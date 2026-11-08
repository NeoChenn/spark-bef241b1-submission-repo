from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np

from integration.dual_argon import HOLD, RESUME_RUST, RETREAT, SCRUBBER, STOP_LEL, Command
from policy.orchestrator import TERMINAL, S
from policy.orchestrator import overlay_text as rust_overlay

try:
    from integration.fleet_paths import safe_pose as _fleet_safe_pose
except ImportError:
    _fleet_safe_pose = None

CAPTIONS = {
    S.PATROL: "Unit 1 patrolling the pipe deck",
    S.DETECT: "Unit 1 scanning the pipe for rust",
    S.SECOND_VIEW: "Unit 1 taking a second look",
    S.APPROACH: "Unit 1 moving in on the rust patch",
    S.SPRAY: "Unit 1 spraying rust remover",
    S.SOAK: "Rust remover soaking in",
    S.SCRUB: "Unit 1 scrubbing the rust",
    S.VERIFY: "Unit 1 checking the steel is clean",
    S.PAINT: "Unit 1 painting the bare steel",
    S.VERIFY_PAINT: "Unit 1 checking the paint coverage",
    S.DONE: "Rust repair complete",
    S.ESCALATE_TO_HUMAN: "Unit 1 needs a human",
    S.BACK_OFF: "Contact force high: brush backing off",
    S.HOLD: "Unit 1 holding",
    S.PAUSE_PERSON: "Person nearby: Unit 1 paused",
}

Pose = np.ndarray
SafePose = Callable[[str, str, np.ndarray], Pose]


def stub_safe_pose(bridge, hold_step_m: float = 1.5, retreat_m: float = 3.0) -> SafePose:
    def pose(unit: str, kind: str, base: np.ndarray) -> Pose:
        base = np.asarray(base, float)
        away = np.array([0.0, -1.0, 0.0])
        pipe = bridge.pipe() if hasattr(bridge, "pipe") else None
        if pipe is not None and hasattr(pipe, "origin") and len(base) == 3:
            ax = np.asarray(getattr(pipe, "axis", [1.0, 0.0, 0.0]), float)
            rel = base - pipe.origin
            off = rel - ax * float(rel @ ax)
            off[2] = 0.0
            if np.linalg.norm(off) > 1e-6:
                away = off / np.linalg.norm(off)
        out = base.copy()
        n = min(len(out), 3)
        out[:n] = base[:n] + away[:n] * (retreat_m if kind == RETREAT else hold_step_m)
        return out
    return pose


class Scrubber:
    def __init__(self, orch, name: str = SCRUBBER, safe_pose: Optional[SafePose] = None, lift_m: float = 0.05,
                 hold_dt: float = 1.0, hold_step_m: float = 1.5, retreat_m: float = 3.0,
                 stop_lel: Optional[float] = STOP_LEL, frame: Optional[str] = None):
        self.orch = orch
        orch.leak = None
        orch._interrupt = self._interrupt
        self.name = name
        self.frame = frame or getattr(orch.bridge, "frame", "local")
        world = self.frame == "world" and _fleet_safe_pose is not None
        self.safe_pose = safe_pose or (_fleet_safe_pose if world else stub_safe_pose(orch.bridge, hold_step_m,
                                                                                    retreat_m))
        self.lift_m, self.hold_dt, self.stop_lel = lift_m, hold_dt, stop_lel
        self.held = False
        self.hold_kind: Optional[str] = None
        self.hold_reason = ""
        self.resume_state: Optional[S] = None
        self.work_base: Optional[np.ndarray] = None
        self.soak_left = 0.0
        self.scrub_saved: Optional[tuple[int, int]] = None
        self._want: Optional[tuple[str, str]] = None
        self._stepping = False
        self.log: list[dict] = []

    @property
    def bridge(self):
        return self.orch.bridge

    @property
    def arm(self):
        return self.orch.arm

    @property
    def t(self) -> float:
        return self.orch.t

    @property
    def state(self) -> str:
        return self.orch.state.value

    @property
    def done(self) -> bool:
        return self.orch.state in TERMINAL and not self.held

    @property
    def onboard_lel(self) -> Optional[float]:
        if not hasattr(self.bridge, "gas_reading"):
            return None
        g = self.bridge.gas_reading()
        if g is None or not math.isfinite(float(g)):
            return None
        return round(100.0 * float(g), 2)

    @property
    def force_n(self) -> float:
        return round(float(self.arm.current_force()), 1)

    @property
    def rust_fraction(self) -> Optional[float]:
        r = self.orch.rust_frac
        return None if r is None else round(float(r), 4)

    @property
    def pos(self) -> Optional[np.ndarray]:
        if hasattr(self.bridge, "base_pose"):
            return np.asarray(self.bridge.base_pose(), float)
        r = getattr(self.bridge, "robot", None)
        return None if r is None else np.asarray(r, float).copy()

    @property
    def caption(self) -> str:
        if self.held:
            what = "stepped back" if self.hold_kind == HOLD else "retreated"
            paused = self.resume_state.value if self.resume_state else "-"
            return f"Unit 1 on hold: brush lifted, {what} to a safe pose; {paused} paused, not cancelled"
        if self.orch.state == S.SOAK:
            return f"{CAPTIONS[S.SOAK]} ({max(0.0, self.orch.soak_until - self.t):.0f} s left)"
        if self.orch.state == S.SCRUB and self.rust_fraction is not None:
            return f"{CAPTIONS[S.SCRUB]} ({self.rust_fraction:.0%} left)"
        return CAPTIONS.get(self.orch.state, self.state)

    def hud(self) -> dict:
        return {"robot": self.name, "state": self.state, "caption": self.caption, "held": self.held,
                "hold_kind": self.hold_kind, "paused_state": self.resume_state.value if self.resume_state else None,
                "onboard_lel": self.onboard_lel, "force_n": self.force_n, "rust_fraction": self.rust_fraction,
                "soak_left_s": round(self._soak_left(), 1) if self.orch.state == S.SOAK or
                (self.held and self.resume_state == S.SOAK) else None,
                "pos": None if self.pos is None else [round(float(v), 3) for v in self.pos]}

    def robot(self) -> dict:
        p = self.pos
        return {"state": self.state, "held": self.held, "paused": self.held, "job": "rust",
                "onboard_lel": self.onboard_lel, "rust_fraction": self.rust_fraction,
                "xy": None if p is None else [round(float(p[0]), 2), round(float(p[1]), 2)]}

    def overlay_text(self) -> list[str]:
        lines = rust_overlay(self.orch)
        g = self.onboard_lel
        lines[1] = f"{lines[1]}   sniffer: {'--' if g is None else f'{g:.1f}'} %LEL"
        if self.held:
            lines[0] = f"{lines[0]}  [{self.hold_kind.upper()}: {self.resume_state.value} paused]"
        return lines

    def snapshot(self) -> dict:
        o, w = self.orch, self.orch.world
        out = {"state": (self.resume_state if self.held else o.state).value, "target": o.target.copy(),
               "pending_spray": o.pending_spray.copy(), "patches": sorted(o.patches),
               "baseline_cells": o.baseline_cells, "rust_frac": o.rust_frac,
               "scrub": self.scrub_saved if self.held and self.scrub_saved else (o.scrub_steps_left, o.scrub_cycles),
               "soak_left": round(self._soak_left(), 6)}
        for k in ("rust", "sprayed", "soak_time", "painted"):
            v = getattr(w, k, None)
            if v is not None:
                out[k] = v.detach().cpu().numpy().copy() if hasattr(v, "detach") else np.array(v, copy=True)
        painted = getattr(o.paint_skill, "painted", None)
        if painted is not None:
            out["paint_skill_painted"] = np.array(painted, copy=True)
        return out

    def _soak_left(self) -> float:
        return self.soak_left if self.held else self.orch.soak_until - self.t

    def handle(self, cmd: Union[Command, str, dict]) -> bool:
        kind, robot, why = _parse(cmd)
        if robot and robot != self.name:
            return False
        if kind in (HOLD, RETREAT):
            return self.hold(kind, why)
        if kind == RESUME_RUST:
            return self.resume(why)
        return False

    def hold(self, kind: str = HOLD, why: str = "") -> bool:
        why = why or ("Argon spark rule: wire brushing is hot work" if kind == HOLD else "Argon: retreat")
        if self.held:
            if kind == RETREAT and self.hold_kind != RETREAT:
                self.hold_kind = RETREAT
                self._move(self.safe_pose(self.name, RETREAT, self.work_base))
                self._note("retreat", why)
                return True
            return False
        if self._stepping:
            if self._want is None or kind == RETREAT:
                self._want = (kind, why)
            return True
        self._enter(kind, why)
        return True

    def resume(self, why: str = "") -> bool:
        if self._want is not None and not self.held:
            self._want = None
            return True
        if not self.held:
            return False
        o = self.orch
        if self.work_base is not None:
            self._move(self.work_base)
        elif hasattr(self.bridge, "return_to_work"):
            self.bridge.return_to_work()
        if hasattr(self.arm, "unstow"):
            self.arm.unstow()
        o.soak_until = self.t + self.soak_left
        if self.scrub_saved is not None:
            o.scrub_steps_left, o.scrub_cycles = self.scrub_saved
        nxt = self.resume_state
        self.held, self.hold_kind, self.resume_state, self.scrub_saved = False, None, None, None
        o._record(nxt, f"{why or 'Argon: zone clear'} -> back to the work pose, resume {nxt.value}",
                  o._evidence(fleet="resume"))
        self._note("resume", why)
        return True

    def step(self) -> None:
        if self.held:
            if self.hold_dt:
                self.bridge.advance(self.hold_dt)
            return
        if self._gas_stop():
            return
        if self.orch.state in TERMINAL:
            return
        self._stepping = True
        try:
            self.orch.step()
        finally:
            self._stepping = False
        if self._want is not None and not self.held:
            kind, why = self._want
            self._want = None
            self._enter(kind, why)

    def run(self, max_steps: int = 500) -> str:
        for _ in range(max_steps):
            if self.done:
                break
            self.step()
        return self.state

    def _gas_stop(self) -> bool:
        g = self.onboard_lel
        if self.stop_lel is None or g is None or g < self.stop_lel:
            return False
        if self.orch.state in TERMINAL:
            return False
        self.hold(RETREAT, f"onboard sniffer {g:.1f} %LEL >= stop {self.stop_lel:.0f}")
        return True

    def _interrupt(self):
        if self._want is None and self._gas_hot():
            self._want = (RETREAT, f"onboard sniffer {self.onboard_lel:.1f} %LEL >= stop {self.stop_lel:.0f}")
        if self._want is None:
            return None
        o = self.orch
        kind, why = self._want
        self._want = None
        self.scrub_saved = (o.scrub_steps_left, o.scrub_cycles)
        return self._enter(kind, why, record=False)

    def _gas_hot(self) -> bool:
        g = self.onboard_lel
        return self.stop_lel is not None and g is not None and g >= self.stop_lel

    def _enter(self, kind: str, why: str, record: bool = True):
        o = self.orch
        self.resume_state = o.state
        self.soak_left = o.soak_until - self.t
        self.work_base = self.pos
        if self.arm is not None:
            self.arm.back_off(self.lift_m)
            if hasattr(self.arm, "stow"):
                self.arm.stow()
        if self.work_base is not None:
            self._move(self.safe_pose(self.name, kind, self.work_base))
        self.held, self.hold_kind, self.hold_reason = True, kind, why
        what = "step back" if kind == HOLD else "retreat"
        self._note(kind, why)
        rec = (S.HOLD, f"{why} -> lift brush, arm to safe pose, {what}; {self.resume_state.value} paused",
               o._evidence(fleet=kind, paused=self.resume_state.value))
        if not record:
            return rec
        o._record(*rec)
        return None

    def _move(self, pose) -> None:
        pose = np.asarray(pose, float)
        if hasattr(self.bridge, "move_base"):
            self.bridge.move_base(pose)
        elif hasattr(self.bridge, "robot"):
            self.bridge.robot = pose.copy()

    def _note(self, kind: str, why: str) -> None:
        self.log.append({"t": round(self.t, 2), "kind": kind, "reason": why, "state": self.state})


def _parse(cmd) -> tuple[str, str, str]:
    if isinstance(cmd, str):
        return cmd, "", ""
    if isinstance(cmd, dict):
        return str(cmd.get("kind") or cmd.get("command") or ""), str(cmd.get("robot") or ""), str(cmd.get("reason") or "")
    return cmd.kind, cmd.robot or "", cmd.reason or ""


def build_scrubber(log_path=None, world_frame: bool = False, **kw) -> tuple[Scrubber, dict]:
    from policy.leak.mock import build_rig

    orch, bridge, world, leak_world, perc, arm, pipe = build_rig(log_path, leak="none")
    if world_frame:
        from integration.fleet_paths import SCRUB

        bridge.frame = "world"
        bridge.work_pos = np.array([SCRUB[0], SCRUB[1], bridge.work_pos[2]])
        bridge.robot = bridge.work_pos.copy()
    return Scrubber(orch, **kw), {"bridge": bridge, "world": world, "leak_world": leak_world, "arm": arm,
                                  "pipe": pipe, "perception": perc}
