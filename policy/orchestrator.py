from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from detection.localize import _components, pixel_to_theta_z, project_mask_to_grid
from policy.interfaces import DecisionRecord
from policy.skills.coverage import dilate
from policy.skills.paint import PaintSkill
from policy.skills.spray import SpraySkill


class S(str, Enum):
    PATROL = "PATROL"
    DETECT = "DETECT"
    SECOND_VIEW = "SECOND_VIEW"
    APPROACH = "APPROACH"
    SPRAY = "SPRAY"
    SOAK = "SOAK"
    SCRUB = "SCRUB"
    VERIFY = "VERIFY"
    PAINT = "PAINT"
    VERIFY_PAINT = "VERIFY_PAINT"
    DONE = "DONE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    BACK_OFF = "BACK_OFF"
    HOLD = "HOLD"
    PAUSE_PERSON = "PAUSE_PERSON"
    GAS_CHECK = "GAS_CHECK"
    LEAK_APPROACH = "LEAK_APPROACH"
    LEAK_INSPECT = "LEAK_INSPECT"
    LEAK_SECOND_VIEW = "LEAK_SECOND_VIEW"
    RETORQUE = "RETORQUE"
    LEAK_VERIFY = "LEAK_VERIFY"
    GAS_RETREAT = "GAS_RETREAT"
    GAS_SEEK = "GAS_SEEK"


TERMINAL = {S.DONE, S.ESCALATE_TO_HUMAN}


@dataclass
class OrchestratorConfig:
    soak_s: float = 30.0
    redetect_s: float = 5.0
    max_second_views: int = 2
    max_scrub_cycles: int = 3
    verify_frac: float = 0.05
    force_limit: float = 40.0
    max_backoffs: int = 3
    person_radius: float = 3.0
    hold_dt: float = 1.0
    hold_max: int = 5
    scrub_steps: int = 120
    scrub_dt: float = 0.1
    scrub_redetect_every: int = 30
    new_patch_min_cells: int = 3
    target_margin: int = 1
    max_paint_attempts: int = 2
    occluded_std: float = 6.0
    clean_min_visible: float = 0.8
    clean_max_uncertain: float = 0.05
    max_verify_views: int = 2
    log_path: Optional[str] = "runs/decisions.jsonl"


@dataclass
class TrackedPatch:
    id: int
    mask: np.ndarray
    centroid: tuple[float, float]
    conf: float
    found_t: float
    found_in: str
    sprayed_t: Optional[float] = None


@dataclass
class Perceived:
    result: object
    grid: np.ndarray
    visible: np.ndarray
    conf: np.ndarray


class Orchestrator:
    def __init__(self, bridge, perception, localizer, world, arm, scrub_policy,
                 config: OrchestratorConfig | None = None,
                 spray: SpraySkill | None = None, paint: PaintSkill | None = None,
                 advisor=None):
        self.bridge, self.perception, self.localizer = bridge, perception, localizer
        self.advisor = advisor
        self.advisor_last = None
        self.advisor_vetoes = 0
        self.last_image: Optional[np.ndarray] = None
        self.world, self.arm, self.scrub_policy = world, arm, scrub_policy
        self.cfg = config or OrchestratorConfig()
        self.spray_skill = spray or SpraySkill()
        self.paint_skill = paint or PaintSkill()
        self.pipe = bridge.pipe()
        self.state = S.PATROL
        self.records: list[DecisionRecord] = []
        self.patches: dict[int, TrackedPatch] = {}
        self.target = np.zeros(self.pipe.shape, bool)
        self.baseline_cells = 0
        self.pending_spray = np.zeros(self.pipe.shape, bool)
        self.soak_until = 0.0
        self.last: Optional[Perceived] = None
        self.rust_frac = None
        self.last_reason = ""
        self.patrol_idx = 0
        self.second_views = 0
        self.scrub_cycles = 0
        self.scrub_steps_left = 0
        self.backoffs = 0
        self.press_scale = 1.0
        self.hold_count = 0
        self.paint_attempts = 0
        self.view_idx = 0
        self.verify_views = 0
        self.n_views = 3
        self.resume: Optional[S] = None
        self.pause_resume: Optional[S] = None
        self._log_f = None
        if self.cfg.log_path:
            Path(self.cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_f = open(self.cfg.log_path, "w")

    @property
    def t(self) -> float:
        return float(self.bridge.sim_time())

    def _advance(self, dt: float):
        self.world.step_soak(dt)
        self.bridge.advance(dt)

    def _evidence(self, **extra) -> dict:
        ev = {"detector_conf": None, "rust_fraction": self.rust_frac,
              "patch_ids": sorted(self.patches), "force": round(float(self.arm.current_force()), 2)}
        if self.last is not None:
            r = self.last.result
            confs = [p.mean_conf for p in r.patches]
            ev["detector_conf"] = round(float(max(confs) if confs else r.max_conf), 3)
            ev["unsure"] = bool(r.unsure)
        if ev["rust_fraction"] is not None:
            ev["rust_fraction"] = round(float(ev["rust_fraction"]), 4)
        ev.update(extra)
        return ev

    def _record(self, to: S, reason: str, evidence: dict):
        rec = DecisionRecord(self.t, self.state.value, to.value, reason, evidence)
        self.records.append(rec)
        self.last_reason = reason
        if self._log_f:
            self._log_f.write(json.dumps(rec.to_json(), default=_json_default) + "\n")
            self._log_f.flush()
        self.state = to
        if hasattr(self.bridge, "publish_overlay"):
            self.bridge.publish_overlay(self.overlay_text())

    def _perceive(self) -> Optional[Perceived]:
        img = self.perception.observe()
        if img is None or float(np.asarray(img, np.float32).std()) < self.cfg.occluded_std:
            return None
        self.last_image = img
        res = self.localizer.localize(img)
        self.pipe = self.bridge.pipe()
        grid, vis, conf = project_mask_to_grid(res.mask, self.perception.camera(), self.pipe, res.prob_map)
        self.last = Perceived(res, grid, vis, conf)
        if self.baseline_cells:
            self.rust_frac = float((grid & self.target).sum() / self.baseline_cells)
        return self.last

    def _interrupt(self) -> Optional[tuple[S, str, dict]]:
        return None

    def _hold(self, reason_state: S) -> tuple[S, str, dict]:
        self.resume = reason_state
        self.hold_count = 0
        return S.HOLD, "perception lost (no frame / occluded) -> hold position", self._evidence()

    def _new_patches(self, p: Perceived) -> list[TrackedPatch]:
        known = dilate(self.target, 2)
        out = []
        for comp in _components(p.grid & ~known):
            if len(comp) < self.cfg.new_patch_min_cells:
                continue
            m = np.zeros(self.pipe.shape, bool)
            rows, cols = zip(*comp)
            m[rows, cols] = True
            out.append(self._register(m, p, self.state.value))
        return out

    def _register(self, m: np.ndarray, p: Perceived, where: str) -> TrackedPatch:
        ii, jj = np.nonzero(m)
        th, z = self.pipe.cell_to_theta_z(ii, jj)
        th_c = float(np.angle(np.exp(1j * th).mean()) % (2 * np.pi))
        tp = TrackedPatch(len(self.patches), m, (round(th_c, 3), round(float(z.mean()), 3)),
                          round(float(p.conf[m].mean()), 3), self.t, where)
        self.patches[tp.id] = tp
        self.target |= dilate(m, self.cfg.target_margin)
        self.baseline_cells += int(m.sum())
        self.pending_spray |= m
        return tp

    def run(self, max_steps: int = 500) -> S:
        for _ in range(max_steps):
            if self.state in TERMINAL:
                break
            self.step()
        if self._log_f:
            self._log_f.flush()
        return self.state

    def step(self):
        if self.state in TERMINAL:
            return
        d = self.bridge.person_distance() if hasattr(self.bridge, "person_distance") else None
        if d is not None and d < self.cfg.person_radius and self.state != S.PAUSE_PERSON:
            self.pause_resume = self.state
            self._record(S.PAUSE_PERSON, f"person at {d:.1f} m < {self.cfg.person_radius} m -> pause",
                         self._evidence(person_m=round(d, 2)))
            return
        nxt, reason, ev = getattr(self, "_s_" + self.state.value.lower())()
        if nxt is not None:
            self._record(nxt, reason, ev)

    def _s_patrol(self):
        poses = self.bridge.patrol_poses()
        if self.patrol_idx >= len(poses):
            return S.DONE, "patrol complete: no rust perceived at any pose", self._evidence()
        pose = poses[self.patrol_idx]
        self.patrol_idx += 1
        self.bridge.go_to_patrol_pose(pose)
        self.second_views = 0
        return S.DETECT, f"at patrol pose {self.patrol_idx}/{len(poses)} -> inspect", self._evidence()

    def _s_detect(self):
        p = self._perceive()
        if p is None:
            return self._hold(S.DETECT)
        r = p.result
        if r.unsure:
            if self.second_views < self.cfg.max_second_views:
                return (S.SECOND_VIEW, f"ambiguous: {len(r.unsure_cells)} patches with p in [0.35,0.65] "
                        f"-> take view {self.second_views + 1}", self._evidence(unsure_cells=len(r.unsure_cells)))
            decided = self._advisor_detect(p)
            if decided:
                return decided
            return (S.ESCALATE_TO_HUMAN, f"still ambiguous after {self.second_views} extra views -> ask human",
                    self._evidence(unsure_cells=len(r.unsure_cells), heatmap_max=round(r.max_conf, 3)))
        new = []
        for comp in _components(p.grid):
            if len(comp) >= self.cfg.new_patch_min_cells:
                m = np.zeros(self.pipe.shape, bool)
                rows, cols = zip(*comp)
                m[rows, cols] = True
                new.append(self._register(m, p, "DETECT"))
        if not new:
            return S.PATROL, f"no rust (max p={r.max_conf:.2f} < thr {r.threshold:.2f}) -> continue patrol", \
                self._evidence()
        self.rust_frac = 1.0
        return (S.APPROACH, f"rust found: {len(new)} patch(es), {self.baseline_cells} cells on pipe grid",
                self._evidence(new_patches=[_pdesc(t) for t in new]))

    def _clean_evidence(self, p: Perceived) -> tuple[bool, float, float]:
        n = max(int(self.target.sum()), 1)
        vis = float(p.visible[self.target].sum() / n)
        unc = float(((p.conf >= 0.2) & (p.conf < p.result.threshold) & self.target).sum() / n)
        return (vis >= self.cfg.clean_min_visible and unc <= self.cfg.clean_max_uncertain), vis, unc

    def _weak_clean(self, p: Perceived, where: S):
        ok, vis, unc = self._clean_evidence(p)
        if ok:
            return None
        ev = self._evidence(target_visible=round(vis, 3), target_uncertain=round(unc, 3))
        if self.verify_views >= self.cfg.max_verify_views:
            return (S.ESCALATE_TO_HUMAN, f"cannot confirm clean: {vis:.0%} of target visible, {unc:.0%} "
                    f"uncertain after {self.verify_views} extra views", ev)
        self.verify_views += 1
        self.view_idx = (self.view_idx + 1) % self.n_views
        self.perception.request_view(self.view_idx)
        return (where, f"looks clean but weak evidence ({vis:.0%} visible, {unc:.0%} uncertain) "
                f"-> re-verify from view {self.view_idx}", ev)

    def _ask_advisor(self, question: str):
        if self.advisor is None or self.last_image is None:
            return None
        verdict = self.advisor.verify(self.last_image, question)
        self.advisor_last = verdict
        return verdict

    def _advisor_evidence(self, verdict, **extra) -> dict:
        return self._evidence(gemini=verdict.summary() if verdict is not None else
                              {"error": getattr(self.advisor, "last_error", None)}, **extra)

    def _advisor_detect(self, p: Perceived):
        verdict = self._ask_advisor("rust")
        if verdict is None or not self.advisor.confident(verdict):
            return None
        if verdict.clean:
            return (S.PATROL, f"Gemini second opinion: clean ({verdict.confidence:.2f}) - {verdict.reason[:60]}",
                    self._advisor_evidence(verdict))
        lo = self.localizer.unsure_band[0] if hasattr(self.localizer, "unsure_band") else 0.35
        candidate = (p.conf >= lo) & p.visible
        cells = set()
        for x, y in verdict.points:
            hit = pixel_to_theta_z((x, y), self.perception.camera(), self.pipe)
            if hit is not None:
                cells.add(tuple(int(v) for v in self.pipe.theta_z_to_cell(*hit)))
        new = []
        for comp in _components(candidate):
            if len(comp) < self.cfg.new_patch_min_cells or (cells and not cells & set(comp)):
                continue
            m = np.zeros(self.pipe.shape, bool)
            rows, cols = zip(*comp)
            m[rows, cols] = True
            new.append(self._register(m, p, "DETECT+GEMINI"))
        if not new:
            return None
        self.rust_frac = 1.0
        return (S.APPROACH, f"Gemini second opinion: rust ({verdict.confidence:.2f}) -> {len(new)} patch(es), "
                f"{self.baseline_cells} cells", self._advisor_evidence(verdict, new_patches=[_pdesc(t) for t in new]))

    def _advisor_verify(self, frac: float):
        if self.advisor_vetoes >= 2:
            return None
        verdict = self._ask_advisor("clean")
        if verdict is None or not self.advisor.confident(verdict) or not verdict.rust:
            return None
        self.advisor_vetoes += 1
        ev = self._advisor_evidence(verdict)
        if self.scrub_cycles >= self.cfg.max_scrub_cycles:
            return (S.ESCALATE_TO_HUMAN, f"Gemini sees rust left ({verdict.confidence:.2f}) after "
                    f"{self.scrub_cycles} cycles - {verdict.reason[:50]}", ev)
        return (S.SCRUB, f"Gemini veto: detector says {frac:.1%} but rust still visible "
                f"({verdict.confidence:.2f}) -> scrub again", ev)

    def _s_second_view(self):
        self.second_views += 1
        self.view_idx = self.second_views
        ok = self.perception.request_view(self.second_views)
        if not ok:
            return S.ESCALATE_TO_HUMAN, f"view {self.second_views} unreachable while ambiguous", self._evidence()
        return S.DETECT, f"moved camera to view {self.second_views} -> re-detect", self._evidence()

    def _s_approach(self):
        ii, jj = np.nonzero(self.target)
        th, z = self.pipe.cell_to_theta_z(ii, jj)
        th_c = float(np.angle(np.exp(1j * th).mean()) % (2 * np.pi))
        self.bridge.approach(th_c, float(z.mean()), self.spray_skill.standoff)
        return S.SPRAY, f"in reach of target centroid (theta={th_c:.2f}, z={z.mean():.2f})", \
            self._evidence()

    def _s_spray(self):
        plan = self.spray_skill.plan(self.pending_spray, self.pipe)
        sprayed = self.spray_skill.execute(plan, self.arm, self.world)
        ids = [pid for pid, tp in self.patches.items() if tp.sprayed_t is None and (tp.mask & plan.target).any()]
        for pid in ids:
            self.patches[pid].sprayed_t = self.t
        self.pending_spray[:] = False
        self.soak_until = max(self.soak_until, self.t + self.cfg.soak_s)
        return (S.SOAK, f"sprayed patch(es) {ids}: {len(plan.waypoints)} wp @ {plan.standoff} m, "
                f"{plan.flow_rate} L/min; soak until t={self.soak_until:.0f}s",
                self._evidence(sprayed_cells=int(sprayed.sum()), sprayed_ids=ids))

    def _s_soak(self):
        if self.t >= self.soak_until - 1e-6:
            return S.SCRUB, f"soak complete ({self.cfg.soak_s:.0f} s since last spray)", self._evidence()
        dt = min(self.cfg.redetect_s, self.soak_until - self.t)
        self._advance(dt)
        p = self._perceive()
        if p is None:
            return self._hold(S.SOAK)
        new = self._new_patches(p)
        if new:
            return (S.SPRAY, "RE-PLAN: new rust " + ", ".join(_pdesc(t) for t in new) +
                    " appeared during soak -> spray it, extend soak", self._evidence(new_patches=[_pdesc(t) for t in new]))
        left = self.soak_until - self.t
        return S.SOAK, f"re-detect: no new rust; soak {left:.0f} s left", self._evidence()

    def _s_scrub(self):
        cfg = self.cfg
        if self.scrub_steps_left <= 0:
            self.scrub_cycles += 1
            self.scrub_steps_left = cfg.scrub_steps
            self.scrub_policy.reset()
        if self.last is None:
            return self._hold(S.SCRUB)
        while self.scrub_steps_left > 0:
            hit = self._interrupt()
            if hit is not None:
                self.scrub_steps_left = 0
                self.scrub_cycles -= 1
                return hit
            f = float(self.arm.current_force())
            if f > cfg.force_limit:
                self.resume = S.SCRUB
                return (S.BACK_OFF, f"contact force {f:.0f} N > {cfg.force_limit:.0f} N limit -> retract",
                        self._evidence(force=round(f, 1)))
            obs = {"ee_theta_z": np.asarray(self.arm.ee_theta_z(), float),
                   "target_mask": self.target.copy(),
                   "perceived_rust": np.where(self.target, self.last.conf * self.last.grid, 0.0),
                   "force": f}
            if hasattr(self.bridge, "scrub_obs"):
                obs.update(self.bridge.scrub_obs() or {})
            a = np.asarray(self.scrub_policy.step(obs), float).copy()
            a[2] = float(np.clip(a[2], 0, 1)) * self.press_scale
            self.arm.apply_ee_delta(a)
            self._advance(cfg.scrub_dt)
            self.scrub_steps_left -= 1
            if self.scrub_steps_left % cfg.scrub_redetect_every == 0:
                p = self._perceive()
                if p is None:
                    return self._hold(S.SCRUB)
                new = self._new_patches(p)
                if new:
                    self.scrub_steps_left = 0
                    self.scrub_cycles -= 1
                    return (S.SPRAY, "RE-PLAN: new rust " + ", ".join(_pdesc(t) for t in new) +
                            " appeared mid-scrub -> pause scrub, spray + soak it first",
                            self._evidence(new_patches=[_pdesc(t) for t in new]))
        return S.VERIFY, f"scrub cycle {self.scrub_cycles} finished ({cfg.scrub_steps} steps)", self._evidence()

    def _s_verify(self):
        p = self._perceive()
        if p is None:
            return self._hold(S.VERIFY)
        new = self._new_patches(p)
        if new:
            return (S.SPRAY, "RE-PLAN: new rust " + ", ".join(_pdesc(t) for t in new) + " at verify -> spray it",
                    self._evidence(new_patches=[_pdesc(t) for t in new]))
        frac = self.rust_frac
        if frac <= self.cfg.verify_frac:
            weak = self._weak_clean(p, S.VERIFY)
            if weak:
                return weak
            veto = self._advisor_verify(frac)
            if veto:
                return veto
            self.verify_views = 0
            return S.PAINT, f"verified clean: {frac:.1%} rust left <= {self.cfg.verify_frac:.0%}", self._evidence()
        if self.scrub_cycles >= self.cfg.max_scrub_cycles:
            return (S.ESCALATE_TO_HUMAN, f"{frac:.1%} rust remains after {self.scrub_cycles} scrub cycles "
                    "-> escalate with evidence", self._evidence(
                        residual_cells=[list(map(int, c)) for c in np.argwhere(p.grid & self.target)][:50],
                        heatmap_max=round(p.result.max_conf, 3)))
        return S.SCRUB, f"{frac:.1%} rust left > {self.cfg.verify_frac:.0%} -> scrub again " \
            f"(cycle {self.scrub_cycles + 1}/{self.cfg.max_scrub_cycles})", self._evidence()

    def _s_paint(self):
        self.paint_attempts += 1
        info = self.paint_skill.execute(self.target, self.pipe, self.arm, self.world)
        return (S.VERIFY_PAINT, f"painted {info['target_cells']} cells in {info['passes']} pass(es), "
                f"coverage {info['coverage']:.1%}", self._evidence(paint=info, paint_passes=self.paint_skill.passes[-info['passes']:]))

    def _s_verify_paint(self):
        p = self._perceive()
        if p is None:
            return self._hold(S.VERIFY_PAINT)
        cov = self.paint_skill.coverage(self.paint_skill.target_for(self.target))
        if cov >= self.paint_skill.min_coverage and self.rust_frac <= self.cfg.verify_frac:
            weak = self._weak_clean(p, S.VERIFY_PAINT)
            if weak:
                return weak
            return S.DONE, f"paint coverage {cov:.1%} >= 98% and no rust visible -> repair complete", \
                self._evidence(paint_coverage=round(cov, 4))
        if self.paint_attempts < self.cfg.max_paint_attempts:
            return S.PAINT, f"paint coverage {cov:.1%} / rust {self.rust_frac:.1%} -> repaint gaps", \
                self._evidence(paint_coverage=round(cov, 4))
        return S.ESCALATE_TO_HUMAN, f"paint coverage {cov:.1%} after {self.paint_attempts} attempts", \
            self._evidence(paint_coverage=round(cov, 4))

    def _s_back_off(self):
        self.backoffs += 1
        self.arm.back_off(0.05)
        self._advance(1.0)
        if self.backoffs > self.cfg.max_backoffs:
            return S.ESCALATE_TO_HUMAN, f"force limit hit {self.backoffs}x -> stop, escalate", self._evidence()
        self.press_scale *= 0.7
        nxt = self.resume or S.SCRUB
        return nxt, f"retracted 5 cm, press scaled to {self.press_scale:.2f} -> retry {nxt.value}", self._evidence()

    def _s_hold(self):
        self.hold_count += 1
        self._advance(self.cfg.hold_dt)
        img = self.perception.observe()
        if img is not None and float(np.asarray(img, np.float32).std()) >= self.cfg.occluded_std:
            nxt = self.resume or S.DETECT
            return nxt, f"camera view restored after {self.hold_count} s hold -> resume {nxt.value}", self._evidence()
        if self.hold_count >= self.cfg.hold_max:
            return S.ESCALATE_TO_HUMAN, f"perception lost for {self.hold_count} s -> escalate", self._evidence()
        return None, "", {}

    def _s_pause_person(self):
        self.bridge.advance(1.0)
        d = self.bridge.person_distance()
        if d is None or d >= self.cfg.person_radius:
            nxt = self.pause_resume or S.DETECT
            return nxt, f"person clear ({'none' if d is None else f'{d:.1f} m'}) -> resume {nxt.value}", self._evidence()
        return None, "", {}

    def overlay_text(self) -> list[str]:
        return overlay_text(self)


def overlay_text(orch: "Orchestrator") -> list[str]:
    st = orch.state.value
    if orch.state == S.SOAK:
        st += f" ({max(0.0, orch.soak_until - orch.t):.0f} s left)"
    elif orch.state == S.SCRUB:
        st += f" (cycle {orch.scrub_cycles}/{orch.cfg.max_scrub_cycles})"
    conf = None
    if orch.last is not None:
        cs = [p.mean_conf for p in orch.last.result.patches]
        conf = max(cs) if cs else orch.last.result.max_conf
    rust = "--" if orch.rust_frac is None else f"{orch.rust_frac:.1%}"
    lines = [f"t={orch.t:6.1f}s  STATE: {st}",
             f"rust left: {rust}   det conf: {'--' if conf is None else f'{conf:.2f}'}",
             f"patches: {', '.join(f'#{k}' for k in orch.patches) or '-'}   force: {orch.arm.current_force():.0f} N",
             f"why: {orch.last_reason[:70]}"]
    v = getattr(orch, "advisor_last", None)
    if v is not None:
        lines.append(f"gemini: {v.verdict} {v.confidence:.2f} ({v.latency_ms:.0f} ms) {v.reason[:48]}")
    return lines


def _pdesc(tp: TrackedPatch) -> str:
    return f"#{tp.id}(theta={tp.centroid[0]:.2f},z={tp.centroid[1]:.2f},conf={tp.conf:.2f})"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def draw_hud(rgb: np.ndarray, lines: list[str], state: str = "") -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    d = ImageDraw.Draw(im, "RGBA")
    h = 14 * len(lines) + 10
    d.rectangle([0, 0, min(im.width, 560), h], fill=(0, 0, 0, 150))
    alert = any(k in (lines[0] if lines else "") for k in ("ESCALATE", "BACK_OFF", "HOLD", "PAUSE", "GAS_RETREAT"))
    for k, line in enumerate(lines):
        col = (255, 90, 60) if (alert and k == 0) else ((120, 255, 140) if k == 0 else (235, 235, 235))
        d.text((6, 5 + 14 * k), line, fill=col)
    return np.asarray(im)
