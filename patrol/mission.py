"""Patrol mission: shuttle between the start and end bins, pick up objects lying in the path, dump them.

States
  FOLLOW    path-following policy (PPO or pure pursuit) drives; non-pickable objects are driven around
  APPROACH  a confirmed pickable object is within `pick_trigger_ahead` m: P-controller to `pick_standoff`
  PICK      robot stopped, arm goes to the object, gripper closes for `pick_time` s -> items += 1
  DUMP      at a path end (a bin): gripper opens for `dump_time` s, items = 0, wait `dwell_at_bin`, turn around
  BLOCKED   no progress for `blocked_timeout` s (path blocked by something it cannot pick or pass): turn around
  TURN      after any turn-around: rotate in place until facing along the (reversed) path, then FOLLOW

    mission = PatrolMission(PatrolController("policy.onnx", path_xy))
    cmd = mission.step(t, ranges, angles, (x, y, yaw), v, w)   # 10 Hz; cmd.vx, cmd.wz, cmd.gripper, cmd.arm_target_xy
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from patrol.controller import PatrolController
from patrol.detect import DetectCfg, Detection, Tracker, detect_frame, hits_world

FOLLOW, APPROACH, PICK, DUMP, BLOCKED, TURN = "FOLLOW", "APPROACH", "PICK", "DUMP", "BLOCKED", "TURN"


@dataclass
class MissionCfg:
    capacity_items: int = 6
    pick_trigger_ahead: float = 1.3
    pick_standoff: float = 0.45
    pick_time: float = 3.0
    dump_time: float = 3.0
    dwell_at_bin: float = 0.0
    blocked_timeout: float = 8.0
    bin_tol: float = 0.4
    dt: float = 0.1
    lost_timeout: float = 1.0
    approach_timeout: float = 10.0
    ignore_radius: float = 0.3
    turn_done: float = 0.35
    turn_rate: float = 1.2
    bin_ignore: float = 0.7
    detect: DetectCfg = field(default_factory=lambda: DetectCfg(confirm=2))


@dataclass
class Command:
    vx: float
    wz: float
    gripper: str | None
    arm_target_xy: np.ndarray | None
    state: str
    event: str | None = None


class PatrolMission:
    def __init__(self, controller: PatrolController, cfg: MissionCfg | None = None):
        self.ctl = controller
        self.cfg = cfg or MissionCfg()
        self.tracker = Tracker(self.cfg.detect)
        self.state = FOLLOW
        self.items = 0
        self.legs = 0
        self.picked = 0
        self.blocked_events = 0
        self.target: Detection | None = None
        self.handled: list[np.ndarray] = []
        self.timer = 0.0
        self.lost = 0.0
        self.best_idx = -1
        self.stall = 0.0
        self.last_event: str | None = None
        self.n_confirmed = 0
        self.max_width = 0.0
        self.widths: list[float] = []
        self.reversed_count = 0

    # ------------------------------------------------------------------ helpers
    def _is_handled(self, xy) -> bool:
        return any(np.linalg.norm(xy - h) < self.cfg.ignore_radius for h in self.handled)

    def _turn_around(self, pose):
        self.ctl.reverse(pos=pose[:2])
        self.tracker.reset()
        self.best_idx = self.ctl.progress_idx
        self.stall = 0.0
        self.reversed_count += 1
        self.state, self.timer = TURN, 0.0

    def _cmd(self, vx, wz, gripper=None, arm=None, event=None) -> Command:
        if event:
            self.last_event = event
        return Command(float(vx), float(wz), gripper, arm, self.state, event)

    # ------------------------------------------------------------------ main step
    def step(self, t: float, ranges, angles, pose, v: float, w: float) -> Command:
        c = self.cfg
        pose = tuple(float(q) for q in pose)
        self._scan = (ranges, angles, pose)
        vx, wz, _ = self.ctl.step(ranges, angles, pose, v, w)  # keeps index / last_act current every tick
        dets = self.tracker.update(detect_frame(ranges, angles, pose, self.ctl.ref, self.ctl.progress_idx, c.detect))
        s0, L = self.ctl.dist_from_start, (self.ctl.M - 1) * self.ctl.ds
        dets = [d for d in dets if c.bin_ignore < s0 + d.ahead < L - c.bin_ignore]  # the bins are not debris
        self.n_confirmed = len(dets)

        if self.state == FOLLOW:
            if self.ctl.dist_to_goal < c.bin_tol:
                self.state, self.timer = DUMP, 0.0
                self.legs += 1
                return self._cmd(0.0, 0.0, "open", None, "arrived")
            if self.items >= c.capacity_items and self.ctl.dist_from_start < self.ctl.dist_to_goal:
                self._turn_around(pose)
                return self._cmd(0.0, 0.0, None, None, "full_return")
            cands = [d for d in dets if d.pickable and d.ahead < c.pick_trigger_ahead and not self._is_handled(d.xy)]
            if cands:
                self.target = min(cands, key=lambda d: d.range)
                self.state, self.timer, self.lost = APPROACH, 0.0, 0.0
                return self._approach(dets, event="target", pose=pose)
            if self.ctl.progress_idx > self.best_idx:
                self.best_idx, self.stall = self.ctl.progress_idx, 0.0
            else:
                self.stall += c.dt
            if self.stall >= c.blocked_timeout:
                self.blocked_events += 1
                self._turn_around(pose)
                return Command(0.0, 0.0, None, None, BLOCKED, "blocked")
            return self._cmd(vx, wz)

        if self.state == APPROACH:
            self.timer += c.dt
            return self._approach(dets, pose=pose)

        if self.state == PICK:
            self.timer += c.dt
            if self.timer >= c.pick_time:
                self.items += 1
                self.picked += 1
                arm = self.target.xy.copy()
                self.handled.append(arm)
                self.target = None
                self.state = FOLLOW
                self.best_idx, self.stall = self.ctl.progress_idx, 0.0
                return Command(0.0, 0.0, "close", arm, PICK, "picked")
            return self._cmd(0.0, 0.0, "close", self.target.xy.copy())

        if self.state == DUMP:
            self.timer += c.dt
            if self.timer >= c.dump_time + c.dwell_at_bin:
                self.items = 0
                self._turn_around(pose)
                return self._cmd(0.0, 0.0, "open", None, "dumped")
            return self._cmd(0.0, 0.0, "open")

        if self.state == TURN:
            self.timer += c.dt
            err = self.ctl.heading_error(pose[2])
            if abs(err) < c.turn_done or self.timer > 8.0:
                self.state = FOLLOW
                self.best_idx, self.stall = self.ctl.progress_idx, 0.0
                return self._cmd(vx, wz)
            return self._cmd(0.0, -math.copysign(c.turn_rate, err))

        return self._cmd(0.0, 0.0)

    def _approach(self, dets, event=None, pose=None) -> Command:
        c = self.cfg
        match = min(dets, key=lambda d: np.linalg.norm(d.xy - self.target.xy), default=None)
        if event == "target":
            self.widths = [self.target.width]
        if match is not None and np.linalg.norm(match.xy - self.target.xy) < c.ignore_radius:
            self.target, self.lost = match, 0.0
            self.widths.append(match.width)
        else:
            self.lost += c.dt
        # median, not max: single-frame width estimates are noisy (about +-5 cm)
        self.max_width = float(np.median(self.widths))
        too_big = len(self.widths) >= 3 and self.max_width > c.detect.pick_max_width
        if too_big or self.lost > c.lost_timeout or self.timer > c.approach_timeout:
            self.handled.append(self.target.xy.copy())
            self.target, self.state = None, FOLLOW
            return self._cmd(0.0, 0.0, None, None, "too_big" if too_big else "target_lost")
        # range / bearing from the target's world position and the current pose (stays valid between hits)
        dx, dy = self.target.xy[0] - pose[0], self.target.xy[1] - pose[1]
        rng = math.hypot(dx, dy)
        b = (math.atan2(dy, dx) - pose[2] + math.pi) % (2 * math.pi) - math.pi
        self.target.range, self.target.bearing = rng, b
        err = rng - c.pick_standoff
        if abs(err) < 0.05 and abs(b) < 0.15:
            if self._extent_near(self.target.xy, 0.18) > c.detect.pick_max_width:
                self.handled.append(self.target.xy.copy())
                self.target, self.state = None, FOLLOW
                return self._cmd(0.0, 0.0, None, None, "too_big")
            self.state, self.timer = PICK, 0.0
            return self._cmd(0.0, 0.0, "open", self.target.xy.copy(), "pick_start")
        wz = float(np.clip(2.0 * b, -1.5, 1.5))
        vx = 0.4 * float(np.clip(err, -0.25, 1.0)) if abs(b) < 0.6 else 0.0
        if 0 < err and abs(b) < 0.3:
            vx = max(vx, 0.03)
        return self._cmd(vx, wz, "open", None, event)

    def _extent_near(self, xy, radius, gap: float = 0.05) -> float:
        """Width of the object at xy from the raw scan (no path-tube clipping): the run of consecutive
        beams that starts at a hit within `radius` of xy and continues while neighbouring hits are
        closer than `gap`. A wall behind or beside the object sits across a range jump, so it is not
        counted; an object that really is wider than the gripper still is."""
        ranges, angles, pose = self._scan
        pts, r, a, _ = hits_world(ranges, angles, pose, self.cfg.detect.lidar_range)
        if len(pts) == 0:
            return 0.0
        order = np.argsort(a)
        pts, r = pts[order], r[order]
        d = np.linalg.norm(pts - xy, axis=1)
        seed = int(d.argmin())
        if d[seed] > radius:
            return 0.0
        n = len(pts)
        lo = hi = seed
        while hi - lo < n - 1 and np.linalg.norm(pts[(hi + 1) % n] - pts[hi % n]) < gap:
            hi += 1
        while hi - lo < n - 1 and np.linalg.norm(pts[(lo - 1) % n] - pts[lo % n]) < gap:
            lo -= 1
        dbeam = 2 * math.pi / max(len(ranges), 1)
        chord = float(np.linalg.norm(pts[hi % n] - pts[lo % n])) if hi > lo else 0.0
        return chord + float(r[seed]) * dbeam

    def overlay_text(self) -> list[str]:
        tgt = f"target {self.target.range:.2f} m" if self.target is not None else f"{self.n_confirmed} objects in view"
        return [
            f"PATROL {self.state}  leg {self.legs}",
            f"to bin {self.ctl.dist_to_goal:.1f} m  |  {tgt}",
            f"bin {self.items}/{self.cfg.capacity_items}  picked {self.picked}  blocked {self.blocked_events}",
            f"last: {self.last_event or '-'}",
        ]
