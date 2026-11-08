"""CPU mock of the whole rig robot with both jobs (rust + gas leak). No Isaac Sim.

Reuses the team's rust mock (policy/demo_mock.py) unchanged and adds: a bolted flange near the rust work
position, a LeakWorld with a gas field, a wrist-camera renderer for the flange, an arm that can re-torque a
bolt, and a bridge with the leak methods the Isaac bridge must also provide (see integration/INTERFACE.md):

    gas_reading() -> float | None     flanges() / flange(id) / nearest_flange()
    approach_flange(id) -> bool       retreat(distance)      return_to_work()
    arm.torque(flange_id, bolt) -> bool

Run:  python -m policy.leak.mock --leak loose          (also: crack, big, none, dust; --gas-during-scrub)
      python -m policy.leak.mock --leak none --debris litter --person-at 2   (deck sweep: debris + camera person)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from detection.leak import LeakLocalizer
from detection.localize import RustLocalizer, fake_colour_classifier
from policy.deck.job import DeckJob
from policy.deck.mock import DeckMockMixin
from policy.deck.scene import Debris
from policy.demo_mock import GreedyScrub, MockArm, MockBridge, MockPerception, MockRustWorld, scene_geometry
from policy.interfaces import CameraModel
from policy.leak.geometry import FlangeGeometry, look_at
from policy.leak.job import LeakJob, RigOrchestrator
from policy.leak.render import render_flange
from policy.leak.world import LeakWorld
from policy.orchestrator import OrchestratorConfig, S


class LeakPerception:
    """Wrist camera pointed at a flange; renders it with the mock renderer."""

    def __init__(self, world: LeakWorld, W=640, H=480, focal_mm=35.0, h_aperture_mm=20.955):
        self.world, self.W, self.H = world, W, H
        f = focal_mm / h_aperture_mm * W
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        self.fid = next(iter(world.flanges))
        self.view = 0
        self.dust = 0.0
        self.glare = False
        self.occluded_frames = 0
        self.n_frames = 0

    def focus(self, fid: str) -> None:
        self.fid, self.view = fid, 0

    def _poses(self):
        return self.world.flanges[self.fid].geom.view_poses()

    def camera(self) -> CameraModel:
        eye, target = self._poses()[self.view]
        return CameraModel(self.K, look_at(eye, target), self.W, self.H)

    def request_view(self, index: int) -> bool:
        if index >= len(self._poses()):
            return False
        self.view = index
        return True

    def observe(self):
        self.n_frames += 1
        if self.occluded_frames > 0:
            self.occluded_frames -= 1
            return np.full((self.H, self.W, 3), 8, np.uint8)
        return render_flange(self.camera(), self.world.flanges[self.fid], dust=self.dust, glare=self.glare)


class RigPerception:
    """One wrist camera, two things it can be looking at: the rust work area or a flange."""

    def __init__(self, rust: MockPerception, leak: LeakPerception):
        self.rust, self.leak = rust, leak
        self.focus = "rust"

    @property
    def active(self):
        return self.leak if self.focus == "leak" else self.rust

    def observe(self):
        return self.active.observe()

    def camera(self):
        return self.active.camera()

    def request_view(self, index: int) -> bool:
        return self.active.request_view(index)


class LeakArm(MockArm):
    def __init__(self, world, pipe, leak_world: LeakWorld, seed=0):
        super().__init__(world, pipe, seed=seed)
        self.leak_world = leak_world
        self.miss = set()            # (flange, bolt) the tool fails to reach once
        self.torques: list[tuple[str, int, float]] = []

    def torque(self, fid: str, bolt: int) -> bool:
        if (fid, bolt) in self.miss:
            self.miss.discard((fid, bolt))
            return False
        self.torques.append((fid, bolt, self.leak_world.tighten(fid, bolt)))
        return True


class RigMockBridge(DeckMockMixin, MockBridge):
    def __init__(self, pipe, world, perception: RigPerception, arm, leak_world: LeakWorld, work_pos,
                 events=None, verbose=True):
        super().__init__(pipe, world, perception, arm, events=events, verbose=verbose)
        self.leak_world = leak_world
        self.work_pos = np.asarray(work_pos, np.float64)
        self.robot = self.work_pos.copy()
        self.gas_override = None     # e.g. float("nan") = failed sensor
        self._deck_init()

    def advance(self, dt):
        super().advance(dt)
        self.leak_world.step(dt)

    def gas_reading(self):
        if self.gas_override is not None:
            return self.gas_override
        return self.leak_world.gas_at(self.robot)

    def flanges(self):
        return [s.geom for s in self.leak_world.flanges.values()]

    def flange(self, fid):
        return self.leak_world.flanges[fid].geom

    def nearest_flange(self):
        if not self.leak_world.flanges:
            return None
        return min(self.leak_world.flanges,
                   key=lambda f: float(np.linalg.norm(self.leak_world.flanges[f].geom.center - self.robot)))

    def approach_flange(self, fid) -> bool:
        g = self.flange(fid)
        self.robot = g.center + 0.85 * g.normal
        self.perception.focus = "leak"
        self.perception.leak.focus(fid)
        return True

    def return_to_work(self):
        self.robot = self.work_pos.copy()
        self.perception.focus = "rust"

    def retreat(self, distance=3.0):
        fid = self.nearest_flange()
        d = self.robot - self.flange(fid).center if fid else np.array([-1.0, 0, 0])
        d[2] = 0.0
        self.robot = self.robot + d / (np.linalg.norm(d) or 1.0) * distance
        self.perception.focus = "rust"

    def go_to_patrol_pose(self, pose):
        self.return_to_work()
        return True

    def approach(self, theta, z, standoff):
        self.return_to_work()
        return True


SCENARIOS = {
    "none": lambda w: None,
    "loose": lambda w: (w.loosen("F1", 1, 0.15), w.loosen("F1", 5, 0.12)),
    "crack": lambda w: w.crack("F1", 0.3),
    "big": lambda w: w.gas_spike("F1"),
}


def _crack_detector(crack_ckpt):
    """None: the shipped trained model (falls back to classical); "classical": the line check; else a checkpoint dir."""
    if crack_ckpt is None:
        return LeakLocalizer.default()
    if crack_ckpt == "classical":
        return LeakLocalizer()
    return LeakLocalizer.from_checkpoint(crack_ckpt)


DEBRIS_SCENES = {
    "none": [],
    "can": [Debris(0.15, 3.4, "can", length_m=0.12, seed=3)],
    "litter": [Debris(0.15, 3.4, "can", length_m=0.12, seed=3), Debris(0.5, 4.3, "debris_can_red.png", length_m=0.12),
               Debris(-0.05, 5.0, "debris_bottle.png", length_m=0.22)],
}


def _rl_pick():
    from policy.deck.pick_skill import RLPickSkill
    return RLPickSkill.default()


def deck_job(debris=True, person=True) -> DeckJob | None:
    """The shipped debris and person models (None for any whose checkpoint is missing)."""
    from detection.rig_vision import DebrisDetector, PersonDetector
    job = DeckJob(debris=DebrisDetector.default() if debris else None,
                  person=PersonDetector.default() if person else None)
    return job if (job.debris or job.person) else None


def build_rig(log_path=None, leak="none", rust=True, glare=False, dust=0.0, verbose=False, seed=0, crack_ckpt=None,
              deck: DeckJob | None = None, debris="none", person_m=None, deck_dust=0.0, pick_skill=None):
    pipe, dmg_theta = scene_geometry()
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(world.blob(dmg_theta, 0.0, size=0.05, level=0.85) if rust
                   else torch.zeros(1, pipe.n_z, pipe.n_theta))
    rust_perc = MockPerception(world, pipe, dmg_theta)
    if not glare:
        rust_perc.glare_views = set()
    flange = FlangeGeometry("F1", center=[pipe.origin[0] + 1.2, -0.2, 0.9], normal=[0, -1, 0], ref=[0, 0, 1])
    leak_world = LeakWorld([flange])
    SCENARIOS[leak](leak_world)
    leak_world.settle()
    leak_perc = LeakPerception(leak_world)
    leak_perc.dust = dust
    perc = RigPerception(rust_perc, leak_perc)
    arm = LeakArm(world, pipe, leak_world, seed=seed)
    arm.theta = dmg_theta
    work = [pipe.origin[0], -1.0, 0.6]       # 1.5 m from the flange: a real leak shows on the gas sensor here
    bridge = RigMockBridge(pipe, world, perc, arm, leak_world, work, verbose=verbose)
    bridge.dmg_theta = dmg_theta
    bridge._deck_init([Debris(**vars(d)) for d in DEBRIS_SCENES[debris]], person_m=person_m, deck_dust=deck_dust)
    bridge.pick_skill = pick_skill
    loc = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.5, temperature=1.0)
    orch = RigOrchestrator(bridge, perc, loc, world, arm, GreedyScrub(pipe),
                           OrchestratorConfig(log_path=log_path, scrub_steps=200, scrub_redetect_every=40),
                           leak=LeakJob(_crack_detector(crack_ckpt)), deck=deck)
    return orch, bridge, world, leak_world, perc, arm, pipe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak", default="loose", choices=sorted(SCENARIOS) + ["dust"])
    ap.add_argument("--no-rust", action="store_true")
    ap.add_argument("--gas-during-scrub", action="store_true", help="a bolt works loose while the robot scrubs rust")
    ap.add_argument("--log", default="runs/decisions.jsonl")
    ap.add_argument("--debris", default="none", choices=sorted(DEBRIS_SCENES), help="loose objects on the deck")
    ap.add_argument("--person-at", type=float, default=None, help="a person this many metres ahead of the body camera")
    ap.add_argument("--no-deck", action="store_true", help="no debris / camera person skills (team behaviour)")
    ap.add_argument("--crack-ckpt", default="", help="crack detector: empty = shipped trained model, \"classical\" = line check, or a checkpoint dir")
    a = ap.parse_args(argv)
    t0 = time.time()
    leak = "loose" if a.leak == "dust" else a.leak
    if a.gas_during_scrub:
        leak = "none"
    orch, bridge, world, leak_world, perc, arm, pipe = build_rig(a.log, leak=leak, rust=not a.no_rust,
                                                                  dust=0.4 if a.leak == "dust" else 0.0, verbose=True,
                                                                  crack_ckpt=a.crack_ckpt or None,
                                                                  deck=None if a.no_deck else deck_job(),
                                                                  debris=a.debris, person_m=a.person_at,
                                                                  pick_skill=None if a.no_deck else _rl_pick())
    leave = None
    if a.person_at is not None:          # the person walks away after a few seconds
        leave = (bridge.t + 4.0, f"person at {a.person_at} m walks away", lambda: setattr(bridge, "camera_person_m", None))
        bridge.events.append(leave)
    scheduled = False
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN) and len(orch.records) < 500:
        prev = orch.state
        orch.step()
        if a.gas_during_scrub and not scheduled and orch.state == S.SCRUB and prev != S.SCRUB:
            scheduled = True
            bridge.events.append((bridge.t + 5.0, "bolt #2 on flange F1 works loose (unscripted)",
                                  lambda: leak_world.loosen("F1", 2, 0.18)))
            bridge.events.sort(key=lambda e: e[0])
    for r in orch.records:
        ev = r.evidence
        print(f"t={r.t:6.1f}s {r.frm:>16} -> {r.to:<17} gas={ev.get('gas')} | {r.reason}")
    print("\nHUD:", *orch.overlay_text(), sep="\n  ")
    for al in orch.alerts:
        print(f"ALERT [{al['severity']}] {al['message']}")
    if orch.deck is not None:
        print(f"deck: picked {len(bridge.picked)} object(s), {len(bridge.debris)} left")
        for p in bridge.pick_log:
            print(f"  RL pick {p['kind']}: {'ok' if p['ok'] else 'missed'} in {p['seconds']} s, knocks {p['knocks']}")
    print(f"\nfinal: {orch.state.value}  flange truth: {leak_world.truth('F1')}  torques: {arm.torques}  "
          f"rust left={float(world.fraction_remaining()[0]):.3f}  decisions={len(orch.records)}  wall={time.time() - t0:.1f}s")
    return orch


if __name__ == "__main__":
    main()
