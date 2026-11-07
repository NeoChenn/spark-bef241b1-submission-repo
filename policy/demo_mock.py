from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from detection.localize import RustLocalizer, fake_colour_classifier, pixel_rays, ray_cylinder
from policy.interfaces import CameraModel, PipeGeometry
from policy.orchestrator import Orchestrator, OrchestratorConfig, S

try:
    from policy.envs.rust_state import RustState as _RustStateBase
except Exception:
    _RustStateBase = None

TWO_PI = 2 * math.pi


class _MiniRustState:

    def __init__(self, num_envs=1, H=32, W=64, length=0.6, **_):
        self.num_envs, self.H, self.W, self.length = 1, H, W, length
        self.rust = torch.zeros(1, H, W)
        self.sprayed = torch.zeros(1, H, W, dtype=torch.bool)
        self.soak_time = torch.zeros(1, H, W)
        self.radius = torch.full((1,), 0.1)
        self.initial_total = torch.ones(1)
        self.theta_c = (torch.arange(W).float() + 0.5) / W * TWO_PI
        self.z_c = (torch.arange(H).float() + 0.5) / H * length - length / 2

    def spray(self, mask, env_ids=None, soak_time=0.0):
        mask = mask.bool()
        self.soak_time = torch.where(mask & ~self.sprayed, torch.tensor(float(soak_time)), self.soak_time)
        self.sprayed |= mask

    def step_soak(self, dt):
        self.soak_time += dt * self.sprayed.float()

    def surface_to_uv(self, theta, z):
        return torch.stack([torch.remainder(theta, TWO_PI) / TWO_PI, (z + self.length / 2) / self.length], -1)

    def abrade(self, contact_uv, contact_radius, normal_force, slide_speed, dt):
        th = contact_uv[:, 0] * TWO_PI
        zc = contact_uv[:, 1] * self.length - self.length / 2
        darc = (torch.remainder(self.theta_c.view(1, 1, -1) - th.view(-1, 1, 1) + math.pi, TWO_PI) - math.pi) * 0.1
        dz = self.z_c.view(1, -1, 1) - zc.view(-1, 1, 1)
        w = torch.clamp(1 - (darc ** 2 + dz ** 2) / contact_radius ** 2, min=0)
        soaked = self.sprayed & (self.soak_time >= 30)
        mult = torch.where(soaked, 4.0, torch.where(self.sprayed, 1.5, 1.0))
        rate = 0.4 * normal_force.float() * slide_speed.float().clamp(0, 0.5) * (normal_force >= 5).float()
        delta = torch.minimum(self.rust, rate.view(-1, 1, 1) * mult * w * dt)
        self.rust = self.rust - delta
        return delta.sum(dim=(1, 2))

    def fraction_remaining(self):
        return self.rust.sum(dim=(1, 2)) / self.initial_total

    def add_patch(self, *a, **k):
        raise NotImplementedError

    def to_texture(self, env_ids=None, overlay=False):
        r = self.rust.unsqueeze(-1)
        steel = torch.tensor([150.0, 152.0, 158.0])
        rc = torch.tensor([176.0, 92.0, 38.0]) + (torch.tensor([96.0, 44.0, 18.0]) - torch.tensor([176.0, 92.0, 38.0])) * r
        rgb = steel + (rc - steel) * r.clamp(0, 1) ** 0.7 + torch.tensor([-10.0, 0.0, 25.0]) * self.sprayed.unsqueeze(-1)
        return torch.cat([rgb, torch.full_like(r, 255.0)], -1).round().clamp(0, 255).to(torch.uint8)


_Base = _RustStateBase or _MiniRustState


class MockRustWorld(_Base):

    def __init__(self, H=32, W=64, length=0.6, radius=0.1):
        super().__init__(num_envs=1, H=H, W=W, length=length)
        self.radius[:] = radius
        self.painted = torch.zeros(1, H, W, dtype=torch.bool)

    def blob(self, theta, z, size=0.05, level=0.85):
        dth = (torch.remainder(self.theta_c.view(1, -1) - theta + math.pi, TWO_PI) - math.pi) * float(self.radius[0])
        dz = self.z_c.view(-1, 1) - z
        f = torch.exp(-0.5 * ((dth / size) ** 2 + (dz / (0.8 * size)) ** 2))
        return (torch.clamp((f - 0.3) / 0.25, 0, 1) * level).unsqueeze(0)

    def set_rust(self, field):
        self.rust = field.clone()
        self.sprayed[:] = False
        self.soak_time[:] = 0
        self.initial_total = self.rust.sum(dim=(1, 2)).clamp_min(1e-6)

    def inject_patch(self, theta, z, size=0.04, level=0.8):
        self.rust = torch.maximum(self.rust, self.blob(theta, z, size, level))
        self.initial_total = torch.maximum(self.initial_total, self.rust.sum(dim=(1, 2)))

    def paint(self, mask):
        self.painted |= torch.as_tensor(mask).bool()


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, -up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xc, yc, zc, eye
    return T


class MockPerception:

    def __init__(self, world: MockRustWorld, pipe: PipeGeometry, focus_theta: float,
                 W=320, H=240, focal_mm=35.0, h_aperture_mm=20.955):
        self.world, self.pipe = world, pipe
        self.W, self.H = W, H
        f = focal_mm / h_aperture_mm * W
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        surf, n = pipe.surface_point(focus_theta, 0.0)
        n2 = pipe.surface_point(focus_theta + 0.35, 0.0)[1]
        n3 = pipe.surface_point(focus_theta - 0.35, 0.0)[1]
        self.views = [look_at(surf + 0.85 * n, surf),
                      look_at(surf + 0.7 * n2 + 0.12 * pipe.axis, surf + 0.03 * pipe.axis),
                      look_at(surf + 0.7 * n3 - 0.12 * pipe.axis, surf)]
        self.view = 0
        self.glare_views = {0}
        self.glare_alpha = 0.6
        self.glare_sx = 300.0
        self.occluded_frames = 0
        self.n_frames = 0
        self._cache = {}

    def camera(self) -> CameraModel:
        return CameraModel(self.K, self.views[self.view], self.W, self.H)

    def request_view(self, index: int) -> bool:
        if index >= len(self.views):
            return False
        self.view = index
        return True

    def _rays(self):
        if self.view not in self._cache:
            o, d = pixel_rays(self.camera())
            hit, th, z, pts = ray_cylinder(o, d, self.pipe)
            i, j = self.pipe.theta_z_to_cell(th, z)
            n = pts - self.pipe.origin
            n -= (n @ self.pipe.axis)[..., None] * self.pipe.axis
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
            shade = np.clip(-(d * n).sum(-1), 0, 1) * 0.55 + 0.45
            self._cache[self.view] = (hit, i, j, shade)
        return self._cache[self.view]

    def observe(self):
        self.n_frames += 1
        if self.occluded_frames > 0:
            self.occluded_frames -= 1
            return np.full((self.H, self.W, 3), 8, np.uint8)
        hit, i, j, shade = self._rays()
        tex = self.world.to_texture()[0, ..., :3].float().numpy()
        painted = self.world.painted[0].numpy()
        tex[painted] = (112, 122, 110)
        img = np.empty((self.H, self.W, 3), np.float32)
        img[:] = (38, 44, 52)
        img[hit] = tex[i[hit], j[hit]] * shade[hit, None]
        if self.view in self.glare_views:
            yy, xx = np.mgrid[0:self.H, 0:self.W]
            g = np.exp(-(((xx - self.W * 0.5) / self.glare_sx) ** 2 + ((yy - self.H * 0.5) / (0.6 * self.glare_sx)) ** 2))
            a = (self.glare_alpha * g)[..., None] * hit[..., None]
            img = img * (1 - a) + 250 * a
        return np.clip(img, 0, 255).astype(np.uint8)


class MockArm:
    def __init__(self, world: MockRustWorld, pipe: PipeGeometry, seed=0):
        self.world, self.pipe = world, pipe
        self.theta, self.z = math.pi / 2, 0.0
        self.force = 0.0
        self.spike_steps = 0
        self.spike_at_step = None
        self.n_delta = 0
        self.rng = np.random.default_rng(seed)
        self.drop_prob = 0.06
        self.contact_radius = 0.03

    def move_to(self, waypoints):
        reached = [bool(self.rng.random() > self.drop_prob) for _ in waypoints]
        if waypoints:
            self.theta, self.z = waypoints[-1].theta, waypoints[-1].z
        self.force = 0.0
        return reached

    def apply_ee_delta(self, a):
        self.n_delta += 1
        if self.spike_at_step is not None and self.n_delta == self.spike_at_step:
            self.spike_steps = 2
        self.theta = (self.theta + float(np.clip(a[0], -0.15, 0.15))) % TWO_PI
        self.z = float(np.clip(self.z + np.clip(a[1], -0.03, 0.03), -self.pipe.length / 2, self.pipe.length / 2))
        press = float(np.clip(a[2], 0, 1))
        self.force = 25.0 * press
        if self.spike_steps > 0:
            self.spike_steps -= 1
            self.force = 65.0
        if self.force >= 5:
            uv = self.world.surface_to_uv(torch.tensor([self.theta]), torch.tensor([self.z]))
            self.world.abrade(uv, self.contact_radius, torch.tensor([self.force]), torch.tensor([0.3]), 0.1)

    def ee_theta_z(self):
        return (self.theta, self.z)

    def current_force(self):
        return self.force

    def back_off(self, distance=0.05):
        self.force = 0.0
        self.spike_steps = 0


class GreedyScrub:

    def __init__(self, pipe: PipeGeometry, dwell=3):
        self.pipe, self.dwell = pipe, dwell
        self.reset()

    def reset(self):
        self.done = set()
        self.cur, self.k = None, 0
        self._last_map = None

    def step(self, obs):
        rust = obs["perceived_rust"]
        if self._last_map is None or not np.array_equal(rust > 0, self._last_map):
            self.done, self.cur = set(), None
            self._last_map = rust > 0
        th, z = obs["ee_theta_z"]
        if self.cur is None:
            cand = [tuple(c) for c in np.argwhere(rust > 0) if tuple(c) not in self.done]
            if not cand:
                cand = [tuple(c) for c in np.argwhere(obs["target_mask"]) if tuple(c) not in self.done]
            if not cand:
                self.done = set()
                return np.array([0.0, 0.0, 0.0])
            cth, cz = self.pipe.cell_to_theta_z(*np.array(cand).T)
            d = (np.angle(np.exp(1j * (cth - th))) * self.pipe.radius) ** 2 + (cz - z) ** 2
            self.cur, self.k = cand[int(np.argmin(d))], 0
        tth, tz = self.pipe.cell_to_theta_z(*self.cur)
        dth = float(np.angle(np.exp(1j * (tth - th))))
        dz = float(tz - z)
        near = abs(dth) * self.pipe.radius < 0.008 and abs(dz) < 0.008
        if near:
            self.k += 1
            if self.k >= self.dwell:
                self.done.add(self.cur)
                self.cur = None
        return np.array([dth, dz, 0.7 if near else 0.3])


class MockBridge:
    def __init__(self, pipe, world, perception, arm, events=None, verbose=True):
        self._pipe, self.world, self.perception, self.arm = pipe, world, perception, arm
        self.t = 0.0
        self.events = sorted(events or [], key=lambda e: e[0])
        self.person_window = None
        self.hud = []
        self.verbose = verbose

    def sim_time(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        while self.events and self.events[0][0] <= self.t:
            _, name, fn = self.events.pop(0)
            fn()
            if self.verbose:
                print(f"   >>> [scene change t={self.t:5.1f}s] {name}")

    def pipe(self):
        return self._pipe

    def patrol_poses(self):
        return ["pipe_A_station_1"]

    def go_to_patrol_pose(self, pose):
        return True

    def approach(self, theta, z, standoff):
        return True

    def person_distance(self):
        if self.person_window and self.person_window[0] <= self.t < self.person_window[1]:
            return 2.4
        return None

    def scrub_obs(self):
        return {}

    def publish_overlay(self, lines):
        self.hud = lines


SCENE_META = "sim/output/rig_deck_meta.json"


def scene_geometry(meta_path: str = SCENE_META):
    import json
    import os
    meta = {"main_pipe": {"z_m": 1.1, "y_m": 0.0, "radius_m": 0.15},
            "damage": {"position_m": [-0.4933, 0.095, 1.2199], "surface_normal": [0.0, 0.6212, 0.7836]}}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    mp, dmg = meta["main_pipe"], meta["damage"]
    dx = float(dmg["position_m"][0])
    n = dmg["surface_normal"]
    theta = math.atan2(n[2], n[1]) % TWO_PI
    pipe = PipeGeometry(origin=[dx, mp["y_m"], mp["z_m"]], axis=[1, 0, 0], ref=[0, 1, 0],
                        radius=mp["radius_m"], length=0.8, n_z=40, n_theta=64)
    return pipe, theta


def build(log_path="runs/decisions.jsonl", glare=True, verbose=True, seed=0):
    pipe, dmg_theta = scene_geometry()
    world = MockRustWorld(H=pipe.n_z, W=pipe.n_theta, length=pipe.length, radius=pipe.radius)
    world.set_rust(world.blob(dmg_theta, 0.0, size=0.05, level=0.85))
    perception = MockPerception(world, pipe, dmg_theta)
    if not glare:
        perception.glare_views = set()
    arm = MockArm(world, pipe, seed=seed)
    arm.theta = dmg_theta
    bridge = MockBridge(pipe, world, perception, arm, verbose=verbose)
    bridge.dmg_theta = dmg_theta
    loc = RustLocalizer(fake_colour_classifier, patch=16, stride=8, threshold=0.5, temperature=1.0)
    orch = Orchestrator(bridge, perception, loc, world, arm, GreedyScrub(pipe),
                        OrchestratorConfig(log_path=log_path, scrub_steps=200, scrub_redetect_every=40))
    return orch, bridge, world, perception, arm, pipe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/decisions.jsonl")
    args = ap.parse_args(argv)
    t0 = time.time()
    orch, bridge, world, perception, arm, pipe = build(args.log)

    def on_spray_hook():
        t_spray = bridge.t
        th_new = bridge.dmg_theta + 0.75
        bridge.events.append((t_spray + 12.0, f"NEW rust patch spawned at theta={th_new:.2f}, z=+0.13",
                              lambda: world.inject_patch(th_new, 0.13, size=0.04, level=0.8)))
        bridge.events.sort(key=lambda e: e[0])

    arm.spike_at_step = 40
    spawned = {"done": False}
    while orch.state not in (S.DONE, S.ESCALATE_TO_HUMAN):
        prev = orch.state
        orch.step()
        if prev == S.SPRAY and not spawned["done"]:
            spawned["done"] = True
            on_spray_hook()
        if orch.state == S.VERIFY and prev == S.SCRUB and not getattr(perception, "_occl_done", False):
            perception._occl_done = True
            perception.occluded_frames = 2
            bridge.person_window = (bridge.t + 0.5, bridge.t + 2.5)
        if len(orch.records) > 400:
            break

    for r in orch.records:
        ev = r.evidence
        print(f"t={r.t:6.1f}s  {r.frm:>13} -> {r.to:<17} conf={ev.get('detector_conf')}  "
              f"rust={ev.get('rust_fraction')}  F={ev.get('force')}  | {r.reason}")
    print("\nHUD:", *orch.overlay_text(), sep="\n  ")
    print(f"\nfinal: {orch.state.value}  true rust remaining={float(world.fraction_remaining()[0]):.3f}  "
          f"painted cells={int(world.painted.sum())}  decisions={len(orch.records)} -> {args.log}  "
          f"wall={time.time() - t0:.1f}s")
    return orch


if __name__ == "__main__":
    main()
