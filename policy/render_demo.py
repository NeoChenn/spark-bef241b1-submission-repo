"""Demo render: the Go2 + lift + UR3 scrubber running a trained SCRUB policy on the rig deck (sim/output/rig_deck.usda),
with the rust grid drawn as a textured overlay on the main pipe. One episode per shot, MP4 per shot + combined.

    infra/watson/run.sh policy.render_demo --worker --ckpt /repo/<run>/model_149.pt --out_dir /repo/runs/scrub_demo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEX_H, TEX_W = 512, 1024
DYN_NAME = "rust_overlay"
# env frame: robot trunk at the origin facing +x, pipe axis along x at y = -0.5, z = 1.1; damage spot ~(0, -0.41, 1.22)
SHOTS = {
    "close": dict(focal=45.0, keys=[
        (0.0, (0.80, -0.30, 1.68), (0.0, -0.42, 1.20)),
        (15.0, (0.62, -0.34, 1.62), (0.0, -0.42, 1.20))]),
    "medium": dict(focal=24.0, keys=[
        (0.0, (2.6, 2.1, 1.75), (0.0, -0.25, 0.8)),
        (12.0, (2.3, 2.4, 1.7), (-0.1, -0.25, 0.8))]),
    "push": dict(focal=28.0, keys=[
        (0.0, (5.5, 4.2, 3.0), (0.0, -0.3, 0.7)),
        (10.0, (2.1, 1.5, 1.75), (0.0, -0.35, 1.0))]),
}


def build_overlay(stage, base, tex_dir: str):
    """Sleeve mesh around the rust-grid segment + OmniPBR material fed by a dynamic texture."""
    import numpy as np
    import omni.kit.commands
    import omni.ui as ui
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    from policy.rust_overlay import sleeve_mesh

    o = (base.pipe_o[0] + base.scene.env_origins[0]).cpu().numpy()
    a, e, b = (t[0].cpu().numpy() for t in (base.pipe_a, base.pipe_e, base.pipe_b))
    r = float(base.pipe_r[0]) + 0.004
    pts, counts, idx, uv = sleeve_mesh(o, a, e, b, r, base.cfg.pipe_length, n_theta=128, n_z=32)
    mesh = UsdGeom.Mesh.Define(stage, "/World/RustOverlay")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    n = pts - o - ((pts - o) @ a)[:, None] * a
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy((n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    mpath = "/World/Looks/RustOverlay"
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=mpath)
    shader = UsdShade.Shader(stage.GetPrimAtPath(mpath + "/Shader"))
    tex = f"dynamic://{DYN_NAME}"
    for name, typ, val in (
            ("diffuse_texture", Sdf.ValueTypeNames.Asset, tex),
            ("enable_opacity", Sdf.ValueTypeNames.Bool, True),
            ("enable_opacity_texture", Sdf.ValueTypeNames.Bool, True),
            ("opacity_texture", Sdf.ValueTypeNames.Asset, tex),
            ("opacity_mode", Sdf.ValueTypeNames.Int, 0),
            ("opacity_threshold", Sdf.ValueTypeNames.Float, 0.5),
            ("reflection_roughness_constant", Sdf.ValueTypeNames.Float, 0.85),
            ("metallic_constant", Sdf.ValueTypeNames.Float, 0.0),
            ("normalmap_texture", Sdf.ValueTypeNames.Asset, os.path.join(tex_dir, "nor_gl.jpg")),
            ("bump_factor", Sdf.ValueTypeNames.Float, 0.6),
            ("texture_scale", Sdf.ValueTypeNames.Float2, (1.0, 1.0))):
        shader.CreateInput(name, typ).Set(val)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(mpath)))
    return ui.DynamicTextureProvider(DYN_NAME)


class RustPainter:
    def __init__(self, provider, tex_dir: str, flip_v: bool):
        import numpy as np
        from PIL import Image

        from policy.rust_overlay import periodic_noise, tile_texture

        self.provider, self.flip_v = provider, flip_v
        tiles_u, tiles_v = 3, 2
        img = Image.open(os.path.join(tex_dir, "diff.jpg")).convert("RGB")
        img = img.resize((TEX_W // tiles_u, TEX_H // tiles_v), Image.LANCZOS)
        lin = (np.asarray(img, np.float32) / 255.0) ** 2.2 * 255.0  # dynamic textures are read as linear
        self.tex = tile_texture(lin.astype(np.uint8), TEX_H, TEX_W, tiles_v, tiles_u)
        self.noise = periodic_noise(TEX_H, TEX_W, seed=3)

    def update(self, grid):
        from policy.rust_overlay import grid_to_mask, rust_rgba

        rgba = rust_rgba(grid_to_mask(grid, TEX_H, TEX_W, flip_v=self.flip_v), self.tex, self.noise)
        self.provider.set_data_array(rgba, [TEX_W, TEX_H])


def seed_rust(base, seed: int, abrasion: float):
    """Two larger patches than the training mean (still inside its size range) so the scrub is visible."""
    import torch

    from policy.envs.rust_state import random_patches

    rs = base.rust
    g = torch.Generator(device=base.device).manual_seed(seed)
    rust = random_patches(1, rs.H, rs.W, rs.radius[:1], rs.length, n_min=2, n_max=2, size_range=(0.06, 0.09),
                          device=base.device, generator=g, theta_window=rs.theta_window) * rs.reachable[:1].float()
    rs.reset(torch.tensor([0], device=base.device), rust=rust, soak_time=60.0, abrasion_coef=abrasion)


def draw_hud(rgb, lines, font, small):
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, w = 28, im.height - 28 - 34 - 26 * (len(lines) - 1), 640
    d.rounded_rectangle([x0 - 14, y0 - 12, x0 + w, im.height - 22], radius=8, fill=(10, 14, 18, 165))
    state, color = lines[0]
    d.text((x0, y0), state, fill=color, font=font)
    for k, line in enumerate(lines[1:]):
        d.text((x0, y0 + 36 + 26 * k), line, fill=(230, 232, 235), font=small)
    return np.asarray(im)


def worker(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--meta", default=os.path.join(ROOT, "sim/output/rig_deck_meta.json"))
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--shots", default="push,close,medium")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--robot", default="lift", choices=["lift", "lift_v2"])
    p.add_argument("--patch_when", type=float, default=0.25, help="inject a new patch once rust left < this")
    p.add_argument("--abrasion", type=float, default=0.7, help="abrasion coefficient (training DR 0.7-1.3)")
    p.add_argument("--preroll", type=float, default=1.5)
    p.add_argument("--renderer", default="RaytracedLighting", choices=["RaytracedLighting", "PathTracing",
                                                                        "RealTimePathTracing"])
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--subframes", type=int, default=1)
    p.add_argument("--max_s", type=float, default=18.0)
    p.add_argument("--still", action="store_true", help="render a few frames per shot only (look-dev)")
    p.add_argument("--no_flip_v", action="store_true")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(argv)
    args.headless, args.enable_cameras = True, True
    app = AppLauncher(args).app

    import carb
    import gymnasium as gym
    import imageio
    import numpy as np
    import omni.usd
    import torch
    from PIL import ImageFont
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ViewerCfg
    from isaaclab.sim import RenderCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from policy.envs.scrub_env import TASKS
    from policy.envs import scrub_env_cfg as C
    from policy.envs.scrub_env_cfg import ScrubPPORunnerCfg
    from policy.play_scrub import apply_meta
    from policy.rust_overlay import camera_path, rig_translation

    task, cfg_name = TASKS[args.robot]
    env_cfg = getattr(C, cfg_name)()
    env_cfg.scene.num_envs = 1
    meta = apply_meta(env_cfg, args.meta)
    env_cfg.prob_new_patch = 0.0
    env_cfg.prob_not_soaked = 0.0
    env_cfg.init_joint_noise = 0.0
    env_cfg.episode_length_s = 120.0
    env_cfg.viz = False
    env_cfg.spawn_pipe_visual = False
    env_cfg.backdrop_usd = os.path.abspath(args.scene)
    env_cfg.backdrop_translation = tuple(rig_translation(meta))
    env_cfg.viewer = ViewerCfg(eye=(2.0, 2.0, 1.6), lookat=(0, -0.4, 1.1), resolution=(1280, 720), origin_type="world")
    env_cfg.sim.render = RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", carb_settings={
        "/rtx/rendermode": args.renderer, "/rtx/pathtracing/spp": args.spp,
        "/rtx/pathtracing/totalSpp": args.spp, "/rtx/pathtracing/optixDenoiser/enabled": True,
        "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False})
    agent_cfg = ScrubPPORunnerCfg()
    agent_cfg.device = "cuda:0"

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    print("[demo] env origin", base.scene.env_origins[0].tolist(), "backdrop", env_cfg.backdrop_translation, flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=base.device)

    get_dones = base._get_dones

    def no_terminate():
        s, t = get_dones()
        return torch.zeros_like(s), torch.zeros_like(t)

    base._get_dones = no_terminate

    stage = omni.usd.get_context().get_stage()
    for path in ("/World/Backdrop/Pipes/damage_patch",):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
    tex_dir = os.path.join(ROOT, "sim/assets/third_party/textures/rust_coarse_01")
    painter = RustPainter(build_overlay(stage, base, tex_dir), tex_dir, flip_v=not args.no_flip_v)
    cam = stage.GetPrimAtPath(env_cfg.viewer.cam_prim_path)
    settings = carb.settings.get_settings()
    for key, val in env_cfg.sim.render.carb_settings.items():
        settings.set(key, val)
    font_dir = "/usr/share/fonts/truetype/dejavu"
    font = ImageFont.truetype(os.path.join(font_dir, "DejaVuSans-Bold.ttf"), 26)
    small = ImageFont.truetype(os.path.join(font_dir, "DejaVuSans.ttf"), 19)
    origin = base.scene.env_origins[0].cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    zero = torch.zeros(1, 4, device=base.device)
    dt = base.step_dt

    def render_frame():
        for _ in range(args.subframes - 1):
            base.sim.render()
        return base.render()

    clips = {}
    for shot in args.shots.split(","):
        spec = SHOTS[shot]
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            base.reset(seed=args.seed)
            seed_rust(base, args.seed, args.abrasion)
        t_patch = None
        obs = env.get_observations()
        cam.GetAttribute("focalLength").Set(spec["focal"])
        painter.update(base.rust.rust[0].cpu().numpy())
        eye, look = camera_path(spec["keys"], 0.0)
        base.sim.set_camera_view((eye + origin).tolist(), (look + origin).tolist())
        for _ in range(30):
            base.sim.render()
        frames, info, t0 = [], [], time.time()
        n_pre = int(args.preroll / dt)
        done_at, retract_at, k = None, None, 0
        max_steps = int(args.max_s / dt) if not args.still else int((args.preroll + 2.0) / dt)
        while k < max_steps:
            t = k * dt
            frac = float(base.rust.fraction_remaining()[0])
            fired = t_patch is not None
            if not fired and k >= n_pre and frac < args.patch_when:
                with torch.inference_mode():
                    base.rust.add_patch(torch.tensor([0], device=base.device), size_range=(0.05, 0.08))
                t_patch, frac = t, float(base.rust.fraction_remaining()[0])
            if done_at is None and fired and frac < env_cfg.success_fraction:
                done_at = t
            if k < n_pre:
                act, state = zero, ("APPROACH  brush to rust", (120, 200, 255))
            elif done_at is None or t < done_at + 0.8:
                with torch.inference_mode():
                    act = policy(obs)
                state = ("SCRUB  removing rust", (120, 255, 140))
                if fired and t - t_patch < 2.0:
                    state = ("RE-PLAN  new rust patch", (255, 190, 60))
                if done_at is not None:
                    state = ("VERIFY  segment clean", (120, 200, 255))
            elif retract_at is None or t < retract_at + 0.5:
                retract_at = retract_at if retract_at is not None else t
                th = float(base.theta[0])
                act = torch.tensor([[0.0, math.cos(th), math.sin(th), -1.0]], device=base.device)
                state = ("DONE  retract brush", (120, 200, 255))
            else:
                act = zero
                state = (("DONE  segment clean", (120, 200, 255)) if frac < 0.01 else
                         (f"DONE  {100 * frac:.0f} % residual flagged for touch-up", (255, 190, 60)))
                if t > retract_at + 2.5:
                    break
            with torch.inference_mode():
                obs, _, _, _ = env.step(act)
            painter.update(base.rust.rust[0].cpu().numpy())
            eye, look = camera_path(spec["keys"], t)
            base.sim.set_camera_view((eye + origin).tolist(), (look + origin).tolist())
            rgb = render_frame()
            force = float(base.force[0])
            lines = [state, f"rust remaining   {100 * frac:5.1f} %",
                     f"contact force    {force:5.1f} N   (limit 40 N)",
                     f"t = {t:4.1f} s     policy: PPO scrub (Isaac Lab)"]
            info.append({"t": round(t, 3), "frac": frac, "force": force, "state": state[0]})
            if rgb is not None and rgb.size and k > 0:
                frames.append(draw_hud(rgb, lines, font, small))
            k += 1
        wall = time.time() - t0
        frames = [f for f in frames if f.mean() > 3]
        fps = int(round(1 / dt))
        path = os.path.join(args.out_dir, f"demo_{shot}.mp4")
        imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=8)
        for q in (0.1, 0.35, 0.6, 0.9):
            imageio.imwrite(os.path.join(args.out_dir, f"key_{shot}_{int(q * 100):02d}.png"), frames[int(q * (len(frames) - 1))])
        with open(os.path.join(args.out_dir, f"demo_{shot}.json"), "w") as f:
            json.dump({"wall_s": round(wall, 1), "frames": len(frames), "fps": fps, "ckpt": args.ckpt,
                       "renderer": args.renderer, "spp": args.spp, "subframes": args.subframes, "log": info}, f)
        clips[shot] = frames
        print(f"[demo] {shot}: {len(frames)} frames in {wall:.0f}s -> {path}", flush=True)
    if len(clips) > 1:
        out = [f for fr in clips.values() for f in fr]
        imageio.mimsave(os.path.join(args.out_dir, "demo_combined.mp4"), out, fps=fps, quality=8, macro_block_size=8)
    print(f"[demo] settings rendermode={settings.get('/rtx/rendermode')}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        worker(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
