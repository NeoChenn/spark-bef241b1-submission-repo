from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORAL = (217, 119, 87)
DARK = (7, 9, 13)
EDGE = (190, 222, 255)
SEAM = (120, 220, 255)
HDRI = os.path.join(ROOT, "sim/assets/third_party/hdris/belfast_sunset_puresky.hdr")
DECK_XY = (1.5, -1.9)


def surface_poly(pts):
    return [tuple(map(float, q)) for q in pts]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="runs/weld/layers")
    p.add_argument("--scene", default=os.path.join(ROOT, "sim/output/rig_deck.usda"))
    p.add_argument("--frames", type=int, default=225)
    p.add_argument("--stow", type=int, default=24)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--weld", default="10,180")
    p.add_argument("--cuts", default="36,48,90,102")
    p.add_argument("--subframes", type=int, default=4)
    p.add_argument("--dome", type=float, default=380.0)
    p.add_argument("--sun", type=float, default=900.0)
    p.add_argument("--key", type=float, default=2.0e4)
    p.add_argument("--arc", type=float, default=0.8)
    p.add_argument("--arc_light", type=float, default=0.12)
    p.add_argument("--bead_glow", type=float, default=1.0)
    p.add_argument("--rain", type=float, default=0.8)
    p.add_argument("--eye", default="1.55,1.1,1.5")
    p.add_argument("--at", default="0.0,-0.28,0.55")
    p.add_argument("--eye2", default="")
    p.add_argument("--at2", default="")
    p.add_argument("--focal", type=float, default=18.0)
    p.add_argument("--probe", type=int, default=0)
    p.add_argument("--renderer", default="RaytracedLighting")
    args = p.parse_args(argv)
    W, H, F = args.width, args.height, args.frames
    w0, w1 = (int(v) for v in args.weld.split(","))

    from policy.weld import clip, fx

    plan = clip.plan(frames=F, fps=args.fps, weld=(w0, w1))
    from integration.kinematics import ARM_STOWED, RobotModel

    model = RobotModel.from_urdf()
    r0 = w1 + 14
    q_end = dict(plan["frames"][min(r0, F - 1)]["q"])
    for fi in range(r0, F):
        u = min((fi - r0) / max(args.stow, 1), 1.0)
        u = u * u * (3 - 2 * u)
        q = {k: v + (ARM_STOWED[k] - v) * u if k in ARM_STOWED else v for k, v in q_end.items()}
        Tt = model.tool_pose(np.asarray(plan["base"]), q)
        plan["frames"][fi].update(q=q, tip=Tt[:3, 3].tolist(), axis=Tt[:3, 0].tolist(), arc=False)
    os.makedirs(args.out_dir, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": args.renderer, "width": W, "height": H})

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFilter
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    from integration import repair_scene as rsc
    from policy import night_look
    from policy.pickcan.layers import edges, project, wipe_mask
    from policy.pickcan.video import cv_pose, intrinsics

    sys.path.insert(0, os.path.join(ROOT, "sim"))
    import look

    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        app.update()
    st = ctx.get_stage()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")

    bd = st.DefinePrim("/World/Backdrop", "Xform")
    bd.GetReferences().AddReference(os.path.abspath(args.scene))
    UsdGeom.XformCommonAPI(bd).SetTranslate(Gf.Vec3d(-DECK_XY[0], -DECK_XY[1], 0.0))
    for path in ("/World/Backdrop/Robots", "/World/Backdrop/Pipes/damage_patch"):
        pr = st.GetPrimAtPath(path)
        if pr.IsValid():
            pr.SetActive(False)
    for pr in list(Usd.PrimRange(bd)):
        if pr.IsA(UsdPhysics.Scene):
            pr.SetActive(False)
    night_look.wet_down(st, "/World/Backdrop")
    sky = UsdLux.DomeLight(st.GetPrimAtPath("/World/Backdrop/Environment/sky"))
    if sky:
        sky.GetTextureFileAttr().Set(Sdf.AssetPath(HDRI))
        sky.GetIntensityAttr().Set(args.dome)
    sun = UsdLux.DistantLight(st.GetPrimAtPath("/World/Backdrop/Environment/sun"))
    if sun:
        sun.GetIntensityAttr().Set(args.sun)
        sun.GetColorAttr().Set(Gf.Vec3f(1.0, 0.58, 0.3))
        ops = UsdGeom.Xformable(sun.GetPrim()).GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3f(84.0, 0.0, 215.0))

    pipe = plan["pipe"]
    P0, A, R = np.asarray(pipe["origin"]), np.asarray(pipe["axis"]), float(pipe["radius"])
    steel = night_look._material(st, "/World/Looks/weld_pipe", (0.24, 0.25, 0.25), 0.5, 0.5)
    paint = night_look._material(st, "/World/Looks/weld_stand", (0.32, 0.2, 0.05), 0.6, 0.1)
    crack_mat = night_look._material(st, "/World/Looks/weld_crack", (0.004, 0.004, 0.004), 0.9, 0.0)
    cyl = UsdGeom.Cylinder.Define(st, "/World/Weld/pipe")
    cyl.CreateRadiusAttr(R)
    cyl.CreateHeightAttr(9.0)
    cyl.CreateAxisAttr("X")
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(*(P0 + A * 0.1)))
    rsc.bind(cyl.GetPrim(), steel)
    for i, sx in enumerate((-2.1, -0.9, 1.1, 2.3)):
        leg = UsdGeom.Cube.Define(st, f"/World/Weld/stand{i}")
        leg.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(leg)
        xf.SetTranslate(Gf.Vec3d(P0[0] + sx, P0[1], (P0[2] - R) / 2))
        xf.SetScale(Gf.Vec3f(0.12, 0.3, float(P0[2] - R)))
        rsc.bind(leg.GetPrim(), paint)

    def ribbon(path, pts, width, lift, mat):
        pts = np.asarray(pts, float)
        n = pts - P0 - np.outer((pts - P0) @ A, A)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        t = np.gradient(pts, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        s = np.cross(n, t)
        base = pts + n * lift
        v = np.concatenate([base - s * width / 2, base + s * width / 2])
        k = len(pts)
        idx = [[i, i + 1, k + i + 1, k + i] for i in range(k - 1)]
        m = UsdGeom.Mesh.Define(st, path)
        m.CreatePointsAttr([Gf.Vec3f(*q) for q in v])
        m.CreateFaceVertexCountsAttr([4] * (k - 1))
        m.CreateFaceVertexIndicesAttr([j for f in idx for j in f])
        m.CreateDoubleSidedAttr(True)
        rsc.bind(m.GetPrim(), mat)
        return m

    ribbon("/World/Weld/crack", plan["crack_world"], 0.0016, 0.0004, crack_mat)

    poser = rsc.RobotPoser(st, model, path="/World/Robot")
    night_look.paint_robot(st, "/World/Robot", "graphite")
    ee = next(q for q in Usd.PrimRange(st.GetPrimAtPath("/World/Robot")) if q.GetName() == "ur3_ee_link")
    for q in [q for q in Usd.PrimRange(ee) if q.GetName() == "scrubber_link"]:
        UsdGeom.Imageable(q).MakeInvisible()
    black = night_look._material(st, "/World/Looks/torch_black", (0.02, 0.02, 0.022), 0.5, 0.2)
    copper = night_look._material(st, "/World/Looks/torch_copper", (0.75, 0.42, 0.25), 0.28, 1.0)
    alu = night_look._material(st, "/World/Looks/torch_alu", (0.55, 0.56, 0.58), 0.35, 0.9)
    torch_root = str(ee.GetPath()) + "/weld_torch"
    for name, x0, x1, r, mat in (("body", 0.0, 0.07, 0.017, black), ("neck", 0.07, 0.096, 0.011, alu),
                                 ("nozzle", 0.096, 0.114, 0.0095, copper), ("tip", 0.114, 0.1195, 0.0028, copper)):
        c = UsdGeom.Cylinder.Define(st, f"{torch_root}/{name}")
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(x1 - x0)
        c.CreateAxisAttr("X")
        UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d((x0 + x1) / 2, 0, 0))
        rsc.bind(c.GetPrim(), mat)
    sensor = UsdGeom.Cube.Define(st, f"{torch_root}/laser_head")
    sensor.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(sensor).SetTranslate(Gf.Vec3d(0.055, 0.0, 0.034))
    UsdGeom.XformCommonAPI(sensor).SetScale(Gf.Vec3f(0.045, 0.03, 0.022))
    rsc.bind(sensor.GetPrim(), alu)

    base = np.asarray(plan["base"])
    work = P0 + np.array([0.0, 0.6, 0.6])
    night_look._lamp(st, "/World/WeldKey", tuple(work + [0.9, 0.9, 1.2]), (1.0, 0.86, 0.7), 0.12, args.key)
    night_look._lamp(st, "/World/WeldRim", tuple(work + [-1.4, -0.4, 0.9]), night_look.LED, 0.1, 0.35 * args.key)
    arc = fx.arc_light(st, "/World/Weld/arc", scale=args.arc_light)
    bead = fx.WeldBead(st, "/World/Weld/bead", emissive=60.0 * args.bead_glow, crown=0.003)
    for k_, v_ in (("diffuseColor", Gf.Vec3f(0.78, 0.74, 0.66)), ("metallic", 0.2), ("roughness", 0.35)):
        bead.shaders[-1].GetInput(k_).Set(v_)
    sparks, smoke = fx.Sparks(seed=1), fx.Smoke(seed=2)

    s = carb.settings.get_settings()
    for k_, v_ in {**look.TIERS["offline"]["rtx"], "/rtx/rendermode": args.renderer,
                   "/rtx/post/motionblur/enabled": False, "/rtx/post/dof/enabled": False}.items():
        s.set(k_, v_)
    for k_, v_ in (("/rtx/fog/enabled", True), ("/rtx/fog/fogZup/enabled", True),
                   ("/rtx/fog/fogColorIntensity", 1.0), ("/rtx/fog/fogStartDist", 4.0),
                   ("/rtx/fog/fogEndDist", 60.0), ("/rtx/fog/fogDistanceDensity", 0.55),
                   ("/rtx/fog/fogHeightDensity", 0.0)):
        s.set(k_, v_)
    s.set_float_array("/rtx/fog/fogColor", [0.42, 0.27, 0.17])

    cam = rsc.Camera(st, "/World/Cameras/weld", focal=args.focal)
    views = [(np.array([float(v) for v in e.split(",")]), np.array([float(v) for v in a.split(",")]))
             for e, a in zip(args.eye.split("/"), args.at.split("/"))]
    cam.look(*views[0])
    rp = rep.create.render_product(cam.path, (W, H))
    ann = {}
    for kind in ("rgb", "distance_to_image_plane", "normals"):
        a_ = rep.AnnotatorRegistry.get_annotator(kind)
        a_.attach(rp)
        ann[kind] = a_
    K = intrinsics(args.focal, cam.aperture, W, H)
    T = cv_pose(cam.world_matrix())
    sun_dir = look.sun_direction((84.0, 0.0, 215.0))
    for _ in range(20):
        app.update()

    def grab():
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=False)
        rgb = np.asarray(ann["rgb"].get_data())[..., :3].copy()
        depth = np.asarray(ann["distance_to_image_plane"].get_data()).squeeze().astype(np.float32)
        nrm = np.asarray(ann["normals"].get_data()).reshape(H, W, -1)[..., :3].astype(np.float32)
        return rgb, depth, nrm

    cuts = [int(c) for c in args.cuts.split(",")]
    rng = np.random.default_rng(0)
    streaks = np.column_stack([rng.uniform(0, W, 420), rng.uniform(0, H, 420), rng.uniform(0.5, 1.0, 420)])
    seen = []
    stills, log = {}, []
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    order = range(F) if not args.probe else [int(v) for v in np.linspace(0, F - 1, args.probe)]
    dt = 1.0 / args.fps
    sc = W / 1920.0

    def render_frame(fi, fr, tip, ax, normal, on, flick, T):
        rgb, depth, nrm = grab()
        full = look.inscatter(rgb, depth, {"K": K, "T": T}, "dusk", sun_dir, 1.0)
        full = look.grade(full, "offline", fi * dt, "dusk")
        im = Image.fromarray(full).convert("RGBA")
        rl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(rl)
        for sx, sy, sv in streaks:
            y0 = (sy + fi * 55 * sv * sc) % H
            x0 = (sx + fi * 9 * sv * sc) % W
            dr.line([x0, y0, x0 + 9 * sv * sc, y0 + 55 * sv * sc], fill=(205, 212, 225, int(40 * args.rain * sv)),
                    width=max(1, int(2 * sc)))
        full = np.asarray(Image.alpha_composite(im, rl.filter(ImageFilter.GaussianBlur(0.8))).convert("RGB"))
        full = smoke.draw(full, K, T, depth)
        full = sparks.draw(full, K, T, depth)
        if on:
            uv, z = project(tip, K, T)
            full = fx.glare(full, uv, float(z), depth, args.arc, flick)

        e = edges(nrm, depth)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 60)
        near = np.clip(1.6 - depth / 6.0, 0.35, 1.0) * valid
        wire = np.asarray(DARK, np.float32) + (np.asarray(EDGE, np.float32) - np.asarray(DARK, np.float32)) * (e * near)[..., None]
        wim = Image.fromarray(np.clip(wire, 0, 255).astype(np.uint8))

        l_ = np.asarray([0.55, 0.65, 0.5]) / np.linalg.norm([0.55, 0.65, 0.5])
        lam = np.clip(nrm @ l_, 0, 1)
        floor = np.where(nrm[..., 2] > 0.95, 0.3, 1.0)
        shade = ((0.2 + 0.75 * lam) * floor)[..., None] * np.asarray([150, 160, 176], np.float32)
        geo = np.where(valid[..., None], shade * np.clip(1.25 - depth / 8.0, 0.3, 1.0)[..., None], np.asarray(DARK, np.float32))
        gim = Image.fromarray(np.clip(geo, 0, 255).astype(np.uint8))
        dg = ImageDraw.Draw(gim, "RGBA")
        lw = max(1, int(round(3 * sc)))
        cuv, cz = project(np.asarray(plan["crack_world"]), K, T)
        dg.line(surface_poly(cuv[cz > 0.05]), fill=(20, 24, 30, 200), width=lw)
        tw = fr["track_world"]
        if len(tw) > 1:
            tuv, tz = project(np.asarray(tw), K, T)
            dg.line(surface_poly(tuv[tz > 0.05]), fill=(255, 255, 255, 255), width=lw + 2)
        if len(seen) > 1:
            suv, sz = project(np.asarray(seen), K, T)
            dg.line(surface_poly(suv[sz > 0.05]), fill=SEAM + (255,), width=max(1, lw - 1))
        lz = np.asarray([q if q is not None else [np.nan] * 3 for q in fr["laser_world"] or []], float).reshape(-1, 3)
        ok = np.isfinite(lz).all(1)
        if ok.sum() > 1 and on:
            emit = tip - ax * 0.055 + normal * 0.0 + np.asarray(fr["laser_dir"]) * 0.0
            emit = emit + normal * 0.03
            euv, _ = project(emit, K, T)
            luv, _ = project(lz[ok], K, T)
            fan = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            df = ImageDraw.Draw(fan)
            df.polygon([tuple(euv), tuple(luv[0]), tuple(luv[-1])], fill=CORAL + (34,))
            gim = Image.alpha_composite(gim.convert("RGBA"), fan).convert("RGB")
            dg = ImageDraw.Draw(gim, "RGBA")
            dg.line(surface_poly(luv), fill=CORAL + (255,), width=max(1, int(round(4 * sc))))
            for q in luv[::4]:
                dg.ellipse([q[0] - 2 * sc, q[1] - 2 * sc, q[0] + 2 * sc, q[1] + 2 * sc], fill=CORAL + (255,))
        if fr["center_est_world"] is not None and on:
            cuv1, _ = project(np.asarray(fr["center_est_world"]), K, T)
            r_ = 9 * sc
            dg.ellipse([cuv1[0] - r_, cuv1[1] - r_, cuv1[0] + r_, cuv1[1] + r_], outline=SEAM + (255,), width=lw)
        tuv1, _ = project(tip, K, T)
        r_ = 7 * sc
        dg.ellipse([tuv1[0] - r_, tuv1[1] - r_, tuv1[0] + r_, tuv1[1] + r_], fill=(255, 255, 255, 255))
        layers = [np.asarray(wim, np.float32), np.asarray(gim, np.float32), np.asarray(full, np.float32)]
        return layers

    for fi in order:
        fr = plan["frames"][fi]
        poser.set(base, fr["q"])
        tip, ax = np.asarray(fr["tip"]), np.asarray(fr["axis"])
        normal = -ax / np.linalg.norm(ax)
        travel = A
        on = bool(fr["arc"])
        flick = fx.arc_flicker(fi * dt, seed=3)
        arc(tip + normal * 0.002, on, flick, normal=normal, dt=dt)
        bw = fr["bead_world"]
        if len(bw) > 1:
            bead.update(np.asarray(bw), None, np.asarray(fr["bead_width_m"]), np.asarray(fr["bead_age_s"]), pipe=(P0, A))
        sparks.step(dt, tip, normal, travel, on, pipe=(P0, A, R))
        smoke.step(dt, tip, on)
        if fr["center_est_world"] is not None:
            seen.append(fr["center_est_world"])
        if args.probe:
            for vi, view in enumerate(views):
                cam.look(*view)
                for k_, L_ in zip(("wire", "geo", "full"), render_frame(fi, fr, tip, ax, normal, on, flick, cv_pose(cam.world_matrix()))):
                    Image.fromarray(L_.astype(np.uint8)).save(os.path.join(args.out_dir, f"probe_{fi:03d}_v{vi}_{k_}.jpg"), quality=92)
            continue
        if args.eye2:
            u = fi / max(F - 1, 1)
            u = 1 - (1 - u) ** 2.5
            e2 = np.array([float(v) for v in args.eye2.split(",")])
            a2 = np.array([float(v) for v in args.at2.split(",")])
            cam.look(views[0][0] + (e2 - views[0][0]) * u, views[0][1] + (a2 - views[0][1]) * u)
            T = cv_pose(cam.world_matrix())
        layers = render_frame(fi, fr, tip, ax, normal, on, flick, T)
        if fi < cuts[0]:
            out = layers[0]
        elif fi < cuts[1]:
            m = wipe_mask(W, H, (fi - cuts[0]) / (cuts[1] - cuts[0]))
            out = layers[1] * m + layers[0] * (1 - m)
        elif fi < cuts[2]:
            out = layers[1]
        elif fi < cuts[3]:
            m = wipe_mask(W, H, (fi - cuts[2]) / (cuts[3] - cuts[2]))
            out = layers[2] * m + layers[1] * (1 - m)
        else:
            out = layers[2]
        arr = np.clip(out, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(frames_dir, f"{fi:04d}.png"), compress_level=1)
        for key, at_ in (("wire", cuts[0] // 2), ("geo", (cuts[1] + cuts[2]) // 2), ("full", (cuts[3] + w1) // 2)):
            if fi == at_:
                stills[key] = arr
        log.append({"frame": fi, "arc": on, "progress_mm": fr["progress_mm"], "tip": np.round(tip, 4).tolist()})
        print(f"[weld-layers] frame {fi}/{F}", flush=True)
    if len(stills) == 3:
        strip = np.concatenate([np.asarray(Image.fromarray(stills[k_]).resize((W // 2, H // 2))) for k_ in ("wire", "geo", "full")], 1)
        Image.fromarray(strip).save(os.path.join(args.out_dir, "weld_layers.png"))
    with open(os.path.join(args.out_dir, "layers.json"), "w") as f:
        json.dump({"metrics": plan.get("metrics"), "frames": log}, f)
    print("[weld-layers] RESULT " + json.dumps({"frames": F, "metrics": plan.get("metrics")}), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
