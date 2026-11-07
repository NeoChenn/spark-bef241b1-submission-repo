import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libxrender1", "libxxf86vm1", "libxfixes3", "libxi6", "libxkbcommon0", "libsm6", "libgl1", "libegl1")
    .pip_install("bpy==4.2.0")
    .add_local_dir(REPO / "sim", "/root/sim", ignore=["**/__pycache__", "output/*.png"])
)
app = modal.App("rig-render-blender", image=image)


@app.function(gpu="L4", timeout=1800, cpu=4, memory=16384)
def render(usd_rel: str, cameras: list[str], width: int = 1280, height: int = 720, samples: int = 64) -> dict:
    import time

    import bpy

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=f"/root/{usd_rel}", import_cameras=True, import_lights=True,
                          import_materials=True, import_usd_preview=True, set_material_blend=True)
    t_import = time.time() - t0

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    backend = None
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                backend = kind
                break
        except TypeError:
            continue
    scene.cycles.device = "GPU" if backend else "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.image_settings.file_format = "PNG"

    cams = {o.name: o for o in scene.objects if o.type == "CAMERA"}
    if not any(o.type == "LIGHT" for o in scene.objects) and scene.world is None:
        scene.world = bpy.data.worlds.new("sky")
        scene.world.color = (0.6, 0.65, 0.7)

    out = {"backend": backend or "CPU", "import_s": round(t_import, 1), "cameras_found": sorted(cams), "frames": {}}
    for name in cameras:
        cam = cams.get(name) or next((o for n, o in cams.items() if name in n), None)
        if cam is None:
            out["frames"][name] = None
            continue
        scene.camera = cam
        path = f"/tmp/{name}.png"
        scene.render.filepath = path
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        out["frames"][name] = {"seconds": round(time.time() - t1, 1), "png": open(path, "rb").read()}
    return out


@app.local_entrypoint()
def main(usd: str = "sim/output/rig_deck.usda", camera: str = "repair_closeup,hero", out: str = "runs/render",
         samples: int = 64):
    res = render.remote(usd, camera.split(","), samples=samples)
    outdir = REPO / out
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"backend={res['backend']} import={res['import_s']}s cameras in scene: {res['cameras_found']}")
    for name, f in res["frames"].items():
        if f is None:
            print(f"  {name}: camera not found")
            continue
        p = outdir / f"blender_{name}.png"
        p.write_bytes(f["png"])
        print(f"  {name}: {f['seconds']}s -> {p.relative_to(REPO)}")
