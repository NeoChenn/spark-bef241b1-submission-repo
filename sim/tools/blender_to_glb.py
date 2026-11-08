import sys
from pathlib import Path

import addon_utils
import bpy

src, dst = (Path(p) for p in sys.argv[sys.argv.index("--") + 1 :][:2])
suffix = src.suffix.lower()

if suffix == ".blend":
    bpy.ops.wm.open_mainfile(filepath=str(src))
    bpy.ops.file.find_missing_files(directory=str(src.parent))
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    importers = {
        ".dae": (None, lambda p: bpy.ops.wm.collada_import(filepath=p)),
        ".3ds": ("io_scene_3ds", lambda p: bpy.ops.import_scene.max3ds(filepath=p)),
        ".x3d": ("io_scene_x3d", lambda p: bpy.ops.import_scene.x3d(filepath=p)),
        ".wrl": ("io_scene_x3d", lambda p: bpy.ops.import_scene.x3d(filepath=p)),
        ".abc": (None, lambda p: bpy.ops.wm.alembic_import(filepath=p)),
    }
    addon, run = importers[suffix]
    if addon:
        addon_utils.enable(addon, default_set=True)
    run(str(src))

dst.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(dst),
    export_format="GLB",
    export_apply=True,
    export_yup=True,
    export_image_format="AUTO",
    export_cameras=False,
    export_lights=False,
    export_animations=False,
)
print(f"BLENDER_OK {dst}")
