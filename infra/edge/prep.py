from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 224


def crop_u8(im: Image.Image, box=None) -> np.ndarray:
    if box:
        im = im.crop(box)
    return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.uint8)


def calib_crops(render_dir: Path, exclude: set[str], n: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    files = sorted(p for p in render_dir.glob("*.png") if p.name not in exclude)
    ims = [Image.open(p).convert("RGB") for p in files]
    out = []
    for i in range(n):
        im = ims[i % len(ims)]
        w, h = im.size
        s = int(rng.integers(96, min(w, h) + 1))
        x, y = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
        out.append(crop_u8(im, (x, y, x + s, y + s)))
    return np.stack(out), [p.name for p in files]


def eval_set(render_dir: Path, synth_test: Path | None):
    from detection.eval import SIM_HELDOUT

    xs, ys, src = [], [], []
    for f, box, label, _ in SIM_HELDOUT:
        p = render_dir / f
        if label < 0 or not p.exists():
            continue
        xs.append(crop_u8(Image.open(p), box)), ys.append(label), src.append("heldout_render")
    if synth_test and synth_test.exists():
        for cls, label in (("clean", 0), ("rust", 1)):
            for p in sorted((synth_test / cls).glob("*.jpg")):
                xs.append(crop_u8(Image.open(p))), ys.append(label), src.append("synth_test")
    return np.stack(xs), np.asarray(ys, np.int64), np.asarray(src)


def policy_obs(onnx_path: Path, n: int, seed: int = 0) -> np.ndarray:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    dim = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    mean = next((v for k, v in init.items() if "mean" in k), np.zeros((1, dim), np.float32)).reshape(-1)
    std = next((v for k, v in init.items() if k.startswith("onnx::Div")), np.ones((1, dim), np.float32)).reshape(-1)
    rng = np.random.default_rng(seed)
    return (mean + np.abs(std) * rng.standard_normal((n, dim))).astype(np.float32)


def random_detector(out: Path):
    from detection.export import export_onnx
    from detection.models import build_model

    m, meta = build_model(lora=False, pretrained=False, verbose=False)
    export_onnx(m, out / "detector.onnx")
    meta.update({"threshold_logit": 0.0, "temperature": 1.0, "weights": "random"})
    (out / "detector_meta.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="runs/detector/best")
    ap.add_argument("--policy", default="runs/scrub_v2/policy.onnx")
    ap.add_argument("--renders", default="sim/output")
    ap.add_argument("--synth-test", default="runs/edge/data/rust_sim/test")
    ap.add_argument("--out", default="runs/edge/bundle")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--n-obs", type=int, default=2048)
    a = ap.parse_args(argv)
    out, det, renders = Path(a.out), Path(a.detector), Path(a.renders)
    out.mkdir(parents=True, exist_ok=True)
    if (det / "model.onnx").exists():
        shutil.copy(det / "model.onnx", out / "detector.onnx")
        meta = json.loads((det / "meta.json").read_text())
        meta.setdefault("weights", str(det))
        (out / "detector_meta.json").write_text(json.dumps(meta, indent=2))
    else:
        random_detector(out)
    shutil.copy(a.policy, out / "policy.onnx")
    from detection.eval import SIM_HELDOUT

    held = {f for f, *_ in SIM_HELDOUT}
    cx, cfiles = calib_crops(renders, held, a.n_calib)
    np.savez_compressed(out / "calib.npz", x=cx, files=np.asarray(cfiles))
    ex, ey, es = eval_set(renders, Path(a.synth_test))
    np.savez_compressed(out / "eval.npz", x=ex, y=ey, src=es)
    np.savez_compressed(out / "policy_obs.npz", obs=policy_obs(out / "policy.onnx", a.n_obs))
    print(json.dumps({"calib": list(cx.shape), "calib_files": cfiles, "eval": list(ex.shape),
                      "eval_pos": int(ey.sum()), "detector_weights": json.loads(
                          (out / "detector_meta.json").read_text()).get("weights")}, indent=1))


if __name__ == "__main__":
    main()
