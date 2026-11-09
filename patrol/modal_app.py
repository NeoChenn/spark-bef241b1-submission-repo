"""Train the patrol path-following policy on Modal CPU containers (no Isaac Sim needed).

  modal run --detach patrol/modal_app.py --runs nav_v1:0,nav_v1b:1 --iters 250
  modal volume get rust-ckpt patrol/nav_v1 runs/patrol/
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra.common import CKPT_DIR, CKPT_VOL  # noqa: E402

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.8.0", "numpy", "onnx", "onnxruntime", "onnxscript",
                    extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("patrol", "infra")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.8.0", "numpy", "onnx", "onnxruntime", "onnxscript")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("patrol", "infra")
)
app = modal.App("patrol-nav", image=image)


@app.function(image=gpu_image, gpu=["L4", "A10G"], cpu=4.0, memory=16384, timeout=3 * 3600, volumes={CKPT_DIR: CKPT_VOL})
def train_gpu(run: str, seed: int = 0, iters: int = 250, envs: int = 4096,
              init_from: str = "", overrides_json: str = "") -> dict:
    import json

    from patrol.ppo import train

    def log(msg):
        print(f"[{run}] {msg}", flush=True)

    init = f"{CKPT_DIR}/patrol/{init_from}/model.pt" if init_from else None
    res = train(iters=iters, envs=envs, out=f"{CKPT_DIR}/patrol/{run}", seed=seed, log=log, device="cuda",
                sim_overrides=json.loads(overrides_json) if overrides_json else None, init=init)
    CKPT_VOL.commit()
    return res


@app.function(cpu=16.0, memory=16384, timeout=3 * 3600, volumes={CKPT_DIR: CKPT_VOL})
def train_remote(run: str, seed: int = 0, iters: int = 250, envs: int = 1024,
                 init_from: str = "", overrides_json: str = "") -> dict:
    import json
    from patrol.ppo import train

    out = f"{CKPT_DIR}/patrol/{run}"

    def log(msg):
        print(f"[{run}] {msg}", flush=True)

    init = f"{CKPT_DIR}/patrol/{init_from}/model.pt" if init_from else None
    res = train(iters=iters, envs=envs, out=out, seed=seed, threads=16, log=log,
                sim_overrides=json.loads(overrides_json) if overrides_json else None, init=init)
    CKPT_VOL.commit()
    return res


@app.local_entrypoint()
def main(runs: str = "nav_v1:0", iters: int = 250, envs: int = 1024, init_from: str = "", overrides: str = "",
         gpu: bool = False):
    jobs = [r.split(":") for r in runs.split(",")]
    fn = train_gpu if gpu else train_remote
    calls = [fn.spawn(name, int(seed), iters, envs, init_from, overrides) for name, seed in jobs]
    for (name, _), c in zip(jobs, calls):
        res = c.get()
        print(name, {k: res[k] for k in ("ppo", "pure_pursuit_baseline")})
