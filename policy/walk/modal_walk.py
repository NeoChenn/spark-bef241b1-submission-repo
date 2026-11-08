from __future__ import annotations

import os
import subprocess
import sys
import time

import modal

from infra.common import CKPT_DIR, CKPT_VOL
from infra.isaac_image import ISAAC_GPU, ISAACLAB_PY, isaac_env, isaac_image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
image = (isaac_image.add_local_dir(os.path.join(ROOT, "integration"), "/root/integration")
         .add_local_dir(os.path.join(ROOT, "sim/assets/go2_ur3"), "/root/sim/assets/go2_ur3"))
app = modal.App("rig-walk-train")


@app.function(image=image, gpu=ISAAC_GPU, cpu=8, memory=32768, timeout=6 * 3600, volumes={CKPT_DIR: CKPT_VOL})
def train(run: str, num_envs: int, iters: int, extra: list[str]) -> dict:
    run_dir = f"{CKPT_DIR}/walk/{run}"
    os.makedirs(run_dir, exist_ok=True)
    cmd = [ISAACLAB_PY, "-m", "policy.walk.train_walk", "--num_envs", str(num_envs), "--iters", str(iters),
           "--run_dir", run_dir, "--headless", *extra]
    proc = subprocess.Popen(cmd, cwd="/root", env=isaac_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last, done = time.time(), False
    with open(os.path.join(run_dir, "train.log"), "a") as log:
        for line in proc.stdout:
            if "Unresolved reference" in line:
                continue
            sys.stdout.write(line)
            log.write(line)
            done = done or "TRAINING_DONE" in line
            if time.time() - last > 120:
                log.flush()
                CKPT_VOL.commit()
                last = time.time()
    proc.wait()
    CKPT_VOL.commit()
    return {"rc": proc.returncode, "done": done, "run_dir": run_dir}


@app.local_entrypoint()
def main(run: str = "", num_envs: int = 4096, iters: int = 1500, resume: str = ""):
    run = run or time.strftime("walk_%m%d_%H%M")
    extra = ["--resume", resume] if resume else []
    print(train.remote(run, num_envs, iters, extra))
