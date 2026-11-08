import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = """
import json, os, sys
sys.path[:] = [os.getcwd()] + [p for p in sys.path if p and os.path.abspath(p) != os.environ["REPO"]]
from fastapi.testclient import TestClient
from rig.web.server import create_app, default_fleet
with TestClient(create_app(sim_port=0, fleet=default_fleet())) as c:
    out = {p: c.get(p).status_code for p in ("/api/fleet", "/api/fleet/map")}
    out["mock"] = [c.get(p).status_code for p in ("/api/mock", "/api/mock/shot/1", "/api/mock/shot/3")]
    out["bench"] = sorted(b["name"] for b in c.get("/api/bench").json())
    import policy.fleet
    from rig.web.server import default_control
    out["control"] = type(default_control()).__name__
    out["from_image"] = policy.fleet.__path__[0].startswith(os.getcwd())
    print(json.dumps(out))
"""


def image_tree(dst: Path) -> Path:
    app = dst / "app"
    for line in (REPO / "docker/web.Dockerfile").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY" or parts[1].startswith("--from"):
            continue
        *srcs, to = parts[1:]
        for src in (p for s in srcs for p in sorted(REPO.glob(s))):
            target = app / to
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                (target if to.endswith("/") else target.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy(src, target / src.name if to.endswith("/") else target)
    return app


@pytest.mark.parametrize("fleet", ["off", "judges"])
def test_web_image_serves_fleet_and_bench(tmp_path, fleet):
    app = image_tree(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "RIG_FLEET": fleet, "HOME": str(tmp_path), "REPO": str(REPO)}
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=app, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["/api/fleet"] == 200 and got["/api/fleet/map"] == 200 and got["from_image"]
    assert {"orch", "detect", "safety", "scrub"} <= set(got["bench"])
    assert got["mock"] == [200, 200, 200]
    assert got["control"] == "ControlCentre"
