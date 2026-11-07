# Rig Sim Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a fixed Isaac Sim camera, running headless in Docker on watson, to a web page at `https://rig.zeecloud.org`.

**Architecture:** Three compose services run on host networking.
- `sim` (Isaac Sim 6.2.0) renders a simple scene. It JPEG-encodes the camera's RGB frames and sends them over TCP to `127.0.0.1:9000`.
- `web` (FastAPI) accepts that TCP stream, keeps only the latest frame, and pushes it to browsers over a WebSocket on `127.0.0.1:8080`.
- `tunnel` (cloudflared) publishes `:8080` as `rig.zeecloud.org`.

A `fakesim` sends synthetic frames, so `web` and the tunnel can be tested without Isaac.

**Tech Stack:**
- Python 3.12 and uv
- FastAPI and uvicorn
- Pillow, plus numpy in the sim
- Isaac Sim 6.2.0 (`omni.replicator.core`, `pxr`)
- Plain HTML/JS
- Docker Compose with CDI GPU devices
- cloudflared 2026.9.3

**Spec:** `docs/superpowers/specs/2026-11-07-rig-sim-infra-design.md`

## Global Constraints

- Host: `watson` (aarch64 GB10). Clone it to `~/Documents/spark-bef241b1-submission-repo`. Touch nothing on watson outside that clone, except this project's own Docker images and volumes. Don't use `sudo` or `brev port-forward`. If a host-level fix seems needed, stop and ask.
- Compose project name is `rig`. Every named volume is prefixed `rig-`.
- Image: `nvcr.io/nvidia/isaac-sim:6.2.0`, with env `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y` and `NVIDIA_DRIVER_CAPABILITIES=all`. The entrypoint is `/isaac-sim/python.sh`.
- GPU: `devices: [nvidia.com/gpu=all]` (CDI).
- Ports: web on `127.0.0.1:8080`, frame ingest on `127.0.0.1:9000`. Bind to loopback only and never to `0.0.0.0`.
- Cloudflared: `cloudflare/cloudflared:2026.9.3`, running `tunnel --no-autoupdate run`. It reads `TUNNEL_TOKEN` from the gitignored `.env`, and its compose profile is `public`.
- Commits: no co-author or "Generated with" lines. No narrative code comments.
- Python tooling uses `uv`. Node isn't used.
- The sim's Python code imports only stdlib, numpy, PIL, `pxr`, `omni.*`, `carb`, `isaacsim` and `rig.frames`/`rig.sim.*`. It must never import fastapi.

## Review Focus

1. **The page is served over HTTPS by the tunnel**, so the WebSocket must use `wss://` when the page is `https:`. Otherwise the browser blocks it as mixed content. This is pinned by the `test_viewer_uses_page_scheme` test in Task 2 and checked by hand in Task 7.
2. **The sim starts before `web`, or `web` restarts mid-run.** The sender must keep retrying without blocking the render loop. Pinned in Task 4 by `test_send_without_server_does_not_block` and `test_reconnects_after_server_restart`.
3. **A slow browser or tab on a bad link** must not make memory grow or delay other viewers. Each client only ever gets the newest frame. Pinned in Task 2 by `test_slow_client_gets_latest_only`.
4. **A second sim (or fakesim) connects while one is already connected.** The newest connection wins and the old one is closed. Pinned in Task 2 by `test_new_sim_replaces_old`.
5. **First-run shader compile shows black frames for minutes.** `/api/health` reports frame age and the page shows "waiting for frames" instead of looking broken. Pinned in Task 2 by `test_health_before_frames` and in Task 7 by checking `/api/frame.jpg` twice for non-black, changing frames.

---

## File Structure

```
pyproject.toml              uv project "rig": fastapi, uvicorn[standard], pillow; dev: pytest, httpx
uv.lock
rig/__init__.py             empty
rig/frames.py               wire format, stdlib only (shared by sim, web, fakesim)
rig/web/__init__.py
rig/web/server.py           create_app(), FrameHub, TCP ingest, routes
rig/web/static/index.html   canvas + status line + fps
rig/web/static/viewer.js    WebSocket -> canvas, fps, reconnect
rig/fakesim.py              synthetic moving frames -> ingest (uses Pillow)
rig/sim/__init__.py
rig/sim/sender.py           FrameSender: background TCP client, latest-only, stdlib only
rig/sim/scene.py            build_scene(stage) -> Scene (Isaac/USD only)
rig/sim/main.py             SimulationApp loop: render -> JPEG -> FrameSender
docker/web.Dockerfile       python:3.12-slim + uv; used by web and fakesim
docker/sim.Dockerfile       FROM isaac-sim:6.2.0
docker-compose.yml          name: rig; services sim, web, fakesim (profile fake), tunnel (profile public)
scripts/remote.sh           check | up | down | logs | status | frame
.env.example                TUNNEL_TOKEN=
.gitignore                  .env, .venv, __pycache__, logs/
README.md                   first run on watson, Cloudflare steps, troubleshooting
tests/test_frames.py
tests/test_server.py
tests/test_sender.py
```

The tests run on the Mac with `uv run pytest`. Isaac-only code (`scene.py`, `main.py`) is verified on watson in Tasks 5 and 7.

---

### Task 1: Project skeleton and frame wire format

**Files:**
- Create: `pyproject.toml`, `rig/__init__.py`, `rig/frames.py`, `.gitignore`
- Test: `tests/test_frames.py`

**Interfaces:**
- Produces, in `rig/frames.py`:
  - `PREFIX = struct.Struct("<II")`
  - `MAX_HEADER = 64 * 1024`
  - `MAX_PAYLOAD = 16 * 1024 * 1024`
  - `encode(header: dict, payload: bytes = b"") -> bytes`: `PREFIX.pack(len(h), len(payload)) + h + payload`, where `h` is compact JSON utf-8.
  - `read_message(sock: socket.socket) -> tuple[dict, bytes]`: blocking. Raises `ConnectionError` on EOF or oversize.
  - `async read_message_async(reader: asyncio.StreamReader) -> tuple[dict, bytes]`: same rules. Turns `asyncio.IncompleteReadError` into `ConnectionError`.
  - A frame message has header `{"type": "frame", "w": int, "h": int, "seq": int, "t": float}` and a JPEG payload.

- [ ] **Step 1:** Run `uv init --package --name rig --python 3.12`, or hand-write `pyproject.toml` with the package `rig` at the repo root. Then `uv add fastapi "uvicorn[standard]" pillow` and `uv add --dev pytest httpx`. Add `.gitignore` with `.env`, `.venv/`, `__pycache__/`, `logs/`.
- [ ] **Step 2: Write failing tests** in `tests/test_frames.py`:
  - `test_roundtrip_socketpair`: with `a, b = socket.socketpair()`, send `encode({"type":"frame","w":2,"h":1,"seq":7,"t":1.5}, b"\xff\xd8xyz")` on `a`. `read_message(b)` returns that exact header dict and payload.
  - `test_roundtrip_async`: feed `encode(...)` bytes into an `asyncio.StreamReader` and call `feed_eof()`. `read_message_async` returns the same pair.
  - `test_eof_mid_message_raises`: feed only the first 10 bytes of an encoded message, then EOF. Both readers raise `ConnectionError`.
  - `test_oversize_header_rejected`: feed `PREFIX.pack(MAX_HEADER + 1, 0)`. This raises `ConnectionError` without trying to read 64 KB.
- [ ] **Step 3:** Run `uv run pytest tests/test_frames.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 4:** Implement `rig/frames.py` using stdlib only (`json`, `struct`, `socket`, `asyncio`). The mining reference is `/Users/desmondzee/Software/mining/isaac/sim/protocol.py`, minus numpy and PIL.
- [ ] **Step 5:** Run `uv run pytest tests/test_frames.py -v`. Expected: 4 passed.
- [ ] **Step 6:** Commit `"Add uv project and frame wire format"`.

---

### Task 2: Web server and viewer page

**Files:**
- Create: `rig/web/__init__.py`, `rig/web/server.py`, `rig/web/static/index.html`, `rig/web/static/viewer.js`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `rig.frames.read_message_async`
- Produces, in `rig/web/server.py`:
  - `class FrameHub`:
    - `publish(jpeg: bytes, header: dict) -> None`
    - `latest: tuple[int, bytes, dict] | None`, holding `(seq, jpeg, header)`
    - `async wait_newer(seq: int) -> tuple[int, bytes, dict]`: returns as soon as the hub's seq is greater than `seq`. Built on an `asyncio.Condition`. Keeps no per-client queue.
    - Counters: `frames: int`, `last_frame_at: float | None` (monotonic), `sim_connected: bool`, `fps: float`. `fps` is an exponential moving average or a rolling 2 s window.
  - `create_app(sim_host: str = "127.0.0.1", sim_port: int = 9000) -> FastAPI`:
    - The lifespan starts `asyncio.start_server` for ingest. `sim_port=0` picks an ephemeral port, and the bound port goes in `app.state.sim_port`. The hub lives in `app.state.hub`.
    - A newer ingest connection closes the previous writer.
  - Module-level `app = create_app(os.environ.get("SIM_HOST","127.0.0.1"), int(os.environ.get("SIM_PORT","9000")))` for uvicorn.
  - Routes:
    - `GET /` serves `static/index.html`. `GET /static/*` is mounted.
    - `GET /api/health` returns `{"sim_connected": bool, "frames": int, "fps": float, "frame_age_s": float | None, "w": int | None, "h": int | None}`.
    - `GET /api/frame.jpg` returns the latest JPEG as `image/jpeg`, or 503 with `{"detail":"no frames yet"}`.
    - `WS /ws/video` loops `wait_newer(last_seq)` and sends each frame as one binary message of raw JPEG bytes.

- [ ] **Step 1: Write failing tests** in `tests/test_server.py`. Use `fastapi.testclient.TestClient(create_app(sim_port=0))` as a context manager so the lifespan runs. A helper `push(app, n, start_seq=0)` opens a plain `socket.create_connection(("127.0.0.1", app.state.sim_port))` and sends `n` encoded frames whose payloads are `b"\xff\xd8" + bytes([seq])`. Poll `/api/health` until `frames` reaches the target, with a 2 s timeout.
  - `test_health_before_frames`: health is `{"sim_connected": False, "frames": 0, "frame_age_s": None, ...}`, and `/api/frame.jpg` is 503.
  - `test_frames_reach_health_and_jpeg`: after `push(3)`, health shows `sim_connected` True and `frames == 3`. `/api/frame.jpg` returns 200, `image/jpeg`, and body `b"\xff\xd8\x02"`.
  - `test_ws_receives_jpeg`: open `/ws/video`, `push(1)`, and `receive_bytes()` starts with `b"\xff\xd8"`.
  - `test_slow_client_gets_latest_only`: open the WS, push 50 frames without reading, wait for health `frames == 50`, then read one message. It is the seq-49 payload, or a later read reaches seq 49 within 3 messages. The hub has no per-client buffer.
  - `test_new_sim_replaces_old`: connect socket A and push 1, then connect socket B and push 1. A gets EOF (`recv` returns `b""` within 2 s), and health still shows `sim_connected` True.
  - `test_index_serves_viewer`: `GET /` is 200 and the HTML contains `viewer.js`.
  - `test_viewer_uses_page_scheme`: the text of `rig/web/static/viewer.js` contains `location.protocol === "https:"` and `"wss:"`.
- [ ] **Step 2:** Run `uv run pytest tests/test_server.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3:** Implement `server.py` to the interfaces above. An ingest handler that hits `ConnectionError` sets `sim_connected=False`, but only if it is still the current connection.
- [ ] **Step 4:** Write `index.html` and `viewer.js`:
  - A dark full-width page with a `<canvas>` scaled to fit with aspect ratio preserved.
  - A status line showing `connecting…`, then `waiting for frames…`, then `live`.
  - An fps readout computed in the browser over a 1 s window.
  - A frame-age readout taken from `/api/health`, polled every 2 s.
  - `viewer.js` builds the WS URL as `(location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/video"` and sets `binaryType = "arraybuffer"`.
  - It decodes with `createImageBitmap(new Blob([data], {type: "image/jpeg"}))`. If a frame arrives while one is still decoding, it drops the older one.
  - It reconnects on close with backoff from 0.5 s up to 5 s.
- [ ] **Step 5:** Run `uv run pytest -v`. Expected: all pass.
- [ ] **Step 6:** Commit `"Add web relay server and viewer page"`.

---

### Task 3: Fakesim, web image and compose (web, fakesim, tunnel)

**Files:**
- Create: `rig/fakesim.py`, `docker/web.Dockerfile`, `docker-compose.yml`, `.env.example`

**Interfaces:**
- Consumes: `rig.frames.encode`
- Produces:
  - `python -m rig.fakesim --host 127.0.0.1 --port 9000 --fps 15 --size 640x360`. It loops forever and reconnects every 1 s if the connection fails.
  - Each frame is a Pillow image with a gradient background, a square rotating once every 4 s, and the text `FAKESIM seq=N`. It's JPEG quality 80, with header fields as in Task 1.

- [ ] **Step 1:** Implement `rig/fakesim.py`.
- [ ] **Step 2:** Run a local end-to-end check with no Docker:
  - Start `uv run uvicorn rig.web.server:app --host 127.0.0.1 --port 8080` and `uv run python -m rig.fakesim` in the background.
  - Then run `curl -s localhost:8080/api/health`. Expected: `frames` rising and `fps` around 15.
  - Run `curl -s -o "$SCRATCH/fake.jpg" localhost:8080/api/frame.jpg`, then look at the image with Read. Expected: a rotating square with the text visible.
  - Open `http://localhost:8080` in the browser (chrome-devtools MCP is fine) and confirm the canvas is animating and the status reads `live`.
  - Stop both processes.
- [ ] **Step 3:** Write `docker/web.Dockerfile`:
  - FROM `python:3.12-slim`, then COPY `uv` from `ghcr.io/astral-sh/uv:<pinned tag that exists>`.
  - `WORKDIR /app`, copy `pyproject.toml` and `uv.lock`, then `uv sync --frozen --no-dev --no-install-project`. After that copy `rig/` and run `uv sync --frozen --no-dev`.
  - CMD `["/app/.venv/bin/uvicorn", "rig.web.server:app", "--host", "127.0.0.1", "--port", "8080"]`.
- [ ] **Step 4:** Write `docker-compose.yml`:
  - `name: rig`.
  - `web`: builds `docker/web.Dockerfile` with context `.`, image `rig-web`, `network_mode: host`, env `SIM_PORT: "9000"`, `restart: unless-stopped`.
  - `fakesim`: `profiles: ["fake"]`, same image, `command: ["/app/.venv/bin/python", "-m", "rig.fakesim"]`, host network.
  - `tunnel`: `profiles: ["public"]`, image `cloudflare/cloudflared:2026.9.3`, `command: tunnel --no-autoupdate run`, `environment: TUNNEL_TOKEN: ${TUNNEL_TOKEN:?set TUNNEL_TOKEN in .env}`, host network, `restart: unless-stopped`.
  - The `sim` service comes in Task 5.
  - Also write `.env.example`, containing only `TUNNEL_TOKEN=`.
- [ ] **Step 5:** Run `docker compose config -q` locally (with no `.env` and no profile). Expected: exit 0. Then `docker compose --profile public config -q`. Expected: it fails with "set TUNNEL_TOKEN in .env".
- [ ] **Step 6:** Commit `"Add fakesim, web image and compose"`.

---

### Task 4: Sim-side frame sender

**Files:**
- Create: `rig/sim/__init__.py`, `rig/sim/sender.py`
- Test: `tests/test_sender.py`

**Interfaces:**
- Consumes: `rig.frames.encode`
- Produces, in `rig/sim/sender.py`, `class FrameSender(host: str, port: int, retry_s: float = 1.0)`:
  - `start() -> FrameSender`: starts a daemon thread.
  - `send(header: dict, payload: bytes) -> None`: never blocks and never raises. It replaces any frame still waiting to go out (a single slot with a lock and an Event), and every replaced frame counts toward `dropped`.
  - `stop() -> None`
  - Attributes: `connected: threading.Event`, `sent: int`, `dropped: int`.
  - The thread connects with `socket.create_connection(timeout=2)`, sets `TCP_NODELAY`, and sends `rig.frames.encode(header, payload)` with `sendall`. If an `OSError` occurs, it clears `connected`, closes the socket, sleeps `retry_s` and reconnects.

- [ ] **Step 1: Write failing tests**, each with a 3 s timeout. Each test runs a tiny listening socket on an ephemeral port and reads with `rig.frames.read_message`.
  - `test_delivers_frames`: after `start()`, `send` 3 frames spaced 50 ms apart. The server reads at least one frame and the last one read has `seq == 2`. `connected` is set.
  - `test_send_without_server_does_not_block`: point the sender at a closed port with `retry_s=0.05`. 100 calls to `send` take under 50 ms in total. `connected` isn't set and `dropped >= 99`.
  - `test_reconnects_after_server_restart`: deliver one frame, close the server and its connection, then reopen a listener on the same port. Keep calling `send` every 20 ms. The new listener receives a frame within 2 s.
- [ ] **Step 2:** Run `uv run pytest tests/test_sender.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3:** Implement `sender.py`, stdlib only.
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: all pass.
- [ ] **Step 5:** Commit `"Add sim frame sender"`.

---

### Task 5: Isaac test scene, sim loop and sim container

**Files:**
- Create: `rig/sim/scene.py`, `rig/sim/main.py`, `docker/sim.Dockerfile`
- Modify: `docker-compose.yml`, to add `sim`

**Interfaces:**
- Consumes: `rig.sim.sender.FrameSender`, `rig.frames`
- Produces:
  - `rig/sim/scene.py`:
    - `CAM_W, CAM_H = 640, 360`
    - `CAMERA_PATH = "/World/Camera"`
    - `@dataclass Scene: spinner_rotate: UsdGeom.XformOp`
    - `build_scene(stage: Usd.Stage) -> Scene`
    - `spin(scene: Scene, t: float) -> None` sets the spinner's Z rotation to `(45.0 * t) % 360` degrees.
  - `rig/sim/main.py` takes `--sim 127.0.0.1:9000`, `--fps 15`, `--renderer RealTimePathTracing`, `--quality 80`, `--duration 0` (0 means forever) and `--snapshot PATH`. `--snapshot` saves the first non-black frame as a PNG; it's off by default.

- [ ] **Step 1:** Implement `build_scene` with pure `pxr` APIs. Mining's `/Users/desmondzee/Software/mining/isaac/sim/scene.py` lines 53–75 and 135–160 are the reference for the camera, render product and annotator calls that worked on watson.
  - Z-up stage at 1 m per unit, with an `/World` Xform.
  - Ground: a `UsdGeom.Cube` scaled to 20×20×0.1 at z=-0.05, grey `displayColor`.
  - Lights: a `UsdLux.DomeLight` at intensity 500 and a `UsdLux.DistantLight` at intensity 1600, tilted about 45°.
  - Shapes, each with its own `displayColor`:
    - a red 1 m cube at (0,0,0.5), which is the spinner (one rotateZ op)
    - a blue sphere of radius 0.5 at (2,1,0.5)
    - a green cylinder at (-2,1,0.75)
    - a yellow cone at (1,-2,0.5)
  - Camera: `UsdGeom.Camera` at `CAMERA_PATH`, eye (6,-6,4) looking at (0,0,0.5) with Z up. Use a transform op from `Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()`. Set horizontal FOV to about 60° through focal length and aperture as in mining's `pinhole()`, with clipping 0.1–1000.
- [ ] **Step 2:** Implement `main.py`:
  - Start with `SimulationApp({"headless": True, "renderer": args.renderer, "width": CAM_W, "height": CAM_H})` as the first thing, before any `omni`/`pxr` import.
  - Build the scene on `omni.usd.get_context().get_stage()`.
  - Set up rendering with `rp = rep.create.render_product(CAMERA_PATH, (CAM_W, CAM_H))`, then `ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach(rp)`. Turn off motion blur, DOF and lens flares through `carb.settings`, as in mining's `config.json` `rtx` block. Play the timeline, then call `app.update()` 30 times.
  - Loop at `--fps`:
    - `spin(scene, t)`
    - `rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)`
    - `data = np.asarray(ann.get_data())`. Skip the frame if `data.ndim != 3` or it's empty.
    - JPEG-encode `data[..., :3]` with PIL at `--quality`.
    - `sender.send({"type":"frame","w":CAM_W,"h":CAM_H,"seq":seq,"t":t}, jpeg)`
  - Every 5 s, print `fps=… sent=… dropped=… connected=… mean=…`, where `mean` is the pixel mean of the last frame, so you can tell when it goes from black to rendered.
  - On exit, stop the sender and call `os._exit(0)` (`app.close()` can hang, as mining found).
- [ ] **Step 3:** Write `docker/sim.Dockerfile`:
  - `FROM nvcr.io/nvidia/isaac-sim:6.2.0`
  - ENV `ACCEPT_EULA=Y PRIVACY_CONSENT=Y NVIDIA_DRIVER_CAPABILITIES=all PYTHONUNBUFFERED=1 PYTHONPATH=/rig`
  - `WORKDIR /rig`
  - `ENTRYPOINT ["/isaac-sim/python.sh"]` and `CMD ["/rig/rig/sim/main.py"]`
- [ ] **Step 4:** Add the `sim` service to `docker-compose.yml`:
  - Build `docker/sim.Dockerfile`, image `rig-sim:6.2.0`, `devices: [nvidia.com/gpu=all]`, `network_mode: host`, `restart: unless-stopped`, `depends_on: [web]`.
  - Volumes: `./rig:/rig/rig:ro` and `./logs:/logs`, plus named volumes:
    - `rig-isaac-cache-main:/isaac-sim/.cache`
    - `rig-isaac-cache-compute:/isaac-sim/.nv/ComputeCache`
    - `rig-isaac-cache-kit:/isaac-sim/kit/cache`
    - `rig-isaac-ovlogs:/isaac-sim/.nvidia-omniverse/logs`
    - `rig-isaac-config:/isaac-sim/.nvidia-omniverse/config`
    - `rig-isaac-data:/isaac-sim/.local/share/ov/data`
  - Each named volume is declared with an explicit `name:`.
- [ ] **Step 5:** Run `python3 -m py_compile rig/sim/scene.py rig/sim/main.py` and `docker compose config -q`. Expected: both exit 0. Real verification happens in Task 7.
- [ ] **Step 6:** Commit `"Add Isaac test scene, sim loop and sim container"`.

---

### Task 6: Remote helper script and README

**Files:**
- Create: `scripts/remote.sh`, `README.md`

**Interfaces:**
- Produces `scripts/remote.sh <cmd>`. It runs on watson from the repo root, with `set -euo pipefail` and `cd "$(dirname "$0")/.."`.
  - `check` prints one `PASS`/`WARN`/`FAIL` line per item and never changes anything:
    - driver version from `nvidia-smi` (PASS if 580.x)
    - `docker info` works without sudo
    - `nvidia-ctk cdi list` includes `nvidia.com/gpu=all`
    - image `nvcr.io/nvidia/isaac-sim:6.2.0` present (WARN if missing: "first up will pull ~27 GB")
    - ports 8080 and 9000 either free or held by `rig` (`ss -ltnp`)
    - `.env` has a non-empty `TUNNEL_TOKEN` (WARN if not: "tunnel will not start")
    - exit 1 if any FAIL
  - `up [fake]`:
    - `mkdir -p logs`
    - run `docker compose up -d --build web`
    - then start `sim`, or `fakesim` if `fake` is given (in that case stop `sim` first, since the newest connection wins)
    - add the `--profile public` tunnel if `.env` has a token
    - print the next-step hint (`status`, and the URL)
  - `down`: `docker compose --profile fake --profile public down`. It keeps the volumes and never uses `-v`.
  - `logs [svc]`: `docker compose logs -f --tail 100 ${svc:-sim}`.
  - `status`: `docker compose ps`, `curl -s localhost:8080/api/health`, and the last `fps=` line from the sim logs.
  - `frame [path]`: saves `curl -s localhost:8080/api/frame.jpg` to `${path:-logs/frame.jpg}`.

- [ ] **Step 1:** Write `scripts/remote.sh` (chmod +x) and run `bash -n scripts/remote.sh`. Expected: exit 0. If shellcheck is installed, `shellcheck scripts/remote.sh` should show no errors.
- [ ] **Step 2:** Write `README.md` covering:
  - what this is (one paragraph)
  - architecture (the three services, the ports)
  - the local dev loop (`uv run pytest`, and the uvicorn plus fakesim commands from Task 3)
  - first run on watson: clone path, `scripts/remote.sh check`, then `up fake` to test the page, then `up`, plus the shader warm-up warning ("black frames for several minutes on first start")
  - viewing without the tunnel: `ssh -N -L 8080:127.0.0.1:8080 watson`, and never `brev port-forward`
  - Cloudflare setup: a new tunnel, public hostname `rig.zeecloud.org` → `http://localhost:8080`, an Access email policy, and the token in `.env`
  - troubleshooting: rows from mining's `isaac/CHECKLIST.md` that apply (CDI, black frames, the RaytracedLighting fallback via `command:` override, low fps)
- [ ] **Step 3:** Commit `"Add remote helper script and README"`, then push to `origin main`.

---

### Task 7: Deploy and verify on watson

No new files. Everything runs over `ssh watson`, inside `~/Documents/spark-bef241b1-submission-repo` only. Fixes found here go back through the Mac: edit, test, commit, push, then `git pull` on watson.

- [ ] **Step 1:** `ssh watson 'cd ~/Documents && git clone https://github.com/NeoChenn/spark-bef241b1-submission-repo.git'`, or `git pull` if it already exists. Then run `scripts/remote.sh check`. Expected: no FAIL. A WARN for the tunnel token is OK.
- [ ] **Step 2:** `scripts/remote.sh up fake`, wait 10 s, then `scripts/remote.sh status`. Expected: `sim_connected: true` and fps around 15. Run `ssh -N -L 8080:127.0.0.1:8080 watson` on the Mac, open `http://localhost:8080`, and confirm the fakesim animation shows.
- [ ] **Step 3:** `scripts/remote.sh up` (the real sim). Follow `logs sim` until the `fps=` lines show `connected=True` and `mean` above about 20. Allow up to 10 minutes on first start for shader compile. If it's still black after 10 minutes, switch to the RaytracedLighting fallback and say so.
- [ ] **Step 4:** Run `scripts/remote.sh frame logs/a.jpg`, wait 1 s, then `scripts/remote.sh frame logs/b.jpg`. Copy both to the scratchpad with `ssh watson 'cat …/logs/a.jpg' > …` and look at them. Expected:
  - the ground, the four coloured shapes and the red cube are visible
  - the image isn't mirrored or black
  - the cube's angle differs between a and b
- [ ] **Step 5:** Ask the user for the tunnel token to be placed in `.env` on watson. The user does the dashboard steps from the README, and the token is never echoed into chat or logs. Then run `scripts/remote.sh up` again and check that `docker compose ps` shows `tunnel` running. Open `https://rig.zeecloud.org`. Expected: the Access login appears, and after login the page shows `live`, the cube spinning and fps updating. This checks Review Focus #1.
- [ ] **Step 6:** Check that watson is clean. `ssh watson 'cd ~/Documents && git -C spark-bef241b1-submission-repo status --short'` should show only `logs/` (gitignored) and `.env`. `docker volume ls` should show only `rig-*` volumes. Report fps and the first-start warm-up time to the user.
