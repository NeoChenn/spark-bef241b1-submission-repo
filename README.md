# Offshore rig sim

Isaac Sim running headless in Docker on a GB10 (watson), streaming a camera view to a web page at
`https://rig.zeecloud.org`. This first milestone is a test world (ground, lights, a few shapes, one spinning red cube,
a fixed camera) to prove the plumbing. The rig itself, quadrupeds and policies come later.

## Architecture

```
sim (Isaac Sim 6.2.0, GPU)  --JPEG over TCP 127.0.0.1:9000-->  web (FastAPI, 127.0.0.1:8080)  <--  tunnel (cloudflared)  <--  browser
```

- `sim`: `rig/sim/main.py` builds the scene (`rig/sim/scene.py`), reads the camera's RGB annotator, JPEG-encodes and
  sends frames with `rig/sim/sender.py`. Isaac's WebRTC livestream does not work on aarch64, hence our own path.
- `web`: `rig/web/server.py` keeps only the latest frame and pushes it to browsers over `/ws/video`.
  `/api/health` and `/api/frame.jpg` are for checks.
- `tunnel`: compose profile `public`, started only when `.env` has `TUNNEL_TOKEN`.

All three use host networking and bind to loopback only.

## Local dev (Mac)

```bash
uv run pytest
uv run uvicorn rig.web.server:app --port 8080   # page at http://localhost:8080, waits for a sim
```

## Run on watson

Everything happens inside `~/Documents/spark-bef241b1-submission-repo`. Edit on the Mac, push, `git pull` on watson.

```bash
ssh watson
cd ~/Documents && git clone https://github.com/NeoChenn/spark-bef241b1-submission-repo.git   # first time
cd spark-bef241b1-submission-repo && git pull
scripts/remote.sh check    # driver, docker, GPU (CDI), image, ports, tunnel token. Reports only.
scripts/remote.sh up       # builds and starts web + sim (+ tunnel if .env has a token)
scripts/remote.sh status   # containers, /api/health, last sim fps line
scripts/remote.sh logs sim
scripts/remote.sh frame logs/frame.jpg
scripts/remote.sh down     # keeps the shader cache volumes
```

**First start:** Isaac compiles shaders for several minutes. Frames are black until then (`mean=` near 0 in the sim's
`fps=` log line). The `rig-isaac-*` volumes keep the cache, so later starts are quick.

**Without the tunnel:** `ssh -N -L 8080:127.0.0.1:8080 watson`, then open http://localhost:8080. Do not use
`brev port-forward` (it leaves a persistent binding on the node).

## Cloudflare setup (one time)

1. Zero Trust → Networks → Tunnels → create a tunnel (cloudflared), e.g. `rig-watson`. Copy its token.
2. Public hostname: `rig.zeecloud.org` → service `http://localhost:8080`.
3. Access → Applications → self-hosted app for `rig.zeecloud.org` with an email allow policy.
4. On watson: `cp .env.example .env` and set `TUNNEL_TOKEN=...`. Never commit `.env`.
5. `scripts/remote.sh up`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unresolvable CDI devices nvidia.com/gpu=all` | Host needs `nvidia-ctk cdi generate` (sudo). Ask before changing the host. |
| Black frames after 10+ minutes | Try the faster renderer: set the sim `command:` to add `--renderer RaytracedLighting`, then `up`. |
| Page says "waiting for sim" | `scripts/remote.sh logs sim`: look for a Traceback, or `connected=False` (web not up). |
| Low fps | Lower `--fps`, or use `RaytracedLighting`. |
| `Permission denied` under `/logs` | `chmod 777 logs` (Isaac runs as uid 1234). `remote.sh up` does this. |
