# Offshore rig sim: infrastructure milestone

## Intent

The long-term goal is a full offshore oil rig rendered in Isaac Sim, with
quadruped robots running policies inside it. It mirrors the earlier `mining`
project (`/Users/desmondzee/Software/mining`), which ran autonomous vehicles in
Isaac Sim on a GB10.

This milestone only proves the plumbing. Isaac Sim runs in Docker on the
GB10 host `watson`, a fixed camera's view streams to a web page, and that page
is reachable at `https://rig.zeecloud.org` through a Cloudflare tunnel. If you
can open the URL on a laptop and watch a live, moving image, the milestone is
done.

## Goals

- A simple test world in Isaac Sim: ground plane, light, a few primitive
  shapes, and one shape that keeps spinning so a frozen stream is obvious.
- One fixed camera looking at the shapes. Its frames stream live to a browser.
- Everything on watson runs in Docker, started with one `docker compose up`.
- The page is served publicly at `rig.zeecloud.org` through a new named
  Cloudflare tunnel, behind a Cloudflare Access email policy.
- The page shows the image and a live fps counter.

## Non-goals (for this milestone)

- Quadrupeds, robot models, policies, and Isaac Lab.
- The oil rig itself or any real assets.
- Controls, multiple cameras, depth or lidar, and H.264/NVENC encoding.
- A separate "brain" service like mining had. The web server only relays frames.

## Anti-goals

- Touching anything on watson outside the cloned repo directory. That means no
  edits to system config, no `sudo` fixes and no `brev port-forward`. If a
  host-level fix turns out to be needed (driver, NVIDIA container toolkit,
  CDI spec), stop and ask the user. Docker images and named volumes created by
  this project's compose project are allowed.
- Reusing mining's tunnel token. If two machines connect with one token, they
  split traffic between them.
- Committing secrets. `TUNNEL_TOKEN` lives only in a gitignored `.env`.

## Constraints

- **Host:** `watson`, a Brev GB10 (DGX Spark), aarch64, hostname
  `promaxgb10-64f1`, driver 580.178.04. Reach it with `ssh watson`. The plan
  was jarvis, but the user isn't in jarvis's docker group, so it moved to
  watson. Checked 2026-11-07:
  - Docker works for the user without sudo.
  - CDI is generated (`nvidia.com/gpu=all`).
  - No containers are running and nothing is using the GPU.
  - Ports 8080 and 9000 are free.
  - `nvcr.io/nvidia/isaac-sim:6.2.0` is already pulled.
  - The 3 TB disk has room.
- **Neighbours on watson:** `~/Documents/mining` and other projects live there.
  Clone this repo to `~/Documents/spark-bef241b1-submission-repo` and leave
  everything else alone, including mining's images, volumes and compose
  project. Use a distinct compose project name, for example `rig`.
- **Repo:** public, and we have write access. watson needs no git credentials,
  and the Isaac image needs no NGC login.
- **Who deploys:** Claude builds, pushes, clones or pulls on watson, runs the
  stack and verifies it. The user handles the Cloudflare dashboard and the token.
- **Workflow:** edit in this repo on the Mac, push to
  `github.com/NeoChenn/spark-bef241b1-submission-repo`, then clone or `git pull`
  on watson. Work on watson happens only inside that clone.
- **Isaac Sim:** `nvcr.io/nvidia/isaac-sim:6.2.0`. It's multi-arch and is the
  same image mining ran on its GB10. Run it headless as a standalone script
  through `/isaac-sim/python.sh`. Isaac's WebRTC livestream doesn't work on
  aarch64, so frames must be pulled from a render product and sent out by our
  own code.
- **GPU in Docker:** use CDI (`devices: [nvidia.com/gpu=all]`) as mining did.
  The check script falls back to reporting what's missing.
- **Commits:** no co-author or "Generated with" lines. No narrative code
  comments.
- **Tooling:** Python uses `uv` and Node uses `pnpm`, though Node probably isn't
  needed.

## Decided shape

Three compose services on watson, all on host networking like mining:

1. **`sim`**
   - FROM `nvcr.io/nvidia/isaac-sim:6.2.0`, with `ACCEPT_EULA=Y`,
     `PRIVACY_CONSENT=Y` and `NVIDIA_DRIVER_CAPABILITIES=all`.
   - Builds the test scene in code and attaches a replicator RGB annotator to
     the camera's render product at roughly 640x360.
   - Encodes each frame as JPEG and sends it to `web`.
   - Named volumes cache shaders and kit data so only the first start is slow.
     Mining's list: `/isaac-sim/.cache`, `.nv/ComputeCache`, `kit/cache`,
     `.nvidia-omniverse/{logs,config}`, `.local/share/ov/data`.
2. **`web`**
   - A small Python server, FastAPI or similar, run with uv, on
     `127.0.0.1:8080`.
   - Receives frames from `sim`, keeps only the latest one, and pushes JPEGs to
     browsers over a WebSocket.
   - Serves one static page that shows the image and fps.
   - Plain HTML/JS, no framework.
3. **`tunnel`**
   - `cloudflare/cloudflared` (mining pinned `2026.9.3`), running
     `tunnel --no-autoupdate run` with `TUNNEL_TOKEN` from `.env`.
   - Ships with a `.env.example` that holds only the variable name.
   - Sits behind compose profile `public`, so `sim` and `web` can run before a
     token exists without cloudflared crash-looping. `scripts/remote.sh up`
     turns the profile on when `.env` has a token. (Claude's call.)

**Dashboard work (the user does this):** create a new tunnel, add a public
hostname `rig.zeecloud.org` → `http://localhost:8080`, attach an Access email
policy, then put the token in the clone's `.env` on watson.

A helper script, `scripts/remote.sh` or similar, with these commands:

- `check`: driver version (mining needed the 580 branch), whether docker sees
  the GPU via CDI, and whether `.env` has a token. It only reports and never
  fixes.
- `up`, `down`, `logs` and `status`.

The README covers first-run steps. The image is already on watson, but the
first start compiles shaders for several minutes and shows black frames until
then, so warn about that up front.

## Left to the builder

- Exact transport between `sim` and `web`. A length-prefixed TCP socket on
  `127.0.0.1:9000` like mining is fine, and so is a WebSocket client in the sim.
  Prefer whatever needs no extra packages inside the Isaac image.
- JPEG quality, target fps (around 15 is fine) and frame size.
- File layout, names and the clone path on watson. Suggested: `sim/`, `web/`,
  `docker-compose.yml` and `scripts/`.
- A `fakesim` that sends synthetic frames to test `web` and the tunnel without
  Isaac is welcome but optional.
- Copy from mining freely (`isaac/docker/`, `isaac/sim/main.py`,
  `brain/brain/protocol.py`, `web/video.js`, `isaac/CHECKLIST.md`), but strip it
  down to this milestone.

## Done when

- `scripts/remote.sh up` on watson brings all three services up.
- `https://rig.zeecloud.org` asks for Access login, then shows the camera view
  with a shape visibly spinning and fps updating.
- Nothing on watson changed outside the clone, apart from Docker images and
  volumes.
