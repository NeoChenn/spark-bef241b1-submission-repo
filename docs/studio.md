# Studio feed: our gas-leak robot on the team's web page

`flange_robot/studio.py` draws a live 2D dashboard of what our module perceives and decides, and sends it
to the studio's web relay with the studio's own frame protocol (`rig/frames.py`, `rig/sim/sender.py`).
Anyone who opens the studio page sees it.

**What it is not:** it is a 2D schematic. Our module is not inside the Isaac Sim 3D rig, and the studio's
Isaac stream is untouched. Putting our scenario in the 3D world would need a bridge running inside Isaac
Sim (see `integration/INTERFACE.md` on the `rust-detection` branch). That is not built.

## What the page shows

- **Approach strip:** the robot walking to the flange, with the gas plume growing around it.
- **Flange close-up:** the eight bolts as the robot *believes* them: green ok, red loose, grey unsure or
  unseen (anything under its 0.7 confidence gate), yellow re-reading after a tightening. A red crack line
  appears once it has diagnosed a crack.
- **Right panel:** phase, diagnosis, gas gauge with the "clear" and "STOP" marks and a trend line,
  perception numbers, the current action with the safety verdict and reason, and the **supervisor alert**.
- **Red banner:** appears for a few seconds when a live change is injected.

This puts perception next to actions on screen, which the video requirement asks for.

## Run it

**Use its own relay and its own ports. Do not point it at the Isaac relay (port 9000).** The relay accepts
one source at a time and a new connection kicks the old one off. Both senders retry every second, so two
sources on one relay would fight and the page would flicker.

```bash
pip install mujoco numpy pillow fastapi "uvicorn[standard]"

# terminal 1: a second copy of the studio's own web server (code unchanged), on its own ports
SIM_PORT=9001 uvicorn rig.web.server:app --host 127.0.0.1 --port 8081

# terminal 2: our feed
python -m flange_robot.studio --sim 127.0.0.1:9001

# view it: http://localhost:8081  (on watson, from your laptop: ssh -N -L 8081:127.0.0.1:8081 watson)
```

The Docker `web` service uses ports 8080 and 9000, so these do not clash. To put it on the public site, the
Cloudflare tunnel needs a second public hostname pointing at `http://localhost:8081` (a dashboard change by
whoever owns the tunnel; we cannot do that from the repo).

`mujoco` on watson's ARM chip: wheels probably exist, but this is **unverified**.

## Live scene changes for the unscripted change in the video

Type in the terminal while it runs:

| Command | Effect |
|---|---|
| `leak [bolt] [radians]` | loosen a bolt and start a leak, e.g. `leak 2 0.15` |
| `crack [severity]` | a crack at the flange, e.g. `crack 0.3` |
| `dust [level]` | degrade the camera, 0 to 1 |
| `clear` | remove the dust |
| `next` | skip to the next scenario |

The five built-in scenarios cycle forever: loose bolts, crack, dusty camera, big leak, and a leak that starts
mid-run. Options: `--fps`, `--once`, `--show-truth` (debug: rings the bolts that are truly loose),
`--no-stdin`.

## What is tested

12 tests, including one that sends our frames through the studio's real `FrameSender` into its real FastAPI
app and reads them back from `/api/frame.jpg`. A manual run against a second relay showed `no_sim` change to
`live` at 10 fps, 1280x720, with a frame age under 0.1 s. A regression test covers a bug found by looking at
the output: the screen once said "no crack seen" while the diagnosis said crack.

**Not tested:** anything on watson itself, other browsers, a Cloudflare tunnel, or running next to the Isaac
stream.
