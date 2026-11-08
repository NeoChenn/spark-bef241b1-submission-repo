# Fleet scene and replay

Renders a fleet timeline (`runs/fleet/timeline.jsonl`) on the full offshore rig as one MP4 with a HUD.

```
C:\isaacsim\python.bat sim\fleet_replay.py                      # full replay -> sim/output/fleet/fleet_replay.mp4
C:\isaacsim\python.bat sim\fleet_replay.py --max-frames 48      # quick test
C:\isaacsim\python.bat sim\fleet_replay.py --at 2,19,44 --width 960 --height 540   # stills only
C:\isaacsim\python.bat sim\fleet_scene.py                       # placement previews (fixed cameras)
C:\isaacsim\python.bat -m pytest tests\test_fleet_replay.py     # CPU tests (pytest on PYTHONPATH)
```

Options: `--timeline`, `--fake` (overwrite the timeline with the built-in fake one), `--width/--height`
(default 1280x720), `--fps` (24), `--subframes` (RTX subframes per frame, default 4), `--max-frames`,
`--start`, `--preset golden|night|storm|none` (`none` loads `full_rig.usda` without the sky/ocean layer),
`--no-ocean`, `--bitrate` (Mbit/s, default 4: a 60 s 720p replay is about 30 MB; the encoder default of
16 Mbit/s gave 97 MB).

Render cost on the RTX PRO 500 laptop (6 GB): 0.32 s per 1280x720 frame at 4 subframes, so a 60 s replay
at 24 fps (1440 frames) takes about 8 minutes, plus about 40 s to load the stage.

## Files

| file | what |
|---|---|
| `sim/fleet_replay.py` | timeline parsing, per-frame interpolation, shot planning, HUD, camera projection (pure numpy/PIL, unit-tested), plus the Isaac render loop in `main()` |
| `sim/fleet_models.py` | USD robots: Q (Go2 + lift + UR3 via `RobotPoser`), A (quadrotor), C (magnetic crawler), R (ROV block) |
| `sim/fleet_scene.py` | opens `cinematic.usda` (which sublayers `full_rig.usda`), adds `/World/Fleet` on the session layer, job markers, gas plumes, camera, lamp-bulb hiding; preview script |
| `tests/test_fleet_replay.py` | interpolation, yaw wrap, deck snapping, plume fade, shot plan, projection, HUD |

## Timeline format

One line per robot per second, `{"t","robot","kind","position":[x,y,z],"yaw","state","job"}`, plus event lines
`{"t","event","job","robot","detail"}`. `detail.position` (if present) is used for the gas plume and the
event camera. If the file is missing, `fleet_replay.py` writes a fake 60 s shift in exactly this format:
A1 flies a gas survey from the helideck and reports a leak on the blue process line at t = 18 s, Q2 is
dispatched to isolate valve_03 and closes it at t = 43 s, Q1 reads gauge_02, C1 runs a UT thickness scan
along the orange line and reports wall loss at t = 42 s.

Conventions assumed: yaw in radians (a file whose yaws exceed 2*pi is read as degrees); positions in rig
metres, Z up. Q and C are pinned to the deck below their logged z (decks at 20, 31.25, 42.5 m); a flying A is
kept at least 4 m above the top deck, a landed one (state IDLE/DOCKED/CHARGING/LANDED) may sit on the
helideck (pad surface z = 51.27 m). Job markers go at the position of the job's latest event that has one,
else at a rig asset id that appears in the job string (e.g. `gauge_02`, from `full_rig_meta.json`).

## Camera language

0-6 s: establishing orbit of the whole rig. Then one cut per event or job change; cuts are at least 4 s
apart, safety events (gas, leak, alert, finding) may cut after 3 s, and cues pushed back more than 6 s are
dropped. Shots longer than 12 s are split by cutting to another busy robot. Ground robots are filmed from
inboard looking out (sea behind them, no deck railings in front of the lens), drones from outboard looking
in. Event shots frame the event position with the reporting robot. Deck lamp bulbs within 9 m of the camera
are hidden (their light stays). The HUD lists every robot with its state colour and job, the last events
(8 s), and an ID tag projected over each robot.

## What is simulated and what is not

Nothing in this scene is physically simulated. It is a kinematic replay of whatever the timeline says.

- Robot poses come straight from the timeline, linearly interpolated between the 1 Hz samples. There is no
  PhysX, collision or contact. A robot logged inside a wall would be drawn inside it.
- Q legs: a scripted trot (diagonal thigh/calf swing phased on distance travelled, as in
  `integration/isaac_bridge.py` `_walk`), not a locomotion policy. Feet can slide. The arm unfolds to the
  repair start pose while INSPECTING/WORKING, and WORKING adds a wrist roll ("turning a valve"). That is
  animation, not IK on a real handwheel.
- Drone: tilt proportional to the interpolated velocity, prop spin, hover bob and strobe are cosmetic.
  There is no flight dynamics, downwash or wind.
- Crawler: wheels turn by distance / radius, and the sensor head pans while scanning. There is no magnetic
  adhesion model, and it drives on the deck rather than on the pipe.
- Gas plume: translucent puffs that rise and drift on a fixed wind vector and fade after an "isolated" event.
  It is a visual cue, not a dispersion simulation, and concentrations in the HUD come from the timeline.
- Status lights, ID plates, rings and beams are overlays for readability.
- The rendering is real RTX path-traced output of the rig scene (`omni.replicator` rgb annotator,
  `omni.videoencoding` MP4).

## Placement checks done

Previews in `sim/output/fleet/preview_*.png`: Q1/Q2 stand on the top deck (trunk at 42.82 = deck +
0.32 m) along `PATROL_ROUTE`. C1's wheels touch the deck. A1 sits on the helideck at t = 0 and otherwise
flies at 46.8 m or higher. Its route to the leak goes round the outside of the rig (west edge, then along
the north face) because a straight line from the helideck passes through a module whose roof is above 52 m;
after the leak it climbs to 60 m before crossing the topsides. The drone chase camera is kept at 57.5 m or
higher for the same reason. The fake timeline keeps every ground robot on the
patrol route; `test_fake_timeline_format_and_shots` checks this.
