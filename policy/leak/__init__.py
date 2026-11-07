"""Gas-leak job for the rig robot: a second job alongside rust repair, in the same plan engine.

See docs/gas-leak-integration.md. Layout mirrors the rust job:
  geometry.py  flange geometry (bolt circle, torque stripes), shared by the sim builder and perception
  world.py     LeakWorld: bolt looseness, cracks and the gas they release (like RustState)
  diagnose.py  loose bolt / crack / unsure / unexplained / clean, from camera readings
  job.py       RigOrchestrator: the team Orchestrator plus the leak states and gas pre-emption
  mock.py      CPU mock: renders the flange for the wrist camera; demo and tests run without Isaac
Perception lives in detection/leak.py, like detection/localize.py for rust.
"""
