from datetime import date

from fleet.contract import Robot, RobotKind, World
from fleet.coordinator import Coordinator
from fleet.rigplanner import RigPlanner
from fleet.sim import FleetSim
from policy.control.brain import ArgonBrain, VerifySensor
from policy.control.centre import ControlCentre
from policy.control.fleet_port import FleetPort
from policy.control.rigsensors import RigSensors
from policy.control.schema import VERIFY
from policy.fleet.rigmap import DECKS, DOCK_XY

DAY = date(2026, 11, 8)
DOCK = (DOCK_XY[0], DOCK_XY[1], DECKS["top_deck"])
Q, C, A = RobotKind.QUADRUPED, RobotKind.CRAWLER, RobotKind.AERIAL
SPEED = {Q: 1.0, C: 0.3, A: 4.0}


def coordinator(robots, executors=None):
    planner = RigPlanner()
    world = World.from_rig_meta()
    world.state["flanges"] = {}
    rs = [Robot(rid, kind, set(caps), tuple(xyz), battery=soc, speed_mps=SPEED[kind]) for rid, kind, caps, xyz, soc in robots]
    return Coordinator(rs, world, {VERIFY: VerifySensor(), **(executors or {})}, planner,
                       docks={r[0]: DOCK if r[1] == Q else tuple(r[3]) for r in robots})


def world(robots, backend=None, operator_s=None, **kw):
    c = coordinator(robots)
    sim = FleetSim(c, timeline_path=None)
    rig = RigSensors(seed=3)
    port = FleetPort(c, rig.specs)
    centre = ControlCentre(backend, log_path=None, today=lambda: DAY, timeline=port, **kw)
    return ArgonBrain(sim, rig, centre, port, operator_s=operator_s)


def run(brain, n=1, until=None):
    for _ in range(int(n)):
        brain.step()
        if until is not None and until():
            return True
    return until is None


def events(brain, name, **match):
    return [e for e in brain.c.timeline if e["event"] == name and all(e["detail"].get(k) == v for k, v in match.items())]
