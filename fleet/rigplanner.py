from __future__ import annotations

import math
from typing import Optional

import numpy as np

from fleet.contract import Robot, RobotKind
from fleet.coordinator import Leg, Route, Vec, dist
from fleet.sim import air_route, sea_route, spur_end
from policy.fleet.rigmap import DECKS, RigMap, load_rig_map

DECK_Z_WEIGHT = 4.0
SPUR_SPEED = 0.5


class RigPlanner:
    def __init__(self, rm: Optional[RigMap] = None):
        self.rm = rm or load_rig_map()
        self.blocked: frozenset[int] = frozenset()
        self._sp: dict = {}
        self._pos = np.array(self.rm.pos)

    def _version(self):
        rm = self.rm
        return (frozenset(getattr(rm, "closed", ())), tuple(sorted(getattr(rm, "factor", {}).items())),
                tuple(sorted(getattr(rm, "kind_factor", {}).items())))

    def nearest(self, p: Vec, deck: Optional[str] = None) -> int:
        if len(self._pos) != len(self.rm.pos):
            self._pos = np.array(self.rm.pos)
        d = np.hypot(self._pos[:, 0] - p[0], self._pos[:, 1] - p[1]) + DECK_Z_WEIGHT * np.abs(self._pos[:, 2] - p[2])
        if deck:
            d = d + np.array([0.0 if k == deck else 1e6 for k in self.rm.deck])
        return int(np.argmin(d))

    def xyz(self, i: int) -> Vec:
        return tuple(float(v) for v in self.rm.pos[i])

    def set_obstacles(self, points: list[tuple[Vec, float]]) -> bool:
        out = set()
        for p, radius in points:
            near = (np.hypot(self._pos[:, 0] - p[0], self._pos[:, 1] - p[1]) < radius) & (np.abs(self._pos[:, 2] - p[2]) < 3.0)
            out |= set(np.flatnonzero(near).tolist())
        new = frozenset(out)
        changed = new != self.blocked
        self.blocked = new
        return changed

    def tree(self, src: int, blocked: bool = True):
        key = (src, self.blocked if blocked else None, self._version())
        if key not in self._sp:
            if len(self._sp) > 4096:
                self._sp.clear()
            self._sp[key] = self.rm.dijkstra(src, (self.blocked - {src}) if blocked else frozenset())
        return self._sp[key]

    def path(self, a: int, b: int) -> tuple[list[int], bool]:
        for blocked in (True, False):
            dist_t, prev, _ = self.tree(a, blocked)
            if math.isfinite(dist_t[b]):
                out = [b]
                while out[-1] != a:
                    out.append(int(prev[out[-1]]))
                return out[::-1], not blocked
        return [], True

    def goal_deck(self, goal: Vec) -> str:
        return self.rm.deck_of(float(goal[2]))

    def route(self, robot: Robot, goal: Vec, standoff: float = 0.0) -> Route:
        start = tuple(float(v) for v in robot.position)
        goal = tuple(float(v) for v in goal)
        if robot.kind == RobotKind.AERIAL:
            return air_route(start, goal)
        if robot.kind == RobotKind.ROV:
            return sea_route(start, goal)
        rm = self.rm
        a = self.nearest(start)
        b = self.nearest(goal, self.goal_deck(goal))
        nodes, blocked = self.path(a, b)
        if not nodes:
            return Route([Leg(start, goal)], blocked=True)
        legs = [Leg(start, self.xyz(a), None, SPUR_SPEED, "spur", a)]
        for u, v in zip(nodes, nodes[1:]):
            e = rm.adj[u][v]
            legs.append(Leg(self.xyz(u), self.xyz(v), e.corridor, e.length / max(rm.travel_s(u, v), 1e-9),
                            e.kind, v))
        on_goal = self.xyz(b)
        legs.append(Leg(on_goal, spur_end(robot, on_goal, goal, standoff), None, SPUR_SPEED, "spur"))
        return Route([l for l in legs if l.length > 1e-6], blocked=blocked)

    def dock_cost(self, frm: Vec, dock: Vec, speed_mps: float) -> tuple[float, float]:
        d = self.nearest(dock, self.goal_deck(dock))
        a = self.nearest(frm)
        t, _, st = self.tree(d, blocked=False)
        if not math.isfinite(t[a]):
            return math.inf, 0.0
        walk = dist(frm, self.rm.pos[a]) / SPUR_SPEED + dist(dock, self.rm.pos[d]) / SPUR_SPEED
        v = max(speed_mps, 1e-6)
        return (float(t[a]) + walk) / v, float(st[a]) / v

    def deck_name(self, z: float) -> str:
        d = min(DECKS, key=lambda k: abs(DECKS[k] - z))
        return d if abs(DECKS[d] - z) < 0.5 else "stairs"

    def to_json(self) -> dict:
        return self.rm.to_json()
