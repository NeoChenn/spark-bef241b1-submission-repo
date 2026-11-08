from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class Leg:
    u: int
    v: int
    t0: float
    t1: float
    kind: str
    corridor: Optional[str] = None
    back: bool = False


class Reservations:
    def __init__(self, clearance: float = 1.0):
        self.clearance = clearance
        self.table: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

    def earliest(self, corridor: str, t: float, dur: float, robot: str) -> float:
        s = t
        others = sorted((a, b) for a, b, r in self.table[corridor] if r != robot)
        moved = True
        while moved:
            moved = False
            for a, b in others:
                if s < b + self.clearance and a < s + dur + self.clearance:
                    s = b + self.clearance
                    moved = True
        return s

    def reserve(self, corridor: str, t0: float, t1: float, robot: str) -> None:
        self.table[corridor].append((t0, t1, robot))

    def claim(self, corridor: str, t0: float, t1: float, robot: str) -> set[str]:
        hit = {r for a, b, r in self.table[corridor] if r != robot and a < t1 + self.clearance and t0 < b + self.clearance}
        self.table[corridor] = [x for x in self.table[corridor] if x[2] not in hit] + [(t0, t1, robot)]
        return hit

    def release(self, robot: str, after: float = -math.inf) -> None:
        for c in self.table:
            self.table[c] = [x for x in self.table[c] if x[2] != robot or x[0] < after]

    def prune(self, t: float) -> None:
        for c in self.table:
            self.table[c] = [x for x in self.table[c] if x[1] + self.clearance >= t]


def plan_legs(rm, res: Reservations, robot: str, path: list[int], t: float) -> list[Leg]:
    legs: list[Leg] = []
    i = 0
    edges = list(zip(path, path[1:]))
    while i < len(edges):
        c = rm.adj[edges[i][0]][edges[i][1]].corridor
        j = i + 1
        if c:
            while j < len(edges) and rm.adj[edges[j][0]][edges[j][1]].corridor == c:
                j += 1
            dur = sum(rm.travel_s(a, b) for a, b in edges[i:j])
            t = res.earliest(c, t, dur, robot)
            res.reserve(c, t, t + dur, robot)
        for a, b in edges[i:j]:
            e = rm.adj[a][b]
            dt = rm.travel_s(a, b)
            legs.append(Leg(a, b, t, t + dt, e.kind, e.corridor))
            t += dt
        i = j
    return legs
