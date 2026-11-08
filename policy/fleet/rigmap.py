from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RIG_META = "sim/output/full_rig_meta.json"
FLANGE_META = "sim/output/leak_flanges_meta.json"
DECKS = {"lower_deck": 20.0, "middle_deck": 31.25, "top_deck": 42.5}
MAX_SEG_M = 3.0
STANDOFF_M = 0.85
SPEED = {"walk": 1.0, "catwalk": 0.7, "spur": 0.5, "stairs": 0.45}
DOCK_XY = (-12.0, 9.4)
DOCK_PADS = 2

MIDDLE_LOOP = [(-21, -6), (0, -6), (22, -6), (22, 2), (22, 14), (0, 14), (-21, 14), (-21, 0), (-21, -6)]
LOWER_LOOP = [(-18, -4), (1, -4), (20, -4), (20, 12), (1, 12), (-18, 12), (-18, 5), (-18, -4)]
CATWALKS = [("catwalk_mid", "middle_deck", [(0, -6), (0, 14)]), ("catwalk_low", "lower_deck", [(1, -4), (1, 3), (1, 12)])]
STAIRS = [("stairs_west_upper", "top_deck", (-21, 11), "middle_deck", (-21, 0)),
          ("stairs_east_upper", "top_deck", (24, -9), "middle_deck", (22, 2)),
          ("stairs_north_lower", "middle_deck", (0, 14), "lower_deck", (1, 3)),
          ("stairs_west_lower", "middle_deck", (-21, -6), "lower_deck", (-18, 5))]


@dataclass
class Edge:
    length: float
    kind: str
    corridor: str | None = None


class RigMap:
    def __init__(self):
        self.pos: list[np.ndarray] = []
        self.deck: list[str] = []
        self.adj: list[dict[int, Edge]] = []
        self.key: dict[tuple, int] = {}
        self.coarse: list[dict] = []
        self.poi_node: dict[str, int] = {}
        self.poi_pos: dict[str, np.ndarray] = {}
        self.flanges: dict[str, list[float]] = {}
        self.inspection: list[dict] = []
        self.speed = dict(SPEED)
        self.dock = -1
        self.closed: set[tuple[int, int]] = set()
        self.factor: dict[tuple[int, int], float] = {}
        self.kind_factor: dict[str, float] = {}

    def node(self, deck: str, xy, z: float | None = None) -> int:
        k = (deck, round(float(xy[0]), 3), round(float(xy[1]), 3))
        if k not in self.key:
            self.key[k] = len(self.pos)
            self.pos.append(np.array([float(xy[0]), float(xy[1]), DECKS[deck] if z is None else z]))
            self.deck.append(deck)
            self.adj.append({})
        return self.key[k]

    def _link(self, a: int, b: int, kind: str, corridor: str | None):
        L = float(np.linalg.norm(self.pos[a] - self.pos[b]))
        if L < 1e-9:
            return
        n = max(1, math.ceil(L / MAX_SEG_M - 1e-9))
        prev = a
        for i in range(1, n + 1):
            if i == n:
                cur = b
            else:
                p = self.pos[a] + (self.pos[b] - self.pos[a]) * i / n
                cur = len(self.pos)
                self.pos.append(p)
                self.deck.append(self.deck[a] if self.deck[a] == self.deck[b] else f"{self.deck[a]}/{self.deck[b]}")
                self.adj.append({})
            e = Edge(L / n, kind, corridor)
            self.adj[prev][cur] = e
            self.adj[cur][prev] = e
            prev = cur

    def edge(self, deck_a: str, a, deck_b: str, b, kind: str, corridor: str | None = None):
        na, nb = self.node(deck_a, a), self.node(deck_b, b)
        self.coarse.append({"id": corridor or f"w{len(self.coarse)}",
                            "a": self.pos[na].round(3).tolist(), "b": self.pos[nb].round(3).tolist(), "kind": kind,
                            "narrow": corridor is not None, "corridor": corridor,
                            "deck": deck_a if deck_a == deck_b else f"{deck_a}/{deck_b}"})
        self._link(na, nb, kind, corridor)

    def polyline(self, deck: str, pts, kind="walk", corridor=None):
        pts = [(float(a), float(b)) for a, b in pts]
        verts = set(pts) | {tuple(map(float, c[k][:2])) for c in self.coarse if c["deck"] == deck for k in "ab"}
        seen = {frozenset((tuple(map(float, c["a"][:2])), tuple(map(float, c["b"][:2])))) for c in self.coarse
                if c["deck"] == deck}
        for a, b in zip(pts, pts[1:]):
            d = np.subtract(b, a)
            L2 = float(np.dot(d, d))
            cuts = sorted({0.0, 1.0} | {s for v in verts if L2 > 0
                                         and abs(d[0] * (v[1] - a[1]) - d[1] * (v[0] - a[0])) < 1e-6
                                         and 1e-6 < (s := float(np.dot(np.subtract(v, a), d) / L2)) < 1 - 1e-6})
            for s0, s1 in zip(cuts, cuts[1:]):
                p0 = tuple(np.round(np.add(a, d * s0), 3).tolist())
                p1 = tuple(np.round(np.add(a, d * s1), 3).tolist())
                if frozenset((p0, p1)) in seen:
                    continue
                seen.add(frozenset((p0, p1)))
                self.edge(deck, p0, deck, p1, kind, corridor)

    def deck_of(self, z: float) -> str:
        return min((d for d in DECKS if DECKS[d] <= z + 0.5), key=lambda d: z - DECKS[d])

    def add_poi(self, pid: str, xyz, deck: str | None = None, standoff: float = STANDOFF_M) -> int:
        xyz = np.asarray(xyz, np.float64)
        deck = deck or self.deck_of(float(xyz[2]))
        best = None
        for a in range(len(self.pos)):
            if self.deck[a] != deck:
                continue
            for b, e in self.adj[a].items():
                if b < a or e.kind not in ("walk", "catwalk") or self.deck[b] != deck:
                    continue
                pa, pb = self.pos[a][:2], self.pos[b][:2]
                d = pb - pa
                s = float(np.clip(np.dot(xyz[:2] - pa, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0))
                q = pa + s * d
                dist = float(np.linalg.norm(xyz[:2] - q))
                if best is None or dist < best[0]:
                    best = (dist, a, b, s, q)
        dist, a, b, s, q = best
        e = self.adj[a][b]
        if s * e.length < 0.05:
            j = a
        elif (1 - s) * e.length < 0.05:
            j = b
        else:
            j = len(self.pos)
            self.pos.append(np.array([q[0], q[1], DECKS[deck]]))
            self.deck.append(deck)
            self.adj.append({})
            del self.adj[a][b], self.adj[b][a]
            for u in (a, b):
                eu = Edge(float(np.linalg.norm(self.pos[u] - self.pos[j])), e.kind, e.corridor)
                self.adj[u][j] = eu
                self.adj[j][u] = eu
        if dist > standoff + 0.05:
            w = xyz[:2] + (q - xyz[:2]) / dist * standoff
            n = self.node(deck, w)
            self.coarse.append({"id": f"spur_{pid}", "a": self.pos[j].round(3).tolist(),
                                "b": self.pos[n].round(3).tolist(), "kind": "spur",
                                "narrow": True, "corridor": f"spur_{pid}", "deck": deck})
            self._link(j, n, "spur", f"spur_{pid}")
            j = n
        self.poi_node[pid] = j
        self.poi_pos[pid] = xyz
        return j

    def dijkstra(self, src: int, blocked=frozenset()):
        n = len(self.pos)
        dist = np.full(n, math.inf)
        stair = np.zeros(n)
        prev = np.full(n, -1, int)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, e in self.adj[u].items():
                if v in blocked:
                    continue
                dt = self.edge_time(u, v, e)
                if dt == math.inf:
                    continue
                if d + dt < dist[v] - 1e-12:
                    dist[v] = d + dt
                    stair[v] = stair[u] + (dt if e.kind == "stairs" else 0.0)
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        return dist, prev, stair

    def route(self, a: int, b: int, blocked=frozenset()):
        blocked = set(blocked) - {a}
        if b in blocked:
            return math.inf, []
        dist, prev, _ = self.dijkstra(a, blocked)
        if not math.isfinite(dist[b]):
            return math.inf, []
        path = [b]
        while path[-1] != a:
            path.append(int(prev[path[-1]]))
        return float(dist[b]), path[::-1]

    def is_closed(self, a: int, b: int) -> bool:
        return ((a, b) if a < b else (b, a)) in self.closed

    def edge_time(self, a: int, b: int, e: Edge | None = None) -> float:
        k = (a, b) if a < b else (b, a)
        if k in self.closed:
            return math.inf
        e = e or self.adj[a][b]
        return e.length / (self.speed[e.kind] * self.kind_factor.get(e.kind, 1.0) * self.factor.get(k, 1.0))

    def travel_s(self, a: int, b: int) -> float:
        k = (a, b) if a < b else (b, a)
        e = self.adj[a][b]
        return e.length / (self.speed[e.kind] * self.kind_factor.get(e.kind, 1.0) * self.factor.get(k, 1.0))

    def trapped(self, a: int) -> bool:
        return bool(self.adj[a]) and all(self.is_closed(a, b) for b in self.adj[a])

    def escape(self, a: int) -> list[int]:
        dist, prev, pq = {a: 0.0}, {a: -1}, [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if not self.trapped(u):
                path = [u]
                while path[-1] != a:
                    path.append(prev[path[-1]])
                return path[::-1]
            for v in self.adj[u]:
                nd = d + self.travel_s(u, v)
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        return []

    def edges(self):
        for a, nb in enumerate(self.adj):
            for b, e in nb.items():
                if a < b:
                    yield a, b, e

    def to_json(self) -> dict:
        xy = np.array([p[:2] for p in self.pos])
        return {"schema": "fleet_map.v1", "units": "m, Z up, sea level z=0",
                "decks": [{"name": k, "z": v} for k, v in DECKS.items()],
                "walkways": self.coarse,
                "dock": {"xyz": self.pos[self.dock].round(3).tolist(), "pads": DOCK_PADS},
                "flanges": [{"id": k, "xyz": v, "work_xyz": self.pos[self.poi_node[k]].round(3).tolist()}
                            for k, v in self.flanges.items()],
                "inspection_points": [{"id": p["id"], "type": p["type"], "xyz": p["position_m"],
                                       "work_xyz": self.pos[self.poi_node[p["id"]]].round(3).tolist()}
                                      for p in self.inspection if p["id"] in self.poi_node],
                "bounds": {"xmin": float(xy[:, 0].min()) - 3, "xmax": float(xy[:, 0].max()) + 3,
                           "ymin": float(xy[:, 1].min()) - 3, "ymax": float(xy[:, 1].max()) + 3}}


def load_rig_map(rig_meta: str = RIG_META, flange_meta: str = FLANGE_META) -> RigMap:
    meta = json.loads(Path(rig_meta).read_text())
    rm = RigMap()
    route = [(p[0], p[1]) for p in meta["patrol_route_m"]]
    rm.polyline("top_deck", route)
    rm.polyline("middle_deck", MIDDLE_LOOP)
    rm.polyline("lower_deck", LOWER_LOOP)
    for name, deck, pts in CATWALKS:
        rm.polyline(deck, pts, "catwalk", name)
    for name, da, a, db, b in STAIRS:
        rm.edge(da, a, db, b, "stairs", name)
    for f in json.loads(Path(flange_meta).read_text())["flanges"]:
        rm.flanges[f["id"]] = [round(float(c), 3) for c in f["center_m"]]
        rm.add_poi(f["id"], f["center_m"], "top_deck")
    rm.inspection = meta["inspection_points"]
    for p in rm.inspection:
        if p["type"] == "gauge":
            rm.add_poi(p["id"], p["position_m"], "top_deck")
    rm.dock = rm.add_poi("dock", [*DOCK_XY, DECKS["top_deck"]], "top_deck", standoff=0.0)
    return rm
