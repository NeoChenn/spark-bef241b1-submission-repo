"""Time-stepped CPU world for the fleet: walkway graph, per-kind motion, scripted events, timeline file.

Motion model (the coordinator moves robots along the legs this planner returns):
  Q, C  walkway graph built from the rig patrol route (sim/output/full_rig_meta.json "patrol_route_m", else
        sim/rig_details.py PATROL_ROUTE at TOP_DECK_Z). Straight segments; a spur leg joins the walkway to
        the asset. Deck changes only on edges added in `stairs`. Q stays at deck height; C may climb the
        spur onto the vessel or pipe it inspects.
  A     climbs to cruise altitude, flies straight, descends to a standoff above the target
  R     stays below sea level (z <= ROV_MAX_Z) and moves straight

Timeline file (runs/fleet/timeline.jsonl), read by the renderer, one JSON object per line:
  robot lines, one per robot per second: {"t","robot","kind","position":[x,y,z],"yaw","state","job"}
  event lines:                            {"t","event","job","robot","detail"}
"""
from __future__ import annotations

import ast
import heapq
import json
import math
from pathlib import Path
from typing import Callable, Optional

from fleet.contract import REPO, RIG_META, Job, Robot, RobotKind, World
from fleet.coordinator import Coordinator, Leg, Route, dist

TOP_DECK_Z = 42.5
CRUISE_Z = 60.0
DRONE_STANDOFF_Z = 4.0
ROV_MAX_Z = -3.0
TIMELINE = REPO / "runs/fleet/timeline.jsonl"

Vec = tuple[float, float, float]


def patrol_route(meta_path: Path = RIG_META) -> list[Vec]:
    if meta_path.exists():
        pts = json.loads(meta_path.read_text(encoding="utf-8")).get("patrol_route_m")
        if pts:
            return [tuple(float(v) for v in p) for p in pts]
    # rig_details imports pxr at module level, so read the constants from source instead of importing it
    tree = ast.parse((REPO / "sim/rig_details.py").read_text(encoding="utf-8"))
    consts = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and t.id in ("PATROL_ROUTE", "TOP_DECK_Z")}
    z = ast.literal_eval(consts.get("TOP_DECK_Z", ast.Constant(TOP_DECK_Z)))
    return [(float(x), float(y), float(z)) for x, y in ast.literal_eval(consts["PATROL_ROUTE"])]


def _on_segment(p: Vec, a: Vec, b: Vec, tol: float = 1e-6) -> bool:
    ab, ap = dist(a, b), dist(a, p)
    return ap > tol and dist(p, b) > tol and abs(ap + dist(p, b) - ab) < tol


def _closest(p: Vec, a: Vec, b: Vec) -> tuple[Vec, float]:
    ab = [bi - ai for ai, bi in zip(a, b)]
    L2 = sum(v * v for v in ab)
    f = 0.0 if L2 == 0 else max(0.0, min(1.0, sum((pi - ai) * v for pi, ai, v in zip(p, a, ab)) / L2))
    q = tuple(ai + v * f for ai, v in zip(a, ab))
    return q, f


class Walkways:
    """Undirected graph of straight walkway segments. Node ids are indices into `nodes`."""

    def __init__(self, polyline: list[Vec], stairs: Optional[list[tuple[Vec, Vec]]] = None):
        nodes: list[Vec] = []

        def nid(p):
            p = tuple(round(float(v), 3) for v in p)
            if p not in nodes:
                nodes.append(p)
            return nodes.index(p)

        raw = [(nid(a), nid(b)) for a, b in zip(polyline, polyline[1:])]
        raw += [(nid(a), nid(b)) for a, b in (stairs or [])]
        edges: set[tuple[int, int]] = set()
        for i, j in raw:                       # split edges at T-junctions
            inner = sorted((k for k in range(len(nodes)) if _on_segment(nodes[k], nodes[i], nodes[j])),
                           key=lambda k: dist(nodes[i], nodes[k]))
            chain = [i] + inner + [j]
            for a, b in zip(chain, chain[1:]):
                if a != b:
                    edges.add((min(a, b), max(a, b)))
        self.nodes = nodes
        self.edges = sorted(edges)
        self.seg_id = {e: f"w{k:02d}" for k, e in enumerate(self.edges)}
        self.adj: dict[int, list[int]] = {k: [] for k in range(len(nodes))}
        for a, b in self.edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        self._sp: dict[int, tuple] = {}

    def snap(self, p: Vec) -> tuple[Vec, tuple[int, int]]:
        best = None
        for e in self.edges:
            q, _ = _closest(p, self.nodes[e[0]], self.nodes[e[1]])
            d = dist((p[0], p[1], 0), (q[0], q[1], 0)) + abs(p[2] - q[2]) * 4.0   # prefer the same deck
            if best is None or d < best[0]:
                best = (d, q, e)
        return best[1], best[2]

    def shortest(self, src: int) -> tuple[dict[int, float], dict[int, int]]:
        if src not in self._sp:
            self._sp[src] = self._dijkstra(src)
        return self._sp[src]

    def _dijkstra(self, src: int) -> tuple[dict[int, float], dict[int, int]]:
        d, prev, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            c, u = heapq.heappop(pq)
            if c > d.get(u, math.inf):
                continue
            for v in self.adj[u]:
                nc = c + dist(self.nodes[u], self.nodes[v])
                if nc < d.get(v, math.inf):
                    d[v], prev[v] = nc, u
                    heapq.heappush(pq, (nc, v))
        return d, prev

    def path(self, a: Vec, b: Vec) -> list[Leg]:
        """Walkway legs from point a to point b (both already on the graph)."""
        pa, ea = self.snap(a)
        pb, eb = self.snap(b)
        if ea == eb:
            return [Leg(pa, pb, self.seg_id[ea])]
        best = None
        for u in ea:
            d, prev = self.shortest(u)
            for v in eb:
                if v in d:
                    c = dist(pa, self.nodes[u]) + d[v] + dist(self.nodes[v], pb)
                    if best is None or c < best[0]:
                        best = (c, u, v, prev)
        if best is None:
            return [Leg(pa, pb, None)]
        _, u, v, prev = best
        chain = [v]
        while chain[-1] != u:
            chain.append(prev[chain[-1]])
        chain.reverse()
        legs = [Leg(pa, self.nodes[u], self.seg_id[ea])]
        for x, y in zip(chain, chain[1:]):
            legs.append(Leg(self.nodes[x], self.nodes[y], self.seg_id[(min(x, y), max(x, y))]))
        legs.append(Leg(self.nodes[v], pb, self.seg_id[eb]))
        return legs


class WalkwayPlanner:
    def __init__(self, walkways: Walkways, deck_z: float = TOP_DECK_Z):
        self.w = walkways
        self.deck_z = deck_z

    def route(self, robot: Robot, goal: Vec, standoff: float = 0.0) -> Route:
        start = tuple(float(v) for v in robot.position)
        goal = tuple(float(v) for v in goal)
        if robot.kind == RobotKind.AERIAL:
            g = (goal[0], goal[1], goal[2] + DRONE_STANDOFF_Z)
            alt = max(CRUISE_Z, start[2], g[2])
            pts = [start, (start[0], start[1], alt), (g[0], g[1], alt), g]
            return Route([Leg(a, b) for a, b in zip(pts, pts[1:]) if dist(a, b) > 1e-6])
        if robot.kind == RobotKind.ROV:
            g = (goal[0], goal[1], min(goal[2], ROV_MAX_Z))
            s = (start[0], start[1], min(start[2], ROV_MAX_Z))
            pts = [start, s, g]
            return Route([Leg(a, b) for a, b in zip(pts, pts[1:]) if dist(a, b) > 1e-6])
        on_start, _ = self.w.snap(start)
        on_goal, _ = self.w.snap(goal)
        if robot.kind == RobotKind.QUADRUPED:
            end = (goal[0], goal[1], on_goal[2])          # stand on the deck in front of the asset
        else:
            end = goal                                     # crawler climbs onto the asset
        spur = dist(on_goal, end)
        if standoff and spur > standoff:
            f = (spur - standoff) / spur
            end = tuple(a + (b - a) * f for a, b in zip(on_goal, end))
        legs = [Leg(start, on_start)] + self.w.path(on_start, on_goal) + [Leg(on_goal, end)]
        return Route([l for l in legs if l.length > 1e-6])


class FleetSim:
    """Steps the coordinator once per second, runs scripted events and ambient sensing, writes the timeline."""

    def __init__(self, coordinator: Coordinator, timeline_path: Optional[Path] = TIMELINE,
                 gas_sense_radius_m: float = 6.0):
        self.c = coordinator
        self.world: World = coordinator.world
        self.path = Path(timeline_path) if timeline_path else None
        self.lines: list[dict] = []
        self.scripted: list[tuple[float, str, Callable[["FleetSim"], None]]] = []
        self.people_paths: list[Callable[[float], Optional[Vec]]] = []
        self.gas_sense_radius_m = gas_sense_radius_m
        self.on_gas: Optional[Callable[["FleetSim", Optional[str], dict, str], None]] = None
        self.world.state.setdefault("gas_leaks", [])
        coordinator.on_event = self._event

    def _event(self, e: dict):
        self.lines.append(e)

    def at(self, t: float, label: str, fn: Callable[["FleetSim"], None]):
        self.scripted.append((t, label, fn))
        self.scripted.sort(key=lambda s: s[0])

    def _sense(self):
        """Robots with a gas sensor pick up leaks within range; fixed detectors alarm after `alarm_after_s`."""
        for leak in self.world.state["gas_leaks"]:
            if leak.get("reported") or leak.get("isolated"):
                continue
            for rt in self.c.rt.values():
                if "gas_sensor" in rt.robot.capabilities and dist(rt.robot.position, leak["position"]) < self.gas_sense_radius_m:
                    leak["reported"] = True
                    if self.on_gas:
                        self.on_gas(self, rt.robot.id, leak, "gas_detected")
                    break
            else:
                if self.world.t >= leak["t0"] + leak.get("alarm_after_s", 60.0):
                    leak["reported"] = True
                    if self.on_gas:
                        self.on_gas(self, None, leak, "gas_alarm")

    def step(self, dt: float = 1.0):
        while self.scripted and self.scripted[0][0] <= self.world.t:
            _, label, fn = self.scripted.pop(0)
            fn(self)
        self.world.people = [p for p in (f(self.world.t) for f in self.people_paths) if p is not None]
        self._sense()
        t = self.world.t
        for rt in self.c.rt.values():          # state at time t, then the coordinator acts for [t, t + dt)
            r = rt.robot
            self.lines.append({"t": round(t, 2), "robot": r.id, "kind": r.kind.value,
                               "position": [round(float(v), 3) for v in r.position],
                               "yaw": round(rt.yaw, 4), "state": rt.state, "job": rt.job})
        self.c.tick(dt)

    def run(self, duration_s: float, dt: float = 1.0):
        end = self.world.t + duration_s
        while self.world.t < end - 1e-9:
            self.step(dt)
        self.write()

    def write(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for line in self.lines:
                fh.write(json.dumps(line) + "\n")


def walk(points: list[Vec], t0: float, speed: float = 0.8, linger_s: float = 0.0) -> Callable[[float], Optional[Vec]]:
    """Person path for FleetSim.people_paths: walks the polyline from t0, lingers at the end, then leaves."""
    lengths = [dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    t_end = t0 + total / speed

    def pos(t: float) -> Optional[Vec]:
        if t < t0 or t > t_end + linger_s:
            return None
        s = min((t - t0) * speed, total)
        for (a, b), L in zip(zip(points, points[1:]), lengths):
            if s <= L:
                f = 0.0 if L == 0 else s / L
                return tuple(ai + (bi - ai) * f for ai, bi in zip(a, b))
            s -= L
        return tuple(points[-1])
    return pos


def new_job(c: Coordinator, kind: str, target: str, position: Vec, level, **kw) -> Job:
    return Job(id=c.new_id(kind), kind=kind, target=target, position=tuple(position), level=level, **kw)
