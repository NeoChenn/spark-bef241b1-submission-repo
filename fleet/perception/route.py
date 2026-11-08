"""Patrol / escape route geometry without importing Isaac or pxr.

sim/rig_details.py imports pxr at module level, so we pull PATROL_ROUTE, TOP_DECK_Z and route_blockages out of
its source with ``ast`` and run just those definitions. The code that runs is the sim's own.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RIG_DETAILS = REPO / "sim/rig_details.py"
RIG_META = REPO / "sim/output/full_rig_meta.json"

_NS: Optional[dict] = None


def rig_details_symbols() -> dict:
    """{'PATROL_ROUTE', 'TOP_DECK_Z', 'route_blockages'} executed from sim/rig_details.py."""
    global _NS
    if _NS is None:
        tree = ast.parse(RIG_DETAILS.read_text(encoding="utf-8"))
        keep = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("PATROL_ROUTE", "TOP_DECK_Z")
                                                    for t in node.targets):
                keep.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == "route_blockages":
                keep.append(node)
        ns: dict = {"np": np, "math": math}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(RIG_DETAILS), "exec"), ns)
        _NS = ns
    return _NS


def patrol_route(meta_path: Path = RIG_META) -> list[tuple[float, float]]:
    """Top-deck patrol loop (x, y) in metres: the rig meta copy if present, else rig_details.PATROL_ROUTE."""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("patrol_route_m"):
            return [(float(p[0]), float(p[1])) for p in meta["patrol_route_m"]]
    return [tuple(map(float, p)) for p in rig_details_symbols()["PATROL_ROUTE"]]


def meta_blocked_samples(meta_path: Path = RIG_META) -> list[list[float]]:
    if meta_path.exists():
        return list(json.loads(meta_path.read_text(encoding="utf-8")).get("patrol_route_blocked_samples") or [])
    return []


def route_blockages(triangles: np.ndarray, route: Sequence, clearance: float = 0.45) -> list[list[float]]:
    return rig_details_symbols()["route_blockages"](np.asarray(triangles, float), list(route), clearance)


def route_length(route: Sequence) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(route, route[1:])))


def project_on_route(route: Sequence, point: Sequence[float]) -> tuple[float, float]:
    """(distance from the route centreline, chainage along the route) for an (x, y) point."""
    p = np.asarray(point[:2], float)
    best, chain, walked = (math.inf, 0.0), 0.0, 0.0
    for a, b in zip(route, route[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        seg = b - a
        L = float(np.linalg.norm(seg))
        t = 0.0 if L == 0 else float(np.clip((p - a) @ seg / (L * L), 0.0, 1.0))
        d = float(np.linalg.norm(a + t * seg - p))
        if d < best[0]:
            best, chain = (d, 0.0), walked + t * L
        walked += L
    return best[0], chain


def cluster_samples(samples: Sequence[Sequence[float]], gap: float = 1.0) -> list[list[list[float]]]:
    """Group blocked route samples that lie within ``gap`` metres of each other (single linkage)."""
    groups: list[list[list[float]]] = []
    for s in samples:
        s = [float(s[0]), float(s[1])]
        near = [g for g in groups if any(math.dist(s, q) <= gap for q in g)]
        merged = [s]
        for g in near:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)
    return groups
