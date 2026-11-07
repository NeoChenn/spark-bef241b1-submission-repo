"""Classical brush-paint skill (primer/topcoat) with coverage verification.

The brush is in contact (0.05 m standoff = brush bristle length to the tool
flange). Coverage is tracked in ``painted`` (n_theta, n_z); gaps (unreached
waypoints) are re-painted until coverage >= ``min_coverage`` or passes run out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from policy.interfaces import PipeGeometry
from policy.skills.coverage import coverage_path, covered_by, dilate
from policy.skills.spray import to_world_mask


@dataclass
class PaintSkill:
    standoff: float = 0.05
    flow_rate: float = 0.2       # L/min paint feed to brush
    margin: int = 1
    lane_width: int = 1          # brush is one cell wide
    min_coverage: float = 0.98
    max_passes: int = 3
    painted: Optional[np.ndarray] = None
    passes: list = field(default_factory=list)

    def target_for(self, mask: np.ndarray) -> np.ndarray:
        return dilate(mask, self.margin)

    def coverage(self, target: np.ndarray) -> float:
        if self.painted is None or not target.any():
            return 1.0 if not target.any() else 0.0
        return float((self.painted & target).sum() / target.sum())

    def paint_pass(self, mask: np.ndarray, pipe: PipeGeometry, arm, world) -> float:
        """One boustrophedon pass over ``mask`` (no extra margin). Returns coverage gain."""
        wps, tgt = coverage_path(mask, pipe, margin=0, lane_width=self.lane_width,
                                 standoff=self.standoff, flow_rate=self.flow_rate)
        if self.painted is None:
            self.painted = np.zeros(mask.shape, bool)
        reached = arm.move_to(wps) if wps else []
        reached = list(reached) if reached is not None else [True] * len(wps)
        done = [w.cell for w, ok in zip(wps, reached) if ok]
        new = covered_by(done, self.lane_width, mask.shape) & tgt
        self.painted |= new
        if new.any() and hasattr(world, "paint"):
            world.paint(to_world_mask(world, new))
        self.passes.append({"n_wp": len(wps), "n_reached": len(done), "cells": int(new.sum())})
        return float(new.sum())

    def execute(self, rust_mask: np.ndarray, pipe: PipeGeometry, arm, world) -> dict:
        """Paint target = dilate(rust_mask); re-paint gaps until >= min_coverage."""
        target = self.target_for(rust_mask)
        self.paint_pass(target, pipe, arm, world)
        n = 1
        while self.coverage(target) < self.min_coverage and n < self.max_passes:
            gaps = target & ~self.painted
            self.paint_pass(gaps, pipe, arm, world)
            n += 1
        return {"coverage": self.coverage(target), "passes": n,
                "gaps": int((target & ~self.painted).sum()), "target_cells": int(target.sum())}
