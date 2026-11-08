from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from policy.interfaces import PipeGeometry, Waypoint
from policy.skills.coverage import coverage_path, covered_by


def to_world_mask(world, mask: np.ndarray):
    lv = getattr(world, "rust", getattr(world, "levels", None))
    try:
        import torch
        if isinstance(lv, torch.Tensor):
            return torch.as_tensor(np.asarray(mask, bool), device=lv.device)
    except ImportError:
        pass
    return np.asarray(mask, bool)


@dataclass
class SprayPlan:
    waypoints: list[Waypoint]
    target: np.ndarray
    standoff: float
    flow_rate: float
    lane_width: int


@dataclass
class SpraySkill:
    standoff: float = 0.15
    flow_rate: float = 0.6
    margin: int = 1
    lane_width: int = 3
    history: list = field(default_factory=list)

    def plan(self, mask: np.ndarray, pipe: PipeGeometry) -> SprayPlan:
        wps, target = coverage_path(mask, pipe, self.margin, self.lane_width,
                                    self.standoff, self.flow_rate)
        return SprayPlan(wps, target, self.standoff, self.flow_rate, self.lane_width)

    def execute(self, plan: SprayPlan, arm, world) -> np.ndarray:
        reached = arm.move_to(plan.waypoints) if plan.waypoints else []
        reached = list(reached) if reached is not None else [True] * len(plan.waypoints)
        done = [w.cell for w, ok in zip(plan.waypoints, reached) if ok]
        sprayed = covered_by(done, plan.lane_width, plan.target.shape, z_half=1) & plan.target
        if sprayed.any():
            world.spray(to_world_mask(world, sprayed))
        self.history.append({"n_wp": len(plan.waypoints), "n_reached": len(done),
                             "cells": int(sprayed.sum())})
        return sprayed
