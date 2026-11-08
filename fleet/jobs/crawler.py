from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from fleet.aerial.patches import grade_classical
from fleet.contract import Finding, Job, JobResult, JobStatus, Level, Robot, RobotKind, World, register
from fleet.jobs.aerial import patch_image


def mark_repaired(world: World, patch_id: str, robot_id: str, job_id: str) -> dict:
    p = world.state["topsides_patches"][patch_id]
    p.update({"repaired": True, "repaired_by": robot_id, "repair_job": job_id, "repaired_t": world.t})
    p.pop("verified", None)
    return p


def verify_job(parent: Job, patch_id: str, position, t: float) -> Job:
    return Job(f"{parent.id}-verify", "corrosion_survey", patch_id, tuple(position), Level.AUTO, {"camera"},
               {RobotKind.AERIAL}, 3, 120.0, {"patch_ids": [patch_id], "verify_of": parent.id}, t, parent.id)


def spray_path(region: np.ndarray, m_per_px: float, fan_m: float = 0.10, overlap: float = 0.5,
               margin_m: Optional[float] = None) -> list[tuple[float, float]]:
    ys, xs = np.nonzero(region)
    margin = fan_m / 2 if margin_m is None else margin_m
    x0, x1 = xs.min() * m_per_px - margin, (xs.max() + 1) * m_per_px + margin
    y0, y1 = ys.min() * m_per_px, (ys.max() + 1) * m_per_px
    pitch = fan_m * (1.0 - overlap)
    n = max(1, int(math.ceil((y1 - y0) / pitch)) + 1)
    path = []
    for i in range(n):
        y = y0 + i * pitch
        a, b = (x0, x1) if i % 2 == 0 else (x1, x0)
        path += [(a, y), (b, y)]
    return path


def coverage(region: np.ndarray, m_per_px: float, path: list[tuple[float, float]], fan_m: float) -> dict:
    h, w = region.shape
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) * m_per_px
    count = np.zeros(region.shape, np.int16)
    for (ax, ay), (bx, by) in zip(path[0::2], path[1::2]):
        lo, hi = min(ax, bx), max(ax, bx)
        count += ((np.abs(yy - ay) <= fan_m / 2) & (xx >= lo) & (xx <= hi)).astype(np.int16)
    c = count[region]
    return {"covered_frac": float((c >= 1).mean()), "two_coat_frac": float((c >= 2).mean()),
            "min_passes": int(c.min()), "mean_passes": float(c.mean())}


class CoatingTouchup:
    kind = "coating_touchup"
    level = Level.SUPV
    needs = {"spray"}
    robot_kinds = {RobotKind.CRAWLER}

    def __init__(self, fan_m: float = 0.10, overlap: float = 0.5, feather_m: float = 0.025,
                 spray_speed_mps: float = 0.25, prep_m2_per_h: float = 1.0, dft_um: float = 250.0,
                 volume_solids: float = 0.7, transfer_eff: float = 0.6):
        self.fan_m, self.overlap, self.feather_m = fan_m, overlap, feather_m
        self.spray_speed, self.prep_rate = spray_speed_mps, prep_m2_per_h
        self.dft_um, self.vs, self.te = dft_um, volume_solids, transfer_eff

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        if job.params.get("approved") is False:
            return JobResult(job.id, robot.id, JobStatus.FAILED, world.t, world.t,
                             log=["coating_touchup is SUPV: operator approval missing"])
        pid = job.params.get("patch_id", job.target)
        patch = world.state["topsides_patches"][pid]
        img = patch_image(pid, patch)
        size_m = patch.get("size_m", [0.6, 0.6])
        m_per_px = float(size_m[0]) / img.shape[1]
        _, rust = grade_classical(img)
        if rust.any():
            region = distance_transform_edt(~rust) * m_per_px <= self.feather_m
        else:
            region = np.ones(rust.shape, bool)
        prep_m2 = float(region.sum()) * m_per_px ** 2
        path = spray_path(region, m_per_px, self.fan_m, self.overlap)
        cov = coverage(region, m_per_px, path, self.fan_m)
        length = sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))
        spray_s = length / self.spray_speed + 1.5 * (len(path) // 2)
        prep_s = prep_m2 / self.prep_rate * 3600.0
        sprayed_m2 = (len(path) // 2) * self.fan_m * (1 - self.overlap) * abs(path[0][0] - path[1][0])
        paint_l = sprayed_m2 * self.dft_um * 1e-6 / (self.vs * self.te) * 1000.0
        rust_m2 = float(rust.sum()) * m_per_px ** 2
        evidence = {"patch_id": pid, "rust_area_m2": round(rust_m2, 5), "prep_area_m2": round(prep_m2, 4),
                    "prep": "ISO 8501-1 St 3 power-tool cleaning, 25 mm feather", "passes": len(path) // 2,
                    "path_m": round(length, 2), "spray_s": round(spray_s), "prep_s": round(prep_s),
                    "paint_l": round(float(paint_l), 3), "dft_um": self.dft_um, **{k: round(v, 4) if isinstance(v, float)
                                                                          else v for k, v in cov.items()}}
        log = [f"prep {prep_m2:.4f} m2 over {rust_m2:.5f} m2 rust; {len(path) // 2} passes, {length:.2f} m path; "
               f"coverage {cov['covered_frac']:.1%} (two coats {cov['two_coat_frac']:.1%})"]
        done_t = world.t + prep_s + spray_s
        if cov["covered_frac"] < 0.99:
            return JobResult(job.id, robot.id, JobStatus.FAILED, world.t, done_t,
                             [Finding("touchup_incomplete", "warning", f"{pid}: coverage {cov['covered_frac']:.1%}",
                                      tuple(patch["position"]), evidence)], log)
        mark_repaired(world, pid, robot.id, job.id)
        finding = Finding("coating_touchup_done", "info",
                          f"{pid}: touched up {prep_m2:.3f} m2, coverage {cov['covered_frac']:.1%}; drone to verify",
                          tuple(patch["position"]), evidence, [verify_job(job, pid, patch["position"], done_t)])
        return JobResult(job.id, robot.id, JobStatus.DONE, world.t, done_t, [finding], log)


DEFAULT_VESSEL = {"nominal_mm": 12.7, "ca_mm": 3.0, "radius_m": 1.2, "height_m": 4.0, "uniform_loss_mm": 0.6,
                  "pits": []}


def true_thickness(v: dict, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    t = np.full(np.broadcast(theta, z).shape, v["nominal_mm"] - v.get("uniform_loss_mm", 0.0), float)
    for pit in v.get("pits", []):
        dth = (theta - math.radians(pit["theta_deg"]) + math.pi) % (2 * math.pi) - math.pi
        d2 = (dth * v["radius_m"]) ** 2 + (z - pit["z_m"]) ** 2
        t = t - pit["depth_mm"] * np.exp(-d2 / (2 * pit["sigma_m"] ** 2))
    return t


class UtThickness:
    kind = "ut_thickness"
    level = Level.AUTO
    needs = {"ut_probe"}
    robot_kinds = {RobotKind.CRAWLER}

    def __init__(self, noise_mm: float = 0.05, dropout: float = 0.03, s_per_point: float = 6.0,
                 warn_margin_mm: float = 1.0):
        self.noise, self.dropout, self.s_per_point, self.warn = noise_mm, dropout, s_per_point, warn_margin_mm

    def run(self, job: Job, robot: Robot, world: World) -> JobResult:
        v = {**DEFAULT_VESSEL, **world.state.get("vessels", {}).get(job.target, {})}
        rng = np.random.default_rng(int(job.params.get("seed", 0)))
        grid = float(job.params.get("grid_m", 0.3))
        r, height = v["radius_m"], v["height_m"]
        if "region" in job.params:
            reg = job.params["region"]
            half = reg["half_m"]
            ths = math.radians(reg["theta_deg"]) + np.arange(-half, half + 1e-9, grid) / r
            zs = np.clip(np.arange(reg["z_m"] - half, reg["z_m"] + half + 1e-9, grid), 0, height)
        else:
            ths = np.arange(0, 2 * math.pi, grid / r)
            zs = np.arange(grid / 2, height, grid)
        TH, Z = np.meshgrid(ths, zs)
        truth = true_thickness(v, TH, Z)
        reading = truth + rng.normal(0, self.noise, truth.shape)
        lost = rng.random(truth.shape) < self.dropout
        retakes = int(lost.sum())
        reading[lost] = truth[lost] + rng.normal(0, self.noise, retakes)
        t_min = v["nominal_mm"] - v["ca_mm"]
        i = np.unravel_index(np.argmin(reading), reading.shape)
        tmin_read, th_at, z_at = float(reading[i]), float(TH[i]), float(Z[i])
        cx, cy, cz = job.position
        pos = (cx + r * math.cos(th_at), cy + r * math.sin(th_at), cz + z_at)
        n = reading.size
        duration = (n + retakes) * self.s_per_point + 0.3 * n * grid / max(robot.speed_mps, 0.05)
        evidence = {"vessel": job.target, "points": n, "retakes": retakes, "grid_m": grid,
                    "nominal_mm": v["nominal_mm"], "ca_mm": v["ca_mm"], "t_min_allowed_mm": round(t_min, 2),
                    "min_mm": round(tmin_read, 2), "mean_mm": round(float(reading.mean()), 2),
                    "min_at": {"theta_deg": round(math.degrees(th_at) % 360, 1), "z_m": round(z_at, 2)},
                    "points_below_allowed": int((reading <= t_min).sum())}
        if "previous_min_mm" in job.params and job.params.get("years_since"):
            rate = (job.params["previous_min_mm"] - tmin_read) / job.params["years_since"]
            evidence["corrosion_rate_mm_y"] = round(rate, 3)
            evidence["remaining_life_y"] = round((tmin_read - t_min) / rate, 1) if rate > 0 else None
        log = [f"{n} UT points ({retakes} re-takes), min {tmin_read:.2f} mm at {evidence['min_at']}, "
               f"allowed {t_min:.2f} mm"]
        regrid = Job(f"{job.id}-regrid", "ut_thickness", job.target, pos, Level.AUTO, {"ut_probe"},
                     {RobotKind.CRAWLER}, 2, 600.0,
                     {"grid_m": 0.05, "region": {"theta_deg": math.degrees(th_at), "z_m": z_at, "half_m": 0.25},
                      "seed": int(job.params.get("seed", 0)) + 1}, world.t, job.id)
        findings = []
        if tmin_read <= t_min:
            ffs = Job(f"{job.id}-ffs", "ffs_assessment", job.target, pos, Level.HUMAN, set(), set(), 1, 0.0,
                      {"standard": "API 579-1 Part 4/5", **evidence}, world.t, job.id)
            follow = [ffs] + ([] if "region" in job.params else [regrid])
            findings.append(Finding("wall_below_corrosion_allowance", "critical",
                                    f"{job.target}: {tmin_read:.2f} mm < allowed {t_min:.2f} mm "
                                    f"(nominal {v['nominal_mm']} - CA {v['ca_mm']})", pos, evidence, follow))
        elif tmin_read <= t_min + self.warn:
            findings.append(Finding("wall_loss_near_allowance", "warning",
                                    f"{job.target}: {tmin_read:.2f} mm, within {self.warn} mm of {t_min:.2f} mm",
                                    pos, evidence, [] if "region" in job.params else [regrid]))
        status = JobStatus.FINDING if findings else JobStatus.DONE
        return JobResult(job.id, robot.id, status, world.t, world.t + duration, findings, log)


COATING_TOUCHUP = register(CoatingTouchup())
UT_THICKNESS = register(UtThickness())
