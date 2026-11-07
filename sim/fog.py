"""Exponential height fog: optical depth of a straight segment through density rho0 * exp(-z / h). Pure numpy.

sim/ocean_mdl.py evaluates the same formula per shading point; keep them in step.
"""

from __future__ import annotations

import numpy as np


def optical_depth(cam_z, p_z, length, density: float, scale_height: float):
    zc = np.maximum(np.asarray(cam_z, float), 0.0)
    zp = np.maximum(np.asarray(p_z, float), 0.0)
    dz = zc - zp
    ep, ec = np.exp(-zp / scale_height), np.exp(-zc / scale_height)
    mean = np.where(np.abs(dz) > 1e-3, scale_height * (ep - ec) / np.where(np.abs(dz) > 1e-3, dz, 1.0), ep)
    return density * np.asarray(length, float) * mean


def transmittance(cam_z, p_z, length, density: float, scale_height: float, haze_distance: float = 0.0):
    tau = optical_depth(cam_z, p_z, length, density, scale_height)
    if haze_distance > 0:
        tau = tau + np.asarray(length, float) / haze_distance
    return np.exp(-tau)
