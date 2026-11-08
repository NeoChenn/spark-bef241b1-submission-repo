from sim.plume.field import Field, sample_ppm
from sim.plume.release import Gas, Release, orifice

__all__ = ["Field", "Gas", "Plume", "Release", "Wind", "orifice", "sample_ppm"]


def __getattr__(name):
    if name in ("Plume", "Wind"):
        from sim.plume import solver

        return getattr(solver, name)
    raise AttributeError(name)
