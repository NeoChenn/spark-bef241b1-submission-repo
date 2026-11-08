from __future__ import annotations

import bisect
from functools import lru_cache

from policy.fleet.jobs import SETUP_S
from policy.weld.job import make_job
from policy.weld.overlay import thumb


class Replay:
    def __init__(self, scenario: str = "crack", seed: int = 1, every_s: float = 0.5):
        j = make_job(scenario, seed)
        self.frames: list[tuple[float, str, dict, dict]] = []
        last = [-1e9]

        def grab(job):
            if job.t - last[0] >= every_s:
                last[0] = job.t
                self.frames.append((job.t, job.state, dict(job.ev), thumb(job, len(self.frames))))
        j.on_frame = grab
        j.run()
        self.frames.append((j.t, j.state, dict(j.ev), thumb(j, len(self.frames))))
        self.ts = [f[0] for f in self.frames]
        self.result = j.result

    def at(self, elapsed: float) -> dict:
        if elapsed < 0:
            t, _, ev, th = self.frames[0]
            return {"state": "SETUP", "evidence": {"job": "weld"}, "pip": th, "t": 0.0}
        i = max(bisect.bisect_right(self.ts, elapsed) - 1, 0)
        t, st, ev, th = self.frames[i]
        return {"state": st, "evidence": ev, "pip": th, "t": t}


@lru_cache(maxsize=8)
def replay(scenario: str = "crack") -> Replay:
    return Replay(scenario)


def weld_view(scenario: str, elapsed: float) -> dict:
    return replay(scenario).at(elapsed)


def fleet_weld_views(fleet) -> dict[str, dict]:
    out = {}
    for r in fleet.robots:
        t = r.task
        if t is not None and t.kind == "weld" and t.status == "active":
            out[r.id] = weld_view(t.scenario, t.progress - SETUP_S["weld"])
    return out
