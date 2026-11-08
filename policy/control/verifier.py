from __future__ import annotations

import copy
import math
from typing import Callable, Optional

PASS = ("read_fleet_state", "require_operator")
WORSE_UNSAFE = 0
WORSE_ISOLATE_S = 10.0


def delta(base: dict, cand: dict) -> dict:
    d = {"unsafe": cand.get("unsafe", 0) - base.get("unsafe", 0)}
    bi, ci = base.get("isolate_s"), cand.get("isolate_s")
    d["isolate_s"] = None if bi is None and ci is None else (math.inf if ci is None else
                                                             -math.inf if bi is None else round(ci - bi, 1))
    return d


def vetoed(d: dict, tol_s: float = WORSE_ISOLATE_S) -> Optional[str]:
    if d["unsafe"] > WORSE_UNSAFE:
        return f"+{d['unsafe']} unsafe event(s)"
    iso = d["isolate_s"]
    if iso is not None and iso > tol_s:
        return "leak never isolated in rollout" if math.isinf(iso) else f"leak isolation +{iso:.0f} s"
    return None


class VerifiedEnvelope:
    def __init__(self, inner, fork: Callable[[], object], horizon_s: float = 600.0, tol_s: float = WORSE_ISOLATE_S):
        self.inner, self.fork, self.horizon_s, self.tol_s = inner, fork, horizon_s, tol_s
        self.vetoes: list[dict] = []
        self.checked = 0

    @property
    def reserved(self):
        return self.inner.reserved

    def check(self, port, name, args):
        return self.inner.check(port, name, args)

    def call(self, port, name: str, args: dict) -> dict:
        args = dict(args or {})
        if name in PASS or self.inner.check(port, name, dict(args))[0]:
            return self.inner.call(port, name, args)
        base, cand = self.fork(), self.fork()
        cand.apply(lambda p: copy.deepcopy(self.inner).call(p, name, dict(args)))
        kb, kc = base.advance(self.horizon_s), cand.advance(self.horizon_s)
        self.checked += 1
        d = delta(kb, kc)
        why = vetoed(d, self.tol_s)
        if why is None:
            return self.inner.call(port, name, args)
        rec = {"call": name, "args": args, "delta": d, "why": why, "horizon_s": self.horizon_s}
        self.vetoes.append(rec)
        port.log("plan_vetoed", args.get("job_id"), None, call=name, args=args, delta=d, why=why,
                 summary=f"plan vetoed by simulation: {why}")
        return {"name": name, "args": args, "ok": False, "refused": f"plan vetoed by simulation: {why}", "note": "",
                "result": None}
