TONE = {
    "BACK_OFF": "warn", "HOLD": "warn", "GAS_CHECK": "warn", "WAIT": "warn", "BLOCKED": "warn", "PAUSED": "warn",
    "TO_DOCK": "warn", "LEAK_APPROACH": "warn", "REPLAN": "warn",
    "PAUSE_PERSON": "stop", "ESCALATE_TO_HUMAN": "stop", "GAS_RETREAT": "stop", "FAILED": "stop",
    "DONE": "ok", "DOCKED": "ok", "CHARGING": "ok",
}
FORCE_LIMIT = 40.0
GAS_STOP = 0.20


def _pct(x) -> str:
    return f"{100 * float(x):.0f} %"


def _join(*parts) -> str:
    return ", ".join(p for p in parts if p)


def _dash(rest: str) -> str:
    return f" — {rest}" if rest else ""


def _gas(ev) -> str:
    g = ev.get("gas")
    return "" if g is None else f"gas {_pct(g)} LEL"


def _scrub(ev):
    f, r = ev.get("force"), ev.get("rust_fraction")
    return "Scrubbing rust", _dash(_join(None if f is None else f"{float(f):.0f} N", None if r is None else f"{_pct(r)} left"))


def _soak(ev):
    left = ev.get("soak_left")
    return "Soaking remover", _dash(None if left is None else f"{float(left):.0f} s left")


def _verify(ev):
    g = ev.get("gemini") or {}
    if g.get("verdict") in ("clean", "rust"):
        says = "clean" if g["verdict"] == "clean" else "rust still visible"
        return "Verifying", f" — Gemini says {says}, {_pct(g.get('confidence', 0))} sure"
    r = ev.get("rust_fraction")
    return "Verifying", " the clean" if r is None else f" — {_pct(r)} rust left"


def _detect(ev):
    c = ev.get("detector_conf")
    return "Inspecting", " the pipe for rust" if c is None else f" the pipe — detector {_pct(c)} sure"


def _spray(ev):
    n = ev.get("patches") or 0
    return "Spraying", f" rust remover on {n} patches" if n > 1 else " rust remover on the patch"


def _back_off(ev):
    f, lim = ev.get("force"), float(ev.get("force_limit") or FORCE_LIMIT)
    if f is not None and float(f) > lim:
        return "Backing off", f": {float(f):.0f} N over the {lim:.0f} N limit"
    return "Backing off", ": force spike, lifting the brush"


def _person(ev):
    d = ev.get("person_m")
    return "Paused", " — person nearby" if d is None else f" — person {float(d):.1f} m away"


def _gas_lead(rest):
    def f(ev):
        g = ev.get("gas")
        return ("Tracing the leak", f" — {rest}") if g is None else (f"Gas at {_pct(g)} LEL", f" — {rest}")
    return f


def _inspect(ev):
    fl = ev.get("flange")
    return (f"Inspecting flange {fl}" if fl else "Inspecting the flange"), " — reading bolt angles"


def _retorque(ev):
    b = ev.get("bolt")
    return (f"Re-torquing bolt {b}" if b is not None else "Re-torquing the flange"), _dash(_gas(ev))


def _leak_verify(ev):
    return "Checking the leak", _dash(_gas(ev)) or " is fixed"


def _retreat(ev):
    g = ev.get("gas")
    stop = float(ev.get("gas_stop") or GAS_STOP)
    if g is None:
        return "Retreating", " — gas sensor failed, failing safe"
    return "Retreating", f" — gas {_pct(g)} LEL over the {_pct(stop)} limit"


def _escalate(ev):
    g = ev.get("gas")
    if g is not None and float(g) >= float(ev.get("gas_stop") or GAS_STOP):
        return "Calling a human", " — gas over the safe limit"
    return "Calling a human", " — can't finish this safely alone"


def _done(ev):
    job, r = ev.get("job"), ev.get("rust_fraction")
    if job == "gas":
        return "Done", " — leak fixed, gas back to normal"
    if job == "scrub":
        return "Done", f" — {_pct(r)} left, flagged for touch-up" if r is not None and r > 0.01 else " — segment clean"
    return "Done", " — pipe clean and painted"


def _transit(ev):
    eta = ev.get("eta_s")
    return "Walking", " to the job" + (f" — {float(eta):.0f} s away" if eta else "")


def _soc(lead, prefix=" — "):
    def f(ev):
        s = ev.get("soc")
        return lead, "" if s is None else f"{prefix}{_pct(s)}"
    return f


FIXED = {
    "PATROL": ("Patrolling", " the deck — scanning pipework for rust"),
    "SECOND_VIEW": ("Taking a second look", " — moving the camera for a clearer view"),
    "APPROACH": ("Approaching", " the rust patch"),
    "PAINT": ("Painting", " primer over the bare steel"),
    "VERIFY_PAINT": ("Checking the paint", " coat for gaps"),
    "HOLD": ("Holding position", " — camera view lost"),
    "LEAK_SECOND_VIEW": ("Second look", " at the flange from another angle"),
    "REPLAN": ("New rust", " — re-planning the job"),
    "IDLE": ("Idle", " — waiting for a job"),
    "WAIT": ("Waiting", " for the walkway to clear"),
    "BLOCKED": ("Blocked", " — finding another route"),
    "PAUSED": ("Paused", " — person on the walkway"),
    "WORK": ("Working", " on the job"),
    "SETUP": ("Setting up", " at the work point"),
    "DOCK_QUEUE": ("Queued", " for a charging pad"),
    "DOCKED": ("Docked", ""),
    "FAILED": ("Robot down", " — its job goes to another unit"),
}
DYNAMIC = {
    "DETECT": _detect, "SPRAY": _spray, "SOAK": _soak, "SCRUB": _scrub, "VERIFY": _verify, "BACK_OFF": _back_off,
    "PAUSE_PERSON": _person, "ESCALATE_TO_HUMAN": _escalate,
    "GAS_CHECK": _gas_lead("pausing work to find the leak"), "LEAK_APPROACH": _gas_lead("tracing the source upwind"),
    "LEAK_INSPECT": _inspect, "RETORQUE": _retorque, "LEAK_VERIFY": _leak_verify, "GAS_RETREAT": _retreat,
    "DONE": _done, "TRANSIT": _transit, "TO_DOCK": _soc("Heading to dock", " — battery "), "CHARGING": _soc("Charging"),
}


def caption(state: str, ev: dict | None = None) -> dict:
    ev = ev or {}
    if state in DYNAMIC:
        lead, rest = DYNAMIC[state](ev)
    else:
        lead, rest = FIXED.get(state, (state.replace("_", " ").capitalize(), ""))
    return {"state": state, "lead": lead, "rest": rest, "text": lead + rest, "tone": TONE.get(state, "act")}


def robot_entry(rid: str, name: str, state: str, ev: dict | None, world_pos, occluded: bool = False) -> dict:
    pos = None if world_pos is None else [round(float(c), 3) for c in world_pos]
    out = {"id": rid, "name": name, "state": state, "caption": caption(state, ev), "world_pos": pos}
    if occluded:
        out["occluded"] = True
    return out


def fleet_robots(snapshot: dict) -> list[dict]:
    out = []
    for r in snapshot.get("robots", []):
        state = r.get("job_state") or r.get("state") or "IDLE"
        p = r.get("pose") or {}
        pos = [p["x"], p["y"], p["z"]] if {"x", "y", "z"} <= set(p) else r.get("world_pos")
        ev = {**(r.get("evidence") or {}), "soc": (r.get("battery") or {}).get("soc"), "eta_s": r.get("eta_s"),
              "job": r.get("kind") or (r.get("evidence") or {}).get("job")}
        out.append(robot_entry(r["id"], r.get("name") or r["id"].replace("-", " ").title(), state, ev, pos,
                               bool(r.get("occluded"))))
    return out
