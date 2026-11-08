from __future__ import annotations

import re
from typing import Optional

DECKS = ((20.0, "lower deck"), (31.25, "middle deck"), (42.5, "top deck"))
DECK_WORD = {"low": "lower deck", "lower": "lower deck", "mid": "middle deck", "middle": "middle deck", "top": "top deck"}
VERB = {"pick_can": "pick up", "weld": "weld repair", "weld_repair": "weld repair", "rust": "rust repair",
        "rust_repair": "rust repair", "rust_inspect": "rust check", "paint": "paint", "touch_up": "touch-up",
        "leak_check": "leak check", "flange_check": "flange check", "flange_retorque": "flange re-torque",
        "retorque": "re-torque", "inspect": "gauge reading", "inspection": "inspection round",
        "gauge_read": "gauge reading", "survey": "survey", "gas": "gas survey", "gas_survey": "gas survey",
        "gas_leak": "gas leak", "crack": "crack check", "crack_check": "crack check", "valve": "valve job",
        "valve_isolate": "valve isolation", "valve_check": "valve check", "evacuate": "evacuation",
        "drop_can": "can dropped", "drop_obstacle": "obstacle dropped", "close_walkway": "walkway closed",
        "close_stairs": "stairs closed", "set_weather": "weather change", "battery_fault": "battery fault",
        "person_at": "worker on deck", "weld_job": "weld job", "clear": "walkway cleared"}
STATE = {"IDLE": "idle", "TRANSIT": "walking", "MOVING": "walking", "CHARGING": "charging",
         "TO_DOCK": "walking to the dock", "DOCK_QUEUE": "waiting for the dock", "DOCKED": "docked",
         "PAUSED": "paused, person nearby", "PAUSED_PERSON": "paused, person nearby", "WAIT": "waiting",
         "WORKING": "working", "PATROL": "on patrol", "SCRUB": "scrubbing rust", "SPRAY": "spraying rust remover",
         "SOAK": "letting the spray soak", "PAINT": "painting", "VERIFY": "checking its work",
         "VERIFY_PAINT": "checking the paint", "GRIND": "grinding", "WELD": "welding", "INSPECT_WELD": "inspecting the weld",
         "RETORQUE": "re-torquing a flange", "LEAK_APPROACH": "closing in on the leak",
         "LEAK_INSPECT": "inspecting for the leak", "LEAK_VERIFY": "confirming the leak has stopped",
         "GAS_CHECK": "checking gas", "VALVE_CLOSE": "closing a valve", "PICK": "picking up", "DETECT": "looking for rust",
         "APPROACH": "approaching", "BACK_OFF": "backing off", "SECOND_VIEW": "taking a second look"}
STATUS = {"QUEUED": "queued", "WAITING_APPROVAL": "waiting for operator approval", "ASSIGNED": "assigned",
          "RUNNING": "in progress", "BLOCKED": "blocked by a hazard", "HANDED_TO_HUMAN": "handed to a human"}
TOOL = {"require_operator": "keep approvals with the operator", "queue_job": "queue a job", "cancel_job": "cancel a job",
        "approve_request": "approve the request", "reprioritise": "re-prioritise"}
META = ("understand", "dive", "diving", "delv", "analy", "clarif", "examin", "consider", "focus", "review", "think",
        "look", "assess", "evaluat", "explor", "try", "process", "interpret", "pars", "formulat", "determin", "figur",
        "ponder", "re-evaluat", "reassess", "investigat", "stud", "zero", "map", "weigh", "digest", "read", "check",
        "note", "noting", "realiz", "realis", "see", "notic", "recogniz", "start", "begin", "began", "work", "refin", "recall",
        "remember", "confirm", "double", "verif", "identif", "gather", "summar", "unpack", "break", "craft", "prepar", "observ", "test", "interrogat", "question", "debat",
        "draft", "outlin", "comb", "scan", "tackl", "piec", "connect", "reconcil", "trac", "make sense", "wonder")
DROP = re.compile(r"diving deeper|deeper dive|in this context|\bthe (?:instructions|prompt|system prompt|guidelines|"
                  r"established rules|defined constraints|constraints|rules)\b|function (?:call|result)|tool call|\bjson\b|"
                  r"read_fleet_state|fleet_state|\boutput at\b|\bschema\b|finding structure|\bis an? [\w\" ]*job\b|"
                  r"requested by the coordinator|^this (?:should|will|ensures|approach|plan)\b|"
                  r"^(?:let'?s|let me|here'?s|time to)\b|\b(?:history|the system(?:'s)?|simulator|benchmark|the code|"
                  r"sentence|phrase|wording|intriguing|interesting|curious|puzzling|hard limits?)\b|\?$", re.I)
LEAD = re.compile(r"^(?:okay|ok|alright|right|so|now|well|first|firstly|next|then|finally|specifically|essentially|"
                  r"basically|additionally|also|therefore|thus|hence|it looks like|it seems(?: that)?|"
                  r"(?:given|based on|according to|per|regarding|looking at) (?:the )?(?:instructions|rules|prompt|"
                  r"constraints|guidelines|fleet state|state|findings|situation)[^,:]*|"
                  r"my (?:actions|plan|next steps?|approach) (?:are|is) (?:clear|as follows|simple|now))\s*[,:]?\s+", re.I)
SELF = re.compile(r"^(?:I'm|I am|I've|I have|I'll|I will|I'd|I need to|I must|I should|I want to|I can|I)\s+"
                  r"((?:now|just|also|then|first|currently|still|really|going to|starting to)\s+)*", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def side(x: float) -> str:
    return "east end" if x > 14.0 else "west end" if x < -8.0 else "centre"


def deck(z: float) -> str:
    return min(DECKS, key=lambda d: abs(d[0] - float(z)))[1]


def place(xyz) -> str:
    try:
        x, _, z = (float(v) for v in list(xyz)[:3])
    except (TypeError, ValueError):
        return ""
    return f"{deck(z)}, {side(x)}"


def verb(kind) -> str:
    k = str(kind or "")
    return VERB.get(k, k.replace("_", " "))


def name(x) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    if ":" in s:
        k, _, poi = s.partition(":")
        return f"{verb(k)}{' ' if k == 'pick_can' else ' at '}{name(poi.split('#')[0])}"
    s = s.split("#")[0]
    for pat, fn in ((r"gauge_?0*(\d+)", lambda m: f"pressure gauge {m[1]}"),
                    (r"valve_?0*(\d+)", lambda m: f"valve {m[1]}"),
                    (r"(?:GD-)?F(\d+)", lambda m: f"flange F{m[1]}"),
                    (r"w(\d+)", lambda m: f"walkway W{m[1]}"),
                    (r"deck-(\d+)", lambda m: f"deck {m[1]}"),
                    (r"ca\d+", lambda m: "the can"), (r"cr\d+", lambda m: "the crate"),
                    (r"rust_(low|lower|mid|middle|top)_(\w+)", lambda m: f"rust patch, {DECK_WORD[m[1]]} {m[2].replace('_', ' ')}"),
                    (r"weld_(\w+)", lambda m: " ".join(reversed(m[1].split("_"))) + " weld")):
        m = re.fullmatch(pat, s)
        if m:
            return fn(m)
    return s.replace("_", " ")


def pct(v) -> str:
    try:
        return f"{round(float(v) * 100)}%"
    except (TypeError, ValueError):
        return "?"


def lel(v) -> str:
    p = float(v or 0.0) * 100.0
    return f"{p:.0f}% LEL" if p >= 10 or p == 0 else f"{p:.1f}% LEL"


def _num(m) -> str:
    v = float(m[0])
    if 0 < v < 1:
        return pct(v)
    r = round(v, 1)
    return str(int(r)) if r == int(r) else str(r)


def plain(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]",
               lambda m: place((m[1], m[2], m[3])), s)
    s = re.sub(r"\s*\(x\d+\)", "", s)
    s = re.sub(r"\b(?:call(?:ing|ed)?|use|using|invoke|invoking)\s+(?:the\s+)?(" + "|".join(TOOL) + r")(?:\s+(?:tool|function))?",
               lambda m: TOOL[m[1]], s, flags=re.I)
    s = re.sub(r"\b(" + "|".join(TOOL) + r")\b", lambda m: TOOL[m[1]], s)
    s = re.sub(r"\b([a-z_]+):([A-Za-z0-9_]+)(?:#\d+)?", lambda m: name(m[0]), s)
    s = re.sub(r"\b(can|crate|drum|puddle) (?:ca|cr|dr|pu)\d+\b", r"\1", s, flags=re.I)
    s = re.sub(r"\b(?:gauge|valve)_\d+(?:#\d+)?\b|\bca\d+\b|\bcr\d+\b|\bw\d+\b|\brust_[a-z]+_[a-z_]+\b|\bweld_[a-z_]+\b",
               lambda m: name(m[0]), s)
    s = re.sub(r"\b(" + "|".join(sorted(VERB, key=len, reverse=True)) + r")\b(?= job| task)",
               lambda m: verb(m[1]), s)
    s = re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: VERB.get(m[0], m[0].replace("_", " ")), s)
    s = re.sub(r"\b[A-Z]{3,5}-\d{3}\b|\b[a-z_]+-\d{3}\b", "the job", s)
    s = re.sub(r",?\s*\b(?:at |as of )?(?:time|t\s*=)\s*\d+(?:\.\d+)?\s*s?\b", "", s)
    s = re.sub(r"(?<![\w.])\d+\.\d{3,}(?![\w.])", _num, s)
    s = re.sub(r"(?<![\w.])(\d+)\.0(?![\w.])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _sentence(s: str) -> Optional[str]:
    s = s.strip()
    for _ in range(3):
        t = LEAD.sub("", s)
        if t == s:
            break
        s = t
    if not s or DROP.search(s):
        return None
    m = SELF.match(s)
    if m:
        rest = s[m.end():]
        word = rest.split(" ", 1)[0].lower()
        if not rest or any(word.startswith(p) for p in META):
            return None
        s = rest
    first = s.split(" ", 1)[0].lower()
    if first.endswith("ing") and any(first.startswith(p) for p in META):
        return None
    if re.match(r"(?i)my (?:focus|analysis|understanding|attention|goal|task|current|immediate|next|priority|thinking)", s):
        return None
    return _cap(s)


def tidy(text: str) -> Optional[str]:
    s = " ".join(str(text or "").replace("**", "").replace("`", "").split())
    if not s:
        return None
    if not re.search(r"[.!?:;\"')]$", s) and len(s.split()) <= 8:
        return None
    kept = [_cap(plain(k)) for k in (_sentence(x) for x in SENT.split(s)) if k]
    return " ".join(k for k in kept if k) or None


def robot_line(r: dict, jobs: dict) -> str:
    rid = str(r.get("id", "?"))
    d = "".join(c for c in rid if c.isdigit())
    who = f"Unit {d} ({rid})" if d else rid
    job = r.get("job")
    j = jobs.get(job) or {}
    st = str(r.get("state") or "")
    words = STATE.get(st, st.lower().replace("_", " "))
    walking = st in ("TRANSIT", "MOVING") and bool(j.get("target") or job)
    if walking:
        words = f"walking to {name(j.get('target') or job)}"
    doing = (verb(j.get("kind")) if walking else f"{verb(j.get('kind'))} · {name(j.get('target'))}") if j else \
        (name(job) if job else "no job")
    bits = [who, doing, pct(r["battery"]) + " battery" if r.get("battery") is not None else "", words]
    pos = r.get("position")
    if pos is not None and len(pos) >= 3:
        bits.append(place(pos))
    return " · ".join(b for b in bits if b)


def job_line(j: dict) -> str:
    bits = [verb(j.get("kind")), name(j.get("target")), f"P{j.get('priority')}",
            STATUS.get(j.get("status"), str(j.get("status") or "").lower().replace("_", " "))]
    if j.get("robot"):
        bits.append(str(j["robot"]))
    if j.get("level") == "SUPV" and j.get("status") == "QUEUED":
        bits.append("needs approval")
    p = j.get("progress")
    if p:
        bits.append(f"{pct(p)} done")
    return " · ".join(b for b in bits if b)


def fleet_lines(state: dict) -> list[str]:
    jobs = {j.get("id"): j for j in state.get("jobs") or []}
    out = ["FLEET:"] + [f"- {robot_line(r, jobs)}" for r in state.get("robots") or []]
    active = [j for j in jobs.values() if j.get("status") not in ("DONE", "CANCELLED", "REJECTED")]
    lines = [f"- {job_line(j)}" for j in sorted(active, key=lambda j: j.get("priority") or 5)[:12]]
    out += ["JOBS (most urgent first):"] + (lines or ["- none"])
    esd = state.get("esd") or {}
    out.append(f"ESD: active in {name(esd.get('zone')) or 'the platform'}" if esd.get("active") else "ESD: not active")
    paused = [str(k) for k in (state.get("person_pauses") or {})]
    if paused:
        out.append("PEOPLE: " + ", ".join(paused) + " paused, worker within reach")
    crew = state.get("crew") or []
    if crew:
        out.append(f"CREW ON DECK: {len(crew)}")
    return out


def finding_line(f: dict) -> str:
    kind = str(f.get("kind") or "finding")
    d = f.get("detail") or {}
    where = name(f.get("target") or f.get("zone") or "")
    bits = [verb(kind)]
    if where:
        bits.append(f"at {where}")
    if isinstance(d, dict):
        k = d.get("kind")
        if k and k != kind:
            bits.append(str(k))
        for key in ("robot", "battery_fault"):
            if d.get(key):
                bits.append(str(d[key]))
        if d.get("lel") is not None:
            bits.append(lel(d["lel"]))
    if f.get("summary"):
        bits.append(plain(f["summary"]))
    return " · ".join(bits)
