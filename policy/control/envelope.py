from __future__ import annotations

from typing import Optional

from policy.control.safety import FORCE_LIMIT_N, PERSON_MIN_M

AGENT = "gemini"
FUNCTIONS = ("queue_job", "cancel_job", "approve_request", "reprioritise", "read_fleet_state", "require_operator")
ROBOT_FUNCTIONS = FUNCTIONS + ("hold_robot", "resume_robot")
MOTOR_WORDS = ("motor", "joint", "torque", "velocity", "cmd_vel", "twist", "wheel", "servo", "pwm", "actuator")
MOTOR_REFUSAL = "Argon never commands motors; it can only queue, cancel, reprioritise, hold or resume jobs"
LEVELS = ("AUTO", "SUPV", "ASSIST", "HUMAN")
SUPV_KINDS = {"paint", "rust_repair", "retorque", "valve_isolate", "touch_up", "spray"}
SAFETY_KINDS = {"valve_isolate", "esd_hold", "evacuate", "gas_retreat", "gas_isolate"}
RESUME_KINDS = {"resume", "unpause", "continue"}


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _why(args) -> str:
    w = args.get("why")
    return w.strip() if isinstance(w, str) else ""


class Envelope:
    def __init__(self, agent: str = AGENT, functions=FUNCTIONS):
        self.agent, self.functions = agent, tuple(functions)
        self.reserved: set[str] = set()

    def _job_rules(self, kind: str, target: str, params: dict, state: dict) -> Optional[str]:
        k = kind.lower()
        keys = [str(x).lower() for x in params]
        if any(w in k or any(w in x for x in keys) for w in MOTOR_WORDS):
            return MOTOR_REFUSAL
        if "esd" in k or any("esd" in x for x in keys):
            return "an ESD can only be reset by a person on site"
        if "force" in k:
            return f"the {FORCE_LIMIT_N:.0f} N force limit is fixed on the edge"
        for key, v in params.items():
            n = _num(v)
            if "force" in str(key).lower() and (n is None or n > FORCE_LIMIT_N):
                return f"force setting {v!r} exceeds the {FORCE_LIMIT_N:.0f} N limit"
            if ("person" in str(key).lower() or "radius" in str(key).lower()) and (n is None or n < PERSON_MIN_M):
                return f"the {PERSON_MIN_M:.0f} m person pause cannot be reduced"
        if "pause" in k or "person" in k:
            return f"the {PERSON_MIN_M:.0f} m person pause can only clear itself"
        if k in RESUME_KINDS and target in (state.get("person_pauses") or {}):
            return f"{target} is paused for a person within {PERSON_MIN_M:.0f} m"
        return None

    def check(self, port, name: str, args: dict) -> tuple[Optional[str], dict, str]:
        if any(w in name.lower() for w in MOTOR_WORDS):
            return f"{name!r} is not an allowed function: {MOTOR_REFUSAL}", args, ""
        if name not in self.functions:
            return f"{name!r} is not an allowed function", args, ""
        if name == "read_fleet_state":
            return None, args, ""
        if not _why(args):
            return "every call needs a one-line why", args, ""
        state = port.fleet_state()
        if name == "require_operator":
            return (None, args, "") if isinstance(args.get("kind"), str) and args["kind"] else ("kind missing", args, "")
        if name in ("hold_robot", "resume_robot"):
            rid = str(args.get("robot") or "")
            if rid not in {r["id"] for r in state.get("robots", [])}:
                return f"unknown robot {rid!r}", args, ""
            if name == "resume_robot" and rid in (state.get("person_pauses") or {}):
                return f"{rid} is paused for a person within {PERSON_MIN_M:.0f} m; only the person moving away clears it", args, ""
            esd = state.get("esd") or {}
            if name == "resume_robot" and esd.get("active"):
                return "an ESD is active; only a person on site can reset it", args, ""
            return None, args, ""
        if name == "queue_job":
            kind, target = str(args.get("kind") or ""), str(args.get("target") or "")
            params = args.get("params") or {}
            if not kind or not target or not isinstance(params, dict):
                return "queue_job needs kind, target and object params", args, ""
            level = str(args.get("level") or "AUTO").upper()
            if level not in LEVELS:
                return f"level {level!r} not in {LEVELS}", args, ""
            p = args.get("priority", 3)
            if not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 5:
                return f"priority {p!r} must be an integer 1..5", args, ""
            bad = self._job_rules(kind, target, params, state)
            if bad:
                return bad, args, ""
            note = ""
            if level == "AUTO" and (kind in SUPV_KINDS or kind in self.reserved):
                level, note = "SUPV", f"level raised to SUPV: {kind} needs operator approval"
            return None, {**args, "level": level, "params": params}, note
        job = port.job(str(args.get("job_id") or ""))
        if job is None:
            return f"unknown job {args.get('job_id')!r}", args, ""
        k = job["kind"].lower()
        if "esd" in k:
            return "an ESD can only be reset by a person on site", args, ""
        safety = k in SAFETY_KINDS or job["priority"] == 1
        if name == "cancel_job" and safety:
            return f"{job['id']} is a safety job; only the operator can cancel it", args, ""
        if name == "reprioritise":
            p = args.get("priority")
            if not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 5:
                return f"priority {p!r} must be an integer 1..5", args, ""
            if safety and p > job["priority"]:
                return f"{job['id']} is a safety job; it cannot be pushed down", args, ""
        if name == "approve_request":
            if job.get("requested_by") == self.agent:
                return f"{job['id']} was requested by the agent; it cannot approve its own SUPV job", args, ""
            if job["kind"] in self.reserved:
                return f"the operator reserved approval of {job['kind']} jobs", args, ""
        return None, args, ""

    def call(self, port, name: str, args: dict) -> dict:
        args = dict(args or {})
        refused, args, note = self.check(port, name, args)
        out = {"name": name, "args": args, "ok": refused is None, "refused": refused, "note": note, "result": None}
        if refused:
            port.log("agent_refused", args.get("job_id"), None, call=name, args=args, why=refused)
            return out
        why = _why(args)
        if name == "read_fleet_state":
            out["result"] = port.fleet_state()
        elif name == "require_operator":
            self.reserved.add(args["kind"])
            port.log("operator_reserved", None, None, kind=args["kind"], by=self.agent, why=why)
            out["result"] = {"reserved": sorted(self.reserved)}
        elif name == "queue_job":
            jid = port.queue(args["kind"], args["target"], args["level"], args.get("priority", 3), args["params"],
                             by=self.agent, why=why)
            out["result"] = {"job_id": jid, "status": port.job(jid)["status"]}
        elif name in ("hold_robot", "resume_robot"):
            fn = port.hold if name == "hold_robot" else port.resume
            ok = fn(args["robot"], self.agent, why)
            out["ok"], out["result"] = ok, {"robot": args["robot"], "done": ok}
            if not ok:
                out["refused"] = "coordinator rejected the call (robot not in a state that allows it)"
        elif name == "approve_request":
            port.log("approval_requested", args["job_id"], None, by=self.agent, why=why)
            out["result"] = {"job_id": args["job_id"], "status": port.job(args["job_id"])["status"]}
            note = out["note"] = (note + " " if note else "") + "sent to the operator inbox; only a human approves"
        else:
            fn = {"cancel_job": lambda: port.cancel(args["job_id"], self.agent, why),
                  "approve_request": lambda: port.approve_request(args["job_id"], self.agent, why),
                  "reprioritise": lambda: port.reprioritise(args["job_id"], args["priority"], self.agent, why)}[name]
            ok = fn()
            out["ok"], out["result"] = ok, {"job_id": args["job_id"], "done": ok}
            if not ok:
                out["refused"] = "coordinator rejected the call (job not in a state that allows it)"
        if note:
            port.log("agent_note", out["result"].get("job_id") if out["result"] else None, None, note=note)
        return out
