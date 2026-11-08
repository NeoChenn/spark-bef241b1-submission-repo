from __future__ import annotations

from rig.captions import robot_entry

MISSION = ("WALK", "STANCE", "DETECT", "DEPLOY", "SPRAY", "SOAK", "SCRUB", "VERIFY", "REPOSITION", "PAINT",
           "VERIFY_PAINT", "DONE")
ORCH = {"PATROL": "WALK", "APPROACH": "DEPLOY", "BACK_OFF": "SCRUB", "SCRUB_LIFT": "SCRUB",
        "PAUSE_PERSON": "PAUSE", "HOLD": "HOLD"}
CORE = ("WALK", "DEPLOY", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT", "VERIFY_PAINT", "DONE")


def mission_phase(orch_state: str, sub: str | None) -> str:
    if sub:
        return sub
    return ORCH.get(orch_state, orch_state)


class MissionTimeline:
    def __init__(self, robot_id: str = "Q1", name: str = "Unit 1 · rust"):
        self.rid, self.name = robot_id, name
        self.spans: list[dict] = []

    def mark(self, t: float, phase: str) -> bool:
        if self.spans and self.spans[-1]["phase"] == phase:
            return False
        if self.spans:
            self.spans[-1]["t1"] = float(t)
        self.spans.append({"phase": phase, "t0": float(t), "t1": None})
        return True

    def close(self, t: float) -> None:
        if self.spans and self.spans[-1]["t1"] is None:
            self.spans[-1]["t1"] = float(t)

    @property
    def phase(self) -> str | None:
        return self.spans[-1]["phase"] if self.spans else None

    def durations(self) -> dict:
        out: dict = {}
        for s in self.spans:
            t1 = s["t1"] if s["t1"] is not None else s["t0"]
            out[s["phase"]] = round(out.get(s["phase"], 0.0) + t1 - s["t0"], 3)
        return out

    def phase_at(self, t: float) -> str | None:
        cur = None
        for s in self.spans:
            if s["t0"] <= t:
                cur = s["phase"]
        return cur

    def in_order(self) -> bool:
        firsts = []
        for s in self.spans:
            if s["phase"] in CORE and s["phase"] not in firsts:
                firsts.append(s["phase"])
        return firsts == [p for p in CORE if p in firsts] and firsts[-1:] == ["DONE"] and len(firsts) == len(CORE)

    def hud_record(self, t: float, state: str, force_n: float, world_pos, ev: dict | None = None) -> dict:
        robot = robot_entry(self.rid, self.name, state, {**(ev or {}), "force_n": round(float(force_n), 1)}, world_pos)
        return {"t": round(float(t), 3), "state": state, "phase": self.phase_at(t) or state,
                "force_n": round(float(force_n), 2), "robots": [robot]}
