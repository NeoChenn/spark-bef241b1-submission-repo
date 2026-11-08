from __future__ import annotations

import hashlib
import json

from policy.control.safety import FORCE_LIMIT_N, LEL_CLEAR, LEL_EVACUATE, PERSON_MIN_M
from policy.control.schema import CLOUD_KINDS, RESPONSE_SCHEMA
from policy.control.snapshot import SensorSnapshot

MAX_EVENTS = 12
MAX_CANDIDATES = 4
MAX_OUTPUT_TOKENS = 16384

SYSTEM = (
    "You are the fleet control centre for inspection-and-repair quadruped robots on an offshore oil platform. "
    "Every few seconds you get one fused sensor snapshot: per-robot state, caption, pose, battery, rust detector "
    "confidence and rust %, brush contact force, gas %LEL and ppm per fixed sensor with trends (%LEL/min), acoustic "
    "(ultrasonic) dB with trend, plume flow-rate estimate, Gemini Robotics-ER verifier verdicts, storm level 0-1, "
    "wind, the job list and recent decision events; sometimes the latest inspection image.\n"
    "Your role is ADVISORY. A deterministic edge safety layer on the robots always wins and runs without you: "
    f"contact force > {FORCE_LIMIT_N:.0f} N -> back off; gas >= {LEL_EVACUATE:.0f}% LEL -> retreat upwind and evacuate "
    f"the zone, latched until < {LEL_CLEAR:.0f}% LEL; person < {PERSON_MIN_M:.0f} m -> pause. Never recommend anything that undoes an active edge rule; "
    "any such recommendation is discarded.\n"
    "Return: (1) situation: plain English, at most 3 short lines for an operator HUD; (2) actions: ranked best "
    "first, only kinds dispatch | second_view | hold | resume | escalate | evacuate, only robot/job/zone ids that "
    "appear in the snapshot, each with confidence 0-1 and a one-sentence rationale citing the evidence; "
    "(3) anomalies: patterns that no single sensor shows, fused across sensors and robots, e.g. gas %LEL and "
    "acoustic dB rising together at one flange -> probable leak; force spikes with rust residual -> surface damage "
    "or weld bead; on-board detector and verifier disagreeing; storm level rising during exposed work. "
    "Prefer the conservative action when evidence is mixed. Empty lists are fine when all is nominal.\n"
    "Fixed rig sensors (gas heads, thermal, acoustic, pressure gauges) are checked by an on-rig health monitor; "
    "health lists each faulty sensor (DEAD, STUCK, DRIFT, OUT_OF_RANGE, NOISY) with evidence numbers and the robots "
    "that could verify it, ranked by path time over the rig graph (stairs, corridors), with the onboard sensor they "
    "would use and whether their battery covers the round trip (ok). To verify a suspect sensor, return a dispatch "
    "with purpose verify_sensor, target = sensor id and a robot whose candidate entry is ok; prefer the nearest. "
    "Skip faults that already have a job. Hazard first: a faulty gas head marked hazard sits beside live gas; treat "
    "its zone as real gas, never as just a broken sensor."
)


def build(snap: SensorSnapshot, edge=(), max_images: int = 1) -> dict:
    d = snap.to_dict(images=False)
    d["events"] = d["events"][-MAX_EVENTS:]
    d["health"] = [{**{k: v for k, v in h.items() if k != "candidates"},
                    "candidates": h["candidates"][:MAX_CANDIDATES]} for h in d["health"] if h["status"] != "OK"]
    ids = {"robots": [r.id for r in snap.robots], "jobs": [j.get("id") for j in snap.jobs],
           "zones": sorted(snap.zones()), "suspect_sensors": [h["id"] for h in d["health"]]}
    text = (
        "Fleet snapshot (JSON):\n" + json.dumps(d, separators=(",", ":"), default=str) +
        "\nValid ids: " + json.dumps(ids, separators=(",", ":")) +
        "\nAllowed action kinds: " + ", ".join(CLOUD_KINDS) +
        "\nEdge rules active now: " + json.dumps([a.to_dict() for a in edge], separators=(",", ":")) +
        ("\nThe attached image is the latest inspection camera frame." if snap.images and max_images else "")
    )
    return {"system": SYSTEM, "text": text, "images": list(snap.images[-max_images:]) if max_images else []}


def prompt_hash(p: dict) -> str:
    h = hashlib.sha256((p["system"] + "\0" + p["text"]).encode("utf-8"))
    for img in p["images"]:
        h.update(img.encode("ascii"))
    return h.hexdigest()[:12]


def request_body(p: dict, model: str, thinking: str = "low", max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    parts = [{"type": "text", "text": p["text"]}]
    parts += [{"type": "image", "data": img, "mime_type": "image/jpeg"} for img in p["images"]]
    return {
        "model": model,
        "store": False,
        "system_instruction": p["system"],
        "input": parts,
        "generation_config": {"thinking_level": thinking, "max_output_tokens": max_output_tokens},
        "response_format": {"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA},
    }
