from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy.control.adapters import from_file, from_header, merge
from policy.control.centre import from_env
from policy.control.mock import FleetScenario

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "runs/hud/mock_session.json"


def mock_snapshot(pt: float, session):
    scenario = FleetScenario()
    snaps = session["snapshots"]
    s = min(snaps, key=lambda x: abs(x["pt"] - pt))
    return merge(from_header(s, "go2-1", zone="A"), scenario.at(pt))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the fleet control centre (gemini-4-argon) about a snapshot.")
    ap.add_argument("--mode", choices=["auto", "live", "replay", "edge"], default="auto")
    ap.add_argument("--at", type=float, nargs="*", default=[26.0], help="mock playback times to evaluate")
    ap.add_argument("--snapshot", default="", help="HUD header JSON/JSONL file instead of the mock fleet")
    ap.add_argument("--log", default=str(REPO / "runs/control/calls.jsonl"))
    ap.add_argument("--full", action="store_true", help="print the whole control header")
    a = ap.parse_args(argv)
    centre = from_env(a.mode, a.log, sync=True, interval_s=0.0, alarm_gap_s=0.0)
    print(f"backend: {centre.backend.name if centre.backend else 'none (edge rules only)'}")
    if a.snapshot:
        snaps = [(0.0, from_file(a.snapshot))]
    else:
        if not SESSION.exists():
            from rig.hud import record_mock
            record_mock(SESSION, str(SESSION.parent / "decisions.jsonl"))
        session = json.loads(SESSION.read_text())
        snaps = [(pt, mock_snapshot(pt, session)) for pt in a.at]
    for k, (pt, snap) in enumerate(snaps):
        h = centre.tick(snap, now=float(k) * 10.0)
        if a.full:
            print(json.dumps(h, indent=1))
            continue
        print(f"\n== pt={pt:.1f}  mode={h['mode']} status={h['status']} latency={h['latency_ms']} ms "
              f"tokens={h['tokens']} cost=${h['cost_usd']} today=${h['cost_today_usd']}")
        if h["banner"]:
            print("   ", h["banner"], h["error"] or "")
        for line in h["situation"]:
            print("   |", line)
        for x in h["actions"]:
            print(f"   {x['rank']}. [{x['source']}] {x['label']} ({x['confidence']:.2f}) - {x['rationale']}")
        for v in h["vetoed"]:
            print(f"   VETO {v['label']}: {v['why']}")
        for x in h["anomalies"]:
            print(f"   ! {x['severity']} {x['kind']} @ {x['zone']} ({x['confidence']:.2f}): {x['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
