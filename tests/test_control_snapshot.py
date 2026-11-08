import json

import pytest

from policy.control.adapters import TrendTracker, from_file, from_header, from_leak, merge
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot, Verdict


def snap(**kw):
    base = dict(t=10.0, robots=[Robot("go2-1", state="SCRUB", zone="A", force_n=22.0)],
                gas=[GasSensor("G-F3", zone="B", lel=4.0, acoustic_db=40.0)])
    base.update(kw)
    return SensorSnapshot(**base)


def test_roundtrip_dict_json():
    s = snap(verdicts=[Verdict("go2-1", "rust", 0.82, "residue")], storm_level=0.4, images=["abc"],
             jobs=[{"id": "R-1", "kind": "rust", "zone": "A", "status": "active"}])
    d = s.to_dict()
    json.dumps(d)
    back = SensorSnapshot.from_dict(d)
    assert back == s
    assert back.robot("go2-1").force_n == 22.0
    assert back.robot("nope") is None


def test_gas_ppm_from_lel_and_lookup():
    g = GasSensor("G1", zone="B", lel=10.0)
    assert g.ppm == pytest.approx(5000.0)
    s = snap()
    assert s.max_lel() == 4.0
    assert s.zones() == {"A", "B"}


def test_from_dict_ignores_unknown_keys():
    d = snap().to_dict()
    d["robots"][0]["laser"] = 1
    d["wat"] = 2
    assert SensorSnapshot.from_dict(d).robot("go2-1").state == "SCRUB"


def test_from_header_maps_hud_fields():
    h = {"t": 49.0, "state": "BACK_OFF", "force_n": 65.0, "detector_conf": 0.91, "rust_fraction": 0.69,
         "hud": ["BACK_OFF", "force 65 N"],
         "gemini": {"verdict": "rust", "confidence": 0.82, "reason": "residue", "latency_ms": 1200},
         "events": [{"t": 49.0, "from": "SCRUB", "to": "BACK_OFF", "reason": "contact force 65 N > 40 N limit",
                     "evidence": {"person_m": 2.4}}],
         "inspection": {"jpg": "Zm9v", "w": 640, "h": 480}}
    s = from_header(h, "go2-1", zone="A")
    r = s.robot("go2-1")
    assert (r.state, r.force_n, r.rust_conf, r.zone) == ("BACK_OFF", 65.0, 0.91, "A")
    assert r.rust_pct == pytest.approx(69.0)
    assert r.person_m == 2.4
    assert r.caption
    assert s.verdicts == [Verdict("go2-1", "rust", 0.82, "residue")]
    assert s.events[0]["robot"] == "go2-1"
    assert s.images == ["Zm9v"]
    assert "hud" in s.sources


def test_from_header_reads_forward_compatible_fleet_keys():
    h = {"t": 3.0, "robots": [{"id": "go2-2", "state": "PATROL", "battery": 0.7, "caption": "walking deck B",
                               "pose": [1, 2, 0.5], "zone": "B"}],
         "gas": [{"id": "G-F3", "zone": "B", "lel": 12.0, "acoustic_db": 66.0}],
         "plume": {"flow_kg_h": 0.8, "source": "F3"}, "storm_level": 0.6, "wind_deg": 270,
         "tasks": [{"id": "L-F3", "kind": "leak", "zone": "B"}]}
    s = from_header(h)
    assert s.robot("go2-2").battery == 0.7 and s.robot("go2-2").pose == (1.0, 2.0, 0.5)
    assert s.gas[0].lel == 12.0 and s.plume_kg_h == 0.8 and s.plume_source == "F3"
    assert s.storm_level == 0.6 and s.wind_deg == 270
    assert s.jobs[0]["id"] == "L-F3"


def test_merge_overrides_non_none_and_unions():
    a = snap()
    b = SensorSnapshot(t=12.0, robots=[Robot("go2-1", battery=0.5), Robot("go2-2", zone="B")],
                       gas=[GasSensor("G-F3", zone="B", lel=9.0)], storm_level=0.3, sources=["fleet"])
    m = merge(a, b)
    assert m.t == 12.0
    r = m.robot("go2-1")
    assert r.state == "SCRUB" and r.force_n == 22.0 and r.battery == 0.5
    assert m.robot("go2-2") is not None
    assert [g.lel for g in m.gas] == [9.0]
    assert m.storm_level == 0.3


def test_trend_tracker_slopes():
    tr = TrendTracker(window_s=10.0)
    for k in range(6):
        s = SensorSnapshot(t=float(k * 2), gas=[GasSensor("G", zone="B", lel=2.0 + k, acoustic_db=40.0 + 3 * k)])
        tr.update(s)
    g = s.gas[0]
    assert g.lel_trend == pytest.approx(30.0, rel=0.05)
    assert g.db_trend == pytest.approx(90.0, rel=0.05)


def test_from_leak_reads_mock_world():
    from policy.leak.mock import build_rig
    orch, bridge, world, leak_world, *_ = build_rig(leak="loose")
    s = from_leak(bridge, robot_id="go2-1")
    assert s.gas and s.gas[0].zone == "F1"
    assert 0 < s.gas[0].lel < 100
    assert s.gas[0].acoustic_db > 40


def test_from_file_jsonl_last_line(tmp_path):
    p = tmp_path / "telemetry.jsonl"
    p.write_text(json.dumps({"t": 1, "state": "SCRUB", "force_n": 10}) + "\n" +
                 json.dumps({"t": 2, "state": "SCRUB", "force_n": 31.5}) + "\n")
    s = from_file(p, "go2-1")
    assert s.robot("go2-1").force_n == 31.5
    assert from_file(tmp_path / "missing.json") is None
