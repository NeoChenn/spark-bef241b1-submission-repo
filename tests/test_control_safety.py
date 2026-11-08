import json

import pytest

from policy.control.safety import FORCE_LIMIT_N, LEL_EVACUATE, PERSON_MIN_M, arbitrate, edge_rules
from policy.control.schema import CLOUD_KINDS, RESPONSE_SCHEMA, Action, SchemaError, validate
from policy.control.snapshot import GasSensor, Robot, SensorSnapshot


def fleet(force=20.0, lel=3.0, person=None):
    return SensorSnapshot(
        t=5.0,
        robots=[Robot("go2-1", state="SCRUB", zone="A", force_n=force, person_m=person),
                Robot("go2-2", state="PATROL", zone="B"), Robot("go2-3", state="DOCKED", zone="dock")],
        gas=[GasSensor("G-F3", zone="B", lel=lel, acoustic_db=40.0)],
        jobs=[{"id": "R-1", "kind": "rust", "zone": "A"}, {"id": "L-F3", "kind": "leak", "zone": "B"}])


def reply(*actions, situation="All nominal.", anomalies=()):
    return {"situation": situation, "actions": list(actions), "anomalies": list(anomalies)}


def act(kind, robot="go2-2", target="L-F3", conf=0.8, why="because"):
    return {"kind": kind, "robot": robot, "target": target, "confidence": conf, "rationale": why}


def test_constants():
    assert (FORCE_LIMIT_N, LEL_EVACUATE, PERSON_MIN_M) == (40.0, 50.0, 3.0)


def test_no_edge_rules_when_nominal():
    assert edge_rules(fleet()) == []


def test_force_rule_backs_off():
    rules = edge_rules(fleet(force=40.5))
    assert [(a.kind, a.robot, a.source, a.confidence) for a in rules] == [("back_off", "go2-1", "edge", 1.0)]
    assert edge_rules(fleet(force=40.0)) == []


def test_gas_rule_retreats_zone_and_evacuates():
    rules = edge_rules(fleet(lel=50.0))
    kinds = {(a.kind, a.robot, a.target) for a in rules}
    assert ("retreat", "go2-2", "upwind") in kinds
    assert ("evacuate", "", "B") in kinds
    assert not any(a.robot == "go2-1" and a.kind == "retreat" for a in rules)


def test_person_rule_pauses():
    rules = edge_rules(fleet(person=2.9))
    assert [(a.kind, a.robot) for a in rules] == [("pause", "go2-1")]
    assert edge_rules(fleet(person=3.0)) == []


def test_edge_beats_cloud_resume_and_dispatch():
    s = fleet(force=65.0, lel=55.0)
    plan = validate(reply(act("resume", "go2-1", ""), act("dispatch", "go2-2", "L-F3"),
                          act("dispatch", "go2-3", "B"), act("escalate", "", "", 0.6)), s)
    final, vetoed = arbitrate(edge_rules(s), plan.actions, s)
    assert final[0].source == "edge"
    kept = [(a.kind, a.robot) for a in final if a.source == "cloud"]
    assert kept == [("escalate", "")]
    assert {(v["kind"], v["robot"]) for v in vetoed} == {("resume", "go2-1"), ("dispatch", "go2-2"), ("dispatch", "go2-3")}
    assert all(v["why"] for v in vetoed)


def test_cloud_can_make_things_safer():
    s = fleet(person=2.0)
    plan = validate(reply(act("hold", "go2-1", ""), act("evacuate", "", "A", 0.7)), s)
    final, vetoed = arbitrate(edge_rules(s), plan.actions, s)
    assert vetoed == []
    assert [a.kind for a in final] == ["pause", "hold", "evacuate"]


def test_duplicate_cloud_evacuate_merges_into_edge():
    s = fleet(lel=60.0)
    plan = validate(reply(act("evacuate", "", "B", 0.9)), s)
    final, vetoed = arbitrate(edge_rules(s), plan.actions, s)
    assert sum(a.kind == "evacuate" for a in final) == 1


def test_validate_rejects_non_whitelisted_and_unknown_ids():
    s = fleet()
    plan = validate(reply(act("open_valve", "go2-2", "B"), act("dispatch", "go9", "L-F3"),
                          act("dispatch", "go2-2", "MOON"), act("dispatch", "go2-2", "L-F3", 0.9),
                          act("hold", "go2-3", "", float("nan")), act("resume", "go2-1", "", 1.7)), s)
    assert [(a.kind, a.robot, a.target) for a in plan.actions] == [("dispatch", "go2-2", "L-F3")]
    assert len(plan.rejected) == 5
    assert "open_valve" not in CLOUD_KINDS


def test_validate_structure_errors():
    s = fleet()
    for bad in (None, [], {"actions": []}, {"situation": 3, "actions": []}, {"situation": "x", "actions": {}}):
        with pytest.raises(SchemaError):
            validate(bad, s)


def test_validate_trims_situation_and_ranks():
    s = fleet()
    text = "\n".join(f"line {k} " + "x" * 200 for k in range(5))
    plan = validate(reply(act("hold", "go2-3", "", 0.4), act("dispatch", "go2-2", "L-F3", 0.9),
                          act("dispatch", "go2-2", "L-F3", 0.7), situation=text), s)
    assert len(plan.situation) == 3 and all(len(x) <= 120 for x in plan.situation)
    assert [a.confidence for a in plan.actions] == [0.9, 0.4]


def test_validate_anomalies():
    s = fleet()
    plan = validate(reply(anomalies=[{"kind": "probable_leak", "zone": "B", "severity": "critical", "confidence": 0.8,
                                      "sensors": ["G-F3", "acoustic"], "note": "gas and dB rising"},
                                     {"kind": "x", "zone": "B", "severity": "apocalyptic", "confidence": 0.5}]), s)
    assert [a.kind for a in plan.anomalies] == ["probable_leak"]
    assert plan.anomalies[0].to_dict()["sensors"] == ["G-F3", "acoustic"]


def test_action_palette_fields():
    a = Action("dispatch", "go2-2", "L-F3", 0.8, "leak suspected")
    d = a.to_dict()
    assert {"id", "label", "kind", "robot", "target", "confidence", "rationale", "source"} <= set(d)
    assert "go2-2" in d["label"] and "L-F3" in d["label"]
    assert d["id"] == "cloud:dispatch:go2-2:L-F3"


def test_schema_is_json_and_enumerates_whitelist():
    json.dumps(RESPONSE_SCHEMA)
    kinds = RESPONSE_SCHEMA["properties"]["actions"]["items"]["properties"]["kind"]["enum"]
    assert tuple(kinds) == CLOUD_KINDS
    assert set(RESPONSE_SCHEMA["required"]) == {"situation", "actions", "anomalies"}


def test_evacuation_latches_until_clear():
    from policy.control.safety import LEL_CLEAR, hazard_zones
    assert LEL_CLEAR == 10.0
    assert hazard_zones(fleet(lel=36.0)) == {}
    assert hazard_zones(fleet(lel=36.0), latched={"B"}) == {"B": 36.0}
    assert hazard_zones(fleet(lel=9.0), latched={"B"}) == {}
    rules = edge_rules(fleet(lel=36.0), latched={"B"})
    assert {(a.kind, a.target) for a in rules} == {("retreat", "upwind"), ("evacuate", "B")}
    assert "latched" in rules[-1].rationale
    plan = validate(reply(act("dispatch", "go2-2", "L-F3")), fleet(lel=36.0))
    final, vetoed = arbitrate(rules, plan.actions, fleet(lel=36.0))
    assert vetoed and vetoed[0]["target"] == "L-F3"
