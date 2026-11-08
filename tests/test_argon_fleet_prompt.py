import json
import os
from datetime import date
from pathlib import Path

import pytest

from integration import dual_argon as da
from integration import argon_fleet_prompt as ap

DOCK = (-12.0, 2.0)
U1 = (0.0, 0.5)
OUT = Path(__file__).resolve().parents[1] / "runs_sample" / "argon_fleet"
ALL = set(ap.KINDS)


def dets(g2, g1=1.0, stuck1=None):
    d2 = da.FixedDetector("G-F2", "F2", (4.0, 0.0, 1.0))
    d1 = da.FixedDetector("G-F1", "F1", (-9.0, 0.0, 1.0), stuck_at=stuck1)
    d2.read(lambda p: g2 / 100.0)
    d1.read(lambda p: g1 / 100.0)
    return [d2, d1]


def robot(pos, state, lel, bat, job, held=False):
    return {"pos": pos, "state": state, "lel": lel, "battery": bat, "job": job, "held": held}


def scenario(name):
    if name == "leak_start":
        ds = dets(7.0)
        ds[0].trend = 2.5
        rb = {"unit1": robot(U1, "SCRUB", 0.8, 0.81, "rust R3 scrub"), "unit2": robot(DOCK, "DOCKED", 0.2, 0.97, "")}
        return ds, rb, "G-F2 amber and rising", {(ap.HOLD, "unit1")}, {ap.FIX_LEAK, ap.RESUME_RUST,
                                                                    ap.LOG_SENSOR_FAULT, ap.RETREAT}
    if name == "unverified":
        ds = dets(14.0)
        ds[0].trend = 3.0
        rb = {"unit1": robot(U1, "HOLD", 1.5, 0.80, "rust R3 paused", True), "unit2": robot(DOCK, "DOCKED", 0.2, 0.97, "")}
        return ds, rb, "", {(ap.GO_CHECK, "unit2", "F2")}, {ap.FIX_LEAK, ap.RESUME_RUST, ap.LOG_SENSOR_FAULT,
                                                            ap.RETREAT}
    if name == "confirmed":
        ds = dets(15.0)
        ds[0].trend = 0.5
        rb = {"unit1": robot(U1, "HOLD", 1.6, 0.79, "rust R3 paused", True),
              "unit2": robot((4.6, 0.6), "SNIFF", 13.0, 0.91, "check F2")}
        return ds, rb, "", {(ap.FIX_LEAK, "unit2", "F2")}, ALL - {ap.FIX_LEAK, ap.HOLD, ap.ESCALATE}
    if name == "fixed":
        ds = dets(2.0)
        ds[0].trend = -4.0
        rb = {"unit1": robot(U1, "HOLD", 0.6, 0.79, "rust R3 paused", True),
              "unit2": robot((4.6, 0.6), "VERIFY", 1.2, 0.88, "fix F2 done")}
        return ds, rb, "zone clear: G-F2 green after re-torque", {(ap.RESUME_RUST, "unit1")}, ALL - {ap.RESUME_RUST}
    if name == "stuck_sensor":
        ds = dets(1.0, stuck1=16.0)
        ds[1].trend = 0.0
        rb = {"unit1": robot(U1, "SCRUB", 0.5, 0.74, "rust R3 scrub"),
              "unit2": robot((-8.5, 0.8), "TRANSIT", 0.4, 0.86, "return to dock")}
        return ds, rb, "", {(ap.LOG_SENSOR_FAULT, "supv", "G-F1")}, ALL - {ap.LOG_SENSOR_FAULT, ap.ESCALATE}
    if name == "all_clear":
        ds = dets(0.8, g1=0.6)
        rb = {"unit1": robot(U1, "SCRUB", 0.4, 0.83, "rust R3 scrub"), "unit2": robot(DOCK, "DOCKED", 0.1, 0.99, "")}
        return ds, rb, "routine check", set(), ALL
    raise KeyError(name)


NAMES = ("leak_start", "unverified", "confirmed", "fixed", "stuck_sensor", "all_clear")


def snapshot(name):
    ds, rb, event, need, banned = scenario(name)
    sn = {rid: (r["lel"], r["pos"]) for rid, r in rb.items()}
    fds = [da.fuse_fleet(d, sn) for d in ds]
    fd = next((f for f in fds if f.kind != da.OK), None)
    if name == "leak_start":
        fd = fds[0]
    text = ap.build_prompt(fd, rb, ds, 12.0, event)
    return text, fd, da.ids_for(ds), need, banned


def judge(cmds, need, banned):
    got = {(c.kind, c.robot, c.target) for c in cmds}
    missing = [n for n in need if not any(g[:len(n)] == n for g in got)]
    bad = sorted(c.kind for c in cmds if c.kind in banned)
    return not missing and not bad, missing, bad


def reply(*cmds, verdict="UNVERIFIED"):
    return json.dumps({"verdict": verdict, "commands": [
        {"kind": k, "robot": r, "target": t, "confidence": 0.9, "why": "w"} for k, r, t in cmds]})


def test_threshold_constants_match_dual_argon():
    assert (ap.CLEAR_LEL, ap.ALARM_LEL, ap.STOP_LEL, ap.NEAR_M, ap.SPARK_RADIUS_M) == (
        da.CLEAR_LEL, da.ALARM_LEL, da.STOP_LEL, da.NEAR_M, da.SPARK_RADIUS_M)
    assert (ap.SCRUBBER, ap.FIXER, ap.SUPERVISOR) == (da.SCRUBBER, da.FIXER, da.SUPERVISOR)
    assert set(da.COMMANDS) <= set(ap.KINDS)
    assert da.SCHEMA is ap.SCHEMA and da.SYSTEM is ap.SYSTEM


def test_schema_is_strict_and_small():
    item = ap.SCHEMA["properties"]["commands"]["items"]
    assert set(item["required"]) == {"kind", "robot", "target", "confidence", "why"}
    assert set(item["properties"]["kind"]["enum"]) == set(ap.KINDS)
    assert item["properties"]["robot"]["enum"] == ["unit1", "unit2", "supv"]
    assert len(ap.SYSTEM) < 2000


def test_prompt_is_compact_and_has_spark_and_health():
    text, fd, ids, _, _ = snapshot("unverified")
    snap = json.loads(text)
    assert snap["spark"] == ["G-F2"] and snap["finding"]["kind"] == da.UNVERIFIED
    assert snap["robots"]["unit1"]["held"] is True and snap["robots"]["unit2"]["bat"] == 0.97
    g2 = snap["detectors"][0]
    assert g2["led"] == "red" and g2["health"] == da.UNVERIFIED and g2["trend"] == 3.0
    assert len(text) < 900
    assert json.loads(snapshot("stuck_sensor")[0])["spark"] == []


def test_build_prompt_accepts_integrator_call_shapes():
    ds = dets(1.0)
    assert json.loads(ap.build_prompt(None, {}, ds, 3.0, event="zone clear"))["event"] == "zone clear"


def test_parse_routes_and_clips_why():
    text, fd, ids, _, _ = snapshot("unverified")
    long = json.dumps({"verdict": "UNVERIFIED", "commands": [
        {"kind": "go_check", "robot": "unit2", "target": "F2", "confidence": 0.8, "why": "x" * 200}]})
    [c] = ap.parse_reply(long, fd, ids)
    assert (c.kind, c.robot, c.target, c.source, c.finding) == ("go_check", "unit2", "F2", "argon", "UNVERIFIED")
    assert len(c.reason) == 90


@pytest.mark.parametrize("cmd", [("fix_leak", "unit1", "F2"), ("go_check", "unit2", "F9"),
                                 ("log_sensor_fault", "unit2", "G-F2"), ("approve", "supv", ""),
                                 ("resume_rust", "unit2", "")])
def test_parse_rejects_wrong_robot_target_or_kind(cmd):
    text, fd, ids, _, _ = snapshot("unverified")
    with pytest.raises(ValueError):
        ap.parse_reply(reply(cmd), fd, ids)


def test_parse_rejects_bad_payloads():
    _, fd, ids, _, _ = snapshot("unverified")
    for bad in ("", "nope", '{"verdict":"OK"}', '{"commands":[1]}',
                reply(("hold", "unit1", "F2")).replace("0.9", "1.5")):
        with pytest.raises(ValueError):
            ap.parse_reply(bad, fd, ids)


def test_hazard_first_refuses_repair_and_demands_retreat():
    ds = dets(30.0)
    fd = da.fuse(ds[0], 26.0, 0.8, robot="unit2")
    ids = da.ids_for(ds)
    with pytest.raises(ValueError):
        ap.parse_reply(reply(("fix_leak", "unit2", "F2")), fd, ids)
    with pytest.raises(ValueError):
        ap.parse_reply(reply(("hold", "unit1", "F2")), fd, ids)
    out = ap.parse_reply(reply(("hold", "unit1", "F2"), ("retreat", "unit2", "F2"), ("escalate", "supv", "F2")),
                         fd, ids)
    assert [c.robot for c in out] == ["unit1", "unit2", "supv"]


def test_fault_ticket_goes_to_human_inbox_and_blocks_repair():
    text, fd, ids, need, banned = snapshot("stuck_sensor")
    assert fd.kind == da.PIPE_SENSOR_FAULT
    with pytest.raises(ValueError):
        ap.parse_reply(reply(("fix_leak", "unit2", "F1")), fd, ids)
    out = ap.parse_reply(reply(("log_sensor_fault", "supv", "G-F1")), fd, ids)
    assert judge(out, need, banned)[0]


def test_scenarios_fuse_to_the_brief():
    kinds = {n: (snapshot(n)[1].kind if snapshot(n)[1] else None) for n in NAMES}
    assert kinds == {"leak_start": da.OK, "unverified": da.UNVERIFIED, "confirmed": da.LEAK_CONFIRMED,
                     "fixed": None, "stuck_sensor": da.PIPE_SENSOR_FAULT, "all_clear": None}


def test_link_uses_this_prompt_end_to_end(tmp_path):
    text, fd, ids, _, _ = snapshot("unverified")
    tl = tmp_path / "rec.jsonl"
    row = {"type": "argon_call", "status": "ok", "key": "G-F2:UNVERIFIED", "latency_ms": 10.0, "tokens_in": 500,
           "tokens_out": 60, "tokens_thought": 40, "tokens_cached": 0,
           "text": reply(("go_check", "unit2", "F2"))}
    tl.write_text(json.dumps(row) + "\n")
    link = da.ArgonLink("replay", da.Timeline(None), caller=da.ReplayCaller(str(tl)))
    _, rb, _, _, _ = scenario("unverified")
    rb["unit1"]["held"] = False
    ds = dets(14.0)
    out = link.tick(0.0, [fd], rb, ds, ids, scrubber_pos=U1, wait=True)
    out += link.tick(1.0, [fd], rb, ds, ids, scrubber_pos=U1, wait=True)
    assert ("go_check", "unit2", "F2") in {(c.kind, c.robot, c.target) for c in out}


@pytest.mark.skipif(os.environ.get("ARGON_LIVE") != "1", reason="live Gemini calls need ARGON_LIVE=1")
def test_live_scenarios():
    from policy.control.limiter import cost_usd

    caller = da.LiveCaller(thinking=ap.THINKING)
    assert caller.available and caller.backend.model == ap.MODEL
    caller.backend.timeout = 90.0
    runs = int(os.environ.get("ARGON_RUNS", "1"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(runs):
        for name in NAMES:
            text, fd, ids, need, banned = snapshot(name)
            try:
                r = caller(text, name)
            except Exception as e:
                rows.append({"scenario": name, "run": i, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            try:
                cmds = ap.parse_reply(r["text"], fd, ids)
                ok, missing, bad = judge(cmds, need, banned)
                err = ""
            except ValueError as e:
                cmds, ok, missing, bad, err = [], False, [], [], str(e)
            rows.append({"scenario": name, "run": i, "ok": ok, "missing": [list(m) for m in missing], "banned": bad,
                         "invalid": err, "latency_ms": round(r["latency_ms"], 1), "tokens_in": r["tokens_in"],
                         "tokens_out": r["tokens_out"], "tokens_thought": r["tokens_thought"],
                         "cost_usd": round(cost_usd(r["tokens_in"], r["tokens_out"], r["tokens_thought"], date.today(),
                                                    r["tokens_cached"]), 6),
                         "commands": [[c.kind, c.robot, c.target, c.confidence, c.reason] for c in cmds],
                         "prompt": text, "text": r["text"]})
    tag = os.environ.get("ARGON_TAG", "run")
    (OUT / f"{tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert all(x["ok"] for x in rows), [(x["scenario"], x.get("missing"), x.get("banned"), x.get("invalid"), x.get("error")) for x in rows
                                        if not x["ok"]]


def test_resume_rust_is_refused_while_a_leak_is_open():
    resume = json.dumps({"verdict": "UNVERIFIED", "situation": "s", "commands": [
        {"command": "resume_rust", "robot": "unit1", "target": "", "confidence": 0.9, "rationale": "r"}]})
    d2, _ = dets(14.0)
    both = json.dumps({"verdict": "UNVERIFIED", "situation": "s", "commands": [
        {"command": "go_check", "robot": "unit2", "target": "F2", "confidence": 0.9, "rationale": "r"},
        {"command": "resume_rust", "robot": "unit1", "target": "", "confidence": 0.9, "rationale": "r"}]})
    for kind_reading in ((2.0, 4.0), (9.0, 1.0)):
        fd = da.fuse(d2, *kind_reading, robot="unit2")
        with pytest.raises(ValueError):
            ap.parse_reply(resume, fd, {"F2", "G-F2"})
        out = ap.parse_reply(both, fd, {"F2", "G-F2"})
        assert [c.kind for c in out] == ["go_check"]
        assert out.vetoed[0]["why"] == "resume refused: leak open at F2"
    assert [c.kind for c in ap.parse_reply(resume, None, {"F2", "G-F2"})] == ["resume_rust"]
