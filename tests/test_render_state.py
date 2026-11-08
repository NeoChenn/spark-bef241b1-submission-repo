import math

import numpy as np
import pytest

from rig.render.state import (StateLog, _mat_quat, _quat_mat, blend_value, blend_view, interpolate, merge_messages,
                              resample, split)


def msg(seq, attrs=(), specs=(), **kw):
    return {"type": "state", "seq": seq, "t": seq / 30, "attrs": [list(a) for a in attrs], "specs": [list(s) for s in specs], **kw}


def test_log_keeps_latest_value_per_attribute():
    log = StateLog().merge(msg(1, [("/a", "x", "float", 1.0), ("/b", "y", "float", 2.0)]))
    log.merge(msg(2, [("/a", "x", "float", 3.0)]))
    out = log.message()
    assert out["seq"] == 2
    assert sorted(out["attrs"]) == [["/a", "x", "float", 3.0], ["/b", "y", "float", 2.0]]


def test_spec_drops_older_attrs_under_it_but_keeps_newer():
    log = StateLog().merge(msg(1, [("/r/link", "xformOp:translate", "double3", [0, 0, 0]), ("/q", "x", "float", 1.0)]))
    log.merge(msg(2, specs=[("/r", "#usda 1.0")]))
    log.merge(msg(3, [("/r/link", "xformOp:translate", "double3", [1, 0, 0])]))
    out = log.message()
    assert out["specs"] == [["/r", "#usda 1.0"]]
    assert ["/r/link", "xformOp:translate", "double3", [1, 0, 0]] in out["attrs"]
    assert ["/q", "x", "float", 1.0] in out["attrs"]
    assert len(out["attrs"]) == 2


def test_spec_prefix_is_path_aware():
    log = StateLog().merge(msg(1, [("/robot2/a", "x", "float", 1.0)]))
    log.merge(msg(2, specs=[("/robot", None)]))
    assert log.message()["attrs"] == [["/robot2/a", "x", "float", 1.0]]


def test_take_clears_delta_and_keeps_head():
    log = StateLog().merge(msg(4, [("/a", "x", "float", 1.0)], view={"m": [1] * 16}))
    first = log.take()
    assert first["attrs"] and first["view"]
    second = log.take()
    assert second["attrs"] == [] and second["seq"] == 4 and second["view"] == {"m": [1] * 16}
    assert log.empty()


def test_merge_messages_combines_deltas_and_events():
    a = msg(1, [("/a", "x", "float", 1.0)], events=[{"e": 1}])
    b = msg(2, [("/b", "x", "float", 2.0)], events=[{"e": 2}], key=True)
    out = merge_messages(a, b)
    assert out["seq"] == 2 and out["key"] is True
    assert len(out["attrs"]) == 2
    assert out["events"] == [{"e": 1}, {"e": 2}]


def test_split_separates_scene_and_hud():
    scene, meta = split({**msg(3, [("/a", "x", "float", 1.0)]), "ts": 5.0, "scene": {"cinematic": "night"},
                         "events": [1], "fleet": {"robots": []}})
    assert set(scene) == {"type", "seq", "t", "attrs", "specs", "ts", "scene"}
    assert meta == {"events": [1], "fleet": {"robots": []}, "t": 0.1}


def test_blend_value_kinds():
    assert blend_value("float", 0.0, 2.0, 0.25) == pytest.approx(0.5)
    assert blend_value("double3", [0, 0, 0], [2, 4, 6], 0.5) == pytest.approx([1, 2, 3])
    assert blend_value("token", "a", "b", 0.1) == "b"
    assert blend_value("bool", False, True, 0.1) is True
    assert blend_value("float", None, 2.0, 0.5) == 2.0
    h = math.sqrt(0.5)
    q = blend_value("quatf", [1, 0, 0, 0], [0, 0, 0, 1], 0.5)
    assert q == pytest.approx([h, 0, 0, h], abs=1e-6)
    assert blend_value("quatd", [1, 0, 0, 0], [-1, 0, 0, 0], 0.5) == pytest.approx([1, 0, 0, 0])


def test_quat_matrix_roundtrip():
    q = np.array([0.8, 0.2, -0.4, 0.4])
    q /= np.linalg.norm(q)
    assert _mat_quat(_quat_mat(q)) == pytest.approx(q.tolist(), abs=1e-9)


def row_matrix(yaw, pos):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, *pos, 1]


def test_blend_view_interpolates_pose_and_focal():
    a = {"m": row_matrix(0.0, (0, 0, 0)), "focal": 20.0, "path": "/c"}
    b = {"m": row_matrix(math.pi / 2, (10, 0, 2)), "focal": 30.0, "path": "/c"}
    out = blend_view(a, b, 0.5)
    assert out["m"] == pytest.approx(row_matrix(math.pi / 4, (5, 0, 1)), abs=1e-9)
    assert out["focal"] == pytest.approx(25.0)
    assert blend_view(None, b, 0.5) is b


def test_interpolate_states():
    a = msg(0, [("/r", "xformOp:translate", "double3", [0, 0, 0])])
    b = msg(3, [("/r", "xformOp:translate", "double3", [3, 0, 0]), ("/n", "x", "float", 1.0)])
    out = interpolate(a, b, 1 / 3)
    assert out["t"] == pytest.approx(1 / 30)
    assert out["attrs"][0][3] == pytest.approx([1, 0, 0])
    assert out["attrs"][1][3] == 1.0


def test_resample_maps_output_frames_to_segments():
    out = resample([0.0, 0.1, 0.2], 20)
    assert [i for i, _ in out] == [1, 1, 2, 2, 2]
    assert [u for _, u in out] == pytest.approx([0.0, 0.5, 0.0, 0.5, 1.0])
    assert resample([], 30) == []


def test_inbox_merges_until_taken():
    from rig.render.state import Inbox

    box = Inbox()
    assert box.take() is None
    box.put({"type": "event", "name": "x"}, b"")
    box.put(msg(1, [("/a", "x", "float", 1.0)], view={"m": [0] * 16}, ts=1.0))
    box.put(msg(2, [("/b", "x", "float", 2.0)], ts=2.0))
    out = box.take()
    assert out["seq"] == 2 and out["ts"] == 2.0 and len(out["attrs"]) == 2 and out["view"]
    assert box.take() is None
    assert box.received == 2


def test_inbox_records_every_message(tmp_path):
    import json

    from rig.render.state import Inbox

    path = tmp_path / "log.jsonl"
    box = Inbox(record=path)
    box.put(msg(1, [("/a", "x", "float", 1.0)]))
    box.put(msg(2, [("/a", "x", "float", 2.0)]))
    box.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [m["seq"] for m in lines] == [1, 2]


def test_replay_plan_applies_deltas_once_and_blends():
    from rig.render.state import replay_plan

    log = [msg(0, [("/r", "p", "double3", [0, 0, 0])], specs=[("/r", "S")]), msg(3, [("/r", "p", "double3", [3, 0, 0])]),
           msg(6, [("/r", "p", "double3", [6, 0, 0])])]
    plan = list(replay_plan(log, fps=60))
    assert len(plan) == 13
    first_deltas, first = plan[0]
    assert [d["seq"] for d in first_deltas] == [0] and first["specs"] == []
    deltas, mid = plan[1]
    assert [d["seq"] for d in deltas] == [3]
    assert mid["attrs"][0][3] == pytest.approx([0.5, 0, 0])
    assert sum(len(d) for d, _ in plan) == 3
    assert plan[-1][1]["attrs"][0][3] == pytest.approx([6, 0, 0])
    assert [len(d) for d, _ in replay_plan(log, fps=0)] == [1, 1, 1]
