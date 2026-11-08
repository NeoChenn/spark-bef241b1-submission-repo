import json

import numpy as np
import pytest
from PIL import Image

from integration.dual_argon import SCRUBBER, FIXER, ArgonLink, FixedDetector, Timeline, fuse_fleet, ids_for
from policy.fleet_overlay import decision, draw, header_at, headline, load_headers, sample_beats


def frame(w, h, v=128):
    return np.full((h, w, 3), v, np.uint8)


def changed(a, b, box):
    x0, y0, x1, y1 = box
    return bool((a[y0:y1, x0:x1] != b[y0:y1, x0:x1]).any())


@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080)])
def test_every_beat_draws_panels_and_leaves_the_middle_clear(w, h):
    f = frame(w, h)
    for beat in sample_beats():
        out = draw(f, beat)
        assert out.shape == f.shape and out.dtype == np.uint8
        assert changed(f, out, (0, 0, w // 5, h // 6))
        assert changed(f, out, (w - w // 5, 0, w, h // 6))
        assert changed(f, out, (w // 3, h - h // 10, 2 * w // 3, h))
        assert not changed(f, out, (w // 3, h // 3, 2 * w // 3, 2 * h // 3))


def test_pil_in_pil_out_and_empty_header_is_a_no_op():
    img = Image.new("RGB", (1280, 720), (90, 90, 90))
    out = draw(img, {})
    assert isinstance(out, Image.Image) and np.array_equal(np.asarray(out), np.asarray(img))


def test_link_cut_banner_only_when_the_link_is_cut():
    beats = {b["beat"]: b for b in sample_beats()}
    f = frame(1920, 1080)
    top = (860, 30, 1060, 80)
    assert changed(f, draw(f, beats["7_link_cut"]), top)
    assert not changed(f, draw(f, beats["3_hold_dispatch"]), top)


def test_labels_follow_projected_points_and_skip_bad_ones():
    beat = sample_beats()[0]
    f = frame(1920, 1080)
    base = draw(f, beat)
    near = (900, 440, 1100, 560)
    out = draw(f, beat, {"unit1": (1000, 560), "unit2": None, "G-F9": (float("nan"), 3.0), "x": (-50, -50)})
    assert changed(base, out, near)
    assert np.array_equal(draw(f, beat, {"unit2": {"xy": (1000, 560), "visible": False}}), base)


def test_decision_lines_for_cloud_and_edge():
    cloud = {"last_command": {"reason": "Send Unit 2 to F2 | G-F2 14 %LEL, Unit 1 far", "source": "argon"}}
    assert decision(cloud) == ("Send Unit 2 to F2", "G-F2 14 %LEL, Unit 1 far")
    edge = {"last_command": {"reason": "edge rule UNVERIFIED: G-F2 11.2%LEL, robot 0.8%LEL at 4.6 m"}}
    assert decision(edge) == ("Unverified", "G-F2 11.2 %LEL, robot 0.8 %LEL at 4.6 m")
    assert decision({"line": "Hold", "last_command": None}) == ("Hold", "")


def test_headline_prefers_the_most_serious_verdict():
    h = {"argon": {"verdicts": {"G-F1": "OK", "G-F3": "PIPE_SENSOR_FAULT"}},
         "findings": [{"sensor": "G-F2", "kind": "LEAK_CONFIRMED"}]}
    assert headline(h, {"G-F2": h["findings"][0]}) == ("G-F2", "LEAK_CONFIRMED")


def test_jsonl_headers_skip_timeline_events_and_resolve_by_time(tmp_path):
    rows = sample_beats()
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in
                         [{"type": "finding", "t": 1.0, "kind": "OK"}, {"type": "command", "t": 2.0}] + rows))
    hs = load_headers(p)
    assert len(hs) == len(rows)
    assert header_at(hs, 12.0)["beat"] == "2_alarm"
    assert header_at(hs, 0.0) == {}


def test_draws_a_live_argon_link_header():
    d2 = FixedDetector("G-F2", "F2", (4.0, 0.0, 1.0))
    d2.read(lambda p: 0.14)
    link = ArgonLink("edge", Timeline(None))
    fd = fuse_fleet(d2, {SCRUBBER: (1.0, (0.0, 0.5)), FIXER: (0.5, (-12.0, 2.0))})
    link.tick(1.0, [fd], {}, [d2], ids_for([d2]), scrubber_pos=(0.0, 0.5), wait=True)
    header = {"robots": {"unit1": {"state": "HOLD"}, "unit2": {"state": "WALK"}},
              "detectors": [{"id": "G-F2", "flange": "F2", "lel": d2.last, "led": "red"}],
              "findings": [fd.to_dict()], "argon": link.header()}
    f = frame(1280, 720)
    assert changed(f, draw(f, header, {"unit1": (600, 500)}), (400, 640, 880, 720))


def test_caption_dicts_and_robot_lists_render_like_strings_and_dicts(monkeypatch):
    import policy.fleet_overlay as fo

    seen = []
    real = fo.ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        seen.append(str(text))
        return real(self, xy, text, *a, **k)

    monkeypatch.setattr(fo.ImageDraw.ImageDraw, "text", spy)
    cap = {"state": "HOLD", "lead": "Holding", "rest": " clear of F2", "text": "Holding clear of F2", "tone": "warn"}
    header = {"robots": [{"id": "unit1", "state": "HOLD", "caption": cap, "force_n": 0.0},
                         {"id": "unit2", "state": "WALK", "caption": {"text": "Walking to F2"}}]}
    draw(frame(1280, 720), header)
    assert "Holding" in seen and " clear of F2" in seen and "Walking to F2" in seen
    assert not any("{" in t for t in seen)
