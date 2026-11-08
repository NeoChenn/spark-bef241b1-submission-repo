import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.hud import (FORCE_CAP, demo_state, FORCE_LIMIT, GREY, GREEN, AMBER, RED, draw_card, force_color, force_zone,
                        gauge_x, lerp_color, parse_state, pipeline, sparkline_points, state_color)


def test_lerp_color_endpoints_and_clamp():
    assert lerp_color((0, 0, 0), (200, 100, 50), 0.5) == (100, 50, 25)
    assert lerp_color((0, 0, 0), (200, 100, 50), -1) == (0, 0, 0)
    assert lerp_color((0, 0, 0), (200, 100, 50), 3) == (200, 100, 50)


def test_force_color_bands():
    assert force_color(0.0) == GREY
    assert force_color(8.0) == GREY
    assert force_color(30.0) == GREEN
    assert force_color(FORCE_CAP) == AMBER
    assert force_color(FORCE_LIMIT + 1) == RED
    assert force_color(80.0) == RED


def test_force_color_gets_greener_as_force_rises_into_band():
    g = [force_color(f)[1] - force_color(f)[0] for f in (10, 14, 18, 22, 26)]
    assert all(b >= a for a, b in zip(g, g[1:])) and g[-1] > g[0]


def test_force_color_warms_toward_cap():
    r = [force_color(f)[0] for f in (34, 35.5, 37, 38)]
    assert all(b >= a for a, b in zip(r, r[1:])) and r[-1] > r[0]


def test_force_zone_labels():
    assert force_zone(3) == "no contact"
    assert force_zone(15) == "light"
    assert force_zone(30) == "effective"
    assert force_zone(39) == "near cap"
    assert force_zone(45) == "over limit"


def test_gauge_x_maps_and_clamps():
    assert gauge_x(0, 100, 200, fmax=50) == 100
    assert gauge_x(25, 100, 200, fmax=50) == 200
    assert gauge_x(500, 100, 200, fmax=50) == 300
    assert gauge_x(-5, 100, 200, fmax=50) == 100


def test_parse_state_splits_key_and_caption():
    assert parse_state("SCRUB  removing rust") == ("SCRUB", "removing rust")
    assert parse_state("RE-PLAN  new rust patch") == ("RE-PLAN", "new rust patch")
    assert parse_state("BACK_OFF") == ("BACK_OFF", "")


def test_state_colors_distinct_for_alerts():
    assert state_color("SCRUB") != state_color("BACK_OFF")
    assert state_color("BACK_OFF") == RED
    assert state_color("unknown") == state_color("PATROL")


def test_pipeline_marks_done_active_todo():
    p = dict(pipeline("SCRUB"))
    assert p["SPRAY"] == "done" and p["SOAK"] == "done" and p["SCRUB"] == "active"
    assert p["VERIFY"] == "todo" and p["PAINT"] == "todo"
    assert dict(pipeline("APPROACH"))["SCRUB"] == "active"
    assert dict(pipeline("RE-PLAN"))["SCRUB"] == "active"
    v = dict(pipeline("VERIFY"))
    assert v["SCRUB"] == "done" and v["VERIFY"] == "active"
    assert all(s == "done" for _, s in pipeline("DONE"))
    assert all(s == "todo" for _, s in pipeline("PATROL"))


def test_sparkline_points_fit_box():
    pts = sparkline_points([0, 25, 50, 100], 10, 20, 90, 30, vmax=50)
    assert pts[0] == (10, 50) and pts[-1][0] == 100
    assert pts[1] == (40, 35) and pts[2][1] == 20 and pts[3][1] == 20
    assert sparkline_points([], 0, 0, 10, 10, 50) == []
    assert sparkline_points([5], 0, 0, 10, 10, 50) == [(0, 9), (10, 9)]


@pytest.mark.parametrize("state,force", [("SCRUB  removing rust", 29.0), ("BACK_OFF  retract", 44.0),
                                         ("APPROACH  brush to rust", 0.0)])
def test_draw_card_composites_in_corner(state, force):
    frame = np.full((720, 1280, 3), 40, np.uint8)
    out = draw_card(frame, state=state, frac=0.42, force=force, history=[0, 10, 20, force], t=5.0)
    assert out.shape == frame.shape and out.dtype == np.uint8
    diff = np.abs(out.astype(int) - frame).sum(-1) > 0
    ys, xs = np.nonzero(diff)
    assert xs.min() >= 0 and xs.max() < 640 and ys.min() > 300


def test_project_center_and_offsets():
    from policy.hud import project

    assert project((5, 0, 0), (0, 0, 0), (1, 0, 0), 15.0, 1280, 720, 15.0) == pytest.approx((640, 360))
    x, y = project((5, -1, 1), (0, 0, 0), (1, 0, 0), 15.0, 1280, 720, 15.0)
    assert x > 640 and y < 360
    assert x - 640 == pytest.approx(720 / 5, rel=1e-6) and 360 - y == pytest.approx(720 / 5, rel=1e-6)
    assert project((-5, 0, 0), (0, 0, 0), (1, 0, 0), 15.0, 1280, 720) is None


def test_draw_card_with_brush_ring_draws_near_brush():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = draw_card(frame, "SCRUB  removing rust", 0.5, 30.0, [30.0], 1.0, brush_px=(900, 200))
    assert np.abs(out[160:240, 860:940].astype(int)).sum() > 0
    assert out[100:300, 700:800].sum() == 0


def test_demo_state_maps_scrub_clip_end_to_clean():
    assert demo_state("DONE  retract brush") == "CLEAN  retract brush · paint next"
    assert demo_state("SCRUB  removing rust") == "SCRUB  removing rust"
    assert demo_state("DONE  4 % residual flagged for touch-up").startswith("VERIFY")
    p = dict(pipeline("CLEAN"))
    assert p["VERIFY"] == "done" and p["PAINT"] == "todo"
    assert state_color("CLEAN") == GREEN
