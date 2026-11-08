import json

import numpy as np
import pytest

from bench import pick_scenes as ps
from policy.pickgrasp import vlm_grasp as vg

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def down_cam(h=1.0):
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    T[2, 3] = h
    return T


def reply(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 50}}


def test_parse_points_dict_and_list():
    pts, ap = vg.parse_points('{"points": [{"point": [500, 250], "label": "can"}], "approach": "side"}', 640, 480)
    assert pts == [("can", 160.0, 240.0)] and ap == "side"
    pts, ap = vg.parse_points('```json\n[{"point": [1200, -5], "label": "box"}]\n```', 100, 100)
    assert pts == [("box", 0.0, 100.0)] and ap == "top"
    assert vg.parse_points("no idea", 10, 10) == ([], "top")
    assert vg.parse_points('{"points": [[205, 527], "approach": "side"}', 1000, 1000) == ([("", 527.0, 205.0)], "side")
    assert vg.parse_points('{"points": [{"point": [1]}], "approach": "x"}', 10, 10) == ([], "top")


def test_project_and_plane_roundtrip():
    T = down_cam(1.0)
    u, v = vg.project([0.1, -0.2, 0.0], K, T)
    assert (u, v) == pytest.approx((370.0, 340.0))
    assert vg.on_plane(u, v, K, T, 0.0) == pytest.approx([0.1, -0.2, 0.0])
    assert vg.on_plane(u, v, K, T, 2.0) is None


def test_from_depth_snaps_to_top_surface():
    T = down_cam(1.0)
    depth = np.full((480, 640), 1.0)
    depth[200:260, 300:340] = 0.9
    p = vg.from_depth(299.0, 230.0, depth, K, T, win=4)
    assert p[2] == pytest.approx(0.1)
    p = vg.from_depth(100.0, 100.0, depth, K, T)
    assert p[2] == pytest.approx(0.0)
    assert vg.from_depth(5.0, 5.0, np.full((480, 640), np.nan), K, T) is None


def test_grasp_uses_vlm_then_backprojects():
    T = down_cam(1.0)
    seen = []

    def transport(url, body):
        seen.append(url)
        return reply({"points": [{"point": [500, 500], "label": "bottle"}], "approach": "top"})

    g = vg.VlmGrasp(api_key="k", transport=transport)
    t = g.grasp(np.zeros((480, 640, 3), np.uint8), K, T, "bottle", plane_z=0.0)
    assert t.source == "vlm" and t.label == "bottle"
    assert t.xyz == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert t.approach == [0.0, 0.0, -1.0]
    assert "gemini-robotics-er-2-preview" in seen[0]
    assert g.calls[0]["tokens_in"] == 1000


def test_side_approach_is_horizontal():
    T = down_cam(1.0)
    T[0, 3] = -0.5
    g = vg.VlmGrasp(api_key="k", transport=lambda u, b: reply({"points": [{"point": [500, 750]}], "approach": "side"}))
    t = g.grasp(np.zeros((480, 640, 3), np.uint8), K, T, "can", plane_z=0.0)
    assert t.approach == pytest.approx([1.0, 0.0, 0.0])


def test_fallback_on_no_key_error_empty_and_budget():
    T = down_cam(1.0)
    fb = lambda rgb, depth, K, T: vg.GraspTarget([0.1, 0.0, 0.05], [0.0, 0.0, -1.0], [1.0, 2.0], "detector")
    img = np.zeros((480, 640, 3), np.uint8)
    t = vg.VlmGrasp(api_key="", transport=lambda u, b: 1 / 0).grasp(img, K, T, "can", fallback=fb)
    assert t.source == "fallback:detector"

    def boom(url, body):
        raise RuntimeError("down")

    g = vg.VlmGrasp(api_key="k", transport=boom)
    assert g.grasp(img, K, T, "can", fallback=fb).source == "fallback:detector"
    assert "down" in g.last_error
    empty = vg.VlmGrasp(api_key="k", transport=lambda u, b: reply({"points": []}))
    assert empty.grasp(img, K, T, "can") is None
    capped = vg.VlmGrasp(api_key="k", transport=lambda u, b: reply({"points": [{"point": [500, 500]}]}), max_calls=0)
    assert capped.grasp(img, K, T, "can", fallback=fb).source == "fallback:detector"


def test_implausible_point_falls_back():
    T = down_cam(1.0)
    depth = np.full((480, 640), 5.0)
    g = vg.VlmGrasp(api_key="k", transport=lambda u, b: reply({"points": [{"point": [500, 500]}]}))
    assert g.grasp(np.zeros((480, 640, 3), np.uint8), K, T, "can", depth=depth) is None


def test_model_404_falls_through_to_next():
    import io
    import urllib.error

    tried = []

    def transport(url, body):
        tried.append(url)
        if len(tried) == 1:
            raise urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b"{}"))
        return reply({"points": [{"point": [500, 500]}]})

    g = vg.VlmGrasp(api_key="k", transport=transport, models=("a", "b"))
    assert g.grasp(np.zeros((480, 640, 3), np.uint8), K, down_cam(), "can", plane_z=0.0).source == "vlm"
    assert g.model == "b" and "/a:" in tried[0]


def test_detection_target_lifts_centre_to_top():
    T = down_cam(1.0)
    up = vg.detection_target({"xyz": [0.1, 0.0, 0.06], "lying": False, "length": 0.12, "diameter": 0.066}, K, T)
    assert up.xyz == pytest.approx([0.1, 0.0, 0.12])
    ly = vg.detection_target({"xyz": [0.1, 0.0, 0.033], "lying": True, "length": 0.12, "diameter": 0.066}, K, T)
    assert ly.xyz == pytest.approx([0.1, 0.0, 0.066])


def test_synth_scene_gt_visible_and_consistent():
    s = ps.make_scene(3, "bottle")
    assert s.rgb.shape == (ps.H, ps.W, 3) and s.depth.shape == (ps.H, ps.W)
    u, v = s.gt_px
    assert s.inst[int(round(v)), int(round(u))] == s.target
    p = vg.from_depth(u, v, s.depth, s.K, s.T, win=1)
    assert np.linalg.norm(np.array(p) - s.gt_xyz) < 0.01
    assert len(s.objects) >= 3
