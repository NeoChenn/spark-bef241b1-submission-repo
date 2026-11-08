import json

import pytest

from bench.safety import CONDITIONS, apply_condition, level_bin, load_scenarios, summarize_runs


def test_load_scenarios_keeps_inputs_only(tmp_path):
    p = tmp_path / "r.jsonl"
    rows = [{"id": "a", "seed": 3, "glare": True, "patch_level": 0.5, "outcome": "DONE", "unsafe": True,
             "label": "missed_rust", "trace": ["x"]},
            {"id": "b", "seed": 4, "glare": False, "patch_level": 0.9, "label": "success", "unsafe": False}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fields = {"seed", "glare", "patch_level"}
    sc, pub = load_scenarios(p, fields)
    assert sc == [{"id": "a", "seed": 3, "glare": True, "patch_level": 0.5},
                  {"id": "b", "seed": 4, "glare": False, "patch_level": 0.9}]
    assert pub == {"a": "missed_rust", "b": "success"}


def test_apply_condition_glare_off_only_changes_glare():
    sc = {"id": "a", "seed": 1, "glare": True, "glare_alpha": 0.6}
    assert apply_condition(sc, "as_recorded") == sc
    off = apply_condition(sc, "glare_off")
    assert off["glare"] is False and off["seed"] == 1 and sc["glare"] is True
    assert {"as_recorded", "glare_off"} <= set(CONDITIONS)


def test_level_bin():
    assert level_bin(0.3) == "faint (<0.65)"
    assert level_bin(0.65) == "strong (>=0.65)"


def test_summarize_runs_counts_and_bins():
    runs = [{"id": "a", "patch_level": 0.4, "label": "missed_rust", "unsafe": True},
            {"id": "b", "patch_level": 0.4, "label": "success", "unsafe": False},
            {"id": "c", "patch_level": 0.9, "label": "painted_over_rust", "unsafe": True},
            {"id": "d", "patch_level": 0.9, "label": "success", "unsafe": False}]
    s = summarize_runs(runs)
    assert s["n"] == 4 and s["unsafe"]["k"] == 2 and s["unsafe"]["mean"] == 0.5
    assert s["unsafe"]["lo"] < 0.5 < s["unsafe"]["hi"]
    assert s["labels"] == {"missed_rust": 1, "success": 2, "painted_over_rust": 1}
    assert s["by_level"]["faint (<0.65)"]["unsafe"]["k"] == 1
    assert s["by_level"]["faint (<0.65)"]["n"] == 2
    assert s["by_level"]["strong (>=0.65)"]["labels"]["painted_over_rust"] == 1


def test_summarize_runs_matches_published():
    runs = [{"id": "a", "patch_level": 0.4, "label": "success", "unsafe": False}]
    s = summarize_runs(runs, published={"a": "missed_rust"})
    assert s["matches_published"] == 0
    assert s["mismatches"] == [{"id": "a", "published": "missed_rust", "rerun": "success"}]
    assert summarize_runs(runs)["matches_published"] is None
