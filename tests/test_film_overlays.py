from pathlib import Path

from integration.film_overlays.capture import HERE, PAGES, film_data

RUN = "runs_sample/film"


def test_board_follows_the_film_run_in_order_with_sources_and_sim_times():
    b = film_data("board", 11.5, RUN, (0.0, 18.3, 1.83))["board"]
    assert [(d["robot"], d["source"], d["t_run"]) for d in b["decisions"]] == [("unit1", "edge", 4.8), ("unit2", "argon", 18.3)]
    assert abs(b["decisions"][0]["u"] - 4.8 / 1.83) < 0.01 and abs(b["decisions"][1]["u"] - 10.0) < 0.01
    assert b["modelled"] and b["argon"]["calls"] == 3 and b["argon"]["cost_usd"] == 0.008
    seg = next(s for s in b["segments"] if s["id"] == "F5–F6")
    assert seg["index"][0][1] < 0.6 <= seg["index"][-1][1] and seg["coating"][-1][1] == "breached"
    leak = next(d for d in b["detectors"] if d["id"] == "G-F5")
    assert [x for _, x in leak["led"]] == ["green", "amber", "red"]
    assert {f["id"]: f["after"] for f in b["fleet"]} == {"unit1": 0, "unit2": 1}


def test_board_omits_a_decision_outside_the_shot():
    b = film_data("board", 10.0, RUN, (0.0, 10.0))["board"]
    assert [d["source"] for d in b["decisions"]] == ["edge"]


def test_conclusion_cards_short_and_long():
    short = film_data("conclusion", 6.0, RUN, cards="short")["cards"]
    long = film_data("conclusion", 12.0, RUN, cards="long")["cards"]
    assert short[0]["html"] == "Alarm to fixed: <b>48 seconds</b>." and len(short) == 3 and len(long) == 5
    assert long[-1]["lines"] == ["Today, a person walks up to the leak.", "It doesn't have to be that way."]
    assert not any("Argon" in str(c) for c in long)


def test_readouts_and_the_dispatch_note():
    g = film_data("readout", 10.5, RUN, (44.3, 55.0, 10.7 / 10.5), "gas")["readout"]
    done = 10.5 * (50.2 - 44.3) / 10.7
    assert all(v > 10 for t, v in g["series"] if t < done) and g["series"][-1][1] < 5
    n = film_data("readout", 1.5, RUN, None, "note")["readout"]
    assert n["source"] == "argon" and n["t_run"] == 63.2 and "dispatched" in n["html"]


def test_every_page_has_data_without_a_run_and_fonts_are_local(tmp_path):
    for page in PAGES:
        assert film_data(page, 10.0, tmp_path)["run"] is None
    for page in PAGES:
        assert "googleapis" not in (HERE / f"{page}.html").read_text()
    assert len(list((HERE / "fonts").glob("*.woff2"))) == 6 and len(list((HERE / "fonts").glob("*OFL.txt"))) == 2
    assert Path(HERE / "film.css").read_text().count("@font-face") == 6


def test_rust_readout_matches_scrub_motion():
    import pytest

    sm = pytest.importorskip("integration.scrub_motion")
    if not hasattr(sm, "film_job"):
        pytest.skip("scrub_motion.film_job not merged yet")
    from integration.film_overlays.capture import _u, _knots, load_run

    sim = (88.0, 107.9, 19.9 / 8.0)
    r = film_data("readout", 8.0, RUN, sim, "rust")["readout"]
    assert r["source"] in ("scrub_motion", "scrub_track")
    job, kn = sm.film_job(Path(RUN) / "timeline.jsonl"), _knots(load_run(RUN), 8.0, sim)
    pts = dict((u, v) for u, v in r["series"])
    for t in (90.0, 95.0, 100.0, 104.0, 106.0):
        u = min(pts, key=lambda x: abs(x - _u(kn, t)))
        a = job.at(88.0 + u * 19.9 / 8.0)
        assert abs(pts[u] - a["rust_fraction"]) < 1e-3
    assert any(v > 0.04 for u, v in r["series"] if 88.0 + u * 19.9 / 8.0 > 100.5)
    assert r["force_avg_s"] == 0.5
    u, f = r["force"][150]
    sim_t = 88.0 + u * 19.9 / 8.0
    want = sum(job.at(sim_t - k / 50)["brush_contact"]["force_N"] for k in range(25)) / 25
    assert abs(f - want) < (0.1 if r["source"] == "scrub_track" else 0.01)
    raw = [job.at(88.0 + x * 19.9 / 8.0)["brush_contact"]["force_N"] for x, _ in r["force"]]
    jump = lambda xs: max(abs(a - b) for a, b in zip(xs, xs[1:]))
    assert jump([v for _, v in r["force"]]) < jump(raw)
