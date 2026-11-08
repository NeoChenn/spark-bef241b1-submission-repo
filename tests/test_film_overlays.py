from integration.film_overlays.capture import PAGES, film_data


def test_board_decisions_come_from_the_run_with_their_source():
    b = film_data("board", 12.0, "runs_sample/fleet_demo")["board"]
    assert [(d["robot"], d["source"]) for d in b["decisions"]] == [("unit2", "argon"), ("unit1", "edge")]
    leak = next(d for d in b["detectors"] if d["id"] == "G-F5")
    assert leak["climb"][0][0] == 0.0 and leak["climb"][-1][0] == 1.0 and max(v for _, v in leak["climb"]) >= 10
    assert {f["id"]: f["after"] for f in b["fleet"]} == {"unit1": 1, "unit2": 0}


def test_conclusion_uses_run_numbers_and_vetted_lines_only():
    cards = film_data("conclusion", 17.0, "runs_sample/fleet_demo")["cards"]
    text = [c.get("text") or c.get("html") for c in cards]
    assert text[0] == "Alarm to fixed: <b>49 seconds</b>."
    assert "Argon's decisions cost about one cent." in text
    assert "On 6 July 1988, a gas leak on Piper Alpha killed 167 people." in text
    assert text[-1] == "It doesn't have to be that way."


def test_readouts_map_sim_time_onto_the_shot():
    g = film_data("readout", 15.0, "runs_sample/fleet_demo", (56.0, 72.0), "gas")["readout"]
    ts = [t for t, _ in g["series"]]
    assert ts == sorted(ts) and -0.5 <= ts[0] <= 0 and 15 <= ts[-1] <= 15.5
    assert g["series"][-1][1] < 5 < g["series"][0][1]
    r = film_data("readout", 15.0, "runs_sample/fleet_demo", (70.0, 92.0), "rust")["readout"]
    assert r["force"] and r["series"]


def test_every_page_has_data_without_a_run(tmp_path):
    for page in PAGES:
        d = film_data(page, 10.0, tmp_path)
        assert d["page"] == page and d["run"] is None
    assert film_data("board", 10.0, tmp_path)["board"]["decisions"][0]["html"].startswith("Leak at F5")
