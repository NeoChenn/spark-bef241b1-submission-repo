from policy.fleet.jobs import SETUP_S, JobLibrary, JobTrace


def test_real_orchestrator_crack_escalates_and_is_cached():
    calls = []
    lib = JobLibrary()
    real = lib.runner
    lib.runner = lambda k, s: calls.append((k, s)) or real(k, s)
    tr = lib.get("crack", "crack")
    assert tr.outcome == "ESCALATE_TO_HUMAN"
    assert any(a["kind"] == "crack_or_burst" for a in tr.alerts)
    assert lib.get("crack", "crack") is tr and calls == [("crack", "crack")]


def test_real_orchestrator_clean_flange_check_inspects_then_done():
    tr = JobLibrary().get("leak_check", "clean")
    assert tr.outcome == "DONE"
    assert "LEAK_INSPECT" in [s for _, s, _ in tr.segments]
    assert tr.duration_s > 0


def test_real_orchestrator_clean_inspection_done():
    tr = JobLibrary().get("inspect", "clean")
    assert tr.outcome == "DONE" and tr.n_decisions >= 2


def test_state_at_walks_segments():
    tr = JobTrace("rust", "x", 50.0, "DONE", [(0.0, "SOAK", "a"), (30.0, "SCRUB", "b"), (50.0, "DONE", "c")])
    assert tr.state_at(1.0)[0] == "SETUP"
    s = SETUP_S["rust"]
    assert tr.state_at(s + 10)[0] == "SOAK"
    assert tr.state_at(s + 30)[0] == "SCRUB"
    assert tr.state_at(s + 99)[0] == "DONE"
    assert tr.total_s == s + 50.0
