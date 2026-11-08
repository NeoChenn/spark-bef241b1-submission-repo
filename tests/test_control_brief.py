from policy.control.agent import OrchestrationAgent, ReplayAgentBackend
from policy.control.backend import Thoughts
from policy.control.brief import fleet_lines, name, plain, tidy
from policy.control.port import MemoryCoordinator

SCREENSHOT = [
    "Analyzing Blocked Jobs",
    'I\'m now diving deeper into the meaning of a "clear" finding in this context. Specifically, I\'m examining the '
    "finding structure and its implications for job scheduling.",
    'I\'m now clarifying the implications of the "clear" finding on job scheduling, especially given the restored '
    "routes. It looks like inspect:gauge_06 is a gauge_read job, requested by the coordinator.",
    "I've just examined the read_fleet_state output at time 551.0. R1 and R2 are docked with full batteries, and no "
    'current jobs. R3 is in transit, with a slightly depleted battery at 0.9593, assigned to an "inspect" job for '
    "gauge_06#2.",
]


def test_meta_narration_and_headings_dropped():
    assert [tidy(x) for x in SCREENSHOT[:3]] == [None, None, None]


def test_situational_sentences_kept_with_ids_and_floats_cleaned():
    out = tidy(SCREENSHOT[3])
    assert out.startswith("R1 and R2 are docked with full batteries")
    assert "96%" in out and "pressure gauge 6" in out
    for raw in ("read_fleet_state", "551", "0.9593", "gauge_06", "#2", "I've"):
        assert raw not in out


def test_first_person_decisions_become_plain_lines():
    assert tidy("I'm now sending R2 to localise the leak at F7. I'll hold the weld on R1.") == \
        "Sending R2 to localise the leak at F7. Hold the weld on R1."
    assert tidy("Then, I'll call require_operator for painting jobs.") == \
        "Keep approvals with the operator for painting jobs."


def test_plain_names_ids_and_places():
    assert name("gauge_06") == "pressure gauge 6" and name("w12") == "walkway W12"
    assert name("rust_mid_east") == "rust patch, middle deck east" and name("weld_riser_east") == "east riser weld"
    assert plain("crate cr2 dropped at [9.0, -8.2, 42.5] -> walkway blocked") == \
        "crate dropped at top deck, centre -> walkway blocked"
    assert plain("R3 pre-empted from rust:rust_low_east -> pick up can ca1") == \
        "R3 pre-empted from rust repair at rust patch, lower deck east -> pick up can"


def test_thought_feed_filters_summaries_only():
    th = Thoughts()
    assert th.add("Analyzing Fleet State") is None
    assert th.add("Queue gas_survey · deck-2", "decision")["text"] == "Queue gas survey · deck-2"
    assert [x["text"] for x in th.view()] == ["Queue gas survey · deck-2"]


def test_fleet_lines_read_like_an_operator_board():
    state = {"robots": [{"id": "R2", "state": "TRANSIT", "battery": 0.9593, "position": [20.0, -9.0, 42.5],
                         "job": "leak_check:F5"}],
             "jobs": [{"id": "leak_check:F5", "kind": "leak_check", "target": "F5", "priority": 2, "status": "ASSIGNED",
                       "robot": "R2", "progress": 0.25}],
             "esd": {"active": False}, "person_pauses": {}}
    lines = fleet_lines(state)
    assert "- Unit 2 (R2) · leak check · 96% battery · walking to flange F5 · top deck, east end" in lines
    assert "- leak check · flange F5 · P2 · assigned · R2 · 25% done" in lines
    assert "ESD: not active" in lines


def test_agent_prompt_has_situation_board_and_id_table():
    p = MemoryCoordinator([{"id": "R1", "state": "IDLE", "battery": 0.8}],
                          [{"id": "J-3", "kind": "inspection", "target": "deck-2", "status": "QUEUED"}])
    p.situation = lambda: ["Gas: flange F5 22% LEL rising"]
    text = OrchestrationAgent(p, ReplayAgentBackend(), log_path=None).prompt("inspect deck 2", [])
    assert "SITUATION:\n- Gas: flange F5 22% LEL rising" in text
    assert "- Unit 1 (R1) · no job · 80% battery · idle" in text
    assert 'JOB IDS: [{"id":"J-3"' in text and "FLEET STATE" not in text


LIVE = [
    "I'm now focusing on the coordinator's reaction to the dropped can and crate. The system automatically assigned a "
    'P1 priority to "pick_can:ca1," pre-empting an ongoing rust repair. R1\'s re-routing adds time.',
    "I'm now clarifying my next steps, given the updated scenario. I can observe R3 at pick_can:ca1 which blocked an "
    "in-progress repair (rust:rust_low_east), while R1 is re-routing. The prompt says repairs are prioritized.",
    'I\'m now zeroing in on the *exact* triggering event for Argon\'s involvement: the crate drop, not just the can. '
    'The "recent" history proves this.',
    "I'm now considering the implications of the un-grippable crate. The lack of an automated job queue for the crate, "
    "unlike the can, is intriguing.",
    "I'm now zeroing in on a crucial point: if I approve and promote rust:rust_low_east to P1, what happens? Given "
    "that both jobs are at priority 1, can the priority system resolve this?",
]


def test_recorded_argon_rumination_is_condensed():
    out = [tidy(x) for x in LIVE]
    assert out[0] == "R1's re-routing adds time."
    assert out[1:] == [None, None, None, None]
