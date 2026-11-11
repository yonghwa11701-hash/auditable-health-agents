"""The public demo serves the recorded run and runs the same checks offline."""
from web.app import EVENTS, run_checks


def test_demo_serves_recorded_run():
    assert len(EVENTS) == 21 and EVENTS[0]["kind"] == "request"


def test_demo_checks_block_invented_numbers_and_advice():
    assert run_checks({"text": "Docetaxel + Gefitinib: tier 1, score 162.3, 95% interval 127.3 to 197.3."})["all_pass"]
    assert not run_checks({"text": "Score 999 and a 99% response rate."})["all_pass"]
    assert not run_checks({"text": "Patients should take this; it improves survival."})["all_pass"]


def test_demo_limits_input_size():
    out = run_checks({"text": "x " * 100_000, "grounding": "https://a.org\n" * 500})
    assert "gates" in out
