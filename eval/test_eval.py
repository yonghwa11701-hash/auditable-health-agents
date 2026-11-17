"""The evaluation itself is tested: the verifier cases must behave as labelled,
the golden set must pass offline, and the live-mode bookkeeping must work
(checked here with the fake Gemini client, so no key is needed)."""
from agent.tools import LungPairTools
from agent.tests.fakes import FakeApi, FakeGemini
from eval import run_eval


def test_verifier_cases_behave_as_labelled():
    v = run_eval.run_verifier()
    assert v["unexpected"] == []
    assert sum(s["bad"] for s in v["per_gate"].values()) >= 20


def test_golden_offline_all_pass():
    g = run_eval.run_golden(live=False, api=None)
    assert len(g) == 10 and all(x["pass"] for x in g), [x for x in g if not x["pass"]]


def test_trajectory_score_modes():
    s = run_eval.trajectory_score
    assert s(["list_drugs", "rank_partners"], ["rank_partners"], "IN_ORDER") == 1.0
    assert s(["rank_partners"], ["list_drugs", "rank_partners"], "IN_ORDER") == 0.0
    assert s(["rank_pairs", "get_pair"], ["get_pair", "rank_pairs"], "ANY_ORDER") == 1.0
    assert s(["rank_pairs", "get_pair"], ["get_pair", "rank_pairs"], "EXACT") == 0.0


def test_live_bookkeeping_with_fake_model(tmp_path):
    case = {"id": "t", "question": "Which partner should a lab screen with gefitinib next?",
            "expect": {"refused": False, "trajectory": ["rank_partners"], "anchor": "gefitinib"}}
    # Checker says pass, but the research cites a PMID that is not grounded:
    # the plain-Python check fails, so the two disagree and that is counted.
    r = run_eval.run_one(case, LungPairTools(FakeApi()), FakeGemini(), str(tmp_path), "eval-test")
    assert r["checks"]["trajectory"] and r["checks"]["anchor"]
    assert r["checks"]["approval_only_passed"]
    assert r["model_calls"] > 0 and r["tokens"] > 0
    assert r["checker_vs_rules_disagree"]


def test_num_runs_aggregates():
    g = run_eval.run_golden(live=False, api=None, num_runs=2)
    assert all(x["pass_fraction"] == 1.0 and len(x["runs"]) == 2 for x in g)
