"""
OFFLINE UNIT TESTS: fake Gemini client + fake ranker API. They prove the
wiring, gates and logging; they say nothing about live Gemini quality.

    .venv/bin/python -m pytest agent/tests -q
"""
import json
import os

import pytest

from agent import checks, config
from agent.gemini import grounding_urls, cited_ids, output_text
from agent.pipeline import run_pipeline
from agent.runlog import RunLog
from agent.tools import LungPairTools
from agent.tests.fakes import FakeApi, FakeGemini


@pytest.fixture
def log(tmp_path):
    return RunLog(runs_dir=str(tmp_path), run_id="test", mode="offline-test")


def approve_first(cands):
    return {"approved": [1], "approver": "tester", "note": "test"}


def run(client, log, **kw):
    return run_pipeline("Which partner should a lab screen with gefitinib next?",
                        LungPairTools(FakeApi()), log, client=client, approve=approve_first,
                        pubmed_check=False, follow_redirects=False, **kw)


# --- tools ------------------------------------------------------------------
def test_resolve_exact_only_and_osimertinib_unknown():
    t = LungPairTools(FakeApi())
    r = t.resolve_drugs(["gefitinib", "715055", "Osimertinib", "gefit"])
    assert [x["nsc"] for x in r["resolved"]] == [715055, 715055]
    assert r["unknown"] == ["Osimertinib", "gefit"]


def test_rank_partners_keeps_only_anchor_pairs():
    out = LungPairTools(FakeApi()).rank_partners("Gefitinib")
    assert all(715055 in (p["nsc_a"], p["nsc_b"]) for p in out["ranked"])
    assert [p["partner_rank"] for p in out["ranked"]] == [1, 2]
    assert len(out["withheld"]) == 1


def test_unknown_tool_and_bad_args_are_captured():
    t = LungPairTools(FakeApi())
    assert "error" in t.call("delete_everything", {})
    assert "error" in t.call("rank_pairs", {"nonsense": 1})


def test_pubchem_and_chembl_tool_declarations_and_dispatch():
    from agent.tools import TOOL_DECLARATIONS
    names = {d["name"] for d in TOOL_DECLARATIONS}
    assert "resolve_pubchem" in names
    assert "get_chembl_mechanism" in names

    t = LungPairTools(FakeApi())
    # Mocking lookups for offline test
    t.resolve_pubchem = lambda name: {"found": True, "name": name, "cid": 1234, "smiles": "CC", "source": "pubchem"}
    t.get_chembl_mechanism = lambda name: {"found": True, "name": name, "chembl_id": "CHEMBL1",
                                           "mechanisms": [{"mechanism": "EGFR inhibitor"}], "source": "chembl"}

    r_pub = t.call("resolve_pubchem", {"name": "Gefitinib"})
    assert r_pub["found"] and r_pub["cid"] == 1234 and r_pub["source"] == "pubchem"

    r_chem = t.call("get_chembl_mechanism", {"name": "Gefitinib"})
    assert r_chem["found"] and r_chem["chembl_id"] == "CHEMBL1"


# --- gates ------------------------------------------------------------------
def test_citation_gate_grounded_vs_invented():
    g = [{"url": "https://pubmed.ncbi.nlm.nih.gov/11111111/", "title": None}]
    ok = checks.citations_grounded("see PMID: 11111111", g)
    bad = checks.citations_grounded("see PMID: 11111111 and PMID 22222222 and https://x.org/a", g)
    assert ok["pass"] and not bad["pass"]
    assert set(bad["detail"]["not_grounded"]) == {"PMID:22222222", "https://x.org/a"}
    stripped = checks.strip_ungrounded("PMID 22222222 here", ["PMID:22222222"])
    assert "22222222" not in stripped


def test_citation_gate_uses_resolved_redirect():
    g = [{"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
          "resolved_url": "https://pubmed.ncbi.nlm.nih.gov/33333333/"}]
    assert checks.citations_grounded("PMID: 33333333", g)["pass"]


def test_citation_gate_pmcid():
    g = [{"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/", "title": None}]
    ok = checks.citations_grounded("evidence in PMC1234567", g)
    bad = checks.citations_grounded("evidence in PMC1234567 and PMC9999999", g)
    assert ok["pass"] and not bad["pass"]
    assert "PMC9999999" in bad["detail"]["not_grounded"]
    stripped = checks.strip_ungrounded("PMC9999999 here", ["PMC9999999"])
    assert "PMC9999999" not in stripped


def test_tier4_gate():
    ranking = {"ranked": [{"drug_a": "A", "drug_b": "B", "nsc_a": 1, "nsc_b": 2, "tier": 1}],
               "withheld": [{"drug_a": "C", "drug_b": "D", "nsc_a": 3, "nsc_b": 4, "tier": 4}]}
    assert checks.tier4_refused(ranking, ranking["ranked"])["pass"]
    assert not checks.tier4_refused(ranking, ranking["withheld"])["pass"]


def test_clinical_claim_and_synthetic_gates():
    assert not checks.no_clinical_claim("This pair improves overall survival.", strict=True)["pass"]
    assert checks.no_clinical_claim("A trial reported improved survival (PMID: 1).", strict=False)["pass"]
    assert not checks.no_clinical_claim("Patients should receive both.", strict=False)["pass"]
    assert not checks.synthetic_labelled("The simulation shows delay.")["pass"]
    assert checks.synthetic_labelled("The synthetic simulation shows delay.")["pass"]


def test_numbers_gate_catches_invented_score():
    tr = [{"tool": "rank_partners", "result": {"ranked": [{"score": 162.3, "interval_95": [127.3, 197.3]}]}}]
    assert checks.numbers_from_tools("score 162.3 [127.3, 197.3]", tr)["pass"]
    assert not checks.numbers_from_tools("score 170.4", tr)["pass"]
    # Exact match: 162.34 rounds to 162.3 but is not in allowed
    assert not checks.numbers_from_tools("score 162.34", tr)["pass"]


def test_grounding_parser_ignores_model_text_urls():
    it = {"steps": [{"type": "model_output", "content": [{"type": "text",
          "text": "see https://made-up.example/1", "annotations": [
              {"type": "url_citation", "url": "https://pubmed.ncbi.nlm.nih.gov/1234567/"}]}]},
          {"type": "url_context_result", "call_id": "u", "result": [
              {"url": "https://pubmed.ncbi.nlm.nih.gov/7654321/", "status": "success"},
              {"url": "https://evil.example/", "status": "unsafe"}]}]}
    urls = [g["url"] for g in grounding_urls(it)]
    assert urls == ["https://pubmed.ncbi.nlm.nih.gov/1234567/", "https://pubmed.ncbi.nlm.nih.gov/7654321/"]
    assert output_text(it) == "see https://made-up.example/1"
    assert cited_ids("PMID: 1234567 and doi 10.1000/xyz.")["dois"] == ["10.1000/xyz"]


# --- pipeline (fake Gemini) ----------------------------------------------------
def test_pipeline_end_to_end_offline(log):
    fake = FakeGemini()
    rep = run(fake, log)
    models = [c["model"] for c in fake.calls]
    assert config.ORCHESTRATOR_MODEL in models
    assert config.RESEARCH_MODEL in models and config.CHECKER_MODEL in models
    assert rep["ranking_source"] == "rank_partners"            # orchestrator chose the tool
    assert rep["tier4_gate"]["pass"]
    assert all(c["tier"] < 4 for c in rep["candidates"])
    assert rep["orchestrator"]["status"] == "passed"
    ev = rep["candidates"][0]["evidence"]
    # PMID 22222222 was never in grounding -> retried once, then flagged + stripped
    assert ev["status"] == "flagged" and ev["attempts"] == 1 + config.RESEARCH_RETRIES
    assert "22222222" not in ev["text"] and "11111111" in ev["text"]
    # Flagged pairs cannot be approved
    assert rep["human_decision"]["approved_pairs"] == []
    recs = [json.loads(l) for l in open(log.path)]
    kinds = {r["kind"] for r in recs}
    assert {"request", "model_call", "tool_call", "gates", "human_decision", "report"} <= kinds
    assert os.path.exists(log.path.replace(".jsonl", ".report.json"))


def test_research_passes_when_all_citations_grounded(log):
    fake = FakeGemini(research_text=lambda i: "Prior evidence: PMID: 11111111.")
    rep = run(fake, log, top_k=1)
    assert rep["candidates"][0]["evidence"]["status"] == "passed"
    assert rep["human_decision"]["approved_pairs"][0]["drug_a"] == "Docetaxel"
    assert fake.research_calls == 1                               # no retry needed


def test_retry_fixes_citation(log):
    fake = FakeGemini(research_text=lambda i: "PMID: 22222222" if i == 1 else "PMID: 11111111")
    rep = run(fake, log, top_k=1)
    ev = rep["candidates"][0]["evidence"]
    assert ev["status"] == "passed" and ev["attempts"] == 2


def test_orchestrator_invented_number_is_flagged(log):
    fake = FakeGemini(summary="Docetaxel is best with score 199.9.")
    rep = run(fake, log, top_k=1)
    assert rep["orchestrator"]["status"] == "flagged"
    assert any(g["gate"] == "numbers_from_tools" and not g["pass"] for g in rep["orchestrator"]["gates"])


def test_checker_model_cannot_override_gate(log):
    fake = FakeGemini(summary="This combination improves overall survival.")
    rep = run(fake, log, top_k=1)                                # checker fake says passes=True
    assert rep["orchestrator"]["status"] == "flagged"


def test_model_fallback_on_error(log):
    fake = FakeGemini(fail_models=[config.ORCHESTRATOR_MODEL])
    rep = run(fake, log, top_k=1)
    assert rep["orchestrator"]["model"] == config.FALLBACK_MODEL
    errs = [json.loads(l) for l in open(log.path) if '"error": "RuntimeError' in l]
    assert errs, "the failed primary call must be in the receipt"


def test_model_non_recoverable_error_does_not_fallback(log):
    from agent.agents import run_orchestrator
    fake = FakeGemini()
    def bad_request(**kw):
        raise RuntimeError("invalid argument: bad syntax")
    fake.interactions.create = bad_request
    with pytest.raises(RuntimeError, match="invalid argument"):
        run_orchestrator(fake, "Which partner?", LungPairTools(FakeApi()), log)


def test_no_llm_mode_uses_deterministic_fallback(log):
    rep = run(None, log)
    assert rep["ranking_source"] == "fallback:rank_partners"
    assert rep["orchestrator"] is None
    assert all(c["evidence"]["status"] == "not_run" for c in rep["candidates"])


def test_receipt_redacts_key_shaped_strings(log):
    log.write("x", note="AIza" + "A" * 35)
    assert "AIzaAAAA" not in open(log.path).read()


def test_research_runs_in_parallel_and_respects_time_budget(log, monkeypatch):
    """Research agents run at the same time; a slow pair hits the stopping rule."""
    import threading, time as _t
    monkeypatch.setattr("agent.tools.get_chembl_mechanism", lambda *a, **k: {"found": False})
    monkeypatch.setattr("agent.tools.resolve_pubchem", lambda *a, **k: {"found": False})
    fake = FakeGemini(research_text=lambda i: "Prior evidence: PMID: 11111111.")
    inner = fake.interactions.create
    active, peak, lock = [0], [0], threading.Lock()

    def slow_create(**kw):
        if any(t.get("type") == "google_search" for t in (kw.get("tools") or [])):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            _t.sleep(0.3)
            with lock:
                active[0] -= 1
        return inner(**kw)

    monkeypatch.setattr(fake.interactions, "create", slow_create)
    monkeypatch.setattr(config, "RESEARCH_BUDGET_S", 5.0)
    rep = run(fake, log)
    assert peak[0] >= 2                                   # more than one at once
    seqs = [json.loads(l)["seq"] for l in open(log.path)]
    assert len(seqs) == len(set(seqs))                     # log is thread-safe

    monkeypatch.setattr(config, "RESEARCH_BUDGET_S", 0.05)  # stopping rule fires
    rep = run(fake, RunLog(runs_dir=os.path.dirname(log.path), run_id="t2", mode="offline-test"))
    assert any(c["evidence"]["status"] == "timeout" for c in rep["candidates"])


def test_out_of_scope_drug_is_refused_by_name():
    from agent.checks import out_of_scope_drugs
    known = ["Gefitinib", "Docetaxel", "Carboplatin", "Paclitaxel"]
    g = out_of_scope_drugs("Should we combine pembrolizumab with gefitinib?", known)
    assert g["unknown_drugs"] == ["pembrolizumab"]
    assert out_of_scope_drugs("osimertinib plus carboplatin", known)["unknown_drugs"] == ["osimertinib"]
    assert out_of_scope_drugs("docetaxel and paclitaxel", known)["unknown_drugs"] == []


def test_interactive_approve_with_drug_names_and_numbers():
    from agent.run import interactive_approve
    cands = [
        {"drug_a": "Docetaxel", "drug_b": "Gefitinib", "tier": 1, "score": 162.3, "evidence": {"status": "passed"}},
        {"drug_a": "Mitoxantrone", "drug_b": "Gefitinib", "tier": 1, "score": 152.4, "evidence": {"status": "passed"}},
        {"drug_a": "Paclitaxel", "drug_b": "Carboplatin", "tier": 1, "score": 120.0, "evidence": {"status": "flagged"}}
    ]
    # Drug pair exact text
    ap_pair = interactive_approve("tester", "Docetaxel + Gefitinib")(cands)
    assert ap_pair["approved"] == [1]

    # Single drug name
    ap_single = interactive_approve("tester", "mitoxantrone")(cands)
    assert ap_single["approved"] == [2]

    # Comma-separated numbers
    ap_nums = interactive_approve("tester", "1, 2")(cands)
    assert ap_nums["approved"] == [1, 2]

    # Block unpassed candidates
    ap_blocked = interactive_approve("tester", "3")(cands)
    assert ap_blocked["approved"] == []
    assert ap_blocked["blocked"] == [3]

    # Empty / none
    ap_none = interactive_approve("tester", "none")(cands)
    assert ap_none["approved"] == []


def test_replay_run_offline():
    from agent.replay import replay_run
    rc = replay_run(top_k=2, approve_preset="Docetaxel + Gefitinib", delay=0.0)
    assert rc == 0

