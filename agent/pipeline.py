"""
The LungPair agent pipeline (one request, bounded number of calls):

  question
   -> orchestrator (gemini-4-argon, function calling) picks + calls ranker tools
   -> code takes the tiered list from the ranker JSON (tier 4 withheld)
   -> research agent (gemini-3.8-flash + Google Search/URL context) per top pair
   -> deterministic gates + checker (gemini-3.5-flash-lite JSON yes/no)
      fail -> one retry with the failure fed back -> still failing: flag and
      strip the ungrounded citations
   -> human approves which pairs go to the next screening batch (logged)
   -> report + runs/<id>.jsonl receipt
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

from . import config, checks
import time
from .agents import run_orchestrator, research_pair, run_checker
from .runlog import RunLog
from .tools import LungPairTools


DISCLAIMER = ("Preclinical screening triage on public NCI-ALMANAC cell-line data. "
              "Not treatment advice. Not checked by a clinician.")


def _latest_ranking(tools: LungPairTools) -> tuple[dict | None, str | None]:
    for r in reversed(tools.results):
        if r["tool"] in ("rank_partners", "rank_pairs") and isinstance(r["result"], dict) \
                and "ranked" in r["result"]:
            return r["result"], r["tool"]
    return None, None


def _drugs_in_text(question: str, tools: LungPairTools) -> list[str]:
    names = [d["name"] for d in tools.list_drugs()["drugs"]]
    q = question.lower()
    return [n for n in names if re.search(rf"\b{re.escape(n.lower())}\b", q)]


def deterministic_ranking(question: str, tools: LungPairTools, log: RunLog) -> tuple[dict | None, str]:
    """Fallback path (no LLM, or orchestrator never called the ranker):
    exact drug-name match in the question, then the same ranker tools."""
    found = _drugs_in_text(question, tools)
    log.write("fallback_resolve", drugs_found=found)
    if len(found) == 1:
        return tools.call("rank_partners", {"anchor": found[0], "top": 10}), "fallback:rank_partners"
    if len(found) >= 2:
        return tools.call("rank_pairs", {"drugs": found, "top": 20}), "fallback:rank_pairs"
    return None, "fallback:no_drug_found"


def evidence_for(client, pair: dict, withheld_names: list[str], log: RunLog,
                 pubmed_check: bool, follow_redirects: bool) -> dict:
    lookup = checks.pubmed_esummary if pubmed_check else None
    feedback, attempt = None, 0
    while True:
        attempt += 1
        r = research_pair(client, pair, log, feedback=feedback, follow_redirects=follow_redirects)
        gates = [checks.citations_grounded(r["text"], r["grounding"], lookup,
                                           (pair["drug_a"], pair["drug_b"])),
                 checks.no_clinical_claim(r["text"], strict=False),
                 checks.synthetic_labelled(r["text"])]
        ck = run_checker(client, r["text"], withheld_names, log,
                         subject=f'research {pair["drug_a"]}+{pair["drug_b"]}')
        v = ck.get("verdict") or {}
        model_ok = ck["ok"] and v.get("passes") and not v.get("clinical_benefit_claim") \
            and not v.get("simulation_unlabelled")
        log.write("gates", subject=f'research {pair["drug_a"]}+{pair["drug_b"]}',
                  attempt=attempt, gates=gates, checker_ok=bool(model_ok))
        near_deadline = log.deadline is not None and time.monotonic() > log.deadline - 30
        if (checks.all_pass(gates) and model_ok) or attempt > config.RESEARCH_RETRIES or near_deadline:
            break
        fails = [g["gate"] + ": " + json.dumps(g["detail"], default=str)[:300]
                 for g in gates if not g["pass"]]
        if not model_ok:
            fails.append("reviewer: " + (v.get("reason") or ck.get("error") or "failed"))
        feedback = "; ".join(fails)
    cit = gates[0]
    shown = checks.strip_ungrounded(r["text"], cit["detail"]["not_grounded"])
    status = "passed" if (checks.all_pass(gates) and model_ok) else "flagged"
    return {"status": status, "attempts": attempt, "text": shown, "_raw_text": r["text"],
            "grounding": r["grounding"], "gates": gates, "checker": v or ck.get("error"),
            "model": r["model"]}


def run_pipeline(question: str, tools: LungPairTools, log: RunLog, client=None,
                 top_k: int | None = None, approve: Callable[[list[dict]], dict] | None = None,
                 pubmed_check: bool = True, follow_redirects: bool = True) -> dict:
    top_k = top_k or config.RESEARCH_TOP_K
    log.write("request", question=question, llm=client is not None,
              models={"orchestrator": config.ORCHESTRATOR_MODEL,
                      "research": config.RESEARCH_MODEL,
                      "checker": config.CHECKER_MODEL} if client else None)
    report: dict = {"run_id": log.run_id, "question": question, "mode": log.mode,
                    "disclaimer": DISCLAIMER}

    # 0. Deterministic refusal before any model call ----------------------
    g_pre = checks.refuse_before_model(question)
    log.write("gates", subject="question intent", gates=[g_pre])
    if not g_pre["pass"]:
        report["refused"] = g_pre["detail"]
        report["error"] = g_pre["detail"]
        log.write("report", report=report)
        return report

    # 1. Orchestrator ----------------------------------------------------
    orch = None
    if client is not None:
        try:
            orch = run_orchestrator(client, question, tools, log)
        except Exception as e:
            log.write("error", agent="orchestrator", error=f"{type(e).__name__}: {e}")
            orch = {"summary": "", "error": str(e)}
    known = [d["name"] for d in tools.list_drugs()["drugs"]]
    g_scope = checks.out_of_scope_drugs(question, known)
    report["out_of_scope"] = g_scope["unknown_drugs"]
    log.write("gates", subject="question scope", gates=[g_scope])
    ranking, source = _latest_ranking(tools)
    if ranking is None:
        ranking, source = deterministic_ranking(question, tools, log)
    report["ranking_source"] = source
    if ranking is None or "ranked" not in ranking:
        report["error"] = ((f'{", ".join(report["out_of_scope"])}: not in the NCI-ALMANAC panel, '
                            "so LungPair gives no score. " if report["out_of_scope"] else "")
                           + "No ranker drug found in the question. Name a drug from the "
                           "NCI-ALMANAC panel (e.g. gefitinib, docetaxel, carboplatin, pemetrexed).")
        report["ranker_response"] = ranking
        log.write("report", report=report)
        return report

    ranked, withheld = ranking.get("ranked", []), ranking.get("withheld", [])
    withheld_names = [f'{w["drug_a"]} + {w["drug_b"]}' for w in withheld]
    # Defence in depth: the ranker already withholds tier 4; never research or offer one.
    candidates = [c for c in ranked if c.get("tier", 4) < 4][:top_k]
    g_tier = checks.tier4_refused(ranking, candidates)
    report.update(ranked=ranked, withheld=withheld, unknown=ranking.get("unknown", []),
                  ranker_model_version=ranking.get("model_version"))

    # 2. Orchestrator summary gates -------------------------------------
    if orch is not None:
        s = orch.get("summary") or ""
        sg = [checks.no_clinical_claim(s, strict=True), checks.synthetic_labelled(s),
              checks.numbers_from_tools(s, tools.results), g_tier]
        ck = run_checker(client, s, withheld_names, log, subject="orchestrator summary") if s else \
            {"ok": False, "error": "empty summary", "verdict": None}
        v = ck.get("verdict") or {}
        ok = checks.all_pass(sg) and ck["ok"] and v.get("passes") \
            and not v.get("clinical_benefit_claim") and not v.get("recommends_withheld_pair")
        log.write("gates", subject="orchestrator summary", gates=sg, checker_ok=bool(ok))
        report["orchestrator"] = {**orch, "gates": sg, "checker": v or ck.get("error"),
                                  "status": "passed" if ok else "flagged"}
    else:
        report["orchestrator"] = None
        log.write("gates", subject="ranking", gates=[g_tier])
    report["tier4_gate"] = g_tier

    # 3. Research + checks per candidate, in parallel ----------------------
    def _one(c):
        try:
            try:
                p_a = tools.resolve_pubchem(c["drug_a"])
                p_b = tools.resolve_pubchem(c["drug_b"])
                if p_a.get("found"):
                    c["smiles_a"] = p_a.get("smiles")
                    c["cid_a"] = p_a.get("cid")
                if p_b.get("found"):
                    c["smiles_b"] = p_b.get("smiles")
                    c["cid_b"] = p_b.get("cid")
                c_a = tools.get_chembl_mechanism(c["drug_a"])
                c_b = tools.get_chembl_mechanism(c["drug_b"])
                if c_a.get("found") and c_a.get("mechanisms"):
                    c["chembl_mechs_a"] = [m["mechanism"] for m in c_a["mechanisms"]]
                if c_b.get("found") and c_b.get("mechanisms"):
                    c["chembl_mechs_b"] = [m["mechanism"] for m in c_b["mechanisms"]]
            except Exception:
                pass
            return evidence_for(client, c, withheld_names, log, pubmed_check, follow_redirects)
        except Exception as e:
            log.write("error", agent="research", pair=f'{c["drug_a"]}+{c["drug_b"]}',
                      error=f"{type(e).__name__}: {e}")
            return {"status": "error", "text": f"research failed: {e}"}

    if client is None:
        for c in candidates:
            c["evidence"] = {"status": "not_run", "text": "No Gemini call (--no-llm)."}
    elif candidates:
        t0 = time.monotonic()
        workers = max(1, min(config.RESEARCH_WORKERS, len(candidates)))
        log.write("stage", stage="research", parallel_workers=workers,
                  budget_s=config.RESEARCH_BUDGET_S, pairs=len(candidates))
        # Daemon threads: if the budget passes, the run finishes and the process can
        # exit; any late writes from a still-running thread are marked late_after_budget.
        import queue, threading
        log.deadline = t0 + config.RESEARCH_BUDGET_S
        todo, results = queue.Queue(), {}
        for i, c in enumerate(candidates):
            todo.put(i)

        def _worker():
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                if time.monotonic() > log.deadline:
                    return
                results[i] = _one(candidates[i])

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(max(0.0, log.deadline - time.monotonic()))
        unfinished = []
        for i, c in enumerate(candidates):
            if i in results:
                c["evidence"] = results[i]
            else:
                unfinished.append(f'{c["drug_a"]}+{c["drug_b"]}')
                c["evidence"] = {"status": "timeout",
                                 "text": f"research not finished within {config.RESEARCH_BUDGET_S:.0f} s budget"}
        if unfinished:
            log.late = True
            log.write("stop", rule="time_budget", budget_s=config.RESEARCH_BUDGET_S, unfinished=unfinished)
        log.write("stage", stage="research_done", seconds=round(time.monotonic() - t0, 1))
    report["candidates"] = candidates

    # 4. Human in the loop -------------------------------------------------
    decision = approve(candidates) if approve else {"approved": [], "approver": None,
                                                    "note": "no approval step run"}
    approved = [c for i, c in enumerate(candidates, 1) if i in decision.get("approved", [])
                and (client is None or (c.get("evidence") or {}).get("status") == "passed")]
    unpassed_attempted = [f'{c["drug_a"]}+{c["drug_b"]}' for i, c in enumerate(candidates, 1)
                          if i in decision.get("approved", [])
                          and client is not None
                          and (c.get("evidence") or {}).get("status") != "passed"]
    if unpassed_attempted:
        log.write("approval_rejected_unpassed", pairs=unpassed_attempted,
                  reason="Only pairs with evidence status 'passed' can be approved.")
    log.write("human_decision", approver=decision.get("approver"),
              approved=[f'{c["drug_a"]}+{c["drug_b"]}' for c in approved],
              offered=[f'{c["drug_a"]}+{c["drug_b"]}' for c in candidates],
              note=decision.get("note"))
    report["human_decision"] = {**decision,
                                "approved_pairs": [{k: c[k] for k in ("drug_a", "drug_b", "nsc_a", "nsc_b",
                                                                      "tier", "score", "interval_95")}
                                                   for c in approved]}
    log.write("report", report={k: v for k, v in report.items() if k != "ranked"})
    with open(os.path.join(os.path.dirname(log.path), f"{log.run_id}.report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    return report
