"""
Repeatable evaluation for the LungPair agent layer.

Three parts, each with a fixed file of cases:

  verifier   eval/verifier_cases.jsonl  Known-good and known-bad texts, one
             per case, run through each plain-Python check. Reports how many
             bad cases each check catches and how many good cases it wrongly
             flags. Cases marked "known_limit" are misses or false alarms we
             already know about; they are reported, not hidden.
  golden     eval/golden.jsonl          About 10 fixed questions with the
             expected behaviour: refuse or not, drugs out of scope, which
             ranker tools are called and in what order (scored EXACT /
             IN_ORDER / ANY_ORDER, as in ADK's tool_trajectory_avg_score;
             here tool names only, with the anchor drug checked separately,
             whereas ADK also compares arguments unless told to ignore them), and that only passed pairs can be approved.
  live       same golden questions, run against real Gemini and the real
             ranker. Adds tokens, cost, time, research status and how often
             the checker model and the plain-Python checks disagree. Saves
             a results file and compares it with the previous one (drift).

    python -m eval.run_eval                       # verifier + golden, offline (no key)
    python -m eval.run_eval --live --api inprocess --save   # needs GEMINI_API_KEY + ranker
    python -m eval.run_eval --live --api inprocess --num-runs 3 --save   # variance

Offline mode uses the fake ranker in agent/tests/fakes.py and no model: it
tests the checks and the code path, not Gemini's quality.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import tempfile
import time

from agent import checks, config
from agent.pipeline import run_pipeline
from agent.runlog import RunLog
from agent.tools import LungPairTools

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Tool output the number cases are checked against (same shape as the ranker).
RANKING = {"ranked": [{"drug_a": "Docetaxel", "drug_b": "Gefitinib", "nsc_a": 628503, "nsc_b": 715055,
                       "tier": 1, "score": 162.3, "interval_95": [127.3, 197.3],
                       "expected_precision_at_20": 0.89}],
           "withheld": [{"drug_a": "Gefitinib", "drug_b": "NSC 999999", "nsc_a": 715055,
                         "nsc_b": 999999, "tier": 4, "ranked": False}]}
PANEL = ["Gefitinib", "Docetaxel", "Carboplatin", "Pemetrexed", "Paclitaxel", "Erlotinib"]


def _load(name: str) -> list[dict]:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
def _flagged(case: dict) -> bool:
    """True if the check under test flags the case (says 'bad')."""
    g, t = case["gate"], case.get("text", "")
    if g == "citations_grounded":
        return not checks.citations_grounded(t, case["grounding"])["pass"]
    if g == "numbers_from_tools":
        return not checks.numbers_from_tools(t, [{"tool": "rank_partners", "result": RANKING}])["pass"]
    if g == "no_clinical_claim":
        return not checks.no_clinical_claim(t, strict=True)["pass"]
    if g == "synthetic_labelled":
        return not checks.synthetic_labelled(t)["pass"]
    if g == "tier4_refused":
        cands = RANKING["ranked"] if case["candidates"] == "RANKED" else RANKING["withheld"]
        return not checks.tier4_refused(RANKING, cands)["pass"]
    if g == "refuse_before_model":
        return not checks.refuse_before_model(t)["pass"]
    if g == "out_of_scope_drugs":
        return bool(checks.out_of_scope_drugs(t, PANEL)["unknown_drugs"])
    raise ValueError(f"unknown gate {g}")


def run_verifier() -> dict:
    per_gate: dict[str, dict] = {}
    rows, unexpected = [], []
    for c in _load("verifier_cases.jsonl"):
        flagged = _flagged(c)
        correct = flagged == (c["truth"] == "bad")
        kind = ("caught" if flagged else "missed") if c["truth"] == "bad" else \
               ("false_alarm" if flagged else "passed")
        s = per_gate.setdefault(c["gate"], {"bad": 0, "caught": 0, "good": 0, "false_alarm": 0})
        s["bad" if c["truth"] == "bad" else "good"] += 1
        if kind in ("caught", "false_alarm"):
            s[kind] += 1
        known = bool(c.get("known_limit"))
        # A known limit that now behaves correctly is also "unexpected": update the case file.
        if correct == known:
            unexpected.append(c["id"])
        rows.append({"id": c["id"], "gate": c["gate"], "result": kind,
                     "known_limit": c.get("known_limit")})
    return {"per_gate": per_gate, "cases": rows, "unexpected": unexpected}


# --------------------------------------------------------------------------- #
def trajectory_score(actual: list[str], expected: list[str], match: str) -> float:
    if match == "EXACT":
        return float(actual == expected)
    if match == "ANY_ORDER":
        return float(all(e in actual for e in expected))
    it = iter(actual)                                   # IN_ORDER: expected is a subsequence
    return float(all(e in it for e in expected))


def _approve_all(cands):
    return {"approved": list(range(1, len(cands) + 1)), "approver": "eval-harness",
            "note": "eval: tries to approve every offered pair; only passed pairs may go through"}


def run_one(case: dict, tools: LungPairTools, client, runs_dir: str, mode: str) -> dict:
    log = RunLog(runs_dir=runs_dir, run_id=f'{case["id"]}-{int(time.time())}', mode=mode)
    t0 = time.monotonic()
    rep = run_pipeline(case["question"], tools, log, client=client, approve=_approve_all,
                       pubmed_check=False, follow_redirects=client is not None)
    secs = round(time.monotonic() - t0, 1)
    e, res = case["expect"], {}
    refused = bool(rep.get("refused"))
    res["refused"] = refused == e["refused"]
    if "out_of_scope" in e:
        res["out_of_scope"] = sorted(rep.get("out_of_scope") or []) == sorted(e["out_of_scope"])
    if e.get("no_score"):
        res["no_score"] = not rep.get("candidates")
    if "did_you_mean" in e:
        known = [d["name"] for d in tools.list_drugs()["drugs"]]
        res["did_you_mean"] = checks.out_of_scope_drugs(case["question"], known)["did_you_mean"] == e["did_you_mean"]
    traj = [r["tool"] for r in tools.results]
    if "trajectory" in e:
        res["trajectory"] = trajectory_score(traj, e["trajectory"], e.get("match", "IN_ORDER")) == 1.0
    if "anchor" in e:
        anchors = [str(r["args"].get("anchor", "")).lower() for r in tools.results if r["tool"] == "rank_partners"]
        res["anchor"] = e["anchor"] in anchors
    # Invariant on every run: nothing is approved unless its evidence passed.
    passed = {(c["drug_a"], c["drug_b"]) for c in rep.get("candidates", [])
              if client is None or (c.get("evidence") or {}).get("status") == "passed"}
    approved = {(p["drug_a"], p["drug_b"]) for p in (rep.get("human_decision") or {}).get("approved_pairs", [])}
    res["approval_only_passed"] = approved <= passed
    # Tier 4 never offered.
    if rep.get("tier4_gate"):
        res["tier4_refused"] = rep["tier4_gate"]["pass"]

    calls = [r for r in log.records if r["kind"] == "model_call"]
    tokens = sum((r.get("usage") or {}).get("total_input_tokens", 0) +
                 (r.get("usage") or {}).get("total_output_tokens", 0) for r in calls)
    cost = sum(r.get("est_cost_usd") or 0 for r in calls)
    # Where the checker model and the plain-Python checks disagree on research text.
    disagree = [r["subject"] for r in log.records
                if r["kind"] == "gates" and str(r.get("subject", "")).startswith("research")
                and "checker_ok" in r and checks.all_pass(r["gates"]) != r["checker_ok"]]
    return {"id": case["id"], "question": case["question"], "pass": all(res.values()),
            "checks": res, "trajectory": traj, "tool_calls": len(traj), "ranking_source": rep.get("ranking_source"),
            "orchestrator_status": (rep.get("orchestrator") or {}).get("status"),
            "evidence_status": [(c.get("evidence") or {}).get("status") for c in rep.get("candidates", [])],
            "model_calls": len(calls), "tokens": tokens, "est_cost_usd": round(cost, 4),
            "seconds": secs, "checker_vs_rules_disagree": disagree, "receipt": log.path}


def run_golden(live: bool, api: str | None, num_runs: int = 1) -> list[dict]:
    if live:
        from agent.gemini import make_client
        from agent.tools import make_api
        client, new_api, mode = make_client(), (lambda: make_api(api)), "eval-live"
    else:
        from agent.tests.fakes import FakeApi
        client, new_api, mode = None, FakeApi, "eval-offline"
    runs_dir = config.RUNS_DIR if live else tempfile.mkdtemp(prefix="eval-runs-")
    shared = new_api() if live else None                # the real ranker fits its model once
    out = []
    for c in _load("golden.jsonl"):
        runs = [run_one(c, LungPairTools(shared or new_api()), client, runs_dir, mode)
                for _ in range(num_runs)]
        r = dict(runs[0])
        if num_runs > 1:                                 # models vary run to run
            r["runs"] = runs
            r["pass_fraction"] = round(sum(x["pass"] for x in runs) / num_runs, 3)
            r["pass"] = all(x["pass"] for x in runs)
            for k in ("tokens", "model_calls", "tool_calls", "seconds", "est_cost_usd"):
                r[k] = round(sum(x[k] for x in runs) / num_runs, 4)   # mean per run
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
def _summary(golden: list[dict]) -> dict:
    n = len(golden) or 1
    return {"questions": len(golden), "pass_rate": round(sum(g["pass"] for g in golden) / n, 3),
            "trajectory_ok": sum(g["checks"].get("trajectory", True) for g in golden),
            "tool_calls": sum(g["tool_calls"] for g in golden),
            "model_calls": sum(g["model_calls"] for g in golden),
            "tokens": sum(g["tokens"] for g in golden),
            "est_cost_usd": round(sum(g["est_cost_usd"] for g in golden), 4),
            "seconds": round(sum(g["seconds"] for g in golden), 1),
            "checker_vs_rules_disagreements": sum(len(g["checker_vs_rules_disagree"]) for g in golden)}


def _previous(mode: str) -> dict | None:
    files = sorted(glob.glob(os.path.join(RESULTS, f"*-{mode}.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run_eval")
    ap.add_argument("--live", action="store_true", help="real Gemini + real ranker (needs GEMINI_API_KEY)")
    ap.add_argument("--api", default=None, help="ranker: http://127.0.0.1:8000 (default) or inprocess")
    ap.add_argument("--num-runs", type=int, default=1,
                    help="repeat each question N times (live: shows run-to-run variance)")
    ap.add_argument("--save", action="store_true", help="write eval/results/<time>-<mode>.json and compare with the last one")
    a = ap.parse_args(argv)
    mode = "live" if a.live else "offline"

    v = run_verifier()
    print("CHECKS ON KNOWN-GOOD / KNOWN-BAD TEXT")
    print(f'{"check":24} {"bad caught":>11} {"good flagged":>13}')
    for g, s in v["per_gate"].items():
        print(f'{g:24} {s["caught"]:>5} / {s["bad"]:<4} {s["false_alarm"]:>6} / {s["good"]:<4}')
    print("known limits (reported, not hidden):")
    for c in v["cases"]:
        if c["known_limit"]:
            print(f'  - {c["id"]} [{c["result"]}]: {c["known_limit"]}')

    golden = run_golden(a.live, a.api, a.num_runs)
    print(f"\nGOLDEN QUESTIONS ({mode}{'' if a.live else ': fake ranker, no model'})")
    for g in golden:
        bad = [k for k, ok in g["checks"].items() if not ok]
        frac = f' ({g["pass_fraction"]:.0%} of {a.num_runs} runs)' if "pass_fraction" in g else ""
        print(f'  {"PASS" if g["pass"] else "FAIL"}{frac}  {g["id"]:28} tools={g["trajectory"] or "-"}'
              f'{"  failed: " + ", ".join(bad) if bad else ""}')
    s = _summary(golden)
    print("\nsummary:", json.dumps(s))

    if a.save:
        prev = _previous(mode)
        os.makedirs(RESULTS, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = {"when_utc": stamp, "mode": mode,
               "models": {"orchestrator": config.ORCHESTRATOR_MODEL, "research": config.RESEARCH_MODEL,
                          "checker": config.CHECKER_MODEL} if a.live else None,
               "num_runs": a.num_runs, "summary": s, "verifier": v, "golden": golden}
        path = os.path.join(RESULTS, f"{stamp}-{mode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
        print("saved:", path)
        if prev:
            ps = prev["summary"]
            print("drift vs", prev["when_utc"], {k: (ps.get(k), s[k]) for k in s if ps.get(k) != s[k]} or "no change")
            if prev.get("models") != out["models"]:
                print("model versions changed:", prev.get("models"), "->", out["models"])

    if v["unexpected"]:
        print("\nUNEXPECTED check results (fix the check or update the case file):", v["unexpected"])
    failed = [g["id"] for g in golden if not g["pass"]]
    if failed:
        print("golden failures:", failed)
    # Live runs report; they do not fail the build on model variance.
    return 1 if v["unexpected"] or (failed and not a.live) else 0


if __name__ == "__main__":
    sys.exit(main())
