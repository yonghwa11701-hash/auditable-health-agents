"""
CLI:  python -m agent.run "Which partner should a lab screen with gefitinib next?"

  --no-llm         ranking + approval only, no Gemini call (Wi-Fi fallback;
                   output is labelled as such)
  --approve 1,3    non-interactive approval (else you are asked)
  --approver NAME  recorded in the log
  --top-k N        pairs sent to the research agent (default 5)
  --api URL|inprocess
"""
from __future__ import annotations

import argparse
import getpass
import sys

from . import config
from .gemini import make_client, MissingKey
from .pipeline import run_pipeline
from .runlog import RunLog
from .tools import LungPairTools, make_api


def _fmt_iv(iv):
    return f"[{iv[0]}, {iv[1]}]" if iv else "-"


def print_table(cands: list[dict]):
    print(f"\n{'#':>2}  {'pair':<36} {'tier':>4} {'score':>7} {'95% interval':>16}  evidence")
    for i, c in enumerate(cands, 1):
        ev = (c.get("evidence") or {}).get("status", "-")
        print(f"{i:>2}  {c['drug_a'] + ' + ' + c['drug_b']:<36} {c['tier']:>4} "
              f"{c['score']:>7} {_fmt_iv(c.get('interval_95')):>16}  {ev}")


def interactive_approve(approver: str, preset: str | None):
    def _approve(cands: list[dict]) -> dict:
        print_table(cands)
        passable = {i for i, c in enumerate(cands, 1)
                    if (c.get("evidence") or {}).get("status") in ("passed", "not_run")}
        if preset is not None:
            raw = preset
        else:
            prompt = "\nHuman sign-off: which pairs go into the next screening batch? "
            if len(passable) < len(cands):
                prompt += f"(only passed pairs allowed: {sorted(passable) or 'none'}) "
            prompt += "(e.g. 1,3 or drug names like 'docetaxel + gefitinib', or 'none'): "
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                default_cand = str(min(passable)) if passable else "none"
                print(f"\n[Non-interactive input: auto-selected candidate {default_cand} for demo]")
                raw = default_cand

        requested: set[int] = set()
        raw_clean = raw.lower().strip()
        if raw_clean not in ("", "none", "0"):
            # 1. Parse comma-separated tokens (numbers or drug names)
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            for t in tokens:
                if t.isdigit() and 1 <= int(t) <= len(cands):
                    requested.add(int(t))
                else:
                    tl = t.lower()
                    for i, c in enumerate(cands, 1):
                        p1 = f"{c['drug_a']} + {c['drug_b']}".lower()
                        p2 = f"{c['drug_b']} + {c['drug_a']}".lower()
                        if tl in (p1, p2) or (len(tl) > 3 and (tl == c['drug_a'].lower() or tl == c['drug_b'].lower())):
                            requested.add(i)

            # 2. If nothing matched token by token, test raw string directly against candidates
            if not requested:
                for i, c in enumerate(cands, 1):
                    p1 = f"{c['drug_a']} + {c['drug_b']}".lower()
                    p2 = f"{c['drug_b']} + {c['drug_a']}".lower()
                    da = c['drug_a'].lower()
                    db = c['drug_b'].lower()
                    if raw_clean in (p1, p2) or (da in raw_clean and db in raw_clean):
                        requested.add(i)
                    elif len(raw_clean) > 3 and (raw_clean == da or raw_clean == db):
                        requested.add(i)

        sorted_req = sorted(requested)
        picks = [p for p in sorted_req if p in passable]
        blocked = [p for p in sorted_req if p not in passable]
        if blocked:
            print(f"Notice: pair(s) {blocked} cannot be approved because evidence status is not 'passed'.")
        return {"approved": picks, "approver": approver, "raw_input": raw, "blocked": blocked,
                "note": "approved for the next screening batch (lab decision, not clinical)"}
    return _approve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.run")
    ap.add_argument("question", nargs="?", default="Which partner should a lab screen with gefitinib next?")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--replay", action="store_true", help="replay verified live multi-agent execution offline")
    ap.add_argument("--approve", default=None)
    ap.add_argument("--approver", default=None)
    ap.add_argument("--top-k", type=int, default=config.RESEARCH_TOP_K)
    ap.add_argument("--api", default=None)
    ap.add_argument("--no-pubmed-check", action="store_true",
                    help="skip the extra NCBI E-utilities lookup of ungrounded PMIDs")
    a = ap.parse_args(argv)

    if a.replay:
        from .replay import replay_run
        return replay_run(top_k=a.top_k, approve_preset=a.approve, approver=a.approver)

    client = None
    if not a.no_llm:
        try:
            client = make_client()
        except MissingKey as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    log = RunLog(mode="no-llm" if a.no_llm else "live")
    tools = LungPairTools(make_api(a.api))
    approver = a.approver or getpass.getuser()
    print("=" * 80)
    print("🫁 LungPair Multi-Agent Triage — Powered by Google Gemini Developer API")
    print(f"   Architecture: Orchestrator ({config.ORCHESTRATOR_MODEL}) | Research ({config.RESEARCH_MODEL}) | Checker ({config.CHECKER_MODEL})")
    print(f"   Run ID: {log.run_id} | Mode: {log.mode} | Receipt: {log.path}")
    print("=" * 80)
    from .checks import out_of_scope_drugs
    _scope = out_of_scope_drugs(a.question, [d["name"] for d in tools.list_drugs()["drugs"]])
    if _scope["unknown_drugs"]:
        print(f"\nREFUSED for: {', '.join(_scope['unknown_drugs'])} (not in the 105-drug NCI-ALMANAC "
              "panel: no score, no evidence search). Anything below covers only drugs in the panel.")
    rep = run_pipeline(a.question, tools, log, client=client, top_k=a.top_k,
                       approve=interactive_approve(approver, a.approve),
                       pubmed_check=not a.no_pubmed_check)
    if rep.get("error"):
        print("ERROR:", rep["error"]); return 1

    print(f"\nranking source: {rep['ranking_source']}  |  ranker model {rep.get('ranker_model_version')}")
    if rep.get("unknown"):
        print("not in NCI-ALMANAC panel:", rep["unknown"])
    print(f"withheld (tier 4, refused): {len(rep['withheld'])}   tier-4 gate: "
          f"{'PASS' if rep['tier4_gate']['pass'] else 'FAIL'}")
    o = rep.get("orchestrator")
    if o:
        print(f"\n[orchestrator {o.get('model')} | {o.get('status')}]\n{o.get('summary')}")
        for g in o.get("gates", []):
            if not g["pass"]:
                print("  gate FAIL:", g["gate"], g["detail"])
    for i, c in enumerate(rep.get("candidates", []), 1):
        ev = c.get("evidence") or {}
        print(f"\n--- {i}. {c['drug_a']} + {c['drug_b']}  [{ev.get('status')}]")
        print(ev.get("text", ""))
        for g in ev.get("gates", []):
            if not g["pass"]:
                print("  gate FAIL:", g["gate"], (g["detail"] or {}).get("not_grounded", g["detail"]))
    hd = rep["human_decision"]
    print(f"\napproved by {hd.get('approver')}: "
          f"{[p['drug_a'] + ' + ' + p['drug_b'] for p in hd['approved_pairs']] or 'none'}")
    print(f"\n{rep['disclaimer']}\nreceipt: {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
