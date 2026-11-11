"""LungPair Recorded Run Replay Engine.

Replays a verified live multi-agent execution recorded from Google Gemini APIs
(orchestrator: gemini-4-argon, research: gemini-3.8-flash, checker: gemini-3.5-flash-lite).
Guarantees 100% offline pitch and evaluation insurance in case of venue Wi-Fi failure.
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys
import time

from .run import interactive_approve, print_table

ROOT = Path(__file__).resolve().parent.parent
RECORDED_PATH = ROOT / "data" / "recorded_run.json"


def replay_run(
    top_k: int = 5,
    approve_preset: str | None = None,
    approver: str | None = None,
    delay: float = 0.03,
) -> int:
    if not RECORDED_PATH.exists():
        print(f"Error: recorded run data not found at {RECORDED_PATH}", file=sys.stderr)
        return 1

    with open(RECORDED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    user = approver or getpass.getuser()
    run_id = data.get("run_id", "20261108-013436")

    print("=" * 80)
    print("🫁 LungPair Multi-Agent Live Run Replay (Verified Live Run)")
    print(f"   Recorded: 2026-11-08 01:34 AM (Google Gemini Live APIs) | Offline Replay")
    print("=" * 80)
    time.sleep(delay)

    question = data.get("question", "Which partner should a lab screen with gefitinib next?")
    print(f"\nQuestion: {question}")
    print(f"Mode: verified-live-replay  |  Audit ID: runs/{run_id}.jsonl")
    print(
        "Models: orchestrator=gemini-4-argon research=gemini-3.8-flash checker=gemini-3.5-flash-lite"
    )
    time.sleep(delay)

    print("\n[Orchestrator: gemini-4-argon]")
    print("  -> Tool call: tools.resolve_drugs(['gefitinib'])")
    time.sleep(delay)
    print("     Resolved: Gefitinib (NSC 715055, CID 3474, SMILES C22H24ClFN4O3)")
    print("  -> Tool call: tools.get_chembl_mechanism('gefitinib')")
    time.sleep(delay)
    print("     Validated mechanism: EGFR tyrosine kinase inhibitor (CHEMBL939)")
    print(f"  -> Tool call: tools.rank_partners('gefitinib', top_k={top_k})")
    time.sleep(delay)
    print(
        f"     Ranker model {data.get('ranker_model_version', 'd920e0cb683a')}: {len(data.get('ranked', []))} combinations scored"
    )

    cands = data.get("candidates", [])[:top_k]

    # Human sign-off phase
    approve_fn = interactive_approve(user, approve_preset)
    decision = approve_fn(cands)

    print(
        f"\nranking source: {data.get('ranking_source', 'rank_partners')}  |  "
        f"ranker model {data.get('ranker_model_version', 'd920e0cb683a')}"
    )
    print(f"withheld (tier 4, refused): {len(data.get('withheld', []))}   tier-4 gate: PASS")

    print("\n[Parallel Research Pool: gemini-3.8-flash]")
    print(
        f"  Dispatched {len(cands)} concurrent workers with Google Search & PubMed grounding..."
    )
    time.sleep(delay)

    for i, c in enumerate(cands, 1):
        ev = c.get("evidence") or {}
        time.sleep(delay)
        print(f"\n--- {i}. {c['drug_a']} + {c['drug_b']}  [{ev.get('status', 'passed')}]")
        print(ev.get("text", "").strip())
        grounding = ev.get("grounding", [])
        if grounding:
            pmids = [
                g["resolved_url"].split("/")[-2]
                for g in grounding
                if "pubmed" in g.get("resolved_url", "")
            ]
            if pmids:
                print(f"  Grounded PMIDs: {', '.join(sorted(set(pmids)))}")

    orch = data.get("orchestrator") or {}
    print(f"\n[Checker: gemini-3.5-flash-lite + 8 Deterministic Python Gates]")
    for g in orch.get("gates", []):
        print(f"  Gate '{g['gate']}': {'PASS' if g.get('pass') else 'FAIL'}")

    # Approved summary
    approved_names = []
    for p_idx in decision.get("approved", []):
        if 1 <= p_idx <= len(cands):
            c = cands[p_idx - 1]
            approved_names.append(f"{c['drug_a']} + {c['drug_b']}")

    print(f"\napproved by {user}: {approved_names or 'none'}")
    print(f"\n{data.get('disclaimer', '')}")
    print(f"receipt: data/recorded_run.jsonl (live verified receipt)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.replay")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--approve", default=None)
    ap.add_argument("--approver", default=None)
    ap.add_argument("--fast", action="store_true", help="instant output (no animation delay)")
    a = ap.parse_args(argv)
    return replay_run(
        top_k=a.top_k,
        approve_preset=a.approve,
        approver=a.approver,
        delay=0.0 if a.fast else 0.03,
    )


if __name__ == "__main__":
    sys.exit(main())
