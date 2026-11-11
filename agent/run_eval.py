#!/usr/bin/env python3
import os
import sys
import json
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agent.evaluator import evaluate_run

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Persona Evaluation on an Agent Run")
    parser.add_argument("receipt", help="Path to the JSONL run receipt (e.g. agent/runs/X.jsonl)")
    args = parser.parse_args()

    print(f"Loading run receipt: {args.receipt}...")
    results = evaluate_run(args.receipt)
    
    total = len(results)
    passed = sum(1 for r in results if r.composite_pass)
    
    print("\n" + "="*50)
    print("CROSS-DISCIPLINARY MULTI-PERSONA SCORECARD")
    print("="*50)
    
    for r in results:
        print(f"\n[{r.persona}]")
        print(f"Pass: {'[YES]' if r.composite_pass else '[NO]'}")
        for q, ok in r.scorecard.items():
            print(f"  - {q}: {'YES' if ok else 'NO'}")
        print(f"Rationale: {r.rationale}")
        
    print("\n" + "="*50)
    print(f"FINAL CONSENSUS: {passed}/{total} personas approved this run.")

if __name__ == "__main__":
    main()
