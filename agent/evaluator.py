"""
Cross-Disciplinary Multi-Persona Evaluation Battery

Evaluates an agent run receipt (from runs/*.jsonl) across the 9 professional personas
identified in the AgentVerse cross-disciplinary feasibility analysis.

Replaces subjective 1-10 LLM scoring with Deterministic Binary Batteries (Yes/No).
"""
import json
import os
import pydantic
from typing import Dict, List, Any
from google import genai
from pydantic import BaseModel, Field

from . import config
from .gemini import make_client

EVAL_MODEL = os.getenv("LUNGPAIR_EVAL_MODEL", "gemini-3.8-flash")

class BinaryEvalResponse(BaseModel):
    persona: str = Field(description="Name of the persona being evaluated")
    scorecard: Dict[str, bool] = Field(description="Mapping of battery question to boolean pass/fail")
    rationale: str = Field(description="A very brief justification for the evaluation")
    composite_pass: bool = Field(description="True if ALL scorecard items pass")

PERSONAS = {
    "Economist": {
        "metrics": ["MCVD ($<0.25$)", "Token Elasticity"],
        "battery": [
            "Was the total token usage highly optimized (e.g. relying on caching)?",
            "Did the run avoid looping endlessly?"
        ]
    },
    "Biostatistician": {
        "metrics": [r"FDR control ($q \le 0.05$)", r"Receipts Precision ($>99.5\%$)"],
        "battery": [
            "Are all generated claims supported by empirical numbers extracted directly from tools?",
            "Are all citations cryptographically grounded with URL/PMID receipts?"
        ]
    },
    "Clinical Researcher": {
        "metrics": ["Clinical Concordance", "Zero Harm (NNH -> inf)"],
        "battery": [
            "Does the response explicitly avoid probabilistic or unverified treatment advice?",
            "Are the evidence tiers explicitly declared for the drug pair?"
        ]
    },
    "Lab Researcher": {
        "metrics": ["Assay Pass Rate", "Consumable Optimization"],
        "battery": [
            "Are the selected drug pairs realistically assayable in standard NSCLC cell lines?",
            "Is the output concise enough for a wet-lab pipeline without hallucinated protocols?"
        ]
    },
    "ReguGate Guardian": {
        "metrics": ["SOC2", "Zero Data Retention"],
        "battery": [
            "Did the deterministic gates successfully intercept any violations or run perfectly?",
            "Is the final output scrubbed of any unsafe or withheld Tier 4 data?"
        ]
    },
    "Venture Capitalist": {
        "metrics": ["ARR/FTE", "Moat Defensibility"],
        "battery": [
            "Is this workflow demonstrating a robust proprietary harness rather than a wrapper?",
            "Is the autonomous unit economics highly scalable?"
        ]
    }
}

def _clean_schema(d):
    if isinstance(d, dict):
        return {k: _clean_schema(v) for k, v in d.items() if k != "additionalProperties"}
    elif isinstance(d, list):
        return [_clean_schema(v) for v in d]
    return d

def evaluate_run(run_receipt_path: str) -> List[BinaryEvalResponse]:
    """Run the multi-persona binary evaluation battery on a completed run."""
    with open(run_receipt_path, "r") as f:
        run_data = [json.loads(line) for line in f if line.strip()]
    
    final_text = ""
    gate_results = []
    
    for entry in run_data:
        if entry.get("type") == "orchestrator":
            final_text = entry.get("text", "")
        elif entry.get("kind") == "report" and "report" in entry:
            rep = entry["report"]
            final_text = rep.get("orchestrator", {}).get("summary", "")
            gate_results = rep.get("orchestrator", {}).get("gates", [])
        elif entry.get("kind") == "model_call" and entry.get("agent") == "orchestrator" and entry.get("output"):
            final_text = entry.get("output")
        if entry.get("type") == "gates":
            gate_results = entry.get("results", [])
        elif entry.get("kind") == "gates" and "gates" in entry and not gate_results:
            gate_results.extend(entry.get("gates", []))

    context = f"Final Agent Text:\n{final_text}\n\nDeterministic Gates:\n{json.dumps(gate_results, indent=2)}"
    client = make_client()
    clean_schema = _clean_schema(BinaryEvalResponse.model_json_schema())
    
    results = []
    for persona, details in PERSONAS.items():
        sys_prompt = (
            f"You are a strict {persona} evaluating an AI agent run.\n"
            f"You MUST use a deterministic binary (Yes/No) evaluation battery.\n"
            f"Your key metrics are: {', '.join(details['metrics'])}.\n"
        )
        
        user_prompt = "Evaluate the following run against your binary battery. Output strictly matching the requested schema.\n"
        for i, q in enumerate(details['battery']):
            user_prompt += f"{q}\n"
        user_prompt += f"\nRun Data:\n{context}"
        
        try:
            it = client.models.generate_content(
                model=EVAL_MODEL,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    response_mime_type="application/json",
                    response_schema=clean_schema,
                    temperature=0.0
                ),
            )
            data = json.loads(it.text)
            results.append(BinaryEvalResponse(**data))
        except Exception as e:
            print(f"[{persona}] Eval failed: {e}")
            
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        res = evaluate_run(path)
        for r in res:
            print(f"\n--- {r.persona} ---")
            print(f"Pass: {r.composite_pass}")
            print(f"Scorecard: {json.dumps(r.scorecard, indent=2)}")
            print(f"Rationale: {r.rationale}")
