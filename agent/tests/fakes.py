"""
OFFLINE TEST DOUBLES. Nothing here is Gemini output. FakeGemini returns
hand-written, Interactions-API-shaped dicts so the wiring, gates and logging
can be tested without a key or network. Never use these in a demo.
"""
import json

DRUGS = [{"nsc": 715055, "name": "Gefitinib"}, {"nsc": 628503, "name": "Docetaxel"},
         {"nsc": 241240, "name": "Carboplatin"}, {"nsc": 698037, "name": "Pemetrexed"}]


class FakeApi:
    """Stands in for Duncan's API. Returns fixed, API-shaped JSON."""
    def __init__(self):
        self.calls = []

    def get(self, path):
        self.calls.append(("GET", path))
        if path == "/drugs":
            return {"count": len(DRUGS), "drugs": DRUGS}
        return {"http_status": 404, "error": "not in fake"}

    def post(self, path, body):
        self.calls.append(("POST", path, body))
        r = lambda a, b, na, nb, t, s: {"drug_a": a, "drug_b": b, "nsc_a": na, "nsc_b": nb,  # noqa
                                         "tier": t, "evidence": "e", "ranked": True, "score": s,
                                         "interval_95": [round(s - 35, 1), round(s + 35, 1)],
                                         "expected_precision_at_20": 0.89}
        return {"model_version": "fake", "unknown": [],
                "ranked": [r("Docetaxel", "Gefitinib", 628503, 715055, 1, 162.3),
                           r("Pemetrexed", "Gefitinib", 698037, 715055, 2, 40.5),
                           r("Docetaxel", "Carboplatin", 241240, 628503, 1, 30.0)],
                "withheld": [{"drug_a": "Gefitinib", "drug_b": "NSC 999999", "nsc_a": 715055,
                              "nsc_b": 999999, "tier": 4, "ranked": False}],
                "note": "Tier 4 pairs are deliberately not ranked."}


def _text(t, annotations=None):
    c = {"type": "text", "text": t}
    if annotations:
        c["annotations"] = annotations
    return {"type": "model_output", "content": [c]}


class _Interactions:
    def __init__(self, owner):
        self.o = owner

    def create(self, **kw):
        self.o.calls.append(kw)
        model = kw.get("model")
        if model in self.o.fail_models:
            raise RuntimeError(f"503 fake overload for {model}")
        tools = kw.get("tools") or []
        n = len(self.o.calls)
        if kw.get("response_format"):                       # checker
            return {"id": f"fake-{n}", "status": "completed",
                    "steps": [_text(json.dumps(self.o.checker_verdict))],
                    "usage": {"total_input_tokens": 10, "total_output_tokens": 5}}
        if any(t.get("type") == "google_search" for t in tools):   # research
            self.o.research_calls += 1
            return {"id": f"fake-{n}", "status": "completed",
                    "steps": [{"type": "google_search_call", "id": "s1",
                               "arguments": {"queries": ["docetaxel gefitinib NSCLC"]}},
                              {"type": "google_search_result", "call_id": "s1", "result": []},
                              _text(self.o.research_text(self.o.research_calls),
                                    [{"type": "url_citation", "url": "https://pubmed.ncbi.nlm.nih.gov/11111111/",
                                      "title": "pubmed.ncbi.nlm.nih.gov", "start_index": 0, "end_index": 10}])],
                    "usage": {"total_input_tokens": 100, "total_output_tokens": 50}}
        if isinstance(kw.get("input"), str):                # orchestrator turn 1
            return {"id": f"fake-{n}", "status": "requires_action",
                    "steps": [{"type": "function_call", "id": "c1", "name": "rank_partners",
                               "arguments": {"anchor": "gefitinib", "top": 10}}],
                    "usage": {"total_input_tokens": 1000, "total_output_tokens": 20}}
        return {"id": f"fake-{n}", "status": "completed",           # orchestrator turn 2
                "steps": [_text(self.o.summary)],
                "usage": {"total_input_tokens": 2000, "total_output_tokens": 100}}


class FakeGemini:
    def __init__(self, summary=None, research_text=None, checker_verdict=None, fail_models=()):
        self.calls = []
        self.research_calls = 0
        self.fail_models = set(fail_models)
        self.summary = summary or ("Top partner for gefitinib: docetaxel (tier 1, score 162.3, "
                                   "95% interval 127.3 to 197.3). One tier-4 pair was withheld.")
        self.research_text = research_text or (lambda i: (
            "Mechanism: microtubule + EGFR. Prior evidence: PMID: 11111111; PMID: 22222222."))
        self.checker_verdict = checker_verdict or {
            "clinical_benefit_claim": False, "recommends_withheld_pair": False,
            "unsupported_specific_claims": [], "simulation_unlabelled": False,
            "passes": True, "reason": "ok"}
        self.interactions = _Interactions(self)
