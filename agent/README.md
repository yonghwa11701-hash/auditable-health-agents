# LungPair agent pipeline (Google Gemini)

A multi-step, multi-agent layer on top of the LungPair ranker (`backend/api.py`).
A researcher types a question; three Gemini agents and plain-Python checks turn
it into a short, cited, human-approved screening batch.

**Preclinical screening triage on public NCI-ALMANAC cell-line data. Not
treatment advice. Not checked by a clinician.**

## Models used (exact IDs, all in `agent/config.py`)

| Agent | Model ID | Gemini feature | Job |
|---|---|---|---|
| Orchestrator | `gemini-4-argon` | function calling (Interactions API) | reads the question, resolves drugs, calls the ranker tools, writes a short summary |
| Research | `gemini-3.8-flash` | Google Search grounding + URL context | mechanism, prior evidence and PubMed IDs for each top pair |
| Checker | `gemini-3.5-flash-lite` | structured JSON output | yes/no review of every model text (clinical claims, withheld pairs, unlabelled simulation) |
| Fallback | `gemini-3.8-flash` | — | used once if a primary model call errors (404/503) |

Each one can be overridden by an environment variable (`LUNGPAIR_ORCH_MODEL`,
`LUNGPAIR_RESEARCH_MODEL`, `LUNGPAIR_CHECKER_MODEL`, `LUNGPAIR_FALLBACK_MODEL`).
SDK: `google-genai` (`client.interactions.create`).

## The loop

```
question
 -> Orchestrator (Argon) calls tools: resolve_drugs / rank_partners / rank_pairs / get_pair
 -> code takes the tiered list from the ranker's JSON (tier 4 = withheld, never ranked)
 -> Research agent (3.8 Flash + Google Search) on the top 5 pairs
 -> deterministic gates + Checker (Flash-Lite JSON)
      fail -> one retry with the failure fed back -> still failing: FLAGGED,
      ungrounded citations removed from the text shown
 -> HUMAN sign-off: researcher picks which pairs go to the next screening batch
 -> report + receipt in runs/<timestamp>.jsonl
```

All scores, tiers and 95% intervals come from the ranker. The language models
never compute a number, and a gate checks that.

### Verification (plain Python, cannot be overridden by a model)

| Gate | Passes only if |
|---|---|
| `citations_grounded` | every PMID, DOI and URL the research agent wrote appears in the grounding metadata returned by the API (`url_citation` annotations, `url_context_result` URLs; Google redirect links are resolved by reading their `Location` header). Optional extra: an NCBI E-utilities lookup reports whether an ungrounded PMID exists and whether its title names a drug of the pair; that is reported, but the citation still counts as not grounded. |
| `tier4_refused` | no tier-4 (withheld) pair is ranked, researched or offered for approval |
| `no_clinical_claim` | no treatment-advice wording anywhere, and no patient-benefit wording in our own summary |
| `synthetic_labelled` | any mention of simulation also says "synthetic" |
| `numbers_from_tools` | every decimal number in the orchestrator's summary exists in the ranker's JSON |

The checker model adds a second, semantic review. A checker "pass" never
cancels a gate "fail".

### Receipt

Every model call, tool call, gate result and the human decision is one line in
`runs/<timestamp>.jsonl` (model ID, interaction ID, prompt summary, output,
grounding URLs, search queries, token usage, estimated cost for Argon at the
Builder Guide price). `runs/<timestamp>.report.json` holds the final report.
Key-shaped strings are redacted before writing.

## Run it

```bash
pip install -r backend/requirements.txt -r agent/requirements.txt
cd backend && python api.py            # terminal 1, http://127.0.0.1:8000
export GEMINI_API_KEY=...              # terminal 2; never write it into a file
python -m agent.run "Which partner should a lab screen with gefitinib next?"
```

Options: `--approve 1,3` (non-interactive sign-off), `--approver NAME`,
`--top-k 3`, `--api inprocess` (no server: imports `backend/api.py`),
`--no-pubmed-check`, and `--no-llm` (ranking + sign-off only, no Gemini call;
clearly labelled `mode=no-llm` in the output and receipt; the Wi-Fi fallback).

Good example drugs: gefitinib, docetaxel, carboplatin, pemetrexed.
Osimertinib is **not** in NCI-ALMANAC and is reported as unknown.

## Tests

```bash
python -m pytest agent/tests -q
```

These are **offline unit tests** with a fake Gemini client and a fake ranker
(`agent/tests/fakes.py`). They test the wiring, gates, retry, fallback, human
sign-off and logging. They are not Gemini results and say nothing about live
model quality.

## Limits

- Research quality depends on Google Search results; web pages are treated as
  data, and identifiers are only shown if grounding backs them.
- The tumour simulator (`tumour_simulator/`) is not wired in yet.
- No design-build-test-learn replay loop yet: approved pairs are logged, not
  replayed against held-out data.

## Cross-Disciplinary Evaluation Battery (AgentVerse Integration)

We integrated the **Cross-Disciplinary Multi-Persona Evaluator** from our research into this codebase.
Run receipts generated in `runs/` can be rigorously audited against 9 distinct professional personas (e.g. Biostatistician, ReguGate Guardian, Venture Capitalist) using strict Binary (Yes/No) evaluation batteries:

```bash
python agent/run_eval.py agent/runs/20261108-013436.jsonl
```

### Antigravity Custom Agents

This repo also includes a `.agents/` folder conforming to the **Google Antigravity IDE** standard for isolated custom agent skills. This guarantees that `regugate-guardian` and `biostatistician` skills can be independently invoked directly inside your IDE or CI/CD pipelines without monolithic context rot.
