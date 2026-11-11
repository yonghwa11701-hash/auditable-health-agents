# auditable-health-agents

**Gemini multi-agent pipelines for regulated health AI that you can run without watching them, and audit afterwards.**

In regulated health work, the hard question isn't whether the agent can answer. It's whether you can prove what each agent did, which sources it used, which rules it passed, and who signed off. This repo is a working pattern for that. It's built on Google Gemini and was first used at AgentVerse 2026 (UCL, 7–8 Nov 2026) for preclinical drug-combination triage.

> Preclinical screening triage on public NCI-ALMANAC cell-line data. Not treatment advice. Not checked by a clinician.

## What it does

```
question
 -> Orchestrator   gemini-4-argon          function calling: resolve drugs, call the ranker tools
 -> Research pool  gemini-3.8-flash        parallel workers, Google Search grounding + URL context
 -> Gates          plain Python            5 deterministic checks; a model cannot override them
 -> Checker        gemini-3.5-flash-lite   structured-JSON semantic review
      fail -> one retry with the failure fed back -> still failing: FLAGGED, ungrounded citations removed
 -> Human sign-off  only passed items can be approved
 -> Receipt         runs/<timestamp>.jsonl: one line per model call, tool call, gate and decision
```

## Three design rules

1. **Code owns state and numbers; models only make judgements.** Every score, tier and 95% interval comes from the deterministic ranker. A gate (`numbers_from_tools`) fails the run if a model writes a number that isn't in the tool output.
2. **Citations must be grounded, not just plausible.** `citations_grounded` passes only if every PMID, DOI or URL the research agent writes also appears in the grounding metadata the Gemini API returned. Optionally, NCBI E-utilities is called to show whether an ungrounded PMID exists at all. Either way, an ungrounded citation never counts as grounded.
3. **A human approves, and the approval is recorded.** The pipeline can't approve its own output. Only items that passed the gates can be offered for sign-off, and the decision (who, what, when) is written to the receipt.

| Gate | Passes only if |
|---|---|
| `citations_grounded` | every citation appears in the API's grounding metadata |
| `tier4_refused` | no withheld (low-evidence) item is ranked, researched or offered for approval |
| `no_clinical_claim` | no treatment-advice or patient-benefit wording |
| `synthetic_labelled` | any mention of simulation also says "synthetic" |
| `numbers_from_tools` | every number in the summary exists in the tool JSON |

A checker "pass" never cancels a gate "fail".

## The audit trail (receipt)

Each run writes `runs/<timestamp>.jsonl` (append-only, sequence-numbered) and `runs/<timestamp>.report.json`. Each line records the model ID, interaction ID, prompt summary, output, grounding URLs, search queries, token usage and estimated cost, plus every gate result and the human decision. Key-shaped strings are redacted before anything is written, and a test checks this.

The verified live run from 8 Nov 2026 (`data/recorded_run.jsonl`) has 21 events: 8 model calls, 2 tool calls, 5 gate sets, 1 human decision and 1 final report.

## Try it (offline, no API key)

```bash
uv venv && uv pip install -r agent/requirements.txt pytest
.venv/bin/python -m pytest -q agent/tests      # 25 tests, fake Gemini client
.venv/bin/python -m agent.replay               # replays the recorded live run
```

To run live, you also need the LungPair ranker backend (see Credits; not included here) and a key in your shell. Don't put the key in a file.

```bash
export GEMINI_API_KEY=...
python -m agent.run "Which partner should a lab screen with gefitinib next?" --api inprocess
```

`--no-llm` runs ranking and sign-off only, with no Gemini call. The output and the receipt are clearly labelled `mode=no-llm`. It's the fallback when there's no network.

## Honest limits

- This is a hackathon prototype. One live end-to-end run is recorded; the rest of the testing is offline with a fake client.
- The underlying ranker transferred weakly to an independent screen (Merck O'Neil: 40% vs 19% top-10 hit rate, p = 0.087, not significant). The pipeline makes that kind of uncertainty visible. It doesn't remove it.
- It isn't a medical device and has no regulatory status.

## What I'd build next with Gemini / Antigravity

- **Unattended runs:** scheduled runs that stop at the human sign-off step and send a notification, with the receipt as the hand-off.
- **Audit trail as a first-class artifact:** signed, hash-chained receipts that a QMS reviewer (ISO 13485 / IEC 62304 style) can check without rerunning the models.
- **ADK port:** the same gate-then-checker-then-human pattern as reusable Agent Development Kit components.

See [`PATTERNS.md`](PATTERNS.md) for the build rules behind this.

## Credits

Agent layer, gates, receipts, replay and tests: Yong Hwa Hong. The LungPair ranker, data pipeline and 3D viewer are by team Backprop to the Future (AgentVerse 2026, Biotech track) and live in the team's submission repo.
