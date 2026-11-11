# auditable-health-agents

[![tests](https://github.com/yonghwa11701-hash/auditable-health-agents/actions/workflows/tests.yml/badge.svg)](https://github.com/yonghwa11701-hash/auditable-health-agents/actions/workflows/tests.yml)

**For a lab scientist choosing the next drug combination to screen, this tool gives a short, cited shortlist. Every step is logged, and nothing goes ahead until a person approves it.**

It's built on Google Gemini. The same pattern fits any health-AI workflow where you need to show, afterwards, what each AI agent did, which sources it used, which rules it passed and who signed off.

> Preclinical screening triage on public NCI-ALMANAC cell-line data. Not treatment advice. Not reviewed by a clinician.

## How it works

```
question
 -> Orchestrator   gemini-4-argon          calls the ranking tools
 -> Research       gemini-3.8-flash        parallel workers with Google Search grounding
 -> Rule checks    plain Python            fixed checks that no model can override
 -> Checker        gemini-3.5-flash-lite   second review, structured JSON
      fail -> one retry with the reason -> still failing: flagged, unsupported citations removed
 -> Human sign-off  only items that passed can be approved
 -> Run log         one JSON line per model call, tool call, check and decision
```

## Three rules

1. **Numbers come from code, not from the model.** Scores, tiers and 95% intervals come from the ranker. The summary fails its check if it contains a decimal, a percentage or an integer of 10 or more that isn't in the tool output. Small counts in prose ("top 3") aren't checked.
2. **A citation counts only if the search actually returned it.** Every PMID, DOI or URL must exactly match a source in the grounding data the Gemini API sent back. This shows the source was retrieved. It doesn't prove that the source supports the sentence; the checker model and the human reviewer judge that.
3. **A person approves the result.** The pipeline never approves its own output. If no one answers, nothing is approved. If the summary fails a check, nothing can be approved. The decision (who, what, when) is written to the log.

| Check | Passes only if |
|---|---|
| `citations_grounded` | every citation exactly matches a source in the API's grounding data |
| `tier4_refused` | no withheld (low-evidence) item is ranked, researched or offered for approval |
| `no_clinical_claim` | there's no treatment-advice or patient-benefit wording |
| `synthetic_labelled` | any mention of simulation also says "synthetic" |
| `numbers_from_tools` | every decimal, percentage and integer ≥ 10 in the summary appears in the tool output |

Two more checks run on the question itself: is it in scope, and are all the named drugs in the panel? A pass from the checker model never cancels a failed rule check.

## The run log

Each run writes `runs/<timestamp>.jsonl` (append-only, numbered lines) and `runs/<timestamp>.report.json`. Each line records the model ID, interaction ID, output, grounding URLs, search queries, token use and estimated cost, plus every check result and the human decision. Anything shaped like an API key is removed before writing.

`data/recorded_run.jsonl` is a real run from 8 Nov 2026 with 21 lines: 8 model calls, 2 tool calls, 5 sets of checks, 1 human decision and 1 report.

## Try it without a key

```bash
uv venv && uv pip install -r agent/requirements.txt pytest
.venv/bin/python -m pytest -q agent/tests    # offline tests, fake Gemini client
.venv/bin/python -m agent.replay             # print the recorded live run, event by event
```

The replay prints only what the log recorded, including the citation check that failed and the stop when the time budget ran out.

### Live run

A live run also needs the LungPair ranker from the [team repo](https://github.com/leventkoymen/backprop-to-the-future-b56f8f5e-submission-repo), pinned to commit `3532b3c`, and a Gemini API key set in your shell. Never put the key in a file.

```bash
git clone https://github.com/leventkoymen/backprop-to-the-future-b56f8f5e-submission-repo lungpair
git -C lungpair checkout 3532b3c
uv pip install -r lungpair/requirements.txt
(cd lungpair/backend && python api.py)       # terminal 1: ranker on http://127.0.0.1:8000
export GEMINI_API_KEY=...                    # terminal 2
.venv/bin/python -m agent.run "Which partner should a lab screen with gefitinib next?"
```

More options are in [`agent/README.md`](agent/README.md).

## Limits

- This is a prototype built at a two-day event. One live end-to-end run is recorded; everything else is tested offline.
- The ranker underneath transferred weakly to an independent screen (Merck O'Neil: 40% vs 19% top-10 hit rate, p = 0.087, not significant). This pipeline shows that uncertainty. It doesn't remove it.
- It isn't a medical device.

## Next steps

- Scheduled runs that stop at sign-off and notify a reviewer.
- Hash-chained, signed run logs that a quality reviewer can check without rerunning the models.
- The same check-then-review-then-approve pattern as reusable Agent Development Kit components.

[`PATTERNS.md`](PATTERNS.md) lists the build rules behind this repo.

## Credits

Agent layer, checks, run logs, replay and tests: Yong Hwa Hong. The LungPair ranker, data pipeline and 3D viewer are by team Backprop to the Future (AgentVerse 2026, UCL, Biotech track).
