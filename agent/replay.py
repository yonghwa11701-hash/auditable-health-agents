"""Replay a recorded live run.

Prints the events in a run log (JSONL) in order, exactly as recorded: model
calls, tool calls, gate results, stops and the human decision. Nothing is
added or re-decided, so anyone can inspect a real run without an API key or
network access.

    python -m agent.replay                      # the bundled run from 8 Nov 2026
    python -m agent.replay runs/<id>.jsonl      # any other run log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDED_LOG = ROOT / "data" / "recorded_run.jsonl"


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _gate_line(g: dict) -> str:
    return f"{g['gate']}: {'PASS' if g.get('pass') else 'FAIL'}"


def render_event(e: dict) -> list[str]:
    kind = e.get("kind")
    seq = f"[{e.get('seq', '?'):>2}]"
    if kind == "request":
        models = e.get("models") or {}
        lines = [f"{seq} request: {e.get('question')}"]
        if models:
            lines.append("     models: " + ", ".join(f"{k}={v}" for k, v in models.items()))
        return lines
    if kind == "model_call":
        usage = e.get("usage") or {}
        tokens = (usage.get("total_input_tokens") or 0) + (usage.get("total_output_tokens") or 0)
        head = f"{seq} model call  {e.get('agent')} ({e.get('model')})  status={e.get('status')}  tokens={tokens}"
        lines = [head]
        if e.get("pair"):
            lines.append(f"     pair: {e['pair']}")
        for fc in e.get("function_calls") or []:
            lines.append(f"     requests tool: {fc.get('name')}({json.dumps(fc.get('arguments'))})")
        if e.get("verdict"):
            lines.append(f"     verdict: {_short(json.dumps(e['verdict']))}")
        elif e.get("output"):
            lines.append(f"     output: {_short(e['output'])}")
        if e.get("grounding_urls"):
            lines.append(f"     grounding sources returned: {len(e['grounding_urls'])}")
        if e.get("error"):
            lines.append(f"     error: {e['error']}")
        return lines
    if kind == "tool_call":
        return [f"{seq} tool call   {e.get('tool')}({json.dumps(e.get('args'))})",
                f"     result: {_short(json.dumps(e.get('result_summary')))}"]
    if kind == "gates":
        gates = e.get("gates") or []
        lines = [f"{seq} checks      {e.get('subject')}"
                 + (f" (attempt {e['attempt']})" if e.get("attempt") else "")]
        lines += [f"     {_gate_line(g)}" for g in gates]
        if "checker_ok" in e:
            lines.append(f"     checker model ok: {e['checker_ok']}")
        return lines
    if kind == "stage":
        rest = {k: v for k, v in e.items() if k not in ("seq", "ts", "run_id", "mode", "kind", "stage")}
        return [f"{seq} stage       {e.get('stage')}  {json.dumps(rest)}"]
    if kind == "stop":
        return [f"{seq} stop        rule={e.get('rule')}  unfinished={e.get('unfinished')}"]
    if kind == "human_decision":
        return [f"{seq} sign-off    approver={e.get('approver')}",
                f"     offered:  {e.get('offered')}",
                f"     approved: {e.get('approved')}",
                f"     note: {e.get('note')}"]
    if kind == "report":
        rep = e.get("report") or {}
        pairs = [f"{x['drug_a']}+{x['drug_b']}" for x in (rep.get("human_decision") or {}).get("approved_pairs") or []]
        return [f"{seq} report      approved pairs: {pairs}",
                f"     {rep.get('disclaimer', '')}"]
    return [f"{seq} {kind}"]


def replay_run(path: str | Path = RECORDED_LOG, delay: float = 0.0, out=sys.stdout) -> int:
    path = Path(path)
    if not path.exists():
        print(f"Error: run log not found at {path}", file=sys.stderr)
        return 1
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events:
        print(f"Error: run log is empty: {path}", file=sys.stderr)
        return 1
    first = events[0]
    print("=" * 80, file=out)
    print(f"Replay of recorded run {first.get('run_id')}  (mode={first.get('mode')}, "
          f"started {first.get('ts')})", file=out)
    print(f"Source: {path.name}, {len(events)} events, printed as recorded", file=out)
    print("=" * 80, file=out)
    for e in events:
        for line in render_event(e):
            print(line, file=out)
        if delay:
            time.sleep(delay)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.replay")
    ap.add_argument("log", nargs="?", default=str(RECORDED_LOG), help="run log (JSONL)")
    ap.add_argument("--slow", action="store_true", help="print events with a short pause")
    a = ap.parse_args(argv)
    return replay_run(a.log, delay=0.05 if a.slow else 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
