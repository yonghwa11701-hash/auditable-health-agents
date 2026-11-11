"""
The run "receipt": one JSON line per step (model call, tool call, gate,
human decision) in runs/<timestamp>.jsonl. Never contains the API key.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import os
import re
from typing import Any

from . import config

_SECRETISH = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")   # Google API key shape


def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return _SECRETISH.sub("[REDACTED-KEY]", obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


class RunLog:
    def __init__(self, runs_dir: str | None = None, run_id: str | None = None,
                 mode: str = "live"):
        self.run_id = run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        d = runs_dir or config.RUNS_DIR
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"{self.run_id}.jsonl")
        self.mode = mode
        self.seq = 0
        self.records: list[dict] = []
        self._lock = threading.Lock()  # research agents write in parallel
        self.late = False      # set when the research time budget has passed
        self.deadline = None   # time.monotonic() deadline for the research stage

    def write(self, kind: str, **fields) -> dict:
        with self._lock:
            return self._write(kind, **fields)

    def _write(self, kind: str, **fields) -> dict:
        self.seq += 1
        rec = {"seq": self.seq, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "run_id": self.run_id, "mode": self.mode, "kind": kind, **fields}
        if self.late and threading.current_thread() is not threading.main_thread():
            rec["late_after_budget"] = True  # written by a research thread after the stopping rule fired
        rec = _scrub(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.records.append(rec)
        return rec

    def model_call(self, agent: str, model: str, prompt: str, interaction: dict | None,
                   output: str, extra: dict | None = None, error: str | None = None):
        from .gemini import usage, cost_usd
        u = usage(interaction or {})
        return self.write("model_call", agent=agent, model=model,
                          interaction_id=(interaction or {}).get("id"),
                          status=(interaction or {}).get("status"),
                          prompt_summary=_clip(prompt, 600),
                          output=_clip(output or "", 20000),
                          usage=u, est_cost_usd=cost_usd(model, u),
                          error=error, **(extra or {}))
