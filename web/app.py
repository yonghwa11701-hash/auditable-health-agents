"""Public demo: inspect a recorded live run and try the rule checks.

No API key, no model calls, nothing stored. Standard library only.

    python -m web.app            # http://127.0.0.1:8080
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import checks

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "recorded_run.jsonl"
PAGE = (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")
MAX_BODY = 20_000          # bytes accepted per request
MAX_TEXT = 8_000           # characters checked per field

EVENTS = [json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
REPORT = next(e["report"] for e in EVENTS if e.get("kind") == "report")
# Tool output used by the numbers check in the playground: the ranker rows the
# recorded run returned for its candidates (scores, tiers, intervals, NSC IDs).
TOOL_OUTPUT = [{"result": {"candidates": [
    {k: c.get(k) for k in ("drug_a", "drug_b", "nsc_a", "nsc_b", "tier", "score",
                           "interval_95", "expected_precision_at_20", "rank")}
    for c in REPORT.get("candidates", [])]}}]


def run_checks(payload: dict) -> dict:
    text = str(payload.get("text", ""))[:MAX_TEXT]
    urls = [u.strip() for u in str(payload.get("grounding", ""))[:MAX_TEXT].splitlines() if u.strip()][:50]
    gates = [
        checks.no_clinical_claim(text, strict=True),
        checks.synthetic_labelled(text),
        checks.numbers_from_tools(text, TOOL_OUTPUT),
        checks.citations_grounded(text, [{"url": u} for u in urls]),
    ]
    return {"all_pass": checks.all_pass(gates), "gates": gates}


class Handler(BaseHTTPRequestHandler):
    server_version = "auditable-health-agents-demo"

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; img-src 'self' https://github.com")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/run":
            self._json(200, {"events": EVENTS, "tool_output": TOOL_OUTPUT[0]["result"]})
        elif path == "/recorded_run.jsonl":
            self._send(200, LOG_PATH.read_bytes(), "application/x-ndjson",
                       {"Content-Disposition": 'attachment; filename="recorded_run.jsonl"'})
        elif path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/check":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return self._json(413, {"error": "request too large"})
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "send JSON: {\"text\": ..., \"grounding\": ...}"})
        self._json(200, run_checks(payload))

    def log_message(self, fmt, *args):   # one line per request, no request bodies
        print(f"{self.command} {self.path.split('?', 1)[0]} -> {args[1] if len(args) > 1 else ''}")


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0" if "K_SERVICE" in os.environ else "127.0.0.1")
    print(f"serving on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
