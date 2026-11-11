"""
Every model ID and limit in one place. Each value can be overridden by an
environment variable so a model that 404s or is overloaded on the event key
can be swapped without touching code.

The API key is read ONLY from the GEMINI_API_KEY environment variable
(see agent/gemini.py). It is never read from files, printed or logged.
"""
import os


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v or default


# --- Google Gemini models (names from Google's gemini-api-dev skill) ---------
ORCHESTRATOR_MODEL = _env("LUNGPAIR_ORCH_MODEL", "gemini-4-argon")
RESEARCH_MODEL = _env("LUNGPAIR_RESEARCH_MODEL", "gemini-3.8-flash")
CHECKER_MODEL = _env("LUNGPAIR_CHECKER_MODEL", "gemini-3.5-flash-lite")
# Used once if a primary model call raises (404 / 503 / overloaded).
FALLBACK_MODEL = _env("LUNGPAIR_FALLBACK_MODEL", "gemini-3.8-flash")

# --- Duncan's ranker API ------------------------------------------------------
# "http://127.0.0.1:8000" (default) or "inprocess" to import backend/api.py
# directly through Flask's test client (no server needed; fits the model once).
LUNGPAIR_API = _env("LUNGPAIR_API", "http://127.0.0.1:8000")

# --- Limits (hard caps per request: cost and time) ---------------------------
MAX_ORCH_TURNS = int(_env("LUNGPAIR_MAX_ORCH_TURNS", "6"))
RESEARCH_TOP_K = int(_env("LUNGPAIR_RESEARCH_TOP_K", "5"))
RESEARCH_RETRIES = int(_env("LUNGPAIR_RESEARCH_RETRIES", "1"))   # bounded retry
# Research agents run in parallel, one per candidate pair (Gemini 3.8 Flash is
# the Builder Guide's model for "fast parallel subagents").
RESEARCH_WORKERS = int(_env("LUNGPAIR_RESEARCH_WORKERS", "5"))
# Stopping rule: wall-clock budget for the whole research stage. Pairs not
# finished in time are marked "timeout", never silently dropped.
RESEARCH_BUDGET_S = float(_env("LUNGPAIR_RESEARCH_BUDGET_S", "90"))
HTTP_TIMEOUT_S = float(_env("LUNGPAIR_HTTP_TIMEOUT_S", "120"))

# --- Logs ---------------------------------------------------------------------
RUNS_DIR = _env("LUNGPAIR_RUNS_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))

# Introductory price for gemini-4-argon quoted in the organisers' Builder Guide
# (USD per 1M tokens, until 30 Nov 2026). Not checked with Google. Other models:
# token counts are logged, no cost estimate is made.
PRICE_PER_M_TOKENS = {
    "gemini-4-argon": {"input": 2.0, "output": 10.0},
}
