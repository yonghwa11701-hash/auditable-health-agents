"""
Small helpers around the google-genai Interactions API
(`client.interactions.create`). Responses are turned into plain dicts so the
same parsing code works for real SDK objects and for the offline fake client
used in agent/tests.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from typing import Any

from . import config


class MissingKey(RuntimeError):
    pass


def make_client():
    """Real Gemini client. Key from GEMINI_API_KEY or macOS keychain (lungpair-gemini); never printed."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        import subprocess, getpass
        try:
            user = os.environ.get("USER") or getpass.getuser()
            key = subprocess.check_output(
                ["security", "find-generic-password", "-s", "lungpair-gemini", "-a", user, "-w"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            key = ""
    if not key:
        raise MissingKey("GEMINI_API_KEY is not set and lungpair-gemini not found in keychain. "
                         "Run `export GEMINI_API_KEY=...` in this shell (do not put it in a file), "
                         "or use --no-llm for the ranking-only fallback.")
    from google import genai
    return genai.Client(api_key=key)


def to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    for m in ("model_dump",):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)(mode="json", by_alias=True, exclude_none=True)
            except TypeError:
                return getattr(obj, m)()
    return obj


def steps(interaction: dict) -> list[dict]:
    return interaction.get("steps") or []


def output_text(interaction: dict) -> str:
    """Text of the trailing model_output steps (same idea as SDK output_text)."""
    if interaction.get("output_text"):
        return interaction["output_text"]
    texts: list[str] = []
    for s in reversed(steps(interaction)):
        if s.get("type") != "model_output":
            if texts:
                break
            continue
        for c in reversed(s.get("content") or []):
            if c.get("type") == "text":
                texts.insert(0, c.get("text", ""))
    return "".join(texts)


def function_calls(interaction: dict) -> list[dict]:
    return [s for s in steps(interaction) if s.get("type") == "function_call"]


def grounding_urls(interaction: dict) -> list[dict]:
    """URLs that came from the server's grounding, NOT from model free text:
    url_citation annotations on model_output text, and url_context_result
    retrievals. Each item: {url, title, source}."""
    out, seen = [], set()

    def add(url, title, source):
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "title": title, "source": source})

    for s in steps(interaction):
        t = s.get("type")
        if t == "model_output":
            for c in s.get("content") or []:
                for a in c.get("annotations") or []:
                    if a.get("type") == "url_citation":
                        add(a.get("url"), a.get("title"), "url_citation")
        elif t == "url_context_result":
            for r in s.get("result") or []:
                if r.get("status") in (None, "success", "SUCCESS", "URL_RETRIEVAL_STATUS_SUCCESS"):
                    add(r.get("url"), None, "url_context_result")
    return out


def search_queries(interaction: dict) -> list[str]:
    q: list[str] = []
    for s in steps(interaction):
        if s.get("type") == "google_search_call":
            args = s.get("arguments") or {}
            q.extend(args.get("queries") or [])
    return q


def usage(interaction: dict) -> dict:
    u = interaction.get("usage") or {}
    return {k: u.get(k) for k in ("total_input_tokens", "total_output_tokens",
                                  "total_thought_tokens", "total_tool_use_tokens",
                                  "total_tokens") if u.get(k) is not None}


def cost_usd(model: str, u: dict) -> float | None:
    p = config.PRICE_PER_M_TOKENS.get(model)
    if not p or "total_input_tokens" not in u:
        return None
    out_tok = (u.get("total_output_tokens") or 0) + (u.get("total_thought_tokens") or 0)
    return round(u["total_input_tokens"] / 1e6 * p["input"] + out_tok / 1e6 * p["output"], 6)


_REDIRECT_HOSTS = ("vertexaisearch.cloud.google.com",)


def resolve_redirect(url: str, timeout: float = 5.0) -> str | None:
    """Grounding citations are often Google redirect links. Read the Location
    header (no body download) so a PubMed URL behind it can be matched."""
    if not any(h in url for h in _REDIRECT_HOSTS):
        return None

    class _NoFollow(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(_NoFollow)
    try:
        opener.open(urllib.request.Request(url, method="HEAD"), timeout=timeout)
    except urllib.error.HTTPError as e:            # 30x arrives as HTTPError
        return e.headers.get("Location")
    except Exception:
        return None
    return None


# --- citation extraction from free text --------------------------------------
PMID_RE = re.compile(r"\b(?:PubMed\s*(?:ID)?|PMID)[:\s#]*?(\d{6,9})\b", re.I)
PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})", re.I)
PMC_RE = re.compile(r"\b(PMC\d{5,9})\b", re.I)
PMC_URL_RE = re.compile(r"ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d{5,9})", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>)\]]+)", re.I)
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.I)


def cited_ids(text: str) -> dict:
    """Every identifier the model WROTE. These must be backed by grounding."""
    pmids = set(PMID_RE.findall(text)) | set(PUBMED_URL_RE.findall(text))
    pmcids = {p.upper() for p in (set(PMC_RE.findall(text)) | set(PMC_URL_RE.findall(text)))}
    urls = {u.rstrip(".,;") for u in URL_RE.findall(text)}
    dois = {d.rstrip(".,;") for d in DOI_RE.findall(text)}
    return {"pmids": sorted(pmids), "pmcids": sorted(pmcids),
            "dois": sorted(dois), "urls": sorted(urls)}
