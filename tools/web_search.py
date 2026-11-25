"""
TOOL A - Web Search (Tavily).

Makes a real HTTPS call to the Tavily Search API and returns normalised
results. Every failure mode (bad key, quota, timeout, network, empty result)
is turned into a clear `SearchError` or an empty list - never a crash and never
fake data.
"""
from __future__ import annotations

import requests

from utils.config import redact

TAVILY_URL = "https://api.tavily.com/search"


class SearchError(Exception):
    """Search provider failed. The message is safe to show to the user."""


def web_search(query: str, api_key: str, max_results: int = 5, search_depth: str = "basic",
               topic: str = "general", timeout: int = 20, retries: int = 1) -> list[dict]:
    """Run one Tavily search.

    Returns a list of dicts: title, url, snippet, score, published_date, query.
    Returns [] when the search worked but found nothing.
    """
    query = (query or "").strip()
    if not query:
        raise SearchError("Empty search query.")
    if not api_key:
        raise SearchError("TAVILY_API_KEY is missing. Add it to your .env file or Streamlit secrets.")

    payload = {
        "query": query[:400],
        "max_results": max(1, min(int(max_results), 10)),
        "search_depth": search_depth if search_depth in ("basic", "advanced") else "basic",
        "topic": topic if topic in ("general", "news") else "general",
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = "unknown error"
    for attempt in range(retries + 1):
        try:
            resp = requests.post(TAVILY_URL, json=payload, headers=headers, timeout=timeout)
        except requests.Timeout:
            last_err = f"Search timed out after {timeout}s."
            continue
        except requests.ConnectionError:
            last_err = "Could not reach the Tavily API (network problem)."
            continue
        except requests.RequestException as e:
            raise SearchError(redact(f"Search request failed: {e}")) from None

        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                raise SearchError("Tavily returned an invalid (non-JSON) response.") from None
            return _normalise(data.get("results") or [], query)
        if resp.status_code in (401, 403):
            raise SearchError("Tavily rejected the API key (401/403). Check TAVILY_API_KEY.")
        if resp.status_code == 429:
            raise SearchError("Tavily rate limit reached (429). Wait a moment and retry.")
        if resp.status_code in (432, 433):
            raise SearchError("Tavily plan/usage limit exceeded. Check your Tavily dashboard.")
        if resp.status_code == 400:
            raise SearchError(f"Tavily rejected the request (400): {resp.text[:150]}")
        last_err = f"Tavily server error ({resp.status_code})."  # 5xx -> retry

    raise SearchError(last_err)


def _normalise(raw: list[dict], query: str) -> list[dict]:
    out = []
    for r in raw:
        url = (r.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue  # never keep a result without a real URL
        out.append({
            "title": (r.get("title") or url).strip(),
            "url": url,
            "snippet": (r.get("content") or "").strip(),
            "score": r.get("score"),
            "published_date": r.get("published_date") or None,
            "query": query,
        })
    return out


def ping(api_key: str, timeout: int = 15) -> tuple[bool, str]:
    """Connectivity/key test used by the UI."""
    try:
        web_search("test", api_key, max_results=1, timeout=timeout, retries=0)
        return True, "Tavily connected"
    except SearchError as e:
        return False, str(e)
