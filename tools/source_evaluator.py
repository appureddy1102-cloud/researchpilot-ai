"""
TOOL C - Source Analysis.

Transparent, rule-based scoring of every source. No LLM is involved, so the
scores are reproducible and explainable:

  relevance   - Tavily's relevance score blended with keyword overlap
  authority   - publisher type (government, university, journal, ...) by domain
  freshness   - how recent the publication date is (neutral if unknown)
  support     - did we get full page text / extracted evidence from it?

The overall score is a weighted sum. When the question needs *current*
information, freshness gets a larger weight.

Also detects duplicates (same normalised URL, near-identical title, or
identical content fingerprint).
"""
from __future__ import annotations

import difflib
import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse

from dateutil import parser as dateparser

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "ref", "ref_src"}
TWO_LEVEL_SUFFIXES = {"co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "edu.au", "gov.au", "co.in", "ac.in",
                      "gov.in", "nic.in", "res.in", "co.jp", "ac.jp", "go.jp", "com.cn", "edu.cn", "gov.cn", "co.nz", "ac.nz"}

# domain (registrable) -> (score, tier label)
KNOWN_DOMAINS: dict[str, tuple[float, str]] = {
    # scholarly / peer-reviewed
    "nature.com": (0.92, "Peer-reviewed journal / publisher"), "science.org": (0.92, "Peer-reviewed journal / publisher"),
    "thelancet.com": (0.92, "Peer-reviewed journal / publisher"), "nejm.org": (0.92, "Peer-reviewed journal / publisher"),
    "bmj.com": (0.9, "Peer-reviewed journal / publisher"), "cell.com": (0.9, "Peer-reviewed journal / publisher"),
    "springer.com": (0.85, "Academic publisher"), "sciencedirect.com": (0.85, "Academic publisher"),
    "wiley.com": (0.85, "Academic publisher"), "ieee.org": (0.88, "Professional / scholarly society"),
    "acm.org": (0.88, "Professional / scholarly society"), "plos.org": (0.85, "Peer-reviewed journal / publisher"),
    "frontiersin.org": (0.8, "Peer-reviewed journal / publisher"), "mdpi.com": (0.7, "Peer-reviewed journal / publisher"),
    "pubmed.ncbi.nlm.nih.gov": (0.92, "Government biomedical database"), "arxiv.org": (0.72, "Preprint server (not peer-reviewed)"),
    # intergovernmental / institutional
    "who.int": (0.93, "Intergovernmental organisation"), "un.org": (0.9, "Intergovernmental organisation"),
    "europa.eu": (0.92, "Intergovernmental / government"), "oecd.org": (0.9, "Intergovernmental organisation"),
    "worldbank.org": (0.88, "Intergovernmental organisation"), "imf.org": (0.88, "Intergovernmental organisation"),
    "iea.org": (0.88, "Intergovernmental organisation"), "irena.org": (0.88, "Intergovernmental organisation"),
    "ipcc.ch": (0.92, "Intergovernmental scientific body"), "nasa.gov": (0.95, "Government"),
    # reputable industry / news / consultancies
    "reuters.com": (0.78, "Reputable news agency"), "apnews.com": (0.78, "Reputable news agency"),
    "bbc.com": (0.75, "Reputable news organisation"), "bbc.co.uk": (0.75, "Reputable news organisation"),
    "nytimes.com": (0.72, "Reputable news organisation"), "ft.com": (0.74, "Reputable news organisation"),
    "economist.com": (0.74, "Reputable news organisation"), "wsj.com": (0.72, "Reputable news organisation"),
    "bloomberg.com": (0.72, "Reputable news organisation"), "technologyreview.com": (0.75, "Reputable industry publication"),
    "spectrum.ieee.org": (0.8, "Reputable industry publication"), "wired.com": (0.65, "Reputable industry publication"),
    "hbr.org": (0.7, "Reputable industry publication"), "mckinsey.com": (0.68, "Industry research / consultancy"),
    "gartner.com": (0.68, "Industry research / consultancy"), "statista.com": (0.6, "Statistics aggregator"),
    "ourworldindata.org": (0.82, "Research-based data publisher"), "ibm.com": (0.6, "Vendor / company publication"),
    "aws.amazon.com": (0.55, "Vendor / company publication"), "cloud.google.com": (0.55, "Vendor / company publication"),
    "openai.com": (0.6, "Vendor / company publication"), "anthropic.com": (0.6, "Vendor / company publication"),
    "deepmind.google": (0.65, "Vendor / company research"), "nvidia.com": (0.55, "Vendor / company publication"),
    "wikipedia.org": (0.5, "Community encyclopedia (check its citations)"),
    "britannica.com": (0.65, "Editorial encyclopedia"),
    # user-generated / low-editorial
    "medium.com": (0.28, "User-generated blog"), "quora.com": (0.2, "User-generated Q&A"), "reddit.com": (0.2, "User-generated forum"),
    "pinterest.com": (0.1, "Social media"), "facebook.com": (0.15, "Social media"), "x.com": (0.15, "Social media"),
    "twitter.com": (0.15, "Social media"), "linkedin.com": (0.25, "Social media / user posts"), "substack.com": (0.3, "User-generated newsletter"),
    "blogspot.com": (0.25, "User-generated blog"), "wordpress.com": (0.25, "User-generated blog"), "tumblr.com": (0.15, "Social media"),
    "youtube.com": (0.25, "Video platform"),
}

STOPWORDS = set("""a an the of in on at for to and or is are was were be been being what which who whom how why when where
does do did can could should would will shall may might must about between among from with without into over under
latest recent new current vs versus than then that this these those it its their there as by not no more most""".split())


# ---------------------------------------------------------------- URL helpers
def normalize_url(url: str) -> str:
    """Canonical form used for duplicate detection."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower().removeprefix("www.")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING_PARAMS))
    path = re.sub(r"/+$", "", p.path or "")
    return f"{host}{path}" + (f"?{query}" if query else "")


def registrable_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def publisher_name(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return host or "unknown"


def keywords_from(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    out: list[str] = []
    for w in words:
        if w not in STOPWORDS and w not in out:
            out.append(w)
    return out


# ------------------------------------------------------------------ scoring
def credibility(url: str) -> tuple[float, str, str]:
    """Return (score 0-1, tier label, human-readable reason)."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    reg = registrable_domain(url)
    for dom in (host, reg):
        if dom in KNOWN_DOMAINS:
            s, tier = KNOWN_DOMAINS[dom]
            return s, tier, f"{dom} is a recognised {tier.lower()}"
    # also match sub-domains of known domains (e.g. spectrum.ieee.org -> ieee.org)
    for dom, (s, tier) in KNOWN_DOMAINS.items():
        if host.endswith("." + dom):
            return s, tier, f"sub-domain of {dom} ({tier.lower()})"
    labels = host.split(".")
    if labels[-1] == "gov" or "gov" in labels[-2:] or labels[-1] == "mil" or reg in ("nic.in",) or reg.endswith(".gov.in"):
        return 0.95, "Government", "government domain"
    if labels[-1] == "edu" or "ac" in labels[-2:] or "edu" in labels[-2:]:
        return 0.9, "University / academic institution", "academic domain"
    if labels[-1] == "int":
        return 0.88, "Intergovernmental organisation", ".int domain"
    if labels[-1] == "org":
        return 0.55, "Non-profit / organisation", ".org domain (not automatically authoritative)"
    return 0.4, "General website", "no recognised authority signals - treated cautiously"


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = dateparser.parse(str(value))
        if d is None:
            return None
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, TypeError):
        return None


def freshness(date_value: str | None, now: datetime | None = None) -> tuple[float, str]:
    d = parse_date(date_value)
    if not d:
        return 0.5, "publication date unknown (neutral score)"
    now = now or datetime.now(timezone.utc)
    age_years = (now - d).days / 365.25
    if age_years < 0:
        return 0.5, "date is in the future - ignored"
    if age_years <= 1:
        return 1.0, f"published {age_years:.1f} years ago"
    if age_years <= 2:
        return 0.8, f"published {age_years:.1f} years ago"
    if age_years <= 5:
        return 0.5, f"published {age_years:.1f} years ago"
    return 0.3, f"published {age_years:.0f} years ago (may be outdated)"


def relevance(keywords: list[str], title: str, snippet: str, tavily_score: float | None) -> float:
    text = f"{title} {snippet}".lower()
    overlap = sum(1 for k in keywords if k in text) / max(len(keywords), 1)
    if tavily_score is None:
        return round(min(overlap, 1.0), 3)
    return round(0.6 * min(max(float(tavily_score), 0.0), 1.0) + 0.4 * overlap, 3)


def evaluate_source(src: dict, keywords: list[str], needs_current: bool, has_evidence: bool = False) -> dict:
    """Score one source; returns the score breakdown dict (also stored on the source)."""
    auth, tier, auth_reason = credibility(src["url"])
    fresh, fresh_reason = freshness(src.get("published_date"))
    rel = relevance(keywords, src.get("title", ""), src.get("snippet", ""), src.get("score"))
    support = 1.0 if has_evidence else (0.6 if src.get("text") else 0.3)
    w_fresh = 0.25 if needs_current else 0.10
    w_rel, w_auth, w_sup = 0.40, 0.35, 0.10
    total_w = w_rel + w_auth + w_fresh + w_sup
    overall = (w_rel * rel + w_auth * auth + w_fresh * fresh + w_sup * support) / total_w
    return {
        "overall": round(overall, 3), "relevance": rel, "authority": auth, "freshness": fresh, "support": support,
        "tier": tier, "authority_reason": auth_reason, "freshness_reason": fresh_reason,
    }


def _fingerprint(src: dict) -> str | None:
    body = (src.get("text") or src.get("snippet") or "").lower()
    body = re.sub(r"\W+", " ", body).strip()[:300]
    return hashlib.md5(body.encode()).hexdigest() if len(body) > 120 else None


def analyze_sources(sources: dict[int, dict], keywords: list[str], needs_current: bool,
                    evidence_source_ids: set[int] | None = None) -> dict:
    """Score all sources and flag duplicates. Mutates sources in place.

    Returns {"ranked": [ids best-first], "duplicates": [(dup_id, kept_id, reason)]}.
    A duplicate source gets `duplicate_of` set and is excluded from the ranking.
    """
    evidence_source_ids = evidence_source_ids or set()
    duplicates: list[tuple[int, int, str]] = []
    seen_fp: dict[str, int] = {}
    seen_titles: list[tuple[str, int]] = []
    for sid in sorted(sources):
        s = sources[sid]
        if s.get("duplicate_of"):
            continue
        reason, kept = None, None
        fp = _fingerprint(s)
        if fp and fp in seen_fp:
            reason, kept = "identical content", seen_fp[fp]
        else:
            t = re.sub(r"\W+", " ", s.get("title", "").lower()).strip()
            for other_t, other_id in seen_titles:
                if t and difflib.SequenceMatcher(None, t, other_t).ratio() >= 0.93 \
                        and registrable_domain(s["url"]) == registrable_domain(sources[other_id]["url"]):
                    reason, kept = "near-identical title on same publisher", other_id
                    break
        if reason:
            s["duplicate_of"] = kept
            duplicates.append((sid, kept, reason))
            continue
        if fp:
            seen_fp[fp] = sid
        seen_titles.append((re.sub(r"\W+", " ", s.get("title", "").lower()).strip(), sid))

    for sid, s in sources.items():
        if not s.get("duplicate_of"):
            s["scores"] = evaluate_source(s, keywords, needs_current, sid in evidence_source_ids)
    ranked = sorted((sid for sid, s in sources.items() if not s.get("duplicate_of")),
                    key=lambda i: sources[i]["scores"]["overall"], reverse=True)
    return {"ranked": ranked, "duplicates": duplicates}
