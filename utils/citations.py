"""
STAGE 6 - Citation handling.

The LLM writes the report body using numeric markers like [1]. Code - not the
LLM - is responsible for everything that must be factual:

  * markers that do not map to a real retrieved source are removed
  * the References list is built from the real source records (title,
    publisher, URL, publication date, access date) - the LLM never writes URLs
  * inline [n] markers are turned into clickable links
  * a lightweight support check flags cited sentences that share almost no
    content words with the evidence collected from the cited source
"""
from __future__ import annotations

import re
from datetime import date

from tools.evidence_extractor import _tokens
from tools.source_evaluator import parse_date

_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")
_CITE = re.compile(r"\[(\d+)\]")
_LINKED = re.compile(r"\[(\d+)\](?!\()")


def normalize_citations(text: str) -> str:
    """Turn '[1, 2]' into '[1][2]' so every marker is a single number."""
    return _GROUP.sub(lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))), text)


def validate_citations(text: str, valid_ids: set[int]) -> tuple[str, list[int], list[int]]:
    """Strip markers that do not match a real source. Returns (clean_text, invalid_ids, used_ids)."""
    text = normalize_citations(text)
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid_ids:
            return m.group(0)
        invalid.append(n)
        return ""

    clean = _CITE.sub(repl, text)
    clean = re.sub(r"[ \t]+([.,;:])", r"\1", clean)
    used = sorted({int(n) for n in _CITE.findall(clean)})
    return clean, sorted(set(invalid)), used


def format_date(value: str | None) -> str:
    d = parse_date(value)
    return d.strftime("%d %b %Y") if d else "Not available"


def build_references(sources: list[dict], accessed: str | None = None, only_ids: set[int] | None = None) -> str:
    """Markdown reference list built exclusively from real source records.

    Each source dict needs: cite (int), title, publisher, url, published_date.
    """
    accessed = accessed or date.today().strftime("%d %b %Y")
    lines = []
    for s in sorted(sources, key=lambda x: x["cite"]):
        if only_ids is not None and s["cite"] not in only_ids:
            continue
        title = s["title"].replace("[", "(").replace("]", ")").strip()
        lines.append(
            f"{s['cite']}. **[{title}]({s['url']})** — {s['publisher']}. "
            f"URL: <{s['url']}>. Published: {format_date(s.get('published_date'))}. Accessed: {accessed}."
        )
    return "\n".join(lines)


def linkify(text: str, sources: list[dict]) -> str:
    """Make inline [n] markers clickable: [n] -> [[n]](url)."""
    urls = {s["cite"]: s["url"] for s in sources}
    return _LINKED.sub(lambda m: f"[[{m.group(1)}]]({urls[int(m.group(1))]})" if int(m.group(1)) in urls else m.group(0), text)


def check_claim_support(text: str, evidence: list[dict], min_overlap: float = 0.15) -> dict:
    """Heuristic check: does each cited sentence overlap with evidence from the cited source?

    `evidence` items need: cite (int), statement, quote.
    Returns {"checked": n, "supported": n, "flagged": [sentences]}.
    This is a sanity check, not proof - the UI labels it accordingly.
    """
    by_cite: dict[int, set[str]] = {}
    for e in evidence:
        by_cite.setdefault(e["cite"], set()).update(_tokens(e["statement"] + " " + e.get("quote", "")))
    checked, supported, flagged = 0, 0, []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        cites = [int(n) for n in _CITE.findall(sent)]
        if not cites or sent.lstrip().startswith(("#", "|")):
            continue
        words = _tokens(_CITE.sub("", sent))
        if len(words) < 4:
            continue
        checked += 1
        best = max((len(words & by_cite.get(c, set())) / len(words) for c in cites), default=0)
        if best >= min_overlap:
            supported += 1
        else:
            flagged.append(sent.strip()[:220])
    return {"checked": checked, "supported": supported, "flagged": flagged}
