"""
TOOL D - Evidence Extraction.

Turns source text into structured evidence records:

    {"source_id": 3, "statement": "...", "quote": "exact words from the page",
     "type": "fact" | "statistic" | "interpretation", "verified": True}

Anti-hallucination guard: every statement must come with a verbatim `quote`
and that quote is checked against the real page text. Statements whose quote
cannot be found in the source are discarded.

If the LLM is unavailable, a heuristic fallback ranks real sentences from the
text by keyword overlap - still source-grounded, just less polished.
"""
from __future__ import annotations

import re

from agent.llm import LLMError, parse_json

EXTRACT_PROMPT = """You extract evidence from ONE web source to help answer a research question.

Research question: {question}

Source title: {title}
Source text:
\"\"\"
{text}
\"\"\"

Return JSON: {{"evidence": [{{"statement": "...", "quote": "...", "type": "fact|statistic|interpretation"}}]}}
Rules:
- Max {max_items} items; only information that helps answer the question.
- "statement": one concise, self-contained sentence in your own words.
- "quote": a SHORT passage (under 200 characters) copied EXACTLY from the source text that supports the statement.
- type = "statistic" if it contains a number/percentage/date figure, "interpretation" if it is the author's opinion or forecast, else "fact".
- Do not add anything that is not stated in the source. If nothing is relevant, return {{"evidence": []}}."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.$]", " ", s.lower())).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote really appears in the text (tolerant to punctuation/spacing)."""
    q, t = _norm(quote), _norm(text)
    if len(q) < 15:
        return False
    if q in t:
        return True
    qt = q.split()
    tset = set(t.split())
    return len(qt) >= 5 and sum(1 for w in qt if w in tset) / len(qt) >= 0.9 and _ordered_overlap(qt, t)


def _ordered_overlap(qwords: list[str], text: str) -> bool:
    """At least 60% of the quote's word-pairs appear adjacent in the text."""
    pairs = [f"{a} {b}" for a, b in zip(qwords, qwords[1:])]
    return bool(pairs) and sum(1 for p in pairs if p in text) / len(pairs) >= 0.6


def extract_evidence(llm, source_id: int, source: dict, question: str, keywords: list[str],
                     max_items: int = 5) -> list[dict]:
    text = (source.get("text") or source.get("snippet") or "").strip()
    if not text:
        return []
    items: list[dict] = []
    if llm is not None:
        try:
            msg = llm.chat([{"role": "user", "content": EXTRACT_PROMPT.format(
                question=question, title=source.get("title", ""), text=text[:7000], max_items=max_items)}],
                json_mode=True, temperature=0)
            data = parse_json(msg.content)
            raw = data.get("evidence", []) if isinstance(data, dict) else []
            for r in raw[:max_items]:
                if not isinstance(r, dict):
                    continue
                stmt, quote = str(r.get("statement", "")).strip(), str(r.get("quote", "")).strip()
                if stmt and quote_in_text(quote, text):
                    typ = r.get("type") if r.get("type") in ("fact", "statistic", "interpretation") else "fact"
                    items.append({"source_id": source_id, "statement": stmt, "quote": quote, "type": typ, "verified": True})
        except LLMError:
            items = []  # fall through to heuristic
    if not items:
        items = heuristic_extract(source_id, text, keywords, max_items)
    return items


def heuristic_extract(source_id: int, text: str, keywords: list[str], max_items: int = 4) -> list[dict]:
    """No-LLM fallback: pick the real sentences that best match the keywords."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for s in sentences:
        s = s.strip()
        if not (50 <= len(s) <= 320) or s.startswith("## "):
            continue
        low = s.lower()
        score = sum(1 for k in keywords if k in low) + (1 if re.search(r"\d", s) else 0)
        if score >= 2:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"source_id": source_id, "statement": s, "quote": s[:200],
             "type": "statistic" if re.search(r"\d", s) else "fact", "verified": True, "heuristic": True}
            for _, s in scored[:max_items]]


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", s.lower())}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def dedupe_evidence(existing: list[dict], new: list[dict], max_per_publisher: int,
                    publisher_of: dict[int, str]) -> tuple[list[dict], int]:
    """Drop repeated statements and cap evidence per publisher (consolidation)."""
    kept, dropped = [], 0
    per_pub: dict[str, int] = {}
    for e in existing:
        per_pub[publisher_of.get(e["source_id"], "")] = per_pub.get(publisher_of.get(e["source_id"], ""), 0) + 1
    pool = list(existing)
    for e in new:
        pub = publisher_of.get(e["source_id"], "")
        if per_pub.get(pub, 0) >= max_per_publisher or any(jaccard(e["statement"], x["statement"]) >= 0.7 for x in pool):
            dropped += 1
            continue
        kept.append(e)
        pool.append(e)
        per_pub[pub] = per_pub.get(pub, 0) + 1
    return kept, dropped


def find_corroboration(evidence: list[dict], threshold: float = 0.35) -> dict[int, set[int]]:
    """source_id -> other source_ids that state something similar (agreement signal)."""
    out: dict[int, set[int]] = {}
    for i, a in enumerate(evidence):
        for b in evidence[i + 1:]:
            if a["source_id"] != b["source_id"] and jaccard(a["statement"], b["statement"]) >= threshold:
                out.setdefault(a["source_id"], set()).add(b["source_id"])
                out.setdefault(b["source_id"], set()).add(a["source_id"])
    return out
