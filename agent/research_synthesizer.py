"""
STAGES 5 & 7 - Research synthesis and report generation.

The LLM receives ONLY the numbered evidence (statement + source + credibility)
and must write the report from it, citing with [n]. If the LLM is unavailable,
`template_report()` builds a plain but fully cited report directly from the
evidence, so the user still gets a usable result.
"""
from __future__ import annotations

from datetime import date

from agent.llm import LLMError

SYNTH_PROMPT = """You are the report-writing module of a research agent. Today is {today}.
Write a professional research report answering the question using ONLY the numbered evidence below.

Research question: {question}
Research intent: {intent}

SOURCES (number = citation marker):
{sources}

EVIDENCE (each line: [source number] type | statement):
{evidence}

STRICT RULES
1. Every factual claim or statistic must end with its citation marker(s), e.g. [2] or [1][3]. Use ONLY source numbers listed above.
2. Do NOT add facts, numbers, names, dates or URLs that are not in the evidence. Do not write a references list (it is added automatically).
3. Prefer higher-credibility sources; when sources disagree, say so explicitly and cite both sides.
4. Clearly separate verified facts from interpretations/forecasts (label opinions "According to ..." or "Interpretation:").
5. If the evidence is thin or missing for part of the question, say so in Limitations instead of guessing.
6. Do not repeat the same point. Length should fit the complexity - be concise but complete.

Use exactly these Markdown sections (## headings) in this order:
## Executive Summary
## Introduction
## Main Findings            (bulleted, each with citations)
## Detailed Analysis
## Evidence Comparison      (a Markdown table: Aspect | What sources say | Sources, plus a sentence on agreement/disagreement)
## Important Statistics     (bullets; write "No reliable statistics were found in the retrieved sources." if none)
## Limitations and Uncertainties
## Conclusion"""


def _fmt_sources(sources: list[dict]) -> str:
    return "\n".join(f"[{s['cite']}] {s['title'][:100]} — {s['publisher']} — {s['scores']['tier']} "
                     f"(credibility {s['scores']['authority']:.2f}, published: {s.get('published_date') or 'unknown'})"
                     for s in sources)


def _fmt_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"[{e['cite']}] {e['type']} | {e['statement']}" for e in evidence)


def synthesize(llm, question: str, analysis: dict, sources: list[dict], evidence: list[dict]) -> str:
    """Return the Markdown report body (may raise LLMError)."""
    prompt = SYNTH_PROMPT.format(
        today=date.today().isoformat(), question=question, intent=analysis.get("intent", ""),
        sources=_fmt_sources(sources), evidence=_fmt_evidence(evidence))
    msg = llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
    body = (msg.content or "").strip()
    if len(body) < 200:
        raise LLMError("The LLM returned an unusably short report.")
    return body


def template_report(question: str, sources: list[dict], evidence: list[dict], reason: str = "") -> str:
    """No-LLM fallback: a plain report assembled directly from collected evidence."""
    stats = [e for e in evidence if e["type"] == "statistic"]
    facts = [e for e in evidence if e["type"] != "interpretation"]
    interp = [e for e in evidence if e["type"] == "interpretation"]
    bullet = lambda e: f"- {e['statement']} [{e['cite']}]"
    note = f" (automatic report: {reason})" if reason else ""
    return "\n\n".join([
        f"## Executive Summary\nThis is an evidence listing assembled without AI synthesis{note}. "
        f"It contains {len(evidence)} statements drawn from {len(sources)} sources.",
        f"## Introduction\nQuestion investigated: {question}",
        "## Main Findings\n" + "\n".join(bullet(e) for e in facts[:10]),
        "## Detailed Analysis\n" + ("\n".join(bullet(e) for e in facts[10:20]) or "See main findings."),
        "## Evidence Comparison\n" + "\n".join(f"- {s['publisher']} ({s['scores']['tier']}), credibility {s['scores']['authority']:.2f} [{s['cite']}]" for s in sources),
        "## Important Statistics\n" + ("\n".join(bullet(e) for e in stats) or "No reliable statistics were found in the retrieved sources."),
        "## Limitations and Uncertainties\nNo cross-source comparison or synthesis was performed by an AI model for this report. "
        + ("Interpretations/forecasts found:\n" + "\n".join(bullet(e) for e in interp) if interp else ""),
        "## Conclusion\nReview the evidence above and the cited sources directly; no further conclusion is drawn automatically.",
    ])
