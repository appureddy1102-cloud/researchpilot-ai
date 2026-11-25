"""
Report export: Markdown and PDF.

PDF is produced with fpdf2 (pure Python, no system dependencies). The built-in
PDF fonts only cover Latin-1, so text is transliterated for the PDF; the
Markdown export keeps full Unicode. URLs in the references are clickable.
"""
from __future__ import annotations

import re
from datetime import date

from fpdf import FPDF

_LINKED_CITE = re.compile(r"\[\[(\d+)\]\]\((?:[^)]+)\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_REPL = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
         "\u00b7": "|", "\u2022": "-", "\u2192": "->", "\u2248": "~", "\u2265": ">=", "\u2264": "<=", "\u00a0": " "}


def to_markdown(record: dict) -> str:
    """Markdown export = the stored report (inline citations are clickable links)."""
    return record["report_markdown"]


def _latin1(text: str) -> str:
    for k, v in _REPL.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _clean_inline(text: str) -> str:
    text = _LINKED_CITE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    text = text.replace("`", "")
    return _latin1(text)


def to_pdf(record: dict) -> bytes:
    """Render the report to PDF bytes."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def para(text: str, size=10.5, style="", indent=0.0, h=5.6):
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(W - indent, h, text, markdown=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(99, 102, 241)
    pdf.cell(0, 6, "ResearchPilot AI - Research Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(25, 25, 35)

    in_refs = False
    sources_by_cite = {s["cite"]: s for s in record.get("sources", [])}
    for raw in record["report_markdown"].splitlines():
        line = raw.rstrip()
        if not line.strip() or set(line.strip()) <= set("|-: "):
            pdf.ln(1.5)
            continue
        if line.startswith("# "):
            pdf.ln(2)
            para(f"**{_clean_inline(line[2:])}**", 16, h=7.5)
            pdf.ln(1)
        elif line.startswith("## "):
            in_refs = line[3:].strip().lower() == "references"
            pdf.ln(3)
            pdf.set_text_color(55, 48, 163)
            para(f"**{_clean_inline(line[3:])}**", 13, h=7)
            pdf.set_text_color(25, 25, 35)
        elif line.startswith("### "):
            para(f"**{_clean_inline(line[4:])}**", 11.5, h=6.5)
        elif in_refs and re.match(r"^\d+\.\s", line):
            n = int(line.split(".", 1)[0])
            s = sources_by_cite.get(n)
            if s:  # build from the structured record so the URL is a real clickable link
                para(f"**[{n}] {_clean_inline(s['title'])}**", 10, h=5.4)
                para(f"Publisher: {_clean_inline(s['publisher'])}", 9.5, indent=5, h=5)
                pdf.set_font("Helvetica", "U", 9)
                pdf.set_text_color(37, 99, 235)
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(W - 5, 5, _latin1(s["url"]), link=s["url"], new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(25, 25, 35)
                m = re.search(r"Published: (.*?)\. Accessed: (.*?)\.$", line)
                if m:
                    para(f"Published: {m.group(1)}  |  Accessed: {m.group(2)}", 9, indent=5, h=5)
            else:
                para(_clean_inline(line), 9.5)
            pdf.ln(1.5)
        elif line.lstrip().startswith(("- ", "* ")):
            para("-  " + _clean_inline(line.lstrip()[2:]), indent=4)
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            para("-  " + "  |  ".join(_clean_inline(c) for c in cells), 9.5, indent=4)
        else:
            para(_clean_inline(line))
    pdf.set_y(-14)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 130)
    pdf.cell(0, 5, f"Exported {date.today():%d %b %Y}. Verify important claims against the cited sources.", align="C")
    return bytes(pdf.output())
