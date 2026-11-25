"""Tests for citation validation/formatting, export and the SQLite layer."""
from database.database import Database
from utils import citations as c
from utils.export import to_markdown, to_pdf

SOURCES = [
    {"cite": 1, "title": "First [study]", "publisher": "nih.gov", "url": "https://nih.gov/a", "published_date": "2025-05-06"},
    {"cite": 2, "title": "Second", "publisher": "bbc.com", "url": "https://bbc.com/b", "published_date": None},
]


def test_group_citations_are_split():
    assert c.normalize_citations("Claim [1, 2] and [1; 2].") == "Claim [1][2] and [1][2]."


def test_invalid_citations_are_removed_never_invented():
    clean, invalid, used = c.validate_citations("A [1]. B [7]. C [2][9].", {1, 2})
    assert invalid == [7, 9] and used == [1, 2]
    assert "[7]" not in clean and "[9]" not in clean and "[1]" in clean


def test_references_contain_all_required_fields_and_only_real_data():
    refs = c.build_references(SOURCES, accessed="25 Nov 2026")
    assert "https://nih.gov/a" in refs and "nih.gov" in refs
    assert "Published: 06 May 2025" in refs and "Published: Not available" in refs
    assert refs.count("Accessed: 25 Nov 2026") == 2
    assert "First (study)" in refs  # brackets in titles cannot break Markdown links
    only = c.build_references(SOURCES, "x", only_ids={2})
    assert "bbc.com" in only and "nih.gov" not in only


def test_linkify_makes_inline_citations_clickable_and_ignores_unknown():
    out = c.linkify("Fact [1] and [2] and [5].", SOURCES)
    assert "[[1]](https://nih.gov/a)" in out and "[[2]](https://bbc.com/b)" in out and "[5]" in out and "[[5]]" not in out


def test_claim_support_check_flags_unsupported_sentences():
    evidence = [{"cite": 1, "statement": "Solar capacity grew 24 percent in 2024", "quote": "solar capacity grew 24 percent"}]
    text = "Solar capacity grew by about 24 percent during 2024 [1]. Quantum bananas dominate global energy markets completely [1]."
    r = c.check_claim_support(text, evidence)
    assert r["checked"] == 2 and r["supported"] == 1 and len(r["flagged"]) == 1 and "bananas" in r["flagged"][0]


def test_pdf_and_markdown_export():
    body = "## Executive Summary\nText with “smart quotes” and a table.\n| A | B |\n|---|---|\n| 1 | 2 |\n- bullet [[1]](https://nih.gov/a)\n"
    rec = {"report_markdown": "# Q?\n\n" + body + "\n## References\n" + c.build_references(SOURCES, "25 Nov 2026"), "sources": SOURCES}
    pdf = to_pdf(rec)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
    assert to_markdown(rec) == rec["report_markdown"]


def test_pdf_handles_unicode_and_very_long_urls():
    src = [dict(SOURCES[0], url="https://example.org/" + "a" * 300, title="Ünïcödé 日本語 title")]
    rec = {"report_markdown": "# Q\n## References\n" + c.build_references(src, "x") + "\n", "sources": src}
    assert to_pdf(rec)[:4] == b"%PDF"


def test_database_crud(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    result = {"question": "Q?", "depth": "Quick", "report_markdown": "# r", "sources": SOURCES, "evidence": [], "trace": {"queries": ["a"]}}
    rid = db.save_report(result)
    assert db.get_report(rid)["report_markdown"] == "# r"  # regression: UI reads this key
    assert db.get_report(rid)["sources"][0]["url"] == "https://nih.gov/a"
    assert db.get_report(rid)["trace"]["queries"] == ["a"]
    assert [r["id"] for r in db.list_reports()] == [rid]
    assert db.list_reports(saved_only=True) == []
    db.set_saved(rid, True)
    assert len(db.list_reports(saved_only=True)) == 1
    assert db.delete_report(rid) and db.get_report(rid) is None and not db.delete_report(rid)
