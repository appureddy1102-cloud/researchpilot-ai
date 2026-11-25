"""Tests for Tools A-D: search, page reader, source evaluation, evidence extraction."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from tools import evidence_extractor as ev
from tools import source_evaluator as se
from tools import web_search as ws
from tools.webpage_reader import extract_content, read_webpage


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = "body"
    return r


# ---------------------------------------------------------------- Tool A
def test_search_returns_real_urls_and_metadata():
    payload = {"results": [
        {"title": "A", "url": "https://a.org/x", "content": "snip", "score": 0.9, "published_date": "2025-01-01"},
        {"title": "no url", "url": "", "content": "x"},
        {"title": "bad scheme", "url": "javascript:alert(1)", "content": "x"}]}
    with patch("tools.web_search.requests.post", return_value=_resp(200, payload)) as post:
        out = ws.web_search("q", "tvly-key", max_results=3)
    assert [r["url"] for r in out] == ["https://a.org/x"]  # invalid URLs dropped
    assert out[0]["snippet"] == "snip" and out[0]["published_date"] == "2025-01-01"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer tvly-key"
    assert post.call_args.kwargs["json"]["query"] == "q"


def test_search_empty_results_returns_empty_list():
    with patch("tools.web_search.requests.post", return_value=_resp(200, {"results": []})):
        assert ws.web_search("q", "k") == []


@pytest.mark.parametrize("status,needle", [(401, "rejected"), (429, "rate limit"), (432, "limit exceeded"), (400, "rejected the request")])
def test_search_http_errors_become_clear_messages(status, needle):
    with patch("tools.web_search.requests.post", return_value=_resp(status)):
        with pytest.raises(ws.SearchError, match=needle):
            ws.web_search("q", "k")


def test_search_timeout_and_network_errors_retry_then_fail():
    with patch("tools.web_search.requests.post", side_effect=requests.Timeout()) as post:
        with pytest.raises(ws.SearchError, match="timed out"):
            ws.web_search("q", "k", retries=1)
    assert post.call_count == 2
    with patch("tools.web_search.requests.post", side_effect=requests.ConnectionError()):
        with pytest.raises(ws.SearchError, match="network"):
            ws.web_search("q", "k", retries=0)


def test_search_requires_key_and_query():
    with pytest.raises(ws.SearchError, match="missing"):
        ws.web_search("q", "")
    with pytest.raises(ws.SearchError, match="Empty"):
        ws.web_search("   ", "k")


# ---------------------------------------------------------------- Tool B
HTML = """<html><head><title>T</title><meta property="article:published_time" content="2025-03-04T10:00:00Z"></head>
<body><nav>Home About Contact menu menu menu</nav><div class="cookie-banner">Accept all cookies now please do it</div>
<article><h1>Real heading</h1><p>%s</p><p>Short.</p><div class="advert"><p>%s</p></div></article>
<footer>Copyright footer text that is long enough to otherwise be kept as a paragraph ok</footer></body></html>""" % (
    "This is the real article paragraph with plenty of useful words to be kept by the extractor. " * 4,
    "BUY NOW amazing advertisement text that must be removed from extracted output completely.")


def test_extract_content_strips_noise_and_finds_date():
    r = extract_content(HTML, "https://x.org")
    assert r["ok"] and "real article paragraph" in r["text"]
    for junk in ("cookies", "BUY NOW", "Copyright footer", "menu menu"):
        assert junk not in r["text"]
    assert r["published_date"].startswith("2025-03-04")


def test_extract_content_flags_thin_pages():
    r = extract_content("<html><body><p>tiny</p></body></html>")
    assert not r["ok"] and "too little" in r["error"]


@pytest.mark.parametrize("url", ["http://localhost/admin", "http://127.0.0.1:8000", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x.org/a"])
def test_reader_blocks_unsafe_urls(url):
    r = read_webpage(url)
    assert not r["ok"] and "Blocked" in r["error"]


def test_reader_handles_unreachable_site_gracefully():
    with patch("tools.webpage_reader._is_public_host", return_value=True), \
         patch("tools.webpage_reader.requests.get", side_effect=requests.Timeout()):
        r = read_webpage("https://example.org/a")
    assert not r["ok"] and "Timed out" in r["error"]


# ---------------------------------------------------------------- Tool C
def test_url_normalisation_detects_duplicates():
    a = se.normalize_url("https://www.Example.com/path/?utm_source=x&b=1&a=2#frag")
    b = se.normalize_url("http://example.com/path?a=2&b=1")
    assert a == b


def test_credibility_ranks_government_and_university_above_blogs():
    gov, _, _ = se.credibility("https://www.cdc.gov/page")
    edu, _, _ = se.credibility("https://web.mit.edu/x")
    ac, _, _ = se.credibility("https://www.iisc.ac.in/x")
    journal, _, _ = se.credibility("https://www.nature.com/articles/x")
    blog, _, _ = se.credibility("https://someone.medium.com/post")
    unknown, tier, _ = se.credibility("https://random-site.com/a")
    assert min(gov, edu, ac, journal) > unknown > blog
    assert "General" in tier


def test_freshness_scoring():
    assert se.freshness("2026-06-01")[0] == 1.0
    assert se.freshness("2010-01-01")[0] == 0.3
    assert se.freshness(None)[0] == 0.5
    assert se.freshness("not a date")[0] == 0.5


def test_analyze_sources_flags_duplicates_and_ranks():
    same = "Identical article body text repeated across mirrors. " * 6
    sources = {
        1: {"title": "Study on X", "url": "https://nih.gov/a", "snippet": same, "score": 0.9, "published_date": "2026-01-01"},
        2: {"title": "Study on X (mirror)", "url": "https://mirror.com/a", "snippet": same, "score": 0.8},
        3: {"title": "Random blog", "url": "https://x.blogspot.com/p", "snippet": "something about X " * 10, "score": 0.5},
    }
    res = se.analyze_sources(sources, ["study", "x"], needs_current=True)
    assert sources[2]["duplicate_of"] == 1 and res["duplicates"][0][:2] == (2, 1)
    assert res["ranked"] == [1, 3]
    assert sources[1]["scores"]["overall"] > sources[3]["scores"]["overall"]


# ---------------------------------------------------------------- Tool D
def test_quote_verification_rejects_invented_quotes():
    text = "The WHO reported that 2.1 billion people lacked safely managed drinking water in 2022."
    assert ev.quote_in_text("WHO reported that 2.1 billion people lacked safely managed drinking water", text)
    assert not ev.quote_in_text("Scientists discovered water on the moon yesterday evening", text)
    assert not ev.quote_in_text("short", text)


def test_heuristic_extract_uses_real_sentences():
    text = "Solar power capacity grew 24 percent in 2024 according to the agency report. Unrelated short. " * 2
    out = ev.heuristic_extract(1, text, ["solar", "capacity", "grew"])
    assert out and all(e["statement"] in text for e in out)


def test_dedupe_evidence_and_publisher_cap():
    existing = [{"source_id": 1, "statement": "Solar capacity grew 24 percent in 2024"}]
    new = [{"source_id": 2, "statement": "Solar capacity grew 24 percent in 2024!"},
           {"source_id": 2, "statement": "Wind power is intermittent and needs storage solutions"}]
    kept, dropped = ev.dedupe_evidence(existing, new, 6, {1: "a.org", 2: "b.org"})
    assert dropped == 1 and len(kept) == 1
    kept2, dropped2 = ev.dedupe_evidence([], [{"source_id": 1, "statement": f"{word} reports distinct finding"} for word in ["solar","wind","hydro","nuclear","geothermal"]], 2, {1: "a.org"})
    assert len(kept2) == 2 and dropped2 == 3


def test_corroboration_between_sources():
    evid = [{"source_id": 1, "statement": "Agentic AI systems act autonomously using tools"},
            {"source_id": 2, "statement": "Agentic AI systems can act autonomously with tools"},
            {"source_id": 3, "statement": "Bananas are yellow"}]
    assert ev.find_corroboration(evid) == {1: {2}, 2: {1}}
