"""End-to-end agent tests using a scripted LLM + fake search (offline, deterministic)."""
import json
import re

import pytest

from agent.research_agent import ResearchAgent, STAGES, validate_question
from agent.research_planner import analyze_and_plan, fallback_plan
from agent.state import ResearchState
from agent.tool_selector import ToolExecutor
from tools.web_search import SearchError
from tests.conftest import FakeLLM, fake_reader, fake_search

REAL_URLS = {"https://mit.edu/agentic", "https://www.example-blog.com/genai?utm_source=x", "https://www.nist.gov/ai-agents"}


def make_agent(settings, llm=None, search=fake_search, events=None):
    return ResearchAgent(settings, llm=llm or FakeLLM(), search_fn=search, reader_fn=fake_reader,
                         progress=(lambda *a: events.append(a)) if events is not None else None)


def test_full_run_executes_real_tool_functions_and_produces_cited_report(settings):
    events = []
    res = make_agent(settings, events=events).run("How does agentic AI differ from generative AI?", "Standard", 10)
    assert res.status == "ok"
    # tools were selected by the LLM and executed
    tools = [(t["tool"], t["origin"]) for t in res.trace["tools"]]
    assert ("web_search", "llm") in tools and ("analyze_sources", "llm") in tools and ("extract_evidence", "llm") in tools
    assert res.trace["queries"] == ["agentic ai vs generative ai", "agentic ai definition"]
    # duplicate URL (www / trailing slash) was removed: 4 raw results -> 3 sources
    assert res.trace["results_retrieved"] == 4 and res.trace["sources_found"] == 3 and res.trace["duplicates_removed"] >= 1
    # every cited source URL is one the search tool actually returned
    assert res.sources and {s["url"] for s in res.sources} <= REAL_URLS
    # invalid marker [9] was stripped, and warning recorded
    assert "[9]" not in res.report_markdown and any("Removed 1 citation" in w for w in res.trace["warnings"])
    # group citation expanded & linked
    assert "[[1]](https://mit.edu/agentic)" in res.report_markdown
    # references section has required fields
    refs = res.report_markdown.split("## References")[1]
    assert "https://mit.edu/agentic" in refs and "Accessed:" in refs and "Published:" in refs
    # fabricated quote ("never written on the page") was discarded by verification
    assert all("NOT in the page" not in e["statement"] for e in res.evidence)
    # all 7 stages reported
    assert res.trace["stages_completed"] == STAGES
    assert ("stage", 6, "done") in events


def test_only_cited_sources_listed_in_references(settings):
    res = make_agent(settings).run("How does agentic AI differ from generative AI?")
    cited = {s["cite"] for s in res.sources if s["cited"]}
    refs = res.report_markdown.split("## References")[1]
    assert len(re.findall(r"^\d+\. ", refs, flags=re.M)) == len(cited)


def test_autopilot_completes_pipeline_when_llm_never_calls_tools(settings):
    res = make_agent(settings, llm=FakeLLM("silent")).run("How does agentic AI differ from generative AI?")
    assert res.status == "ok"
    origins = {t["origin"] for t in res.trace["tools"]}
    assert origins == {"autopilot"}
    assert any("stopped calling tools" in w for w in res.trace["warnings"])


def test_llm_down_entirely_uses_fallback_plan_and_template_report(settings):
    res = make_agent(settings, llm=FakeLLM("down")).run("What are the applications of machine learning in healthcare?")
    # fallback planner queries are not in FAKE_RESULTS, so search yields nothing -> honest no_results
    assert res.status == "no_results" and "rephras" in res.message
    assert any("rule-based plan" in w for w in res.trace["warnings"])


def test_llm_down_after_planning_still_gives_cited_template_report(settings):
    res = make_agent(settings, llm=FakeLLM("down_after_plan")).run("How does agentic AI differ from generative AI?")
    assert res.status == "ok"
    assert any("template report" in w for w in res.trace["warnings"])
    assert "## Executive Summary" in res.report_markdown and "[[1]](" in res.report_markdown
    assert all(e["verified"] for e in res.evidence)  # heuristic evidence = real sentences


def test_search_failure_is_reported_not_raised(settings):
    def broken(*a, **k):
        raise SearchError("Tavily rejected the API key (401/403). Check TAVILY_API_KEY.")
    res = make_agent(settings, search=broken).run("How does agentic AI differ from generative AI?")
    assert res.status == "search_failed" and "Tavily rejected" in res.message and res.sources == []


def test_empty_search_results_handled(settings):
    res = make_agent(settings, search=lambda *a, **k: []).run("How does agentic AI differ from generative AI?")
    assert res.status == "no_results" and res.sources == [] and res.message


@pytest.mark.parametrize("q", ["", "   ", "hi", "12345 67890 !!!", "x" * 1001])
def test_invalid_input_rejected_before_any_tool_runs(settings, q):
    llm = FakeLLM()
    res = make_agent(settings, llm=llm).run(q)
    assert res.status == "invalid_input" and res.message and llm.calls == []


def test_validate_question_accepts_normal_input():
    ok, msg, q = validate_question("  What is   agentic AI?  ")
    assert ok and q == "What is agentic AI?"


def test_search_budget_and_duplicate_query_guard(settings):
    st = ResearchState("q", "Quick", {"max_searches": 1, "max_steps": 5, "max_pages": 1, "results_per_search": 3, "search_depth": "basic"}, 5)
    st.analysis = {"keywords": ["agentic"]}
    ex = ToolExecutor(st, settings, FakeLLM(), fake_search, fake_reader)
    assert "new_sources" in json.loads(ex.execute("web_search", {"query": "agentic ai vs generative ai"}))
    assert "budget exhausted" in json.loads(ex.execute("web_search", {"query": "another"}))["error"]
    assert "Unknown tool" in json.loads(ex.execute("hack_system", {}))["error"]
    assert "error" in json.loads(ex.execute("read_webpage", {"source_id": 999}))


def test_reader_failure_falls_back_to_snippet(settings):
    st = ResearchState("q", "Quick", {"max_searches": 2, "max_steps": 5, "max_pages": 2, "results_per_search": 3, "search_depth": "basic"}, 5)
    st.analysis = {"keywords": ["agentic", "autonomously", "tools"]}
    bad_reader = lambda url, timeout=15: {"url": url, "ok": False, "text": "", "title": "", "published_date": None, "error": "HTTP 403"}
    ex = ToolExecutor(st, settings, None, fake_search, bad_reader)
    ex.execute("web_search", {"query": "agentic ai vs generative ai"})
    out = json.loads(ex.execute("read_webpage", {"source_id": 1}))
    assert "snippet will be used" in out["error"]
    ex.execute("extract_evidence", {"source_ids": [1]})
    assert st.evidence and st.evidence[0]["source_id"] == 1  # evidence still extracted from the snippet


def test_planner_validates_and_limits_output(settings):
    plan = analyze_and_plan(FakeLLM(), "How does agentic AI differ from generative AI?")
    assert 2 <= len(plan["search_queries"]) <= 4 and plan["plan"]["objective"] and not plan["fallback"]
    fb = fallback_plan("What are the latest developments in artificial intelligence?")
    assert fb["needs_current_info"] and len(fb["search_queries"]) >= 2
