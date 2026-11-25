"""Shared test fixtures: a scripted fake LLM and fake search/reader functions.

These fixtures exist ONLY for tests (so they run offline and for free). The
application itself never ships fake data - see tools/web_search.py.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import LLMError, LLMMessage, ToolCall  # noqa: E402
from utils.config import Settings  # noqa: E402

PAGE_TEXT = ("Agentic AI systems plan, call tools and act autonomously towards goals, unlike traditional generative AI models "
             "that only produce content in response to a prompt. Surveys report that 35 percent of organisations piloted agents in 2025. ")

FAKE_RESULTS = {
    "agentic ai vs generative ai": [
        {"title": "Agentic AI explained - MIT", "url": "https://mit.edu/agentic", "snippet": PAGE_TEXT, "score": 0.9, "published_date": "2025-09-01", "query": "q"},
        {"title": "Generative AI overview", "url": "https://www.example-blog.com/genai?utm_source=x", "snippet": PAGE_TEXT.replace("Agentic", "Autonomous"), "score": 0.6, "published_date": None, "query": "q"},
    ],
    "agentic ai definition": [
        {"title": "Agentic AI explained - MIT", "url": "https://www.mit.edu/agentic/", "snippet": PAGE_TEXT, "score": 0.9, "published_date": "2025-09-01", "query": "q"},  # duplicate URL
        {"title": "AI agents report", "url": "https://www.nist.gov/ai-agents", "snippet": "NIST describes AI agents as software that perceives and acts; the report discusses evaluation of agentic systems in detail across sectors.", "score": 0.8, "published_date": "2025-11-10", "query": "q"},
    ],
}


def fake_search(query, api_key, max_results=5, search_depth="basic", topic="general", timeout=20, **kw):
    return [dict(r, query=query) for r in FAKE_RESULTS.get(query, [])]


def fake_reader(url, timeout=15, **kw):
    return {"url": url, "ok": True, "text": PAGE_TEXT * 3, "title": "t", "published_date": None, "error": ""}


class FakeLLM:
    """Scripted LLM. mode: 'tools' (drives tool loop), 'silent' (never calls tools), 'down' (always errors),
    'down_after_plan' (planner ok, everything else fails)."""

    def __init__(self, mode="tools"):
        self.mode, self.calls, self.turn = mode, [], 0

    def chat(self, messages, tools=None, json_mode=False, temperature=0.2, max_tokens=None):
        prompt = messages[-1]["content"] if messages and isinstance(messages[-1].get("content"), str) else ""
        first = messages[0]["content"] or ""
        self.calls.append("tools" if tools else "json" if json_mode else "text")
        if self.mode == "down":
            raise LLMError("fake LLM down")
        if "planning module" in first or "planning module" in prompt:
            return LLMMessage(json.dumps({
                "main_topic": "agentic ai", "keywords": ["agentic", "generative", "ai", "autonomous"], "intent": "Compare agentic and generative AI",
                "needs_current_info": True, "sub_questions": ["What is agentic AI?"],
                "search_queries": ["agentic ai vs generative ai", "agentic ai definition"],
                "plan": {"objective": "compare", "subtopics": ["a"], "information_requirements": ["b"], "search_strategy": "c",
                         "expected_evidence": ["d"], "potential_limitations": ["e"]}}))
        if self.mode == "down_after_plan":
            raise LLMError("fake LLM down")
        if tools:
            if self.mode == "silent":
                return LLMMessage("I will answer from memory.")
            self.turn += 1
            script = {
                1: [ToolCall("c1", "web_search", {"query": "agentic ai vs generative ai"}), ToolCall("c2", "web_search", {"query": "agentic ai definition"})],
                2: [ToolCall("c3", "analyze_sources", {})],
                3: [ToolCall("c4", "read_webpage", {"source_id": 1}), ToolCall("c5", "extract_evidence", {"source_ids": [1, 2, 3]})],
                4: [ToolCall("c6", "finish_research", {"reason": "enough evidence"})],
            }
            return LLMMessage("", script.get(self.turn, [ToolCall("c9", "finish_research", {"reason": "done"})]))
        if "extract evidence" in prompt.lower() or "You extract evidence" in prompt:
            text = prompt.split('"""')[1].strip()
            quote = re.split(r"(?<=[.])\s", text)[0][:150]
            return LLMMessage(json.dumps({"evidence": [{"statement": quote, "quote": quote, "type": "fact"},
                                                      {"statement": "Quote that is NOT in the page", "quote": "this sentence was never written on the page at all", "type": "fact"}]}))
        if "report-writing module" in prompt:
            return LLMMessage("## Executive Summary\nAgentic AI acts autonomously using tools while generative AI responds to prompts [1][2]. A fabricated claim [9].\n\n"
                              "## Introduction\nIntro text about agentic systems and generative models that is long enough to be useful here.\n\n"
                              "## Main Findings\n- Agentic systems plan and act [1, 2].\n\n## Detailed Analysis\nDetails about tools and autonomy [1].\n\n"
                              "## Evidence Comparison\n| Aspect | Sources | Cites |\n|---|---|---|\n| Autonomy | agents act | [1] |\n\n"
                              "## Important Statistics\n- Surveys report 35 percent of organisations piloted agents in 2025 [1].\n\n"
                              "## Limitations and Uncertainties\nOnly few sources.\n\n## Conclusion\nThey differ in autonomy.")
        return LLMMessage("{}")


@pytest.fixture
def settings():
    return Settings(llm_api_key="sk-test-123456789", tavily_api_key="tvly-test-123456789", db_path=":memory:")
