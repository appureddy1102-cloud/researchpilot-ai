"""
STAGE 3 - Tool selection & execution.

`TOOL_SCHEMAS` is the menu of tools the LLM can choose from (OpenAI
function-calling format). `ToolExecutor.execute()` receives the tool name and
arguments the LLM picked and runs the matching *real* Python function:

    web_search       -> tools.web_search.web_search            (Tavily HTTP API)
    read_webpage     -> tools.webpage_reader.read_webpage      (HTTP GET + HTML clean)
    analyze_sources  -> tools.source_evaluator.analyze_sources (scoring + dedupe)
    extract_evidence -> tools.evidence_extractor.extract_evidence
    finish_research  -> stops the loop

Each call is logged as a ToolCallRecord so the UI can show exactly which tools
ran. Tool results returned to the LLM are compact JSON strings.
"""
from __future__ import annotations

import json
from typing import Callable

from agent.state import ResearchState, ToolCallRecord
from tools import evidence_extractor as ev
from tools import source_evaluator as se
from tools import webpage_reader, web_search as ws
from utils.config import Settings

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (Tavily) for a query. Returns source ids, titles, publishers and snippets. "
                       "Use specific queries; call again with a DIFFERENT query to fill evidence gaps.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"},
            "topic": {"type": "string", "enum": ["general", "news"], "description": "Use 'news' for very recent events"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_webpage",
        "description": "Download and clean the full text of a previously found source (by id) so richer evidence can be extracted. "
                       "Limited budget - choose the most promising sources.",
        "parameters": {"type": "object", "properties": {"source_id": {"type": "integer"}}, "required": ["source_id"]}}},
    {"type": "function", "function": {
        "name": "analyze_sources",
        "description": "Score all found sources for relevance, authority, freshness; detect duplicates; returns a ranking. "
                       "Call after searching and before choosing which pages to read.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "extract_evidence",
        "description": "Extract verifiable factual statements (with exact quotes) from the given source ids and store them as evidence.",
        "parameters": {"type": "object", "properties": {
            "source_ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["source_ids"]}}},
    {"type": "function", "function": {
        "name": "finish_research",
        "description": "Stop researching when the evidence is sufficient (or the budget is used) and hand over to report writing.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}, "known_gaps": {"type": "array", "items": {"type": "string"}}},
                       "required": ["reason"]}}},
]

MAX_EVIDENCE_PER_PUBLISHER = 6


class ToolExecutor:
    def __init__(self, state: ResearchState, settings: Settings, llm,
                 search_fn: Callable = ws.web_search, reader_fn: Callable = webpage_reader.read_webpage,
                 on_action: Callable[[str], None] | None = None):
        self.state, self.settings, self.llm = state, settings, llm
        self.search_fn, self.reader_fn = search_fn, reader_fn
        self.on_action = on_action or (lambda _t: None)
        self.step = 0

    # ------------------------------------------------------------ dispatch
    def execute(self, name: str, args: dict, origin: str = "llm") -> str:
        self.step += 1
        handlers = {"web_search": self._web_search, "read_webpage": self._read_webpage,
                    "analyze_sources": self._analyze, "extract_evidence": self._extract,
                    "finish_research": self._finish}
        handler = handlers.get(name)
        if handler is None:
            result, ok = {"error": f"Unknown tool '{name}'. Available: {', '.join(handlers)}"}, False
        else:
            try:
                result = handler(**args)
                ok = "error" not in result
            except TypeError as e:
                result, ok = {"error": f"Bad arguments for {name}: {e}"}, False
            except ws.SearchError as e:
                self.state.warnings.append(f"Search failed: {e}")
                result, ok = {"error": str(e)}, False
        self.state.tool_log.append(ToolCallRecord(self.step, name, args, self._summ(result), ok, origin))
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _summ(result: dict) -> str:
        if "error" in result:
            return f"error: {result['error']}"
        return result.get("summary", "ok")

    # --------------------------------------------------------------- tools
    def _web_search(self, query: str, topic: str = "general") -> dict:
        st = self.state
        q = (query or "").strip()
        if not q:
            return {"error": "query is empty"}
        if len(st.queries_run) >= st.cfg["max_searches"]:
            return {"error": f"search budget exhausted ({st.cfg['max_searches']}). Use analyze/extract/finish."}
        if q.lower() in {x.lower() for x in st.queries_run}:
            return {"error": "this exact query was already run - use a different query"}
        self.on_action(f"Searching the web: “{q}”")
        results = self.search_fn(q, self.settings.tavily_api_key, max_results=st.cfg["results_per_search"],
                                 search_depth=st.cfg["search_depth"], topic=topic,
                                 timeout=self.settings.request_timeout)
        st.queries_run.append(q)
        st.results_retrieved += len(results)
        known = {se.normalize_url(s["url"]): sid for sid, s in st.sources.items()}
        new_items, dup = [], 0
        for r in results:
            key = se.normalize_url(r["url"])
            if key in known:
                dup += 1
                continue
            sid = st.next_id
            st.next_id += 1
            r.update(publisher=se.publisher_name(r["url"]), domain=se.registrable_domain(r["url"]))
            st.sources[sid] = r
            known[key] = sid
            new_items.append({"id": sid, "title": r["title"][:110], "publisher": r["publisher"],
                              "date": r.get("published_date"), "snippet": r["snippet"][:180]})
        st.duplicates_removed += dup
        if not results:
            return {"summary": f"0 results for “{q}”", "new_sources": [], "note": "No results - try a broader or differently worded query."}
        self.on_action(f"Found {len(results)} results ({len(new_items)} new, {dup} duplicates skipped)")
        return {"summary": f"{len(results)} results, {len(new_items)} new, {dup} duplicates", "new_sources": new_items}

    def _read_webpage(self, source_id: int) -> dict:
        st = self.state
        s = st.sources.get(int(source_id))
        if not s:
            return {"error": f"unknown source_id {source_id}"}
        if s.get("text"):
            return {"summary": f"source {source_id} already read", "note": "already read"}
        if st.pages_read >= st.cfg["max_pages"]:
            return {"error": f"page-read budget exhausted ({st.cfg['max_pages']})"}
        self.on_action(f"Reading page: {s['publisher']}")
        st.pages_read += 1
        page = self.reader_fn(s["url"], timeout=self.settings.request_timeout)
        if not page["ok"]:
            s["read_error"] = page["error"]
            return {"error": f"could not read {s['publisher']}: {page['error']} (snippet will be used instead)"}
        s["text"] = page["text"]
        if page.get("published_date") and not s.get("published_date"):
            s["published_date"] = page["published_date"]
        return {"summary": f"read {s['publisher']} ({len(page['text'])} chars)", "source_id": source_id, "chars": len(page["text"]),
                "published_date": s.get("published_date")}

    def _analyze(self) -> dict:
        st = self.state
        if not st.sources:
            return {"error": "no sources yet - run web_search first"}
        self.on_action("Evaluating source quality, freshness and duplicates")
        res = se.analyze_sources(st.sources, st.analysis.get("keywords", []), bool(st.analysis.get("needs_current_info")),
                                 {e["source_id"] for e in st.evidence})
        st.analyzed = True
        st.duplicates_removed += len(res["duplicates"])
        ranking = [{"id": sid, "score": st.sources[sid]["scores"]["overall"], "tier": st.sources[sid]["scores"]["tier"],
                    "publisher": st.sources[sid]["publisher"], "date": st.sources[sid].get("published_date")}
                   for sid in res["ranked"][:12]]
        return {"summary": f"ranked {len(res['ranked'])} sources, {len(res['duplicates'])} duplicates flagged",
                "ranking": ranking, "duplicates": [{"dup": d, "of": k, "why": w} for d, k, w in res["duplicates"]]}

    def _extract(self, source_ids: list[int]) -> dict:
        st = self.state
        if not isinstance(source_ids, list):
            source_ids = [source_ids]
        report, total_new, dropped_total = {}, 0, 0
        publisher_of = {sid: s.get("domain", "") for sid, s in st.sources.items()}
        for raw in source_ids[:6]:
            try:
                sid = int(raw)
            except (TypeError, ValueError):
                continue
            s = st.sources.get(sid)
            if not s or s.get("duplicate_of"):
                report[str(raw)] = "skipped (unknown or duplicate)"
                continue
            if any(e["source_id"] == sid for e in st.evidence):
                report[str(sid)] = "already extracted"
                continue
            self.on_action(f"Extracting evidence from {s['publisher']}")
            items = ev.extract_evidence(self.llm, sid, s, st.question, st.analysis.get("keywords", []))
            kept, dropped = ev.dedupe_evidence(st.evidence, items, MAX_EVIDENCE_PER_PUBLISHER, publisher_of)
            st.evidence.extend(kept)
            total_new += len(kept)
            dropped_total += dropped
            report[str(sid)] = f"{len(kept)} items"
        return {"summary": f"{total_new} new evidence items ({dropped_total} repeats dropped)", "per_source": report,
                "total_evidence": len(st.evidence)}

    def _finish(self, reason: str = "", known_gaps: list[str] | None = None) -> dict:
        self.state.finished = True
        self.state.finish_reason = reason
        for g in known_gaps or []:
            self.state.warnings.append(f"Gap noted by agent: {g}")
        return {"summary": f"research finished: {reason[:120]}"}
