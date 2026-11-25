"""
The research agent - orchestrates the whole workflow.

    Stage 1-2  analyze_and_plan()          (research_planner.py)
    Stage 3-5  LLM tool-calling loop       (this file + tool_selector.py)
                  LLM picks a tool -> Python executes it -> result goes back to
                  the LLM -> LLM decides the next tool, until it calls
                  finish_research or the budget (search/step limits) runs out.
    Safety net `_autopilot()` guarantees the pipeline completes even if the
                  model never calls a tool (e.g. provider without tool support).
                  Autopilot calls are labelled as such in the trace.
    Stage 6-7  select sources -> synthesize() -> validate citations -> report

The agent never invents sources: every cited item comes from a real Tavily
result that is stored in `ResearchState.sources`.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable

from agent import research_planner, research_synthesizer
from agent.llm import LLMClient, LLMError
from agent.state import ResearchState
from agent.tool_selector import TOOL_SCHEMAS, ToolExecutor
from tools import evidence_extractor as ev
from tools import source_evaluator as se
from tools import webpage_reader, web_search as ws
from utils import citations
from utils.config import DEPTH_PRESETS, Settings

STAGES = [
    "Understanding research question", "Planning search strategy", "Searching the web", "Collecting evidence",
    "Evaluating sources", "Synthesizing findings", "Preparing final report",
]

SYSTEM_PROMPT = """You are the controller of a web research agent. You work ONLY by calling tools; never answer from memory.
Tools: web_search, read_webpage, analyze_sources, extract_evidence, finish_research.

Recommended policy:
1. Run the planned search queries (refine them if useful). Each new query must differ from earlier ones.
2. Call analyze_sources to rank sources by relevance, authority (prefer government, university, journals, official bodies) and freshness.
3. read_webpage for the most promising, authoritative sources (limited budget); sources you do not read can still be used via their snippets.
4. extract_evidence for the best sources (several ids per call).
5. Check the STATUS line. If key sub-questions lack evidence or only one viewpoint/source type is covered, run another, different web_search and extract again.
6. Call finish_research when you have solid evidence from at least 3 different sources, or when budgets are used up.
Be efficient: do not repeat the same call."""


@dataclass
class ResearchResult:
    status: str                      # "ok" | "no_results" | "search_failed" | "invalid_input"
    question: str
    depth: str = "Standard"
    report_markdown: str = ""        # full report incl. clickable citations + references
    body: str = ""                   # report body with plain [n] markers
    sources: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    message: str = ""                # human-readable explanation when status != ok

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def validate_question(q: str) -> tuple[bool, str, str]:
    """Return (ok, message, cleaned_question)."""
    q = re.sub(r"\s+", " ", (q or "")).strip()
    if not q:
        return False, "Please enter a research question.", q
    if len(q) < 10:
        return False, "Your question is too short - please give at least a short sentence (10+ characters).", q
    if len(q) > 1000:
        return False, "Your question is too long (max 1000 characters). Please shorten it.", q
    if len(re.findall(r"[A-Za-z\u00C0-\u024F\u0900-\u097F\u0C00-\u0C7F]", q)) < 5:
        return False, "Your question needs to contain words, not just numbers or symbols.", q
    return True, "", q


class ResearchAgent:
    def __init__(self, settings: Settings, llm=None, search_fn: Callable = ws.web_search,
                 reader_fn: Callable = webpage_reader.read_webpage,
                 progress: Callable[[str, object, object], None] | None = None):
        """
        progress(kind, a, b): kind="stage" -> (stage_index, "active"|"done"); kind="action" -> (text, None)
        `llm`, `search_fn`, `reader_fn` can be injected (used by the unit tests).
        """
        self.settings = settings
        self.llm = llm or LLMClient(settings)
        self.search_fn, self.reader_fn = search_fn, reader_fn
        self.progress = progress or (lambda *_a: None)
        self._stage = -1

    # --------------------------------------------------------- progress
    def _enter(self, idx: int) -> None:
        if idx <= self._stage:
            return
        for i in range(max(self._stage, 0), idx):
            if self._stage >= 0 or i < idx:
                self.progress("stage", i, "done")
        self._stage = idx
        self.progress("stage", idx, "active")

    def _action(self, text: str) -> None:
        self.progress("action", text, None)

    # -------------------------------------------------------------- run
    def run(self, question: str, depth: str = "Standard", source_limit: int = 10) -> ResearchResult:
        t0 = time.time()
        ok, msg, question = validate_question(question)
        if not ok:
            return ResearchResult("invalid_input", question, depth, message=msg)
        depth = depth if depth in DEPTH_PRESETS else "Standard"
        st = ResearchState(question=question, depth=depth, cfg=dict(DEPTH_PRESETS[depth]), source_limit=int(source_limit))

        # ---- Stage 1 + 2: analysis and plan
        self._enter(0)
        self._action("Analysing the question and extracting keywords")
        analysis = research_planner.analyze_and_plan(self.llm, question, max_queries=st.cfg["max_searches"])
        st.analysis, st.plan = analysis, analysis["plan"]
        if analysis.get("fallback"):
            st.warnings.append("The planner LLM call failed, so a rule-based plan was used. " + analysis["plan"]["potential_limitations"][-1])
        self._enter(1)
        self._action(f"Plan ready: {len(analysis['search_queries'])} search queries, topic “{analysis['main_topic']}”")

        # ---- Stage 3-5: tool-calling loop
        executor = ToolExecutor(st, self.settings, self.llm, self.search_fn, self.reader_fn, on_action=self._action)
        self._tool_loop(st, executor)
        self._autopilot(st, executor)

        # ---- no usable sources -> honest failure
        if not st.sources:
            search_errors = [w for w in st.warnings if w.startswith("Search failed")]
            status = "search_failed" if search_errors else "no_results"
            message = search_errors[-1] if search_errors else \
                "The search returned no results for this question. Try rephrasing it or making it more specific."
            return ResearchResult(status, question, depth, message=message, trace=self._trace(st, [], t0))

        # ---- final scoring + selection
        self._enter(4)
        res = se.analyze_sources(st.sources, analysis.get("keywords", []), bool(analysis.get("needs_current_info")),
                                 {e["source_id"] for e in st.evidence})
        with_ev = {e["source_id"] for e in st.evidence}
        ranked_with_evidence = [sid for sid in res["ranked"] if sid in with_ev][:st.source_limit]
        if not ranked_with_evidence:
            return ResearchResult("no_results", question, depth, trace=self._trace(st, [], t0),
                                  message="Sources were found but no usable evidence could be extracted from them.")
        cite_of = {sid: n for n, sid in enumerate(ranked_with_evidence, start=1)}
        corro = ev.find_corroboration([e for e in st.evidence if e["source_id"] in cite_of])
        sources = []
        for sid in ranked_with_evidence:
            s = st.sources[sid]
            sources.append({
                "cite": cite_of[sid], "id": sid, "title": s["title"], "publisher": s["publisher"], "url": s["url"],
                "snippet": s.get("snippet", "")[:300], "published_date": s.get("published_date"),
                "scores": s["scores"], "page_read": bool(s.get("text")),
                "evidence_count": sum(1 for e in st.evidence if e["source_id"] == sid),
                "corroborated_by": sorted(cite_of[o] for o in corro.get(sid, set()) if o in cite_of),
            })
        evidence = [dict(e, cite=cite_of[e["source_id"]]) for e in st.evidence if e["source_id"] in cite_of]
        evidence.sort(key=lambda e: e["cite"])

        # ---- Stage 6: synthesis
        self._enter(5)
        self._action(f"Synthesizing findings from {len(sources)} sources / {len(evidence)} evidence items")
        try:
            body = research_synthesizer.synthesize(self.llm, question, analysis, sources, evidence[:70])
        except LLMError as e:
            st.warnings.append(f"Report writing by the LLM failed ({e}); an evidence-based template report was generated instead.")
            body = research_synthesizer.template_report(question, sources, evidence, "LLM unavailable")

        # ---- Stage 7: citations + final report
        self._enter(6)
        self._action("Validating citations and building references")
        body, invalid, used = citations.validate_citations(body, set(cite_of.values()))
        if invalid:
            st.warnings.append(f"Removed {len(invalid)} citation marker(s) that did not match any retrieved source: {invalid}")
        if not used:
            st.warnings.append("The LLM report contained no valid citations, so an evidence-based template report was used instead.")
            body = research_synthesizer.template_report(question, sources, evidence, "no valid citations in LLM output")
            body, _, used = citations.validate_citations(body, set(cite_of.values()))
        check = citations.check_claim_support(body, evidence)
        if check["flagged"]:
            st.warnings.append(f"{len(check['flagged'])} cited sentence(s) have weak overlap with the extracted evidence - verify them against the sources.")
        for s in sources:
            s["cited"] = s["cite"] in used
        accessed = date.today().strftime("%d %b %Y")
        refs = citations.build_references(sources, accessed, only_ids=set(used))
        report = (
            f"# {question}\n\n"
            f"*Generated by ResearchPilot AI on {accessed} · depth: {depth} · {len(used)} sources cited*\n\n"
            f"{citations.linkify(body, sources)}\n\n## References\n{refs}\n"
        )
        st.stages_completed = list(STAGES)
        self.progress("stage", 6, "done")
        trace = self._trace(st, sources, t0)
        trace["citation_check"] = check
        return ResearchResult("ok", question, depth, report, body, sources, evidence, trace)

    # --------------------------------------------------- tool-calling loop
    def _tool_loop(self, st: ResearchState, ex: ToolExecutor) -> None:
        a = st.analysis
        brief = (
            f"Research question: {st.question}\nMain topic: {a['main_topic']}\nIntent: {a['intent']}\n"
            f"Needs current info: {a['needs_current_info']}\nSub-questions: {json.dumps(a['sub_questions'])}\n"
            f"Planned search queries: {json.dumps(a['search_queries'])}\n"
            f"Budgets: max {st.cfg['max_searches']} searches, {st.cfg['max_pages']} page reads, {st.cfg['max_steps']} tool-selection steps.\n"
            f"Start now."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": brief}]
        nudged = False
        for _ in range(st.cfg["max_steps"]):
            try:
                reply = self.llm.chat(messages, tools=TOOL_SCHEMAS, temperature=0.1)
            except LLMError as e:
                st.warnings.append(f"Tool-selection LLM call failed: {e}")
                return
            messages.append(reply.to_dict())
            if not reply.tool_calls:
                if nudged:
                    st.warnings.append("The model stopped calling tools; the agent completed the remaining steps automatically.")
                    return
                nudged = True
                messages.append({"role": "user", "content": "Please continue by calling a tool (or finish_research). " + st.status_line()})
                continue
            for call in reply.tool_calls:
                self._enter({"web_search": 2, "read_webpage": 3, "extract_evidence": 3, "analyze_sources": 4}.get(call.name, self._stage))
                out = ex.execute(call.name, call.arguments, origin="llm")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out[:3500]})
            if st.finished:
                return
            messages.append({"role": "user", "content": st.status_line()})

    # ------------------------------------------------------ safety net
    def _autopilot(self, st: ResearchState, ex: ToolExecutor) -> None:
        """Make sure the minimum pipeline ran even if the LLM did not drive it."""
        if not st.queries_run:
            self._enter(2)
            for q in st.analysis["search_queries"][:st.cfg["max_searches"]]:
                before = len(ex.state.warnings)
                ex.execute("web_search", {"query": q}, origin="autopilot")
                if len(ex.state.warnings) > before and not st.sources:
                    break  # search is failing (e.g. bad key) - do not hammer the API
        if not st.sources:
            return
        if not st.analyzed:
            self._enter(4)
            ex.execute("analyze_sources", {}, origin="autopilot")
        with_ev = {e["source_id"] for e in st.evidence}
        if len(with_ev) < min(3, len(st.sources)):
            self._enter(3)
            ranked = sorted((i for i, s in st.sources.items() if not s.get("duplicate_of") and "scores" in s),
                            key=lambda i: st.sources[i]["scores"]["overall"], reverse=True)
            for sid in [i for i in ranked if i not in with_ev][:max(st.cfg["max_pages"] - st.pages_read, 0)][:2]:
                ex.execute("read_webpage", {"source_id": sid}, origin="autopilot")
            todo = [i for i in ranked if i not in with_ev][:4]
            if todo:
                ex.execute("extract_evidence", {"source_ids": todo}, origin="autopilot")

    # ------------------------------------------------------------ trace
    def _trace(self, st: ResearchState, sources: list[dict], t0: float) -> dict:
        return {
            "queries": st.queries_run, "results_retrieved": st.results_retrieved, "sources_found": len(st.sources),
            "duplicates_removed": st.duplicates_removed, "sources_selected": len(sources),
            "pages_read": st.pages_read, "evidence_items": len(st.evidence),
            "tools": [t.to_dict() for t in st.tool_log], "stages_completed": st.stages_completed or STAGES[:self._stage + 1],
            "warnings": st.warnings, "analysis": st.analysis, "finish_reason": st.finish_reason,
            "model": self.settings.llm_model, "duration_s": round(time.time() - t0, 1),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
