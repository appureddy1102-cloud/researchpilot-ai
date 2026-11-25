"""
ResearchPilot AI - Streamlit dashboard.

Run with:  streamlit run app.py

This file is only the user interface. All research logic lives in `agent/`
and `tools/`; storage in `database/`; helpers in `utils/`.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.llm import LLMClient, LLMError  # noqa: E402
from agent.research_agent import STAGES, ResearchAgent, validate_question  # noqa: E402
from database.database import Database  # noqa: E402
from tools import web_search as ws  # noqa: E402
from utils import config as cfg  # noqa: E402
from utils.citations import format_date  # noqa: E402
from utils.export import to_markdown, to_pdf  # noqa: E402

st.set_page_config(page_title=f"{cfg.APP_NAME}", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

NAV = ["🔎  New Research", "🕘  Research History", "⭐  Saved Reports", "⚙️  Settings", "🔑  API Configuration"]
THEMES = {
    "dark": dict(bg="#0b0b14", sidebar="#0f0f1c", card="#13131f", card_hover="#1a1a2b", input="#10101b", border="#26263a",
                 text="#e8e8f2", muted="#9494ad", accent="#8b8cf8", accent_soft="rgba(129,140,248,.12)", accent_border="rgba(129,140,248,.45)",
                 shadow="0 10px 30px rgba(0,0,0,.35)"),
    "light": dict(bg="#f6f7fb", sidebar="#ffffff", card="#ffffff", card_hover="#f1f2fa", input="#ffffff", border="#e2e4ef",
                  text="#1b1b2b", muted="#6b6b85", accent="#4f46e5", accent_soft="rgba(99,102,241,.09)", accent_border="rgba(99,102,241,.4)",
                  shadow="0 8px 24px rgba(30,30,70,.07)"),
}

S = st.session_state
for k, v in {"theme": "dark", "page": NAV[0], "overrides": {}, "view_id": None, "current": None, "service_ok": None,
             "service_msg": "", "question": "", "default_depth": "Standard", "default_limit": 10, "last_error": None}.items():
    S.setdefault(k, v)


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


def settings() -> cfg.Settings:
    return cfg.load_settings().with_overrides(**S.overrides)


db = get_db(settings().db_path)


# ================================================================= sidebar
with st.sidebar:
    st.markdown(f'<div class="rp-brand"><div class="rp-logo">🧭</div><b>{cfg.APP_NAME}</b></div>', unsafe_allow_html=True)
    st.radio("Navigation", NAV, key="page", label_visibility="collapsed")
    st.divider()
    dark = st.toggle("🌙 Dark mode", value=S.theme == "dark")
    S.theme = "dark" if dark else "light"
    n_hist = len(db.list_reports())
    st.caption(f"{n_hist} report(s) in local history")

t = THEMES[S.theme]
css = (Path(__file__).parent / "assets" / "custom.css").read_text(encoding="utf-8")
root = ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items()) + "}"
st.markdown(f"<style>{root}\n{css}</style>", unsafe_allow_html=True)


# ================================================================== header
def header() -> None:
    s = settings()
    missing = s.missing()
    if missing:
        dot, label = "bad", f"AI service not configured · missing {', '.join(missing)}"
    elif S.service_ok is True:
        dot, label = "ok", "AI service connected"
    elif S.service_ok is False:
        dot, label = "bad", "Connection problem · " + html.escape(S.service_msg[:70])
    else:
        dot, label = "warn", "Keys configured · not yet verified"
    st.markdown(
        f'<div class="rp-header"><div><div class="rp-title">{cfg.APP_NAME}</div><div class="rp-sub">{cfg.APP_SUBTITLE}</div></div>'
        f'<div class="rp-status"><span class="rp-dot {dot}"></span>{label}</div></div>', unsafe_allow_html=True)


# ============================================================ small helpers
def esc(x) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


def badge(text: str, kind: str = "") -> str:
    return f'<span class="rp-badge {kind}">{esc(text)}</span>'


def metric_row(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="rp-metric"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div></div>' for l, v in items)
    st.markdown(f'<div class="rp-metrics">{cells}</div>', unsafe_allow_html=True)


def source_card(s: dict) -> str:
    sc = s["scores"]
    kind = "good" if sc["authority"] >= 0.75 else "mid" if sc["authority"] >= 0.5 else "low"
    badges = badge(sc["tier"], kind) + badge(f"Relevance {sc['relevance']:.0%}") + badge(f"Freshness: {format_date(s.get('published_date'))}")
    badges += badge("Cited in report", "good") if s.get("cited") else badge("Not cited", "mid")
    badges += badge("Full page read" if s.get("page_read") else "Snippet only")
    if s.get("corroborated_by"):
        badges += badge("Corroborated by " + ", ".join(f"[{c}]" for c in s["corroborated_by"]), "good")
    snippet = esc(s.get("snippet", "")[:220]) + ("…" if len(s.get("snippet", "")) > 220 else "")
    return (
        f'<div class="rp-card"><div class="rp-src-title">[{s["cite"]}] {esc(s["title"])}</div>'
        f'<div class="rp-src-meta">{esc(s["publisher"])} · {esc(s["url"])}</div>'
        f'<div>{badges}</div><div class="rp-bar" title="Overall score"><div style="width:{sc["overall"] * 100:.0f}%"></div></div>'
        f'<div class="rp-src-meta">Overall score {sc["overall"]:.2f} · authority {sc["authority"]:.2f} · {esc(sc["authority_reason"])}</div>'
        f'<div class="rp-snip">{snippet}</div>'
        f'<a class="rp-open" href="{esc(s["url"])}" target="_blank" rel="noopener noreferrer">Open source ↗</a></div>')


@st.cache_data(show_spinner=False, max_entries=20)
def make_pdf(report_md: str, sources_json: str) -> bytes:
    return to_pdf({"report_markdown": report_md, "sources": json.loads(sources_json)})


# ======================================================== result rendering
def render_record(rec: dict, key: str, allow_delete: bool = False) -> None:
    tr = rec["trace"]
    chk = tr.get("citation_check", {})
    metric_row([
        ("Search queries", str(len(tr.get("queries", [])))), ("Results retrieved", str(tr.get("results_retrieved", 0))),
        ("Sources cited", str(sum(1 for s in rec["sources"] if s.get("cited")))), ("Evidence items", str(len(rec["evidence"]))),
        ("Tools executed", str(len(tr.get("tools", [])))),
        ("Claim check", f"{chk.get('supported', 0)}/{chk.get('checked', 0)}" if chk else "–"), ("Time", f"{tr.get('duration_s', 0)}s"),
    ])
    a, b, c, d, _ = st.columns([1.2, 1.2, 1.2, 1.2, 3])
    rid = rec.get("id")
    if rid is not None:
        if a.button("★ Saved" if rec.get("saved") else "☆ Save report", key=f"save_{key}", width="stretch"):
            db.set_saved(rid, not rec.get("saved"))
            if S.current and S.current.get("id") == rid:
                S.current = db.get_report(rid)
            st.rerun()
    b.download_button("⬇ Markdown", to_markdown(rec), file_name=f"research_{rid or 'report'}.md", mime="text/markdown",
                      key=f"md_{key}", width="stretch")
    try:
        c.download_button("⬇ PDF", make_pdf(rec["report_markdown"], json.dumps(rec["sources"])), file_name=f"research_{rid or 'report'}.pdf",
                          mime="application/pdf", key=f"pdf_{key}", width="stretch")
    except Exception as e:  # PDF problems must never break the page
        c.caption(f"PDF unavailable: {type(e).__name__}")
    if allow_delete and rid is not None:
        if d.button("🗑 Delete", key=f"del_{key}", width="stretch"):
            delete_report(rid)
            st.rerun()

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        tab_r, tab_e, tab_p = st.tabs(["📄 Report", "🧾 Evidence", "🛰️ Research process"])
        with tab_r:
            with st.container(border=True):
                st.markdown(rec["report_markdown"])
        with tab_e:
            if rec["evidence"]:
                df = pd.DataFrame([{"Cite": f"[{e['cite']}]", "Type": e["type"], "Statement": e["statement"],
                                    "Exact quote from source": e.get("quote", ""), "Verified in source": "✓" if e.get("verified") else "✗"}
                                   for e in rec["evidence"]])
                st.dataframe(df, width="stretch", hide_index=True)
            st.markdown("**Source evaluation**")
            sdf = pd.DataFrame([{"Cite": f"[{s['cite']}]", "Publisher": s["publisher"], "Type": s["scores"]["tier"],
                                 "Authority": s["scores"]["authority"], "Relevance": s["scores"]["relevance"],
                                 "Freshness": s["scores"]["freshness"], "Overall": s["scores"]["overall"],
                                 "Evidence items": s["evidence_count"]} for s in rec["sources"]])
            st.dataframe(sdf, width="stretch", hide_index=True)
            st.caption("Scores are rule-based and transparent: authority from publisher type/domain, freshness from the publication date "
                       "(neutral if unknown), relevance from the search score + keyword overlap.")
        with tab_p:
            render_process(tr)
    with right:
        st.markdown("#### Sources")
        for s in rec["sources"]:
            st.markdown(source_card(s), unsafe_allow_html=True)


def render_process(tr: dict) -> None:
    st.markdown("**Stages completed**")
    st.markdown("  \n".join(f"✅ {i + 1}. {s}" for i, s in enumerate(tr.get("stages_completed", []))))
    an = tr.get("analysis", {})
    if an:
        st.markdown("**Research plan**")
        plan = an.get("plan", {})
        st.markdown(f"- **Main topic:** {an.get('main_topic')}\n- **Intent:** {an.get('intent')}\n- **Needs current information:** {an.get('needs_current_info')}\n"
                    f"- **Objective:** {plan.get('objective')}\n- **Strategy:** {plan.get('search_strategy')}")
        if plan.get("potential_limitations"):
            st.markdown("- **Anticipated limitations:** " + "; ".join(plan["potential_limitations"]))
    st.markdown("**Search queries used**")
    st.markdown("\n".join(f"{i + 1}. `{q}`" for i, q in enumerate(tr.get("queries", []))) or "_none_")
    st.markdown(f"**Sources:** {tr.get('results_retrieved', 0)} results retrieved → {tr.get('sources_found', 0)} unique "
                f"({tr.get('duplicates_removed', 0)} duplicates removed) → {tr.get('sources_selected', 0)} with usable evidence · "
                f"{tr.get('pages_read', 0)} full pages read · model `{tr.get('model', '?')}`")
    if tr.get("tools"):
        st.markdown("**Tools executed** (who chose the call: `llm` = the model decided, `autopilot` = safety-net fallback)")
        st.dataframe(pd.DataFrame([{"#": t_["step"], "Tool": t_["tool"], "Chosen by": t_["origin"],
                                    "Arguments": json.dumps(t_["arguments"])[:90], "Result": t_["summary"][:110], "OK": "✓" if t_["ok"] else "✗"}
                                   for t_ in tr["tools"]]), width="stretch", hide_index=True)
    for w in tr.get("warnings", []):
        st.warning(w)
    chk = tr.get("citation_check", {})
    for sent in chk.get("flagged", [])[:5]:
        st.caption(f"⚠ Weak evidence overlap: “{sent}”")


def delete_report(rid: int) -> None:
    db.delete_report(rid)
    if S.view_id == rid:
        S.view_id = None
    if S.current and S.current.get("id") == rid:
        S.current = None


# ================================================================== pages
def stage_html(states: list[str]) -> str:
    out = []
    for i, name in enumerate(STAGES):
        stt = states[i]
        icon = "✓" if stt == "done" else str(i + 1)
        out.append(f'<div class="rp-stage {stt}"><span class="ic">{icon}</span><span>{name}</span></div>')
    return "".join(out)


def page_new_research() -> None:
    header()
    s = settings()
    with st.container(border=True):
        st.markdown("##### What would you like to research?")
        st.text_area("Research question", key="question", height=110, label_visibility="collapsed",
                     placeholder="e.g. How does agentic AI differ from traditional generative AI?")
        c1, c2, c3 = st.columns([1.6, 1.4, 1])
        depth = c1.radio("Research depth", list(cfg.DEPTH_PRESETS), index=list(cfg.DEPTH_PRESETS).index(S.default_depth), horizontal=True,
                         help="Quick: 2 searches · Standard: 4 searches · Deep: 7 searches with advanced Tavily search")
        limit = c2.select_slider("Maximum sources to keep", cfg.SOURCE_LIMIT_OPTIONS, value=S.default_limit)
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        go = c3.button("🔍  Start research", type="primary", width="stretch")
        p = cfg.DEPTH_PRESETS[depth]
        st.caption(f"{depth}: up to {p['max_searches']} searches · {p['max_pages']} full pages read · {p['results_per_search']} results per search")
        st.markdown("<span class='rp-muted'>Try an example:</span>", unsafe_allow_html=True)
        cols = st.columns(len(cfg.EXAMPLE_QUESTIONS))
        for i, (col, q) in enumerate(zip(cols, cfg.EXAMPLE_QUESTIONS)):
            col.button(q, key=f"ex{i}", on_click=lambda q=q: S.update(question=q), width="stretch")

    if go:
        ok, msg, q = validate_question(S.question)
        if not ok:
            st.warning(msg)
        elif not s.ready:
            st.error("**API keys are missing.** Missing: " + ", ".join(f"`{m}`" for m in s.missing()) +
                     ". Add them in **🔑 API Configuration** (this session only) or put them in a `.env` file / Streamlit secrets "
                     "(see the README), then try again.")
        else:
            run_research(s, q, depth, limit)

    if S.last_error:
        st.error(S.last_error)
    if S.current:
        st.markdown("---")
        render_record(S.current, key="current")
    elif not go:
        st.info("Enter a question above and press **Start research**. The agent will plan queries, search the web with Tavily, read pages, "
                "score sources, extract evidence and write a cited report.")


def run_research(s: cfg.Settings, question: str, depth: str, limit: int) -> None:
    S.last_error, S.current = None, None
    states = ["pending"] * len(STAGES)
    log: list[str] = []
    box = st.container(border=True)
    with box:
        st.markdown("##### Research in progress")
        ph_stages, ph_log = st.empty(), st.empty()
    ph_stages.markdown(stage_html(states), unsafe_allow_html=True)

    def progress(kind, a, b):
        if kind == "stage":
            states[a] = b
        else:
            log.append(str(a))
        ph_stages.markdown(stage_html(states), unsafe_allow_html=True)
        ph_log.markdown("".join(f'<div class="rp-log">› {esc(x)}</div>' for x in log[-5:]), unsafe_allow_html=True)

    try:
        agent = ResearchAgent(s, progress=progress)
        result = agent.run(question, depth, limit)
    except LLMError as e:
        box.empty()
        S.last_error, S.service_ok, S.service_msg = f"LLM error: {e}", False, str(e)
        return
    except Exception as e:  # last-resort guard: never show a stack trace or leak keys
        box.empty()
        S.last_error = cfg.redact(f"Unexpected error ({type(e).__name__}): {e}", s)
        return

    box.empty()
    if result.status == "ok":
        S.service_ok = True
        rid = db.save_report(result.to_dict())
        S.current = db.get_report(rid)
        for w in result.trace.get("warnings", []):
            st.toast(w[:120], icon="⚠️")
    elif result.status == "search_failed":
        S.service_ok, S.service_msg = False, result.message
        S.last_error = f"🔌 Search failed: {result.message}"
    else:
        S.last_error = f"⚠️ {result.message}"


def history_list(saved_only: bool) -> None:
    header()
    st.markdown(f"### {'⭐ Saved Reports' if saved_only else '🕘 Research History'}")
    recs = db.list_reports(saved_only=saved_only)
    if S.view_id is not None:
        rec = db.get_report(S.view_id)
        if rec and (rec["saved"] or not saved_only):
            if st.button("← Back to list"):
                S.view_id = None
                st.rerun()
            st.markdown(f"#### {esc(rec['question'])}")
            render_record(rec, key=f"view{rec['id']}", allow_delete=True)
            return
        S.view_id = None
    if not recs:
        st.info("No saved reports yet. Use **☆ Save report** on a finished report." if saved_only else "No research yet. Run your first query in **New Research**.")
        return
    for r in recs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([7, 1.2, 1.2], vertical_alignment="center")
            c1.markdown(f"**{esc(r['question'])}**  \n<span class='rp-muted'>{r['created_at'].replace('T', ' ')} · {r['depth']}"
                        f"{' · ⭐ saved' if r['saved'] else ''}</span>", unsafe_allow_html=True)
            if c2.button("View", key=f"v{r['id']}", width="stretch"):
                S.view_id = r["id"]
                st.rerun()
            if saved_only:
                if c3.button("Unsave", key=f"u{r['id']}", width="stretch"):
                    db.set_saved(r["id"], False)
                    st.rerun()
            elif c3.button("🗑 Delete", key=f"d{r['id']}", width="stretch"):
                delete_report(r["id"])
                st.rerun()


def page_settings() -> None:
    header()
    st.markdown("### ⚙️ Settings")
    with st.container(border=True):
        st.markdown("**Appearance**")
        choice = st.radio("Theme", ["Dark", "Light"], index=0 if S.theme == "dark" else 1, horizontal=True)
        if choice.lower() != S.theme:
            S.theme = choice.lower()
            st.rerun()
    with st.container(border=True):
        st.markdown("**Research defaults** (applied to the New Research page)")
        S.default_depth = st.radio("Default depth", list(cfg.DEPTH_PRESETS), index=list(cfg.DEPTH_PRESETS).index(S.default_depth), horizontal=True)
        S.default_limit = st.select_slider("Default source limit", cfg.SOURCE_LIMIT_OPTIONS, value=S.default_limit)
    with st.container(border=True):
        st.markdown("**Data**")
        st.caption(f"Local SQLite database: `{settings().db_path}` · {len(db.list_reports())} report(s)")
        confirm = st.checkbox("I understand this permanently deletes ALL research history")
        if st.button("Delete all history", disabled=not confirm):
            n = db.clear_all()
            S.current, S.view_id = None, None
            st.success(f"Deleted {n} report(s).")


PRESETS = {
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    "Groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "Google Gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash"),
    "Ollama (local)": ("http://localhost:11434/v1", "llama3.1"),
    "Custom": None,
}


def page_api() -> None:
    header()
    st.markdown("### 🔑 API Configuration")
    s = settings()
    S.setdefault("cfg_base", s.llm_base_url)
    S.setdefault("cfg_model", s.llm_model)

    def apply_preset():
        p = PRESETS.get(S.cfg_preset)
        if p:
            S.cfg_base, S.cfg_model = p

    with st.container(border=True):
        st.markdown("**LLM provider** (any OpenAI-compatible API)")
        st.selectbox("Provider preset", list(PRESETS), key="cfg_preset", on_change=apply_preset)
        c1, c2 = st.columns(2)
        c1.text_input("Base URL", key="cfg_base")
        c2.text_input("Model", key="cfg_model")
        llm_key = st.text_input("LLM API key", type="password", placeholder=f"current: {cfg.mask(s.llm_api_key)}")
        st.markdown("**Search provider**")
        tav_key = st.text_input("Tavily API key", type="password", placeholder=f"current: {cfg.mask(s.tavily_api_key)}")
        a, b = st.columns(2)
        if a.button("Apply for this session", type="primary", width="stretch"):
            S.overrides = {k: v for k, v in {"llm_base_url": S.cfg_base.strip(), "llm_model": S.cfg_model.strip(),
                                             "llm_api_key": llm_key.strip() or S.overrides.get("llm_api_key", ""),
                                             "tavily_api_key": tav_key.strip() or S.overrides.get("tavily_api_key", "")}.items() if v}
            S.service_ok = None
            st.success("Applied. Keys are kept in memory for this browser session only - they are never written to disk or the database.")
            st.rerun()
        if b.button("Test connections", width="stretch"):
            s = settings()
            if s.missing():
                st.error("Missing: " + ", ".join(s.missing()))
            else:
                for w in s.warnings():
                    st.warning(w)
                with st.spinner("Testing LLM and Tavily..."):
                    ok1, m1 = LLMClient(s).ping()
                    ok2, m2 = ws.ping(s.tavily_api_key)
                (st.success if ok1 else st.error)(("✅ " if ok1 else "❌ ") + m1)
                (st.success if ok2 else st.error)(("✅ " if ok2 else "❌ ") + m2)
                S.service_ok = ok1 and ok2
                S.service_msg = "; ".join(m for ok, m in ((ok1, m1), (ok2, m2)) if not ok)
    st.markdown(f"**Status:** LLM key `{cfg.mask(s.llm_api_key)}` · Tavily key `{cfg.mask(s.tavily_api_key)}` · model `{s.llm_model}`")
    with st.expander("How to configure keys permanently"):
        st.markdown(
            "**Local:** copy `.env.example` to `.env` and fill in `LLM_API_KEY`, `TAVILY_API_KEY` (and optionally `LLM_BASE_URL`, `LLM_MODEL`). "
            "Restart the app.\n\n**Streamlit Community Cloud:** App → Settings → Secrets, paste the same keys in TOML format "
            "(see `.streamlit/secrets.toml.example`).\n\nGet a Tavily key (free tier) at https://app.tavily.com.")


# =================================================================== router
page = S.page
if page == NAV[0]:
    page_new_research()
elif page == NAV[1]:
    history_list(False)
elif page == NAV[2]:
    history_list(True)
elif page == NAV[3]:
    page_settings()
else:
    page_api()
