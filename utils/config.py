"""
Configuration & secret management.

Secrets are read (in this priority order) from:
  1. Real environment variables
  2. A local `.env` file (never committed - see .gitignore)
  3. Streamlit secrets (`.streamlit/secrets.toml` locally, or the "Secrets"
     box on Streamlit Community Cloud)

Keys are never written to disk by the app, never logged, and never added to
generated reports. `redact()` scrubs any known key out of text before it is
logged or shown in an error message.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_NAME = "ResearchPilot AI"
APP_SUBTITLE = "Your Intelligent AI Research Assistant"

# Depth presets: how much work the agent is allowed to do.
#   max_searches  - number of Tavily queries the agent may run
#   max_steps     - maximum LLM "decide the next tool" turns
#   max_pages     - maximum webpages fetched in full (Tool B)
#   results_per_search - Tavily results requested per query
DEPTH_PRESETS: dict[str, dict] = {
    "Quick": {"max_searches": 2, "max_steps": 6, "max_pages": 2, "results_per_search": 4, "search_depth": "basic"},
    "Standard": {"max_searches": 4, "max_steps": 10, "max_pages": 4, "results_per_search": 5, "search_depth": "basic"},
    "Deep": {"max_searches": 7, "max_steps": 16, "max_pages": 8, "results_per_search": 6, "search_depth": "advanced"},
}
SOURCE_LIMIT_OPTIONS = [5, 8, 10, 15, 20]

EXAMPLE_QUESTIONS = [
    "What are the latest developments in artificial intelligence?",
    "How does agentic AI differ from traditional generative AI?",
    "What are the recent advances in drug delivery systems?",
    "What are the applications of machine learning in healthcare?",
    "What are the advantages and limitations of renewable energy?",
]


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:  # Streamlit secrets are optional
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


@dataclass(frozen=True)
class Settings:
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    tavily_api_key: str = ""
    request_timeout: int = 20
    db_path: str = str(BASE_DIR / "data" / "researchpilot.db")

    # ---- validation -------------------------------------------------
    def missing(self) -> list[str]:
        """Names of required settings that are empty."""
        out = []
        if not self.llm_api_key:
            out.append("LLM_API_KEY")
        if not self.tavily_api_key:
            out.append("TAVILY_API_KEY")
        return out

    def warnings(self) -> list[str]:
        """Soft format checks (we never reject a key, providers differ)."""
        w = []
        if self.tavily_api_key and not self.tavily_api_key.startswith("tvly-"):
            w.append("Tavily keys normally start with 'tvly-'. Double-check yours.")
        if self.llm_base_url and not self.llm_base_url.startswith(("http://", "https://")):
            w.append("LLM_BASE_URL must start with http:// or https://")
        return w

    @property
    def ready(self) -> bool:
        return not self.missing()

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v not in (None, "")})


def load_settings() -> Settings:
    return Settings(
        llm_api_key=_secret("LLM_API_KEY") or _secret("OPENAI_API_KEY"),
        llm_base_url=_secret("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_model=_secret("LLM_MODEL", "gpt-4o-mini"),
        tavily_api_key=_secret("TAVILY_API_KEY"),
        request_timeout=int(_secret("REQUEST_TIMEOUT", "20") or 20),
        db_path=_secret("DB_PATH", str(BASE_DIR / "data" / "researchpilot.db")),
    )


def mask(key: str) -> str:
    if not key:
        return "(not set)"
    return key[:4] + "…" + key[-2:] if len(key) > 10 else "••••"


def redact(text: str, settings: Settings | None = None) -> str:
    """Remove any configured API key from a string (logs, error messages)."""
    s = settings or load_settings()
    for k in (s.llm_api_key, s.tavily_api_key):
        if k and len(k) > 6:
            text = text.replace(k, "***REDACTED***")
    return text
