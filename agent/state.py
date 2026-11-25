"""Shared data structures for one research run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    step: int
    tool: str
    arguments: dict
    summary: str
    ok: bool = True
    origin: str = "llm"  # "llm" = chosen by the model, "autopilot" = safety-net fallback

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ResearchState:
    """Mutable working memory of the agent while it researches."""
    question: str
    depth: str
    cfg: dict
    source_limit: int
    analysis: dict = field(default_factory=dict)     # Stage 1 output
    plan: dict = field(default_factory=dict)         # Stage 2 output
    queries_run: list[str] = field(default_factory=list)
    sources: dict[int, dict] = field(default_factory=dict)   # internal id -> source
    results_retrieved: int = 0                       # raw hits before de-duplication
    duplicates_removed: int = 0
    evidence: list[dict] = field(default_factory=list)
    tool_log: list[ToolCallRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    pages_read: int = 0
    analyzed: bool = False
    finished: bool = False
    finish_reason: str = ""
    next_id: int = 1

    def status_line(self) -> str:
        with_ev = len({e["source_id"] for e in self.evidence})
        return (
            f"STATUS: searches {len(self.queries_run)}/{self.cfg['max_searches']}, "
            f"sources found {len(self.sources)}, pages read {self.pages_read}/{self.cfg['max_pages']}, "
            f"evidence items {len(self.evidence)} from {with_ev} sources, "
            f"sources analyzed: {'yes' if self.analyzed else 'no'}."
        )
