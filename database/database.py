"""
SQLite persistence for research history and saved reports.

Table `reports` keeps the question, the final Markdown report, the source list
(JSON), the evidence (JSON) and the research trace (JSON). "Saved" reports are
simply history rows with saved=1 (a bookmark).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    depth TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_md TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    saved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at DESC);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn = sqlite3.connect(":memory:") if path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._memory_conn or sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

    def save_report(self, result: dict) -> int:
        """Store a completed ResearchResult (as dict). Returns the new row id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reports (question, depth, created_at, report_md, sources_json, evidence_json, trace_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (result["question"], result.get("depth", "Standard"), datetime.now().isoformat(timespec="seconds"),
                 result["report_markdown"], json.dumps(result["sources"]), json.dumps(result["evidence"]),
                 json.dumps(result["trace"])))
            return int(cur.lastrowid)

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["report_markdown"] = d.pop("report_md")
        d["sources"] = json.loads(d.pop("sources_json"))
        d["evidence"] = json.loads(d.pop("evidence_json"))
        d["trace"] = json.loads(d.pop("trace_json"))
        d["saved"] = bool(d["saved"])
        return d

    def get_report(self, report_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return self._row(r) if r else None

    def list_reports(self, saved_only: bool = False, limit: int = 200) -> list[dict]:
        """Lightweight listing (no big JSON blobs)."""
        sql = "SELECT id, question, depth, created_at, saved FROM reports"
        if saved_only:
            sql += " WHERE saved=1"
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(sql, (limit,)).fetchall()
        return [dict(r, saved=bool(r["saved"])) for r in rows]

    def set_saved(self, report_id: int, saved: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE reports SET saved=? WHERE id=?", (int(saved), report_id))

    def delete_report(self, report_id: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM reports WHERE id=?", (report_id,)).rowcount > 0

    def clear_all(self) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM reports").rowcount
