"""
Real-API validation: runs the agent on the 5 suggested test questions and checks the results.

    python scripts/run_validation.py            # Quick depth (cheap)
    python scripts/run_validation.py --depth Standard

Needs LLM_API_KEY and TAVILY_API_KEY (see .env.example). For each question it verifies
  * the Tavily search tool actually ran (queries + results were retrieved)
  * every cited URL is reachable (HTTP status check)  -> "citations open the right sources"
  * how many cited sentences matched extracted evidence (heuristic claim check)
and saves each report to validation_results/ plus a summary table (validation_results/summary.md).
Nothing here is pre-baked: the numbers come from the live run on your machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.research_agent import ResearchAgent  # noqa: E402
from utils.config import EXAMPLE_QUESTIONS, load_settings  # noqa: E402


def url_status(url: str) -> str:
    try:
        r = requests.get(url, timeout=15, stream=True, headers={"User-Agent": "Mozilla/5.0 ResearchPilotValidator"})
        code = r.status_code
        r.close()
        return str(code)
    except requests.RequestException as e:
        return type(e).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="Quick", choices=["Quick", "Standard", "Deep"])
    args = ap.parse_args()
    s = load_settings()
    if s.missing():
        print("Missing:", ", ".join(s.missing()), "- copy .env.example to .env and fill it in.")
        return 2
    out = Path("validation_results")
    out.mkdir(exist_ok=True)
    rows = ["| # | Question | Status | Queries | Retrieved | Cited | URLs OK | Claim check | Warnings |", "|---|---|---|---|---|---|---|---|---|"]
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"[{i}/5] {q}")
        res = ResearchAgent(s).run(q, args.depth, 8)
        if res.status != "ok":
            rows.append(f"| {i} | {q} | **{res.status}**: {res.message[:60]} | – | – | – | – | – | – |")
            continue
        cited = [x for x in res.sources if x.get("cited")]
        ok = sum(1 for x in cited if url_status(x["url"]).startswith(("2", "3")))
        chk = res.trace["citation_check"]
        (out / f"q{i}.md").write_text(res.report_markdown, encoding="utf-8")
        rows.append(f"| {i} | {q} | ok | {len(res.trace['queries'])} | {res.trace['results_retrieved']} | {len(cited)} | {ok}/{len(cited)} | "
                    f"{chk['supported']}/{chk['checked']} | {len(res.trace['warnings'])} |")
        print("   ", rows[-1])
    (out / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
