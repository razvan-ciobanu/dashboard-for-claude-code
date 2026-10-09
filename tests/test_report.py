"""Tests for report.py — the per-project usage report."""
from __future__ import annotations

import json
from datetime import date

from claude_dashboard.report import build_report, to_markdown


def _bucket(cost, ms):
    return {"tokens_by_model": {}, "cost_usd": cost, "time_ms": ms, "calls": 1}


def _row(sid, by_day, lines, programs_by_day=None, reviews=None):
    return {
        "session_id": sid, "started_at": None, "code_lines_added": 0, "code_lines_removed": 0,
        "activities_json": json.dumps({"by_day": by_day, "lines_by_day": lines,
                                       "programs_by_day": programs_by_day or {}}),
        "reviews_json": json.dumps({"passes": reviews or []}),
    }


def test_a_long_session_spreads_over_its_days():
    # One session working in both periods: cost, lines and commands land on their days.
    rows = [_row("s", {
        "2026-10-01": {"test": _bucket(10.0, 3_600_000)},
        "2026-10-08": {"code": _bucket(30.0, 3_600_000)},
    }, {"2026-10-01": 500, "2026-10-08": 1500}, programs_by_day={
        "2026-10-08": {"code": {"Edit *.py": _bucket(30.0, 3_600_000)}},
    })]
    rep = build_report(rows, today=date(2026, 10, 9), weeks=2, period_days=7)
    cur, prev = rep["current"], rep["previous"]
    assert (cur["cost_usd"], cur["lines_changed"], cur["sessions"]) == (30.0, 1500, 1)
    assert (prev["cost_usd"], prev["lines_changed"], prev["sessions"]) == (10.0, 500, 1)
    assert cur["cost_per_1k_lines"] == 20.0
    assert list(cur["programs"]["code"]) == ["Edit *.py"]
    assert set(rep["weekly"]) == {"2026-09-28", "2026-10-05"}
    md = to_markdown(rep, "/repo")
    assert "`Edit *.py` 60 min" in md
