"""Usage report for one project: where Claude's money and agent-time go, and
how that moves over time.

    python -m claude_dashboard.report /path/to/repo [/path/to/other-clone ...]
        [--weeks 8] [--period-days 14] [--json] [--snapshot DIR]

A project is one or more path prefixes (a repo and its worktrees). The report
refreshes the cache from ~/.claude first, then prints:

- a weekly series: cost, agent-hours, sessions, lines changed, cost per 1k
  lines changed, and each activity's share of cost;
- the last `period-days` against the period before: activity cost / time
  shares, the heaviest commands of each activity, and review passes.

Activities, commands and review passes come from activities.py. Cost, time,
commands and lines changed count on the day they happened, so a session that
runs for weeks spreads over them; a session counts as active in every week /
period it worked in; review passes count by their start.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from claude_dashboard.activities import ACTIVITIES, OTHER_COMMANDS, summarize_reviews
from claude_dashboard.scanner import refresh
from claude_dashboard.store import (
    Store,
    _add_flat,
    _as_obj,
    _loads_obj,
    _num,
    _review_passes,
    _top_items,
    db_path,
)

LABELS = {
    "code": "Writing code", "docs": "Docs / decisions", "test": "Testing",
    "eval": "Evaluation", "train": "Training", "review": "Review",
    "build": "Build / release", "git": "Git / PRs", "ops": "Infra / ops",
    "coord": "Coordination", "other": "Other", "explore": "Reading code & files",
    "data": "Data queries", "web": "Web research", "think": "Thinking / answers",
}


def _day(ts: str | None) -> date | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).astimezone().date()
    except ValueError:
        return None


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def build_report(rows: list, today: date, weeks: int, period_days: int) -> dict:
    """The report data from session rows (Store.session_rows)."""
    first_week = _monday(today) - timedelta(weeks=weeks - 1)
    cur_start = today - timedelta(days=period_days - 1)
    prev_start = cur_start - timedelta(days=period_days)

    weekly: dict[str, dict] = {}
    periods = {"current": _empty_period(cur_start, today),
               "previous": _empty_period(prev_start, cur_start - timedelta(days=1))}
    passes = []
    for r in rows:
        acts = _loads_obj(r["activities_json"])
        lines = _as_obj(acts.get("lines_by_day"))
        progs_by_day = _as_obj(acts.get("programs_by_day"))
        active: set[str] = set()  # weeks / periods this session worked in
        days_in = {name: 0 for name in periods}
        for day_s, by_act in _as_obj(acts.get("by_day")).items():
            try:
                d = date.fromisoformat(day_s)
            except ValueError:
                continue
            buckets = []
            if d >= first_week:
                wk = _monday(d).isoformat()
                buckets.append((wk, weekly.setdefault(wk, _empty_week())))
            buckets += [(name, p) for name, p in periods.items() if p["start"] <= d <= p["end"]]
            for name, p in periods.items():
                if p["start"] <= d <= p["end"]:
                    days_in[name] += 1
                    cost = sum(_num(_as_obj(b).get("cost_usd")) for b in _as_obj(by_act).values())
                    top = p["top_sessions"].setdefault(r["session_id"], {
                        "session_id": r["session_id"], "title": r["custom_title"],
                        "started": (r["started_at"] or "")[:10], "cost_usd": 0.0, "active_days": 0})
                    top["cost_usd"] += cost
                    top["active_days"] += 1
            for key, x in buckets:
                _add_flat(x["activities"], _as_obj(by_act))
                x["lines_changed"] += int(lines.get(day_s) or 0)
                if key not in active:
                    active.add(key)
                    x["sessions"] += 1
                if "programs" in x:
                    for act, sigs in _as_obj(progs_by_day.get(day_s)).items():
                        _add_flat(x["programs"].setdefault(act, {}), _as_obj(sigs))
        passes += _review_passes(r["session_id"], r["reviews_json"])

    for p in periods.values():
        p["top_sessions"] = sorted(p["top_sessions"].values(), key=lambda t: -t["cost_usd"])[:5]
        p["reviews"] = summarize_reviews(
            [x for x in passes if p["start"] <= (_day(x.get("started_at")) or date.min) <= p["end"]])
        for act, sigs in p["programs"].items():
            p["programs"][act] = _top_items(sigs, keep=10)
        _totals(p)
        p["start"], p["end"] = p["start"].isoformat(), p["end"].isoformat()
    for w in weekly.values():
        _totals(w)
    return {"generated": today.isoformat(), "weekly": dict(sorted(weekly.items())), **periods}


def _empty_week() -> dict:
    return {"activities": {}, "sessions": 0, "lines_changed": 0}


def _empty_period(start: date, end: date) -> dict:
    return {"start": start, "end": end, "activities": {}, "programs": {},
            "sessions": 0, "lines_changed": 0, "top_sessions": {}}


def _totals(x: dict):
    acts = x["activities"].values()
    x["cost_usd"] = sum(b["cost_usd"] for b in acts)
    x["agent_hours"] = sum(b["time_ms"] for b in acts) / 3.6e6
    x["cost_per_1k_lines"] = (1000 * x["cost_usd"] / x["lines_changed"]) if x["lines_changed"] else None
    # Tokens per API call ≈ the context each call re-reads (cache reads dominate):
    # the session-lifetime lever.
    calls = sum(b["calls"] for b in acts)
    x["context_k_per_call"] = sum(b["tokens"] for b in acts) / calls / 1000 if calls else None


# ── markdown ─────────────────────────────────────────────────────────────────

def _share(x: dict, act: str, key: str) -> float:
    total = sum(b[key] for b in x["activities"].values())
    return 100 * x["activities"].get(act, {}).get(key, 0) / total if total else 0.0


def to_markdown(rep: dict, project: str) -> str:
    cur, prev = rep["current"], rep["previous"]
    acts = [a for a in ACTIVITIES if a in cur["activities"] or a in prev["activities"]]
    top = sorted(acts, key=lambda a: -cur["activities"].get(a, {}).get("cost_usd", 0))[:6]
    out = [f"# Claude usage — {project}", f"Generated {rep['generated']}.", ""]

    out += ["## Weekly", "", "| Week | Cost | Agent-h | Active sessions | Lines changed | $/1k lines | Context k/call | "
            + " | ".join(f"{LABELS[a]} %" for a in top) + " |",
            "|" + "---|" * (7 + len(top))]
    for wk, w in rep["weekly"].items():
        per_k = f"{w['cost_per_1k_lines']:.2f}" if w["cost_per_1k_lines"] is not None else "—"
        ctx = f"{w['context_k_per_call']:.0f}" if w["context_k_per_call"] is not None else "—"
        out.append(f"| {wk} | ${w['cost_usd']:.0f} | {w['agent_hours']:.1f} | {w['sessions']} | "
                   f"{w['lines_changed']} | {per_k} | {ctx} | "
                   + " | ".join(f"{_share(w, a, 'cost_usd'):.0f}" for a in top) + " |")

    def headline(p):
        per_k = f"${p['cost_per_1k_lines']:.2f}" if p["cost_per_1k_lines"] is not None else "—"
        ctx = f"{p['context_k_per_call']:.0f}k" if p["context_k_per_call"] is not None else "—"
        return (f"${p['cost_usd']:.0f}, {p['agent_hours']:.1f} agent-h, {p['sessions']} sessions, "
                f"{p['lines_changed']} lines changed, {per_k} per 1k lines, {ctx} tokens of context per call")

    out += ["", f"## Last {cur['start']}..{cur['end']} vs {prev['start']}..{prev['end']}", "",
            f"- Current: {headline(cur)}", f"- Previous: {headline(prev)}", "",
            "| Activity | Cost now | Cost % now | Cost % before | Δ pp | Time % now | Time % before | Δ pp |",
            "|---|---|---|---|---|---|---|---|"]
    for a in sorted(acts, key=lambda a: -cur["activities"].get(a, {}).get("cost_usd", 0)):
        c_now, c_prev = _share(cur, a, "cost_usd"), _share(prev, a, "cost_usd")
        t_now, t_prev = _share(cur, a, "time_ms"), _share(prev, a, "time_ms")
        out.append(f"| {LABELS[a]} | ${cur['activities'].get(a, {}).get('cost_usd', 0):.0f} | "
                   f"{c_now:.0f} | {c_prev:.0f} | {c_now - c_prev:+.0f} | "
                   f"{t_now:.0f} | {t_prev:.0f} | {t_now - t_prev:+.0f} |")

    out += ["", "## Costliest sessions (current period)", ""]
    for t in cur["top_sessions"]:
        share = 100 * t["cost_usd"] / cur["cost_usd"] if cur["cost_usd"] else 0
        out.append(f"- ${t['cost_usd']:.0f} ({share:.0f}%) — {t['title'] or t['session_id'][:8]}, "
                   f"started {t['started']}, active {t['active_days']} day(s) in the period")

    out += ["", "## Heaviest commands (current period, by time)", ""]
    for a in top:
        progs = cur["programs"].get(a) or {}
        rows = sorted(((k, v) for k, v in progs.items() if k != OTHER_COMMANDS),
                      key=lambda kv: -kv[1]["time_ms"])[:5]
        if rows:
            out.append(f"- **{LABELS[a]}**: " + "; ".join(
                f"`{k}` {v['time_ms'] / 60000:.0f} min / {v['calls']} calls / ${v['cost_usd']:.0f}"
                for k, v in rows))

    out += ["", "## Reviews", ""]
    for name, p in (("Current", cur), ("Previous", prev)):
        rv = p["reviews"]
        if not rv["passes"]:
            out.append(f"- {name}: no review subagents")
            continue
        kinds = ", ".join(f"{k} {v['count']} (${v['cost_usd']:.0f})" for k, v in rv["kinds"].items())
        dist = ", ".join(f"{k} pass{'es' if k != '1' else ''}: {n}"
                         for k, n in sorted(rv["passes_per_target"].items()))
        out.append(f"- {name}: {rv['passes']} passes over {rv['targets']} targets "
                   f"({rv['passes'] / rv['targets']:.2f} per target); {kinds}; {dist}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="project path prefix(es): a repo (its worktrees "
                    "included) and any other clone, as separate arguments")
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--period-days", type=int, default=14)
    ap.add_argument("--json", action="store_true", help="print the report data as JSON")
    ap.add_argument("--no-refresh", action="store_true", help="skip re-scanning ~/.claude")
    ap.add_argument("--snapshot", metavar="DIR",
                    help="also write the report data to DIR/snapshots/<today>.json")
    args = ap.parse_args(argv)

    store = Store(db_path())
    if not args.no_refresh:
        report = refresh(store)
        for err in report.errors:
            print(f"warning: {err}", file=sys.stderr)
    today = datetime.now().astimezone().date()
    rows = store.session_rows(args.paths)
    if not rows:
        known = sorted({p["project_path"] for p in store.list_projects(include_hidden=True)})
        store.close()
        print(f"error: no session under {args.paths}. Known project paths:\n  "
              + "\n  ".join(known), file=sys.stderr)
        return 2
    rep = build_report(rows, today, args.weeks, args.period_days)
    store.close()
    if args.snapshot:
        out = Path(args.snapshot).expanduser() / "snapshots" / f"{today.isoformat()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str))
        print(f"snapshot: {out}", file=sys.stderr)
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(to_markdown(rep, ", ".join(args.paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
