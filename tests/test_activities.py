"""Tests for activities.py — per-session split of tokens and time by activity."""
from __future__ import annotations

import json

import pytest

from claude_dashboard.activities import (
    ActivityTracker,
    classify_bash,
    classify_path,
    classify_tool,
    is_review_agent,
)
from claude_dashboard.parser import merge_stats, parse_file


@pytest.mark.parametrize("path,want", [
    ("src/app/main.py", "code"),
    ("config/pipeline.yaml", "code"),
    ("tests/test_store.py", "test"),
    ("pkg/test_utils.py", "test"),
    ("web/button.spec.ts", "test"),
    ("README.md", "docs"),
    ("openspec/changes/x/tasks.md", "docs"),
    ("docs/diagram.svg", "docs"),
])
def test_classify_path(path, want):
    assert classify_path(path) == want


@pytest.mark.parametrize("cmd,want", [
    ("uv run pytest -m 'not slow'", "test"),
    ("npm run test", "test"),
    ('grep -n "^A\\|^B" src/x.py | head -5', "explore"),
    ("cd /repo && git -C sub log --oneline -3 && sed -n 1,20p a.py", "explore"),
    ("env -u SSL_CERT_FILE gh pr view 65 --json state", "explore"),
    ("aws batch describe-jobs --jobs 1", "explore"),
    ("cat > notes/plan.md <<'EOF'\nhello\nEOF", "docs"),
    ("echo x >> src/mod.py", "code"),
    ("cat >> $WS/progress.md <<'EOF'\nTask 4 done\nEOF", "docs"),
    ("ls > /dev/null", "explore"),
    ("python3 - <<'EOF'\nimport json\nprint(json.load(open('a.json')))\nEOF", "explore"),
    ("python3 - <<'EOF'\nimport pathlib\npathlib.Path('tests/test_a.py').write_text('x')\nEOF", "test"),
    ("git commit -m 'x' && git push", "other"),
    ("git worktree add ../wt -b me/DP-1-test-hygiene", "other"),
    ("nohup uv run python scripts/validation/run_attribution.py --max-folds 4 > /tmp/a.log 2>&1 &", "eval"),
    ("uv run python scripts/training/ecpm_grid_search.py --round r3 > run.log", "train"),
    ("uv run python batch_run.py --name golden-dau --tag x", "test"),
    ("aws batch submit-job --job-name retrain-ecpm --job-queue q", "train"),
    ("docker run --rm img python -m swc_forecasting.smoke run", "test"),
    ("uv run python scripts/deploy.py", "other"),
    ("until grep -q DONE /tmp/a.log; do sleep 30; done", "wait"),
    ("gh run watch 123 --exit-status", "wait"),
    ("sed -i '' 's/a/b/' src/x.py", "other"),
])
def test_classify_bash(cmd, want):
    assert classify_bash(cmd) == want


def test_classify_tool():
    assert classify_tool("Edit", {"file_path": "src/a.py"}) == "code"
    assert classify_tool("Write", {"file_path": "docs/a.md"}) == "docs"
    assert classify_tool("Read", {"file_path": "src/a.py"}) == "explore"
    assert classify_tool("mcp__jira__jira_get_issue", {}) == "explore"
    assert classify_tool("mcp__jira__jira_create_issue", {}) == "other"
    assert classify_tool("Agent", {}) == "other"


def test_is_review_agent():
    assert is_review_agent("Review PR #101")
    assert is_review_agent("Delta review of commit")
    assert not is_review_agent("Implement task 3.7")
    assert not is_review_agent(None)


# ── tracker ────────────────────────────────────────────────────────────────

def _ts(sec: int) -> str:
    return f"2026-01-01T00:{sec // 60:02d}:{sec % 60:02d}.000Z"


def _prompt(sec, text="go"):
    return {"type": "user", "timestamp": _ts(sec), "promptId": "p",
            "message": {"role": "user", "content": text}}


def _call(sec, rid, tools=(), out=10, cache_read=100):
    content = [{"type": "tool_use", "id": f"{rid}-{i}", "name": n, "input": inp}
               for i, (n, inp) in enumerate(tools)] or [{"type": "text", "text": "ok"}]
    return {"type": "assistant", "timestamp": _ts(sec), "requestId": rid,
            "message": {"model": "claude-sonnet-4-5", "content": content,
                        "usage": {"input_tokens": 1, "output_tokens": out,
                                  "cache_read_input_tokens": cache_read}}}


def _result(sec, tool_id):
    return {"type": "user", "timestamp": _ts(sec),
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]}}


def _run(lines, review=False):
    t = ActivityTracker(review=review)
    for line in lines:
        t.feed(line)
    return t.result()


def _tok(bucket):
    return sum(sum(v.values()) for v in bucket["tokens_by_model"].values())


def test_tracker_attributes_tokens_and_time():
    r = _run([
        _prompt(0),
        _call(3, "r1", [("Read", {"file_path": "a.py"})]),        # 3s generating
        _result(5, "r1-0"),                                       # 2s reading
        _call(9, "r2", [("Edit", {"file_path": "a.py"})]),        # 4s generating
        _result(10, "r2-0"),                                      # 1s editing
        _call(12, "r3", [("Bash", {"command": "pytest -q"})]),    # 2s
        _result(42, "r3-0"),                                      # 30s tests
        _call(44, "r4"),                                          # 2s final answer
        _prompt(600),                                             # user idle: not counted
        _call(601, "r5"),
    ])
    by = r["by_activity"]
    assert by["explore"]["time_ms"] == 5000 + 2000 + 1000
    assert by["code"]["time_ms"] == 5000
    assert by["test"]["time_ms"] == 32000
    assert by["explore"]["calls"] == 3
    assert _tok(by["code"]) == 111
    # r1 led to code; r4 and r5 ended their turns without an action.
    assert r["explore_next"]["code"]["calls"] == 1
    assert r["explore_next"]["other"]["calls"] == 2


def test_tracker_dedupes_streamed_lines_last_wins():
    first = _call(1, "r1", [("Edit", {"file_path": "a.py"})], out=1)
    last = _call(2, "r1", out=50)  # same requestId, final usage
    r = _run([_prompt(0), first, last])
    assert r["by_activity"]["code"]["calls"] == 1
    assert _tok(r["by_activity"]["code"]) == 1 + 50 + 100


def test_tracker_skips_foreground_agent_wait():
    r = _run([
        _prompt(0),
        _call(1, "r1", [("Agent", {"description": "Implement x"})]),
        _result(901, "r1-0"),  # 15 min of subagent work, timed in its own file
    ])
    assert r["by_activity"]["other"]["time_ms"] == 1000


def test_review_agent_is_all_review():
    r = _run([_prompt(0), _call(1, "r1", [("Edit", {"file_path": "a.py"})])], review=True)
    assert set(r["by_activity"]) == {"review"}


def test_parse_and_merge_carry_activities(tmp_path):
    main = tmp_path / "s.jsonl"
    sub = tmp_path / "agent.jsonl"
    main.write_text("\n".join(json.dumps(x) for x in [
        _prompt(0), _call(2, "r1", [("Edit", {"file_path": "a.py"})])]))
    sub.write_text("\n".join(json.dumps(x) for x in [
        _prompt(0), _call(4, "s1", [("Read", {"file_path": "a.py"})])]))
    stats = parse_file(main)
    merge_stats(stats, parse_file(sub, review=True))
    by = stats["activities"]["by_activity"]
    assert by["code"]["time_ms"] == 2000
    assert by["review"]["time_ms"] == 4000


def test_wait_takes_the_activity_of_the_run_it_waits_on():
    r = _run([
        _prompt(0),
        _call(1, "r1", [("Bash", {"command": "nohup uv run python run_attribution.py > /tmp/s/att.log 2>&1 &"})]),
        _result(2, "r1-0"),
        _call(3, "r2", [("Bash", {"command": "until grep -q exit /tmp/s/att.log; do sleep 30; done"})]),
        _result(603, "r2-0"),  # 10 min waiting on the evaluation
    ])
    assert set(r["by_activity"]) == {"eval"}
    assert r["by_activity"]["eval"]["time_ms"] == 603000


@pytest.mark.parametrize("cmd,want", [
    ("gh run watch 36412627335 --exit-status --interval 30", "test"),
    ("timeout 590 bash -c 'while ! grep -qE \"[0-9]+ (passed|failed)\" /tmp/x; do sleep 10; done'", "test"),
    ("until grep -q 'eval_revenue_torch exit' /tmp/j.txt; do sleep 60; done", "eval"),
    ("until [ -s /tmp/unknown.txt ]; do sleep 5; done", "other"),
])
def test_unlinked_wait_uses_its_own_words(cmd, want):
    r = _run([_prompt(0), _call(1, "r1", [("Bash", {"command": cmd})])])
    assert set(r["by_activity"]) == {want}


def test_monitor_is_a_wait():
    r = _run([
        _prompt(0),
        _call(1, "r1", [("Bash", {"command": "uv run pytest tests > /tmp/s/t.out 2>&1 &"})]),
        _call(2, "r2", [("Monitor", {"command": "tail -f /tmp/s/t.out | grep --line-buffered FAIL"})]),
    ])
    assert set(r["by_activity"]) == {"test"}


def test_user_answers_are_not_timed():
    r = _run([
        _prompt(0),
        _call(1, "r1", [("AskUserQuestion", {"questions": []})]),
        _result(3601, "r1-0"),  # an hour until the user answered
    ])
    assert r["by_activity"]["other"]["time_ms"] == 1000


def test_wait_on_a_background_run_output_file():
    run = _call(1, "r1", [("Bash", {"command": "uv run pytest -m slow", "run_in_background": True})])
    started = _result(2, "r1-0")
    started["message"]["content"][0]["content"] = (
        "Command running in background. Output is being written to: /tmp/s/tasks/b7.output")
    r = _run([
        _prompt(0), run, started,
        _call(3, "r2", [("Bash", {"command": "until [ -s /tmp/s/tasks/b7.output ]; do sleep 5; done"})]),
        _result(303, "r2-0"),
    ])
    assert set(r["by_activity"]) == {"test"}


def test_wait_on_a_batch_job_id():
    submit = _call(1, "r1", [("Bash", {"command": "aws batch submit-job --job-name grid-search-ecpm"})])
    out = _result(2, "r1-0")
    out["message"]["content"][0]["content"] = '{"jobId": "0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b"}'
    r = _run([
        _prompt(0), submit, out,
        _call(3, "r2", [("Bash", {"command": (
            "while aws batch describe-jobs --jobs 0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b"
            " | grep -q RUNNING; do sleep 60; done")})]),
        _result(3603, "r2-0"),
    ])
    assert set(r["by_activity"]) == {"train"}
