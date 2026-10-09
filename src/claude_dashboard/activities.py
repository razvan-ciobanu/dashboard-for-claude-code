"""Split a session's tokens and time across activities (code, docs, tests, ...).

The unit is one API call (one requestId). Its tokens are that call's final usage;
its time is the generation latency before it plus the run time of the tools it
issued. Time spent waiting on the user (or on a queued/background notification)
is not attributed to anything.

A call's activity comes from the tools it issued — see `classify_tool`. A call
that only reads (or issues no tool) is `explore`; for those we also record which
activity came next in the same turn (`explore_next`), so the UI can show the
reading either as its own activity or folded into the work it led to.

Every call of a review subagent (its description mentions "review") is `review`.
Subagent time is summed (agent-time), so it can exceed the session's wall time.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

ACTIVITIES = ("code", "docs", "test", "review", "explore", "other")

# Highest first: a call that both reads and edits code is `code`.
_PRIORITY = {"test": 0, "code": 1, "docs": 2, "other": 3, "explore": 4}

_EDIT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}
_READ_TOOLS = {"Read", "Grep", "Glob", "LS", "WebSearch", "WebFetch", "ToolSearch"}
_READ_MCP = re.compile(r"(^|_)(get|list|read|search|query|fetch|describe|view)(_|$)")

_DOC_EXT = {".md", ".mdx", ".txt", ".rst", ".adoc"}
_DOC_DIR = re.compile(r"(^|/)(docs?|openspec|memory)/")
_TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)test_[^/]+$|_test\.\w+$|\.(test|spec)\.\w+$")

_TEST_CMD = re.compile(
    r"\b(pytest|py\.test|unittest|tox|nox|jest|vitest|mocha"
    r"|(npm|pnpm|yarn|bun)\s+(run\s+)?test|go\s+test|cargo\s+test|make\s+test)\b"
)
# A shell write into a file: `> path`, `>> path`, `tee [-a] path`.
_SHELL_WRITE = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*['\"]?([~\w./-]+\.\w+)")
_READ_CMDS = {
    "cat", "head", "tail", "less", "grep", "rg", "ls", "find", "fd", "wc", "du",
    "tree", "jq", "file", "stat", "diff", "which", "pwd", "echo", "printf", "sqlite3",
    "sort", "uniq", "cut", "tr", "awk", "column", "nl", "comm", "basename", "dirname",
    "realpath", "date", "true", "false", "test", "[", "unset", "export", "sleep",
}
_READ_GIT = {
    "log", "diff", "show", "status", "blame", "branch", "remote", "ls-files", "ls-tree",
    "grep", "fetch", "rev-parse", "cat-file", "describe", "shortlog", "reflog",
}
_READ_SUB = re.compile(r"^(view|list|ls|checks|diff|status|get|describe|search|api)\b|^(get|describe|list)-")
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)(?:\n\1\s*(?:\n|$)|$)", re.DOTALL)
_PY_WRITE = re.compile(r"write_text\(|write_bytes\(|open\([^)]*['\"][wa]b?['\"]|\.to_(csv|parquet|json)\(")
_PATH_LITERAL = re.compile(r"['\"]([~\w./-]+\.(?:py|md|mdx|txt|rst|yaml|yml|json|toml|js|ts|sql|sh))['\"]")
_SCRATCH = re.compile(r"/(tmp|scratchpad)/|^/private/")


def classify_path(path: str) -> str:
    p = path.replace("\\", "/")
    if _TEST_PATH.search(p):
        return "test"
    dot = p.rfind(".")
    if (dot > p.rfind("/") and p[dot:].lower() in _DOC_EXT) or _DOC_DIR.search(p):
        return "docs"
    return "code"


def classify_bash(command: str) -> str:
    bodies = [m.group(2) for m in _HEREDOC.finditer(command)]
    shell = _HEREDOC.sub(" ", command)
    if _TEST_CMD.search(shell):
        return "test"
    # A heredoc script that writes a file (python - <<EOF ... write_text ...).
    for body in bodies:
        if _PY_WRITE.search(body):
            m = _PATH_LITERAL.search(body)
            return classify_path(m.group(1)) if m else "code"
    bare = _QUOTED.sub("Q", shell)
    for target in _SHELL_WRITE.findall(bare):
        if not target.startswith("/dev/"):
            return classify_path(target)
    # Read-only when every command of a pipeline/sequence is a reader.
    parts = [p.strip() for p in re.split(r"\|\||&&|[|;\n()]", bare) if p.strip()]
    if parts and all(_is_read_cmd(p, bool(bodies)) for p in parts):
        return "explore"
    return "other"


def _is_read_cmd(part: str, heredoc: bool) -> bool:
    words = part.split()
    # Leading wrappers: env [-u X] VAR=v, cd, time, timeout N, xargs, uv run [--flags].
    while words:
        w = words[0]
        if w in {"cd", "do", "done", "then", "fi", "else", "{", "}"}:
            return True
        if "=" in w or w in {"time", "nohup", "xargs", "command"}:
            words = words[1:]
        elif w in {"env", "timeout"} or (w.startswith("-") and len(words) > 1):
            words = words[2:] if w in {"-u", "-C", "timeout"} else words[1:]
        elif w == "uv" and len(words) > 1 and words[1] == "run":
            words = [x for x in words[2:] if not x.startswith("--")]
        else:
            break
    if not words:
        return True
    head = words[0].rsplit("/", 1)[-1]
    rest = words[1:]
    if head == "sed":
        return "-n" in rest and "-i" not in rest
    if head == "git":
        while rest and rest[0] in {"-C", "-c"}:
            rest = rest[2:]
        return bool(rest) and (rest[0] in _READ_GIT or rest[:2] == ["worktree", "list"])
    if head in {"gh", "aws", "kubectl"}:
        return any(_READ_SUB.match(x) for x in rest[:3] if not x.startswith("-"))
    if head.startswith("python"):
        # Inline analysis (`python - <<EOF` without writes) or a scratch script.
        return (rest[:1] in (["-"], ["-c"]) and heredoc) or rest[:1] == ["-c"] \
            or bool(rest and _SCRATCH.search(rest[0]))
    return head in _READ_CMDS


def classify_tool(name: str, inp: dict) -> str:
    if name in _EDIT_TOOLS:
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return classify_path(path) if path else "code"
    if name == "Bash":
        return classify_bash(inp.get("command") or "")
    if name in _READ_TOOLS:
        return "explore"
    if name.startswith("mcp__") and _READ_MCP.search(name.rsplit("__", 1)[-1]):
        return "explore"
    return "other"


def is_review_agent(description: str | None) -> bool:
    return bool(description) and "review" in description.lower()


def _ms(ts: Any) -> float | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp() * 1000
    except ValueError:
        return None


def empty_bucket() -> dict:
    return {"tokens_by_model": {}, "time_ms": 0, "calls": 0}


class ActivityTracker:
    """Fed every transcript line in order by parse_file; `result()` at the end."""

    def __init__(self, review: bool = False):
        self.review = review
        self.calls: dict[str, dict] = {}   # requestId -> call
        self.order: list[str] = []
        self.tool_owner: dict[str, tuple[str, str]] = {}  # tool_use id -> (requestId, tool)
        self.turn = 0                       # bumped at every user/notification boundary
        self.last_ms: float | None = None

    def feed(self, line: dict):
        ltype = line.get("type")
        now = _ms(line.get("timestamp"))
        if ltype == "assistant":
            self._assistant(line, now)
        elif ltype == "user":
            self._user(line, now)

    def _gap(self, now: float | None) -> int:
        if now is None or self.last_ms is None:
            return 0
        return max(0, int(now - self.last_ms))

    def _assistant(self, line: dict, now: float | None):
        msg = line.get("message") or {}
        model = msg.get("model") or ""
        rid = line.get("requestId")
        if not rid or not model or model == "<synthetic>" or line.get("isApiErrorMessage"):
            return
        call = self.calls.get(rid)
        if call is None:
            call = {"model": model, "usage": {}, "tools": [], "time_ms": 0, "turn": self.turn}
            self.calls[rid] = call
            self.order.append(rid)
        call["time_ms"] += self._gap(now)  # generation latency (and streaming)
        if now is not None:
            self.last_ms = now
        call["usage"] = msg.get("usage") or call["usage"]  # last line wins
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                call["tools"].append(classify_tool(block.get("name", ""), block.get("input") or {}))
                if block.get("id"):
                    self.tool_owner[block["id"]] = (rid, block.get("name", ""))

    def _user(self, line: dict, now: float | None):
        content = (line.get("message") or {}).get("content")
        owner = None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    owner = self.tool_owner.get(block.get("tool_use_id"))
                    if owner:
                        break
        if owner:
            rid, tool = owner
            # Tool run time. Not for a foreground Agent: that wait is the subagent's
            # own work, already timed in its transcript.
            if tool != "Agent":
                self.calls[rid]["time_ms"] += self._gap(now)
        else:
            self.turn += 1  # user prompt / notification: the wait before it is nobody's
        if now is not None:
            self.last_ms = now

    def _activity(self, call: dict) -> str:
        if self.review:
            return "review"
        if not call["tools"]:
            return "explore"
        return min(call["tools"], key=_PRIORITY.__getitem__)

    def result(self) -> dict:
        """{"by_activity": {act: bucket}, "explore_next": {act: bucket}} where a
        bucket is {"tokens_by_model", "time_ms", "calls"}."""
        from claude_dashboard.parser import _apply_tokens  # avoid an import cycle

        acts = [self._activity(self.calls[rid]) for rid in self.order]
        by: dict[str, dict] = {}
        nxt: dict[str, dict] = {}
        for i, rid in enumerate(self.order):
            call, act = self.calls[rid], acts[i]
            buckets = [by.setdefault(act, empty_bucket())]
            if act == "explore":
                follow = "other"  # the turn ended on reading/answering
                for j in range(i + 1, len(self.order)):
                    if self.calls[self.order[j]]["turn"] != call["turn"]:
                        break
                    if acts[j] != "explore":
                        follow = acts[j]
                        break
                buckets.append(nxt.setdefault(follow, empty_bucket()))
            for b in buckets:
                _apply_tokens(b["tokens_by_model"], call["model"], call["usage"])
                b["time_ms"] += call["time_ms"]
                b["calls"] += 1
        return {"by_activity": by, "explore_next": nxt}


def merge_activities(base: dict, extra: dict) -> dict:
    """Add `extra` (a result() dict) into `base` in place."""
    for section in ("by_activity", "explore_next"):
        dst = base.setdefault(section, {})
        for act, b in (extra.get(section) or {}).items():
            d = dst.setdefault(act, empty_bucket())
            d["time_ms"] += b.get("time_ms", 0)
            d["calls"] += b.get("calls", 0)
            for model, counts in (b.get("tokens_by_model") or {}).items():
                t = d["tokens_by_model"].setdefault(model, {})
                for k, v in counts.items():
                    t[k] = t.get(k, 0) + (v or 0)
    return base
