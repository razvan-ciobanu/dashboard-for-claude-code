"""Split a session's tokens and time across activities (code, docs, tests, ...).

The unit is one API call (one requestId). Its tokens are that call's final usage;
its time is the generation latency before it plus the run time of the tools it
issued. Time spent waiting on the user (or on a queued/background notification)
is not attributed to anything.

A call's activity comes from the tools it issued — see `classify_tool`. A call
that only reads (or issues no tool) is `explore`; for those we also record which
activity came next in the same turn (`explore_next`), so the UI can show the
reading either as its own activity or folded into the work it led to.

A run (script, container, batch job) is test / eval / train by the words of what
it runs; other commands fall into build (docker build/push, tags, releases,
dependency installs), git (commits, PRs, worktrees), ops (ssh, aws, terraform,
...) or coord (agents, messages, task tracking, Jira); a wait (polling loop, Monitor, `gh run watch`) takes the activity of the
run it waits on — see `ActivityTracker._resolve_wait`. AskUserQuestion and
foreground Agent waits are not timed.

Every call of a review subagent (its description mentions "review") is `review`.
Subagent time is summed (agent-time), so it can exceed the session's wall time.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# Bump when a classification rule changes: the scanner then re-parses every session.
RULES_VERSION = "7"

ACTIVITIES = (
    "code", "docs", "test", "eval", "train", "review",
    "build", "git", "ops", "coord", "explore", "other",
)

# Highest first: a call that both reads and edits code is `code`.
_PRIORITY = {
    "test": 0, "train": 1, "eval": 2, "code": 3, "docs": 4,
    "build": 5, "git": 6, "ops": 7, "coord": 8, "other": 9, "explore": 10,
}

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
_SHELL_WRITE = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*['\"]?([~${}\w./-]+\.\w+)")
_RUN_ACTS = {"test", "eval", "train", "build"}
# Orchestrating other agents and tracking the work (agents, messages, tasks, Jira).
_COORD_TOOLS = {
    "Agent", "SendMessage", "SubagentHandback", "TaskStop", "ListAgents", "TodoWrite",
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "CronCreate", "CronDelete",
    "ScheduleWakeup", "PushNotification",
}
_GIT_TOOLS = {"EnterWorktree", "ExitWorktree"}
_COORD_MCP = re.compile(r"^mcp__(jira|atlassian|linear|todoist|monday|plugin_slack\w*)__", re.IGNORECASE)
_BUILD_WORDS = re.compile(r"(?<![a-z])(build|release|publish|deploy|docker-build\w*)(?![a-z])")
_OPS_HEADS = {
    "ssh", "scp", "rsync", "terraform", "ansible", "ansible-playbook", "aws", "gcloud",
    "kubectl", "helm", "systemctl", "journalctl", "sudo",
}
_BUILD_HEADS = {"sbt", "mvn", "gradle", "cargo", "go", "npm", "pnpm", "yarn", "pip", "pip3", "poetry"}
_UNTIMED_TOOLS = {"Agent", "AskUserQuestion", "ExitPlanMode", "SubagentHandback"}
# MCP OAuth flows: the result arrives when the person finishes logging in.
_AUTH_TOOLS = ("__authenticate", "__complete_authentication")
_PATH_TOKEN = re.compile(r"[~\w.-]*/[\w./-]*[\w-]\.\w+")
# How a tool result names the background task it started (bash, MCP call, agent).
_TASK_ID = re.compile(r"(?:with ID:|as task|agentId:)\s*([A-Za-z0-9]+)")
_JOB_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_LOG_EXT = (".log", ".out", ".err")
# Commands that wait on something else: polling loops, CI/batch watchers, sleeps.
_WAIT = re.compile(
    r"\b(until|while)\b[^\n]*\b(do|sleep)\b|\bgh\s+run\s+watch\b|\bgh\s+pr\s+checks\b[^\n]*--watch"
    r"|\baws\s+batch\s+wait\b|^\s*sleep\s+\d|\btimeout\s+\d+\s+(ba|z)?sh\s+-c\b"
    r"|(?s:\bfor\b.*\bdo\b.*\bsleep\s+\d)"
)
_CI_WAIT = re.compile(r"\bgh\s+(run\s+watch|pr\s+checks)\b")
_RUNNERS = {"docker", "podman", "bash", "sh", "zsh", "make", "node", "npx", "sbatch", "kubectl"}
_RUN_WORDS = [
    ("test", re.compile(r"(?<![a-z])(pytest|tests?|gate|golden|parity|smoke|anchors?)(?![a-z])")),
    ("train", re.compile(r"(?<![a-z])(train(ing)?|retrain|grid[_-]?search|fine[_-]?tune)")),
    ("eval", re.compile(r"(?<![a-z])(eval(uat\w*)?|attribution|backtest|bake[_-]?off|benchmark)")),
]
# Wait conditions on a test outcome ("N passed", "FAILED") once nothing else matched.
_TEST_OUTCOME = re.compile(r"\b(passed|failed)\b", re.IGNORECASE)
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


def classify_run(text: str) -> str | None:
    """test / train / eval from the words of a run or wait (job names, scripts,
    log names), or None. `(?<![a-z])` keeps "latest", "constraint" or
    "aggregate" from matching while still matching `run_release_gate`."""
    t = text.lower()
    for act, rx in _RUN_WORDS:
        if rx.search(t):
            return act
    return None


def classify_bash(command: str) -> str:
    """An activity, or "wait" for a command that waits on something else
    (resolved by ActivityTracker against what it waits on)."""
    bodies = [m.group(2) for m in _HEREDOC.finditer(command)]
    shell = _HEREDOC.sub(" ", command)
    if _TEST_CMD.search(shell):
        return "test"
    if _WAIT.search(shell):
        return "wait"
    bare = _QUOTED.sub("Q", shell)
    parts = [p.strip() for p in re.split(r"\|\||&&|[|;\n()&]", shell) if p.strip()]
    words = [_strip_wrappers(p.split()) for p in parts]
    # A run (script, container, batch job): what it runs decides.
    runs = [p for p, w in zip(parts, words) if w and _is_run(w)]
    act = next((a for a in map(classify_run, runs) if a), None)
    if act:
        return act
    # A heredoc script that writes a file (python - <<EOF ... write_text ...).
    for body in bodies:
        if _PY_WRITE.search(body):
            m = _PATH_LITERAL.search(body)
            return classify_path(m.group(1)) if m else "code"
    for target in _SHELL_WRITE.findall(bare):
        if not target.startswith("/dev/") and not target.endswith(_LOG_EXT):
            return classify_path(target)
    # Read-only when every command of a pipeline/sequence is a reader.
    bare_parts = [p.strip() for p in re.split(r"\|\||&&|[|;\n()]", bare) if p.strip()]
    if bare_parts and all(_is_read_cmd(p, bool(bodies)) for p in bare_parts):
        return "explore"
    # Otherwise the most significant kind of tooling the non-reading parts use.
    acts = [_tool_kind(w) for p in bare_parts
            if (w := _strip_wrappers(p.split())) and not _is_read_cmd(p, bool(bodies))]
    return min(acts, key=_PRIORITY.__getitem__, default="other")


def _tool_kind(words: list[str]) -> str:
    """build / git / ops for one non-reading command, else other."""
    head = words[0].rsplit("/", 1)[-1]
    rest = [w for w in words[1:] if not w.startswith("-")]
    if head == "git":
        return "build" if rest[:1] == ["tag"] else "git"
    if head == "gh":
        if rest[:1] in (["release"], ["workflow"]):
            return "build"
        return "git" if rest[:1] in (["pr"], ["repo"], ["issue"]) else "ops"
    if head in {"docker", "podman"}:
        return "build" if rest[:1] in (["build"], ["push"], ["buildx"], ["compose"]) else "ops"
    if head == "uv":
        return "build" if rest[:1] in (["sync"], ["build"], ["publish"], ["lock"], ["add"], ["pip"]) else "other"
    if head in _BUILD_HEADS:
        return "build"
    if head in _OPS_HEADS:
        return "ops"
    return "other"


def _strip_wrappers(words: list[str]) -> list[str] | None:
    """Drop leading wrappers (env -u X, VAR=v, time, nohup, timeout N, xargs,
    uv run --flags). None for shell syntax that does nothing by itself (cd, do...)."""
    while words:
        w = words[0]
        if w in {"cd", "do", "done", "then", "fi", "else", "{", "}"}:
            return None
        if "=" in w or w in {"time", "nohup", "xargs", "command", "exec"}:
            words = words[1:]
        elif w in {"env", "timeout"} or (w.startswith("-") and len(words) > 1):
            words = words[2:] if w in {"-u", "-C", "timeout"} else words[1:]
        elif w == "uv" and len(words) > 1 and words[1] == "run":
            words = [x for x in words[2:] if not x.startswith("--")]
        else:
            break
    return words


def _is_run(words: list[str]) -> bool:
    head = words[0].rsplit("/", 1)[-1]
    rest = words[1:]
    if head.startswith("python"):
        # A repo script; inline code and scratch scripts are analysis (explore).
        return bool(rest) and rest[0] not in {"-", "-c"} and not _SCRATCH.search(rest[0])
    if head == "aws":
        return rest[:2] == ["batch", "submit-job"]
    return head in _RUNNERS or head.endswith((".sh", ".py")) or words[0].startswith("./")


def _is_read_cmd(part: str, heredoc: bool) -> bool:
    words = _strip_wrappers(part.split())
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
    if name in _COORD_TOOLS or _COORD_MCP.match(name):
        return "coord"
    if name in _GIT_TOOLS:
        return "git"
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
        self.tool_owner: dict[str, tuple[str, str, str]] = {}  # tool_use id -> (requestId, tool, act)
        self.turn = 0                       # bumped at every user/notification boundary
        self.last_ms: float | None = None
        self.task_acts: dict[str, str] = {}  # background task id -> act ("agent" for a subagent)
        self.untimed: set[str] = set()       # tool_use ids whose run time is not ours
        self.last_run: str | None = None     # activity of the latest test/eval/train/build run
        self.run_paths: dict[str, str] = {}  # path / job id named by a test/eval/train run -> act

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
                name, inp = block.get("name", ""), block.get("input") or {}
                act = self._classify(name, inp)
                if name == "TaskOutput":
                    # Waiting on a background task: it is that task's work. A
                    # subagent's work is timed in its own transcript.
                    act = self.task_acts.get(str(inp.get("task_id")), "other")
                    if act == "agent":
                        act = "coord"
                        self.untimed.add(block.get("id"))
                call["tools"].append(act)
                if block.get("id"):
                    self.tool_owner[block["id"]] = (rid, block.get("name", ""), act)

    def _user(self, line: dict, now: float | None):
        content = (line.get("message") or {}).get("content")
        owner, owner_id = None, None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    found = self.tool_owner.get(block.get("tool_use_id"))
                    if found:
                        text = str(block.get("content"))[:4000]
                        for tid in _TASK_ID.findall(text):
                            self.task_acts[tid] = "agent" if found[1] == "Agent" else found[2]
                    if found and found[2] in _RUN_ACTS:
                        # A run's result names where it writes / what it started
                        # (background output file, batch job id) — waits poll those.
                        self._remember(str(block.get("content"))[:4000], found[2])
                    if found and not owner:
                        owner, owner_id = found, block.get("tool_use_id")
        if owner:
            rid, tool, _ = owner
            if owner_id in self.untimed:
                tool = "Agent"
            # Tool run time — except a foreground Agent (the subagent times its own
            # work) and tools whose result is a person's answer (user wait).
            if tool not in _UNTIMED_TOOLS and not tool.endswith(_AUTH_TOOLS):
                self.calls[rid]["time_ms"] += self._gap(now)
        else:
            self.turn += 1  # user prompt / notification: the wait before it is nobody's
        if now is not None:
            self.last_ms = now

    def _classify(self, name: str, inp: dict) -> str:
        cmd = inp.get("command") or ""
        act = "wait" if name == "Monitor" else classify_tool(name, inp)
        if act in _RUN_ACTS:
            self.last_run = act
            if cmd:
                self._remember(_HEREDOC.sub(" ", cmd), act)
        if act == "wait":
            act = self._resolve_wait(cmd)
        return act

    def _remember(self, text: str, act: str):
        for token in _PATH_TOKEN.findall(text) + _JOB_ID.findall(text):
            self.run_paths[token] = act

    def _resolve_wait(self, cmd: str) -> str:
        """A wait belongs to what it waits on: a run whose log/output path it
        names (latest wins), else the words of the wait itself, else CI = test,
        else the latest run of this thread."""
        hits = [act for path, act in self.run_paths.items() if path in cmd]
        if hits:
            return hits[-1]
        act = classify_run(_QUOTED.sub(lambda m: m.group(0).replace("/", " "), cmd))
        if act:
            return act
        if _BUILD_WORDS.search(cmd.lower()):
            return "build"
        if _CI_WAIT.search(cmd) or _TEST_OUTCOME.search(cmd):
            return "test"
        # A bare timer (`sleep 600`, then check) waits on whatever was last launched.
        return self.last_run or "other"

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
