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
run it waits on — see `ActivityTracker._resolve_wait`. Waits on a person are
not timed: AskUserQuestion, refused tools, MCP logins, and the run time of local
file tools (Edit, Read, ...), which only ever means a pending permission prompt.
Foreground Agent waits are not timed either (the subagent times its own work).

Every call of a review subagent (its description mentions "review") is `review`.
Subagent time is summed (agent-time), so it can exceed the session's wall time.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# Bump when a classification rule changes: the scanner then re-parses every session.
RULES_VERSION = "16"

ACTIVITIES = (
    "code", "docs", "test", "eval", "train", "review",
    "build", "git", "ops", "coord", "other", "explore", "data", "web", "think",
)
# Reading / querying / answering: the activities "fold exploration" redistributes.
EXPLORATION = frozenset({"explore", "data", "web", "think"})

# Highest first: a call that both reads and edits code is `code`.
_PRIORITY = {
    "test": 0, "train": 1, "eval": 2, "code": 3, "docs": 4,
    "build": 5, "git": 6, "ops": 7, "coord": 8, "other": 9,
    "data": 10, "web": 11, "explore": 12, "think": 13,
}

_EDIT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}
_READ_TOOLS = {"Read", "Grep", "Glob", "LS", "ToolSearch"}
_WEB_TOOLS = {"WebSearch", "WebFetch"}
# Database / warehouse MCP servers and query tools.
_DATA_MCP = re.compile(r"^mcp__[^_]*(clickhouse|bigquery|postgres|mysql|snowflake|duckdb|sql|database)"
                       r"|__(run_query|query|execute_sql|list_tables|list_databases)$", re.IGNORECASE)
_DATA_CMDS = {"sqlite3", "psql", "mysql", "clickhouse", "clickhouse-client", "duckdb", "bq", "ch"}
_SQL = re.compile(r"\bselect\b[\s\S]{0,2000}?\bfrom\b", re.IGNORECASE)
_WEB_CMDS = {"curl", "wget", "http", "xh"}
# A tool result saying a person refused it: the time until then was theirs.
_REFUSED = re.compile(r"Permission for this tool use was denied|tool use was rejected|doesn't want to proceed")
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
# Local file tools finish in milliseconds; any longer is a permission prompt
# waiting on the person, so their run time is not counted.
_INSTANT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit", "Read", "Grep", "Glob", "LS", "ToolSearch"}
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
    "psql", "mysql", "clickhouse", "clickhouse-client", "duckdb", "bq", "ch",
    "curl", "wget", "http", "xh",
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
    if runs and _SQL.search(shell):
        return "data"  # a query wrapper script (q.sh "select ...")
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
        return _read_kind(bare_parts, shell, bool(bodies))
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


def _read_kind(parts: list[str], shell: str, heredoc: bool) -> str:
    """data (SQL, warehouse clients, analysis scripts, S3), web (curl) or explore."""
    heads = []
    for p in parts:
        w = _strip_wrappers(p.split())
        if w:
            heads.append((w[0].rsplit("/", 1)[-1], w[1:]))
    if _SQL.search(shell) or any(
        h in _DATA_CMDS or h.startswith("python") or (h == "aws" and r[:1] == ["s3"])
        for h, r in heads
    ):
        return "data"
    if any(h in _WEB_CMDS for h, _ in heads):
        return "web"
    return "explore"


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
            words = _drop_options(words[2:], _UV_RUN_VALUED)
        else:
            break
    return words


_UV_RUN_VALUED = {"--group", "--with", "--project", "--python", "--extra", "--package", "--env-file",
                  "--directory", "--only-group", "-p"}


def _drop_options(words: list[str], valued: set[str]) -> list[str]:
    """Leading options of a wrapper, with the value of those that take one."""
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in valued else 1
    return words[i:]


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


# ── command signatures (the drill-down under an activity) ────────────────────

WAIT_PREFIX = "wait → "
ITEMS_KEPT = 100
PROGRAMS_KEPT = 300
PROGRAMS_PER_DAY_KEPT = 30
OTHER_COMMANDS = "(other commands)"
_NEUTRAL_HEADS = {"cd", "echo", "printf", "export", "unset", "set", "source", ".", "true", "mkdir", "date"}
_REDIRECT = re.compile(r"\s*\d?>>?&?\s*(\S+)?|\s*<\s*\S+")
_ABS_PATH = re.compile(r"(?<![\w.])(?:~|/)[\w.@+-]*(?:/[\w.@+{}$-]+)+")
_MASK = str.maketrans({"|": "\x00", ";": "\x01", "&": "\x02", "\n": "\x03"})
_UNMASK = str.maketrans({"\x00": "|", "\x01": ";", "\x02": "&", "\x03": "\n"})
_CONTENT_WRITERS = {"cat", "tee", "echo", "printf"}
_NUMBERISH = re.compile(r"\b(?=[\da-f-]*\d)[\da-f]{6,}(?:-[\da-f]{4,})*\b|\b\d+\b")


def signature(name: str, inp: dict) -> str:
    """A short, normalised label of what one tool call did: the main shell
    command without cd/env wrappers, output pipes, redirects, absolute
    directories or ids; for other tools the tool and (for files) the file name."""
    if name in ("Bash", "Monitor"):
        return _command_signature(inp.get("command") or "")
    if name in _EDIT_TOOLS or name == "Read":
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return f"{name} {path.rsplit('/', 1)[-1]}".strip()
    if name == "Agent":
        return f"Agent ({inp.get('subagent_type') or 'general-purpose'})"
    if name.startswith("mcp__"):
        return name.removeprefix("mcp__").replace("__", " ")
    return name


def _command_signature(command: str) -> str:
    shell = _HEREDOC.sub(" <<heredoc\n", command)
    # Commands of the sequence, each cut at its first pipe (the rest only formats
    # output). Separators inside quotes are masked so they do not split.
    masked = _QUOTED.sub(lambda m: m.group(0).translate(_MASK), shell)
    parts, runs = [], []
    for seq in re.split(r"&&|\|\||;|\n", masked):
        seq = seq.split("|", 1)[0].translate(_UNMASK).strip()
        words = _strip_wrappers(seq.split()) if seq else None
        if words and words[0].rsplit("/", 1)[-1] not in _NEUTRAL_HEADS:
            parts.append(" ".join(words))
            if _TEST_CMD.search(seq) or _is_run(words):
                runs.append(parts[-1])
    # The run in a sequence (`rm -rf x && uv run pytest`) is what it did.
    text = (runs or parts or [shell.strip()])[0]
    inline = "<<heredoc" in text
    text = text.replace("<<heredoc", "")
    words = text.split()
    if words and words[0] in _CONTENT_WRITERS and (target := _SHELL_WRITE.search(text)):
        text = f"{words[0]} > {target.group(1)}"  # a file written from the shell
        inline = False
    else:
        text = _REDIRECT.sub("", text).rstrip(" &")
    if inline:
        text += " <<inline script"
    text = _ABS_PATH.sub(lambda m: "…/" + m.group(0).rstrip("/").rsplit("/", 1)[-1], text)
    text = _NUMBERISH.sub("N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:100] or "(empty)"


def program_of(sig: str) -> str:
    """The program part of a signature: up to its first flag, quote or path
    (`pytest tests`, `gh run watch`, `aws batch submit-job`); file tools by
    extension (`Edit *.py`)."""
    if sig.startswith(WAIT_PREFIX):
        return WAIT_PREFIX + program_of(sig.removeprefix(WAIT_PREFIX))
    words = sig.split(" ")
    if words[0] in _EDIT_TOOLS or words[0] == "Read":
        name = words[1] if len(words) > 1 else ""
        return f"{words[0]} *.{name.rsplit('.', 1)[-1]}" if "." in name else words[0]
    if len(words) >= 3 and words[1] in (">", ">>"):
        name = words[2].rsplit("/", 1)[-1]
        return f"{words[0]} > *.{name.rsplit('.', 1)[-1]}" if "." in name else f"{words[0]} >"
    out = []
    for w in words:
        if w.startswith(("-", '"', "'")):
            break
        if "/" in w or "…" in w:
            out.append(w.rsplit("/", 1)[-1])
            break
        out.append(w)
        if len(out) >= 3:
            break
    return " ".join(out) or words[0]


def _call_signature(call: dict, act: str) -> str:
    """The signature of the tool that decided the call's activity."""
    for tool_act, sig in zip(call["tools"], call["sigs"]):
        if tool_act == act:
            return sig
    return call["sigs"][0] if call["sigs"] else "(no tool)"


def classify_tool(name: str, inp: dict) -> str:
    if name in _EDIT_TOOLS:
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        return classify_path(path) if path else "code"
    if name == "Bash":
        return classify_bash(inp.get("command") or "")
    if name in _READ_TOOLS:
        return "explore"
    if name in _WEB_TOOLS:
        return "web"
    if name.startswith("mcp__") and _DATA_MCP.search(name):
        return "data"
    if name.startswith("mcp__") and _READ_MCP.search(name.rsplit("__", 1)[-1]):
        return "explore"
    if name in _COORD_TOOLS or _COORD_MCP.match(name):
        return "coord"
    if name in _GIT_TOOLS:
        return "git"
    return "other"


_REVIEW_START = re.compile(
    r"\s*(/code-review|((scoped|delta|final|second|third|fourth|whole-branch|spec|quality|"
    r"independent|adversarial|fresh|post-merge) )*(re-?)?review\b(?!-))", re.IGNORECASE)
_FIXING = re.compile(r"\b(fix\w*|findings|minors|redo)\b", re.IGNORECASE)
_FINAL_REVIEW = re.compile(r"^\s*final\b|whole-branch", re.IGNORECASE)
_RE_REVIEW = re.compile(
    r"re-?review|\bdelta\b|\b(second|third|fourth|fifth|2nd|3rd|4th)\b|\bround\s*\d|"
    r"post-merge|follow-up|re-?check|\bfinal\b", re.IGNORECASE)
_REVIEW_TARGETS = [
    re.compile(r"#(\d+)"),
    re.compile(r"\bPR\s*(\d+)", re.IGNORECASE),
    re.compile(r"/code-review\s+(\d+)"),
    re.compile(r"\b(Task\s+[A-Z]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\b(Part\s+[A-Z]\b)", re.IGNORECASE),
    re.compile(r"\b(Phase\s+\d+)", re.IGNORECASE),
]


def is_review_agent(description: str | None) -> bool:
    """A subagent whose job is a review. "Fix PR #84 review findings" or
    "Final-review fix wave" fix what a review found: they are not reviews."""
    if not description or "review" not in description.lower():
        return False
    return bool(_REVIEW_START.match(description)) or not _FIXING.search(description)


def review_kind(description: str) -> str:
    """initial / re-review / final (a whole-branch pass at the end)."""
    if _FINAL_REVIEW.search(description):
        return "final"
    return "re-review" if _RE_REVIEW.search(description) else "initial"


PLAN_GAP_MS = 12 * 3600 * 1000
_STRUCTURED_TARGET = re.compile(r"(PR #|Task |Part |Phase )")


def _review_groups(passes: list[dict]) -> dict[tuple, list[dict]]:
    """Group review passes by what they reviewed. A PR is one target per session.
    Task / Part / Phase numbers restart with every plan, and a long session runs
    several plans: a plan ends with its final (whole-branch) review, so an
    initial review after a final one, or after a 12h pause, starts the next."""
    groups: dict[tuple, list[dict]] = {}
    by_session: dict[str, list[dict]] = {}
    for p in passes:
        by_session.setdefault(p.get("session_id") or "", []).append(p)
    for sid, ps in by_session.items():
        ps.sort(key=lambda p: p.get("started_at") or "")
        plan, final_key, last = 0, None, None
        for p in ps:
            t = _ms(p.get("started_at"))
            if p["kind"] == "initial" and (final_key or (t and last and t - last > PLAN_GAP_MS)):
                plan, final_key = plan + 1, None
            last = t or last
            if p["target"].startswith("PR #"):
                key = (sid, p["target"])
            elif p["kind"] == "re-review" and final_key and not _STRUCTURED_TARGET.match(p["target"]):
                key = final_key  # re-review of the final review's fix wave
            else:
                key = (sid, plan, p["target"])
            if p["kind"] == "final":
                final_key = key
            groups.setdefault(key, []).append(p)
    return groups


def summarize_reviews(passes: list[dict]) -> dict:
    """Review passes (scanner._review_pass entries with their session_id) summed
    by kind, by pass number (1st, 2nd, ... review of the same target), and the
    passes-per-target distribution."""
    def add(d, p):
        d["count"] += 1
        d["cost_usd"] += p.get("cost_usd") or 0
        d["time_ms"] += p.get("time_ms") or 0
    def zero():
        return {"count": 0, "cost_usd": 0.0, "time_ms": 0}

    groups = _review_groups(passes)
    kinds: dict[str, dict] = {}
    by_pass: dict[str, dict] = {}
    per_target: dict[str, int] = {}
    targets = []
    for group in groups.values():
        for i, p in enumerate(group, 1):
            add(kinds.setdefault(p["kind"], zero()), p)
            add(by_pass.setdefault(str(i) if i < 4 else "4+", zero()), p)
        n = len(group)
        per_target[str(n) if n < 4 else "4+"] = per_target.get(str(n) if n < 4 else "4+", 0) + 1
        targets.append({
            "target": group[0]["target"],
            "session_id": group[0].get("session_id"),
            "passes": n,
            "kinds": [p["kind"] for p in group],
            "descriptions": [p["description"] for p in group],
            "cost_usd": sum(p.get("cost_usd") or 0 for p in group),
            "time_ms": sum(p.get("time_ms") or 0 for p in group),
        })
    targets.sort(key=lambda t: (t["passes"], t["cost_usd"]), reverse=True)
    return {
        "passes": len(passes),
        "targets": len(groups),
        "kinds": kinds,
        "by_pass": by_pass,
        "passes_per_target": per_target,
        "top_targets": targets[:30],
    }


def review_target(description: str) -> str:
    """What was reviewed, to count its passes: a PR number, else Task / Part /
    Phase, else the description without its review words."""
    for rx in _REVIEW_TARGETS:
        if m := rx.search(description):
            g = m.group(1)
            return f"PR #{g}" if g.isdigit() else " ".join(g.split()).title()
    bare = re.sub(r"(?i)\b(scoped|delta|final|second|third|fourth|whole-branch|re-?review|review|of|the)\b",
                  " ", description)
    return " ".join(bare.split()) or "(whole branch)"


def _local_day(ms: float | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).astimezone().strftime("%Y-%m-%d")


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
        self.task_sigs: dict[str, str] = {}  # background task id -> signature of what it runs
        self.tool_sigs: dict[str, str] = {}  # tool_use id -> signature
        self.lines_by_day: dict[str, int] = {}
        self.untimed: set[str] = set()       # tool_use ids whose run time is not ours
        self.last_run: tuple[str, str] | None = None  # (act, sig) of the latest test/eval/train/build run
        self.run_paths: dict[str, tuple[str, str]] = {}  # path / job id named by a run -> (act, sig)

    def feed(self, line: dict):
        ltype = line.get("type")
        now = _ms(line.get("timestamp"))
        patch = (line.get("toolUseResult") or {}).get("structuredPatch") \
            if isinstance(line.get("toolUseResult"), dict) else None
        if isinstance(patch, list):  # lines changed, on the day of the change
            n = sum(1 for h in patch if isinstance(h, dict) for ln in h.get("lines") or []
                    if isinstance(ln, str) and ln[:1] in "+-")
            day = _local_day(now) or "unknown"
            self.lines_by_day[day] = self.lines_by_day.get(day, 0) + n
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
            call = {"model": model, "usage": {}, "tools": [], "sigs": [], "time_ms": 0,
                    "turn": self.turn, "day": _local_day(now)}
            self.calls[rid] = call
            self.order.append(rid)
        call["time_ms"] += self._gap(now)  # generation latency (and streaming)
        if now is not None:
            self.last_ms = now
        call["usage"] = msg.get("usage") or call["usage"]  # last line wins
        for block in msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name, inp = block.get("name", ""), block.get("input") or {}
                act, sig = self._classify(name, inp)
                if name == "TaskOutput":
                    # Waiting on a background task: it is that task's work. A
                    # subagent's work is timed in its own transcript.
                    act = self.task_acts.get(str(inp.get("task_id")), "other")
                    if act == "agent":
                        act = "coord"
                        self.untimed.add(block.get("id"))
                call["tools"].append(act)
                if name == "TaskOutput":
                    sig = self.task_sigs.get(str(inp.get("task_id")), sig)
                call["sigs"].append(sig)
                if block.get("id"):
                    self.tool_owner[block["id"]] = (rid, block.get("name", ""), act)
                    self.tool_sigs[block["id"]] = sig

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
                            self.task_sigs[tid] = self.tool_sigs.get(block.get("tool_use_id"), "")
                    if found and found[2] in _RUN_ACTS:
                        # A run's result names where it writes / what it started
                        # (background output file, batch job id) — waits poll those.
                        self._remember(str(block.get("content"))[:4000], found[2],
                                       self.tool_sigs.get(block.get("tool_use_id"), ""))
                    if found and not owner:
                        owner, owner_id = found, block.get("tool_use_id")
                        if _REFUSED.search(str(block.get("content"))[:2000]):
                            self.untimed.add(owner_id)  # waited on a person who said no
        if owner:
            rid, tool, _ = owner
            if owner_id in self.untimed:
                tool = "Agent"
            # Tool run time — except a foreground Agent (the subagent times its own
            # work) and tools whose result is a person's answer (user wait).
            if tool not in _UNTIMED_TOOLS and tool not in _INSTANT_TOOLS \
                    and not tool.endswith(_AUTH_TOOLS):
                self.calls[rid]["time_ms"] += self._gap(now)
        else:
            self.turn += 1  # user prompt / notification: the wait before it is nobody's
        if now is not None:
            self.last_ms = now

    def _classify(self, name: str, inp: dict) -> tuple[str, str]:
        """(activity, signature) of one tool call."""
        cmd = inp.get("command") or ""
        act = "wait" if name == "Monitor" else classify_tool(name, inp)
        sig = signature(name, inp)
        if act in _RUN_ACTS:
            self.last_run = (act, sig)
            if cmd:
                self._remember(_HEREDOC.sub(" ", cmd), act, sig)
        if act == "wait":
            act, run_sig = self._resolve_wait(cmd)
            if run_sig:  # label the wait by the run it waits on, never nested
                sig = run_sig if run_sig.startswith(WAIT_PREFIX) else WAIT_PREFIX + run_sig
        return act, sig

    def _remember(self, text: str, act: str, sig: str):
        for token in _PATH_TOKEN.findall(text) + _JOB_ID.findall(text):
            self.run_paths[token] = (act, sig)

    def _resolve_wait(self, cmd: str) -> tuple[str, str | None]:
        """A wait belongs to what it waits on: a run whose log/output path it
        names (latest wins), else the words of the wait itself, else CI = test,
        else the latest run of this thread. Returns (activity, the signature of
        the run waited on when known)."""
        hits = [run for path, run in self.run_paths.items() if path in cmd]
        if hits:
            return hits[-1]
        act = classify_run(_QUOTED.sub(lambda m: m.group(0).replace("/", " "), cmd))
        if act:
            return act, None
        if _BUILD_WORDS.search(cmd.lower()):
            return "build", None
        if _CI_WAIT.search(cmd) or _TEST_OUTCOME.search(cmd):
            return "test", None
        # A bare timer (`sleep 600`, then check) waits on whatever was last launched.
        return self.last_run or ("other", None)

    def _activity(self, call: dict) -> str:
        if self.review:
            return "review"
        if not call["tools"]:
            return "think"  # thinking / answering, no tool
        return min(call["tools"], key=_PRIORITY.__getitem__)

    def result(self) -> dict:
        """{"by_activity": {act: bucket}, "explore_next": {act: bucket},
        "by_day": {local date: {act: bucket}}} where a bucket is
        {"tokens_by_model", "time_ms", "calls"}. A call counts on the day it started."""
        from claude_dashboard.parser import _apply_tokens  # avoid an import cycle

        acts = [self._activity(self.calls[rid]) for rid in self.order]
        by: dict[str, dict] = {}
        items: dict[str, dict] = {}
        programs: dict[str, dict] = {}
        programs_by_day: dict[str, dict] = {}
        nxt: dict[str, dict] = {}
        by_day: dict[str, dict] = {}
        for i, rid in enumerate(self.order):
            call, act = self.calls[rid], acts[i]
            day = by_day.setdefault(call["day"] or "unknown", {})
            sig = _call_signature(call, act)
            buckets = [by.setdefault(act, empty_bucket()), day.setdefault(act, empty_bucket()),
                       items.setdefault(act, {}).setdefault(sig, empty_bucket()),
                       programs.setdefault(act, {}).setdefault(program_of(sig), empty_bucket()),
                       programs_by_day.setdefault(call["day"] or "unknown", {}).setdefault(act, {})
                       .setdefault(program_of(sig), empty_bucket())]
            if act in EXPLORATION:
                follow = "other"  # the turn ended on reading/answering
                for j in range(i + 1, len(self.order)):
                    if self.calls[self.order[j]]["turn"] != call["turn"]:
                        break
                    if acts[j] not in EXPLORATION:
                        follow = acts[j]
                        break
                buckets.append(nxt.setdefault(follow, empty_bucket()))
            for b in buckets:
                _apply_tokens(b["tokens_by_model"], call["model"], call["usage"])
                b["time_ms"] += call["time_ms"]
                b["calls"] += 1
        return {"by_activity": by, "explore_next": nxt, "by_day": by_day,
                "items": items, "programs": programs, "programs_by_day": programs_by_day,
                "lines_by_day": dict(self.lines_by_day)}


def merge_activities(base: dict, extra: dict) -> dict:
    """Add `extra` (a result() dict) into `base` in place."""
    for section in ("by_activity", "explore_next"):
        _merge_buckets(base.setdefault(section, {}), extra.get(section) or {})
    for section in ("by_day", "items", "programs"):
        dst = base.setdefault(section, {})
        for key, buckets in (extra.get(section) or {}).items():
            _merge_buckets(dst.setdefault(key, {}), buckets)
    days = base.setdefault("programs_by_day", {})
    for day, by_act in (extra.get("programs_by_day") or {}).items():
        for act, buckets in by_act.items():
            _merge_buckets(days.setdefault(day, {}).setdefault(act, {}), buckets)
    lines = base.setdefault("lines_by_day", {})
    for day, n in (extra.get("lines_by_day") or {}).items():
        lines[day] = lines.get(day, 0) + n
    return base


def trim_items(activities: dict) -> dict:
    """Bound a session's command lists: per activity, the ITEMS_KEPT full
    commands (PROGRAMS_KEPT programs) with the most time plus as many with the
    most tokens; the rest are summed into OTHER_COMMANDS, so a list still adds up
    to its activity's total."""
    for section, keep in (("items", ITEMS_KEPT), ("programs", PROGRAMS_KEPT)):
        _trim(activities.get(section) or {}, keep)
    for by_act in (activities.get("programs_by_day") or {}).values():
        _trim(by_act, PROGRAMS_PER_DAY_KEPT)
    return activities


def _trim(lists: dict, keep: int):
    for act, sigs in lists.items():
        if len(sigs) <= keep:
            continue

        def tokens(k, sigs=sigs):
            return sum(v for c in sigs[k]["tokens_by_model"].values() for v in c.values())

        named = [k for k in sigs if k != OTHER_COMMANDS]
        kept = set(sorted(named, key=lambda k, sigs=sigs: sigs[k]["time_ms"], reverse=True)[:keep])
        kept |= set(sorted(named, key=tokens, reverse=True)[:keep])
        out = {k: sigs[k] for k in kept}
        for k, bucket in sigs.items():
            if k not in kept:
                _merge_buckets(out, {OTHER_COMMANDS: bucket})
        lists[act] = out


def _merge_buckets(dst: dict, src: dict):
    for act, b in src.items():
        d = dst.setdefault(act, empty_bucket())
        d["time_ms"] += b.get("time_ms", 0)
        d["calls"] += b.get("calls", 0)
        for model, counts in (b.get("tokens_by_model") or {}).items():
            t = d["tokens_by_model"].setdefault(model, {})
            for k, v in counts.items():
                t[k] = t.get(k, 0) + (v or 0)
