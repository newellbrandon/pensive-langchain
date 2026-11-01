"""Sandboxed tool executor for Pensive test agents.

Each agent gets an isolated workspace directory. Tools may only read/write
inside it, and shell commands run with that directory as the working
directory. Every execution returns a plain-text result (with an [ok]/[error]
prefix) so failures become tool results the model can react to — never
exceptions that kill the loop.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from ddgs import DDGS

# --- workspaces --------------------------------------------------------------

def workspace_root(agent_id: str, base: Path | None = None) -> Path:
    """Absolute workspace directory for an agent (created on demand)."""
    root = base or Path(
        os.getenv("PENSIVE_AGENT_WORKSPACES", "~/.pensive/agent-workspaces")
    ).expanduser()
    return (root / agent_id).resolve()


def _safe_path(workspace: Path, raw_path: str) -> Path:
    """Resolve a tool-supplied path and reject anything outside the workspace."""
    base = workspace.resolve()
    target = (workspace / (raw_path or "")).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes the agent workspace: {raw_path!r}")
    return target


def safe_workspace_path(workspace: Path, raw_path: str) -> Path:
    """Public path resolver: reject anything outside the workspace directory.

    Used by the admin service for workspace browsing; tool handlers use
    :func:`_safe_path` internally.
    """
    return _safe_path(workspace, raw_path)


# --- tool handlers -----------------------------------------------------------

def _handle_write_file(args: dict, workspace: Path) -> str:
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise ValueError("path is required (relative to the workspace), e.g. diary.py")
    path = _safe_path(workspace, raw_path)
    content = str(args.get("content") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"[ok] wrote {path.relative_to(workspace)} ({len(content)} chars)"


def _handle_read_file(args: dict, workspace: Path) -> str:
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise ValueError("path is required (relative to the workspace), e.g. diary.py")
    path = _safe_path(workspace, raw_path)
    text = path.read_text(encoding="utf-8")
    return f"[ok] {path.relative_to(workspace)}:\n{text}"


def _handle_run_command(args: dict, workspace: Path) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        raise ValueError("command is required")
    workspace.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = "\n".join(part for part in (proc.stdout or "", proc.stderr or "") if part)
    status = "[ok]" if proc.returncode == 0 else f"[error exit={proc.returncode}]"
    return f"{status} {output}".rstrip()


def _handle_append_file(args: dict, workspace: Path) -> str:
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        raise ValueError("path is required (relative to the workspace), e.g. diary.py")
    path = _safe_path(workspace, raw_path)
    content = str(args.get("content") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(content)
    return f"[ok] appended {len(content)} chars to {path.relative_to(workspace)}"


# --- web tools (ddgs meta-search) ---------------------------------------------

_WEB_BODY_SNIPPET = 220
_WEB_EXTRACT_LIMIT = 8000


def _handle_web_search(args: dict, workspace: Path) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    try:
        max_results = int(args.get("max_results") or 5)
    except (TypeError, ValueError):
        max_results = 5
    max_results = max(1, min(max_results, 10))
    try:
        results = DDGS(timeout=10).text(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 — rate limits/network blips are common
        raise ValueError(
            f"web search failed ({type(exc).__name__}: {exc}); "
            "wait a few seconds and retry the same query"
        ) from exc
    if not results:
        return f"[ok] no web results for {query!r}"
    lines = [f"[ok] {len(results)} web result(s) for {query!r}:"]
    for index, item in enumerate(results[:max_results], 1):
        title = str(item.get("title") or "").strip() or "(no title)"
        href = str(item.get("href") or "").strip()
        body = str(item.get("body") or "").strip().replace("\n", " ")[:_WEB_BODY_SNIPPET]
        lines.append(f"{index}. {title}\n   {href}\n   {body}")
    return "\n".join(lines)


def _handle_web_extract(args: dict, workspace: Path) -> str:
    url = str(args.get("url") or "").strip()
    if not url:
        raise ValueError("url is required")
    try:
        result = DDGS(timeout=20).extract(url, fmt="text_markdown")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"web extract failed ({type(exc).__name__}: {exc}); "
            "check the URL and retry"
        ) from exc
    content = str(result.get("content") or "")[:_WEB_EXTRACT_LIMIT]
    if not content.strip():
        return f"[ok] no readable content at {url}"
    return f"[ok] extracted {url}:\n{content}"


# --- project browsing (host projects dir, read-only) ---------------------------

PROJECT_TOOL_NAMES = frozenset({"project_list", "project_read"})
_PROJECT_READ_LIMIT = 40000
_PROJECT_FILE_LIMIT = 200
_SKIP_PROJECT_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}


def _resolve_projects_root(projects_root: Path | None) -> Path | None:
    """Projects base directory: explicit arg, else PENSIVE_PROJECTS_ROOT env."""
    if projects_root is not None:
        return Path(projects_root).resolve()
    raw = os.getenv("PENSIVE_PROJECTS_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _walk_project_tree(base: Path, root: Path, limit: int) -> list[dict]:
    """Sorted {'path','size','is_dir'} entries under base, relative to root."""
    entries: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_PROJECT_DIRS)
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(dirnames):
            rel = (name if str(rel_dir) == "." else (rel_dir / name).as_posix()) + "/"
            entries.append({"path": rel, "size": 0, "is_dir": True})
            if len(entries) >= limit:
                return entries
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            full = Path(dirpath) / name
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            rel = name if str(rel_dir) == "." else (rel_dir / name).as_posix()
            entries.append({"path": rel, "size": size, "is_dir": False})
            if len(entries) >= limit:
                return entries
    return entries


def _handle_project_list(args: dict, root: Path) -> str:
    rel = str(args.get("path") or "").strip()
    base = _safe_path(root, rel)
    if not base.is_dir():
        return f"[error] not a directory under projects: {rel or '.'}"
    entries = _walk_project_tree(base, root, _PROJECT_FILE_LIMIT)
    label = rel or "."
    if not entries:
        return f"[ok] no files under projects/{label}"
    lines = [f"[ok] {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} under projects/{label}:"]
    for entry in entries:
        suffix = "/" if entry["is_dir"] else f" ({entry['size']} B)"
        lines.append(f"  {entry['path']}{suffix}")
    return "\n".join(lines)


def _handle_project_read(args: dict, root: Path) -> str:
    rel = str(args.get("path") or "").strip()
    if not rel:
        raise ValueError("path is required (relative to the projects root), e.g. okf/index.md")
    path = _safe_path(root, rel)
    if not path.is_file():
        return f"[error] no such project file: {rel}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[error] {type(exc).__name__}: {exc}"
    if len(text) > _PROJECT_READ_LIMIT:
        text = text[:_PROJECT_READ_LIMIT] + "\n…[truncated at 40000 chars]"
    return f"[ok] {rel} ({len(text)} chars):\n{text}"


_PROJECT_HANDLERS: dict[str, Callable[[dict, Path], str]] = {
    "project_list": _handle_project_list,
    "project_read": _handle_project_read,
}


def execute_project_tool_call(
    tool_call: dict,
    projects_root: Path | None = None,
) -> dict:
    """Execute a project-browsing tool call. Never raises; {'name', 'output'}."""
    name = tool_call.get("name") or ""
    if name not in PROJECT_TOOL_NAMES:
        return {"name": name, "output": f"[error] unknown project tool: {name}"}
    root = _resolve_projects_root(projects_root)
    if root is None:
        return {
            "name": name,
            "output": (
                "[error] project access is not configured on this deployment "
                "(PENSIVE_PROJECTS_ROOT is unset)"
            ),
        }
    try:
        args = json.loads(str(tool_call.get("arguments") or "{}"))
        if not isinstance(args, dict):
            args = {}
    except (ValueError, TypeError) as exc:
        return {
            "name": name,
            "output": (
                f"[error] tool arguments are not valid JSON ({exc}); reissue "
                "the call with valid JSON"
            ),
        }
    try:
        return {"name": name, "output": _PROJECT_HANDLERS[name](args, root)}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "output": f"[error] {type(exc).__name__}: {exc}"}


_HANDLERS: dict[str, Callable[[dict, Path], str]] = {
    "write_file": _handle_write_file,
    "append_file": _handle_append_file,
    "read_file": _handle_read_file,
    "run_command": _handle_run_command,
    "web_search": _handle_web_search,
    "web_extract": _handle_web_extract,
}

# --- tool definitions (OpenAI function schema) --------------------------------

def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


DEFAULT_AGENT_TOOLS: list[dict] = [
    _function(
        "write_file",
        "Create or overwrite a file inside the agent workspace. Keep content "
        "under ~1500 characters per call; build large files with append_file.",
        {"path": {"type": "string", "description": "Relative path inside the workspace"},
         "content": {"type": "string", "description": "Full file content"}},
        ["path", "content"],
    ),
    _function(
        "append_file",
        "Append content to a file inside the agent workspace (creates the file "
        "if it does not exist). Use this to build large files incrementally in "
        "pieces of ~1500 characters or less.",
        {"path": {"type": "string", "description": "Relative path inside the workspace"},
         "content": {"type": "string", "description": "Content to append"}},
        ["path", "content"],
    ),
    _function(
        "read_file",
        "Read a file inside the agent workspace.",
        {"path": {"type": "string", "description": "Relative path inside the workspace"}},
        ["path"],
    ),
    _function(
        "run_command",
        "Run a shell command in the agent workspace directory (e.g. python3 app.py, "
        "python3 -m pytest, ls).",
        {"command": {"type": "string"}},
        ["command"],
    ),
    _function(
        "web_search",
        "Search the web (DuckDuckGo text search) for a query. Returns up to "
        "max_results (default 5, max 10) results with title, URL, and a short "
        "snippet. Use web_extract to read the full content of a result URL.",
        {"query": {"type": "string", "description": "Web search query"},
         "max_results": {"type": "integer", "description": "Results to return (default 5, max 10)"}},
        ["query"],
    ),
    _function(
        "web_extract",
        "Fetch a URL and extract its readable content as markdown (truncated "
        "to about 8000 characters). Use on URLs returned by web_search.",
        {"url": {"type": "string", "description": "http(s) URL to fetch"}},
        ["url"],
    ),
    _function(
        "project_list",
        "List files and directories inside the host projects directory "
        "(~/projects, read-only; the OKF knowledge bundle lives under okf/). "
        "Relative paths are from the projects root.",
        {"path": {"type": "string", "description": "Relative directory inside projects (omit for the root listing)"}},
        [],
    ),
    _function(
        "project_read",
        "Read a file from the host projects directory (~/projects, read-only, "
        "truncated to ~40000 characters). Use project_list to discover paths.",
        {"path": {"type": "string", "description": "Relative file path inside projects, e.g. okf/index.md"}},
        ["path"],
    ),
]


def execute_tool_call(tool_call: dict, workspace: Path, projects_root: Path | None = None) -> dict:
    """Execute one tool call. Never raises; returns {'name', 'output'}."""
    name = tool_call.get("name") or ""
    if name in PROJECT_TOOL_NAMES:
        return execute_project_tool_call(tool_call, projects_root=projects_root)
    # Resolve once so handlers compare like-for-like paths (macOS /var → /private/var).
    workspace = workspace.resolve()
    name = tool_call.get("name") or ""
    raw_arguments = str(tool_call.get("arguments") or "{}")
    try:
        args = json.loads(raw_arguments)
    except (ValueError, TypeError) as exc:
        # Models sometimes hand-build large argument payloads that are not
        # valid JSON. Say exactly what is wrong so the retry is a real fix.
        return {
            "name": name,
            "output": (
                f"[error] tool arguments are not valid JSON ({exc}). Reissue "
                "the tool call with valid JSON. Keep individual write_file and "
                "append_file content under ~1500 characters per call; split "
                "large files across multiple append_file calls (never rewrite "
                "the same file with write_file more than once)."
            ),
        }
    if not isinstance(args, dict):
        args = {}
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"name": name, "output": f"[error] unknown tool: {name}"}
    try:
        return {"name": name, "output": handler(args, workspace)}
    except Exception as exc:  # noqa: BLE001 — tool failures are model-visible results
        return {"name": name, "output": f"[error] {type(exc).__name__}: {exc}"}