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
]


def execute_tool_call(tool_call: dict, workspace: Path) -> dict:
    """Execute one tool call. Never raises; returns {'name', 'output'}."""
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