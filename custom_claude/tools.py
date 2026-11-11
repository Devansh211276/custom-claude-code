"""Core tools for Custom Claude Code with human-in-the-loop approval gates."""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

DEFAULT_IGNORES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
}


class ApprovalGate:
    """Manages human-in-the-loop confirmation for dangerous actions."""

    def __init__(self, yolo: bool = False) -> None:
        self.yolo: bool = yolo
        self.always_allowed_tools: Set[str] = set()
        self.custom_prompt_fn: Optional[Callable[[str, str], bool]] = None

    def set_yolo(self, enabled: bool) -> None:
        self.yolo = enabled

    def is_always_allowed(self, tool_name: str) -> bool:
        return self.yolo or (tool_name in self.always_allowed_tools)

    def request_approval(
        self,
        tool_name: str,
        title: str,
        preview_renderable: Any,
        prompt_message: str = "Execute this action?",
    ) -> bool:
        """Prompt user with [Y]es / [N]o / [A]lways allow in this session."""
        if self.is_always_allowed(tool_name):
            return True

        if self.custom_prompt_fn is not None:
            approved = self.custom_prompt_fn(tool_name, title)
            return approved

        console.print()
        console.print(
            Panel(
                preview_renderable,
                title=f"[bold yellow]Approval Required: {title}[/bold yellow]",
                border_style="yellow",
                subtitle="[dim]Safety Gate[/dim]",
            )
        )

        while True:
            try:
                answer = input(f"{prompt_message} [Y]es / [N]o / [A]lways allow in this session: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold red]Action aborted by user.[/bold red]")
                return False

            if answer in ("y", "yes"):
                return True
            elif answer in ("a", "always"):
                self.always_allowed_tools.add(tool_name)
                console.print(f"[bold green]Added '{tool_name}' to session auto-allow list.[/bold green]")
                return True
            elif answer in ("n", "no", ""):
                console.print("[bold red]Action rejected by user.[/bold red]")
                return False
            else:
                console.print("[dim]Please enter 'y', 'n', or 'a'.[/dim]")


# Global approval gate instance
global_gate = ApprovalGate()


def set_yolo_mode(yolo: bool) -> None:
    """Enable or disable global YOLO mode (auto-approve all tools)."""
    global_gate.set_yolo(yolo)


def read_file(path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Read file contents with line numbers."""
    target_path = Path(path)
    if not target_path.exists():
        return f"Error: File not found: {path}"
    if not target_path.is_file():
        return f"Error: Path is not a file: {path}"

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        return f"Error reading file '{path}': {exc}"

    total_lines = len(lines)
    if total_lines == 0:
        return f"[File '{path}' is empty (0 lines)]"

    start_idx = max(1, start_line)
    end_idx = total_lines if end_line is None else min(total_lines, max(start_idx, end_line))

    selected = lines[start_idx - 1 : end_idx]
    padding = len(str(end_idx))

    numbered_lines = []
    for line_num, content in zip(range(start_idx, end_idx + 1), selected):
        numbered_lines.append(f"{str(line_num).rjust(padding)} | {content.rstrip(chr(10))}")

    header = f"=== File: {path} (Lines {start_idx}-{end_idx} of {total_lines}) ==="
    return header + "\n" + "\n".join(numbered_lines)


def format_unified_diff(
    original_lines: List[str],
    modified_lines: List[str],
    from_file: str,
    to_file: str,
) -> str:
    """Generate a clean unified diff string."""
    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=from_file,
            tofile=to_file,
            lineterm="",
        )
    )
    return "\n".join(diff_lines)


def edit_file(
    path: str,
    search_block: str,
    replace_block: str,
    gate: Optional[ApprovalGate] = None,
) -> str:
    """Apply surgical search-and-replace with colored unified diff display and approval."""
    target_gate = gate or global_gate
    target_path = Path(path)
    if not target_path.exists():
        return f"Error: Cannot edit non-existent file: {path}"
    if not target_path.is_file():
        return f"Error: Path is not a file: {path}"

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            original_content = f.read()
    except Exception as exc:
        return f"Error reading '{path}': {exc}"

    # Normalize line endings for reliable matching
    normalized_original = original_content.replace("\r\n", "\n")
    normalized_search = search_block.replace("\r\n", "\n")
    normalized_replace = replace_block.replace("\r\n", "\n")

    if normalized_search not in normalized_original:
        # Check if stripping leading/trailing whitespace helps identify intent
        stripped_search = normalized_search.strip()
        if stripped_search and stripped_search in normalized_original:
            return (
                f"Error: Exact search block not found in '{path}'. "
                f"A similar trimmed block was detected; check indentation or whitespace exactness."
            )
        return (
            f"Error: Search block not found in '{path}'.\n"
            f"Provided search block ({len(normalized_search)} chars):\n{search_block[:200]}..."
        )

    # Perform single surgical replacement
    modified_content = normalized_original.replace(normalized_search, normalized_replace, 1)

    diff_text = format_unified_diff(
        normalized_original.splitlines(),
        modified_content.splitlines(),
        from_file=f"a/{path}",
        to_file=f"b/{path}",
    )

    if not diff_text:
        return f"No changes: replace block was identical to search block for '{path}'."

    # Build rich syntax highlighting for the diff preview
    diff_syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)

    approved = target_gate.request_approval(
        tool_name="edit_file",
        title=f"Edit File: {path}",
        preview_renderable=diff_syntax,
        prompt_message=f"Apply diff to '{path}'?",
    )

    if not approved:
        return f"Edit cancelled by user for file: {path}"

    try:
        # Preserve original line ending convention if present
        if "\r\n" in original_content:
            modified_content = modified_content.replace("\n", "\r\n")

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        return f"Successfully edited '{path}'. Unified diff:\n{diff_text}"
    except Exception as exc:
        return f"Error writing modified content to '{path}': {exc}"


def write_file(
    path: str,
    content: str,
    gate: Optional[ApprovalGate] = None,
) -> str:
    """Create or overwrite a file with approval gate."""
    target_gate = gate or global_gate
    target_path = Path(path)
    is_overwrite = target_path.exists()

    if is_overwrite:
        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()
            diff_text = format_unified_diff(
                old_content.replace("\r\n", "\n").splitlines(),
                content.replace("\r\n", "\n").splitlines(),
                from_file=f"a/{path}",
                to_file=f"b/{path}",
            )
            preview = Syntax(diff_text or content[:500], "diff" if diff_text else "python")
            title = f"Overwrite Existing File: {path}"
        except Exception:
            preview = Text(f"File exists ({target_path.stat().st_size} bytes). Replacing with {len(content)} characters.")
            title = f"Overwrite File: {path}"
    else:
        preview = Text(f"New file will be created at: {path}\nLength: {len(content)} characters\n\nPreview:\n{content[:400]}")
        title = f"Create File: {path}"

    approved = target_gate.request_approval(
        tool_name="write_file",
        title=title,
        preview_renderable=preview,
        prompt_message=f"Write to '{path}'?",
    )

    if not approved:
        return f"Write cancelled by user for file: {path}"

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        action_verb = "Overwrote" if is_overwrite else "Created"
        return f"Successfully {action_verb.lower()} '{path}' ({len(content)} characters)."
    except Exception as exc:
        return f"Error writing file '{path}': {exc}"


def bash_exec(
    command: str,
    timeout: int = 60,
    cwd: Optional[str] = None,
    gate: Optional[ApprovalGate] = None,
) -> Dict[str, Any]:
    """Execute shell commands with timeout, capturing stdout, stderr, and exit codes."""
    target_gate = gate or global_gate
    working_dir = Path(cwd).resolve() if cwd else Path.cwd()

    preview = Syntax(command, "powershell" if sys.platform == "win32" else "bash", theme="monokai")
    approved = target_gate.request_approval(
        tool_name="bash_exec",
        title=f"Execute Command in {working_dir}",
        preview_renderable=preview,
        prompt_message="Run command?",
    )

    if not approved:
        return {
            "command": command,
            "stdout": "",
            "stderr": "Command execution aborted by user.",
            "returncode": -1,
            "cancelled": True,
        }

    try:
        # Cross-platform execution
        if sys.platform == "win32":
            # Use powershell on Windows
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir),
                shell=False,
            )
        else:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir),
                shell=True,
                executable="/bin/bash",
            )

        output_str = proc.stdout
        err_str = proc.stderr
        return {
            "command": command,
            "stdout": output_str,
            "stderr": err_str,
            "returncode": proc.returncode,
            "cancelled": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "stdout": "",
            "stderr": f"Error: Command timed out after {timeout} seconds.",
            "returncode": 124,
            "cancelled": False,
        }
    except Exception as exc:
        return {
            "command": command,
            "stdout": "",
            "stderr": f"Execution error: {exc}",
            "returncode": 1,
            "cancelled": False,
        }


def is_ignored(path: Path, root: Path, custom_ignores: Optional[Set[str]] = None) -> bool:
    """Check if a path matches ignore criteria."""
    ignores = set(DEFAULT_IGNORES)
    if custom_ignores:
        ignores.update(custom_ignores)

    rel_parts = path.relative_to(root).parts
    for part in rel_parts:
        if part in ignores:
            return True
        for pattern in ignores:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def grep_search(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for regex pattern across codebase files."""
    root_path = Path(path).resolve()
    if not root_path.exists():
        return f"Error: Directory or file not found: {path}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: Invalid regular expression pattern '{pattern}': {exc}"

    matches: List[str] = []
    count = 0

    files_to_scan = [root_path] if root_path.is_file() else (
        root_path.rglob("*") if recursive else root_path.glob("*")
    )

    for item in files_to_scan:
        if count >= max_results:
            break
        if not item.is_file():
            continue
        if root_path.is_dir() and is_ignored(item, root_path):
            continue

        try:
            with open(item, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    if regex.search(line):
                        rel_path = item.relative_to(root_path) if root_path.is_dir() else item.name
                        matches.append(f"{rel_path}:{line_idx}: {line.strip()}")
                        count += 1
                        if count >= max_results:
                            break
        except Exception:
            continue

    if not matches:
        return f"No matches found for pattern: '{pattern}'"

    result_header = f"Found {len(matches)} matches for '{pattern}'"
    if count >= max_results:
        result_header += f" (capped at {max_results})"
    return result_header + ":\n" + "\n".join(matches)


def glob_files(pattern: str = "*", path: str = ".") -> str:
    """Find files matching a glob pattern relative to path."""
    root_path = Path(path).resolve()
    if not root_path.exists():
        return f"Error: Directory not found: {path}"

    matched_paths: List[str] = []
    try:
        for item in root_path.glob(pattern):
            if is_ignored(item, root_path):
                continue
            rel = item.relative_to(root_path)
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            matched_paths.append(f"{prefix} {rel}")
    except Exception as exc:
        return f"Error running glob '{pattern}': {exc}"

    if not matched_paths:
        return f"No files matched glob pattern: '{pattern}' in '{path}'"

    return f"Matched {len(matched_paths)} items for '{pattern}':\n" + "\n".join(sorted(matched_paths[:200]))
