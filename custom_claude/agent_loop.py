"""ReAct autonomous agent loop: Plan -> Think -> Execute Tool -> Observe -> Repeat."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from custom_claude.context import ProjectContext
from custom_claude.model_engine import (
    ChunkType,
    ModelEngine,
    ResponseParser,
    StreamChunk,
    ToolCall,
)
from custom_claude.tools import (
    ApprovalGate,
    bash_exec,
    edit_file,
    glob_files,
    grep_search,
    read_file,
    write_file,
)

console = Console()

DEFAULT_SYSTEM_PROMPT = """You are Custom Claude Code (powered by My-Opus), an expert autonomous AI software engineer directly integrated into the developer's terminal.

You operate in an autonomous ReAct loop:
1. <thinking>: You must ALWAYS begin your response with an internal reasoning process wrapped in `<thinking>...</thinking>` tags. Plan your steps carefully.
2. Tool Execution: When you need to read files, search the codebase, modify code, or execute commands, emit structured `<tool_call>` tags:
   <tool_call>
   {"name": "tool_name", "arguments": {"arg1": "val1"}}
   </tool_call>
3. Available Tools:
   - `read_file(path, start_line, end_line)`: Read file contents with line numbers.
   - `edit_file(path, search_block, replace_block)`: Apply surgical search-and-replace with diff preview.
   - `write_file(path, content)`: Create or overwrite files.
   - `bash_exec(command, timeout)`: Run terminal commands (e.g., pytest, linters, git).
   - `grep_search(pattern, path)`: Fast regex search across repository files.
   - `glob_files(pattern, path)`: Find files matching patterns.
4. Autonomous Testing: When fixing bugs or running tests, read failure tracebacks, understand root causes, surgically modify the code, and re-run tests until the test suite passes with 0 errors.
5. Provide concise, direct explanations when answering without unnecessary fluff.
"""


class AgentLoop:
    """Manages multi-turn autonomous tool execution cycles."""

    def __init__(
        self,
        model_engine: Optional[ModelEngine] = None,
        context: Optional[ProjectContext] = None,
        gate: Optional[ApprovalGate] = None,
        max_iterations: int = 10,
    ) -> None:
        self.engine = model_engine or ModelEngine()
        self.context = context or ProjectContext()
        self.gate = gate
        self.max_iterations = max_iterations
        self.messages: List[Dict[str, str]] = []
        self._init_session()

    def _init_session(self) -> None:
        """Initialize system prompt and project memory in context."""
        system_context = self.context.get_system_context()
        full_system = f"{DEFAULT_SYSTEM_PROMPT}\n\n### Current Project Environment:\n{system_context}"
        self.messages = [{"role": "system", "content": full_system}]

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to project root if not absolute."""
        p = Path(path)
        if not p.is_absolute():
            p = (self.context.root_dir / p).resolve()
        return str(p)

    def execute_tool(self, call: ToolCall) -> str:
        """Dispatch tool call to corresponding implementation with safety gate."""
        name = call.name.strip()
        args = call.arguments or {}

        try:
            if name == "read_file":
                raw_path = args.get("path") or args.get("file_path", "")
                path = self._resolve_path(raw_path)
                start = int(args.get("start_line", 1))
                end = int(args["end_line"]) if "end_line" in args and args["end_line"] is not None else None
                return read_file(path=path, start_line=start, end_line=end)

            elif name == "edit_file":
                raw_path = args.get("path") or args.get("file_path", "")
                path = self._resolve_path(raw_path)
                search_block = args.get("search_block") or args.get("search", "")
                replace_block = args.get("replace_block") or args.get("replace", "")
                return edit_file(
                    path=path,
                    search_block=search_block,
                    replace_block=replace_block,
                    gate=self.gate,
                )

            elif name == "write_file":
                raw_path = args.get("path") or args.get("file_path", "")
                path = self._resolve_path(raw_path)
                content = args.get("content", "")
                return write_file(path=path, content=content, gate=self.gate)

            elif name == "bash_exec":
                cmd = args.get("command") or args.get("cmd", "")
                timeout = int(args.get("timeout", 60))
                cwd = str(self.context.root_dir)
                result = bash_exec(command=cmd, timeout=timeout, cwd=cwd, gate=self.gate)
                output = []
                if result["stdout"]:
                    output.append(f"[stdout]\n{result['stdout']}")
                if result["stderr"]:
                    output.append(f"[stderr]\n{result['stderr']}")
                if result.get("cancelled"):
                    output.append("[Command was cancelled by user]")
                output.append(f"[Exit code: {result['returncode']}]")
                return "\n".join(output)

            elif name == "grep_search":
                pattern = args.get("pattern", "")
                path = self._resolve_path(args.get("path", "."))
                return grep_search(pattern=pattern, path=path)

            elif name == "glob_files":
                pattern = args.get("pattern", "*")
                path = self._resolve_path(args.get("path", "."))
                return glob_files(pattern=pattern, path=path)

            else:
                return f"Error: Unknown tool '{name}'. Available: read_file, edit_file, write_file, bash_exec, grep_search, glob_files"
        except Exception as exc:
            return f"Error executing tool '{name}': {exc}"

    def run_turn(
        self,
        user_input: str,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    ) -> str:
        """Run a full autonomous turn (Plan -> Think -> Tool -> Observe -> Repeat)."""
        self.messages.append({"role": "user", "content": user_input})
        iteration = 0
        final_answer = ""

        while iteration < self.max_iterations:
            iteration += 1
            accumulated_thinking = ""
            accumulated_text = ""
            found_tool_calls: List[ToolCall] = []

            # Stream model output
            for chunk in self.engine.generate_stream(self.messages):
                if on_chunk:
                    on_chunk(chunk)

                if chunk.chunk_type == ChunkType.THINKING:
                    accumulated_thinking += chunk.content
                elif chunk.chunk_type == ChunkType.TEXT:
                    accumulated_text += chunk.content
                elif chunk.chunk_type == ChunkType.TOOL_CALL and chunk.tool_call:
                    found_tool_calls.append(chunk.tool_call)

            # Check if any tool calls were embedded in text if parser didn't catch them
            if not found_tool_calls:
                _, _, extra_calls = ResponseParser.parse_full_response(accumulated_text)
                found_tool_calls.extend(extra_calls)

            # Record assistant message in history
            assistant_content = ""
            if accumulated_thinking:
                assistant_content += f"<thinking>\n{accumulated_thinking.strip()}\n</thinking>\n"
            assistant_content += accumulated_text

            if found_tool_calls:
                for call in found_tool_calls:
                    assistant_content += f'\n<tool_call>\n{{"name": "{call.name}", "arguments": {json.dumps(call.arguments)}}}\n</tool_call>'

            self.messages.append({"role": "assistant", "content": assistant_content})

            # If no tools called, we've arrived at the final answer
            if not found_tool_calls:
                final_answer = accumulated_text
                break

            # Execute all tools called by the model
            tool_results_text = []
            for call in found_tool_calls:
                result = self.execute_tool(call)
                tool_results_text.append(f"<tool_result name=\"{call.name}\">\n{result}\n</tool_result>")

            observation = "\n\n".join(tool_results_text)
            self.messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        return final_answer

    def run_autonomous_test_loop(
        self,
        target_file: Optional[str] = None,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
    ) -> str:
        """Specialized autonomous test generation, execution, and self-repair loop."""
        target_desc = f" targeting `{target_file}`" if target_file else " for the project"
        prompt = (
            f"Run the autonomous testing workflow{target_desc}. "
            f"1. Run the test suite via `bash_exec` with `pytest -q`. "
            f"2. If tests fail, read the failure tracebacks, inspect the code using `read_file`, "
            f"apply surgical fixes using `edit_file`, and re-run `pytest`. "
            f"3. Repeat until all tests pass with 0 errors!"
        )
        return self.run_turn(prompt, on_chunk=on_chunk)

    def compact_history(self) -> str:
        """Summarize history and prune earlier messages to save context tokens."""
        if len(self.messages) <= 3:
            return "Conversation history is already minimal (no compaction needed)."

        original_count = len(self.messages)
        system_msg = self.messages[0]
        recent_messages = self.messages[-3:]

        # Create concise summary of pruned messages
        pruned_turns = self.messages[1:-3]
        summary_points = []
        for msg in pruned_turns:
            role = msg.get("role", "unknown")
            text = msg.get("content", "")[:120].replace("\n", " ")
            summary_points.append(f"- [{role}]: {text}...")

        summary_text = (
            f"### Compacted Prior Context ({len(pruned_turns)} turns pruned):\n"
            + "\n".join(summary_points[:8])
            + f"\n[Compacted {original_count - 4} messages to preserve context window]"
        )

        self.messages = [
            system_msg,
            {"role": "user", "content": summary_text},
            {"role": "assistant", "content": "Acknowledged. Prior context compacted; ready to continue."},
        ] + recent_messages

        return f"Compacted conversation from {original_count} messages down to {len(self.messages)} messages."
