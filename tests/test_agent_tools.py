"""Comprehensive pytest suite for custom-claude-code: tools, approval gates, diff engine, and model loop."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from custom_claude.agent_loop import AgentLoop
from custom_claude.context import ProjectContext
from custom_claude.model_engine import (
    ChunkType,
    CustomOpusConfig,
    CustomOpusModel,
    ModelEngine,
    ResponseParser,
    StreamBufferParser,
    ToolCall,
)
from custom_claude.tools import (
    ApprovalGate,
    bash_exec,
    edit_file,
    format_unified_diff,
    glob_files,
    grep_search,
    read_file,
    write_file,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for isolated file testing."""
    tmp = tempfile.mkdtemp(prefix="ccode_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# 1. Tests for read_file
# ==============================================================================

def test_read_file_basic(temp_workspace):
    test_file = temp_workspace / "sample.py"
    test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")

    result = read_file(str(test_file), start_line=2, end_line=4)
    assert "Lines 2-4 of 5" in result
    assert "2 | line 2" in result
    assert "3 | line 3" in result
    assert "4 | line 4" in result
    assert "1 | line 1" not in result
    assert "5 | line 5" not in result


def test_read_file_non_existent(temp_workspace):
    missing_path = temp_workspace / "does_not_exist.txt"
    result = read_file(str(missing_path))
    assert "Error: File not found" in result


def test_read_file_empty(temp_workspace):
    empty_file = temp_workspace / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    result = read_file(str(empty_file))
    assert "empty (0 lines)" in result


# ==============================================================================
# 2. Tests for edit_file & Diff Engine
# ==============================================================================

def test_format_unified_diff():
    orig = ["def hello():", "    return 'world'"]
    mod = ["def hello():", "    return 'universe'"]
    diff = format_unified_diff(orig, mod, "a/test.py", "b/test.py")
    assert "-    return 'world'" in diff
    assert "+    return 'universe'" in diff


def test_edit_file_success_yolo(temp_workspace):
    gate = ApprovalGate(yolo=True)
    f = temp_workspace / "mod.py"
    f.write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")

    res = edit_file(str(f), search_block="y = 20", replace_block="y = 99", gate=gate)
    assert "Successfully edited" in res
    assert "-y = 20" in res
    assert "+y = 99" in res
    assert f.read_text(encoding="utf-8") == "x = 10\ny = 99\nz = 30\n"


def test_edit_file_not_found(temp_workspace):
    gate = ApprovalGate(yolo=True)
    f = temp_workspace / "mod2.py"
    f.write_text("foo = 1\n", encoding="utf-8")

    res = edit_file(str(f), search_block="bar = 2", replace_block="bar = 3", gate=gate)
    assert "Error: Search block not found" in res
    assert f.read_text(encoding="utf-8") == "foo = 1\n"


def test_edit_file_rejection(temp_workspace):
    gate = ApprovalGate(yolo=False)
    # Custom prompt hook simulates user rejecting the edit
    gate.custom_prompt_fn = lambda tool, title: False

    f = temp_workspace / "reject.py"
    f.write_text("a = 1\n", encoding="utf-8")

    res = edit_file(str(f), search_block="a = 1", replace_block="a = 2", gate=gate)
    assert "cancelled by user" in res
    assert f.read_text(encoding="utf-8") == "a = 1\n"


# ==============================================================================
# 3. Tests for write_file
# ==============================================================================

def test_write_file_new_and_overwrite(temp_workspace):
    gate = ApprovalGate(yolo=True)
    target = temp_workspace / "nested" / "output.txt"

    res1 = write_file(str(target), "First content\n", gate=gate)
    assert "created" in res1.lower()
    assert target.read_text(encoding="utf-8") == "First content\n"

    res2 = write_file(str(target), "Overwritten content\n", gate=gate)
    assert "overwrote" in res2.lower()
    assert target.read_text(encoding="utf-8") == "Overwritten content\n"


def test_write_file_rejection(temp_workspace):
    gate = ApprovalGate(yolo=False)
    gate.custom_prompt_fn = lambda tool, title: False
    target = temp_workspace / "no_write.txt"

    res = write_file(str(target), "Should not exist", gate=gate)
    assert "cancelled by user" in res
    assert not target.exists()


# ==============================================================================
# 4. Tests for bash_exec
# ==============================================================================

def test_bash_exec_basic_command(temp_workspace):
    gate = ApprovalGate(yolo=True)
    res = bash_exec("Write-Output 'Hello from CCode'", cwd=str(temp_workspace), gate=gate)
    assert res["returncode"] == 0
    assert "Hello from CCode" in res["stdout"]
    assert not res["cancelled"]


def test_bash_exec_rejection(temp_workspace):
    gate = ApprovalGate(yolo=False)
    gate.custom_prompt_fn = lambda tool, title: False
    res = bash_exec("Write-Output 'Danger'", cwd=str(temp_workspace), gate=gate)
    assert res["cancelled"] is True
    assert res["returncode"] == -1


def test_bash_exec_timeout():
    gate = ApprovalGate(yolo=True)
    # Short timeout on a sleep command
    res = bash_exec("Start-Sleep -Seconds 3", timeout=1, gate=gate)
    assert res["returncode"] == 124
    assert "timed out" in res["stderr"].lower()


# ==============================================================================
# 5. Tests for grep_search & glob_files
# ==============================================================================

def test_grep_and_glob(temp_workspace):
    sub = temp_workspace / "src"
    sub.mkdir()
    (sub / "main.py").write_text("def run_pipeline():\n    pass\n", encoding="utf-8")
    (sub / "helper.py").write_text("def helper_fn():\n    return True\n", encoding="utf-8")
    (temp_workspace / ".git").mkdir()
    (temp_workspace / ".git" / "git_internal.py").write_text("def ignored_fn(): pass\n", encoding="utf-8")

    # Test grep
    grep_res = grep_search("run_pipeline", path=str(temp_workspace))
    assert "main.py:1: def run_pipeline():" in grep_res
    assert "git_internal.py" not in grep_res

    # Test glob
    glob_res = glob_files("*.py", path=str(sub))
    assert "main.py" in glob_res
    assert "helper.py" in glob_res


# ==============================================================================
# 6. Tests for ResponseParser (<thinking> & <tool_call>)
# ==============================================================================

def test_extract_thinking():
    raw = "<thinking>\nStep 1: Check tests.\nStep 2: Run bash.\n</thinking>\nHere is the answer."
    thinking, remaining = ResponseParser.extract_thinking(raw)
    assert "Step 1: Check tests." in thinking
    assert remaining == "Here is the answer."


def test_extract_tool_calls_json_xml():
    raw = """
<thinking>Planning action</thinking>
Let's inspect the file:
<tool_call>
{"name": "read_file", "arguments": {"path": "main.py", "start_line": 1, "end_line": 20}}
</tool_call>
"""
    thinking, clean, calls = ResponseParser.parse_full_response(raw)
    assert thinking == "Planning action"
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "main.py"
    assert calls[0].arguments["start_line"] == 1


def test_extract_tool_calls_anthropic_invoke():
    raw = """
<invoke name="edit_file">
  <parameter name="path">app.py</parameter>
  <parameter name="search_block">x = 1</parameter>
  <parameter name="replace_block">x = 2</parameter>
</invoke>
"""
    calls = ResponseParser.extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "edit_file"
    assert calls[0].arguments["path"] == "app.py"
    assert calls[0].arguments["search_block"] == "x = 1"


def test_stream_buffer_parser():
    parser = StreamBufferParser()
    deltas = ["<think", "ing>\nThinking", " steps\n</think", "ing>\nHello ", "world!"]
    chunks = []
    for d in deltas:
        chunks.extend(parser.process_delta(d))
    chunks.extend(parser.flush())

    thinking_chunks = [c.content for c in chunks if c.chunk_type == ChunkType.THINKING]
    text_chunks = [c.content for c in chunks if c.chunk_type == ChunkType.TEXT]

    assert "Thinking steps" in "".join(thinking_chunks)
    assert "Hello world!" in "".join(text_chunks)


# ==============================================================================
# 7. Tests for PyTorch CustomOpus Architecture & ModelEngine
# ==============================================================================

def test_pytorch_custom_opus_model():
    cfg = CustomOpusConfig(vocab_size=1000, dim=64, n_layers=2, n_heads=4, max_seq_len=128)
    model = CustomOpusModel(cfg)
    input_ids = torch.randint(0, 1000, (1, 10))
    logits = model(input_ids)
    assert logits.shape == (1, 10, 1000)

    # Test autoregressive token generation
    generated = model.generate_tokens(input_ids, max_new_tokens=3)
    assert generated.shape == (1, 13)


def test_model_engine_switch_target():
    engine = ModelEngine(endpoint_or_path="http://localhost:8080/v1")
    assert engine.backend.value == "http"

    res = engine.switch_target("simulated_mode")
    assert engine.backend.value == "simulated"
    assert "Simulated" in res


# ==============================================================================
# 8. Tests for ProjectContext & Memory (/init)
# ==============================================================================

def test_project_context_init_opus_md(temp_workspace):
    ctx = ProjectContext(root_dir=str(temp_workspace))
    res = ctx.initialize_opus_md()
    assert "Successfully generated" in res
    opus_path = temp_workspace / "OPUS.md"
    assert opus_path.exists()
    content = opus_path.read_text(encoding="utf-8")
    assert "Project Memory & Architecture Guide (OPUS.md)" in content
    assert "Run Tests" in content


# ==============================================================================
# 9. Tests for AgentLoop (Autonomous Tool Calling Loop & Compaction)
# ==============================================================================

def test_agent_loop_turn_with_tool_call(temp_workspace):
    # Setup test file
    sample = temp_workspace / "hello.txt"
    sample.write_text("Hello, Autonomous World!\n", encoding="utf-8")

    gate = ApprovalGate(yolo=True)
    ctx = ProjectContext(root_dir=str(temp_workspace))
    engine = ModelEngine(endpoint_or_path="simulated")

    loop = AgentLoop(model_engine=engine, context=ctx, gate=gate, max_iterations=3)

    # User requests to read the file
    turn_output = loop.run_turn(f"Please read the file '{sample.name}'")
    assert len(loop.messages) >= 3

    # Verify tool execution occurred and observation was appended
    observation_messages = [m for m in loop.messages if "Observation:" in m.get("content", "")]
    assert len(observation_messages) >= 1
    assert "Hello, Autonomous World!" in observation_messages[0]["content"]


def test_agent_loop_history_compaction():
    gate = ApprovalGate(yolo=True)
    engine = ModelEngine(endpoint_or_path="simulated")
    loop = AgentLoop(model_engine=engine, gate=gate)

    # Add 10 dummy turns to history
    for i in range(10):
        loop.messages.append({"role": "user", "content": f"Turn {i} question"})
        loop.messages.append({"role": "assistant", "content": f"Turn {i} response"})

    assert len(loop.messages) == 21  # 1 system + 20 dialogue
    compact_res = loop.compact_history()
    assert "Compacted conversation" in compact_res
    assert len(loop.messages) < 10


# ==============================================================================
# 10. Tests for FastAPI Server (serve_my_opus.py)
# ==============================================================================

def test_fastapi_server_health_and_endpoints():
    from fastapi.testclient import TestClient
    from server.serve_my_opus import app, load_model_weights

    load_model_weights(None)
    client = TestClient(app)

    # Test /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["model"] == "my-opus"
    assert health_resp.json()["ready"] is True

    # Test /v1/chat/completions (OpenAI non-streaming)
    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "my-opus",
            "messages": [{"role": "user", "content": "Hello My-Opus"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]

    # Test /v1/chat/completions (OpenAI streaming)
    stream_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "my-opus",
            "messages": [{"role": "user", "content": "Please run pytest"}],
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    stream_text = stream_resp.text
    assert "data: " in stream_text
    assert "[DONE]" in stream_text

    # Test /v1/messages (Anthropic non-streaming)
    anthropic_resp = client.post(
        "/v1/messages",
        json={
            "model": "my-opus",
            "messages": [{"role": "user", "content": "Read a file"}],
            "stream": False,
        },
    )
    assert anthropic_resp.status_code == 200
    adata = anthropic_resp.json()
    assert adata["type"] == "message"
    assert len(adata["content"]) > 0

    # Test /v1/messages (Anthropic streaming)
    astream_resp = client.post(
        "/v1/messages",
        json={
            "model": "my-opus",
            "messages": [{"role": "user", "content": "Read file foo.py"}],
            "stream": True,
        },
    )
    assert astream_resp.status_code == 200
    assert "event: message_start" in astream_resp.text
    assert "event: message_stop" in astream_resp.text


# ==============================================================================
# 11. Tests for CLI Slash Commands
# ==============================================================================

def test_cli_slash_commands(temp_workspace):
    from custom_claude.cli import TerminalUI

    ui = TerminalUI(model_target="simulated", yolo=False)
    ui.context = ProjectContext(root_dir=str(temp_workspace))
    ui.agent.context = ui.context

    # Test /yolo toggle
    assert ui.gate.yolo is False
    assert ui.handle_slash_command("/yolo") is True
    assert ui.gate.yolo is True
    assert ui.handle_slash_command("/yolo") is True
    assert ui.gate.yolo is False

    # Test /help
    assert ui.handle_slash_command("/help") is True

    # Test /model
    assert ui.handle_slash_command("/model http://localhost:9000/v1") is True
    assert ui.engine.endpoint_or_path == "http://localhost:9000/v1"

    # Test /compact
    ui.agent.messages.extend([{"role": "user", "content": f"msg {i}"} for i in range(10)])
    assert ui.handle_slash_command("/compact") is True

    # Test /init creates OPUS.md
    assert ui.handle_slash_command("/init") is True
    assert (temp_workspace / "OPUS.md").exists()
