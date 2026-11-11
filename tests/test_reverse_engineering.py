"""Test suite for Reverse Engineering, White-Box Distillation, and Rebuild Engine."""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from rebuild_engine.breaker import (
    DECEPTIVE_BUG_TRAPS,
    ASTMutator,
    LoopDetector,
    apply_mutation,
    count_possible_mutations,
    generate_poisoned_context,
)
from rebuild_engine.diagnose_and_synth import FlawDatasetManager, FlawRecord
from reverse_eng.context_packer import build_repo_skeleton, compress_traceback
from reverse_eng.recurrent_opus import (
    CFGLogitMasker,
    RecurrentOpusConfig,
    RecurrentOpusModel,
    TaskAdapter,
)
from reverse_eng.trace_synthesizer import (
    ExecutionTracer,
    TraceSynthesizer,
    compute_ast_backward_trace,
)


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="reverse_eng_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# 1. AST Repo Skeleton Builder & Traceback Compressor
# ==============================================================================

def test_ast_repo_skeleton_builder(temp_workspace):
    sample_file = temp_workspace / "sample_service.py"
    sample_file.write_text(
        '''"""Module docstring."""

class Calculator:
    """Calculator service."""
    factor: int = 10

    def multiply(self, a: int, b: int) -> int:
        """Multiply two ints."""
        val = a * b
        return val * self.factor

def standalone_helper(msg: str) -> bool:
    print(msg)
    return True
''',
        encoding="utf-8",
    )

    skeleton = build_repo_skeleton(temp_workspace)
    assert "class Calculator:" in skeleton
    assert '"""Calculator service."""' in skeleton
    assert "def multiply(self, a: int, b: int) -> int:" in skeleton
    assert "def standalone_helper(msg: str) -> bool:" in skeleton
    # Verify method bodies are stripped to '...'
    assert "val = a * b" not in skeleton
    assert "..." in skeleton


def test_compress_traceback():
    sample_output = """
============================= test session starts =============================
collected 1 item

tests/test_calc.py F                                                     [100%]

================================== FAILURES ===================================
________________________________ test_multiply ________________________________

    def test_multiply():
        c = Calculator()
>       assert c.multiply(2, 3) == 50
E       AssertionError: assert 60 == 50
E       where 60 = c.multiply(2, 3)

tests/test_calc.py:12: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_calc.py::test_multiply - AssertionError: assert 60 == 50
"""
    res = compress_traceback(sample_output)
    assert "test_multiply" in res["failing_test"]
    assert "AssertionError" in res["error_message"]
    assert "assert c.multiply(2, 3) == 50" in res["crash_line"]
    assert len(res["code_window"]) > 0


# ==============================================================================
# 2. Variable Tracer & Divergence Synthesizer
# ==============================================================================

def test_sys_settrace_divergence():
    def buggy_fn(x: int) -> int:
        y = x + 1
        return y

    def fixed_fn(x: int) -> int:
        y = x + 2
        return y

    tracer_b = ExecutionTracer("buggy_fn")
    res_b, snaps_b = tracer_b.run_instrumented(buggy_fn, 5)

    tracer_f = ExecutionTracer("fixed_fn")
    res_f, snaps_f = tracer_f.run_instrumented(fixed_fn, 5)

    assert res_b == 6
    assert res_f == 7
    assert len(snaps_b) > 0
    assert len(snaps_f) > 0

    synth = TraceSynthesizer()
    div = synth.find_divergence(snaps_b, snaps_f)
    assert div["step"] != -1


def test_ast_backward_trace():
    code = """
def compute_total(price, tax_rate, discount):
    subtotal = price * tax_rate
    final = subtotal - discount
    return final
"""
    trace_info = compute_ast_backward_trace(code, "final")
    assert any("price" in line or "subtotal" in line for line in trace_info)


def test_trace_synthesizer_sample_generation(temp_workspace):
    jsonl_path = temp_workspace / "traces.jsonl"
    synth = TraceSynthesizer(output_path=str(jsonl_path))

    sample = synth.synthesize_sample(
        bug_id="fencepost",
        buggy_code="def count(a, b): return b - a",
        fixed_code="def count(a, b): return (b - a) + 1",
        test_func_name="test_count",
        failing_traceback="AssertionError: assert 0 == 1",
    )

    assert "[TRACEBACK_SLICE]" in sample.thinking_block
    assert "[AST_BACKWARD_TRACE]" in sample.thinking_block
    assert "[BOUNDARY_PROOF]" in sample.thinking_block
    assert "[SURGICAL_PLAN]" in sample.thinking_block
    assert jsonl_path.exists()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        assert data["bug_id"] == "fencepost"


# ==============================================================================
# 3. RecurrentOpusModel Architecture & CFG Masker
# ==============================================================================

def test_recurrent_opus_model_forward():
    cfg = RecurrentOpusConfig(
        vocab_size=500,
        dim=64,
        n_layers=3,
        n_heads=4,
        max_seq_len=64,
        num_recurrences=3,
        adapter_dim=16,
    )
    model = RecurrentOpusModel(cfg)
    input_ids = torch.randint(0, 500, (1, 8))

    # Test standard mode
    logits_std = model(input_ids, mode="standard")
    assert logits_std.shape == (1, 8, 500)

    # Test thinking mode with recurrence K=3
    logits_think = model(input_ids, mode="thinking")
    assert logits_think.shape == (1, 8, 500)

    # Test tool mode with tool adapter
    logits_tool = model(input_ids, mode="tool")
    assert logits_tool.shape == (1, 8, 500)


def test_cfg_logit_masker():
    masker = CFGLogitMasker()

    # Valid JSON tool
    valid_json = '{"name": "read_file", "arguments": {"path": "main.py"}}'
    ok, err = masker.validate_tool_json(valid_json)
    assert ok is True
    assert err is None

    # Invalid tool name
    invalid_tool = '{"name": "rm_rf_everything", "arguments": {}}'
    ok2, err2 = masker.validate_tool_json(invalid_tool)
    assert ok2 is False
    assert "Unauthorized" in err2

    # Logit masking
    dummy_logits = torch.randn(1, 500)
    masked = masker.mask_invalid_logits(dummy_logits, prefix_text="<tool_call>", in_tool_call=True)
    assert masked.shape == dummy_logits.shape


# ==============================================================================
# 4. Rebuild Engine: Breaker & Flaw Diagnostics
# ==============================================================================

def test_deceptive_bug_traps_validity():
    assert len(DECEPTIVE_BUG_TRAPS) == 20
    for trap in DECEPTIVE_BUG_TRAPS:
        # Verify valid Python syntax for all buggy and fixed code
        ast.parse(trap.buggy_code)
        ast.parse(trap.fixed_code)
        assert trap.name
        assert trap.category


def test_ast_mutator():
    orig_code = "def check(a, b):\n    return a < b and b == 10\n"
    n_mutations = count_possible_mutations(orig_code)
    assert n_mutations >= 2

    mutated_code, desc = apply_mutation(orig_code, 0)
    assert mutated_code != orig_code
    assert desc is not None
    # Code should still be syntactically valid Python
    ast.parse(mutated_code)


def test_loop_detector():
    detector = LoopDetector(max_turns_per_bug=3)
    # Turn 1
    assert detector.record_turn("read_file", '{"path": "a.py"}') is None
    # Turn 2 - verbatim repeat
    err = detector.record_turn("read_file", '{"path": "a.py"}')
    assert err is not None
    assert "Repeated verbatim" in err


def test_flaw_dataset_manager(temp_workspace):
    jsonl_path = temp_workspace / "flaws.jsonl"
    mgr = FlawDatasetManager(output_path=str(jsonl_path))

    trap = DECEPTIVE_BUG_TRAPS[0]
    golden = mgr.synthesize_golden_thinking(trap, "SHALLOW_THINKING")
    assert "<thinking>" in golden
    assert "1. Execution Trace:" in golden
    assert "2. Boundary Analysis:" in golden
    assert "3. Mutant Kill Strategy:" in golden
    assert "4. Tool Plan:" in golden

    record = FlawRecord(
        prompt="Fix binary search",
        rejected_response="Quick fix",
        failure_type="SHALLOW_THINKING",
        chosen_response=golden,
    )
    mgr.log_record(record)
    assert jsonl_path.exists()


# ==============================================================================
# 5. Rebuilder Tests: ReasoningRebuildLoss & ModelRebuilder
# ==============================================================================

def test_byte_level_code_tokenizer():
    from rebuild_engine.rebuilder import ByteLevelCodeTokenizer

    tok = ByteLevelCodeTokenizer()
    text = "Hello <thinking>Analyze bug</thinking> World"
    ids, mask = tok.encode(text, max_len=32)

    assert len(ids) == 32
    assert len(mask) == 32
    # Verify thinking tokens have mask == 1
    assert ByteLevelCodeTokenizer.THINK_START in ids
    assert ByteLevelCodeTokenizer.THINK_END in ids
    assert mask.sum() > 0


def test_reasoning_rebuild_loss():
    from rebuild_engine.rebuilder import ReasoningRebuildLoss

    criterion = ReasoningRebuildLoss(beta=0.1, thinking_weight=1.5)
    B, S, V = 2, 8, 100

    p_c_logits = torch.randn(B, S, V, requires_grad=True)
    p_r_logits = torch.randn(B, S, V, requires_grad=True)
    ref_c_logits = torch.randn(B, S, V)
    ref_r_logits = torch.randn(B, S, V)

    c_labels = torch.randint(0, V, (B, S))
    r_labels = torch.randint(0, V, (B, S))
    c_mask = torch.ones(B, S)
    r_mask = torch.zeros(B, S)

    loss = criterion(
        p_c_logits, p_r_logits,
        ref_c_logits, ref_r_logits,
        c_labels, r_labels,
        c_mask, r_mask,
    )

    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    loss.backward()
    assert p_c_logits.grad is not None


def test_model_rebuilder_train_and_hot_reload(temp_workspace):
    from rebuild_engine.rebuilder import ModelRebuilder
    from reverse_eng.recurrent_opus import RecurrentOpusConfig, RecurrentOpusModel
    from custom_claude.model_engine import CustomOpusModel, ModelEngine

    # Prepare dummy flaw jsonl
    flaw_file = temp_workspace / "flaw_sample.jsonl"
    record = {
        "prompt": "Fix binary search index",
        "chosen_response": "<thinking>1. Trace inputs. 2. Fix high.</thinking>\ndef search(): pass",
        "rejected_response": "<thinking>Looks fine</thinking>\npass",
        "failure_type": "SHALLOW_THINKING",
    }
    flaw_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cfg = RecurrentOpusConfig(vocab_size=32000, dim=64, n_layers=2, n_heads=4, num_recurrences=2)
    model = RecurrentOpusModel(cfg)

    rebuilder = ModelRebuilder(model, lr=1e-3)
    save_path = temp_workspace / "trained_rebuilt.pt"
    metrics = rebuilder.train_on_flaws(
        flaw_jsonl_path=str(flaw_file),
        epochs=1,
        batch_size=1,
        save_path=str(save_path),
    )

    assert metrics["steps"] >= 1
    assert save_path.exists()

    # Test hot-reloading into CustomOpusModel / ModelEngine
    engine = ModelEngine(endpoint_or_path=str(save_path))
    assert engine.backend.value == "pytorch"
    assert engine.pytorch_model is not None
