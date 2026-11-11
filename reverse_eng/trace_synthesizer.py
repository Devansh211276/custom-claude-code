"""Abductive thinking reconstructor: sys.settrace variable divergence and AST backward tracing."""

from __future__ import annotations

import ast
import copy
import difflib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class VariableSnapshot:
    lineno: int
    event: str
    locals_dump: Dict[str, str]


class ExecutionTracer:
    """Captures line-by-line runtime local variables using sys.settrace."""

    def __init__(self, target_func_name: Optional[str] = None) -> None:
        self.target_func_name = target_func_name
        self.snapshots: List[VariableSnapshot] = []
        self._prev_trace: Optional[Callable] = None

    def trace_callback(self, frame, event: str, arg: Any) -> Any:
        if self.target_func_name and frame.f_code.co_name != self.target_func_name:
            return self.trace_callback

        if event in ("line", "return", "exception"):
            # Serialize locals
            dump: Dict[str, str] = {}
            for k, v in frame.f_locals.items():
                if not k.startswith("__"):
                    try:
                        dump[k] = repr(v)
                    except Exception:
                        dump[k] = str(type(v))

            self.snapshots.append(
                VariableSnapshot(lineno=frame.f_lineno, event=event, locals_dump=dump)
            )
        return self.trace_callback

    def run_instrumented(self, func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, List[VariableSnapshot]]:
        """Run callable under trace and return (result, snapshots)."""
        self.snapshots.clear()
        self._prev_trace = sys.gettrace()
        sys.settrace(self.trace_callback)
        try:
            result = func(*args, **kwargs)
        finally:
            sys.settrace(self._prev_trace)
        return result, list(self.snapshots)


# ==============================================================================
# AST Dataflow & Backward Tracer
# ==============================================================================

class DataflowVisitor(ast.NodeVisitor):
    """Traces dataflow dependencies from inputs to expressions in a function."""

    def __init__(self) -> None:
        self.args: List[str] = []
        self.dependencies: Dict[str, Set[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for a in node.args.args:
            self.args.append(a.arg)
            self.dependencies[a.arg] = {a.arg}
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        rhs_names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        for target in node.targets:
            if isinstance(target, ast.Name):
                deps: Set[str] = set()
                for r in rhs_names:
                    deps.update(self.dependencies.get(r, {r}))
                self.dependencies[target.id] = deps
        self.generic_visit(node)


def compute_ast_backward_trace(code_str: str, failing_target_name: Optional[str] = None) -> List[str]:
    """Determine which function arguments feed into the failing variable."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return ["Unable to parse AST."]

    visitor = DataflowVisitor()
    visitor.visit(tree)

    flow_path: List[str] = []
    flow_path.append(f"Function Arguments: {visitor.args}")

    if failing_target_name and failing_target_name in visitor.dependencies:
        srcs = visitor.dependencies[failing_target_name]
        flow_path.append(f"Variable `{failing_target_name}` depends on inputs: {sorted(list(srcs))}")
    else:
        for var, srcs in visitor.dependencies.items():
            flow_path.append(f"`{var}` <- {sorted(list(srcs))}")

    return flow_path


# ==============================================================================
# Abductive Trace Synthesizer
# ==============================================================================

@dataclass
class ReverseEngineeredSample:
    bug_id: str
    failing_test_name: str
    buggy_code: str
    fixed_code: str
    traceback_slice: str
    ast_backward_trace: List[str]
    boundary_proof: str
    surgical_plan: str
    thinking_block: str
    full_prompt: str
    golden_response: str


class TraceSynthesizer:
    """Reverse-engineers thinking traces from code diffs and test executions."""

    def __init__(self, output_path: str = "data/reverse_engineered_traces.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def find_divergence(
        self,
        buggy_snaps: List[VariableSnapshot],
        fixed_snaps: List[VariableSnapshot],
    ) -> Dict[str, Any]:
        """Detect the exact step where variable states diverge between buggy and fixed runs."""
        min_len = min(len(buggy_snaps), len(fixed_snaps))
        divergence_step = -1
        diverged_vars: Dict[str, Tuple[str, str]] = {}

        for i in range(min_len):
            b_dump = buggy_snaps[i].locals_dump
            f_dump = fixed_snaps[i].locals_dump
            if b_dump != f_dump or buggy_snaps[i].lineno != fixed_snaps[i].lineno:
                divergence_step = i
                # Track differing vars
                all_keys = set(b_dump.keys()) | set(f_dump.keys())
                for k in all_keys:
                    b_val = b_dump.get(k, "<absent>")
                    f_val = f_dump.get(k, "<absent>")
                    if b_val != f_val:
                        diverged_vars[k] = (b_val, f_val)
                break

        return {
            "step": divergence_step,
            "buggy_frame": buggy_snaps[divergence_step] if divergence_step != -1 else None,
            "fixed_frame": fixed_snaps[divergence_step] if divergence_step != -1 else None,
            "diverged_vars": diverged_vars,
        }

    def synthesize_sample(
        self,
        bug_id: str,
        buggy_code: str,
        fixed_code: str,
        test_func_name: str,
        failing_traceback: str,
        buggy_callable: Optional[Callable] = None,
        fixed_callable: Optional[Callable] = None,
        test_args: Tuple = (),
    ) -> ReverseEngineeredSample:
        """Synthesize a complete reverse-engineered training record."""

        # 1. Trace divergence if callables are provided
        divergence_info = "Runtime divergence detected at boundary input."
        traceback_slice = failing_traceback.strip()

        if buggy_callable and fixed_callable:
            tracer_b = ExecutionTracer()
            tracer_f = ExecutionTracer()
            try:
                tracer_b.run_instrumented(buggy_callable, *test_args)
            except Exception as exc:
                traceback_slice = f"Exception: {exc}\n{traceback_slice}"

            try:
                tracer_f.run_instrumented(fixed_callable, *test_args)
            except Exception:
                pass

            div = self.find_divergence(tracer_b.snapshots, tracer_f.snapshots)
            if div["diverged_vars"]:
                divergence_info = f"Variable divergence: {div['diverged_vars']}"

        # 2. AST backward dataflow trace
        ast_trace = compute_ast_backward_trace(buggy_code)

        # 3. Boundary condition proof
        boundary_proof = (
            f"Boundary condition triggered under input {test_args}. "
            f"{divergence_info}"
        )

        # 4. Surgical diff plan
        diff_lines = list(
            difflib.unified_diff(
                buggy_code.splitlines(),
                fixed_code.splitlines(),
                fromfile=f"a/{bug_id}.py",
                tofile=f"b/{bug_id}.py",
                lineterm="",
            )
        )
        surgical_diff = "\n".join(diff_lines)

        # 5. Reverse-engineered <thinking> block
        thinking_block = f"""<thinking>
[TRACEBACK_SLICE]
{traceback_slice}

[AST_BACKWARD_TRACE]
{chr(10).join('- ' + s for s in ast_trace)}

[BOUNDARY_PROOF]
{boundary_proof}

[SURGICAL_PLAN]
Apply search-and-replace:
{surgical_diff}
</thinking>
"""

        tool_call = (
            f"<tool_call>\n"
            f'{{"name": "edit_file", "arguments": {{"path": "{bug_id}.py", '
            f'"search_block": {json.dumps(buggy_code.strip())}, '
            f'"replace_block": {json.dumps(fixed_code.strip())}}}}}\n'
            f"</tool_call>"
        )

        golden_response = f"{thinking_block}\nApplying surgical fix to `{bug_id}.py`:\n{tool_call}"
        full_prompt = f"Fix the failing test `{test_func_name}` in `{bug_id}.py`:\n\n{buggy_code}\n\nFailing Traceback:\n{failing_traceback}"

        sample = ReverseEngineeredSample(
            bug_id=bug_id,
            failing_test_name=test_func_name,
            buggy_code=buggy_code,
            fixed_code=fixed_code,
            traceback_slice=traceback_slice,
            ast_backward_trace=ast_trace,
            boundary_proof=boundary_proof,
            surgical_plan=surgical_diff,
            thinking_block=thinking_block,
            full_prompt=full_prompt,
            golden_response=golden_response,
        )

        # Save to JSONL
        self.save_sample(sample)
        return sample

    def save_sample(self, sample: ReverseEngineeredSample) -> None:
        """Append sample to JSONL file."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            rec = {
                "bug_id": sample.bug_id,
                "test": sample.failing_test_name,
                "prompt": sample.full_prompt,
                "thinking": sample.thinking_block,
                "response": sample.golden_response,
                "diff": sample.surgical_plan,
            }
            f.write(json.dumps(rec) + "\n")
