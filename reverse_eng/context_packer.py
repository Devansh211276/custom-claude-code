"""Protocol & context deconstructor: AST repo-skeleton builder and traceback compressor."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from custom_claude.tools import DEFAULT_IGNORES, is_ignored


class SkeletonStripper(ast.NodeTransformer):
    """AST transformer that preserves signatures and docstrings but strips bodies to '...'."""

    def _strip_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        docstring_node = None
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)):
            docstring_node = body[0]

        ellipsis_node = ast.Expr(value=ast.Constant(value=...))
        if docstring_node:
            return [docstring_node, ellipsis_node]
        return [ellipsis_node]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Preserve decorators, name, args, returns
        node.body = self._strip_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Process methods inside class
        new_body: List[ast.stmt] = []
        docstring_node = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            docstring_node = node.body[0]
            new_body.append(docstring_node)

        for stmt in node.body:
            if stmt is docstring_node:
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_body.append(self.visit(stmt))
            elif isinstance(stmt, (ast.AnnAssign, ast.Assign)):
                # Keep class attribute type signatures
                new_body.append(stmt)

        if not new_body or (len(new_body) == 1 and docstring_node):
            new_body.append(ast.Expr(value=ast.Constant(value=...)))

        node.body = new_body
        return node


def build_repo_skeleton(repo_path: Union[str, Path], max_files: int = 50) -> str:
    """Parse Python repository and generate compressed AST skeleton."""
    root = Path(repo_path).resolve()
    if not root.exists():
        return f"Error: Repository path not found: {repo_path}"

    files_to_scan = [root] if root.is_file() else sorted(root.rglob("*.py"))
    skeleton_sections: List[str] = []
    scanned_count = 0

    for py_file in files_to_scan:
        if scanned_count >= max_files:
            skeleton_sections.append(f"# ... (Remaining files truncated at {max_files} files)")
            break

        if root.is_dir() and is_ignored(py_file, root):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            continue

        stripper = SkeletonStripper()
        pruned_tree = stripper.visit(tree)
        ast.fix_missing_locations(pruned_tree)

        try:
            skeleton_code = ast.unparse(pruned_tree)
        except Exception:
            continue

        rel_path = py_file.relative_to(root) if root.is_dir() else py_file.name
        skeleton_sections.append(f"# File: {rel_path}\n{skeleton_code.strip()}\n")
        scanned_count += 1

    return "\n" + ("=" * 70) + "\n\n".join(skeleton_sections)


# ==============================================================================
# Traceback Compressor
# ==============================================================================

def compress_traceback(pytest_output: str, max_window: int = 15) -> Dict[str, Any]:
    """Extract failing assertion, crash frame, local variables, and 15-line code window."""
    lines = pytest_output.splitlines()

    failing_test = "UnknownTest"
    error_message = ""
    crash_line = ""
    locals_captured: Dict[str, str] = {}
    code_window: List[str] = []

    # 1. Detect test name
    for line in lines:
        if "FAILED " in line:
            parts = line.split("FAILED ")
            if len(parts) > 1:
                failing_test = parts[1].split()[0]
                break

    # 2. Detect error line (starts with E )
    error_lines = [l.strip() for l in lines if l.startswith("E   ") or l.startswith("E  ")]
    if error_lines:
        error_message = "\n".join(error_lines)

    # 3. Detect crash frame (starts with > )
    in_failure_block = False
    for i, line in enumerate(lines):
        if line.startswith(">   ") or line.startswith(">  "):
            crash_line = line.strip()
            # Capture around crash line
            start_idx = max(0, i - (max_window // 2))
            end_idx = min(len(lines), i + (max_window // 2) + 1)
            code_window = [l for l in lines[start_idx:end_idx] if not l.startswith("E   ")]
            break

    # 4. Extract local variables / where clauses
    for line in lines:
        if "where " in line:
            # e.g., "where False = binary_search([1], 5)"
            var_match = re.search(r"where\s+(.+?)\s*=\s*(.+)", line.strip())
            if var_match:
                val, expr = var_match.group(1), var_match.group(2)
                locals_captured[expr.strip()] = val.strip()

    compressed_summary = (
        f"Failing Test: {failing_test}\n"
        f"Crash Expression: {crash_line}\n"
        f"Error Details: {error_message}\n"
        f"Captured State: {locals_captured}\n"
        f"Code Window ({len(code_window)} lines):\n"
        + "\n".join(code_window[:max_window])
    )

    return {
        "failing_test": failing_test,
        "crash_line": crash_line,
        "error_message": error_message,
        "locals_captured": locals_captured,
        "code_window": code_window[:max_window],
        "compressed_summary": compressed_summary,
    }
