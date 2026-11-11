"""Failure diagnostics and dataset generator: Logs flaw traces and synthesizes golden thinking corrections."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rebuild_engine.breaker import (
    DECEPTIVE_BUG_TRAPS,
    ASTMutator,
    BugTrap,
    apply_mutation,
    count_possible_mutations,
)


@dataclass
class FlawRecord:
    prompt: str
    rejected_response: str
    failure_type: str  # SHALLOW_THINKING | BROKEN_DIFF | INFINITE_LOOP | MISSED_MUTANT
    chosen_response: str
    metadata: Optional[Dict[str, Any]] = None


class FlawDatasetManager:
    """Manages logging flaw traces to data/flaw_traces.jsonl and synthesizing corrections."""

    def __init__(self, output_path: str = "data/flaw_traces.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_record(self, record: FlawRecord) -> None:
        """Append a flaw trace record in JSONL format."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            data = asdict(record)
            f.write(json.dumps(data) + "\n")

    def synthesize_golden_thinking(
        self,
        trap: BugTrap,
        failure_type: str,
        mutant_info: Optional[str] = None,
    ) -> str:
        """Synthesize a deep, structured <thinking> trace addressing boundary analysis & mutant kill."""
        mutant_clause = mutant_info if mutant_info else "Verify inverted operators (<, <=, ==) fail boundary tests."

        thinking_block = f"""<thinking>
1. Execution Trace:
   - Function: `{trap.name}` (Category: {trap.category})
   - Trace analysis: {trap.description}
   - Input evaluation: Step through arguments leading to failure state.

2. Boundary Analysis:
   - Identify edge values: zero, empty collections, negative numbers, endpoint inclusiveness, float precision limits.
   - Detect failure trigger: State divergence occurs when boundaries or edge conditions are hit.

3. Mutant Kill Strategy:
   - {mutant_clause}
   - Ensure comprehensive assertion coverage kills any flipped comparison or arithmetic operators.

4. Tool Plan:
   - Apply surgical diff via `edit_file` targeting exact lines.
   - Verify replacement satisfies all invariant tests.
</thinking>
"""
        # Format the surgical edit tool call
        edit_call = (
            f"<tool_call>\n"
            f'{{"name": "edit_file", "arguments": {{"path": "{trap.name}.py", '
            f'"search_block": {json.dumps(trap.buggy_code.strip())}, '
            f'"replace_block": {json.dumps(trap.fixed_code.strip())}}}}}\n'
            f"</tool_call>"
        )

        return f"{thinking_block}\nI will surgically patch `{trap.name}.py` to resolve the {trap.category} defect:\n\n{edit_call}"

    def generate_benchmark_flaws(self) -> List[FlawRecord]:
        """Generate structured benchmark flaw traces across all bug traps and mutations."""
        records: List[FlawRecord] = []

        for trap in DECEPTIVE_BUG_TRAPS:
            # 1. Simulate a shallow thinking failure
            shallow_thinking = (
                f"<thinking>\nThe code looks mostly right, just running basic tests.\n</thinking>\n"
                f"Running tests without checking edge boundaries for `{trap.name}`."
            )
            golden = self.synthesize_golden_thinking(trap, "SHALLOW_THINKING")
            rec1 = FlawRecord(
                prompt=f"Fix the bug in {trap.name}.py:\n\n{trap.buggy_code}",
                rejected_response=shallow_thinking,
                failure_type="SHALLOW_THINKING",
                chosen_response=golden,
                metadata={"category": trap.category, "trap": trap.name},
            )
            records.append(rec1)
            self.log_record(rec1)

            # 2. Check mutation kills
            n_mutants = count_possible_mutations(trap.buggy_code)
            if n_mutants > 0:
                mutated_code, mutation_desc = apply_mutation(trap.buggy_code, 0)
                missed_mutant_response = (
                    f"<thinking>\nTests passed on standard input.\n</thinking>\n"
                    f"Test suite passed despite mutant: {mutation_desc}"
                )
                golden_mutant = self.synthesize_golden_thinking(
                    trap,
                    "MISSED_MUTANT",
                    mutant_info=f"Must detect mutant: {mutation_desc}",
                )
                rec2 = FlawRecord(
                    prompt=f"Audit and kill mutations in {trap.name}.py:\n\n{mutated_code}",
                    rejected_response=missed_mutant_response,
                    failure_type="MISSED_MUTANT",
                    chosen_response=golden_mutant,
                    metadata={"category": trap.category, "mutant": mutation_desc},
                )
                records.append(rec2)
                self.log_record(rec2)

        return records
