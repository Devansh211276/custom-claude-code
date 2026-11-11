"""Adversarial stress-test suite: Deceptive bug traps, AST mutators, context poisoning, and loop detection."""

from __future__ import annotations

import ast
import copy
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ==============================================================================
# 1. 20 Deceptive Bug Traps
# ==============================================================================

@dataclass
class BugTrap:
    name: str
    category: str
    description: str
    buggy_code: str
    fixed_code: str
    test_code: str


DECEPTIVE_BUG_TRAPS: List[BugTrap] = [
    # Category 1: Off-by-one errors (1-4)
    BugTrap(
        name="binary_search_midpoint",
        category="off_by_one",
        description="Binary search calculates upper bound without -1 on right split.",
        buggy_code="""def binary_search(arr, target):
    low = 0
    high = len(arr)
    while low <= high:
        mid = (low + high) // 2
        if mid >= len(arr):
            return -1
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return -1
""",
        fixed_code="""def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
        test_code="""def test_binary_search():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0
    assert binary_search([1, 3, 5, 7, 9], 9) == 4
    assert binary_search([1, 3, 5, 7, 9], 4) == -1
    assert binary_search([], 5) == -1
""",
    ),
    BugTrap(
        name="range_slice_inclusive",
        category="off_by_one",
        description="Window slice excludes final item when window equals length.",
        buggy_code="""def get_sliding_windows(items, size):
    windows = []
    for i in range(len(items) - size):
        windows.append(items[i : i + size])
    return windows
""",
        fixed_code="""def get_sliding_windows(items, size):
    windows = []
    if size > len(items) or size <= 0:
        return []
    for i in range(len(items) - size + 1):
        windows.append(items[i : i + size])
    return windows
""",
        test_code="""def test_sliding_windows():
    assert get_sliding_windows([1, 2, 3], 3) == [[1, 2, 3]]
    assert get_sliding_windows([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]
    assert get_sliding_windows([1], 2) == []
""",
    ),
    BugTrap(
        name="paginate_indices",
        category="off_by_one",
        description="Pagination drops the remainder page on non-divisible item counts.",
        buggy_code="""def total_pages(item_count, per_page):
    if item_count == 0:
        return 0
    return item_count // per_page
""",
        fixed_code="""def total_pages(item_count, per_page):
    if item_count <= 0 or per_page <= 0:
        return 0
    return (item_count + per_page - 1) // per_page
""",
        test_code="""def test_total_pages():
    assert total_pages(10, 5) == 2
    assert total_pages(11, 5) == 3
    assert total_pages(0, 5) == 0
    assert total_pages(1, 5) == 1
""",
    ),
    BugTrap(
        name="fencepost_intervals",
        category="off_by_one",
        description="Closed interval counting misses right-endpoint inclusive element.",
        buggy_code="""def count_closed_integers(start, end):
    if end < start:
        return 0
    return end - start
""",
        fixed_code="""def count_closed_integers(start, end):
    if end < start:
        return 0
    return (end - start) + 1
""",
        test_code="""def test_count_closed():
    assert count_closed_integers(5, 5) == 1
    assert count_closed_integers(1, 10) == 10
    assert count_closed_integers(10, 5) == 0
""",
    ),

    # Category 2: Floating-point precision traps (5-8)
    BugTrap(
        name="float_currency_accumulator",
        category="floating_point",
        description="Floating point accumulation introduces epsilon errors in exact ledger balances.",
        buggy_code="""def sum_ledger(transactions):
    total = 0.0
    for t in transactions:
        total += t
    return total == 0.3
""",
        fixed_code="""from decimal import Decimal
def sum_ledger(transactions):
    total = sum(Decimal(str(t)) for t in transactions)
    return total == Decimal('0.3')
""",
        test_code="""def test_sum_ledger():
    assert sum_ledger([0.1, 0.2]) is True
    assert sum_ledger([0.1, 0.1, 0.1]) is True
""",
    ),
    BugTrap(
        name="float_equality_threshold",
        category="floating_point",
        description="Direct equality comparison on float computations without tolerance.",
        buggy_code="""def is_triangle_right_angled(a, b, c):
    sides = sorted([a, b, c])
    return (sides[0] ** 2 + sides[1] ** 2) == (sides[2] ** 2)
""",
        fixed_code="""import math
def is_triangle_right_angled(a, b, c):
    sides = sorted([a, b, c])
    return math.isclose(sides[0] ** 2 + sides[1] ** 2, sides[2] ** 2, rel_tol=1e-9)
""",
        test_code="""def test_right_angled():
    import math
    assert is_triangle_right_angled(1.0, 1.0, math.sqrt(2.0)) is True
    assert is_triangle_right_angled(3.0, 4.0, 5.0) is True
""",
    ),
    BugTrap(
        name="float_discount_rounding",
        category="floating_point",
        description="Floor division instead of standard half-up rounding leads to 1-cent losses.",
        buggy_code="""def apply_discount_cents(price_cents, discount_rate):
    return int(price_cents * (1.0 - discount_rate))
""",
        fixed_code="""from decimal import Decimal, ROUND_HALF_UP
def apply_discount_cents(price_cents, discount_rate):
    p = Decimal(str(price_cents))
    rate = Decimal(str(discount_rate))
    final = p * (Decimal('1.0') - rate)
    return int(final.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
""",
        test_code="""def test_discount():
    # 299 * (1 - 0.15) = 254.15 -> 254
    assert apply_discount_cents(299, 0.15) == 254
    # 150 * (1 - 0.33) = 100.5 -> 101
    assert apply_discount_cents(150, 0.33) == 101
""",
    ),
    BugTrap(
        name="float_average_zero_division",
        category="floating_point",
        description="Subnormal division underflow when computing weighted variance.",
        buggy_code="""def compute_snr(signal, noise):
    if noise == 0.0:
        return 0.0
    return signal / noise
""",
        fixed_code="""def compute_snr(signal, noise):
    if abs(noise) < 1e-12:
        return float('inf') if signal > 0 else (float('-inf') if signal < 0 else 0.0)
    return signal / noise
""",
        test_code="""def test_snr():
    assert compute_snr(10.0, 1e-15) == float('inf')
    assert compute_snr(10.0, 2.0) == 5.0
""",
    ),

    # Category 3: Mutable default arguments (9-12)
    BugTrap(
        name="mutable_list_default",
        category="mutable_defaults",
        description="Persistent list accumulator across function invocations.",
        buggy_code="""def collect_tags(item, tag_list=[]):
    tag_list.append(item)
    return tag_list
""",
        fixed_code="""def collect_tags(item, tag_list=None):
    if tag_list is None:
        tag_list = []
    tag_list.append(item)
    return tag_list
""",
        test_code="""def test_collect_tags():
    first = collect_tags('alpha')
    second = collect_tags('beta')
    assert first == ['alpha']
    assert second == ['beta']
""",
    ),
    BugTrap(
        name="mutable_dict_config",
        category="mutable_defaults",
        description="Default options dictionary shared and mutated across callers.",
        buggy_code="""def configure_worker(name, options={}):
    options['worker'] = name
    return options
""",
        fixed_code="""def configure_worker(name, options=None):
    opts = dict(options) if options is not None else {}
    opts['worker'] = name
    return opts
""",
        test_code="""def test_configure():
    w1 = configure_worker('W1')
    w2 = configure_worker('W2')
    assert w1['worker'] == 'W1'
    assert w2['worker'] == 'W2'
""",
    ),
    BugTrap(
        name="mutable_set_registry",
        category="mutable_defaults",
        description="Shared set accumulator creates cross-session data leakage.",
        buggy_code="""def register_handler(name, registered=set()):
    registered.add(name)
    return len(registered)
""",
        fixed_code="""def register_handler(name, registered=None):
    reg = set(registered) if registered is not None else set()
    reg.add(name)
    return len(reg)
""",
        test_code="""def test_register():
    assert register_handler('a') == 1
    assert register_handler('b') == 1
""",
    ),
    BugTrap(
        name="mutable_nested_cache",
        category="mutable_defaults",
        description="Cached mapping argument shares mutated keys across instances.",
        buggy_code="""def run_cached_step(step_id, cache={'hits': 0}):
    cache['hits'] += 1
    return cache['hits']
""",
        fixed_code="""def run_cached_step(step_id, cache=None):
    c = dict(cache) if cache is not None else {'hits': 0}
    c['hits'] += 1
    return c['hits']
""",
        test_code="""def test_run_cached_step():
    assert run_cached_step(1) == 1
    assert run_cached_step(2) == 1
""",
    ),

    # Category 4: Race conditions / Concurrency traps (13-16)
    BugTrap(
        name="unsynchronized_counter",
        category="race_conditions",
        description="Multiple threads update a shared integer without mutual exclusion.",
        buggy_code="""import threading
class ThreadSafeCounter:
    def __init__(self):
        self.val = 0
    def increment(self):
        cur = self.val
        self.val = cur + 1
""",
        fixed_code="""import threading
class ThreadSafeCounter:
    def __init__(self):
        self.val = 0
        self._lock = threading.Lock()
    def increment(self):
        with self._lock:
            self.val += 1
""",
        test_code="""def test_thread_safe_counter():
    c = ThreadSafeCounter()
    threads = [threading.Thread(target=c.increment) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert c.val == 50
""",
    ),
    BugTrap(
        name="check_then_act_dict",
        category="race_conditions",
        description="Key existence checked before retrieval without thread-safety.",
        buggy_code="""class ConcurrentStore:
    def __init__(self):
        self.data = {}
    def get_or_set(self, key, default):
        if key not in self.data:
            self.data[key] = default
        return self.data[key]
""",
        fixed_code="""import threading
class ConcurrentStore:
    def __init__(self):
        self.data = {}
        self._lock = threading.Lock()
    def get_or_set(self, key, default):
        with self._lock:
            if key not in self.data:
                self.data[key] = default
            return self.data[key]
""",
        test_code="""def test_concurrent_store():
    store = ConcurrentStore()
    assert store.get_or_set('k', 42) == 42
    assert store.get_or_set('k', 99) == 42
""",
    ),
    BugTrap(
        name="double_checked_singleton",
        category="race_conditions",
        description="Singleton instance check not guarded by thread lock.",
        buggy_code="""class Singleton:
    _instance = None
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
""",
        fixed_code="""import threading
class Singleton:
    _instance = None
    _lock = threading.Lock()
    @classmethod
    def get(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
""",
        test_code="""def test_singleton():
    s1 = Singleton.get()
    s2 = Singleton.get()
    assert s1 is s2
""",
    ),
    BugTrap(
        name="producer_consumer_queue_flag",
        category="race_conditions",
        description="Loop terminates on unsynchronized boolean flag before queue drains.",
        buggy_code="""import queue
def drain_queue(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items
""",
        fixed_code="""import queue
def drain_queue(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return items
""",
        test_code="""import queue
def test_drain():
    q = queue.Queue()
    q.put(1); q.put(2)
    assert drain_queue(q) == [1, 2]
""",
    ),

    # Category 5: Silent exception swallowing & Boundary traps (17-20)
    BugTrap(
        name="silent_swallow_error",
        category="exception_swallowing",
        description="Bare except catches KeyboardInterrupt and syntax errors, returning None.",
        buggy_code="""def parse_integer(val):
    try:
        return int(val)
    except:
        return None
""",
        fixed_code="""def parse_integer(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
""",
        test_code="""def test_parse_int():
    assert parse_integer("42") == 42
    assert parse_integer("abc") is None
    import pytest
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt()
        except (ValueError, TypeError):
            pass
""",
    ),
    BugTrap(
        name="swallowed_file_io_corruption",
        category="exception_swallowing",
        description="Swallows IO error during file save, falsely reporting success.",
        buggy_code="""def safe_dump(filename, data):
    try:
        with open(filename, 'w') as f:
            f.write(data)
        return True
    except Exception:
        pass
    return False
""",
        fixed_code="""def safe_dump(filename, data):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)
        return True
    except (OSError, IOError) as exc:
        return False
""",
        test_code="""def test_safe_dump(tmp_path):
    p = str(tmp_path / 'out.txt')
    assert safe_dump(p, 'hello') is True
    assert safe_dump('', 'fail') is False
""",
    ),
    BugTrap(
        name="shallow_copy_nested_mutation",
        category="boundary_mutability",
        description="List copy shallow duplicates nested lists leading to cross-mutation.",
        buggy_code="""def clone_matrix(matrix):
    return list(matrix)
""",
        fixed_code="""import copy
def clone_matrix(matrix):
    return copy.deepcopy(matrix)
""",
        test_code="""def test_clone_matrix():
    m = [[1, 2], [3, 4]]
    c = clone_matrix(m)
    c[0][0] = 99
    assert m[0][0] == 1
""",
    ),
    BugTrap(
        name="dictionary_key_identity_vs_equality",
        category="boundary_mutability",
        description="Boolean True collides with integer 1 as dictionary keys.",
        buggy_code="""def count_unique_tokens(tokens):
    seen = {}
    for t in tokens:
        seen[t] = True
    return len(seen)
""",
        fixed_code="""def count_unique_tokens(tokens):
    seen = set()
    for t in tokens:
        seen.add((type(t), t))
    return len(seen)
""",
        test_code="""def test_token_types():
    assert count_unique_tokens([1, True, "1"]) == 3
    assert count_unique_tokens([0, False]) == 2
""",
    ),
]


# ==============================================================================
# 2. AST Mutator (mutmut replacement in pure Python AST)
# ==============================================================================

class ASTMutator(ast.NodeTransformer):
    """Systematically flips operators in Python AST to test assertion depth."""

    CMP_MAP = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
    }

    BIN_MAP = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.FloorDiv,
        ast.FloorDiv: ast.Mult,
    }

    BOOL_MAP = {
        ast.And: ast.Or,
        ast.Or: ast.And,
    }

    def __init__(self, target_mutation_index: int = 0) -> None:
        super().__init__()
        self.target_index = target_mutation_index
        self.current_index = 0
        self.applied_mutation: Optional[str] = None

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        new_ops = []
        for op in node.ops:
            op_type = type(op)
            if op_type in self.CMP_MAP:
                if self.current_index == self.target_index:
                    replacement = self.CMP_MAP[op_type]()
                    new_ops.append(replacement)
                    self.applied_mutation = f"Flipped {op_type.__name__} to {type(replacement).__name__} at line {getattr(node, 'lineno', '?')}"
                else:
                    new_ops.append(op)
                self.current_index += 1
            else:
                new_ops.append(op)
        node.ops = new_ops
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        op_type = type(node.op)
        if op_type in self.BIN_MAP:
            if self.current_index == self.target_index:
                node.op = self.BIN_MAP[op_type]()
                self.applied_mutation = f"Flipped {op_type.__name__} to {type(node.op).__name__} at line {getattr(node, 'lineno', '?')}"
            self.current_index += 1
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        op_type = type(node.op)
        if op_type in self.BOOL_MAP:
            if self.current_index == self.target_index:
                node.op = self.BOOL_MAP[op_type]()
                self.applied_mutation = f"Flipped {op_type.__name__} to {type(node.op).__name__} at line {getattr(node, 'lineno', '?')}"
            self.current_index += 1
        return node


def count_possible_mutations(source_code: str) -> int:
    """Determine how many mutatable sites exist in the given code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return 0
    mutator = ASTMutator(target_mutation_index=-1)
    mutator.visit(tree)
    return mutator.current_index


def apply_mutation(source_code: str, mutation_index: int) -> Tuple[str, Optional[str]]:
    """Generate a mutated version of the code at the given mutation index."""
    tree = ast.parse(source_code)
    mutator = ASTMutator(target_mutation_index=mutation_index)
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)
    mutated_code = ast.unparse(mutated_tree)
    return mutated_code, mutator.applied_mutation


# ==============================================================================
# 3. Context Poisoning & Long-Context Stress Tester
# ==============================================================================

def generate_poisoned_context(base_traceback: str, noise_depth: int = 15) -> str:
    """Inject deep irrelevant call stacks and noisy logs to test model parsing resilience."""
    frames = []
    for i in range(noise_depth):
        frames.append(
            f'  File "/usr/local/lib/python3.10/site-packages/middleware/framework_{i}.py", line {100 + i * 7}, in execute_wrapper\n'
            f'    yield from dispatch_event(event_id="{1000 + i}", handler="proxy_delegate_{i}")'
        )
    noise_stack = "Traceback (most recent call last):\n" + "\n".join(frames)
    return f"{noise_stack}\n\n[Active Application Error Frame]:\n{base_traceback}"


# ==============================================================================
# 4. Loop Detection Monitor
# ==============================================================================

class LoopDetector:
    """Detects infinite loops, repeated tool calls, or thrashing across turns."""

    def __init__(self, max_turns_per_bug: int = 3) -> None:
        self.max_turns = max_turns_per_bug
        self.call_history: List[Tuple[str, str]] = []
        self.turn_count = 0

    def record_turn(self, tool_name: str, arguments_str: str) -> Optional[str]:
        """Returns error code if a loop condition is detected."""
        self.turn_count += 1
        entry = (tool_name.strip(), arguments_str.strip())

        # Check repeated verbatim call
        if self.call_history and entry == self.call_history[-1]:
            return "INFINITE_LOOP: Repeated verbatim tool call"

        # Check 2-cycle loop
        if len(self.call_history) >= 2 and entry == self.call_history[-2]:
            return "INFINITE_LOOP: Detected oscillating 2-step tool loop"

        self.call_history.append(entry)

        if self.turn_count > self.max_turns:
            return "INFINITE_LOOP: Exceeded turn threshold on single failure"

        return None
