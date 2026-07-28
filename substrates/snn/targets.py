"""
targets.py — pluggable target functions for the evolvable-hardware GA.

A `Target` is the single source of truth for one problem: where the input
seeds are, where/how the outputs are read, and the truth table to match.
The growth automaton, circuit interpretation, fitness scorer and GUI all read
their I/O layout from a Target, so adding a new function is just registering a
new Target — no changes to the core library.

Output encoding (per output terminal):
    complement_inputs   feed (HIGH - current) instead of current
    invert_spike        a spike counts as logic 0 (and no-spike as 1)
These are available for experiments with alternate encodings.  All bundled
arithmetic carry outputs use the ordinary direct encoding: a high source pulse
and a spike at the carry terminal both mean logic 1.

Output strategy (per target):
    "heuristic"   legacy behaviour — outputs are the grown neurons nearest the
                  vertical mid-line in the rightmost columns (used by the
                  original half-adder so its saved genome still scores).
    "terminals"   outputs are read at fixed terminal cells; this is what scales
                  to many outputs (full adder, multi-bit adders, custom tables).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from .genome import GRID_SIZE, MAX_ITER
from substrates.nervous.contracts import BehaviorContract, logic_contract

CURRENT_HIGH = 4.0   # nA — logical "1" input current
MIN_SPIKES   = 1     # spikes needed to count as "fired"

Pos = Tuple[int, int]


@dataclass
class OutputTerminal:
    role:              str            # unique label, e.g. "Sum", "Carry", "S0"
    pos:               Pos            # fixed terminal cell (used by "terminals")
    complement_inputs: bool = False
    invert_spike:      bool = False


@dataclass
class Target:
    name:            str
    inputs:          List[Pos]                              # seed positions, bit order (index 0 = first listed)
    outputs:         List[OutputTerminal]
    cases:           List[Tuple[Tuple[int, ...], Tuple[int, ...]]]  # (input_bits, output_bits)
    grid_size:       int = GRID_SIZE
    iters:           int = MAX_ITER
    output_strategy: str = "terminals"                      # "terminals" | "heuristic"
    high:            float = CURRENT_HIGH
    graded:          bool = False                           # smooth partial-credit fitness
    contract:        BehaviorContract = field(default_factory=logic_contract)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)


# ── layout helpers ────────────────────────────────────────────────────────────

def iters_for(grid_size: int) -> int:
    """
    Growth iterations calibrated to the grid's traversal radius: enough for input
    signals to cross to the far side once, but not so many that the cellular
    automaton converges to a genome-independent saturated fixed point (which
    collapses the fitness gradient). For grid_size 9 this is 12 — exactly the
    half-adder's value — so existing targets are unchanged.
    """
    return max(MAX_ITER, grid_size + 3)


def _spread(count: int, grid_size: int) -> List[int]:
    """Evenly spread `count` y-positions down a column of height grid_size."""
    if count <= 0:
        return []
    if count == 1:
        return [grid_size // 2]
    step = (grid_size - 1) / (count - 1)
    return [int(round(i * step)) for i in range(count)]


def _left_seeds(count: int, grid_size: int) -> List[Pos]:
    return [(0, y) for y in _spread(count, grid_size)]


def _right_terminals(roles_specs, grid_size: int) -> List[OutputTerminal]:
    """roles_specs: list of (role, complement_inputs, invert_spike)."""
    ys = _spread(len(roles_specs), grid_size)
    x  = grid_size - 1
    return [OutputTerminal(role=r, pos=(x, y), complement_inputs=ci, invert_spike=inv)
            for (r, ci, inv), y in zip(roles_specs, ys)]


# ── builders ──────────────────────────────────────────────────────────────────

# 2-input boolean functions, as f(a, b) -> bit
_GATES = {
    "AND":  lambda a, b: a & b,
    "OR":   lambda a, b: a | b,
    "XOR":  lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR":  lambda a, b: 1 - (a | b),
    "XNOR": lambda a, b: 1 - (a ^ b),
}


def gate_target(name: str, grid_size: int = GRID_SIZE) -> Target:
    """Single-output 2-input logic gate (direct spike encoding)."""
    fn    = _GATES[name]
    cases = [((a, b), (fn(a, b),)) for a in (0, 1) for b in (0, 1)]
    return Target(
        name=name,
        inputs=_left_seeds(2, grid_size),
        outputs=_right_terminals([(name, False, False)], grid_size),
        cases=cases,
        grid_size=grid_size,
    )


def truth_table_target(name: str, n_inputs: int, output_roles: List[str],
                       rows, grid_size: int = GRID_SIZE,
                       iters: int | None = None) -> Target:
    """
    Build a Target from an explicit truth table.

    rows: iterable of (input_bits_tuple, output_bits_tuple). Any input
    combinations omitted are simply not scored. `iters` defaults to the
    traversal-calibrated value for the grid (see iters_for).
    """
    cases   = [(tuple(int(b) for b in ib), tuple(int(b) for b in ob)) for ib, ob in rows]
    outputs = _right_terminals([(r, False, False) for r in output_roles], grid_size)
    return Target(
        name=name,
        inputs=_left_seeds(n_inputs, grid_size),
        outputs=outputs,
        cases=cases,
        grid_size=grid_size,
        iters=iters_for(grid_size) if iters is None else iters,
    )


def adder_target(n_bits: int, grid_size: int | None = None) -> Target:
    """
    n-bit ripple-carry adder: inputs A0..A(n-1), B0..B(n-1) (LSB index 0),
    outputs S0..S(n-1) and Cout.  Truth table has 2^(2n) rows.

    NB: this is provided for capacity, not speed — a 4-bit adder is 256 rows
    and 5 outputs, which is very slow to evolve. Use small n for real runs.
    """
    n_in  = 2 * n_bits
    n_out = n_bits + 1
    if grid_size is None:
        grid_size = max(GRID_SIZE, n_in, n_out)
    iters = iters_for(grid_size)

    in_pos = _left_seeds(n_in, grid_size)
    roles  = [("S%d" % i, False, False) for i in range(n_bits)] + [("Cout", False, False)]
    outs   = _right_terminals(roles, grid_size)

    cases = []
    for combo in range(2 ** n_in):
        a_bits = [(combo >> i) & 1 for i in range(n_bits)]
        b_bits = [(combo >> (n_bits + i)) & 1 for i in range(n_bits)]
        a = sum(bit << i for i, bit in enumerate(a_bits))
        b = sum(bit << i for i, bit in enumerate(b_bits))
        s = a + b
        in_bits  = tuple(a_bits + b_bits)
        out_bits = tuple([(s >> i) & 1 for i in range(n_bits)] + [(s >> n_bits) & 1])
        cases.append((in_bits, out_bits))

    return Target(name="%d-bit adder" % n_bits, inputs=in_pos, outputs=outs,
                  cases=cases, grid_size=grid_size, iters=iters)


def _half_adder() -> Target:
    """Half-adder with sum/carry read at DISTINCT, vertically-separated terminals.

    (Was the legacy heuristic, which placed both outputs in adjacent middle-right
    columns — so sum and carry landed right on top of each other. Split them to
    opposite ends of the output edge and use the scalable `terminals` strategy.)"""
    return Target(
        name="Half adder",
        inputs=[(0, 3), (0, 5)],
        outputs=[
            OutputTerminal("sum",   (8, 1), complement_inputs=False, invert_spike=False),
            OutputTerminal("carry", (8, 7), complement_inputs=False, invert_spike=False),
        ],
        cases=[
            ((0, 0), (0, 0)),
            ((0, 1), (1, 0)),
            ((1, 0), (1, 0)),
            ((1, 1), (0, 1)),
        ],
        output_strategy="terminals",
    )


def _full_adder() -> Target:
    """A,B,Cin -> Sum,Carry (3 inputs, 2 outputs, terminal strategy)."""
    cases = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                s = a + b + c
                cases.append(((a, b, c), (s & 1, (s >> 1) & 1)))
    return Target(
        name="Full adder",
        inputs=_left_seeds(3, GRID_SIZE),
        outputs=_right_terminals([("Sum", False, False), ("Carry", False, False)], GRID_SIZE),
        cases=cases,
    )


# ── more complex combinational functions (work on SNN, Nervous and LUT) ─────────
# Each is just an explicit truth table, so nothing in the core library changes.
# They span the interesting classes: data routing (mux/demux/decoder), voting,
# wide parity, arithmetic (multiplier) and relational logic (comparator).

def mux2_target(grid_size: int = GRID_SIZE) -> Target:
    """2-to-1 multiplexer: inputs A, B, Sel -> out = Sel ? B : A. The classic
    'route one of two data lines' primitive — needs the select line to gate two
    different paths, which is harder than any single gate."""
    rows = [((a, b, s), (b if s else a,))
            for s in (0, 1) for a in (0, 1) for b in (0, 1)]
    return truth_table_target("2:1 MUX", 3, ["Y"], rows, grid_size=grid_size)


def majority3_target(grid_size: int = GRID_SIZE) -> Target:
    """3-input majority vote: out = 1 iff at least two inputs are high (the
    carry of a full adder, and a fault-tolerant voter)."""
    rows = [((a, b, c), (1 if a + b + c >= 2 else 0,))
            for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    return truth_table_target("Majority-3", 3, ["M"], rows, grid_size=grid_size)


def parity3_target(grid_size: int = GRID_SIZE) -> Target:
    """3-input odd parity: out = a XOR b XOR c. Wider XOR is the textbook hard
    case for these substrates (not linearly separable, no single-gate shortcut)."""
    rows = [((a, b, c), (a ^ b ^ c,))
            for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    return truth_table_target("Parity-3 (XOR3)", 3, ["P"], rows, grid_size=grid_size)


def decoder2to4_target(grid_size: int = GRID_SIZE) -> Target:
    """2-to-4 one-hot decoder: inputs A1 A0 select which of D0..D3 goes high
    (a demultiplex / address-decode primitive — four outputs, exactly one hot)."""
    rows = []
    for a0 in (0, 1):
        for a1 in (0, 1):
            idx = a0 | (a1 << 1)
            rows.append(((a0, a1), tuple(1 if i == idx else 0 for i in range(4))))
    return truth_table_target("2-to-4 decoder", 2, ["D0", "D1", "D2", "D3"],
                              rows, grid_size=grid_size)


def comparator2_target(grid_size: int | None = None) -> Target:
    """2-bit magnitude comparator: A=(A1 A0), B=(B1 B0) -> GT, EQ, LT (one hot
    per case). Relational logic across two multi-bit operands — three coupled
    outputs that must partition every input."""
    if grid_size is None:
        grid_size = max(GRID_SIZE, 4)
    rows = []
    for a in range(4):
        for b in range(4):
            a0, a1 = a & 1, (a >> 1) & 1
            b0, b1 = b & 1, (b >> 1) & 1
            rows.append(((a0, a1, b0, b1),
                         (1 if a > b else 0, 1 if a == b else 0, 1 if a < b else 0)))
    return truth_table_target("2-bit comparator", 4, ["GT", "EQ", "LT"],
                              rows, grid_size=grid_size)


def multiplier2_target(grid_size: int | None = None) -> Target:
    """2x2 unsigned multiplier: A=(A1 A0), B=(B1 B0) -> 4-bit product P0..P3.
    The hardest combinational preset — four outputs, each a different nonlinear
    function of all four inputs (P0 = A0&B0, P3 = A1&B1&(A0|B0)-ish, etc.)."""
    if grid_size is None:
        grid_size = max(GRID_SIZE, 4)
    rows = []
    for a in range(4):
        for b in range(4):
            a0, a1 = a & 1, (a >> 1) & 1
            b0, b1 = b & 1, (b >> 1) & 1
            p = a * b
            rows.append(((a0, a1, b0, b1), tuple((p >> i) & 1 for i in range(4))))
    return truth_table_target("2x2 multiplier", 4, ["P0", "P1", "P2", "P3"],
                              rows, grid_size=grid_size)


# ── registry ──────────────────────────────────────────────────────────────────

TARGETS = {
    "Half adder": _half_adder(),
    "Full adder": _full_adder(),
    "2-bit adder": adder_target(2),
    "2:1 MUX": mux2_target(),
    "Majority-3": majority3_target(),
    "Parity-3 (XOR3)": parity3_target(),
    "2-to-4 decoder": decoder2to4_target(),
    "2-bit comparator": comparator2_target(),
    "2x2 multiplier": multiplier2_target(),
}
for _g in _GATES:
    TARGETS[_g] = gate_target(_g)

# Default target (keeps the half-adder as the out-of-the-box problem).
DEFAULT_TARGET = "Half adder"


def get_target(name: str) -> Target:
    return TARGETS[name]
