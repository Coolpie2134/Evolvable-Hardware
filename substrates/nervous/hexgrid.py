"""
substrates/nervous/hexgrid.py - honeycomb geometry + the paper's routing table.

The nervous net is a hexagonal (honeycomb) array: each node has exactly THREE
neighbours - Left, Right, and a vertical Down/up that alternates with position
(Edwards EH'02, Fig. 2 & 4: "the concepts right, left and down are rotated as
necessary to match the network topology"). This degree-3 lattice is what forms
the excitatory/inhibitory loops.

We keep integer (x, y) cell coordinates; only the *neighbour* relation is
honeycomb. A cell is drawn offset per row so it reads as a hex lattice.

Directions are given in the node's OWN orientation frame (context rotation),
not absolute compass sides - see hex_dirs.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache


@lru_cache(maxsize=8192)
def hex_dirs(x, y):
    """{'D','R','L'} -> neighbour coordinate, in the node's own orientation frame.

    Each honeycomb node has three neighbours. 'D' is the parity-chosen vertical
    (the node's apex direction); 'R' and 'L' are its clockwise / anticlockwise
    horizontal neighbours *relative to that apex*. Up-oriented nodes (x+y even)
    have their apex one way and down-oriented nodes (x+y odd) are their mirror
    image, so a down-node's 'R' maps to the opposite absolute side.

    Reading L/R/D in this local frame - rather than fixed compass directions -
    is the paper's "context rotation": "the concepts right, left and down are
    rotated as necessary to match the network topology" (Edwards EH'02, Fig. 4).
    It lets one gene express structure symmetrically across both orientations.
    The neighbour *set* is identical to the absolute one (only the R/L labels
    swap by parity), so physical adjacency and frontier growth are unchanged.

    The returned mapping is shared and read-only by convention. Geometry is a
    dominant inner loop for every substrate, while coordinates repeat heavily;
    caching avoids millions of identical three-entry dictionary allocations.
    """
    if (x + y) % 2 == 0:                        # up-node: apex +y, R on the +x side
        return {'D': (x, y + 1), 'R': (x + 1, y), 'L': (x - 1, y)}
    return {'D': (x, y - 1), 'R': (x - 1, y), 'L': (x + 1, y)}  # down-node (mirror)


def hex_frontier_cells(x, y):
    """Honeycomb neighbours of (x, y) for growth frontier expansion. The field
    is UNBOUNDED - "if the field is big enough, the circuit can dictate its own
    boundary" (section 3); size is limited genetically (a chromosome's telomere is a
    Hayflick limit - a cell stops dividing once its telomere runs out) and by
    growth converging to its attractor, not by walls."""
    return list(hex_dirs(x, y).values())


@lru_cache(maxsize=4096)
def _honeycomb_delta_distance(dx, dy, start_parity):
    """Shortest edge distance from the origin on this brick-wall lattice."""
    target = (int(dx), int(dy))
    if target == (0, 0):
        return 0
    pending = deque([((0, 0), 0)])
    seen = {(0, 0)}
    while pending:
        (x, y), distance = pending.popleft()
        if (x + y + int(start_parity)) % 2 == 0:
            neighbours = ((x - 1, y), (x + 1, y), (x, y + 1))
        else:
            neighbours = ((x - 1, y), (x + 1, y), (x, y - 1))
        for neighbour in neighbours:
            if neighbour == target:
                return distance + 1
            if neighbour not in seen:
                seen.add(neighbour)
                pending.append((neighbour, distance + 1))
    raise AssertionError("unbounded honeycomb must be connected")


def honeycomb_distance(left, right):
    """Exact number of honeycomb edges between two integer coordinates."""
    return _honeycomb_delta_distance(
        int(right[0]) - int(left[0]), int(right[1]) - int(left[1]),
        (int(left[0]) + int(left[1])) & 1)


def hex_pixel(x, y):
    """Drawing position of node (x, y): a true honeycomb-VERTEX layout -
    "cells are positioned at the corners of a hex array" (automaton_arrays).
    Nodes sit on zigzag (triangular) rows joined by vertical rungs, so each
    node visibly touches exactly its two diagonal row-neighbours (L, R) and
    one vertical partner (D): the paper's trestle / brick-wall picture, with
    all three edges unit length."""
    return (x * 0.8660254,                                   # sqrt(3)/2
            y * 1.5 + (0.25 if (x + y) % 2 == 0 else -0.25))


# -- routing table: the paper's 16 AND states + 16 OR twins (Edwards EH'02) -------
# Each entry is (E1, E2, I1, OP): two excitatory input directions, one inhibitory
# (None = unused), and the excitatory combining rule. A node fires when
#       (E1 op E2) AND NOT I1
# The paper's Fig. 3 is pure coincidence (op = 'and'): "neither input alone can
# trigger a response." States 0-15 are exactly that table. States 16-31 are the
# same wiring with op = 'or' - fire if EITHER excitatory input is active - a
# deliberate, user-requested extension beyond the paper. For a buffer (E1==E2) or
# an off state OR-mode is identical to AND, so those twins are benign aliases; the
# genuinely new capability is the OR of the coincidence states (either of two
# different lines activates the node). Keeping 0-15 unchanged means every existing
# genome grows and behaves bit-identically (the extra state bit is 0 for them).
#
# Every input comes from a neighbour - no internal source - so with no external
# input the array stays all-zeros ("All switches start with value zero, which
# prevents any signals from propagating"). Memory is a pulse injected by an input
# circulating a loop of buffers "until stopped by an inhibitory input" (section 3), on
# the asynchronous edge-triggered PULSE dynamics (pulse.py).
_ROUTING_BASE = [
    (None, None, None),   # 0  off
    ('D', 'D', None),     # 1  single connection (buffer D)
    ('R', 'R', None),     # 2  buffer R
    ('L', 'L', None),     # 3  buffer L
    ('D', 'R', None),     # 4  coincidence  D & R
    ('L', 'R', None),     # 5  coincidence  L & R
    ('L', 'D', None),     # 6  coincidence  L & D
    ('R', 'R', 'L'),      # 7  buffer R, vetoed by L   (excitatory/inhibitory)
    ('D', 'D', 'L'),      # 8  buffer D, vetoed by L
    ('L', 'L', 'R'),      # 9  buffer L, vetoed by R
    ('D', 'D', 'R'),      # 10 buffer D, vetoed by R
    ('L', 'L', 'D'),      # 11 buffer L, vetoed by D
    ('R', 'R', 'D'),      # 12 buffer R, vetoed by D
    ('R', 'D', 'L'),      # 13 coincidence R & D, vetoed by L
    ('L', 'D', 'R'),      # 14 coincidence L & D, vetoed by R
    ('L', 'R', 'D'),      # 15 coincidence L & R, vetoed by D
]
# 0-15 = AND (the paper's Fig. 3); 16-31 = OR twins (fire on EITHER excitatory).
ROUTING_HEX = ([(e1, e2, i1, 'and') for (e1, e2, i1) in _ROUTING_BASE] +
               [(e1, e2, i1, 'or')  for (e1, e2, i1) in _ROUTING_BASE])


# -- dedicated I/O node types (terminal_nodes io_placement only) ------------------
# Two functionally-REDUNDANT OR-twin alias states are repurposed as dedicated
# input / output NODE TYPES. A cell IS an I/O terminal precisely by growing into
# one of these states (a gene's self_out can encode it), so evolution can grow
# its own I/O. The one-way pulse physics is applied ONLY under the terminal_nodes
# binding strategy:
#   * INPUT  (state 16): source-only - its routing is (None,None,None) so it never
#     fires from neighbours; its wire is driven ONLY by external injection, yet
#     neighbours may still read it, so the input propagates. ("cannot receive")
#   * OUTPUT (state 17): sink-only - it computes/fires from its neighbours (so it
#     is readable as the answer) but NOTHING reads it; it drives nothing into the
#     net. At most ONE output node. ("cannot output")
# Under EVERY OTHER strategy these two states keep their harmless alias behaviour
# (16 == off, 17 == buffer D), so existing runs / checkpoints stay byte-identical.
IO_STATE_INPUT  = 16
IO_STATE_OUTPUT = 17


# -- canonical alphabet ---------------------------------------------------------
# The 5-bit configuration register has 32 settings but only 22 distinct CIRCUITS.
# A routing whose two excitatory inputs are the same cell relays that one line,
# and AND(x, x) == OR(x, x) - so the OR twin of every buffer, and of "off", is
# the identical circuit to its AND original:
#
#     dead <- 0, 16     buffer D <- 1, 17     buffer R <- 2, 18   buffer L <- 3, 19
#     buffer R veto L <- 7, 23    buffer D veto L <-  8, 24   buffer L veto R <-  9, 25
#     buffer D veto R <- 10, 26   buffer L veto D <- 11, 27   buffer R veto D <- 12, 28
#
# Only the six genuine coincidence routings (4, 5, 6, 13, 14, 15) differ from
# their OR twins, giving 12 distinct coincidence circuits.
#
# Left unmanaged this is a 2:1 PRIOR against coincidence - the substrate's only
# computational primitive. Drawing a configuration uniformly over the 32
# encodings makes every buffer (and death) twice as likely as any coincidence
# detector; measured over 60k random genes, buffers landed at ~6.2% each against
# ~3.1% for each coincidence. It also let a genome accumulate alias states, so
# two genes building the identical circuit displayed as different node types and
# a gene emitting state 16 displayed as a node type that is really a dead cell.
#
# Genomes therefore carry CANONICAL states only: one encoding per circuit. The
# register is still physically 5 bits and mutation is still a single bit flip
# (the paper's model); the flip is simply normalised back onto the canonical
# representative afterwards, so a bit that provably changes nothing cannot
# silently consume a mutation event or split one circuit across two node types.
_CANONICAL_ALIAS = {}
for _state in range(len(ROUTING_HEX)):
    _e1, _e2, _i1, _op = ROUTING_HEX[_state]
    # An OR twin is redundant exactly when its two excitatory sources coincide.
    if _state >= 16 and _e1 == _e2:
        _CANONICAL_ALIAS[_state] = _state - 16
del _state, _e1, _e2, _i1, _op

#: Configurations a genome may hold under ordinary binding: one per circuit.
CANONICAL_STATES = tuple(s for s in range(len(ROUTING_HEX))
                         if s not in _CANONICAL_ALIAS)
#: Under ``terminal_nodes`` binding, states 16 and 17 are not aliases at all -
#: they are the dedicated input / output NODE TYPES described above, so they
#: stay drawable and are never normalised away.
CANONICAL_STATES_WITH_TERMINALS = tuple(sorted(
    set(CANONICAL_STATES) | {IO_STATE_INPUT, IO_STATE_OUTPUT}))


#: A redundant encoding is DEAD, not a second name for its AND original. The
#: register really does have 32 settings and only 22 of them are circuits; the
#: other 10 are configurations the hardware cannot usefully hold, so landing on
#: one kills the cell rather than quietly meaning something else.
#:
#: This is deliberately NOT the same as folding an alias onto its twin. Folding
#: made a bit flip onto the AND/OR select line of a buffer a guaranteed no-op -
#: an inert mutation that consumed an event and changed nothing. Death makes the
#: same flip meaningful: it prunes the cell, which is a real and reachable
#: developmental outcome, and it gives mutation a way to remove a cell that does
#: not require finding state 0 exactly.
DEAD_STATE = 0


_IS_ALIAS_TABLE = tuple(s in _CANONICAL_ALIAS for s in range(32))


def canonical_state(state, terminals=False):
    """The stored form of one configuration: itself, or DEAD if it is redundant.

    ``terminals`` keeps the two dedicated I/O node types (16 / 17) distinct,
    which they genuinely are under the ``terminal_nodes`` binding strategy -
    there they are real node types rather than redundant encodings, and killing
    them would erase every terminal an organism grew.
    """
    st = int(state) & 0x1F
    if terminals and (st == IO_STATE_INPUT or st == IO_STATE_OUTPUT):
        return st
    return DEAD_STATE if _IS_ALIAS_TABLE[st] else st



def canonical_states(terminals=False):
    return (CANONICAL_STATES_WITH_TERMINALS if terminals
            else CANONICAL_STATES)


def _entry_op(entry):
    return entry[3] if len(entry) > 3 else 'and'      # tolerate legacy 3-tuples


def node_fires(entry, value):
    """Boolean output of a routed node: out = (E1 op E2) AND NOT I1, where op is
    'and' (coincidence, states 0-15) or 'or' (either input, states 16-31).
    value(d) gives the 0/1 on direction d; a buffer has e1==e2, so AND and OR both
    just relay that one line."""
    e1, e2, i1 = entry[0], entry[1], entry[2]
    if e1 is None or e2 is None:
        return 0
    if _entry_op(entry) == 'or':
        fired = bool(value(e1) or value(e2))
    else:
        fired = bool(value(e1) and value(e2))
    if fired and i1 is not None and value(i1):
        fired = False
    return 1 if fired else 0


def routing_kind(entry):
    """Classify a routing entry for colouring / summaries: off / buffer /
    coincidence (AND) / or (either input) / inhibited (vetoed AND)."""
    e1, e2, i1 = entry[0], entry[1], entry[2]
    if e1 is None:
        return 'off'
    if e1 == e2:
        return 'buffer'                    # single line - OR and AND coincide
    if _entry_op(entry) == 'or':
        return 'or'                        # fires on EITHER of two lines
    if i1 is not None:
        return 'inhibited'
    return 'coincidence'
