"""
substrates/nervous/hexgrid.py — honeycomb geometry + the paper's routing table.

The nervous net is a hexagonal (honeycomb) array: each node has exactly THREE
neighbours — Left, Right, and a vertical Down/up that alternates with position
(Edwards EH'02, Fig. 2 & 4: "the concepts right, left and down are rotated as
necessary to match the network topology"). This degree-3 lattice is what forms
the excitatory/inhibitory loops.

We keep integer (x, y) cell coordinates; only the *neighbour* relation is
honeycomb. A cell is drawn offset per row so it reads as a hex lattice.

Directions are given in the node's OWN orientation frame (context rotation),
not absolute compass sides — see hex_dirs.
"""
from __future__ import annotations


def hex_dirs(x, y):
    """{'D','R','L'} -> neighbour coordinate, in the node's own orientation frame.

    Each honeycomb node has three neighbours. 'D' is the parity-chosen vertical
    (the node's apex direction); 'R' and 'L' are its clockwise / anticlockwise
    horizontal neighbours *relative to that apex*. Up-oriented nodes (x+y even)
    have their apex one way and down-oriented nodes (x+y odd) are their mirror
    image, so a down-node's 'R' maps to the opposite absolute side.

    Reading L/R/D in this local frame — rather than fixed compass directions —
    is the paper's "context rotation": "the concepts right, left and down are
    rotated as necessary to match the network topology" (Edwards EH'02, Fig. 4).
    It lets one gene express structure symmetrically across both orientations.
    The neighbour *set* is identical to the absolute one (only the R/L labels
    swap by parity), so physical adjacency and frontier growth are unchanged.
    """
    if (x + y) % 2 == 0:                        # up-node: apex +y, R on the +x side
        return {'D': (x, y + 1), 'R': (x + 1, y), 'L': (x - 1, y)}
    return {'D': (x, y - 1), 'R': (x - 1, y), 'L': (x + 1, y)}  # down-node (mirror)


def hex_frontier_cells(x, y):
    """Honeycomb neighbours of (x, y) for growth frontier expansion. The field
    is UNBOUNDED — "if the field is big enough, the circuit can dictate its own
    boundary" (§3); size is limited genetically (a chromosome's telomere is a
    Hayflick limit — a cell stops dividing once its telomere runs out) and by
    growth converging to its attractor, not by walls."""
    return list(hex_dirs(x, y).values())


def hex_pixel(x, y):
    """Drawing position of node (x, y): a true honeycomb-VERTEX layout —
    "cells are positioned at the corners of a hex array" (automaton_arrays).
    Nodes sit on zigzag (triangular) rows joined by vertical rungs, so each
    node visibly touches exactly its two diagonal row-neighbours (L, R) and
    one vertical partner (D): the paper's trestle / brick-wall picture, with
    all three edges unit length."""
    return (x * 0.8660254,                                   # sqrt(3)/2
            y * 1.5 + (0.25 if (x + y) % 2 == 0 else -0.25))


# ── routing table: the paper's 16 AND states + 16 OR twins (Edwards EH'02) ───────
# Each entry is (E1, E2, I1, OP): two excitatory input directions, one inhibitory
# (None = unused), and the excitatory combining rule. A node fires when
#       (E1 op E2) AND NOT I1
# The paper's Fig. 3 is pure coincidence (op = 'and'): "neither input alone can
# trigger a response." States 0-15 are exactly that table. States 16-31 are the
# same wiring with op = 'or' — fire if EITHER excitatory input is active — a
# deliberate, user-requested extension beyond the paper. For a buffer (E1==E2) or
# an off state OR-mode is identical to AND, so those twins are benign aliases; the
# genuinely new capability is the OR of the coincidence states (either of two
# different lines activates the node). Keeping 0-15 unchanged means every existing
# genome grows and behaves bit-identically (the extra state bit is 0 for them).
#
# Every input comes from a neighbour — no internal source — so with no external
# input the array stays all-zeros ("All switches start with value zero, which
# prevents any signals from propagating"). Memory is a pulse injected by an input
# circulating a loop of buffers "until stopped by an inhibitory input" (§3), on
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


# ── dedicated I/O node types (terminal_nodes io_placement only) ──────────────────
# Two functionally-REDUNDANT OR-twin alias states are repurposed as dedicated
# input / output NODE TYPES. A cell IS an I/O terminal precisely by growing into
# one of these states (a gene's self_out can encode it), so evolution can grow
# its own I/O. The one-way pulse physics is applied ONLY under the terminal_nodes
# binding strategy:
#   * INPUT  (state 16): source-only — its routing is (None,None,None) so it never
#     fires from neighbours; its wire is driven ONLY by external injection, yet
#     neighbours may still read it, so the input propagates. ("cannot receive")
#   * OUTPUT (state 17): sink-only — it computes/fires from its neighbours (so it
#     is readable as the answer) but NOTHING reads it; it drives nothing into the
#     net. At most ONE output node. ("cannot output")
# Under EVERY OTHER strategy these two states keep their harmless alias behaviour
# (16 == off, 17 == buffer D), so existing runs / checkpoints stay byte-identical.
IO_STATE_INPUT  = 16
IO_STATE_OUTPUT = 17


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
        return 'buffer'                    # single line — OR and AND coincide
    if _entry_op(entry) == 'or':
        return 'or'                        # fires on EITHER of two lines
    if i1 is not None:
        return 'inhibited'
    return 'coincidence'
