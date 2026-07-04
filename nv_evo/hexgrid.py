"""
nv_evo/hexgrid.py — honeycomb geometry + the paper's routing table.

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


# ── the paper's 16 useful routing combinations (Edwards EH'02, Fig. 3) ──────────
# Each entry is (E1, E2, I1): the two excitatory input directions and one
# inhibitory direction (None = unused). A node fires when
#       (E1 AND E2) AND NOT I1
# — genuine coincidence detection: "Excitatory inputs E1 and E2 are coupled such
# that neither input alone can trigger a response. The inhibitory input I1, if
# active, will prevent a response even if both excitatory inputs are active."
# Buffer = E1==E2 (single connection, relays one line); coincidence = AND of two
# different lines. There is NO disjunction in the paper — every state is a
# buffer or an AND, optionally vetoed. The table is exactly Figure 3, in order.
#
# Every input comes from a neighbour — no internal source — so with no external
# input the array stays all-zeros ("All switches start with value zero, which
# prevents any signals from propagating"). Sustained activity / memory comes
# from a pulse injected by an input circulating around a loop of buffers, "until
# stopped by application of an inhibitory input" (§3). NB: that circulation is a
# property of the paper's asynchronous, edge-triggered PULSE dynamics; the
# synchronous level relaxation used here is a discretisation of it.
ROUTING_HEX = [
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


def node_fires(entry, value):
    """Boolean output of a routed node (out = (E1 AND E2) AND NOT I1).
    entry = (e1, e2, i1); value(d) gives the 0/1 on direction d. A buffer has
    e1 == e2, so it simply relays that one line."""
    e1, e2, i1 = entry
    if e1 is None or e2 is None:
        return 0
    fired = bool(value(e1) and value(e2))
    if fired and i1 is not None and value(i1):
        fired = False
    return 1 if fired else 0


def routing_kind(entry):
    """Classify a routing entry (e1, e2, i1) for colouring / summaries."""
    e1, e2, i1 = entry
    if e1 is None:
        return 'off'
    if i1 is not None:
        return 'inhibited'
    if e1 == e2:
        return 'buffer'
    return 'coincidence'
