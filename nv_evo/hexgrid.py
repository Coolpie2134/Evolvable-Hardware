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


# A paper tile contains three independent nervous-network circuits.  The names
# are local to the tile (and therefore rotate with ``hex_dirs``), exactly like
# the L/R/D labels used by the developmental context.
DIRECTIONS = ('L', 'R', 'D')
TILE_ENCODING = 'nv-directional-12bit-v1'
GENOME_ENCODING = 'nv-circuit-4bit-v2'
TILE_STATE_MASK = 0xFFF
CIRCUIT_STATE_COUNT = 16
_DIRECTION_INDEX = {direction: index
                    for index, direction in enumerate(DIRECTIONS)}

# Apply one orientation-independent developmental rule to each output circuit.
# The selected output is rotated into the canonical D/front position, matching
# the per-output context rotation used by the paper's cellular substrates.
_CIRCUIT_CONTEXT_ORDER = {
    'D': ('L', 'R', 'D'),
    'R': ('D', 'L', 'R'),
    'L': ('R', 'D', 'L'),
}


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


# ── Figure-3 routing table + legacy one-wire compatibility aliases ─────────────
# Each entry is (E1, E2, I1, OP): two excitatory input directions, one inhibitory
# (None = unused), and the excitatory combining rule. A node fires when
#       (E1 op E2) AND NOT I1
# The paper's Fig. 3 is pure coincidence (op = 'and'): "neither input alone can
# trigger a response." States 0-15 are exactly that table. States 16-31 are the
# same wiring with op = 'or' — fire if EITHER excitatory input is active — a
# historical extension beyond the paper, retained only so old diagnostics/files
# can be decoded. The active three-circuit tile uses ``PAPER_ROUTING`` below and
# cannot select these aliases. For a buffer (E1==E2) or
# an off state OR-mode is identical to AND, so those twins are benign aliases; the
# genuinely new capability was the OR of the coincidence states (either of two
# different lines activates the node). Keeping indices 0-15 unchanged lets the
# checkpoint migrator preserve an old state's wiring selector honestly.
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

# The active Edwards tile uses Figure 3 verbatim: one 4-bit selector for each
# of its three directional output circuits.  ``ROUTING_HEX`` remains exported
# so old files and low-level diagnostics can still decode the former 5-bit
# single-wire model, but evolved tiles only use these first sixteen entries.
PAPER_ROUTING = tuple(ROUTING_HEX[:16])


def pack_tile_state(left, right, down):
    """Pack the three Figure-3 selectors into one 12-bit developmental state.

    Nibble order is stable and deliberately matches ``DIRECTIONS``: L occupies
    bits 0..3, R bits 4..7, and D bits 8..11.  State zero is the completely
    unprogrammed/dead tile described by the paper.
    """
    selectors = tuple(map(int, (left, right, down)))
    if any(not 0 <= selector < CIRCUIT_STATE_COUNT
           for selector in selectors):
        raise ValueError('each core-circuit state must be 0 through 15')
    left, right, down = selectors
    return left | (right << 4) | (down << 8)


def unpack_tile_state(state):
    """Return ``(left, right, down)`` Figure-3 selectors from a tile state."""
    state = int(state)
    if not 0 <= state <= TILE_STATE_MASK:
        raise ValueError('packed nervous phenotype state must be 0 through 4095')
    return state & 0xF, (state >> 4) & 0xF, (state >> 8) & 0xF


def format_tile_state(state):
    """User-facing form of a tile word: three 4-bit circuit configurations.

    A tile does store 12 SRAM bits, but presenting their packed integer as one
    member of a 4096-state menu is misleading.  Figure 3 defines sixteen states
    for *each* L/R/D core circuit, so editors should display ``L/R/D`` nibbles.
    """
    return '/'.join('%X' % selector for selector in unpack_tile_state(state))


def parse_tile_state(text):
    """Parse ``L/R/D`` hexadecimal selectors into the packed hardware word."""
    parts = str(text).strip().replace(',', '/').split('/')
    if len(parts) != len(DIRECTIONS):
        raise ValueError('expected three Figure-3 configurations: L/R/D')
    selectors = [int(part.strip(), 16) for part in parts]
    if any(not 0 <= selector < CIRCUIT_STATE_COUNT for selector in selectors):
        raise ValueError('each Figure-3 configuration must be 0 through F')
    return pack_tile_state(*selectors)


def promote_legacy_state(state):
    """Embed one old single-route state into the three-circuit tile model.

    The former 16..31 states were non-paper OR aliases.  There is no honest OR
    equivalent in Figure 3, so migration retains their wiring selector (low
    nibble), restores paper coincidence semantics, and replicates it over L/R/D.
    """
    state = int(state)
    if not 0 <= state < 32:
        raise ValueError('legacy nervous state must be a 5-bit value (0..31)')
    route = state & 0xF
    return pack_tile_state(route, route, route)


def decode_tile_routing(state):
    """Return the independent ``(L, R, D)`` nervous-circuit configurations."""
    return tuple(PAPER_ROUTING[n] for n in unpack_tile_state(state))


def tile_channel(pos, direction):
    """Canonical output-wire key ``(x, y, direction)`` for one tile circuit."""
    if direction not in DIRECTIONS:
        raise ValueError('unknown nervous-net direction: %r' % (direction,))
    return int(pos[0]), int(pos[1]), direction


def is_channel(pos):
    return (isinstance(pos, (tuple, list)) and len(pos) == 3
            and pos[2] in DIRECTIONS)


def channel_tile(pos):
    """Return the owning tile of either a tile or directional-channel key."""
    return (pos[0], pos[1]) if is_channel(pos) else tuple(pos)


def tile_channels(pos):
    return tuple(tile_channel(pos, direction) for direction in DIRECTIONS)


def facing_channel(pos, incoming_direction):
    """Wire arriving at ``pos`` from one local direction.

    If ``pos`` reads its local L input, for example, this returns the L/R/D
    output circuit on that adjacent tile which physically points back at
    ``pos``.  Computing the reciprocal label instead of assuming it keeps the
    mapping correct under the honeycomb's alternating orientation.
    """
    pos = (int(pos[0]), int(pos[1]))
    neighbour = hex_dirs(*pos)[incoming_direction]
    for direction, candidate in hex_dirs(*neighbour).items():
        if candidate == pos:
            return tile_channel(neighbour, direction)
    raise ValueError('non-reciprocal honeycomb edge at %r %s' %
                     (pos, incoming_direction))


def circuit_selector(grid, channel):
    """Return one channel's 4-bit configuration from a packed phenotype grid."""
    channel = tuple(channel)
    state = grid.get(channel_tile(channel), 0)
    return unpack_tile_state(state)[_DIRECTION_INDEX[channel[2]]]


def developmental_context(grid, pos, output_direction):
    """Three 4-bit neighbour states seen by one developing core circuit.

    The physical L/R/D inputs are the configurations of the neighbour circuits
    that face ``pos``.  They are cyclically rotated for each selected output so
    the same 4-bit gene can develop L, R, and D circuits independently.
    """
    if output_direction not in DIRECTIONS:
        raise ValueError('unknown nervous-net direction: %r' % output_direction)
    incoming = {
        direction: circuit_selector(grid, facing_channel(pos, direction))
        for direction in DIRECTIONS
    }
    return tuple(incoming[direction]
                 for direction in _CIRCUIT_CONTEXT_ORDER[output_direction])


def is_directional_routing(routing):
    """Whether routing values contain three independent circuit entries."""
    if not routing:
        return False
    value = next(iter(routing.values()))
    return (isinstance(value, (tuple, list)) and len(value) == 3
            and isinstance(value[0], (tuple, list)))


def expand_directional_routing(grid, routing):
    """Resolve tile-local direction labels into physical channel-to-channel wires.

    The returned map is suitable for the generic event engine's resolved-source
    mode: ``output_channel -> (E1_channel, E2_channel, I1_channel, op)``.
    """
    resolved = {}
    off = PAPER_ROUTING[0]
    for pos in grid:
        entries = routing.get(pos, (off, off, off))
        for direction, entry in zip(DIRECTIONS, entries):
            e1, e2, i1 = entry[0], entry[1], entry[2]
            source = lambda d: (facing_channel(pos, d) if d is not None else None)
            resolved[tile_channel(pos, direction)] = (
                source(e1), source(e2), source(i1),
                entry[3] if len(entry) > 3 else 'and')
    return resolved


def directional_connections(grid, routing):
    """Return drawable connections between exact core circuits.

    Items are ``(source_channel, destination_channel, role)`` where ``role`` is
    ``'exc'`` or ``'inh'``.  E1=E2 is the paper's single-input buffer: one nerve
    is fanned into the two required excitatory terminals, so it is deliberately
    returned once.  Every destination is an exact L/R/D circuit, which prevents
    three independent circuits in one physical tile from looking like mutually
    contradictory connections to one output.
    """
    connections = []
    live = set(grid)
    for pos in sorted(grid):
        entries = routing.get(pos, ())
        for output_direction, entry in zip(DIRECTIONS, entries):
            destination = tile_channel(pos, output_direction)
            e1, e2, i1 = entry[0], entry[1], entry[2]
            for incoming_direction in dict.fromkeys((e1, e2)):
                if incoming_direction is None:
                    continue
                source = facing_channel(pos, incoming_direction)
                if channel_tile(source) in live:
                    connections.append((source, destination, 'exc'))
            if i1 is not None:
                source = facing_channel(pos, i1)
                if channel_tile(source) in live:
                    connections.append((source, destination, 'inh'))
    return connections


def active_channels(grid, routing):
    """Stable list of programmed directional output wires in live tiles."""
    out = []
    for pos in sorted(grid):
        for direction, entry in zip(DIRECTIONS, routing.get(pos, ())):
            if entry[0] is not None:
                out.append(tile_channel(pos, direction))
    return out


def normalize_output_channel(grid, pos):
    """Return an exact channel key, promoting a legacy whole-tile readout.

    Old hand-built designs stored only ``(x, y)``.  Choose that tile's first
    programmed circuit in stable L/R/D order; if all three are off, retain an L
    key so validation will correctly observe silence instead of moving readout.
    """
    if pos is None or is_channel(pos):
        return None if pos is None else tuple(pos)
    tile = channel_tile(pos)
    state = grid.get(tile, 0)
    for direction, selector in zip(DIRECTIONS, unpack_tile_state(state)):
        if selector:
            return tile_channel(tile, direction)
    return tile_channel(tile, 'L')


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
    if i1 is not None:
        return 'inhibited'                 # any circuit with a veto input
    if e1 == e2:
        return 'buffer'                    # single line — OR and AND coincide
    if _entry_op(entry) == 'or':
        return 'or'                        # fires on EITHER of two lines
    return 'coincidence'
