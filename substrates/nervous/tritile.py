"""
substrates/nervous/tritile.py - the paper's THREE-circuit tile topology (Architecture 1).

Edwards EH'02 Fig. 2: "each node contains three nervous network circuits. Each
circuit receives inputs from three directions (left, right, and down) and sends
outputs to the same three directions." The legacy engine (pulse.py + the
single-circuit interpretation in nervous.py) collapses that to ONE circuit and
ONE output net per tile, which every listening neighbour reads - so a tile can
only broadcast a single state. This module implements the real tile: three
independently-configured Fig. 3 circuits per tile, one per output direction.

Rather than re-derive the whole event engine, a tri tile is EXPANDED into three
ordinary single-circuit sub-nodes and simulated on the unchanged PulseSim via
its pre-resolved-``sources`` hook:

    tile P, state = 15 bits = chanL | chanR<<5 | chanD<<10
      sub-node (P,'L'): circuit driving P's L output, config = chanL
      sub-node (P,'R'): circuit driving P's R output, config = chanR
      sub-node (P,'D'): circuit driving P's D output, config = chanD

Each sub-node selects E1/E2/I1 among the tile's THREE incoming directional
signals (the paper's 16 useful routing combinations, ROUTING table). The signal
arriving on P's input direction ``e`` is the output of the neighbour Q =
hex_dirs(P)[e] whose own output points back at P - i.e. sub-node
(Q, back_dir(P,e)). An input terminal tile presents its external drive on all
directions through a single (P,'IN') node.

The current target schema names an output TILE, not one of its three pins. As a
compatibility adapter, scoring therefore observes a normalized wired-OR union
of that tile's three output wires. This union is an external readout convention,
not a fourth circuit inside the paper tile. TriSim presents tile-keyed
``rise_times`` / ``pulse_intervals`` / ``ever`` / ``activity_at`` so the temporal
scorer needs no tri-specific branch.
"""
from __future__ import annotations

from .hexgrid import hex_dirs, ROUTING_HEX

# The three output directions of a tile, in bit-field order (chanL = bits 0-4,
# chanR = 5-9, chanD = 10-14).
#
# Each channel is a FIVE-bit config indexing ROUTING_HEX, exactly as a
# single-circuit cell state does: 0-15 are the paper's Fig. 3 coincidence/AND
# routings, 16-31 their OR twins (fire on EITHER excitatory input). The tile was
# originally AND-only, on the grounds that the OR twins are a single-tile
# extension with no place in a paper-faithful tile. That faithfulness had a
# measured cost: under pure AND a lone circulating pulse only survives a node
# whose channel happens to be a buffer, so a sustained ring needs every tile
# around the loop to hold the right buffer on the right channel - and tri tiles
# consequently could not hold a ring at all (Oscillator and Pattern scored ~0
# while input-driven targets scored at or above the single-tile arch). OR
# restores the cheap re-fire path that makes circulating-pulse memory reachable.
TRI_DIRS = ('L', 'R', 'D')
_CHAN_BITS = 5
_CHAN_MASK = (1 << _CHAN_BITS) - 1                       # 0x1F
_DIR_SHIFT = {'L': 0, 'R': _CHAN_BITS, 'D': 2 * _CHAN_BITS}

# Seed tiles start with all three channels as buffer-D (config 1): a live,
# signal-passing tile, mirroring SEED_STATE=1 for the single-circuit array.
TRI_SEED_STATE = 1 | (1 << _CHAN_BITS) | (1 << (2 * _CHAN_BITS))

# Width of the pre-OR layout, kept so old tri3 checkpoints can be widened.
_LEGACY_CHAN_BITS = 4
_LEGACY_CHAN_MASK = (1 << _LEGACY_CHAN_BITS) - 1         # 0xF


def channel_configs(state):
    """(chanL, chanR, chanD): the three 5-bit channel configs packed in a tile
    state. Each indexes ROUTING_HEX (0 = off, 1-15 paper AND, 16-31 OR twin)."""
    return tuple((state >> _DIR_SHIFT[d]) & _CHAN_MASK for d in TRI_DIRS)


def pack_channels(chan_l, chan_r, chan_d):
    """Pack three routing configurations into one 15-bit tile state."""
    values = (int(chan_l), int(chan_r), int(chan_d))
    if any(value < 0 or value > _CHAN_MASK for value in values):
        raise ValueError('tri-tile channel configurations must be in 0..%d'
                         % _CHAN_MASK)
    return (values[0] | (values[1] << _CHAN_BITS)
            | (values[2] << (2 * _CHAN_BITS)))


def widen_legacy_state(state):
    """Re-lay a pre-OR 12-bit tri state into the 15-bit layout.

    The three channel configs are carried across unchanged, and every legacy
    config is 0-15 - the AND half of ROUTING_HEX - so a widened genome grows the
    same grid and simulates identically. Without this an old checkpoint's packed
    bits would be re-cut at the new field boundaries and silently decode as
    different channels.
    """
    value = int(state)
    return pack_channels(value & _LEGACY_CHAN_MASK,
                         (value >> _LEGACY_CHAN_BITS) & _LEGACY_CHAN_MASK,
                         (value >> (2 * _LEGACY_CHAN_BITS)) & _LEGACY_CHAN_MASK)


def back_dir(pos, d):
    """The direction, in the frame of hex_dirs(pos)[d], that points back to
    ``pos``. Adjacency is mutual and this back-direction is unique (asserted in
    tests), so the incoming signal on pos's input ``d`` is unambiguous."""
    nx, ny = hex_dirs(*pos)[d]
    for bd, bp in hex_dirs(nx, ny).items():
        if bp == pos:
            return bd
    raise ValueError('no back-direction from %s toward %s' % ((nx, ny), pos))


def interpret_tri(grid, inputs):
    """Expand a grown tri-tile grid into a single-circuit graph.

    ``grid`` = {tile: 15-bit state}; ``inputs`` = iterable of input-terminal tile
    positions. Returns a dict with:
        nodes      - set of sub-node keys (x, y, d) with d in L/R/D or 'IN'
        routing    - {node: (e1, e2, i1, 'and')} (informational dirs + op)
        sources    - {node: (s1, s2, si)} pre-resolved feeder sub-nodes
        tile_nodes - {tile: [sub-node keys that represent its output]}
        in_nodes   - {input tile: (tile[0], tile[1], 'IN')}
    Off channels (config 0) and channels whose neighbour tile is absent produce
    a dead sub-node (no sources), exactly as an off/edge direction does in the
    single-circuit engine.
    """
    input_set = set(inputs)
    nodes = {}
    sources = {}
    tile_nodes = {tile: [] for tile in grid}
    in_nodes = {}

    # Input terminals present their external drive to every neighbour through
    # one node; their own three circuits are overridden by the drive.
    for tile in grid:
        if tile in input_set:
            key = (tile[0], tile[1], 'IN')
            nodes[key] = (None, None, None, 'and')
            sources[key] = (None, None, None)
            in_nodes[tile] = key
            tile_nodes[tile].append(key)

    def signal_source(tile, e):
        """The sub-node feeding tile's input direction ``e`` (None if the edge
        neighbour is absent)."""
        neigh = hex_dirs(*tile)[e]
        if neigh not in grid:
            return None
        if neigh in input_set:
            return (neigh[0], neigh[1], 'IN')
        return (neigh[0], neigh[1], back_dir(tile, e))

    for tile, state in grid.items():
        if tile in input_set:
            continue
        configs = channel_configs(state)
        for d, cfg in zip(TRI_DIRS, configs):
            key = (tile[0], tile[1], d)
            e1, e2, i1, op = ROUTING_HEX[cfg]
            nodes[key] = (e1, e2, i1, op)
            # A channel drives direction d ONLY if that neighbour exists; a
            # circuit pointing off-grid is inert but harmless (still computed).
            s1 = signal_source(tile, e1) if e1 is not None else None
            s2 = signal_source(tile, e2) if e2 is not None else None
            si = signal_source(tile, i1) if i1 is not None else None
            sources[key] = (s1, s2, si)
            tile_nodes[tile].append(key)

    return {'nodes': nodes, 'routing': nodes, 'sources': sources,
            'tile_nodes': tile_nodes, 'in_nodes': in_nodes}


class _MergedView:
    """Read-only tile-keyed view over a sub-node map, merging each tile's
    sub-nodes. Presents ``.get`` and iteration so it is a drop-in for the
    per-cell dicts the scorer reads (rise_times / pulse_intervals / ever)."""

    def __init__(self, inner, tile_nodes, kind):
        self._inner = inner
        self._tile_nodes = tile_nodes
        self._kind = kind                 # 'rises' | 'intervals' | 'ever'

    def get(self, tile, default=None):
        keys = self._tile_nodes.get(tile)
        if not keys:
            return default
        if self._kind == 'ever':
            return 1 if any(self._inner.get(k, 0) for k in keys) else 0
        merged = []
        for k in keys:
            merged.extend(self._inner.get(k, ()))
        if self._kind == 'rises':
            # For this view ``inner`` is the interval map.  Deriving leading
            # edges from the interval union makes overlapping channel pulses a
            # single physical tile-level wired-OR pulse.
            intervals = sorted((tuple(iv) for iv in merged), key=lambda iv: iv[0])
            union = []
            for start, end in intervals:
                start, end = float(start), float(end)
                if not union or start > union[-1][1]:
                    union.append([start, end])
                elif end > union[-1][1]:
                    union[-1][1] = end
            return [iv[0] for iv in union]
        intervals = sorted((tuple(iv) for iv in merged), key=lambda iv: iv[0])
        union = []
        for start, end in intervals:
            start, end = float(start), float(end)
            if not union or start > union[-1][1]:
                union.append([start, end])
            elif end > union[-1][1]:
                union[-1][1] = end
        return [tuple(iv) for iv in union]

    def __getitem__(self, tile):
        return self.get(tile, [] if self._kind != 'ever' else 0)

    def __contains__(self, tile):
        return tile in self._tile_nodes

    def items(self):
        for tile in self._tile_nodes:
            yield tile, self.get(tile)

    def keys(self):
        return self._tile_nodes.keys()

    def __iter__(self):
        return iter(self._tile_nodes)


class TriSim:
    """Drop-in for PulseSim over a THREE-circuit-per-tile grid.

    Externally keyed by TILE position (x, y): ``inject_pulse``/``activity_at``/
    ``rise_times``/``pulse_intervals``/``ever`` all speak tiles, while the inner
    PulseSim runs on the expanded sub-node graph. This is what lets the temporal
    scorer treat tri and single tiles identically.
    """

    def __init__(self, grid, inputs, config=None, max_events=None,
                 outputs=None):
        from .simulation import create_simulator
        self.grid = grid
        info = interpret_tri(grid, inputs)
        self._tile_nodes = info['tile_nodes']
        self._in_nodes = info['in_nodes']
        output_tiles = set(outputs or ())
        output_nodes = {
            node for tile in output_tiles
            for node in info['tile_nodes'].get(tile, ())}
        # Preserve the external cap's per-tile meaning. Tri3 has three real
        # circuit nodes per non-input tile, so it receives three times the
        # event budget rather than being penalised merely for faithful hardware.
        base_cap = (getattr(config, 'event_cap', None)
                    if max_events is None else max_events)
        inner_cap = None if base_cap is None else 3 * int(base_cap)
        self._sim = create_simulator(
            info['nodes'], info['routing'], max_events=inner_cap,
            config=config, sources=info['sources'],
            input_nodes=set(info['in_nodes'].values()),
            output_nodes=output_nodes)
        self.config = self._sim.config
        self.rise_times = _MergedView(self._sim.pulse_intervals, self._tile_nodes,
                                      'rises')
        self.pulse_intervals = _MergedView(self._sim.pulse_intervals,
                                            self._tile_nodes, 'intervals')
        self.ever = _MergedView(self._sim.ever, self._tile_nodes, 'ever')

    @property
    def overflow(self):
        return self._sim.overflow

    def inject_pulse(self, tile, t, width=None):
        node = self._in_nodes.get(tile)
        if node is not None:
            self._sim.inject_pulse(node, t, width)

    def advance_to(self, when):
        self._sim.advance_to(when)

    def step(self, input_vals):
        inner_vals = {self._in_nodes[tile]: value
                      for tile, value in input_vals.items()
                      if tile in self._in_nodes}
        inner = self._sim.step(inner_vals)
        return {tile: (1 if any(inner.get(k, 0) for k in keys) else 0)
                for tile, keys in self._tile_nodes.items()}

    def activity_at(self, when):
        inner = self._sim.activity_at(when)
        return {tile: (1 if any(inner.get(k, 0) for k in keys) else 0)
                for tile, keys in self._tile_nodes.items()}

    def expanded_signal_graph(self):
        """{node: set(readers)} over sub-nodes - for tri loop_profile."""
        edges = {n: set() for n in self._sim.grid}
        for v, (s1, s2, si) in self._sim.src.items():
            for u in (s1, s2, si):
                if u is not None and u in edges:
                    edges[u].add(v)
        return edges
