"""
lut_evo/lut.py — square LUT array, faithful to sim6 (the reference) and the
paper's Architecture 2 (automaton_arrays.pdf: "square array, 4 neighbours,
4 lookup tables per cell, each 16 states").

Each live cell holds FOUR 16-bit lookup tables — one per output direction
(N, S, E, W). A cell's grown state is therefore the tuple (Ln, Ls, Le, Lw).

Growth (associative-memory ontogeny, sim6 build_step): each output direction's
LUT is looked up separately, with the neighbour context ROTATED so the output
direction is "front" — this is the context rotation that lets one gene express
all four directions. The context for direction d is
    (front, back, right, left, self_d)
= the LUTs the four neighbours present toward this cell, reordered relative to
d, plus this cell's own LUT for d; the minimum-Hamming-distance gene supplies
the new LUT. A cell whose four LUTs are all zero is dead. Growth runs to a
stable attractor ("maturity"), capped by `iters`.

Dynamics (sim6 step_network): latched and synchronous. Each cell emits one bit
per direction; the four bits it receives (one from each neighbour, the bit that
neighbour aims back at this cell) form a 4-bit index into each of the cell's
four LUTs, giving its next four output bits. Inputs are OR-injected onto the
input cells' outputs (the perimeter sensory lines).
"""
from __future__ import annotations

LUT_BITS  = 16
# Seed cells are pinned to a fixed live state during growth. 0xFFFE = the "any
# neighbour high" LUT (output 1 for every index except all-zero): a neutral
# relay that passes signals without self-starting.
SEED_LUT   = 0xFFFE
SEED_STATE = (SEED_LUT, SEED_LUT, SEED_LUT, SEED_LUT)

_N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_PC16 = bytes(bin(i).count('1') for i in range(1 << 16))


def _live(state):
    return state[0] or state[1] or state[2] or state[3]


def _lookup(genome, front, back, right, left, self_lut, iteration):
    """Minimum-Hamming gene lookup for one output direction. Args are already
    rotated so `front` is the neighbour in the output direction. Returns the
    new 16-bit LUT (0 = that direction goes dead).

    sim6-faithful size limits: a chromosome's growth rules (self_in == 0) only
    apply while iteration < its telomere, and a dead direction (self_lut == 0)
    can only be brought to life by a growth rule ("if self is zero and not an
    exact match, return zero") — so expansion provably stops once every
    telomere has expired, and maintenance settles the attractor."""
    if front == 0 and back == 0 and right == 0 and left == 0 and self_lut == 0:
        return 0
    pc = _PC16
    best_gene, best_dist = None, 1 << 30
    for chrom in genome.chromosomes:
        expired = iteration >= getattr(chrom, 'telomere', 1 << 30)
        for g in chrom.genes:
            if expired and g.self_in == 0:
                continue
            d = (pc[(g.ctx_n ^ front) & 0xFFFF] + pc[(g.ctx_s ^ back) & 0xFFFF] +
                 pc[(g.ctx_e ^ right) & 0xFFFF] + pc[(g.ctx_w ^ left) & 0xFFFF] +
                 pc[(g.self_in ^ self_lut) & 0xFFFF])
            if d < best_dist:
                best_dist, best_gene = d, g
    if best_gene is None:
        return 0
    if self_lut == 0 and best_gene.self_in != 0:
        return 0
    return best_gene.self_out


def _grow_step(genome, grid, seeds, iteration, cache):
    # expand the frontier: every empty 4-neighbour of a live cell — the field
    # is unbounded; telomeres + the empty-cell guard bound the organism.
    frontier = {}
    for (x, y) in list(grid):
        for dx, dy in _N4:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in grid and (nx, ny) not in frontier:
                frontier[(nx, ny)] = (0, 0, 0, 0)
    working = {**grid, **frontier}
    nxt = {}
    # Memoized lookup, exactly sim6's table_lookup_cached: neighbourhood
    # contexts repeat massively across cells (sim6 measured ~94%), so cache
    # per (context, telomere-expiry mask) within one growth run.
    mask = tuple(iteration >= getattr(c, 'telomere', 1 << 30)
                 for c in genome.chromosomes)

    def look(f, b, r, l, s):
        key = (f, b, r, l, s, mask)
        v = cache.get(key)
        if v is None:
            v = _lookup(genome, f, b, r, l, s, iteration)
            cache[key] = v
        return v

    for (x, y), st in working.items():
        # LUTs each neighbour presents toward this cell: north neighbour's
        # south-LUT (index 1), south neighbour's north-LUT (0), east's west (3),
        # west's east (2).  Absent neighbour -> 0.
        n = working.get((x, y + 1), (0, 0, 0, 0))[1]
        s = working.get((x, y - 1), (0, 0, 0, 0))[0]
        e = working.get((x + 1, y), (0, 0, 0, 0))[3]
        w = working.get((x - 1, y), (0, 0, 0, 0))[2]
        # each output direction: (front, back, right, left, self_d)
        ln = look(n, s, e, w, st[0])
        ls = look(s, n, w, e, st[1])
        le = look(e, w, s, n, st[2])
        lw = look(w, e, n, s, st[3])
        if ln or ls or le or lw:
            nxt[(x, y)] = (ln, ls, le, lw)
    for pos in seeds:
        nxt[pos] = SEED_STATE
    return nxt


def grow_lut(genome, seeds, grid_size, iters):
    """Grow on the unbounded field to the attractor (fixed point or 2-cycle);
    `grid_size` is only the target's I/O layout scale; `iters` is a safety cap.
    Returns {(x,y): (Ln, Ls, Le, Lw)}."""
    grid = {pos: SEED_STATE for pos in seeds}
    prev, cache = None, {}
    for it in range(iters):
        nxt = _grow_step(genome, grid, seeds, it, cache)
        if nxt == grid or nxt == prev:
            return nxt
        prev, grid = grid, nxt
    return grid


def grow_lut_snapshots(genome, seeds, grid_size, iters):
    snaps = [{pos: SEED_STATE for pos in seeds}]
    grid, prev, cache = dict(snaps[0]), None, {}
    for it in range(iters):
        nxt = _grow_step(genome, grid, seeds, it, cache)
        snaps.append(dict(nxt))
        if nxt == grid or nxt == prev:
            break
        prev, grid = grid, nxt
    return snaps


# ── synchronous latched dynamics (sim6 step_network) ────────────────────────────

class LutSim:
    """Synchronous simulation of a grown LUT array. sim.step({cell: bit})
    advances one tick and returns {cell: 0/1} (1 = the cell emits on any
    direction). sim.ever marks cells that have ever emitted."""

    # per-cell output nibble bit positions: N=8, S=4, E=2, W=1
    def __init__(self, grid, _unused=None):
        self.grid = grid
        self.out  = {c: 0 for c in grid}        # 4-bit output: N S E W
        self.ever = {c: 0 for c in grid}

    def step(self, input_vals):
        prev, nxt = self.out, {}
        grid = self.grid
        for (x, y), (ln, ls, le, lw) in grid.items():
            # index bits: from-north = N-neighbour's S-output, etc.
            idx = (((prev.get((x, y + 1), 0) & 0x4) >> 2)        # from N -> bit0
                   | ((prev.get((x, y - 1), 0) & 0x8) >> 2)      # from S -> bit1
                   | ((prev.get((x + 1, y), 0) & 0x1) << 2)      # from E -> bit2
                   | ((prev.get((x - 1, y), 0) & 0x2) << 2))     # from W -> bit3
            o = (((ln >> idx) & 1) << 3 | ((ls >> idx) & 1) << 2
                 | ((le >> idx) & 1) << 1 | ((lw >> idx) & 1))
            if input_vals.get((x, y)):          # OR-injected sensory line
                o |= 0xF
            nxt[(x, y)] = o
            if o:
                self.ever[(x, y)] = 1
        self.out = nxt
        return {c: (1 if o else 0) for c, o in nxt.items()}
