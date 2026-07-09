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

import numpy as np

LUT_BITS  = 16
# Seed cells are pinned to a fixed live state during growth. 0xFFFE = the "any
# neighbour high" LUT (output 1 for every index except all-zero): a neutral
# relay that passes signals without self-starting.
SEED_LUT   = 0xFFFE
SEED_STATE = (SEED_LUT, SEED_LUT, SEED_LUT, SEED_LUT)

_N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
_PC16 = bytes(bin(i).count('1') for i in range(1 << 16))
_PC16_ARR = np.frombuffer(_PC16, dtype=np.uint8)   # vectorised popcount table
_EXPIRE = 1000                                     # > max Hamming (5*16): excludes a gene


def _live(state):
    return state[0] or state[1] or state[2] or state[3]


def _genome_lut_arrays(genome):
    """Pack a genome's genes into column arrays (chromosome-major order) for the
    vectorised min-Hamming lookup: the five context fields, the outputs, a
    growth-rule mask (self_in == 0) and each gene's chromosome telomere. Built
    once per grow_lut call."""
    n, cn, ce, cs, cw, si, so, tel = 0, [], [], [], [], [], [], []
    big = 1 << 30
    for c in genome.chromosomes:
        ct = getattr(c, 'telomere', big)
        for g in c.genes:
            cn.append(g.ctx_n); ce.append(g.ctx_e); cs.append(g.ctx_s)
            cw.append(g.ctx_w); si.append(g.self_in); so.append(g.self_out)
            tel.append(ct); n += 1
    a = lambda xs: np.array(xs, dtype=np.int64) if xs else np.zeros(0, np.int64)
    Ain = a(si)
    return (a(cn), a(cs), a(ce), a(cw), Ain, so, (Ain == 0), a(tel))


def _lookup_vec(garr, front, back, right, left, self_lut, iteration):
    """Vectorised minimum-Hamming gene lookup for one output direction (args
    already rotated so `front` is the output direction). Behaviour is identical
    to the scalar reference `_lookup`: same 80-bit Hamming metric, FIRST gene
    wins at equal distance (argmin), expired growth rules excluded, and the
    empty-context / empty-cell guards. Returns the new 16-bit LUT (0 = dead).

    sim6-faithful size limits: a chromosome's growth rules (self_in == 0) only
    apply while iteration < its telomere, and a dead direction (self_lut == 0)
    can only be brought to life by a growth rule — so expansion provably stops
    once every telomere has expired, and maintenance settles the attractor."""
    return _lookup_vec_idx(garr, front, back, right, left, self_lut, iteration)[0]


def _lookup_vec_idx(garr, front, back, right, left, self_lut, iteration):
    """As _lookup_vec, but also returns the WINNING flat gene index (chromosome-
    major), or -1 when no gene shaped the output (all-zero context, or only
    expired growth rules remained). The empty-cell guard still returns the
    winner's index: that gene WON the argmin and its identity caused the 0, so
    it counts as expressed — removing it would change the argmin (not neutral).
    Used for expression tracking / neutral compaction; see lut_evo.ga."""
    An, As, Ae, Aw, Ain, Aout, Agrow, Atel = garr
    if An.shape[0] == 0:
        return 0, -1
    if front == 0 and back == 0 and right == 0 and left == 0 and self_lut == 0:
        return 0, -1
    PC = _PC16_ARR
    d = (PC[An ^ front].astype(np.int32) + PC[As ^ back] + PC[Ae ^ right]
         + PC[Aw ^ left] + PC[Ain ^ self_lut])
    if iteration:                              # telomere expiry only bites once t>0
        expired = Agrow & (Atel <= iteration)
        if expired.any():
            d[expired] += _EXPIRE
            j = int(d.argmin())
            if expired[j]:                     # only expired genes left -> no match
                return 0, -1
        else:
            j = int(d.argmin())
    else:
        j = int(d.argmin())
    if self_lut == 0 and Ain[j] != 0:          # empty-cell guard (sim6)
        return 0, j                            # gene j won; its identity mattered
    return Aout[j], j


# scalar reference kept for clarity / tests (growth uses the vectorised path)
def _lookup(genome, front, back, right, left, self_lut, iteration):
    return _lookup_vec(_genome_lut_arrays(genome), front, back, right, left,
                       self_lut, iteration)


def _grow_step(genome, garr, grid, seeds, iteration, cache, counts=None):
    # expand the frontier: every empty 4-neighbour of a live cell — the field
    # is unbounded; telomeres + the empty-cell guard bound the organism.
    # `counts` (list per flat gene, chromosome-major) tallies gene EXPRESSION
    # when tracking (see grow_lut_tracked); None = the zero-overhead fast path.
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

    if counts is None:                          # fast path (unchanged)
        def look(f, b, r, l, s):
            key = (f, b, r, l, s, mask)
            v = cache.get(key)
            if v is None:
                v = _lookup_vec(garr, f, b, r, l, s, iteration)
                cache[key] = v
            return v
        for (x, y), st in working.items():
            n = working.get((x, y + 1), (0, 0, 0, 0))[1]
            s = working.get((x, y - 1), (0, 0, 0, 0))[0]
            e = working.get((x + 1, y), (0, 0, 0, 0))[3]
            w = working.get((x - 1, y), (0, 0, 0, 0))[2]
            ln = look(n, s, e, w, st[0]); ls = look(s, n, w, e, st[1])
            le = look(e, w, s, n, st[2]); lw = look(w, e, n, s, st[3])
            if ln or ls or le or lw:
                nxt[(x, y)] = (ln, ls, le, lw)
    else:                                       # tracking path (expression tally)
        def look(f, b, r, l, s):
            key = (f, b, r, l, s, mask)
            vj = cache.get(key)
            if vj is None:
                vj = _lookup_vec_idx(garr, f, b, r, l, s, iteration)
                cache[key] = vj
            return vj
        for (x, y), st in working.items():
            n = working.get((x, y + 1), (0, 0, 0, 0))[1]
            s = working.get((x, y - 1), (0, 0, 0, 0))[0]
            e = working.get((x + 1, y), (0, 0, 0, 0))[3]
            w = working.get((x - 1, y), (0, 0, 0, 0))[2]
            (ln, jn) = look(n, s, e, w, st[0]); (ls, js) = look(s, n, w, e, st[1])
            (le, je) = look(e, w, s, n, st[2]); (lw, jw) = look(w, e, n, s, st[3])
            # count EVERY argmin winner, even for a cell that stays dead: that
            # gene's identity is what produced the 0, so removing it could change
            # the argmin and bring the cell to life — it is NOT neutral to drop.
            for j in (jn, js, je, jw):
                if j >= 0:
                    counts[j] += 1
            if ln or ls or le or lw:
                nxt[(x, y)] = (ln, ls, le, lw)
    for pos in seeds:
        nxt[pos] = SEED_STATE
    return nxt


def grow_lut(genome, seeds, grid_size, iters):
    """Grow on the unbounded field to the attractor (fixed point or 2-cycle);
    `grid_size` is only the target's I/O layout scale; `iters` is a safety cap.
    Returns {(x,y): (Ln, Ls, Le, Lw)}."""
    garr = _genome_lut_arrays(genome)
    grid = {pos: SEED_STATE for pos in seeds}
    prev, cache = None, {}
    for it in range(iters):
        nxt = _grow_step(genome, garr, grid, seeds, it, cache)
        if nxt == grid or nxt == prev:
            return nxt
        prev, grid = grid, nxt
    return grid


def grow_lut_tracked(genome, seeds, grid_size, iters):
    """Grow exactly like grow_lut, but also return a per-gene EXPRESSION count
    (chromosome-major, one entry per gene) tallying how many cell-directions each
    gene builds across the whole developmental trajectory. A gene with count 0
    won no growth lookup and so is phenotype-neutral — see lut_evo.ga.compact.
    Returns (grid, counts)."""
    garr = _genome_lut_arrays(genome)
    counts = [0] * garr[0].shape[0]
    grid = {pos: SEED_STATE for pos in seeds}
    prev, cache = None, {}
    for it in range(iters):
        nxt = _grow_step(genome, garr, grid, seeds, it, cache, counts)
        if nxt == grid or nxt == prev:
            return nxt, counts
        prev, grid = grid, nxt
    return grid, counts


def grow_lut_snapshots(genome, seeds, grid_size, iters):
    garr = _genome_lut_arrays(genome)
    snaps = [{pos: SEED_STATE for pos in seeds}]
    grid, prev, cache = dict(snaps[0]), None, {}
    for it in range(iters):
        nxt = _grow_step(genome, garr, grid, seeds, it, cache)
        snaps.append(dict(nxt))
        if nxt == grid or nxt == prev:
            break
        prev, grid = grid, nxt
    return snaps


# ── synchronous latched dynamics (sim6 step_network) ────────────────────────────

class LutSim:
    """Synchronous simulation of a grown LUT array, VECTORISED with numpy.

    The dynamics are a pure array update — each cell's next 4-bit output is a
    lookup in its four LUTs at the 4-bit index formed from its neighbours'
    facing bits — so the whole grid advances in one set of numpy ops instead of
    a per-cell Python loop with four dict lookups each (that loop was ~85% of
    LUT evaluation time; dense biomorph organisms are 500-900 cells). Behaviour
    is identical: same nibble layout (N=8 S=4 E=2 W=1), same wired-OR input
    injection, same fixed-point/2-cycle attractor.

    API is unchanged: ``sim.step({cell: bit})`` advances one tick and returns
    {cell: 0/1}; ``sim.out`` is the {cell: nibble} map; ``sim.ever`` marks cells
    that have ever emitted. ``sim.run_bits(streams, in_pos, T)`` is a fast bulk
    runner (T ticks with no per-tick dict) returning a [T, ncells] bit matrix
    for scoring — see lut_evo.ga.place_outputs_by_trace.
    """

    def __init__(self, grid, _unused=None):
        self.grid  = grid
        cells      = list(grid)
        self._cells = cells
        self.n     = n = len(cells)
        self._cidx = {c: i for i, c in enumerate(cells)}
        # four LUT columns as int arrays (index n is a zero-padding sentinel so
        # an absent neighbour reads 0 without a branch)
        self._Ln = np.fromiter((grid[c][0] for c in cells), dtype=np.int64, count=n)
        self._Ls = np.fromiter((grid[c][1] for c in cells), dtype=np.int64, count=n)
        self._Le = np.fromiter((grid[c][2] for c in cells), dtype=np.int64, count=n)
        self._Lw = np.fromiter((grid[c][3] for c in cells), dtype=np.int64, count=n)

        def col(dx, dy):     # column of each cell's (dx,dy) neighbour (n = none)
            ci = self._cidx
            return np.fromiter((ci.get((c[0] + dx, c[1] + dy), n) for c in cells),
                               dtype=np.intp, count=n)
        self._nN, self._nS = col(0, 1), col(0, -1)
        self._nE, self._nW = col(1, 0), col(-1, 0)
        self._out  = np.zeros(n, dtype=np.int64)      # current nibble per cell
        self._pe   = np.zeros(n + 1, dtype=np.int64)  # padded prev (index n = 0)
        self._ever = np.zeros(n, dtype=bool)
        self._out_dict = None                         # lazy {cell: nibble} cache

    def _advance(self, inject_cols):
        """One tick in numpy. inject_cols: iterable of column indices forced high
        (wired-OR sensory injection) this tick. Returns the new nibble array."""
        pe = self._pe
        pe[:self.n] = self._out
        idx = (((pe[self._nN] & 0x4) >> 2)         # from N -> bit0
               | ((pe[self._nS] & 0x8) >> 2)       # from S -> bit1
               | ((pe[self._nE] & 0x1) << 2)       # from E -> bit2
               | ((pe[self._nW] & 0x2) << 2))      # from W -> bit3
        o = ((((self._Ln >> idx) & 1) << 3) | (((self._Ls >> idx) & 1) << 2)
             | (((self._Le >> idx) & 1) << 1) | ((self._Lw >> idx) & 1))
        if inject_cols is not None and len(inject_cols):
            o[inject_cols] = 0xF
        self._out = o
        self._ever |= o.astype(bool)
        self._out_dict = None
        return o

    def _inject_cols(self, input_vals):
        ci = self._cidx
        return [ci[c] for c, b in input_vals.items() if b and c in ci]

    def step(self, input_vals):
        o = self._advance(self._inject_cols(input_vals))
        cells = self._cells
        return {cells[i]: (1 if o[i] else 0) for i in range(self.n)}

    def run_bits(self, streams, in_pos, T):
        """Run T ticks driven by `streams` (streams[t] = input bits for in_pos)
        with NO per-tick dict; return a [T, ncells] uint8 emit-bit matrix. Cell
        i's column is self._cidx[cell]. The scoring hot path (no dict churn).
        T may exceed len(streams): the extra ticks run with zero input (padding),
        so a delayed output's late events are still observed."""
        in_cols = [self._cidx.get(p) for p in in_pos]
        ns = len(streams)
        B = np.zeros((T, self.n), dtype=np.uint8)
        for t in range(T):
            if t < ns:
                row = streams[t]
                inj = [in_cols[i] for i in range(len(in_cols))
                       if row[i] and in_cols[i] is not None]
            else:
                inj = []
            B[t] = self._advance(inj) != 0
        return B

    def reset(self):
        self._out[:] = 0
        self._ever[:] = False
        self._out_dict = None

    @property
    def out(self):
        if self._out_dict is None:
            o = self._out
            self._out_dict = {self._cells[i]: int(o[i]) for i in range(self.n)}
        return self._out_dict

    @property
    def ever(self):
        e = self._ever
        return {self._cells[i]: (1 if e[i] else 0) for i in range(self.n)}
