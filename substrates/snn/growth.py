from __future__ import annotations
from typing import Dict, Tuple
from .genome import Genome, GRID_SIZE, germline_telomere

SEED_A     = (0, 3)
SEED_B     = (0, 5)
SEED_STATE = 1


# ── associative next-state lookup (min Hamming over the 4 sides + self) ──────────
# The per-gene time `limit` is gone: growth is now bounded by the Hayflick
# telomere (see _grow_step), not by which genes are active on which iteration, so
# the lookup depends on CONTEXT ALONE and its result is cacheable for the whole
# grow (mirrors substrates.nervous._next_state / sim6 table_lookup_cached).

_STATE_BITS = 4
_STATE_MASK = (1 << _STATE_BITS) - 1


def _pack_context(sn, ss, se, sw, si):
    return ((sn & _STATE_MASK)
            | ((ss & _STATE_MASK) << _STATE_BITS)
            | ((se & _STATE_MASK) << (2 * _STATE_BITS))
            | ((sw & _STATE_MASK) << (3 * _STATE_BITS))
            | ((si & _STATE_MASK) << (4 * _STATE_BITS)))


def _compile_lookup(genome):
    """Pack immutable gene contexts once for one developmental run."""
    return tuple(
        (_pack_context(
            gene.state_n, gene.state_s, gene.state_e, gene.state_w,
            gene.self_in), gene.self_out)
        for chromosome in genome.chromosomes
        if not getattr(chromosome, 'wiring', False)
        for gene in chromosome.genes
    )


def _lookup_compiled(program, sn, ss, se, sw, si, packed=None):
    if sn == 0 and ss == 0 and se == 0 and sw == 0 and si == 0:
        return 0
    best_out, best_dist = 0, 1 << 30
    context = (
        _pack_context(sn, ss, se, sw, si)
        if packed is None else packed)
    for gene_context, output in program:
        distance = (gene_context ^ context).bit_count()
        if distance < best_dist:
            best_dist, best_out = distance, output
    return best_out if best_out else 1


def _lookup(genome: Genome, sn: int, ss: int, se: int, sw: int, si: int) -> int:
    """Back-compatible one-shot lookup; growth compiles the genome once."""
    return _lookup_compiled(
        _compile_lookup(genome), sn, ss, se, sw, si)


def cell_io_tags(genome: Genome, grid) -> Dict[Tuple[int, int], int]:
    """Map each live cell to its CELL TYPE: the settled 4-bit state itself —
    the node-type number the growth view shows. The ``genome`` parameter is
    kept for the shared call signature but the type is purely phenotypic.
    Powers the evolvable io_placement strategies (substrates/nervous/io_placement.bind_io);
    deterministic and side-effect free."""
    return {pos: int(state) for pos, state in grid.items()}


def _next_state(program, sn, ss, se, sw, si, cache):
    key = _pack_context(sn, ss, se, sw, si)
    v = cache.get(key)
    if v is None:
        v = _lookup_compiled(
            program, sn, ss, se, sw, si, packed=key)
        cache[key] = v
    return v


_FRONT = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _grow_step(program, grid, tel, seeds, L, grid_size, cache):
    """One development step on a telomere-bounded field. `grid` = {pos: state},
    `tel` = {pos: remaining telomere}. Returns (next_grid, next_tel).

      * maintenance: every live cell re-runs its lookup and keeps its telomere;
      * division: an empty orthogonal-neighbour cell is born only if a live
        neighbour still has telomere > 0 (the Hayflick gate) — the daughter
        inherits (max live-neighbour telomere) - 1. So no cell appears past
        radius L from the seeds, and growth halts on its own;
      * seeds: germline / stem cells, always present at full telomere L.

    The `grid_size` walls remain as an outer safety bound so coordinates stay in
    [0, grid_size) (the growth view and fixed I/O terminals rely on that); the
    telomere is what normally stops growth first."""
    nxt: Dict[Tuple[int, int], int] = {}
    nxt_tel: Dict[Tuple[int, int], int] = {}
    for (x, y), state in grid.items():
        sn = grid.get((x, y + 1), 0); ss = grid.get((x, y - 1), 0)
        se = grid.get((x + 1, y), 0); sw = grid.get((x - 1, y), 0)
        ns = _next_state(program, sn, ss, se, sw, state, cache)
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = tel.get((x, y), 0)
    frontier = set()
    for (x, y) in grid:
        for dx, dy in _FRONT:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in grid:
                frontier.add((nx, ny))
    for (x, y) in frontier:
        parent = max((tel.get((x + dx, y + dy), 0) for dx, dy in _FRONT
                      if (x + dx, y + dy) in grid), default=0)
        if parent <= 0:                                   # every neighbour senescent
            continue                                      # -> no division (Hayflick)
        sn = grid.get((x, y + 1), 0); ss = grid.get((x, y - 1), 0)
        se = grid.get((x + 1, y), 0); sw = grid.get((x - 1, y), 0)
        ns = _next_state(program, sn, ss, se, sw, 0, cache)
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = parent - 1
    for pos in seeds:
        nxt[pos] = SEED_STATE
        nxt_tel[pos] = L
    return nxt, nxt_tel


# Growth is bounded by the telomere (Hayflick radius L) — no external iteration
# cap. Cell addition stops after ~L steps; the remaining lookups settle the
# organism to its attractor (fixed point / 2-cycle), which needs O(L) more, so
# a 3L + margin budget suffices and the loop early-exits on convergence.

def _grow_budget(L):
    return 3 * L + 6


def _run(genome, seeds, grid_size, collect):
    """Develop until the body is fully grown. Cells never die (the lookup always
    returns a live state), so the live-cell SET only ever grows — until the
    Hayflick telomere is spent and no daughter can be born. Maturity is therefore
    "no new cell this step", which halts at ~L steps (telomere-bounded). This
    substrate is recurrent and its states never settle, so — unlike the nervous
    net — there's no point running extra settling steps; the STRUCTURE is the
    organism, read at the moment it stops growing."""
    L = germline_telomere(genome)
    program = _compile_lookup(genome)
    grid = {pos: SEED_STATE for pos in seeds}
    tel  = {pos: L for pos in seeds}
    snaps = [dict(grid)] if collect else None
    cache = {}
    for _ in range(_grow_budget(L)):
        nxt, nxt_tel = _grow_step(
            program, grid, tel, seeds, L, grid_size, cache)
        if collect:
            snaps.append(dict(nxt))
        if nxt.keys() == grid.keys():          # no new cell -> body mature
            return snaps if collect else nxt
        grid, tel = nxt, nxt_tel
    return snaps if collect else grid


# `iters` is accepted but ignored (vestigial) — growth self-limits via the
# telomere. `grid_size` is honoured as the outer wall (default GRID_SIZE).

def grow_snn(genome: Genome, seeds=(SEED_A, SEED_B),
             grid_size: int = GRID_SIZE, iters=None
             ) -> Dict[Tuple[int, int], int]:
    return _run(genome, seeds, grid_size or GRID_SIZE, collect=False)


def grow_snn_snapshots(genome: Genome, seeds=(SEED_A, SEED_B),
                       grid_size: int = GRID_SIZE, iters=None):
    """Return list of grid dicts: seed state followed by each growth step."""
    return _run(genome, seeds, grid_size or GRID_SIZE, collect=True)
