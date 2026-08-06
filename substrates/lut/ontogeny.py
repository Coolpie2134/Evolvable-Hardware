"""
substrates/lut/ontogeny.py - sim6-faithful morphogenesis (STAGE 0 diagnostic).

The substrates/lut GA evolves FIXED genomes: a small random gene set, matched by
minimum Hamming distance during growth. sim6 (sim6_faster.py) does something
different and richer - it *generates* the genome during growth via
`table_create`: an on-the-fly ontogeny that invents a new gene for every context
it has not seen, and frequently sets a new cell's output by COPYING a
neighbour's facing LUT (so structure propagates and mutates through space)
rather than a fixed constant. That, plus context-dependent per-gene limits
(`set_limit`) and an asymmetric single-cell seed, is what gives sim6 its varied,
irregular biomorphs instead of the near-uniform diamonds the GA port produces.

This module ports sim6's `table_create` / `set_limit` / build loop faithfully in
MECHANISM (not bit-for-bit node ordering) so we can SEE whether that ontogeny
restores shape variety on our substrate, before deciding whether to fold it back
into the GA (Stage 1). It shares nothing with the GA path; it only borrows the
grid tuple format `(Ln, Ls, Le, Lw)` and directional conventions from lut.py so
the result renders with the existing `draw_lut_net`.

Run it:  py -m substrates.lut.ontogeny [seed] [n_morphs]
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .genome import (LutGene, Chromosome, Genome, MAX_CHROMS)

MAX_ITER = 20                      # sim6 MAX_ITER: growth iterations per biomorph
Pos      = Tuple[int, int]

# 16-bit popcount table (sim6 _POPCOUNT16) for the min-Hamming lookup.
_PC16 = bytes(bin(i).count("1") for i in range(1 << 16))


@dataclass
class Gene:
    """sim6's Gene, in sim6's own field order (N,S,E,W) - kept faithful so the
    ontogeny reads like genetic.c. Converted to substrates/lut's LutGene by
    `to_lut_genome` (which reorders to N,E,S,W)."""
    limit:    int = 255
    state_n:  int = 0
    state_s:  int = 0
    state_e:  int = 0
    state_w:  int = 0
    self_in:  int = 0
    self_out: int = 0


def _set_limit(g: Gene, rng: random.Random) -> None:
    """sim6 set_limit: a growth gene (self_in == 0) with <=2 live context sides
    gets a SHORT random activity window; a well-surrounded one never expires.
    This prunes the sparse growth front irregularly - a key source of ragged,
    non-diamond boundaries."""
    if g.self_in > 0:
        return
    surround = (g.state_n > 0) + (g.state_s > 0) + (g.state_e > 0) + (g.state_w > 0)
    if surround > 2:
        return
    g.limit = MAX_ITER - 2 - rng.randint(0, 9)


def table_create(sn, ss, se, sw, s_self, genome: List[Gene], iteration: int,  # noqa: E501
                 rng: random.Random) -> int:
    """sim6 table_create: the build-phase ontogeny. If the context is unseen,
    invent a gene for it, deciding its output by (usually) copying a same-context
    gene's output, else copying a neighbour's facing LUT, else a random LUT."""
    # Fully surrounded - no gene needed (sim6 line 296).
    if sn > 0 and ss > 0 and se > 0 and sw > 0:
        return s_self
    # All-zero context stays zero (no spontaneous generation).
    if sn == 0 and ss == 0 and se == 0 and sw == 0 and s_self == 0:
        return 0
    # Exact match already in the genome.
    for g in genome:
        if (g.state_n == sn and g.state_s == ss and g.state_e == se
                and g.state_w == sw and g.self_in == s_self):
            return g.self_out
    # A new gene for this context.
    gg = Gene(state_n=sn, state_s=ss, state_e=se, state_w=sw, self_in=s_self)
    genome.append(gg)
    # 70%: inherit an output from an existing gene with the SAME self context.
    if rng.randint(0, 99) > 30:
        cands = [g for g in genome[:-1] if g.self_in == s_self]
        if cands:
            gg.self_out = rng.choice(cands).self_out
            _set_limit(gg, rng)  # note: sim6 calls set_limit here too
            return gg.self_out
    # Otherwise decide the output directly.
    r = rng.randint(0, 99)
    if s_self == 0:                        # empty cell: usually becomes active
        _set_limit(gg, rng)
        if iteration > gg.limit or r > 80: gg.self_out = s_self       # stay dead
        elif r > 70: gg.self_out = sn & 0xFFFF                        # copy N
        elif r > 60: gg.self_out = ss & 0xFFFF                        # copy S
        elif r > 50: gg.self_out = se & 0xFFFF                        # copy E
        elif r > 40: gg.self_out = sw & 0xFFFF                        # copy W
        else:        gg.self_out = rng.randint(0, 0xFFFF)             # random
    else:                                  # active cell: usually persists
        if   r > 20: gg.self_out = s_self & 0xFFFF
        elif r > 16: gg.self_out = sn & 0xFFFF
        elif r > 12: gg.self_out = ss & 0xFFFF
        elif r > 8:  gg.self_out = se & 0xFFFF
        elif r > 4:  gg.self_out = sw & 0xFFFF
        else:        gg.self_out = rng.randint(0, 0xFFFF)
    return gg.self_out


def table_lookup(sn, ss, se, sw, s_self, genome: List[Gene], iteration: int) -> int:
    """sim6 table_lookup: min-Hamming match over the (already-built) genome,
    respecting per-gene limits and the empty-cell guard. Used when growing a
    morph from a COPIED genome (variation), like sim6's later biomorphs."""
    if sn == 0 and ss == 0 and se == 0 and sw == 0 and s_self == 0:
        return 0
    best, best_d = None, 1 << 30
    pc = _PC16
    for g in genome:
        if iteration > g.limit:
            continue
        d = (pc[(g.state_n ^ sn) & 0xFFFF] + pc[(g.state_s ^ ss) & 0xFFFF] +
             pc[(g.state_e ^ se) & 0xFFFF] + pc[(g.state_w ^ sw) & 0xFFFF] +
             pc[(g.self_in ^ s_self) & 0xFFFF])
        if d < best_d:
            best_d, best = d, g
    if best is None:
        return 0
    if s_self == 0 and best.self_in != 0:
        return 0
    return best.self_out


# -- morphogenesis ------------------------------------------------------------
# Grid cells are {pos: [sn, ss, se, sw]}; index 0=N,1=S,2=E,3=W - the same order
# as lut.py's (Ln, Ls, Le, Lw) tuple, and directions match lut.py (N=+y, S=-y,
# E=+x, W=-x) so a grown grid renders directly with draw_lut_net.

_N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _build_step(grid: Dict[Pos, List[int]], genome: List[Gene], iteration: int,
                table_fn, rng: random.Random) -> None:
    # Expand the frontier: every empty 4-neighbour of an existing cell (sim6
    # expands from ALL nodes, live or not, so growth advances one ring/step).
    for (x, y) in list(grid):
        for dx, dy in _N4:
            p = (x + dx, y + dy)
            if p not in grid:
                grid[p] = [0, 0, 0, 0]
    # Compute new states (rotate the context so the output direction is "front").
    saves: Dict[Pos, List[int]] = {}
    for (x, y), st in grid.items():
        sn = grid[(x, y + 1)][1] if (x, y + 1) in grid else 0   # N's south LUT
        ss = grid[(x, y - 1)][0] if (x, y - 1) in grid else 0   # S's north LUT
        se = grid[(x + 1, y)][3] if (x + 1, y) in grid else 0   # E's west LUT
        sw = grid[(x - 1, y)][2] if (x - 1, y) in grid else 0   # W's east LUT
        saves[(x, y)] = [
            table_fn(sn, ss, se, sw, st[0], genome, iteration, rng),   # N
            table_fn(ss, sn, sw, se, st[1], genome, iteration, rng),   # S
            table_fn(se, sw, ss, sn, st[2], genome, iteration, rng),   # E
            table_fn(sw, se, sn, ss, st[3], genome, iteration, rng),   # W
        ]
    for p, s in saves.items():             # parallel commit
        grid[p] = [v & 0xFFFF for v in s]


def _lookup_adapter(sn, ss, se, sw, s_self, genome, iteration, rng):
    return table_lookup(sn, ss, se, sw, s_self, genome, iteration)


def _live(st) -> bool:
    return st[0] or st[1] or st[2] or st[3]


def grow_ontogeny(seed: Optional[int] = None, genome: Optional[List[Gene]] = None,
                  ) -> Tuple[List[Gene], Dict[Pos, Tuple[int, int, int, int]]]:
    """Grow ONE biomorph, sim6-style. If `genome` is None, build a fresh genome
    on the fly with `table_create` (a brand-new organism); otherwise grow from
    the given genome with min-Hamming `table_lookup` (a variation on it, as sim6
    does for its later morphs). Returns (genome, grid) where grid holds only the
    LIVE cells as {pos: (Ln, Ls, Le, Lw)}."""
    rng = random.Random(seed)
    building = genome is None
    if building:
        genome = []
        table_fn = table_create
    else:
        table_fn = _lookup_adapter
    # Asymmetric single-cell seed: one node with a random NORTH state (sim6 550).
    grid: Dict[Pos, List[int]] = {(0, 0): [rng.randint(1, 0xFFFF), 0, 0, 0]}
    for it in range(MAX_ITER):
        _build_step(grid, genome, it, table_fn, rng)
    live = {p: tuple(st) for p, st in grid.items() if _live(st)}
    return genome, live


def _lutgene(g: Gene) -> LutGene:
    """sim6-order gene (N,S,E,W) -> substrates/lut LutGene (N,E,S,W). Limits dropped
    (substrates/lut bounds growth by telomere, not per-gene limits)."""
    return LutGene(ctx_n=g.state_n, ctx_e=g.state_e, ctx_s=g.state_s,
                   ctx_w=g.state_w, self_in=g.self_in, self_out=g.self_out)


def _pack(lg: List[LutGene], n_chroms: int) -> Genome:
    """Partition an ontogeny gene list into exactly ``n_chroms`` chromosomes."""
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    base, extra = divmod(len(lg), n_chroms)
    chroms, start = [], 0
    for index in range(n_chroms):
        size = base + (1 if index < extra else 0)
        genes = lg[start:start + size]
        start += size
        chroms.append(Chromosome(
            genes=genes, split=(0 if size < 2 else size // 2),
            tag=0, telomere=MAX_ITER))
    return Genome(chromosomes=chroms, tag=0)


def to_lut_genome(genes: List[Gene], n_chroms: int = 1) -> Genome:
    """Convert an ontogeny gene list to a substrates/lut Genome (see _lutgene/_pack)."""
    return _pack([_lutgene(g) for g in genes], n_chroms)


def _cap_genes(lg: List[LutGene], cap: int, rng: random.Random) -> List[LutGene]:
    """Trim a gene list to `cap`, preserving growth structure (self_in == 0
    genes, the only kind that can bring a cell to life) but ALWAYS reserving a
    share of the cap for maintenance genes: growth rules stop applying once the
    telomere expires, and a capped genome left with zero maintainers grows
    hundreds of cells and then dies back to nothing at maturity (measured:
    921 cells @iter20 -> 2 @iter30 when growth genes alone filled the cap)."""
    if len(lg) <= cap:
        return lg
    grow  = [g for g in lg if g.self_in == 0]
    maint = [g for g in lg if g.self_in != 0]
    rng.shuffle(maint)
    n_maint = min(len(maint), max(cap // 4, cap - len(grow)))
    keep = grow[:cap - n_maint] + maint[:n_maint]
    rng.shuffle(keep)
    return keep[:cap]


def random_ontogeny_genome(n_chroms: int = 2, cap_genes: int = 350,
                           min_cells: int = 30, min_genes: int = 150,
                           max_tries: int = 60,
                           rng: Optional[random.Random] = None) -> Genome:
    """GA seed factory (Stage 1): grow a sim6-style biomorph and keep it only if
    it is DENSE (raw genome >= `min_genes` - density is what makes shapes rich;
    trivial few-gene biomorphs just fill a uniform diamond) and, capped to
    `cap_genes` (see _cap_genes) and regrown by substrates/lut's own grow_lut, makes at
    least `min_cells` live cells (sim6's `good < 10` retry). Falls back to the
    last dense attempt, or a random-lut genome if none was dense enough."""
    from .lut import grow_lut
    r = rng or random
    last = None
    for _ in range(max_tries):
        genes, _ = grow_ontogeny(seed=r.randint(1, (1 << 31) - 1))
        if len(genes) < min_genes:
            continue                                    # too sparse -> diamond
        last = _pack(_cap_genes([_lutgene(g) for g in genes], cap_genes, r), n_chroms)
        if len(grow_lut(last, seeds=[(0, 0)], grid_size=40, iters=MAX_ITER)) >= min_cells:
            return last
    if last is not None:
        return last
    from .genome import random_lut_genome
    return random_lut_genome(n_chroms)


# -- diagnostic gallery ---------------------------------------------------------

def render_gallery(out_path: str, base_seed: int = 1, n: int = 16) -> str:
    """Grow `n` biomorphs from consecutive seeds and render them in a grid,
    exactly as a sim6 shape-variety check. Saves a PNG to out_path (headless)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .viz import draw_lut_net

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4))
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(n):
        genes, grid = grow_ontogeny(seed=base_seed + i)
        live = len(grid)
        draw_lut_net(flat[i], grid, in_pos=[(0, 0)],
                     title="seed %d\n%d genes, %d cells" % (base_seed + i,
                                                            len(genes), live))
        print("morph %2d: seed=%d  genes=%3d  live_cells=%3d"
              % (i, base_seed + i, len(genes), live))
    for j in range(n, len(flat)):
        flat[j].set_visible(False)
    fig.suptitle("sim6-style ontogeny (table_create) - Stage 0 shape check",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100)
    print("saved gallery -> %s" % out_path)
    return out_path


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    render_gallery("ontogeny_gallery.png", base_seed=seed, n=n)
