"""
lut_evo/reverse.py — best-effort INVERSE of grow_lut (LUT network -> genome).

The mirror of nv_evo/reverse.py for Architecture 2. Growth develops four 16-bit
LUTs per cell by the same associative-memory rule as the nervous net, only the
context is five 16-bit LUTs (the four neighbours' facing tables + the cell's own
table for that direction), looked up with the context ROTATED so the output
direction is "front" (see lut.py). One gene therefore serves all four
directions.

We reconstruct a genome by reading, for every live cell and every direction, the
exact rotated context the growth engine would present and pinning a gene to it:

  * a growth gene   (self_in = 0, self_out = L)      births a direction; and
  * a hold gene     (self_in = L, self_out = L)      maintains it,

so the target grid is a fixed point of the lookup. Directions that are dead in
the target (LUT = 0) get a self_in = 0 -> 0 suppressor so a stray context can't
switch them on. Contexts that demand two different outputs are collisions in
this direct-to-final replay and are reported. They are not proof that no genome
exists: a more complex development may pass through intermediate states first.

Entry point: grid_to_genome_lut(grid, seeds, grid_size, iters) -> (genome, report).
"""
from __future__ import annotations
from collections import deque

from .genome import LutGene, Chromosome, Genome, MAX_TELOMERE
from .lut import grow_lut, SEED_STATE

_Z = (0, 0, 0, 0)
_N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _facing(grid, x, y):
    """The four incoming LUTs at (x, y): (n, s, e, w) = the table each neighbour
    aims at this cell (its S, N, W, E table respectively; 0 where empty)."""
    n = grid.get((x, y + 1), _Z)[1]
    s = grid.get((x, y - 1), _Z)[0]
    e = grid.get((x + 1, y), _Z)[3]
    w = grid.get((x - 1, y), _Z)[2]
    return n, s, e, w


def _dir_contexts(grid, x, y, st):
    """Yield (self_lut, (ctx_n, ctx_s, ctx_e, ctx_w)) for each of the four
    directions, in the exact rotated framing grow_lut's look() presents — so a
    gene pinned here wins that direction's lookup at Hamming distance 0."""
    n, s, e, w = _facing(grid, x, y)
    #        self     (ctx_n, ctx_s, ctx_e, ctx_w)  <- look(f, b, r, l, self)
    yield st[0], (n, s, e, w)        # N: look(n, s, e, w, Ln)
    yield st[1], (s, n, w, e)        # S: look(s, n, w, e, Ls)
    yield st[2], (e, w, s, n)        # E: look(e, w, s, n, Le)
    yield st[3], (w, e, n, s)        # W: look(w, e, n, s, Lw)


def _hop_distances(cells, seeds):
    cellset = set(cells)
    dist = {s: 0 for s in seeds if s in cellset}
    dq = deque(dist)
    while dq:
        c = dq.popleft()
        for dx, dy in _N4:
            nb = (c[0] + dx, c[1] + dy)
            if nb in cellset and nb not in dist:
                dist[nb] = dist[c] + 1
                dq.append(nb)
    return dist


def _record(growth, conflicts, ctx, out):
    if ctx in growth and growth[ctx] != out:
        conflicts.setdefault(ctx, {growth[ctx]}).add(out)
    else:
        growth.setdefault(ctx, out)


def _collect_contexts_lut(grid, seeds, L):
    """Replay a MONOTONE development with the target grid as an oracle (the LUT
    analogue of nv_evo.reverse._collect_contexts). Mirrors grow_lut's step —
    parallel frontier growth, per-direction rotated lookup, telomere-gated
    births — but pins every cell at its TARGET four-LUT state the moment it is
    born. Logs the birth context of each direction as a growth gene (self_in=0 ->
    target LUT, 0 for a dead direction so it can't switch on) and each live
    direction's running context as a hold gene. Returns
    (growth, maint, conflicts, reached)."""
    growth, maint, conflicts = {}, {}, {}
    organism = {s: SEED_STATE for s in seeds}
    for step in range(3 * L + 8):
        # log the contexts every live cell currently presents
        for (x, y), st in organism.items():
            for self_lut, ctx in _dir_contexts(organism, x, y, st):
                if self_lut:
                    maint[(ctx, self_lut)] = self_lut
                else:                      # keep a live cell's dead direction dead
                    _record(growth, conflicts, ctx, 0)
        if step >= L:                      # telomere expired: growth rules stop
            break
        # parallel division into empty target frontier cells
        frontier = set()
        for (x, y) in organism:
            for dx, dy in _N4:
                nb = (x + dx, y + dy)
                if nb in grid and nb not in organism:
                    frontier.add(nb)
        births = {}
        for (x, y) in frontier:
            st = grid[(x, y)]
            for self_lut, ctx in _dir_contexts(organism, x, y, st):
                _record(growth, conflicts, ctx, self_lut)   # 0 for a dead dir
            births[(x, y)] = st
        if not births:
            break
        organism.update(births)
        for s in seeds:
            organism[s] = SEED_STATE
    return growth, maint, conflicts, set(organism)


def repair_genome_lut(source_genome, target_grid, seeds, grid_size=7, iters=30,
                      repair_rounds=12):
    """Conservatively reconcile a retained LUT genome with an edited grid."""
    target = {tuple(pos): tuple(int(v) & 0xFFFF for v in state)
              for pos, state in target_grid.items()}
    seeds = tuple(tuple(seed) for seed in seeds)
    baseline = grow_lut(source_genome, seeds=seeds,
                        grid_size=grid_size, iters=iters)
    unchanged = {pos for pos, state in target.items()
                 if baseline.get(pos) == state}
    edited = {pos for pos in set(baseline) | set(target)
              if baseline.get(pos) != target.get(pos)}
    L = max((getattr(chrom, 'telomere', 1)
             for chrom in source_genome.chromosomes), default=1)
    patch_map = {}
    collisions = {}

    def gene_for(key, output):
        cn, cs, ce, cw, self_in = key
        return LutGene(ctx_n=cn, ctx_s=cs, ctx_e=ce, ctx_w=cw,
                       self_in=self_in, self_out=output)

    def candidate():
        if not patch_map:
            genome = source_genome
        else:
            repair_chrom = Chromosome(
                genes=list(patch_map.values()), split=0, tag=-1, telomere=L)
            genome = Genome(chromosomes=[repair_chrom]
                            + list(source_genome.chromosomes),
                            tag=source_genome.tag)
        grown = grow_lut(genome, seeds=seeds,
                         grid_size=grid_size, iters=iters)
        unchanged_ok = sum(grown.get(pos) == target[pos] for pos in unchanged)
        edited_ok = sum(grown.get(pos) == target.get(pos) for pos in edited)
        matched = sum(grown.get(pos) == state for pos, state in target.items())
        extras = [pos for pos in grown if pos not in target]
        key = (unchanged_ok, edited_ok, -len(extras), matched, -len(patch_map))
        return key, genome, grown, matched, extras

    def request(key, output):
        old = patch_map.get(key)
        if old is not None and old.self_out != output:
            collisions.setdefault(key, {old.self_out}).add(output)
            return False
        if old is None:
            patch_map[key] = gene_for(key, output)
            return True
        return False

    current = candidate()
    best = current
    best_patch_map = dict(patch_map)
    stagnation = 0
    for _ in range(repair_rounds):
        _key, _genome, grown, _matched, extras = current
        added = False
        for pos, desired_state in target.items():
            actual_state = grown.get(pos, _Z)
            if grown.get(pos) == desired_state:
                continue
            for (self_lut, ctx), desired_lut in zip(
                    _dir_contexts(grown, pos[0], pos[1], actual_state),
                    desired_state):
                added |= request(ctx + (self_lut,), desired_lut)
        for pos in extras:
            actual_state = grown[pos]
            for self_lut, ctx in _dir_contexts(
                    grown, pos[0], pos[1], actual_state):
                if self_lut:
                    added |= request(ctx + (self_lut,), 0)
                    added |= request(ctx + (0,), 0)
        if not added:
            break
        current = candidate()
        if current[0] > best[0]:
            best = current
            best_patch_map = dict(patch_map)
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 4:
                break

    patch_map = best_patch_map
    _key, genome, grown, matched, extras = candidate()
    report = {
        'backend': 'lut', 'strategy': 'delta-repair',
        'baseline_matched': sum(baseline.get(pos) == state
                                for pos, state in target.items()),
        'matched': matched, 'target': len(target), 'extra': len(extras),
        'missing': len(target) - matched,
        'unchanged_preserved': sum(grown.get(pos) == target[pos]
                                   for pos in unchanged),
        'unchanged': len(unchanged),
        'edits_reproduced': sum(grown.get(pos) == target.get(pos)
                                for pos in edited),
        'edits': len(edited), 'added_genes': len(patch_map),
        'conflicts': [(key, ['%04X' % value for value in sorted(values)])
                      for key, values in collisions.items()],
        'seed_mismatch': [seed for seed in seeds
                          if target.get(seed, SEED_STATE) != SEED_STATE],
        'unreachable': [], 'radius': 0, 'telomere': L,
        'exact': grown == target, 'grown': grown,
    }
    return genome, report


def grid_to_genome_lut(grid, seeds, grid_size=7, iters=30,
                       telomere_cap=MAX_TELOMERE, repair_rounds=6):
    """Reconstruct a genome whose growth reproduces the LUT `grid` from `seeds`.
    Returns (genome, report) with the same report shape as the nervous inverse."""
    grid = {tuple(p): tuple(int(v) & 0xFFFF for v in st) for p, st in grid.items()}
    seeds = [tuple(s) for s in seeds]
    seedset = set(seeds)
    report = {'backend': 'lut'}

    if not seeds:
        report.update(matched=0, target=len(grid), extra=0, missing=len(grid),
                      conflicts=[], seed_mismatch=[], unreachable=list(grid),
                      radius=0, telomere=1, exact=False, grown={},
                      note='no seeds/inputs — mark at least one input (the growth '
                           'seed) before reversing.')
        return Genome(chromosomes=[Chromosome(genes=[LutGene()], telomere=1)]), report

    seed_mismatch = [s for s in seeds if grid.get(s, SEED_STATE) != SEED_STATE]

    dist = _hop_distances(set(grid) | seedset, seeds)
    unreachable = [p for p in grid if p not in dist and p not in seedset]
    radius = max((d for p, d in dist.items() if p in grid or p in seedset),
                 default=0)
    # Hop k is born at iteration k-1. Growth genes remain active while
    # iteration < telomere, so telomere k is sufficient (not k+1).
    L = max(1, min(int(telomere_cap), radius))

    # replay a monotone development to log the exact (context -> LUT) genes
    growth, maint, conflicts, reached = _collect_contexts_lut(grid, seeds, L)
    unreachable = sorted(set(unreachable) | {p for p in grid if p not in reached})

    def build(suppressors):
        # gene fields are (ctx_n, ctx_e, ctx_s, ctx_w); context tuples are
        # (ctx_n, ctx_s, ctx_e, ctx_w) — keyword args keep the two straight.
        genes = [LutGene(ctx_n=cn, ctx_s=cs, ctx_e=ce, ctx_w=cw,
                         self_in=0, self_out=out)
                 for (cn, cs, ce, cw), out in growth.items()]
        genes += [LutGene(ctx_n=cn, ctx_s=cs, ctx_e=ce, ctx_w=cw,
                          self_in=0, self_out=0)
                  for (cn, cs, ce, cw) in suppressors]
        genes += [LutGene(ctx_n=cn, ctx_s=cs, ctx_e=ce, ctx_w=cw,
                          self_in=self_lut, self_out=self_lut)
                  for ((cn, cs, ce, cw), self_lut) in maint if self_lut]
        return Genome(chromosomes=[Chromosome(genes=genes, split=0, tag=1,
                                              telomere=L)], tag=1)

    def evaluate(genome):
        # Verify with exactly the safety cap the Designer's Grow button uses;
        # silently extending it could report a different phenotype.
        grown = grow_lut(genome, seeds=tuple(seeds), grid_size=grid_size,
                         iters=iters)
        matched = sum(1 for p, st in grid.items() if grown.get(p) == st)
        extras = [p for p in grown if p not in grid]
        return grown, matched, extras

    wanted = set(growth)
    genome = build(())
    grown, matched, extras = evaluate(genome)
    best = (matched, -len(extras), genome, grown, extras)
    suppressors = set()
    for _ in range(repair_rounds):
        # kill each extra cell by suppressing every live direction it grew,
        # unless that context is a wanted birth elsewhere
        new = set()
        for p in extras:
            for self_lut, ctx in _dir_contexts(grown, p[0], p[1], grown[p]):
                if self_lut:
                    new.add(ctx)
        new -= wanted | suppressors
        if not new:
            break
        suppressors |= new
        genome = build(suppressors)
        grown, matched, extras = evaluate(genome)
        if (matched, -len(extras)) >= (best[0], best[1]):
            best = (matched, -len(extras), genome, grown, extras)
        else:
            genome, grown, extras = best[2], best[3], best[4]
            break

    matched, _neg, genome, grown, extras = best
    missing = [p for p in grid if grown.get(p) != grid[p]]
    report.update(
        matched=matched, target=len(grid), extra=len(extras), missing=len(missing),
        conflicts=[(k, ['%04X' % s for s in sorted(v)]) for k, v in conflicts.items()],
        seed_mismatch=seed_mismatch, unreachable=unreachable,
        radius=radius, telomere=L,
        exact=(matched == len(grid) and not extras), grown=grown)
    return genome, report
