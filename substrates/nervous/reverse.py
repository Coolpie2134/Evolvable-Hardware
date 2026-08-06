"""
substrates/nervous/reverse.py - best-effort INVERSE of grow_nervous (network -> genome).

Growth (ontogeny) is many-to-one and not analytically invertible, so a hand-
edited working grid usually has no genome behind it. This module reconstructs
one AS BEST IT CAN: a genome that develops back into the grid, with EXTRA cells
that do not disturb the reproduced states permitted (the caller's contract).

The growth engine is a *deterministic function of the local context*: a cell's
next state depends only on (L, R, D neighbours, self), looked up by minimum
Hamming distance (see nervous.py). Making the target a mere fixed point is not
enough - the growth TRAJECTORY from the bare seeds rarely lands on it. The
reconstruction runs in three stages:

  1. MONOTONE REPLAY (_collect_contexts). Replay a development with the target
     as an oracle - mirroring grow's parallel frontier + Hayflick telomere gate,
     but pinning every cell at its target state the instant it is born. Log only
     the contexts that actually occur: a GROWTH gene (self_in=0 -> s) at each
     birth context and a HOLD gene (self_in=s -> s) at each running context. The
     self bit separates the two kinds, so each cell's genes win its own lookups
     at Hamming distance 0. This is exact when the ideal trajectory is realised.

  1b. MULTI-STEP REPLAY (_collect_multistep_rules). When identical birth
      contexts request different final states, birth a shared temporary state.
      As the frontier grows, differing neighbourhoods can then differentiate
      those cells using the existing 5-bit state vocabulary.

  2. BOUNDARY SEAL (_boundary_seals). The genes above are position-blind, so
     they also fire at frontier positions OUTSIDE the target. Pin a suppressor
     (self_in=0 -> 0) at the mature target's exterior contexts to stop growth
     spilling over - the dominant source of extra cells.

  3. CLOSED-LOOP REPAIR. Actually grow both candidate genomes and compare to the
     target. Real growth diverges from the ideal replay (extras perturb
     neighbourhoods, cascading), so pin an exact PATCH gene at the OBSERVED grown
     context of every wrong/missing cell - forcing it correct even amid the
     divergence a forward replay cannot foresee. Patches are prepended (they win
     the min-Hamming tie at distance 0); repair each trajectory independently,
     then keep the better verified genome.

EXTRA cells that do not disturb the reproduced states are permitted (the caller's
contract). Two births that present an identical context but demand different
final states are a collision in this MONOTONE direct-to-final replay. That means
this reconstruction cannot satisfy both; it does not prove no genome can, because
another developmental trajectory may use intermediate states to separate them.
The collision and the verified misses are reported rather than hidden.

Entry point: grid_to_genome_nervous(grid, seeds) -> (genome, report).
"""
from __future__ import annotations
from collections import deque

from .genome import (HexGene, Chromosome, Genome, MAX_TELOMERE,
                     germline_telomere)
from .hexgrid import hex_dirs, hex_frontier_cells
from .nervous import grow_nervous, SEED_STATE


_DIRS = ('L', 'R', 'D')


def _ctx(grid, x, y):
    """The (L, R, D) neighbour context of cell (x, y) in `grid` (0 = empty)."""
    nb = hex_dirs(x, y)
    return (grid.get(nb['L'], 0) & 0x1F,
            grid.get(nb['R'], 0) & 0x1F,
            grid.get(nb['D'], 0) & 0x1F)


def _collect_contexts(grid, seeds, L):
    """Replay a MONOTONE development with the target grid as an oracle and record
    exactly the contexts that occur - nothing invented.

    This mirrors grow_nervous's own step (parallel frontier division, Hayflick
    telomere gating) but pins every cell at its TARGET state the instant it is
    born, so each cell is born once and then only held. At each step we log:

      * the birth context of every new target cell  -> a growth gene, and
      * the current context of every live cell        -> a hold gene,

    which are precisely the (context -> state) pairs a genome must satisfy to
    reproduce this trajectory. Because only real contexts are logged, the only
    conflicts reported are genuine substrate ambiguities (one context, two
    demanded states). Returns (growth, maint, conflicts, reached)."""
    growth, maint, conflicts = {}, {}, {}
    organism = {s: SEED_STATE for s in seeds}   # grow pins seeds to SEED_STATE
    tel = {s: L for s in seeds}
    for _ in range(3 * L + 6):
        # log the hold context every live cell currently presents
        for (x, y), s in organism.items():
            cl, cr, cd = _ctx(organism, x, y)
            maint[(cl, cr, cd, s)] = s
        # parallel division into empty target frontier cells (Hayflick-gated)
        frontier = {}
        for (x, y) in organism:
            for nb in hex_frontier_cells(x, y):
                if nb in grid and nb not in organism and nb not in frontier:
                    frontier[nb] = None
        births = {}
        for (x, y) in frontier:
            nb = hex_dirs(x, y)
            parent_tel = max((tel.get(nb[d], 0) for d in _DIRS
                              if nb[d] in organism), default=0)
            if parent_tel <= 0:
                continue
            cl, cr, cd = _ctx(organism, x, y)
            s = grid[(x, y)]
            if (cl, cr, cd) in growth and growth[(cl, cr, cd)] != s:
                conflicts.setdefault((cl, cr, cd), {growth[(cl, cr, cd)]}).add(s)
            else:
                growth.setdefault((cl, cr, cd), s)
            births[(x, y)] = (s, parent_tel - 1)
        if not births:
            break
        for pos, (s, t) in births.items():
            organism[pos] = s
            tel[pos] = t
        for s in seeds:                       # seeds stay pinned to SEED_STATE
            organism[s] = SEED_STATE
            tel.setdefault(s, L)
    return growth, maint, conflicts, set(organism)


def _collect_multistep_rules(grid, seeds, L):
    """Synthesize a deterministic trajectory that may use temporary states.

    When several cells have the same complete lookup context but need different
    final states, direct replay cannot birth them differently.  Give the group a
    shared nonzero intermediate state instead.  On later steps their growing
    neighbourhoods may differ, producing distinct contexts that can transition
    to the requested final states without adding coordinates or widening the
    substrate.  Previously recorded rules are immutable: asking one timeless
    context to change its output later would only fake a clock.

    Returns ``(rules, seals, reached, collisions, temporary_states)`` where each
    rule maps ``(L,R,D,self) -> next_state`` and seals are empty-cell contexts
    observed outside the target during the synthesized trajectory.
    """
    seedset = set(seeds)
    organism = {s: SEED_STATE for s in seeds}
    tel = {s: L for s in seeds}
    rules = {}
    seals = set()
    collisions = {}
    reached = set(organism)

    target_states = {state for state in grid.values() if state}
    token_pool = ([s for s in range(1, 32)
                   if s not in target_states and s != SEED_STATE]
                  + [s for s in range(2, 32)])
    token_for = {}

    for _ in range(3 * L + 6):
        # Existing target cells update, while empty target-frontier cells may be
        # born. Seeds are pinned after lookup and need no rule of their own.
        groups = {}
        next_tel = {}
        for (x, y), state in organism.items():
            if (x, y) in seedset:
                continue
            cl, cr, cd = _ctx(organism, x, y)
            key = (cl, cr, cd, state)
            groups.setdefault(key, []).append(((x, y), grid[(x, y)]))
            next_tel[(x, y)] = tel.get((x, y), 0)

        frontier = set()
        for pos in organism:
            frontier.update(nb for nb in hex_frontier_cells(*pos)
                            if nb not in organism)
        for x, y in frontier:
            nb = hex_dirs(x, y)
            parent_tel = max((tel.get(nb[d], 0) for d in _DIRS
                              if nb[d] in organism), default=0)
            if parent_tel <= 0:
                continue
            cl, cr, cd = _ctx(organism, x, y)
            if (x, y) in grid:
                key = (cl, cr, cd, 0)
                groups.setdefault(key, []).append(((x, y), grid[(x, y)]))
                next_tel[(x, y)] = parent_tel - 1
            else:
                seals.add((cl, cr, cd))

        outputs = {}
        for key, requests in groups.items():
            desired = {state for _pos, state in requests}
            if key in rules:
                out = rules[key]
            elif len(desired) == 1:
                out = next(iter(desired))
                rules[key] = out
            else:
                collisions.setdefault(key, set()).update(desired)
                token_key = (key, tuple(sorted(desired)))
                unavailable = desired | {0}
                out = token_for.get(token_key)
                if out is None:
                    out = next((state for state in token_pool
                                if state not in unavailable), SEED_STATE)
                    token_for[token_key] = out
                rules[key] = out
            for pos, _desired in requests:
                outputs[pos] = out

        nxt = {pos: state for pos, state in outputs.items() if state}
        nxt_tel = {pos: next_tel.get(pos, 0) for pos in nxt}
        for seed in seeds:
            nxt[seed] = SEED_STATE
            nxt_tel[seed] = L
        reached.update(nxt)
        if nxt == organism:
            break
        organism, tel = nxt, nxt_tel

    wanted_births = {key[:3] for key, out in rules.items()
                     if key[3] == 0 and out}
    seals -= wanted_births
    return rules, seals, reached, collisions, set(token_for.values())


def _hop_distances(cells, seeds):
    """Honeycomb hop distance from the seeds to each cell, travelling only
    through `cells` (undirected hex adjacency). Cells with no path are absent."""
    cellset = set(cells)
    dist = {s: 0 for s in seeds if s in cellset}
    dq = deque(dist)
    while dq:
        c = dq.popleft()
        for nb in hex_frontier_cells(*c):
            if nb in cellset and nb not in dist:
                dist[nb] = dist[c] + 1
                dq.append(nb)
    return dist


def _boundary_seals(grid, seeds, wanted_births):
    """Contexts of the empty cells just OUTSIDE the mature target. Adding a
    suppressor (self_in=0 -> 0) at each stops the promiscuous, position-blind
    growth genes from spilling past the target boundary - the dominant source of
    extra cells. A boundary context that must ALSO birth a real cell somewhere is
    skipped (it can't be both grown and sealed); the closed-loop repair, whose
    growth patches are prepended and win at Hamming distance 0, overrides a seal
    that turns out to block a wanted cell."""
    seedset = set(seeds)
    seals = set()
    for (x, y) in grid:
        for nb in hex_frontier_cells(x, y):
            if nb in grid or nb in seedset:
                continue
            cl, cr, cd = _ctx(grid, *nb)
            if (cl, cr, cd) not in wanted_births:
                seals.add((cl, cr, cd))
    return seals


def repair_genome_nervous(source_genome, target_grid, seeds, repair_rounds=24):
    """Reconcile a retained genome with a hand-edited nervous-grid phenotype.

    The original chromosomes remain untouched and retain their developmental
    trajectory.  A leading repair chromosome supplies exact observed-context
    rules only where the forward-grown phenotype differs from ``target_grid``.
    Candidate selection is deliberately conservative: preserving cells that
    were unchanged by the user outranks satisfying edits, so reconciliation can
    never silently trade away working circuitry to improve the headline count.
    """
    target = {tuple(pos): int(state) & 0x1F
              for pos, state in target_grid.items()}
    seeds = tuple(tuple(seed) for seed in seeds)
    baseline = grow_nervous(source_genome, seeds=seeds)
    unchanged = {pos for pos, state in target.items()
                 if baseline.get(pos) == state}
    edited = {pos for pos in set(baseline) | set(target)
              if baseline.get(pos) != target.get(pos)}
    L = germline_telomere(source_genome)
    patch_map = {}
    collisions = {}

    def candidate():
        if not patch_map:
            genome = source_genome
        else:
            repair_chrom = Chromosome(
                genes=list(patch_map.values()), split=0, tag=-1, telomere=L)
            genome = Genome(chromosomes=[repair_chrom]
                            + list(source_genome.chromosomes),
                            tag=source_genome.tag)
        grown = grow_nervous(genome, seeds=seeds)
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
            patch_map[key] = HexGene(*key, output)
            return True
        return False

    current = candidate()
    best = current
    best_patch_map = dict(patch_map)
    stagnation = 0
    for _ in range(repair_rounds):
        _key, _genome, grown, _matched, extras = current
        added = False
        # Desired live cells take precedence over suppressors if one position-
        # blind context is shared by an addition and a deletion.
        for pos, desired in target.items():
            actual = grown.get(pos)
            if actual == desired:
                continue
            cl, cr, cd = _ctx(grown, *pos)
            self_in = 0 if actual is None else (actual & 0x1F)
            added |= request((cl, cr, cd, self_in), desired)
        for pos in extras:
            actual = grown[pos] & 0x1F
            cl, cr, cd = _ctx(grown, *pos)
            added |= request((cl, cr, cd, actual), 0)  # remove this live extra
            added |= request((cl, cr, cd, 0), 0)       # suppress its rebirth
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

    # Rebuild the saved best if later exploratory patches were rejected.
    patch_map = best_patch_map
    best = candidate()
    _key, genome, grown, matched, extras = best
    unchanged_ok = sum(grown.get(pos) == target[pos] for pos in unchanged)
    edited_ok = sum(grown.get(pos) == target.get(pos) for pos in edited)
    report = {
        'backend': 'nervous', 'strategy': 'delta-repair',
        'baseline_matched': sum(baseline.get(pos) == state
                                for pos, state in target.items()),
        'matched': matched, 'target': len(target), 'extra': len(extras),
        'missing': len(target) - matched,
        'unchanged_preserved': unchanged_ok, 'unchanged': len(unchanged),
        'edits_reproduced': edited_ok, 'edits': len(edited),
        'added_genes': len(patch_map),
        'conflicts': [(key, sorted(values))
                      for key, values in collisions.items()],
        'seed_mismatch': [seed for seed in seeds
                          if target.get(seed, SEED_STATE) != SEED_STATE],
        'unreachable': [], 'radius': 0, 'telomere': L,
        'exact': grown == target, 'grown': grown,
    }
    return genome, report


def grid_to_genome_nervous(grid, seeds, telomere_cap=MAX_TELOMERE,
                           repair_rounds=24, use_multistep=True):
    """Reconstruct a genome whose growth reproduces `grid` from `seeds`.

    Returns (genome, report). `report` is a dict:
        matched/target      cells reproduced exactly / cells in the target
        extra/missing        grown cells not in the target / vice-versa
        conflicts            contexts where this monotone direct-to-final replay
                             asks one lookup to produce different states
        seed_mismatch        seed cells whose target state != the pinned seed
        unreachable          target cells with no growth path from a seed
        radius/telomere      growth radius needed / telomere used
        exact                True iff the grown grid equals the target exactly
        grown                the verifying grown grid (dict)
    """
    grid = {tuple(p): (s & 0x1F) for p, s in grid.items()}
    seeds = [tuple(s) for s in seeds]
    seedset = set(seeds)
    report = {'backend': 'nervous'}

    if not seeds:
        report.update(matched=0, target=len(grid), extra=0, missing=len(grid),
                      conflicts=[], seed_mismatch=[], unreachable=list(grid),
                      radius=0, telomere=1, exact=False, grown={},
                      note='no seeds/inputs - mark at least one input (the growth '
                           'seed) before reversing.')
        return Genome(chromosomes=[Chromosome(genes=[HexGene()], telomere=1)]), report

    # seeds are pinned to SEED_STATE every growth step, so a target state that
    # differs there can never be reproduced (an inherent substrate constraint).
    seed_mismatch = [s for s in seeds if grid.get(s, SEED_STATE) != SEED_STATE]

    # growth reach: telomere is a Hayflick limit, so a cell at hop k from a seed
    # needs germline length >= k. Size the telomere to the target's own radius.
    dist = _hop_distances(set(grid) | seedset, seeds)
    topological_unreachable = {p for p in grid
                               if p not in dist and p not in seedset}
    radius = max((d for p, d in dist.items() if p in grid or p in seedset),
                 default=0)
    L = max(1, min(int(telomere_cap), radius))

    # replay a monotone development to log the exact (context -> state) genes,
    # then any target cell the replay never reached is unreachable this way
    growth, maint, conflicts, reached = _collect_contexts(grid, seeds, L)
    direct_reached = reached
    direct_wanted = set(growth)                   # contexts that must birth a live cell

    # base genome: growth genes birth each cell, boundary seals stop the spill,
    # maintenance genes hold the fixed point. Patches (below) are PREPENDED so an
    # exact (Hamming-0) patch always wins the min-Hamming lookup.
    direct_base = (
        [HexGene(cl, cr, cd, 0, s) for (cl, cr, cd), s in growth.items()]
        + [HexGene(cl, cr, cd, 0, 0)
           for (cl, cr, cd) in _boundary_seals(grid, seeds, direct_wanted)]
        + [HexGene(cl, cr, cd, s, s) for (cl, cr, cd, s) in maint])

    # Alternative synthesis: use shared temporary states for ambiguous births,
    # then differentiate after their neighbourhoods diverge. It uses the same
    # 5-bit state vocabulary and the same forward growth engine.
    (multi_rules, trajectory_seals, multi_reached, multi_collisions,
     temporary_states) = _collect_multistep_rules(grid, seeds, L)
    multi_wanted = {key[:3] for key, out in multi_rules.items()
                    if key[3] == 0 and out}
    multi_seals = (trajectory_seals
                   | _boundary_seals(grid, seeds, multi_wanted)) - multi_wanted
    multi_base = ([HexGene(cl, cr, cd, si, out)
                   for (cl, cr, cd, si), out in multi_rules.items()]
                  + [HexGene(cl, cr, cd, 0, 0)
                     for cl, cr, cd in multi_seals])

    def evaluate(base, patches=()):
        genome = Genome(chromosomes=[Chromosome(genes=list(patches) + base, split=0,
                                                tag=1, telomere=L)], tag=1)
        grown = grow_nervous(genome, seeds=tuple(seeds))
        matched = sum(1 for p, s in grid.items() if grown.get(p) == s)
        extras = [p for p in grown if p not in grid]
        return genome, grown, matched, extras

    direct_eval = evaluate(direct_base)
    multi_eval = evaluate(multi_base)

    # CLOSED-LOOP repair each trajectory independently. Choosing a trajectory
    # before repair can regress when the initially weaker base responds better
    # to exact observed-context patches.
    def repair(base, wanted, initial):
        genome, grown, matched, extras = initial
        patches = []
        best = (matched, -len(extras), genome, grown, extras, list(patches))
        stagn = 0
        for _ in range(repair_rounds):
            seen = {(g.ctx_l, g.ctx_r, g.ctx_d, g.self_in) for g in patches}
            add = []
            for p, state in grid.items():
                actual = grown.get(p)
                if actual == state:
                    continue
                cl, cr, cd = _ctx(grown, *p)
                self_in = 0 if actual is None else (actual & 0x1F)
                key = (cl, cr, cd, self_in)
                if key not in seen:
                    add.append(HexGene(cl, cr, cd, self_in, state))
                    seen.add(key)
            for p in extras:
                cl, cr, cd = _ctx(grown, *p)
                key = (cl, cr, cd, 0)
                if (cl, cr, cd) not in wanted and key not in seen:
                    add.append(HexGene(cl, cr, cd, 0, 0))
                    seen.add(key)
            if not add:
                break
            patches += add
            genome, grown, matched, extras = evaluate(base, patches)
            if (matched, -len(extras)) > (best[0], best[1]):
                best = (matched, -len(extras), genome, grown, extras, list(patches))
                stagn = 0
            else:
                stagn += 1
                if stagn >= 3:
                    break
        return best

    direct_best = repair(direct_base, direct_wanted, direct_eval)
    multi_best = (repair(multi_base, multi_wanted, multi_eval)
                  if use_multistep else None)

    def repaired_key(candidate):
        matched, neg_extras, genome = candidate[:3]
        genes = sum(len(chrom.genes) for chrom in genome.chromosomes)
        return matched, neg_extras, -genes

    if multi_best is not None and repaired_key(multi_best) > repaired_key(direct_best):
        strategy, best, reached = 'multi-step', multi_best, multi_reached
    else:
        strategy, best, reached = 'direct', direct_best, direct_reached
    unreachable = sorted(topological_unreachable
                         | {p for p in grid if p not in reached})
    matched, _neg, genome, grown, extras, _patches = best
    missing = [p for p in grid if grown.get(p) != grid[p]]
    report.update(
        matched=matched, target=len(grid), extra=len(extras), missing=len(missing),
        conflicts=[(k, sorted(v)) for k, v in conflicts.items()],
        seed_mismatch=seed_mismatch, unreachable=unreachable,
        radius=radius, telomere=L, strategy=strategy,
        intermediate_states=sorted(temporary_states) if strategy == 'multi-step' else [],
        multistep_collisions=len(multi_collisions),
        exact=(matched == len(grid) and not extras), grown=grown)
    return genome, report
