"""
nv_evo/reverse.py — best-effort INVERSE of grow_nervous (network -> genome).

Growth (ontogeny) is many-to-one and not analytically invertible, so a hand-
edited working grid usually has no genome behind it. This module reconstructs
one AS BEST IT CAN: a genome that develops back into the grid, with EXTRA cells
that do not disturb the reproduced states permitted (the caller's contract).

The growth engine is a *deterministic function of the local context*: a cell's
next state depends only on (L, R, D neighbours, self), looked up by minimum
Hamming distance (see nervous.py). Making the target a mere fixed point is not
enough — the growth TRAJECTORY from the bare seeds rarely lands on it. The
reconstruction runs in three stages:

  1. MONOTONE REPLAY (_collect_contexts). Replay a development with the target
     as an oracle — mirroring grow's parallel frontier + Hayflick telomere gate,
     but pinning every cell at its target state the instant it is born. Log only
     the contexts that actually occur: a GROWTH gene (self_in=0 -> s) at each
     birth context and a HOLD gene (self_in=s -> s) at each running context. The
     self bit separates the two kinds, so each cell's genes win its own lookups
     at Hamming distance 0. This is exact when the ideal trajectory is realised.

  1b. MULTI-STEP REPLAY (_collect_multistep_rules). When identical circuit
      contexts request different final selectors, emit a shared temporary
      4-bit selector. As the frontier grows, differing neighbourhoods can then
      differentiate those circuits. The three results are packed back into the
      tile phenotype after every synthesized step.

  2. BOUNDARY SEAL (_boundary_seals). The genes above are position-blind, so
     they also fire at frontier positions OUTSIDE the target. Pin a suppressor
     (self_in=0 -> 0) at the mature target's exterior contexts to stop growth
     spilling over — the dominant source of extra cells.

  3. CLOSED-LOOP REPAIR. Actually grow both candidate genomes and compare to the
     target. Real growth diverges from the ideal replay (extras perturb
     neighbourhoods, cascading), so pin an exact PATCH gene at the OBSERVED grown
     context of every wrong/missing cell — forcing it correct even amid the
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

from .genome import (HexGene, Chromosome, Genome, MAX_STATE, MAX_TELOMERE,
                     germline_telomere)
from .hexgrid import (DIRECTIONS, developmental_context,
                      hex_dirs, hex_frontier_cells, pack_tile_state,
                      unpack_tile_state)
from .nervous import grow_nervous, SEED_STATE


_DIRS = DIRECTIONS


def _ctx(grid, x, y, output_direction='D'):
    """Return one output circuit's rotated 4-bit developmental context.

    ``grid`` always contains packed 12-bit tile words.  Extracting a circuit
    through :func:`developmental_context` is essential here: masking the tile
    word itself to four bits would silently discard its R and D circuits.
    """
    return developmental_context(grid, (x, y), output_direction)


def _selector_items(tile_state):
    """Yield ``(direction, selector)`` pairs from one packed phenotype tile."""
    return zip(_DIRS, unpack_tile_state(tile_state))


def _record_rule(table, key, output, conflicts, conflict_key=None):
    """Keep the first deterministic rule and expose incompatible requests."""
    output = int(output)
    if not 0 <= output < MAX_STATE:
        raise ValueError('reverse rule output must be a 4-bit circuit state')
    old = table.get(key)
    if old is not None and old != output:
        ckey = key if conflict_key is None else conflict_key
        conflicts.setdefault(ckey, {old}).add(output)
        return False
    table.setdefault(key, output)
    return old is None


def _collect_contexts(grid, seeds, L):
    """Replay a MONOTONE development with the target grid as an oracle and record
    exactly the contexts that occur — nothing invented.

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
    seedset = set(seeds)
    organism = {s: SEED_STATE for s in seeds}   # grow pins seeds to SEED_STATE
    tel = {s: L for s in seeds}
    for _ in range(3 * L + 6):
        # Log one hold rule for each independently developed core circuit. Seeds
        # are pinned after every step, so their lookup results need no rule.
        for (x, y), tile_state in organism.items():
            if (x, y) in seedset:
                continue
            for direction, selector in _selector_items(tile_state):
                cl, cr, cd = _ctx(organism, x, y, direction)
                key = (cl, cr, cd, selector)
                _record_rule(maint, key, selector, conflicts, key)
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
            tile_state = grid[(x, y)]
            for direction, selector in _selector_items(tile_state):
                cl, cr, cd = _ctx(organism, x, y, direction)
                key = (cl, cr, cd)
                _record_rule(growth, key, selector, conflicts,
                             (cl, cr, cd, 0))
            births[(x, y)] = (tile_state, parent_tel - 1)
        if not births:
            break
        for pos, (s, t) in births.items():
            organism[pos] = s
            tel[pos] = t
        for s in seeds:                       # seeds stay pinned to SEED_STATE
            organism[s] = SEED_STATE
            tel.setdefault(s, L)
    # A zero-valued circuit in a live tile and an unborn circuit both have
    # self_in=0. Report clashes between their hold and birth requests too.
    for context, output in growth.items():
        key = (*context, 0)
        if key in maint and maint[key] != output:
            conflicts.setdefault(key, {output}).add(maint[key])
    return growth, maint, conflicts, set(organism)


def _collect_multistep_rules(grid, seeds, L):
    """Synthesize a deterministic circuit trajectory with temporary selectors.

    Requests are grouped by one circuit's ``(L,R,D,self)`` lookup key.  Every
    chosen result is a 4-bit selector; the L/R/D results for a position are then
    packed into its 12-bit phenotype tile before the next replay step.  A
    timeless rule is never rewritten merely because a later step wants another
    result.

    Returns ``(rules, seals, reached, collisions, temporary_states)`` where each
    rule maps a four-selector context to one selector and seals contain rotated
    empty-circuit contexts seen just outside the target trajectory.
    """
    seedset = set(seeds)
    organism = {s: SEED_STATE for s in seeds}
    tel = {s: L for s in seeds}
    rules = {}
    seals = set()
    collisions = {}
    reached = set(organism)

    target_selectors = {
        selector
        for tile_state in grid.values()
        for selector in unpack_tile_state(tile_state)
        if selector
    }
    seed_selectors = set(unpack_tile_state(SEED_STATE))
    token_pool = ([selector for selector in range(1, MAX_STATE)
                   if selector not in target_selectors
                   and selector not in seed_selectors]
                  + [selector for selector in range(1, MAX_STATE)
                     if selector not in seed_selectors])
    token_for = {}

    for _ in range(3 * L + 6):
        # Existing target cells update, while empty target-frontier cells may be
        # born. Seeds are pinned after lookup and need no rule of their own.
        groups = {}
        next_tel = {}
        for (x, y), tile_state in organism.items():
            if (x, y) in seedset:
                continue
            desired_tile = grid[(x, y)]
            desired_selectors = unpack_tile_state(desired_tile)
            for (direction, self_selector), desired in zip(
                    _selector_items(tile_state), desired_selectors):
                cl, cr, cd = _ctx(organism, x, y, direction)
                key = (cl, cr, cd, self_selector)
                groups.setdefault(key, []).append(
                    (((x, y), direction), desired))
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
            if (x, y) in grid:
                for direction, desired in _selector_items(grid[(x, y)]):
                    cl, cr, cd = _ctx(organism, x, y, direction)
                    key = (cl, cr, cd, 0)
                    groups.setdefault(key, []).append(
                        (((x, y), direction), desired))
                next_tel[(x, y)] = parent_tel - 1
            else:
                for direction in _DIRS:
                    seals.add(_ctx(organism, x, y, direction))

        circuit_outputs = {}
        for key, requests in groups.items():
            desired = {selector for _channel, selector in requests}
            if key in rules:
                out = rules[key]
                if any(selector != out for selector in desired):
                    collisions.setdefault(key, {out}).update(desired)
            elif len(desired) == 1:
                out = next(iter(desired))
                rules[key] = out
            else:
                collisions.setdefault(key, set()).update(desired)
                token_key = (key, tuple(sorted(desired)))
                unavailable = desired | {0}
                out = token_for.get(token_key)
                if out is None:
                    # If every non-seed selector is unavailable, fall back to
                    # the seed *circuit* selector (normally 1), never the packed
                    # 12-bit SEED_STATE tile word.
                    fallback = next(iter(seed_selectors), 1)
                    out = next((state for state in token_pool
                                if state not in unavailable), fallback)
                    token_for[token_key] = out
                rules[key] = out
            for channel, _desired in requests:
                circuit_outputs[channel] = out

        positions = {pos for pos, _direction in circuit_outputs}
        nxt = {}
        for pos in positions:
            tile_state = pack_tile_state(*(
                circuit_outputs.get((pos, direction), 0)
                for direction in _DIRS))
            if tile_state:
                nxt[pos] = tile_state
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
    growth genes from spilling past the target boundary — the dominant source of
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
            for direction in _DIRS:
                context = _ctx(grid, *nb, direction)
                if context not in wanted_births:
                    seals.add(context)
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
    target = {tuple(pos): int(state) for pos, state in target_grid.items()}
    for state in target.values():
        unpack_tile_state(state)  # validate, never truncate a corrupt tile word
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
            repair_genes = list(patch_map.values())
            repair_chrom = Chromosome(
                genes=repair_genes,
                split=(0 if len(repair_genes) < 2 else len(repair_genes) // 2),
                tag=-1, telomere=L)
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
        for pos, desired_tile in target.items():
            actual_tile = grown.get(pos, 0)
            if actual_tile == desired_tile or pos in seeds:
                continue
            actual_selectors = unpack_tile_state(actual_tile)
            desired_selectors = unpack_tile_state(desired_tile)
            for (direction, self_in), desired in zip(
                    zip(_DIRS, actual_selectors), desired_selectors):
                if self_in == desired:
                    continue
                cl, cr, cd = _ctx(grown, *pos, direction)
                added |= request((cl, cr, cd, self_in), desired)
        for pos in extras:
            for direction, actual in _selector_items(grown[pos]):
                cl, cr, cd = _ctx(grown, *pos, direction)
                if actual:
                    # Remove only this core circuit; all three must be zero
                    # before the packed phenotype tile disappears.
                    added |= request((cl, cr, cd, actual), 0)
                # The same direction is independently queried on rebirth.
                added |= request((cl, cr, cd, 0), 0)
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
    # The phenotype remains one packed L/R/D tile word.  Only individual gene
    # fields are 4-bit; never truncate a target tile to its low nibble here.
    grid = {tuple(p): int(s) for p, s in grid.items()}
    for state in grid.values():
        unpack_tile_state(state)  # validate, never truncate a corrupt tile word
    seeds = [tuple(s) for s in seeds]
    seedset = set(seeds)
    report = {'backend': 'nervous'}

    if not seeds:
        report.update(matched=0, target=len(grid), extra=0, missing=len(grid),
                      conflicts=[], seed_mismatch=[], unreachable=list(grid),
                      radius=0, telomere=1, exact=False, grown={},
                      note='no seeds/inputs — mark at least one input (the growth '
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
    direct_wanted = {context for context, output in growth.items() if output}

    # base genome: growth genes birth each cell, boundary seals stop the spill,
    # maintenance genes hold the fixed point. Patches (below) are PREPENDED so an
    # exact (Hamming-0) patch always wins the min-Hamming lookup.
    direct_base = (
        [HexGene(cl, cr, cd, 0, selector)
         for (cl, cr, cd), selector in growth.items()]
        + [HexGene(cl, cr, cd, 0, 0)
           for (cl, cr, cd) in _boundary_seals(grid, seeds, direct_wanted)]
        + [HexGene(cl, cr, cd, self_in, selector)
           for (cl, cr, cd, self_in), selector in maint.items()
           # A birth rule has priority when an all-zero selector inside a live
           # tile presents the same key as an unborn output circuit.
           if not (self_in == 0 and (cl, cr, cd) in growth)])

    # Alternative synthesis: use shared temporary 4-bit circuit selectors for
    # ambiguous lookups, then repack the three outputs before every next step.
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
        genes = list(patches) + base
        genome = Genome(chromosomes=[Chromosome(
            genes=genes, split=(0 if len(genes) < 2 else len(genes) // 2),
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
            for p, target_tile in grid.items():
                actual_tile = grown.get(p, 0)
                if actual_tile == target_tile or p in seedset:
                    continue
                actual_selectors = unpack_tile_state(actual_tile)
                target_selectors = unpack_tile_state(target_tile)
                for (direction, self_in), output in zip(
                        zip(_DIRS, actual_selectors), target_selectors):
                    if self_in == output:
                        continue
                    cl, cr, cd = _ctx(grown, *p, direction)
                    key = (cl, cr, cd, self_in)
                    if key not in seen:
                        add.append(HexGene(cl, cr, cd, self_in, output))
                        seen.add(key)
            for p in extras:
                for direction, actual in _selector_items(grown[p]):
                    cl, cr, cd = _ctx(grown, *p, direction)
                    if actual:
                        kill_key = (cl, cr, cd, actual)
                        if kill_key not in seen:
                            add.append(HexGene(*kill_key, 0))
                            seen.add(kill_key)
                    birth_key = (cl, cr, cd, 0)
                    if ((cl, cr, cd) not in wanted
                            and birth_key not in seen):
                        add.append(HexGene(*birth_key, 0))
                        seen.add(birth_key)
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
