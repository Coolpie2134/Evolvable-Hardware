"""Constructive FNV placement and associative-v2 compatibility growth."""
from __future__ import annotations

from substrates.nervous.hexgrid import hex_dirs, hex_frontier_cells

from .catalogue import STATE_DISTANCES
from .genome import (
    PlacementGene, germline_telomere, input_seed_grid, is_constructive,
)


def _compile_lookup(genome):
    return tuple(
        (gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in, gene.self_out)
        for chromosome in genome.chromosomes
        for gene in chromosome.genes
    )


def _lookup_compiled(program, s_l, s_r, s_d, self_state):
    if s_l == s_r == s_d == self_state == 0:
        return 0
    best_entry, best_distance = None, 1 << 30
    distances = STATE_DISTANCES
    for entry in program:
        ctx_l, ctx_r, ctx_d, self_in, _ = entry
        distance = (
            distances[ctx_l][s_l]
            + distances[ctx_r][s_r]
            + distances[ctx_d][s_d]
            + distances[self_in][self_state]
        )
        if distance < best_distance:
            best_entry, best_distance = entry, distance
    if best_entry is None:
        return 0
    # Empty cells only express explicit growth rules.  This prevents a
    # maintenance rule from filling the infinite field.
    if self_state == 0 and best_entry[3] != 0:
        return 0
    return int(best_entry[4])


def lookup(genome, s_l, s_r, s_d, self_state):
    return _lookup_compiled(
        _compile_lookup(genome), s_l, s_r, s_d, self_state)


def _next_state(program, s_l, s_r, s_d, self_state, cache):
    key = (s_l, s_r, s_d, self_state)
    if key not in cache:
        cache[key] = _lookup_compiled(program, *key)
    return cache[key]


def _grow_step(program, grid, telomeres, seed_states, germline, cache):
    nxt, nxt_telomeres = {}, {}
    for (x, y), state in grid.items():
        neighbors = hex_dirs(x, y)
        new_state = _next_state(
            program,
            grid.get(neighbors["L"], 0),
            grid.get(neighbors["R"], 0),
            grid.get(neighbors["D"], 0),
            state,
            cache,
        )
        if new_state:
            nxt[(x, y)] = new_state
            nxt_telomeres[(x, y)] = telomeres.get((x, y), 0)

    frontier = set()
    for cell in grid:
        frontier.update(
            neighbor for neighbor in hex_frontier_cells(*cell)
            if neighbor not in grid)
    for x, y in frontier:
        neighbors = hex_dirs(x, y)
        parent_telomere = max(
            (telomeres.get(neighbors[direction], 0)
             for direction in ("L", "R", "D")
             if neighbors[direction] in grid),
            default=0,
        )
        if parent_telomere <= 0:
            continue
        new_state = _next_state(
            program,
            grid.get(neighbors["L"], 0),
            grid.get(neighbors["R"], 0),
            grid.get(neighbors["D"], 0),
            0,
            cache,
        )
        if new_state:
            nxt[(x, y)] = new_state
            nxt_telomeres[(x, y)] = parent_telomere - 1

    for seed, state in seed_states.items():
        nxt[seed] = state
        nxt_telomeres[seed] = germline
    return nxt, nxt_telomeres


def _growth_budget(telomere):
    return 3 * int(telomere) + 6


def grow_functional(genome, seeds, grid_size=None, iters=None):
    """Grow until a fixed point/2-cycle; legacy size arguments are ignored."""
    del grid_size, iters
    seeds = tuple(tuple(seed) for seed in seeds)
    if not seeds:
        seeds = ((0, 0),)
    if is_constructive(genome):
        from .construction import develop_constructive
        return develop_constructive(genome, seeds).grid
    germline = germline_telomere(genome)
    program = _compile_lookup(genome)
    seed_states = input_seed_grid(seeds)
    grid = dict(seed_states)
    telomeres = {seed: germline for seed in seeds}
    previous, cache = None, {}
    for _ in range(_growth_budget(germline)):
        nxt, nxt_telomeres = _grow_step(
            program, grid, telomeres, seed_states, germline, cache)
        if nxt == grid or nxt == previous:
            return nxt
        previous, grid, telomeres = grid, nxt, nxt_telomeres
    return grid


def grow_functional_snapshots(genome, seeds, grid_size=None, iters=None):
    del grid_size, iters
    seeds = tuple(tuple(seed) for seed in seeds) or ((0, 0),)
    if is_constructive(genome):
        from .construction import develop_constructive
        return list(develop_constructive(
            genome, seeds, snapshots=True).snapshots)
    germline = germline_telomere(genome)
    program = _compile_lookup(genome)
    seed_states = input_seed_grid(seeds)
    grid = dict(seed_states)
    telomeres = {seed: germline for seed in seeds}
    snapshots = [dict(grid)]
    previous, cache = None, {}
    for _ in range(_growth_budget(germline)):
        nxt, nxt_telomeres = _grow_step(
            program, grid, telomeres, seed_states, germline, cache)
        snapshots.append(dict(nxt))
        if nxt == grid or nxt == previous:
            break
        previous, grid, telomeres = grid, nxt, nxt_telomeres
    return snapshots


def active_gene_loci_and_contexts(genome, seeds):
    """Return winning rules and encountered mature/frontier contexts.

    This is a search hint only. Development itself still evaluates every gene
    with the unchanged first-wins categorical-distance rule.
    """
    if is_constructive(genome):
        from .construction import develop_constructive
        active = develop_constructive(genome, seeds).active_ids
        loci = tuple(
            (chromosome_index, gene_index)
            for chromosome_index, chromosome in enumerate(genome.chromosomes)
            for gene_index, gene in enumerate(chromosome.genes)
            if isinstance(gene, PlacementGene)
            and int(gene.gene_id) in active)
        return loci, ()
    grid = grow_functional(genome, seeds)
    program = [
        (
            chromosome_index, gene_index,
            gene.ctx_l, gene.ctx_r, gene.ctx_d,
            gene.self_in, gene.self_out,
        )
        for chromosome_index, chromosome in enumerate(genome.chromosomes)
        for gene_index, gene in enumerate(chromosome.genes)
    ]
    contexts = []
    for cell, state in grid.items():
        neighbors = hex_dirs(*cell)
        contexts.append((
            grid.get(neighbors["L"], 0),
            grid.get(neighbors["R"], 0),
            grid.get(neighbors["D"], 0),
            state,
        ))
    frontier = set()
    for cell in grid:
        frontier.update(
            neighbor for neighbor in hex_frontier_cells(*cell)
            if neighbor not in grid)
    for cell in frontier:
        neighbors = hex_dirs(*cell)
        contexts.append((
            grid.get(neighbors["L"], 0),
            grid.get(neighbors["R"], 0),
            grid.get(neighbors["D"], 0),
            0,
        ))

    active = set()
    distances = STATE_DISTANCES
    for context in contexts:
        s_l, s_r, s_d, self_state = context
        best, best_distance = None, 1 << 30
        for (chromosome_index, gene_index, ctx_l, ctx_r, ctx_d,
             self_in, _self_out) in program:
            if self_state == 0 and self_in != 0:
                continue
            distance = (
                distances[ctx_l][s_l]
                + distances[ctx_r][s_r]
                + distances[ctx_d][s_d]
                + distances[self_in][self_state]
            )
            if distance < best_distance:
                best = (chromosome_index, gene_index)
                best_distance = distance
        if best is not None:
            active.add(best)
    return tuple(sorted(active)), tuple(sorted(set(contexts)))


def active_gene_loci(genome, seeds):
    """Return rules that win a lookup on the mature body or its frontier."""
    return active_gene_loci_and_contexts(genome, seeds)[0]
