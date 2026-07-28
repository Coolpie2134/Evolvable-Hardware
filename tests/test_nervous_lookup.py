"""Exact-equivalence checks for the packed nervous growth lookup."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.genome import (Chromosome, Genome, HexGene, germline_telomere,
                           random_hex_genome, MAX_STATE, TRI_STATE_MAX)
from substrates.nervous.hexgrid import hex_dirs, hex_frontier_cells
from substrates.nervous.nervous import (_compile_lookup, _grow_budget, _lookup_compiled,
                            _lookup_nv, _seed_state, grow_nervous,
                            grow_nervous_snapshots)


def _reference_lookup(genome, sL, sR, sD, si):
    """The original fieldwise scalar lookup, retained independently for tests."""
    if sL == 0 and sR == 0 and sD == 0 and si == 0:
        return 0
    # Widths come from the genome-level alphabet, not from the lookup module, so
    # this reference stays an independent check of the packed implementation.
    bits = ((TRI_STATE_MAX - 1).bit_length()
            if getattr(genome, 'arch', 'single') == 'tri3'
            else (MAX_STATE - 1).bit_length())
    mask = (1 << bits) - 1
    best_gene, best_distance = None, 1 << 30
    for chromosome in genome.chromosomes:
        for gene in chromosome.genes:
            distance = (
                ((gene.ctx_l ^ sL) & mask).bit_count()
                + ((gene.ctx_r ^ sR) & mask).bit_count()
                + ((gene.ctx_d ^ sD) & mask).bit_count()
                + ((gene.self_in ^ si) & mask).bit_count()
            )
            if distance < best_distance:
                best_gene, best_distance = gene, distance
    if best_gene is None or (si == 0 and best_gene.self_in != 0):
        return 0
    return best_gene.self_out


def _reference_grow(genome, seeds):
    """Original growth algorithm using the independent scalar lookup above."""
    seeds = tuple(seeds)
    L = germline_telomere(genome)
    seed_state = _seed_state(genome)
    grid = {position: seed_state for position in seeds}
    telomeres = {position: L for position in seeds}
    snapshots = [dict(grid)]
    previous, cache = None, {}

    def lookup(sL, sR, sD, si):
        key = (sL, sR, sD, si)
        if key not in cache:
            cache[key] = _reference_lookup(genome, sL, sR, sD, si)
        return cache[key]

    for _ in range(_grow_budget(L)):
        next_grid, next_telomeres = {}, {}
        for (x, y), state in grid.items():
            neighbors = hex_dirs(x, y)
            state_out = lookup(
                grid.get(neighbors['L'], 0), grid.get(neighbors['R'], 0),
                grid.get(neighbors['D'], 0), state)
            if state_out:
                next_grid[(x, y)] = state_out
                next_telomeres[(x, y)] = telomeres.get((x, y), 0)

        frontier = set()
        for x, y in grid:
            frontier.update(position for position in hex_frontier_cells(x, y)
                            if position not in grid)
        for x, y in frontier:
            neighbors = hex_dirs(x, y)
            parent_telomere = max(
                (telomeres.get(neighbors[direction], 0)
                 for direction in ('L', 'R', 'D')
                 if neighbors[direction] in grid),
                default=0,
            )
            if parent_telomere <= 0:
                continue
            state_out = lookup(
                grid.get(neighbors['L'], 0), grid.get(neighbors['R'], 0),
                grid.get(neighbors['D'], 0), 0)
            if state_out:
                next_grid[(x, y)] = state_out
                next_telomeres[(x, y)] = parent_telomere - 1

        for position in seeds:
            next_grid[position] = seed_state
            next_telomeres[position] = L
        snapshots.append(dict(next_grid))
        if next_grid == grid or next_grid == previous:
            return next_grid, snapshots
        previous, grid, telomeres = grid, next_grid, next_telomeres
    return grid, snapshots


def test_packed_lookup_matches_fieldwise_reference_for_both_architectures():
    random.seed(8101)
    for architecture, state_max in (('single', MAX_STATE),
                                    ('tri3', TRI_STATE_MAX)):
        for _ in range(20):
            genome = random_hex_genome(4, max_telomere=5, arch=architecture)
            program = _compile_lookup(genome)
            for _ in range(100):
                context = tuple(random.randrange(state_max) for _ in range(4))
                expected = _reference_lookup(genome, *context)
                assert _lookup_compiled(program, *context) == expected
                assert _lookup_nv(genome, *context) == expected


def test_packed_lookup_preserves_first_gene_wins_ties():
    first = HexGene(ctx_l=1, ctx_r=2, ctx_d=3, self_in=4, self_out=7)
    second = HexGene(ctx_l=1, ctx_r=2, ctx_d=3, self_in=4, self_out=19)
    genome = Genome([Chromosome(genes=[first, second])], arch='single')
    assert _lookup_nv(genome, 1, 2, 3, 4) == 7
    genome.chromosomes[0].genes.reverse()
    assert _lookup_nv(genome, 1, 2, 3, 4) == 19


def test_packed_lookup_preserves_empty_cell_guards():
    growth = Genome([Chromosome(genes=[
        HexGene(ctx_l=1, ctx_r=0, ctx_d=0, self_in=0, self_out=7)
    ])])
    maintenance = Genome([Chromosome(genes=[
        HexGene(ctx_l=1, ctx_r=0, ctx_d=0, self_in=1, self_out=7)
    ])])
    assert _lookup_nv(growth, 0, 0, 0, 0) == 0  # hard all-zero guard
    assert _lookup_nv(growth, 1, 0, 0, 0) == 7
    assert _lookup_nv(maintenance, 1, 0, 0, 0) == 0


def test_packed_growth_is_bit_identical_to_scalar_reference():
    random.seed(8102)
    seeds = ((0, 0), (0, 4))
    for architecture in ('single', 'tri3'):
        for _ in range(12):
            genome = random_hex_genome(4, max_telomere=5, arch=architecture)
            expected_grid, expected_snapshots = _reference_grow(genome, seeds)
            assert grow_nervous(genome, seeds) == expected_grid
            assert grow_nervous_snapshots(genome, seeds) == expected_snapshots

