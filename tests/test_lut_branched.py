"""
tests/test_lut_branched.py - the branched encoding on the four-directional LUT.

The design decision under test: a branched LUT cell is built from the run's
ENABLED FUNCTION BANKS, not from arbitrary 16-bit tables. The raw state space is
2^64 per cell, in which "nearly the same cell" is meaningless - and an arm's
tolerance budget is exactly a claim about nearness. A finite catalogue of named
gates gives the LUT array the same kind of alphabet FNV has.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.lut.branched import (                              # noqa: E402
    DEPTH_ANY, EMPTY_CELL, OUT_CELL, PAD_CELL,
    BranchedLutChromosome, BranchedLutGenome, LutContextGene, LutControlGene,
    LutOutputGene, catalogue, cell_distance, cell_sources,
    develop_branched_lut, neighbours, output_root_sites, square_ring,
    table_family, table_support)
from substrates.lut.functions import FUNCTION_FAMILIES             # noqa: E402

NAMED = tuple(family for family in FUNCTION_FAMILIES
              if family != 'UNRESTRICTED')


def _genome(genes, families=NAMED, tolerance=6, telomere=14):
    return BranchedLutGenome(
        chromosomes=[BranchedLutChromosome(
            genes=list(genes),
            controls=[LutControlGene(tolerance=tolerance, telomere=telomere),
                      LutControlGene()])],
        outputs=[LutOutputGene(role='Q', bearing=0, distance=2, branch_id=1)],
        families=families)


def test_the_alphabet_is_the_enabled_banks_not_all_65536_tables():
    """A cell is assembled from named gates, so tolerance can mean something."""
    everything = catalogue(NAMED)
    assert len(everything) == 79, len(everything)
    assert len(catalogue(('ROUTING',))) == 4
    assert len(catalogue(('AND', 'XOR'))) == 22
    # A run may only install what its banks allow.
    routing_only = {table for _family, table in catalogue(('ROUTING',))}
    assert routing_only.issubset({table for _f, table in everything})
    assert all(table_family(table, ('ROUTING',)) == 'ROUTING'
               for table in routing_only)
    # Deterministic order: this feeds random draws during construction.
    assert list(everything) == sorted(everything)


def test_growth_buds_are_computed_from_real_table_dependence():
    """A cell's buds are the directions its tables actually read.

    Computed rather than declared, so the buds stay honest for any table: a
    table depends on an input exactly when flipping it changes the output.
    """
    assert table_support(0x0000) == frozenset()      # constant: reads nothing
    assert table_support(0xFFFF) == frozenset()
    for family, table in catalogue(NAMED):
        support = table_support(table)
        assert support, (family, hex(table))         # a named gate reads something
        assert support <= {'N', 'S', 'E', 'W'}
    two_input_and = next(table for family, table in catalogue(('AND',))
                         if len(table_support(table)) == 2)
    assert len(cell_sources((two_input_and, 0, 0, 0))) == 2
    assert cell_sources(EMPTY_CELL) == frozenset()
    assert cell_sources(PAD_CELL) == frozenset()


def test_distance_tracks_function_not_bit_patterns():
    """Hamming over raw tables would call unrelated gates adjacent."""
    and_table = catalogue(('AND',))[0][1]
    xor_table = catalogue(('XOR',))[0][1]
    same = cell_distance((and_table, 0, 0, 0), (and_table, 0, 0, 0), NAMED)
    across = cell_distance((and_table, 0, 0, 0), (xor_table, 0, 0, 0), NAMED)
    assert same == 0
    assert across > 0
    # The two tables differ in many raw bits; the metric does not care.
    assert bin(and_table ^ xor_table).count('1') > across
    # Symmetric, or a budget would depend on match direction.
    assert across == cell_distance((xor_table, 0, 0, 0), (and_table, 0, 0, 0),
                                   NAMED)


def test_square_rings_are_a_usable_polar_coordinate():
    """Bearing slides a site around the anchor; distance moves it in and out."""
    assert square_ring(0) == ((0, 0),)
    for radius in (1, 2, 3):
        ring = square_ring(radius)
        assert len(ring) == 4 * radius
        assert len(set(ring)) == len(ring)
        assert all(abs(x) + abs(y) == radius for x, y in ring)


def test_four_neighbours_are_the_square_lattice_not_the_honeycomb():
    around = neighbours((0, 0))
    assert set(around) == {'N', 'S', 'E', 'W'}
    assert len(set(around.values())) == 4
    assert around['N'] != around['S'] and around['E'] != around['W']


def test_an_arm_starts_only_at_its_genetic_output_root():
    and_table = catalogue(('AND',))[0][1]
    rootless = _genome([LutContextGene(1, self_in=EMPTY_CELL,
                                       self_out=(and_table, 0, 0, 0),
                                       branch_id=1)])
    assert develop_branched_lut(rootless, [(0, 0)]).grid == {}

    rooted = _genome([LutContextGene(1, self_in=OUT_CELL,
                                     self_out=(and_table, 0, 0, 0),
                                     branch_id=1)])
    trace = develop_branched_lut(rooted, [(0, 0)])
    assert len(trace.grid) == 1
    assert set(trace.grid) == set(output_root_sites(rooted, [(0, 0)]).values())


def test_depth_bands_differentiate_on_the_square_lattice_too():
    """The individuation the native LUT ontogeny cannot express."""
    and_table = catalogue(('AND',))[0][1]
    routes = [table for _family, table in catalogue(('ROUTING',))]
    genome = _genome([
        LutContextGene(1, self_in=OUT_CELL, self_out=(and_table, 0, 0, 0),
                       branch_id=1),
        LutContextGene(2, self_in=EMPTY_CELL, self_out=(routes[0], 0, 0, 0),
                       branch_id=1, depth=1),
        LutContextGene(3, self_in=EMPTY_CELL, self_out=(routes[1], 0, 0, 0),
                       branch_id=1, depth=2),
    ])
    trace = develop_branched_lut(genome, [(0, 0)])
    by_depth = {}
    for cell, state in trace.grid.items():
        by_depth.setdefault(trace.depths[cell], set()).add(state)
    assert len(by_depth) >= 3
    assert len({state for states in by_depth.values() for state in states}) >= 3


def test_pads_are_read_only_and_development_is_deterministic():
    and_table = catalogue(('AND',))[0][1]
    routes = [table for _family, table in catalogue(('ROUTING',))]
    genome = _genome([
        LutContextGene(1, self_in=OUT_CELL, self_out=(and_table, 0, 0, 0),
                       branch_id=1),
        LutContextGene(2, self_in=EMPTY_CELL, self_out=(routes[0], 0, 0, 0),
                       branch_id=1, depth=1),
    ])
    pads = [(0, 0), (1, 0)]
    first = develop_branched_lut(genome, pads)
    assert not set(first.grid).intersection(pads)
    for _ in range(4):
        again = develop_branched_lut(genome, pads)
        assert again.grid == first.grid and again.depths == first.depths


def test_the_native_lut_encoding_is_untouched():
    """Additive and opt-in, or every recorded LUT result changes meaning."""
    from substrates.lut.genome import LutGene

    fields = LutGene().__dict__
    assert 'branch_id' not in fields and 'depth' not in fields
    assert set(fields) >= {'ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in'}


# -- construction and variation -------------------------------------------------

def test_random_lut_genomes_live_and_install_only_enabled_gates():
    """The catalogue is a run-level constraint, not a suggestion."""
    import random
    from substrates.lut.branched_ga import (
        input_pads, prepare_branched_lut, random_branched_lut_genome)

    families = ('ROUTING', 'AND', 'OR', 'XOR', 'VETO')

    class _Target:
        n_inputs = 2

    random.seed(7)
    alive = complete = 0
    for _ in range(20):
        genome = random_branched_lut_genome(2, 2, ('Q',), families=families)
        grid = develop_branched_lut(genome, input_pads(genome)).grid
        alive += bool(grid)
        complete += prepare_branched_lut(genome, _Target()) is not None
        for cell in grid.values():
            for table in cell:
                if table:
                    assert table_family(table, families), hex(table)
    assert alive >= 18, alive
    assert complete >= 15, complete


def test_lut_construction_is_reproducible_and_variation_is_safe():
    import random
    from substrates.lut.branched_ga import (
        crossover_branched_lut, input_pads, mutate_branched_lut,
        observed_contexts, random_branched_lut_genome)

    families = ('ROUTING', 'AND', 'XOR')
    bodies = []
    for _ in range(2):
        random.seed(4242)
        genome = random_branched_lut_genome(2, 2, ('Q',), families=families)
        bodies.append(develop_branched_lut(genome, input_pads(genome)).grid)
    assert bodies[0] == bodies[1]

    random.seed(9)
    genome = random_branched_lut_genome(2, 2, ('Q',), families=families)
    seen = observed_contexts(genome, 1)
    assert not isinstance(seen, (set, frozenset))

    parents = [random_branched_lut_genome(2, 2, ('Q',), families=families)
               for _ in range(8)]
    children = [mutate_branched_lut(parent) for parent in parents]
    crossed = [crossover_branched_lut(parents[i], parents[(i + 1) % 8])
               for i in range(8)]
    for pool in (children, crossed):
        alive = sum(1 for genome in pool
                    if develop_branched_lut(genome, input_pads(genome)).grid)
        assert alive >= 6, alive
