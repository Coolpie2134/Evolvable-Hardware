"""
tests/test_nervous_branched.py - FNV's branched encoding on the hex nervous net.

The native hex encoding matches by minimum Hamming distance, so a cell's
configuration is a pure function of its neighbourhood and two cells in identical
surroundings MUST become identical cells. These tests pin the four mechanisms
that lift that ceiling, and the substrate-specific decisions the port required.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.branched import (                          # noqa: E402
    DEPTH_ANY, EMPTY_STATE, OUT_STATE, PAD_STATE,
    BranchedHexChromosome, BranchedHexGenome, HexContextGene, HexControlGene,
    HexOutputGene, IoChromosome, bearing_cell, develop_branched_hex, output_root_sites,
    routing_interface, routing_sources, state_distance)
from substrates.nervous.hexgrid import ROUTING_HEX, hex_dirs       # noqa: E402


def _genome(genes, tolerance=3, telomere=12, bearing=0, distance=2):
    return BranchedHexGenome(
        chromosomes=[BranchedHexChromosome(
            genes=list(genes),
            controls=[HexControlGene(tolerance=tolerance, telomere=telomere),
                      HexControlGene()])],
        io_chromosome=IoChromosome(outputs=[
            HexOutputGene(role='Q', bearing=bearing, distance=distance,
                          branch_id=1)]))


# -- substrate mapping ----------------------------------------------------------

def test_growth_buds_are_the_states_own_signal_sources():
    """The hex analogue of an FNV component's input directions.

    Reverse growth only makes sense if a cell exposes buds where it actually
    draws signal from. ROUTING_HEX names exactly that, so a coincidence node
    offers two buds and a buffer one - the same shape as a binary gate versus a
    unary transport on FNV.
    """
    assert routing_sources(0) == ()                     # dead: nothing to grow
    assert routing_sources(1) == ('D',)                 # buffer: one bud
    assert set(routing_sources(5)) == {'L', 'R'}        # coincidence: two buds
    assert set(routing_sources(15)) == {'L', 'R', 'D'}  # plus a veto source
    for state in (EMPTY_STATE, PAD_STATE, OUT_STATE):
        assert routing_sources(state) == ()


def test_tolerance_measures_circuits_not_integers():
    """A tolerance budget has to mean "nearly the same circuit".

    Routing states are an arbitrary enumeration: 15 and 16 are adjacent numbers
    but an AND of three sources versus an inert OR, while 1 and 17 are 16 apart
    numerically and the same source combined the other way. Numeric distance
    would let a tolerant rule wander across unrelated circuits.
    """
    assert state_distance(15, 15) == 0
    # Operation is a mutable function on the same physical interface, so it
    # costs no context distance; rewiring the interface does.
    assert state_distance(1, 17) == 0        # same source, AND vs OR
    assert state_distance(15, 16) == 3       # adjacent integers, unrelated
    assert state_distance(1, 17) < state_distance(15, 16)
    # and it is symmetric, or the budget would depend on match direction
    for a, b in ((1, 5), (15, 16), (0, 31), (7, 23)):
        assert state_distance(a, b) == state_distance(b, a)


def test_roots_use_ring_geometry_not_a_repeated_direction():
    """Walking one direction repeatedly does not travel in a straight line.

    hex_dirs reports neighbours in the node's OWN rotated frame, so two 'L'
    steps come straight back to the origin - the first attempt at placing roots
    put every one of them on top of the input pad and nothing grew at all.
    """
    origin = (0, 0)
    twice_left = hex_dirs(*hex_dirs(*origin)['L'])['L']
    assert twice_left == origin, 'the rotated frame no longer folds back'
    # Ring placement genuinely leaves the origin, at every bearing.
    for bearing in range(6):
        assert bearing_cell(bearing, 2) != origin


def test_roots_never_land_on_a_pad_or_on_each_other():
    genome = BranchedHexGenome(
        chromosomes=[BranchedHexChromosome(), BranchedHexChromosome()],
        io_chromosome=IoChromosome(outputs=[
            HexOutputGene(role='A', bearing=0, distance=1, branch_id=1),
            HexOutputGene(role='B', bearing=0, distance=1, branch_id=2)]))
    pads = [(0, 0), (-1, 0)]
    roots = output_root_sites(genome, pads)
    assert len(roots) == 2
    assert len(set(roots.values())) == 2
    assert not set(roots.values()).intersection(pads)


def test_cohort_construction_reuses_pads_but_grows_fresh_arms():
    """Compatible crossover needs several independently grown legal mates."""
    from substrates.nervous.branched_ga import (
        input_pads, random_branched_hex_genome)

    import random
    random.seed(73)
    founder = random_branched_hex_genome(2, 3, ('Sum', 'Carry'))
    peer = random_branched_hex_genome(
        2, 3, ('Sum', 'Carry'), input_genes=founder.inputs)

    assert input_pads(peer) == input_pads(founder)
    assert peer.inputs is not founder.inputs
    assert all(a is not b for a, b in zip(peer.inputs, founder.inputs))


def test_crossover_moves_an_output_root_with_its_arm():
    from substrates.nervous.branched_ga import (
        crossover_branched_hex, random_branched_hex_genome)

    import random
    random.seed(75)
    left = random_branched_hex_genome(2, 3, ('Sum', 'Carry'))
    right = random_branched_hex_genome(
        2, 3, ('Sum', 'Carry'), input_genes=left.inputs)
    right.outputs[1].bearing = (left.outputs[1].bearing + 2) % 6
    right.outputs[1].distance = 6
    for seed in range(100):
        random.seed(seed)
        child = crossover_branched_hex(left, right)
        if child.outputs[1].distance == 6:
            assert child.outputs[1].bearing == right.outputs[1].bearing
            break
    else:
        raise AssertionError('crossover never selected Carry arm')


# -- the four mechanisms --------------------------------------------------------

def test_an_arm_starts_only_at_its_own_genetic_output_root():
    """Nothing grows without a root rule - that is what starts a branch."""
    rootless = _genome([HexContextGene(1, self_in=EMPTY_STATE, self_out=1,
                                       branch_id=1)])
    assert develop_branched_hex(rootless, [(0, 0)]).grid == {}

    rooted = _genome([HexContextGene(1, self_in=OUT_STATE, self_out=5,
                                     branch_id=1)])
    trace = develop_branched_hex(rooted, [(0, 0)])
    assert list(trace.grid.values()) == [5]
    assert set(trace.grid) == set(output_root_sites(rooted, [(0, 0)]).values())


def test_depth_bands_make_one_neighbourhood_build_two_different_cells():
    """The individuation the native encoding cannot express.

    Every bud here sees the same neighbourhood; only its distance along the
    branch differs. Under nearest-gene matching all of them would be forced to
    the same state.
    """
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1, depth=1),
        HexContextGene(3, self_in=EMPTY_STATE, self_out=17, branch_id=1, depth=2),
    ])
    trace = develop_branched_hex(genome, [(0, 0)])
    by_depth = {}
    for cell, state in trace.grid.items():
        by_depth.setdefault(trace.depths[cell], set()).add(state)
    assert by_depth[0] == {5}
    assert by_depth[1] == {1}
    assert by_depth[2] == {17}


def test_an_unbuilt_bud_takes_its_parents_depth_not_zero():
    """Regression: depth-banded rules were silently dead.

    A bud has no depth of its own until it is built. Defaulting it to 0 made
    every bud look like a root, so a rule banded to depth 1 could never fire and
    the whole mechanism was inert while appearing to work.
    """
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1, depth=1),
    ])
    trace = develop_branched_hex(genome, [(0, 0)])
    assert len(trace.grid) > 1, 'a depth-1 rule never fired'
    assert set(trace.depths.values()) == {0, 1}


def test_a_rule_is_spent_after_one_cohort():
    """Bounded ontogenic amplification.

    A rule may differentiate a whole synchronous wave, but cannot fire again on
    the next one and extrude an unbounded chain. Without this, one growth rule
    walks to the placement ceiling.
    """
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=5, branch_id=1),
    ], telomere=120)
    trace = develop_branched_hex(genome, [(0, 0)])
    # Root cohort + exactly one expression wave of the growth rule.
    assert 1 < len(trace.grid) <= 4, len(trace.grid)


def test_an_arms_telomere_is_a_lifespan_in_changed_cells():
    """Not a radius. Cutting the lifespan cuts the body, one cell per change."""
    sizes = []
    for telomere in (1, 2, 3):
        genome = _genome([
            HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
            HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1,
                           depth=1),
        ], telomere=telomere)
        sizes.append(len(develop_branched_hex(genome, [(0, 0)]).grid))
    assert sizes == sorted(sizes), sizes
    assert sizes[0] < sizes[-1], sizes


def test_every_built_cell_is_owned_by_the_arm_that_built_it():
    """Ownership is what scopes a rule to its own limb rather than the field."""
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1, depth=1),
    ])
    trace = develop_branched_hex(genome, [(0, 0)])
    assert trace.grid
    assert set(trace.owners.values()) == {1}
    assert set(trace.owners) == set(trace.grid)


def test_pads_are_read_only():
    """Pads are the input interface; development may read them, never write."""
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1),
    ], tolerance=8, telomere=60)
    pads = [(0, 0), (1, 0), (-1, 0)]
    trace = develop_branched_hex(genome, pads)
    assert not set(trace.grid).intersection(pads)


def test_development_is_deterministic():
    """No RNG anywhere in development - the same genome grows the same body."""
    genome = _genome([
        HexContextGene(1, self_in=OUT_STATE, self_out=5, branch_id=1),
        HexContextGene(2, self_in=EMPTY_STATE, self_out=1, branch_id=1, depth=1),
        HexContextGene(3, self_in=EMPTY_STATE, self_out=17, branch_id=1, depth=2),
    ])
    first = develop_branched_hex(genome, [(0, 0)])
    for _ in range(5):
        again = develop_branched_hex(genome, [(0, 0)])
        assert again.grid == first.grid
        assert again.owners == first.owners
        assert again.depths == first.depths


def test_the_native_hex_encoding_is_untouched():
    """This encoding is additive and opt-in.

    Nothing in the branched module may alter the native HexGene path, or every
    recorded nervous result and checkpoint silently changes meaning.
    """
    from substrates.nervous.genome import HexGene

    fields = HexGene().__dict__
    assert 'branch_id' not in fields and 'depth' not in fields
    assert set(fields) >= {'ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'}


# -- construction and variation -------------------------------------------------

def test_random_genomes_are_living_organisms_not_inert_rule_lists():
    """A random context rule almost never fires; drawing from OBSERVED contexts
    is what makes an initial population worth evaluating."""
    import random
    from substrates.nervous.branched_ga import (
        input_pads, random_branched_hex_genome)

    random.seed(7)
    alive = built = 0
    for _ in range(20):
        genome = random_branched_hex_genome(2, n_inputs=2, output_roles=('Q',))
        pads = input_pads(genome)
        grid = develop_branched_hex(genome, pads).grid
        alive += bool(grid)
        roots = output_root_sites(genome, pads)
        built += bool(roots) and all(cell in grid for cell in roots.values())
    assert alive >= 18, alive
    assert built >= 14, built


def test_construction_is_reproducible_from_the_seed():
    """The lesson from the FNV hash-order bug, pinned here from the start.

    Every context carries states and is collected in a set; iterating that set
    to feed random.choice made seeded FNV runs explore a different search in
    every process. observed_contexts sorts, and this test is what keeps it that
    way.
    """
    import random
    from substrates.nervous.branched_ga import (
        input_pads, observed_contexts, random_branched_hex_genome)

    bodies = []
    for _ in range(2):
        random.seed(4242)
        genome = random_branched_hex_genome(2, n_inputs=2, output_roles=('Q',))
        bodies.append(develop_branched_hex(genome, input_pads(genome)).grid)
    assert bodies[0] == bodies[1]

    # And the collection that feeds the draws is ordered, not a raw set.
    random.seed(1)
    genome = random_branched_hex_genome(2, n_inputs=2, output_roles=('Q',))
    seen = observed_contexts(genome, 1)
    assert not isinstance(seen, (set, frozenset))
    assert list(seen) == sorted(seen)


def test_mutation_keeps_organisms_alive_and_actually_changes_them():
    """A variation operator that kills its children, or changes nothing, is
    useless in different directions."""
    import random
    from substrates.nervous.branched_ga import (
        input_pads, mutate_branched_hex, random_branched_hex_genome)

    random.seed(11)
    parents = [random_branched_hex_genome(2, 2, ('Q',)) for _ in range(12)]
    children = [mutate_branched_hex(genome) for genome in parents]
    alive = sum(1 for genome in children
                if develop_branched_hex(genome, input_pads(genome)).grid)
    changed = sum(
        1 for parent, child in zip(parents, children)
        if (develop_branched_hex(parent, input_pads(parent)).grid
            != develop_branched_hex(child, input_pads(child)).grid))
    assert alive >= 10, alive
    assert changed >= 6, changed


def test_crossover_trades_whole_arms():
    """An arm owns a territory and a lifespan, so it is the smallest piece that
    means anything on its own; trading single rules would mix genes selected
    against different bodies."""
    import random
    from substrates.nervous.branched_ga import (
        crossover_branched_hex, input_pads, random_branched_hex_genome)

    random.seed(21)
    left = random_branched_hex_genome(2, 2, ('Q',))
    right = random_branched_hex_genome(2, 2, ('Q',))
    child = crossover_branched_hex(left, right)
    for chromosome in child.chromosomes:
        for gene in chromosome.genes:
            assert 1 <= int(gene.branch_id) <= 2 * len(child.chromosomes)
    ids = [gene.gene_id for c in child.chromosomes for gene in c.genes]
    assert child.next_gene_id > max(ids, default=0)
    assert develop_branched_hex(child, input_pads(child)) is not None


def test_crossover_keeps_an_arm_in_its_input_pad_environment():
    import random
    from substrates.nervous.branched_ga import (
        crossover_branched_hex, input_pads, random_branched_hex_genome)

    random.seed(211)
    left = random_branched_hex_genome(1, 2, ('Q',))
    right = random_branched_hex_genome(1, 2, ('Q',))
    for bearing in range(6):
        right.inputs[0].distance = 6
        right.inputs[0].bearing = bearing
        if input_pads(right) != input_pads(left):
            break
    else:
        raise AssertionError('could not construct distinct input layouts')

    child = crossover_branched_hex(left, right)
    assert child == left


def test_a_grown_body_is_directly_simulable_with_genome_derived_io():
    """The point of output rooting: I/O comes from the genome, not from probes
    fitted after growth."""
    import random
    from substrates.nervous.branched_ga import (
        prepare_branched_hex, random_branched_hex_genome)
    from substrates.nervous.nervous import interpret_nervous
    import tools.benchmark as benchmark

    target = benchmark.targets_for_backend('nervous', 'paper_analog')['AND']
    roles = tuple(terminal.role for terminal in target.outputs)
    random.seed(3)
    complete = 0
    for _ in range(30):
        genome = random_branched_hex_genome(2, target.n_inputs, roles)
        prepared = prepare_branched_hex(genome, target)
        if prepared is None:
            continue
        grid, inputs, outputs = prepared
        routing, _ip, _op = interpret_nervous(grid, None, arch='single')
        assert set(routing) == set(grid)
        assert len(inputs) == target.n_inputs
        assert set(outputs) == set(roles)
        assert all(cell in grid for cell in outputs.values())
        complete += 1
    assert complete >= 20, complete


def test_an_organism_missing_a_root_is_rejected_not_patched():
    """A role whose root was never built has no output. Fitting a probe onto
    some other cell would hide exactly the failure selection needs to see."""
    from substrates.nervous.branched_ga import prepare_branched_hex
    import tools.benchmark as benchmark

    target = benchmark.targets_for_backend('nervous', 'paper_analog')['AND']
    roles = tuple(terminal.role for terminal in target.outputs)
    barren = BranchedHexGenome(
        chromosomes=[BranchedHexChromosome()],
        io_chromosome=IoChromosome(outputs=[
            HexOutputGene(role=roles[0], bearing=0, distance=2,
                          branch_id=1)]))
    assert prepare_branched_hex(barren, target) is None
