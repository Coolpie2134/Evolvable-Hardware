"""The branched FNV encoding: 2n arms, spawns, and global context rules.

A chromosome has two arms and therefore two branches. The rule at the centromere
end of an arm is its SPAWN: it names an input pad and a direction out of it, and
places its component there. Every other rule is a CONTEXT rule naming, per input
pin, the grown node TYPE that must drive it.

Growth is synchronous: each iteration every rule of every living arm is matched
against the grid as it stood at the start of that iteration and fires at EVERY
cell it fits. Which arm a rule sits in never gates where it applies - an arm
decides only where its branch starts and how long it lives.
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime.checkpoint import genome_from_dict, genome_to_dict
from substrates.fnv.catalogue import BY_ID, BY_NAME
from substrates.fnv.construction import (
    arm_telomere, branch_growth_order, develop_constructive)
from substrates.fnv.construction_ga import (
    TYPED_MUT_OPS, branch_cut, chromosome_rules, constructive_signature,
    crossover_typed, mutate_typed, mutate_typed_once, placement_genes,
    random_typed_genome, relabel_branches, typed_branches,
)
from substrates.fnv.genome import (
    BranchRef, Chromosome, Genome, PlacementGene, TYPED_ENCODING,
    is_typed, random_functional_genome, validate_genome,
)

FAMILIES = ("LOGIC", "DELAY")
#: Input D, outputs L and R.
DOWN_TO_LR = BY_NAME["DELAY1_D_TO_LR"].id
#: Input L, outputs R and D - it consumes what DOWN_TO_LR drives leftward.
LEFT_TO_RD = BY_NAME["DELAY1_L_TO_RD"].id
#: Same pins as DOWN_TO_LR, different component, for the conflict test.
DOWN_TO_L = BY_NAME["DELAY1_D_TO_L"].id
PADS = ((0, 0), (-8, -8))


def _genome(chromosomes, pads=PADS, next_gene_id=99):
    return Genome(chromosomes=chromosomes, input_layout=pads,
                  encoding=TYPED_ENCODING, next_gene_id=next_gene_id)


def _arm(spawn_component, rules, telomere, pad=-1):
    """One bottom-arm branch: a spawn on ``pad`` followed by context rules."""
    genes = [PlacementGene(1, spawn_component, (BranchRef(pad, "D"),), 1)]
    for index, (component, source) in enumerate(rules):
        genes.append(PlacementGene(
            index + 2, component, (BranchRef(source, "L"),), 1))
    return Chromosome(genes=genes, split=0, telomere_top=0,
                      telomere_bottom=telomere)


def _body(genome):
    seeds = tuple(genome.input_layout)
    trace = develop_constructive(genome, seeds)
    return trace, len(trace.grid) - len(seeds)


def test_a_chromosome_has_exactly_two_arms():
    random.seed(3)
    for count in (1, 2, 3):
        genome = random_typed_genome(count, FAMILIES, 3)
        branches = typed_branches(genome)
        assert len(branches) == 2 * count
        for index, chromosome in enumerate(genome.chromosomes):
            rules = chromosome_rules(chromosome)
            arms = branch_growth_order(chromosome)
            # The two arms partition the chromosome, each read outward from the
            # centromere, so the top one is stored reversed.
            assert list(reversed(arms[0])) + list(arms[1]) == rules
            assert branches[(index, 0)] == list(arms[0])
            assert branches[(index, 1)] == list(arms[1])


def test_the_spawn_starts_its_branch_one_step_out_of_a_pad():
    # Telomere 1 buys exactly the spawn and nothing else.
    genome = _genome([_arm(DOWN_TO_LR, [], 1)])
    trace, cells = _body(genome)
    assert cells == 1
    # (0, 0) is pad -1 and the spawn declared direction D out of it.
    assert [cell for cell in trace.grid if cell not in PADS] == [(0, 1)]
    assert trace.grid[(0, 1)] == DOWN_TO_LR
    assert sorted(trace.active_ids) == [1]

    # A different pad moves the whole branch.
    moved = _genome([_arm(DOWN_TO_LR, [], 1, pad=-2)])
    assert [cell for cell in _body(moved)[0].grid
            if cell not in PADS] == [(-8, -7)]


def test_a_context_rule_fires_off_the_type_it_names():
    genome = _genome([_arm(DOWN_TO_LR, [(LEFT_TO_RD, DOWN_TO_LR)], 4)])
    trace, cells = _body(genome)
    assert cells == 2
    assert trace.grid[(0, 1)] == DOWN_TO_LR
    assert trace.grid[(1, 1)] == LEFT_TO_RD
    assert sorted(trace.active_ids) == [1, 2]

    # Naming a type nothing grows leaves the rule dormant.
    idle = _genome([_arm(DOWN_TO_LR, [(LEFT_TO_RD, LEFT_TO_RD)], 4)])
    idle_trace, idle_cells = _body(idle)
    assert idle_cells == 1
    assert sorted(idle_trace.dormant_ids) == [2]


def test_an_arms_telomere_is_spent_one_per_placement():
    sizes = [_body(_genome([_arm(
        DOWN_TO_LR, [(LEFT_TO_RD, DOWN_TO_LR)], life)]))[1]
        for life in (0, 1, 2, 3)]
    # Nothing, the spawn, the spawn plus one rule, then the branch is spent.
    assert sizes == [0, 1, 2, 2]

    chromosome = _arm(DOWN_TO_LR, [], 5)
    assert arm_telomere(chromosome, 1) == 5
    assert arm_telomere(chromosome, 0) == 0        # empty top arm


def test_a_burst_may_overshoot_the_telomere_within_one_iteration():
    # One rule, two matching sites in the same iteration, one telomere left.
    # The iteration is checked before it runs, so both placements stand.
    spawn_a = PlacementGene(1, DOWN_TO_LR, (BranchRef(-1, "D"),), 1)
    spawn_b = PlacementGene(2, DOWN_TO_LR, (BranchRef(-2, "D"),), 1)
    rule = PlacementGene(3, LEFT_TO_RD, (BranchRef(DOWN_TO_LR, "L"),), 1)
    # Two spawns on separate pads, then one shared rule with a single life left.
    genome = _genome([
        Chromosome(genes=[spawn_a], split=0, telomere_top=0,
                   telomere_bottom=1),
        Chromosome(genes=[spawn_b, rule], split=0, telomere_top=0,
                   telomere_bottom=2)])
    trace, cells = _body(genome)
    # spawn_a, spawn_b, and the rule firing at BOTH of their sites at once.
    assert cells == 4
    assert sorted(trace.active_ids) == [1, 2, 3]


def test_two_rules_disagreeing_over_a_cell_place_nothing():
    # Both rules match the cell right of the spawn. They name different
    # components, so they disagree and neither wins.
    spawn = PlacementGene(1, DOWN_TO_LR, (BranchRef(-1, "D"),), 1)
    left = PlacementGene(2, LEFT_TO_RD, (BranchRef(DOWN_TO_LR, "L"),), 1)
    other = BY_NAME["DELAY2_L_TO_RD"].id
    right = PlacementGene(3, other, (BranchRef(DOWN_TO_LR, "L"),), 1)
    contested = _genome([Chromosome(
        genes=[spawn, left, right], split=0, telomere_top=0,
        telomere_bottom=6)])
    trace, cells = _body(contested)
    assert cells == 1                       # only the spawn
    assert sorted(trace.dormant_ids) == [2, 3]

    # The same two rules AGREEING place it once, and both count as fired.
    agreed = _genome([Chromosome(
        genes=[spawn, left, PlacementGene(
            3, LEFT_TO_RD, (BranchRef(DOWN_TO_LR, "L"),), 1)],
        split=0, telomere_top=0, telomere_bottom=6)])
    agreed_trace, agreed_cells = _body(agreed)
    assert agreed_cells == 2
    assert sorted(agreed_trace.active_ids) == [1, 2, 3]


def test_a_rule_fires_on_structure_another_arm_built():
    # The rule lives in chromosome 1, whose own branch is far away and of the
    # wrong type to feed it. The only node it can match was grown by chromosome
    # 0, so which arm a rule sits in must not gate where it applies.
    spawn = PlacementGene(1, DOWN_TO_LR, (BranchRef(-1, "D"),), 1)
    elsewhere = PlacementGene(2, LEFT_TO_RD, (BranchRef(-2, "L"),), 1)
    rule = PlacementGene(3, LEFT_TO_RD, (BranchRef(DOWN_TO_LR, "L"),), 1)
    genome = _genome([
        Chromosome(genes=[spawn], split=0, telomere_top=0, telomere_bottom=4),
        Chromosome(genes=[elsewhere, rule], split=0, telomere_top=0,
                   telomere_bottom=4)])
    trace, cells = _body(genome)
    assert cells == 3
    # Chromosome 0's node, chromosome 1's own distant spawn, and the rule's
    # placement hanging off chromosome 0.
    assert trace.grid[(0, 1)] == DOWN_TO_LR
    assert trace.grid[(1, 1)] == LEFT_TO_RD
    assert sorted(trace.active_ids) == [1, 2, 3]


def test_more_chromosomes_give_more_branches_and_a_larger_organism():
    sizes = {}
    for count in (1, 2, 3, 4):
        random.seed(21)
        grown = [_body(random_typed_genome(count, FAMILIES, 3))[1]
                 for _ in range(40)]
        sizes[count] = sorted(grown)[len(grown) // 2]
    assert sizes[1] < sizes[2] < sizes[3] < sizes[4]


def test_every_rule_keeps_the_form_its_position_requires():
    # Position decides meaning, so an edit that moves a rule across the
    # centromere - or deletes the spawn ahead of it - has to restore the form.
    assert TYPED_MUT_OPS == [
        "tweak", "add_gene", "duplicate_branch", "del_rule", "del_branch",
        "telomere"]
    random.seed(9)
    genome = random_typed_genome(2, FAMILIES, 3)
    for _ in range(300):
        mutate_typed_once(genome, FAMILIES, 3)
        validate_genome(genome, FAMILIES)
        for members in typed_branches(genome).values():
            for index, gene in enumerate(members):
                entry = BY_ID[int(gene.component_id)]
                assert tuple(
                    ref.direction for ref in gene.inputs) == entry.inputs
                for ref in gene.inputs:
                    if index == 0:
                        # A spawn names a pad, and takes a single input.
                        assert int(ref.node_id) < 0
                        assert len(gene.inputs) == 1
                    else:
                        # A context rule names a grown node type, never a pad.
                        assert int(ref.node_id) > 0
                        assert ref.direction in BY_ID[
                            int(ref.node_id)].outputs


def test_each_arm_carries_its_own_lifespan_through_crossover():
    random.seed(13)
    for _ in range(120):
        left = random_typed_genome(2, FAMILIES, 3)
        right = random_typed_genome(2, FAMILIES, 3)
        for _ in range(6):
            mutate_typed(left, None, FAMILIES, 3)
            mutate_typed(right, None, FAMILIES, 3)
        relabel_branches(left)
        relabel_branches(right)
        child = crossover_typed(left, right, FAMILIES)
        validate_genome(child, FAMILIES)
        ids = [gene.gene_id for gene in placement_genes(child)]
        assert len(ids) == len(set(ids))
        assert child.next_gene_id > max(ids, default=0)
        assert len(ids) <= 128
        for index, chromosome in enumerate(child.chromosomes):
            # Each arm's lifespan came from whichever parent gave that arm.
            for half, value in enumerate((chromosome.telomere_top,
                                          chromosome.telomere_bottom)):
                field = "telomere_top" if half == 0 else "telomere_bottom"
                assert int(value) in {
                    int(getattr(left.chromosomes[index], field)),
                    int(getattr(right.chromosomes[index], field))}


def test_a_lifespan_or_centromere_change_is_a_real_change():
    genome = _genome([_arm(DOWN_TO_LR, [(LEFT_TO_RD, DOWN_TO_LR)], 4)])
    before = constructive_signature(genome)
    genome.chromosomes[0].telomere_bottom = 2
    assert constructive_signature(genome) != before
    # Moving the centromere changes which rule is the spawn, so it must be
    # visible to anything that dedupes on the signature.
    moved = _genome([_arm(DOWN_TO_LR, [(LEFT_TO_RD, DOWN_TO_LR)], 4)])
    moved.chromosomes[0].split = 1
    assert constructive_signature(moved) != before
    assert branch_cut(moved.chromosomes[0]) == 1


def test_branched_checkpoint_roundtrips_arms_and_lifespans():
    random.seed(17)
    genome = random_typed_genome(2, FAMILIES, 3)
    for _ in range(12):
        mutate_typed(genome, None, FAMILIES, 3)
    document = genome_to_dict(genome, "fnv")
    assert document["encoding"] == TYPED_ENCODING
    # Its own development version: the same gene record read by a different
    # rule must not silently load as constructive_v3.
    assert document["development_version"] == 5
    restored = genome_from_dict(document, "fnv")
    assert restored == genome
    assert is_typed(restored)
    assert [(c.telomere_top, c.telomere_bottom, c.split)
            for c in restored.chromosomes] == [
        (c.telomere_top, c.telomere_bottom, c.split)
        for c in genome.chromosomes]
    assert _body(restored)[1] == _body(genome)[1]


def test_constructive_v3_growth_is_bounded_by_the_germline_radius():
    # The run's Max telomere is the constructive organism's growth RADIUS, and
    # is a different field from the branched per-arm lifespans.
    from substrates.fnv.construction import growth_radius

    random.seed(19)
    genome = random_functional_genome(
        2, max_telomere=20, families=FAMILIES, n_inputs=3)
    assert not is_typed(genome)
    assert [c.telomere for c in genome.chromosomes] == [20, 20]
    seeds = tuple(genome.input_layout)
    grown = {}
    for radius in (2, 4, 20):
        for chromosome in genome.chromosomes:
            chromosome.telomere = radius
        trace = develop_constructive(genome, seeds)
        depths = [trace.depths[node] for node in trace.active_ids]
        assert not depths or max(depths) <= radius
        grown[radius] = len(trace.grid)
    assert grown[2] < grown[4] <= grown[20]

    for chromosome in genome.chromosomes:
        chromosome.telomere = 1
    assert growth_radius(genome) is None       # legacy = unbounded
    assert len(develop_constructive(genome, seeds).grid) >= grown[20]


def test_fnv_default_radius_cannot_bind():
    from runtime.config import default_max_telomere
    from substrates.fnv.genome import MAX_PLACEMENTS

    assert default_max_telomere("fnv") >= MAX_PLACEMENTS
    assert default_max_telomere("lut") == 8
    assert default_max_telomere("nervous") == 20
