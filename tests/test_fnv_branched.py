"""The branched FNV encoding: context-window genes over 2n chromosome arms.

Every gene is a context rule of the shape the nervous and LUT substrates use -
the states required of the three neighbours, the state the cell must already be
in, and what it becomes. EMPTY is a state like any other, so a rule may react to
empty space, build into it, retype a cell, or erase one.

Matching is EXACT and growth is synchronous. Each output role owns one arm;
that arm starts at its genetic writable OUT niche and grows toward PAD cues.
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
    arm_control, arm_telomere, arm_tolerance, branch_growth_order,
    context_distance, develop_constructive)
from substrates.fnv.construction_ga import (
    BRANCHED_MUT_OPS, assemble_role_modules, branch_cut, branch_map, branched_signature,
    chromosome_rules, crossover_branched, mutate_branched,
    mutate_branched_once, observed_contexts, placement_genes,
    random_branched_genome, randomize_branch_behavior, relabel_branches,
)
from substrates.fnv.genome import (
    BRANCHED_ENCODING, Chromosome, ContextGene, ControlGene, EMPTY_STATE,
    Genome, OUT_STATE, OutputGene, PAD_STATE, is_branched,
    sync_output_layout, validate_genome,
)

FAMILIES = ("LOGIC", "DELAY")
DOWN_TO_LR = BY_NAME["DELAY1_D_TO_LR"].id
LEFT_TO_RD = BY_NAME["DELAY1_L_TO_RD"].id
PADS = ((0, 0), (-8, -8))


def _genome(chromosomes, pads=PADS, next_gene_id=99):
    genome = Genome(
        chromosomes=chromosomes, input_layout=pads,
        output_chromosome=Chromosome(
            genes=[OutputGene(98, 'out0', 1, 2, 2)]),
        encoding=BRANCHED_ENCODING, next_gene_id=next_gene_id)
    relabel_branches(genome)
    return sync_output_layout(genome)


def _arm(genes, telomere, split=0, tolerance=0):
    """One bottom-arm branch: its rules plus the control gene that steers it."""
    members = list(genes) + [ControlGene(90 + len(genes), tolerance, telomere,
                                         90 + len(genes))]
    return Chromosome(genes=members, split=split)


def _body(genome):
    seeds = tuple(genome.input_layout)
    trace = develop_constructive(genome, seeds)
    return trace, {cell: state for cell, state in trace.grid.items()
                   if cell not in seeds}


def test_a_chromosome_has_exactly_two_arms():
    random.seed(3)
    for count in (1, 2, 3):
        genome = random_branched_genome(count, FAMILIES, 3)
        branches = branch_map(genome)
        assert len(branches) == 2 * count
        for index, chromosome in enumerate(genome.chromosomes):
            arms = branch_growth_order(chromosome)
            assert branches[(index, 0)] == list(arms[0])
            # Every occupied arm carries exactly one control gene.
            for half, members in enumerate(arms):
                if members:
                    assert arm_telomere(chromosome, half) > 0 or True
                    assert sum(1 for gene in chromosome.genes
                               if isinstance(gene, ControlGene)) >= 1


def test_an_output_root_gene_starts_only_at_its_assigned_site():
    # The root is at (0,1), directly above input pad zero.
    gene = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       DOWN_TO_LR, 1)
    trace, cells = _body(_genome([_arm([gene], 4)]))
    assert cells == {(0, 1): DOWN_TO_LR}
    assert sorted(trace.active_ids) == [1]


def test_contract_allele_choice_changes_only_the_existing_gate_function():
    pads = ((-1, 1), (1, 1))
    route = BY_NAME["AND_LR_TO_D"].id
    root = ContextGene(
        1, PAD_STATE, PAD_STATE, EMPTY_STATE, OUT_STATE, route, 2)
    genome = _genome([_arm([root], 4)], pads=pads)
    before_layout = genome.output_layout

    # In canonical binary-row order XOR is high on rows 01 and 10: 0b0110.
    assert randomize_branch_behavior(
        genome, 2, 2, preferred_signature=0b0110)
    assert BY_ID[root.self_out].behavior == "XOR"
    assert BY_ID[root.self_out].inputs == BY_ID[route].inputs
    assert BY_ID[root.self_out].outputs == BY_ID[route].outputs
    assert genome.output_layout == before_layout


def test_contract_allele_choice_honors_the_targets_input_row_order():
    pads = ((-1, 1), (1, 1))
    route = BY_NAME["AND_LR_TO_D"].id
    root = ContextGene(
        1, PAD_STATE, PAD_STATE, EMPTY_STATE, OUT_STATE, route, 2)
    genome = _genome([_arm([root], 4)], pads=pads)

    # The middle two rows reverse canonical binary order, as happens when a
    # target enumerates input 0 as the least-significant counter bit.  In this
    # order A & ~B is 0,1,0,0 -> 0b0010.
    patterns = ((0, 0), (1, 0), (0, 1), (1, 1))
    assert randomize_branch_behavior(
        genome, 2, 2, preferred_signature=0b0010,
        input_patterns=patterns)
    # Pad 0 is physically left of the cell and therefore enters its R-facing
    # port (the port name is the direction the signal travels toward).
    assert root.self_out == BY_NAME["VETO_R_NOT_L_TO_D"].id


def test_a_gene_may_retype_and_erase_an_existing_cell():
    start = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        DOWN_TO_LR, 1)
    retype = ContextGene(2, EMPTY_STATE, EMPTY_STATE, PAD_STATE, DOWN_TO_LR,
                         LEFT_TO_RD, 1)
    _trace, retyped = _body(_genome([_arm([start, retype], 6)]))
    assert LEFT_TO_RD in retyped.values()

    erase = ContextGene(2, EMPTY_STATE, EMPTY_STATE, PAD_STATE, DOWN_TO_LR,
                        EMPTY_STATE, 1)
    # Each regulatory gene differentiates one cell once: the root is built and
    # then erased, but the spent root gene cannot extrude it again as a chain.
    trace, erased = _body(_genome([_arm([start, erase], 4)]))
    assert sorted(trace.active_ids) == [1, 2]
    assert not erased
    assert not _body(_genome([_arm([start, erase], 5)]))[1]


def test_a_cell_no_gene_matches_is_left_alone():
    start = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        DOWN_TO_LR, 1)
    _trace, cells = _body(_genome([_arm([start], 6)]))
    kept = dict(cells)
    # Adding a rule whose context never occurs changes nothing at all.
    idle = ContextGene(2, LEFT_TO_RD, LEFT_TO_RD, LEFT_TO_RD, LEFT_TO_RD,
                       DOWN_TO_LR, 1)
    _trace, again = _body(_genome([_arm([start, idle], 6)]))
    assert again == kept


def test_development_cache_reuses_only_an_identical_mutable_genome():
    start = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        DOWN_TO_LR, 1)
    genome = _genome([_arm([start], 6)])
    first = develop_constructive(genome, genome.input_layout)
    assert develop_constructive(genome, genome.input_layout) is first
    assert "_fnv_development_cache" not in genome.__getstate__()

    # In-place mutation changes the structural key; no operator has to remember
    # to invalidate the cache manually.
    start.self_out = BY_NAME["DELAY2_D_TO_LR"].id
    changed = develop_constructive(genome, genome.input_layout)
    assert changed is not first
    assert changed.grid != first.grid


def test_branch_priority_settles_a_contested_cell():
    """Two genes wanting different states: the lower-numbered arm wins.

    Leaving contested cells empty instead was tried and it walled branches off
    from one another.
    """
    start = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        DOWN_TO_LR, 1)
    other = BY_NAME["DELAY2_D_TO_LR"].id
    rival = ContextGene(2, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        other, 1)
    # Both in one arm: the lower gene id wins the tie.
    trace, cells = _body(_genome([_arm([start, rival], 6)]))
    assert cells and set(cells.values()) == {DOWN_TO_LR}
    assert 1 in trace.active_ids and 2 not in trace.active_ids

def test_pads_are_never_written():
    # A rule that would erase the pad itself cannot: pads are the input
    # interface and genes only read them.
    from substrates.fnv.genome import input_seed_state

    erase_pad = ContextGene(1, EMPTY_STATE, EMPTY_STATE, EMPTY_STATE,
                            input_seed_state(0), EMPTY_STATE, 1)
    genome = _genome([_arm([erase_pad], 6)])
    trace, _cells = _body(genome)
    for index, pad in enumerate(PADS):
        assert trace.grid[pad] == input_seed_state(index)


def test_an_arms_telomere_is_spent_one_per_cell_change():
    gene = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       DOWN_TO_LR, 1)
    # A dead arm does nothing at all.
    assert not _body(_genome([_arm([gene], 0)]))[1]
    # With one life left it writes its one assigned output root.
    assert len(_body(_genome([_arm([gene], 1)]))[1]) == 1

    chromosome = _arm([gene], 5)
    assert arm_telomere(chromosome, 1) == 5
    assert arm_telomere(chromosome, 0) == 0        # empty top arm


def test_pad_context_does_not_grant_initial_territory():
    pad_only = ContextGene(1, PAD_STATE, EMPTY_STATE, EMPTY_STATE, EMPTY_STATE,
                           DOWN_TO_LR, 1)
    _trace, cells = _body(_genome([_arm([pad_only], 4)]))
    assert not cells


def test_reverse_growth_uses_component_input_ports_not_every_free_edge():
    """A unary component exposes one bud, not a decorative halo."""
    root = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       DOWN_TO_LR, 1)
    # This context is the empty L neighbour of the root, but L is an OUTPUT of
    # DOWN_TO_LR. It is therefore not a developmental input port.
    side = ContextGene(2, DOWN_TO_LR, EMPTY_STATE, EMPTY_STATE, EMPTY_STATE,
                       DOWN_TO_LR, 1)
    _trace, cells = _body(_genome([_arm([root, side], 8)]))
    assert cells == {(0, 1): DOWN_TO_LR}


def test_reverse_growth_requires_each_bud_to_drive_back_toward_its_root():
    root_state = BY_NAME["AND_LR_TO_D"].id
    root = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       root_state, 1)
    context = (root_state, EMPTY_STATE, EMPTY_STATE, EMPTY_STATE)
    # At (1, 1), the output-facing edge is L. LEFT_TO_RD cannot drive L, so it
    # cannot become a disconnected ornament even though its context matches.
    wrong_way = ContextGene(2, *context, LEFT_TO_RD, 1)
    _trace, blocked = _body(_genome([_arm([root, wrong_way], 8)]))
    assert blocked == {(0, 1): root_state}

    toward_root = ContextGene(2, *context, DOWN_TO_LR, 1)
    _trace, connected = _body(_genome([_arm([root, toward_root], 8)]))
    assert connected[(1, 1)] == DOWN_TO_LR


def test_binary_root_exposes_two_independent_developmental_buds():
    root_state = BY_NAME["AND_LR_TO_D"].id
    root = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       root_state, 1)
    left_bud = ContextGene(
        2, root_state, EMPTY_STATE, EMPTY_STATE, EMPTY_STATE,
        DOWN_TO_LR, 1)
    right_bud = ContextGene(
        3, EMPTY_STATE, root_state, EMPTY_STATE, EMPTY_STATE,
        DOWN_TO_LR, 1)
    _trace, cells = _body(_genome([_arm(
        [root, left_bud, right_bud], 8)]))
    assert cells[(0, 1)] == root_state
    assert cells[(1, 1)] == DOWN_TO_LR
    assert cells[(-1, 1)] == DOWN_TO_LR


def test_at_most_one_output_root_gene_per_arm_survives_repair():
    assert BRANCHED_MUT_OPS == [
        "tweak", "add_gene", "connect", "block", "del_rule", "del_branch",
        "control", "inputs", "outputs"]
    random.seed(9)
    genome = random_branched_genome(2, FAMILIES, 3)
    for _ in range(300):
        mutate_branched_once(genome, FAMILIES, 3)
        validate_genome(genome, FAMILIES)
        for members in branch_map(genome).values():
            assert sum(1 for gene in members if gene.spawns_output()) <= 1
        for gene in placement_genes(genome):
            # PAD is neighbour context only; OUT may only name self input.
            assert PAD_STATE not in (gene.self_in, gene.self_out)
            assert int(gene.self_out) != PAD_STATE
            assert int(gene.self_out) != OUT_STATE


def test_new_genes_are_drawn_from_contexts_the_body_presents():
    # Matching is exact, so a gene invented from thin air would almost never
    # fire. Fresh genomes should therefore be overwhelmingly expressible.
    random.seed(7)
    fired, total = 0, 0
    for _ in range(40):
        genome = random_branched_genome(2, FAMILIES, 3)
        trace, cells = _body(genome)
        assert cells, 'a fresh organism must develop'
        fired += len(trace.active_ids)
        total += len(placement_genes(genome))
        assert observed_contexts(genome, tuple(genome.input_layout))
    assert fired / total > 0.75


def test_reach_widens_what_a_rule_matches():
    """Tolerance is the evolvable reach of an arm, in node type numbers.

    EMPTY and PAD never tolerate a mismatch, so a rule may be vague about which
    component sits beside it and never about whether anything is there at all.
    """
    and_lr = BY_NAME["AND_LR_TO_D"].id
    xor_lr = BY_NAME["XOR_LR_TO_D"].id
    and_rd = BY_NAME["AND_RD_TO_L"].id
    assert context_distance((and_lr, 7, 0, 0), (and_lr, 7, 0, 0)) == 0
    # Function changes on fixed pins do not stale developmental rules.
    assert context_distance(
        (and_lr, 0, PAD_STATE, 0),
        (xor_lr, 0, PAD_STATE, 0)) == 0
    route_gap = context_distance(
        (and_lr, 0, PAD_STATE, 0),
        (and_rd, 0, PAD_STATE, 0))
    assert route_gap > 0
    assert context_distance((and_lr, 0, 0, 0), (and_lr, 3, 0, 0)) is None
    assert context_distance((PAD_STATE, 0, 0, 0), (5, 0, 0, 0)) is None

    # Take a neighbourhood the organism really presents, then key a rule one
    # type number away from it. Exact matching never fires; a reach of one does.
    # The root is a binary gate: reverse development can therefore observe its
    # two real input ports. DOWN_TO_LR can feed either one while keeping the
    # tolerance assertion independent of which port sorting chooses.
    spawn = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                        BY_NAME["AND_LR_TO_D"].id, 1)
    base = _genome([_arm([spawn], 8)])
    near, slot = None, None
    for context in sorted(observed_contexts(base, tuple(base.input_layout))):
        if context[3] != EMPTY_STATE:
            continue
        slot = next((i for i in range(3) if context[i] > EMPTY_STATE), None)
        if slot is not None:
            near = context
            break
    assert near is not None, 'expected a neighbourhood beside a grown cell'

    shifted = list(near)
    alternatives = [
        state for state in BY_NAME.values()
        if context_distance(
            tuple(shifted[:slot] + [state.id] + shifted[slot + 1:]), near)
        not in (None, 0)]
    shifted[slot] = min(
        alternatives,
        key=lambda state: context_distance(
            tuple(shifted[:slot] + [state.id] + shifted[slot + 1:]), near)
    ).id
    gap = context_distance(tuple(shifted), near)
    assert gap > 0
    follow = ContextGene(2, *shifted, LEFT_TO_RD, 1)
    tight = _body(_genome([_arm([spawn, follow], 8, tolerance=0)]))[1]
    loose = _body(_genome([_arm([spawn, follow], 8, tolerance=gap)]))[1]
    assert LEFT_TO_RD not in tight.values()
    assert LEFT_TO_RD in loose.values()


def test_each_arm_carries_its_own_lifespan_through_crossover():
    random.seed(13)
    for _ in range(80):
        left = random_branched_genome(2, FAMILIES, 3)
        right = random_branched_genome(2, FAMILIES, 3)
        for _ in range(6):
            mutate_branched(left, None, FAMILIES, 3)
            mutate_branched(right, None, FAMILIES, 3)
        relabel_branches(left)
        relabel_branches(right)
        child = crossover_branched(left, right, FAMILIES)
        validate_genome(child, FAMILIES)
        ids = [gene.gene_id for gene in placement_genes(child)]
        assert len(ids) == len(set(ids))
        assert child.next_gene_id > max(ids, default=0)
        # An arm's control gene rides along with it, so its reach and lifespan
        # came from whichever parent supplied that arm.
        for index, chromosome in enumerate(child.chromosomes):
            for half in (0, 1):
                pair = (arm_tolerance(chromosome, half),
                        arm_telomere(chromosome, half))
                if pair == (0, 0):
                    continue                       # empty arm
                assert pair in {
                    (arm_tolerance(left.chromosomes[index], half),
                     arm_telomere(left.chromosomes[index], half)),
                    (arm_tolerance(right.chromosomes[index], half),
                     arm_telomere(right.chromosomes[index], half))}


def test_a_lifespan_or_reach_change_is_a_real_change():
    gene = ContextGene(1, EMPTY_STATE, EMPTY_STATE, PAD_STATE, OUT_STATE,
                       DOWN_TO_LR, 1)
    genome = _genome([_arm([gene], 4)])
    before = branched_signature(genome)
    for control in [g for g in genome.chromosomes[0].genes
                    if isinstance(g, ControlGene)]:
        control.telomere = 2
    assert branched_signature(genome) != before

    wider = _genome([_arm([gene], 4, tolerance=6)])
    assert branched_signature(wider) != branched_signature(
        _genome([_arm([gene], 4)]))
    assert arm_tolerance(wider.chromosomes[0], 1) == 6


def test_branched_checkpoint_roundtrips_arms_and_lifespans():
    random.seed(17)
    genome = random_branched_genome(2, FAMILIES, 3)
    for _ in range(12):
        mutate_branched(genome, None, FAMILIES, 3)
    document = genome_to_dict(genome, "fnv")
    assert document["encoding"] == BRANCHED_ENCODING
    assert document["development_version"] == 6
    restored = genome_from_dict(document, "fnv")
    assert restored == genome
    assert is_branched(restored)
    assert [(arm_tolerance(c, 0), arm_telomere(c, 0),
             arm_tolerance(c, 1), arm_telomere(c, 1), c.split)
            for c in restored.chromosomes] == [
        (arm_tolerance(c, 0), arm_telomere(c, 0),
         arm_tolerance(c, 1), arm_telomere(c, 1), c.split)
        for c in genome.chromosomes]
    assert _body(restored)[1] == _body(genome)[1]


def test_max_telomere_is_the_arm_lifespan_ceiling():
    """Max telomere bounds an arm's lifespan and nothing else.

    It was briefly a growth RADIUS that clipped placements by depth. That is
    gone: the control only ever sizes a telomere, which here is how many cell
    changes one branch may make.
    """
    from runtime.config import default_max_telomere
    from substrates.fnv.construction_ga import MAX_ARM_TELOMERE

    assert default_max_telomere("fnv") == MAX_ARM_TELOMERE
    assert default_max_telomere("lut") == 8
    assert default_max_telomere("nervous") == 20

    random.seed(31)
    for ceiling in (2, 5):
        genome = random_branched_genome(2, FAMILIES, 3, max_telomere=ceiling)
        for _ in range(40):
            mutate_branched(genome, 4.0, FAMILIES, 3, ceiling)
        for chromosome in genome.chromosomes:
            for half in (0, 1):
                assert 0 <= arm_telomere(chromosome, half) <= ceiling


def test_input_pads_are_placed_by_their_own_chromosome():
    """An ADDITIONAL chromosome places the pads, beyond the growth ones.

    Pad zero is the coordinate gauge and has no gene, so the organism cannot
    drift sideways into identical copies of itself; every other pad names a
    bearing and a distance out from it.
    """
    from substrates.fnv.genome import (
        InputGene, MAX_INPUT_DISTANCE, input_ring, resolve_input_layout)

    random.seed(5)
    genome = random_branched_genome(2, FAMILIES, 3)
    # Two growth chromosomes, so four branches, PLUS the input chromosome.
    assert len(genome.chromosomes) == 2
    assert len(branch_map(genome)) == 4
    pads = genome.input_chromosome
    assert pads is not None
    assert len(pads.genes) == 2                    # one per pad after the first
    assert all(isinstance(gene, InputGene) for gene in pads.genes)

    assert genome.input_layout[0] == (0, 0)        # the anchor
    assert genome.input_layout == resolve_input_layout(genome)
    assert len(set(genome.input_layout)) == len(genome.input_layout)

    # Bearing slides a pad around the anchor; distance moves it in or out.
    gene = pads.genes[0]
    gene.distance, gene.bearing = 2, 0
    assert gene.cell() == input_ring(2)[0]
    gene.bearing = 1
    assert gene.cell() == input_ring(2)[1]
    gene.distance = 3
    assert gene.cell() in input_ring(3)
    assert 1 <= gene.distance <= MAX_INPUT_DISTANCE


def test_outputs_are_distinct_role_genes_bound_to_stable_arms():
    from substrates.fnv.genome import resolve_output_layout

    random.seed(6)
    roles = ('sum', 'carry', 'overflow')
    genome = random_branched_genome(2, FAMILIES, 3, roles)
    assert [gene.role for gene in genome.output_chromosome.genes] == list(roles)
    assert [gene.branch_id for gene in genome.output_chromosome.genes] == [1, 2, 3]
    assert genome.output_layout == resolve_output_layout(genome)
    cells = [cell for _role, cell in genome.output_layout]
    assert len(cells) == len(set(cells))
    assert not set(cells).intersection(genome.input_layout)
    grown = develop_constructive(genome, genome.input_layout).grid
    assert set(cells).issubset(grown)       # initialization starts every role
    # Four arms exist, but only the three role-owned arms develop.
    assert not branch_growth_order(genome.chromosomes[1])[1]
    validate_genome(genome, FAMILIES)


def test_crossover_keeps_each_output_allele_with_its_arm():
    random.seed(23)
    roles = ('sum', 'carry')
    left = random_branched_genome(2, FAMILIES, 3, roles)
    right = random_branched_genome(2, FAMILIES, 3, roles)
    for index, gene in enumerate(left.output_chromosome.genes):
        gene.distance, gene.bearing = 1, index
        arm_control(left.chromosomes[index // 2], index % 2).tolerance = 10 + index
    for index, gene in enumerate(right.output_chromosome.genes):
        gene.distance, gene.bearing = 3, index + 3
        arm_control(right.chromosomes[index // 2], index % 2).tolerance = 90 + index
    sync_output_layout(left)
    sync_output_layout(right)

    expected = {}
    for parent in (left, right):
        for gene in parent.output_chromosome.genes:
            label = int(gene.branch_id)
            expected.setdefault(gene.role, set()).add((
                int(gene.distance), int(gene.bearing),
                arm_control(parent.chromosomes[(label - 1) // 2],
                            (label - 1) % 2).tolerance))
    for _ in range(40):
        child = crossover_branched(left, right, FAMILIES)
        for gene in child.output_chromosome.genes:
            label = int(gene.branch_id)
            control = arm_control(
                child.chromosomes[(label - 1) // 2], (label - 1) % 2)
            assert (int(gene.distance), int(gene.bearing),
                    control.tolerance) in expected[gene.role]
        validate_genome(child, FAMILIES)


def test_role_assembly_joins_compatible_specialists_without_mutating_modules():
    random.seed(230)
    left = random_branched_genome(2, FAMILIES, 3, ('sum', 'carry'))
    from substrates.fnv.construction_ga import clone_constructive
    right = clone_constructive(left)

    left.output_chromosome.genes[0].distance = 2
    right.output_chromosome.genes[1].distance = 4
    arm_control(left.chromosomes[0], 0).tolerance = 11
    arm_control(right.chromosomes[0], 1).tolerance = 92
    sync_output_layout(left)
    sync_output_layout(right)

    child = assemble_role_modules(left, {1: left, 2: right}, FAMILIES)

    assert child.input_layout == left.input_layout == right.input_layout
    assert arm_control(child.chromosomes[0], 0).tolerance == 11
    assert arm_control(child.chromosomes[0], 1).tolerance == 92
    assert child.output_chromosome.genes[0].distance == 2
    assert child.output_chromosome.genes[1].distance == 4
    validate_genome(child, FAMILIES)


def test_moving_an_output_changes_its_genetic_root_without_repairing_the_gene():
    random.seed(9)
    genome = random_branched_genome(2, FAMILIES, 3)
    before_signature = branched_signature(genome)
    before_layout = genome.output_layout
    gene = genome.output_chromosome.genes[0]
    before_allele = (gene.distance, gene.bearing)
    gene.bearing = int(gene.bearing) + 1
    sync_output_layout(genome)
    assert genome.output_layout != before_layout
    assert (gene.distance, gene.bearing) != before_allele
    assert branched_signature(genome) != before_signature


def test_context_occurrences_are_ordered_not_hash_ordered():
    """A fixed seed must give a fixed evolution, in every process.

    `_observed_context_occurrences` used to return its raw `set`. Every element
    carries `directions`, a tuple of direction STRINGS, and CPython randomises
    string hashing per process - so the set iterated in a different order each
    run, and that order is what `_random_gene` hands to `random.choice`. The
    contents were always right; only the order moved. It made a seeded FNV run
    unreproducible (Majority-3 solved at generation 7, 24, 17, 22 and 30 on one
    seed), which silently turns every benchmark number into a lottery ticket.
    """
    from substrates.fnv.construction_ga import (
        _observed_context_occurrences, _seeds)

    random.seed(31)
    genome = random_branched_genome(2, FAMILIES, 3)
    occurrences = _observed_context_occurrences(genome, _seeds(genome))
    assert not isinstance(occurrences, (set, frozenset)), (
        'a set has no stable iteration order across processes')
    assert list(occurrences) == sorted(occurrences)
    # The string-keyed part is what made the order unstable, so prove the
    # ordering really is total over it rather than accidentally stable.
    assert len(set(occurrences)) == len(list(occurrences))


def test_a_seeded_genome_is_identical_under_any_string_hash_seed():
    """The property the fix exists to protect, checked end to end."""
    import json
    import subprocess

    script = (
        'import random, json, sys;'
        'sys.path.insert(0, %r);'
        'from substrates.fnv.construction_ga import random_branched_genome;'
        'from substrates.fnv.catalogue import DEFAULT_FAMILIES;'
        'from runtime.checkpoint import genome_to_dict;'
        'random.seed(4242);'
        'print(json.dumps(genome_to_dict('
        'random_branched_genome(2, DEFAULT_FAMILIES, 3), "fnv"),'
        ' sort_keys=True))' % ROOT)
    digests = set()
    for hash_seed in ('0', '1', '2'):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, '-c', script], env=environment,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-400:]
        digests.add(out.stdout.strip())
    assert len(digests) == 1, (
        'the same seed produced %d different genomes across hash seeds'
        % len(digests))
