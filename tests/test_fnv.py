import copy
import os
import random
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime.checkpoint import genome_from_dict, genome_to_dict
from runtime.config import FNVConfig, RunConfig
from substrates.fnv.catalogue import (
    BY_ID, BY_NAME, CATALOGUE_HASH, COMPONENTS, FAMILIES,
    NODE_TYPE_DICTIONARY,
    enabled_component_ids,
)
from substrates.fnv.ga import mutate_functional, mutate_input_layout, rank_key
from substrates.fnv.genome import (
    FunctionalGene, Genome, Chromosome, functional_input_positions,
    random_functional_genome,
)
from substrates.fnv.growth import grow_functional
from substrates.fnv.simulation import FunctionalSim


SOURCE_STATE = BY_NAME["DELAY1_D_TO_LR"].id


def _assert_raises(error_type, call, text=None):
    try:
        call()
    except error_type as exc:
        if text is not None:
            assert text in str(exc)
        return
    raise AssertionError("expected %s" % error_type.__name__)


def _single(name, pulses, horizon=10):
    grid = {
        (-1, 0): SOURCE_STATE,
        (0, 0): BY_NAME[name].id,
        (1, 0): SOURCE_STATE,
    }
    sim = FunctionalSim(
        grid, input_nodes={cell for cell, _, _ in pulses}, max_events=100)
    for cell, start, width in pulses:
        sim.inject_pulse(cell, start, width)
    sim.advance_to(horizon)
    return sim


def test_catalogue_is_permanent_complete_and_family_first_selectable():
    assert len(COMPONENTS) == 118
    assert COMPONENTS[0].name == "EMPTY"
    assert COMPONENTS[-1].id == 117
    assert NODE_TYPE_DICTIONARY == {
        entry.id: entry.name for entry in COMPONENTS}
    assert NODE_TYPE_DICTIONARY[0] == "EMPTY"
    assert NODE_TYPE_DICTIONARY[117] == COMPONENTS[-1].name
    assert CATALOGUE_HASH == (
        "8487a4ddca738efb0e843cc186fb4c56cb166179cac6ca4afbe586f2eb44dd28")
    assert len([entry for entry in COMPONENTS
                if entry.family == "LOGIC"]) == 15
    assert len([entry for entry in COMPONENTS
                if entry.family == "DELAY"]) == 18
    assert len([entry for entry in COMPONENTS
                if entry.family == "NORMALIZER"]) == 18
    assert len([entry for entry in COMPONENTS
                if entry.family == "HOLD"]) == 18
    assert len([entry for entry in COMPONENTS
                if entry.family == "C_ELEMENT"]) == 3
    assert len([entry for entry in COMPONENTS
                if entry.family == "TOGGLE"]) == 9
    assert len([entry for entry in COMPONENTS
                if entry.family == "GATED_OSCILLATOR"]) == 36
    assert not any(entry.behavior in {
        "NOT", "NAND", "NOR", "FREE_OSC", "FILTER", "TAP_DELAY"
    } for entry in COMPONENTS)
    selected = enabled_component_ids(("TOGGLE",))
    assert len(selected) == 9
    assert all(BY_ID[state].family == "TOGGLE" for state in selected)


def test_all_catalogue_types_are_quiescent_without_input():
    for entry in COMPONENTS[1:]:
        sim = FunctionalSim({(0, 0): entry.id})
        sim.advance_to(50)
        assert sim.levels[(0, 0)] == 0, entry.name
        assert sim.rise_times[(0, 0)] == [], entry.name
        assert not sim.overflow


def test_logic_delay_normalizer_hold_toggle_and_gated_oscillator():
    sim = _single(
        "AND_LR_TO_D",
        [((-1, 0), 0, 5), ((1, 0), 0, 5)])
    assert sim.rise_times[(0, 0)] == [1.0]
    assert sim.fall_times[(0, 0)] == [6.0]

    sim = _single("DELAY2_L_TO_R", [((-1, 0), 0.25, 0.20)])
    assert sim.rise_times[(0, 0)] == [2.25]
    assert sim.fall_times[(0, 0)] == [2.45]

    sim = _single("NORMALIZER2_L_TO_R", [((-1, 0), 0.25, 5.0)])
    assert sim.rise_times[(0, 0)] == [1.25]
    assert sim.fall_times[(0, 0)] == [3.25]

    sim = _single("HOLD2_L_TO_R", [((-1, 0), 0.25, 0.20)])
    assert sim.rise_times[(0, 0)] == [1.25]
    assert sim.fall_times[(0, 0)] == [3.45]

    sim = _single(
        "TOGGLE_L_TO_R",
        [((-1, 0), 0.25, 0.20), ((-1, 0), 2.0, 0.20)])
    assert sim.rise_times[(0, 0)] == [1.25]
    assert sim.fall_times[(0, 0)] == [3.0]

    sim = _single("GOSC_H1_L2_L_TO_R", [((-1, 0), 0.25, 6.0)])
    assert sim.rise_times[(0, 0)][:2] == [1.25, 4.25]
    assert sim.fall_times[(0, 0)][:2] == [2.25, 5.25]
    assert sim.levels[(0, 0)] == 0


def test_fnv_bit_parallel_static_path_matches_physical_basic_gate_replay():
    from substrates.fnv.evaluation import (
        _bit_parallel_logic_runs, _run_logic_case)
    from substrates.fnv.simulation import compile_functional_grid
    from substrates.snn.targets import get_target

    target = get_target("AND")
    inputs = ((-1, 0), (1, 0))
    grid = {cell: SOURCE_STATE for cell in inputs}
    grid[(0, 0)] = BY_NAME["AND_LR_TO_D"].id
    circuit = compile_functional_grid(grid, inputs)
    fast = _bit_parallel_logic_runs(grid, inputs, target, circuit)
    genome = Genome([Chromosome([FunctionalGene()], telomere=3)])
    assert fast is not None
    for fast_run, (bits, _expected) in zip(fast, target.cases):
        physical = _run_logic_case(
            genome, grid, inputs, bits, target, _compiled=circuit)
        assert fast_run[1][(0, 0)] == physical[1][(0, 0)]


def test_directed_antiparallel_wires_require_the_facing_output_port():
    external = (-2, 0)
    middle = (-1, 0)
    receiver = (0, 0)
    base = {
        external: SOURCE_STATE,
        receiver: BY_NAME["DELAY1_L_TO_R"].id,
    }
    connected = dict(base, **{})  # keep the coordinate-keyed mapping explicit
    connected[middle] = BY_NAME["DELAY1_R_TO_L"].id
    sim = FunctionalSim(connected, input_nodes={external})
    sim.inject_pulse(external, 0, 1)
    sim.advance_to(5)
    assert sim.rise_times[receiver] == [2.0]

    disconnected = dict(base)
    disconnected[middle] = BY_NAME["DELAY1_R_TO_D"].id
    sim = FunctionalSim(disconnected, input_nodes={external})
    sim.inject_pulse(external, 0, 1)
    sim.advance_to(5)
    assert sim.rise_times[receiver] == []


def test_input_pads_are_source_only_and_only_external_injection_reactivates_them():
    external, pad, receiver = (-1, 0), (0, 0), (1, 0)
    grid = {
        external: SOURCE_STATE,
        pad: BY_NAME["DELAY1_L_TO_R"].id,
        receiver: BY_NAME["DELAY1_R_TO_L"].id,
    }
    sim = FunctionalSim(grid, input_nodes={external, pad})
    sim.inject_pulse(external, 0.0, 1.0)
    sim.advance_to(3.0)
    assert sim.rise_times[pad] == []
    assert sim.rise_times[receiver] == []

    sim.inject_pulse(pad, 3.0, 1.0)
    sim.advance_to(6.0)
    assert sim.rise_times[pad] == [3.0]
    assert sim.rise_times[receiver] == [4.0]


def test_fnv_topology_counts_only_input_reachable_connections_and_loops():
    from substrates.fnv.evaluation import functional_topology

    chain = {
        (0, 0): SOURCE_STATE,
        (1, 0): BY_NAME["DELAY1_R_TO_L"].id,
        (2, 0): BY_NAME["DELAY1_L_TO_R"].id,
    }
    chain_topology = functional_topology(chain, [(0, 0)])
    assert chain_topology.reachable_nodes == 2
    assert chain_topology.reachable_edges == 2
    assert chain_topology.cyclic_nodes == 0
    assert chain_topology.loop_rank == 0

    # A six-component directed honeycomb ring, entered through an OR so the
    # source pulse and circulating pulse can share the first component.
    loop = {
        (0, 0): SOURCE_STATE,
        (1, 0): BY_NAME["OR_RD_TO_L"].id,
        (2, 0): BY_NAME["DELAY1_L_TO_R"].id,
        (3, 0): BY_NAME["DELAY1_R_TO_D"].id,
        (3, -1): BY_NAME["DELAY1_D_TO_L"].id,
        (2, -1): BY_NAME["DELAY1_L_TO_R"].id,
        (1, -1): BY_NAME["DELAY1_R_TO_D"].id,
    }
    loop_topology = functional_topology(loop, [(0, 0)])
    assert loop_topology.reachable_nodes == 6
    assert loop_topology.reachable_edges == 7
    assert loop_topology.cyclic_nodes == 6
    assert loop_topology.loop_rank == 1
    assert loop_topology.loop_regions == 1
    assert loop_topology.score > chain_topology.score

    unreachable = dict(loop)
    unreachable[(10, 10)] = unreachable.pop((0, 0))
    disconnected_topology = functional_topology(
        unreachable, [(10, 10)])
    assert disconnected_topology.reachable_nodes == 0
    assert disconnected_topology.reachable_edges == 0
    assert disconnected_topology.loop_rank == 0


def test_fnv_topology_recognizes_multi_input_convergence():
    from substrates.fnv.evaluation import functional_topology

    grid = {
        (-1, 0): SOURCE_STATE,
        (1, 0): SOURCE_STATE,
        (0, 0): BY_NAME["AND_LR_TO_D"].id,
    }
    topology = functional_topology(grid, [(-1, 0), (1, 0)])
    assert topology.reachable_nodes == 1
    assert topology.reachable_edges == 2
    assert topology.integrating_nodes == 1
    assert topology.max_input_convergence == 2
    assert topology.fully_integrating_nodes == 1


def test_fnv_logic_repertoire_rewards_input_dependence_without_target_answers():
    from substrates.fnv.evaluation import logic_behavior_diversity
    from substrates.snn.targets import get_target

    target = get_target("Full adder")
    observations = []
    for bits, _expected in target.cases:
        a, b, c = bits
        observations.append({
            "node_levels": {
                (0, 0): a,
                (1, 0): a ^ b,
                (2, 0): a ^ b ^ c,
                (3, 0): 0,
            }
        })
    distinct, multi, full, max_inputs = logic_behavior_diversity(
        observations, target)
    assert distinct == 3
    assert multi == 2
    assert full == 1
    assert max_inputs == 3


def test_fnv_rank_has_no_gene_count_or_telomere_preference():
    from substrates.fnv.ga import _lexicase

    small = Genome(
        [Chromosome([FunctionalGene()], telomere=1)])
    large = Genome(
        [Chromosome([FunctionalGene() for _ in range(20)], telomere=20)])
    for genome in (small, large):
        genome._robustness = 0.25
        genome._juvenile_score = 0.5
        genome._topology_score = 3.0
        genome._topology_rank = (1, 1, 1, 1, 0, 0, 0)
    assert rank_key(small, 0.75) == rank_key(large, 0.75)

    large._topology_score = 4.0
    large._topology_rank = (2, 1, 1, 1, 0, 0, 0)
    assert rank_key(large, 0.75) > rank_key(small, 0.75)
    assert _lexicase(
        [small, large], [(0.5, 0.5), (0.5, 0.5)]) is large
    small._topology_score = 100.0
    small._topology_rank = (3, 1, 1, 1, 0, 0, 0)
    assert rank_key(large, 0.76) > rank_key(small, 0.75)


def test_fnv_binary_lexicase_filters_exactly_on_a_fifty_fifty_split():
    from substrates.fnv.ga import _lexicase

    population = [Genome([Chromosome([FunctionalGene()], telomere=1)])
                  for _ in range(4)]
    for genome in population:
        genome._topology_score = 0.0
    cases = [(1.0,), (1.0,), (0.0,), (0.0,)]
    random.seed(82)
    winners = {id(_lexicase(population, cases)) for _ in range(80)}
    assert winners == {id(population[0]), id(population[1])}


def test_fnv_static_selection_cases_preserve_outputs_and_joint_rows():
    from substrates.fnv.ga import _selection_case_vector
    from substrates.snn.targets import get_target

    target = get_target("Half adder")
    # Row-major cells for (Sum, Carry): Sum is perfect; Carry misses its only
    # asserted row. The extra views must not alter the executable base cells.
    cells = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    selected = _selection_case_vector(cells, target)
    assert selected[:len(cells)] == cells
    assert selected[len(cells):len(cells) + 2] == (1.0, 0.5)
    assert selected[len(cells) + 2:len(cells) + 6] == (
        1.0, 1.0, 1.0, 0.0)
    assert selected[-1] == 0.5


def test_fnv_temporal_selection_cases_are_not_reinterpreted():
    from substrates.fnv.ga import _selection_case_vector
    from substrates.nervous.targets import TEMPORAL_TARGETS

    cases = (0.1, 0.7, 1.0)
    assert _selection_case_vector(
        cases, TEMPORAL_TARGETS["Period stepper"]) == cases


def test_normalizer_does_not_retrigger_while_held_and_rearms_after_inactive():
    sim = _single(
        "NORMALIZER1_L_TO_R",
        [((-1, 0), 0.0, 4.0), ((-1, 0), 6.0, 0.25)],
        horizon=10)
    assert sim.rise_times[(0, 0)] == [1.0, 7.0]
    assert sim.fall_times[(0, 0)] == [2.0, 8.0]


def test_c_element_holds_on_disagreement_and_cancels_unsettled_agreement():
    sim = _single(
        "C_ELEMENT_LR_TO_D",
        [((-1, 0), 0.0, 2.0), ((1, 0), 0.0, 4.0)],
        horizon=7)
    assert sim.rise_times[(0, 0)] == [1.0]
    assert sim.fall_times[(0, 0)] == [5.0]

    sim = _single(
        "C_ELEMENT_LR_TO_D",
        [((-1, 0), 0.0, 3.0), ((1, 0), 0.0, 0.5)],
        horizon=5)
    assert sim.rise_times[(0, 0)] == []


def test_family_restricted_mutation_never_introduces_a_disabled_component():
    random.seed(34)
    families = ("LOGIC", "DELAY")
    genome = random_functional_genome(2, families=families)
    for _ in range(80):
        mutate_functional(
            genome, 4.0, chromosome_count=2, families=families)
    for chromosome in genome.chromosomes:
        for gene in chromosome.genes:
            if hasattr(gene, "component_id"):
                assert BY_ID[gene.component_id].family in families
                continue
            for field in ("ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"):
                state = getattr(gene, field)
                assert state == 0 or BY_ID[state].family in families


def test_growth_is_bounded_by_telomere_and_keeps_the_germline():
    # This rule grows the delay component into every frontier context it wins.
    gene = FunctionalGene(
        ctx_l=SOURCE_STATE, ctx_r=0, ctx_d=0,
        self_in=0, self_out=SOURCE_STATE)
    genome = Genome([Chromosome([gene], telomere=2)], tag=1)
    body = grow_functional(genome, [(0, 0)])
    assert (0, 0) in body
    assert 1 <= len(body) <= 10
    assert max(abs(x) + abs(y) for x, y in body) <= 2


def test_input_germlines_have_four_fixed_physical_identities():
    from substrates.fnv.genome import input_seed_grid

    seeds = [(0, 1), (0, 3), (0, 5), (0, 7)]
    palette = input_seed_grid(seeds)
    assert len(set(palette.values())) == 4
    assert all(
        BY_ID[state].family == "DELAY"
        and len(BY_ID[state].outputs) == 2
        for state in palette.values())


def test_fnv_input_layouts_are_distinct_relative_genes_with_local_mutation():
    from substrates.nervous.hexgrid import hex_frontier_cells

    random.seed(61)
    genome = random_functional_genome(
        2, max_telomere=5, n_inputs=4)
    original = genome.input_layout
    assert original[0] == (0, 0)
    assert len(original) == len(set(original)) == 4

    assert mutate_input_layout(genome, max_telomere=5)
    changed = genome.input_layout
    assert changed[0] == (0, 0)
    assert len(changed) == len(set(changed)) == 4
    moved = [
        index for index, (before, after) in enumerate(zip(original, changed))
        if before != after
    ]
    assert len(moved) == 1
    assert moved[0] != 0
    assert changed[moved[0]] in hex_frontier_cells(*original[moved[0]])

def test_evolved_fnv_inputs_override_target_coordinates_but_legacy_inputs_do_not():
    from substrates.fnv.evaluation import prepare_functional
    from substrates.snn.targets import gate_target

    target = gate_target("AND", grid_size=5)
    layout = ((0, 0), (2, 1))
    genome = Genome(
        [Chromosome([FunctionalGene()], telomere=1)],
        input_layout=layout)
    grown = {
        layout[0]: SOURCE_STATE,
        layout[1]: SOURCE_STATE,
        (1, 0): SOURCE_STATE,
    }
    with (
        mock.patch(
            "substrates.fnv.evaluation.grow_functional",
            return_value=grown) as grow,
        mock.patch(
            "substrates.fnv.evaluation.place_logic_outputs",
            return_value=({target.outputs[0].role: (1, 0)}, [])),
    ):
        prepared = prepare_functional(genome, target)
    assert prepared[1] == list(layout)
    assert grow.call_args.args[1] == list(layout)

    legacy = Genome([Chromosome([FunctionalGene()], telomere=1)])
    assert functional_input_positions(
        legacy, target.inputs) == tuple(target.inputs)


def test_output_assignment_is_global_distinct_and_not_role_greedy():
    # One implementation, shared by both asynchronous substrates so the probe
    # selection rule cannot drift between them.
    from substrates.nervous.scoring import (
        best_distinct_assignment, behavior_representatives,
    )
    import substrates.fnv.evaluation as fnv_evaluation
    import substrates.nervous.temporal as nv_temporal
    assert fnv_evaluation.best_distinct_assignment is best_distinct_assignment
    assert nv_temporal.best_distinct_assignment is best_distinct_assignment

    roles = ("sum", "carry")
    cells = ((1, 0), (2, 0))
    scores = {
        "sum": {(1, 0): 10, (2, 0): 9},
        "carry": {(1, 0): 10, (2, 0): 0},
    }
    # Greedy would give "sum" its best cell (1, 0) and strand "carry" on a zero.
    assert best_distinct_assignment(roles, cells, scores) == {
        "sum": (2, 0),
        "carry": (1, 0),
    }
    # Plain sum prefers a perfect easy output plus a weak hard output. The
    # truth-table contract's mean-and-worst objective prefers balanced probes,
    # and fitted output selection must make that same choice.
    balanced_scores = {
        "sum": {(1, 0): 1.0, (2, 0): 0.69},
        "carry": {(1, 0): 0.69, (2, 0): 0.40},
    }
    assert best_distinct_assignment(roles, cells, balanced_scores) == {
        "sum": (1, 0), "carry": (2, 0)}
    assert best_distinct_assignment(
        roles, cells, balanced_scores, balance_worst=True) == {
            "sum": (2, 0), "carry": (1, 0)}
    assert behavior_representatives(
        ((1, 0), (2, 0), (3, 0)),
        lambda _cell: "same response",
        multiplicity=2,
    ) == ((1, 0), (2, 0))


def test_fnv_output_fitting_can_select_a_node_beyond_the_old_probe_radius():
    from substrates.fnv.evaluation import place_logic_outputs
    from substrates.snn.targets import gate_target

    target = gate_target("AND", grid_size=5)
    genome = Genome([Chromosome([FunctionalGene()], telomere=1)])
    input_cells = list(target.inputs)
    far = (20, 0)
    grid = {cell: SOURCE_STATE for cell in input_cells}
    grid.update({(index, 0): SOURCE_STATE for index in range(1, 21)})

    def fake_run(_genome, _grid, _inputs, bits, _target):
        expected = int(all(bits))
        values = {
            cell: 1 - expected
            for cell in grid
            if cell not in input_cells
        }
        values[far] = expected
        return values, dict(values), False

    with mock.patch(
            "substrates.fnv.evaluation._run_logic_case",
            side_effect=fake_run):
        positions, _ = place_logic_outputs(
            genome, grid, input_cells, target)
    assert tuple(positions.values()) == (far,)


def test_expression_aware_mutation_identifies_live_rules():
    from substrates.fnv.growth import (
        active_gene_loci, active_gene_loci_and_contexts)

    child_state = BY_NAME["DELAY1_L_TO_R"].id
    genome = Genome([Chromosome([
        FunctionalGene(
            ctx_l=SOURCE_STATE, ctx_r=0, ctx_d=0,
            self_in=0, self_out=child_state),
        FunctionalGene(
            ctx_l=SOURCE_STATE, ctx_r=0, ctx_d=0,
            self_in=child_state, self_out=child_state),
    ], telomere=2)])
    loci = active_gene_loci(genome, [(0, 0)])
    assert (0, 0) in loci
    assert loci
    same_loci, contexts = active_gene_loci_and_contexts(genome, [(0, 0)])
    assert same_loci == loci
    assert contexts


def test_fnv_observed_context_mutation_is_reachable_but_target_blind():
    from substrates.fnv.ga import _mutate_observed_context

    genome = Genome([Chromosome([
        FunctionalGene(self_out=BY_NAME["AND_LR_TO_D"].id)
    ], telomere=2)])
    context = (BY_NAME["AND_LR_TO_D"].id, 0, 0, 0)
    random.seed(95)
    assert _mutate_observed_context(
        genome, (context,), frozenset(("LOGIC", "DELAY")),
        ("LOGIC", "DELAY"))
    inserted = genome.chromosomes[0].genes[0]
    assert (inserted.ctx_l, inserted.ctx_r, inserted.ctx_d,
            inserted.self_in) == context
    assert BY_ID[inserted.self_out].family in ("LOGIC", "DELAY")


def test_fnv_public_surface_has_no_inverse_development():
    import substrates.fnv as fnv

    assert not hasattr(fnv, "grid_to_genome_functional")
    assert not hasattr(fnv, "random_scaffold_genome")
    assert not hasattr(fnv, "phenotype_local_variants")


def test_context_specialization_splits_a_basic_gate_without_rewriting_source():
    from substrates.fnv.ga import _specialize_observed_context
    from substrates.fnv.growth import lookup

    source = FunctionalGene(
        ctx_l=0, ctx_r=0, ctx_d=0, self_in=0,
        self_out=BY_NAME["AND_LR_TO_D"].id)
    later = FunctionalGene(
        ctx_l=SOURCE_STATE, ctx_r=SOURCE_STATE, ctx_d=0, self_in=0,
        self_out=BY_NAME["VETO_L_NOT_R_TO_D"].id)
    genome = Genome([
        Chromosome([source], telomere=2),
        Chromosome([later], telomere=2),
    ])
    context = (SOURCE_STATE, SOURCE_STATE, 0, 0)
    replacement = BY_NAME["OR_LR_TO_D"].id
    assert _specialize_observed_context(
        genome, context, ("LOGIC",), replacement=replacement)
    # Priority is genome-wide, so the split must lead chromosome zero rather
    # than merely lead a randomly selected later chromosome.
    split = genome.chromosomes[0].genes[0]
    assert tuple(getattr(split, field) for field in (
        "ctx_l", "ctx_r", "ctx_d", "self_in")) == context
    assert split.self_out == replacement
    assert genome.chromosomes[0].genes[1] == source
    assert lookup(genome, *context) == replacement


def test_checkpoint_roundtrip_checks_the_catalogue_hash():
    import tempfile
    from runtime.checkpoint import load_checkpoint, save_checkpoint
    from substrates.snn.targets import gate_target

    random.seed(7)
    genome = random_functional_genome(2, n_inputs=2)
    data = genome_to_dict(genome, "fnv")
    assert data["catalogue_hash"] == CATALOGUE_HASH
    assert data["development_version"] == 3
    assert data["encoding"] == "constructive_v3"
    assert data["gene_fields"] == []
    assert data["input_layout"] == [
        list(cell) for cell in genome.input_layout]
    assert genome_from_dict(data, "fnv") == genome
    legacy = copy.deepcopy(data)
    del legacy["input_layout"]
    assert genome_from_dict(legacy, "fnv").input_layout is None
    changed = copy.deepcopy(data)
    changed["catalogue_hash"] = "wrong"
    _assert_raises(
        ValueError, lambda: genome_from_dict(changed, "fnv"),
        "catalogue hash mismatch")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "fnv.json")
        config = RunConfig(fnv=FNVConfig(("LOGIC", "DELAY")))
        save_checkpoint(
            path, genome, 0.5, gate_target("AND"), None, 7, "fnv",
            config)
        loaded = load_checkpoint(path)
        assert loaded["backend"] == "fnv"
        assert loaded["best_genome"] == genome
        assert loaded["run_config"].fnv.families == ("LOGIC", "DELAY")


def test_run_config_roundtrips_family_bank_selection():
    config = RunConfig(fnv=FNVConfig(("LOGIC", "TOGGLE")))
    loaded = RunConfig.from_dict({
        "ga": {},
        "pulse": {},
        "fnv": {"families": ["LOGIC", "TOGGLE"]},
    })
    assert config.fnv == loaded.fnv
    _assert_raises(ValueError, lambda: FNVConfig(()))
    _assert_raises(ValueError, lambda: FNVConfig(("FREE_OSC",)))


def test_native_baseline_is_informational_and_respects_family_bank():
    from substrates.fnv.evaluation import native_component_baseline
    from substrates.nervous.targets import echo, toggle_ff
    from substrates.snn.targets import gate_target

    assert native_component_baseline(
        gate_target("XOR"), ("LOGIC",)) == "XOR"
    assert native_component_baseline(
        gate_target("XOR"), ("DELAY",)) is None
    assert native_component_baseline(
        echo(delay=2), ("DELAY",)) == "DELAY2"
    assert native_component_baseline(
        echo(delay=3), ("DELAY",)) is None
    assert native_component_baseline(
        toggle_ff(), ("TOGGLE",)) == "TOGGLE"


def test_combinational_initialization_focuses_logic_without_narrowing_bank():
    from substrates.fnv.ga import initialization_families
    from substrates.nervous.targets import echo, periodic_combinational_target
    from substrates.snn.targets import get_target

    all_families = tuple(FAMILIES)
    logic_target = periodic_combinational_target(get_target("Half adder"))
    assert initialization_families(
        all_families, logic_target) == frozenset(("LOGIC", "DELAY"))
    assert initialization_families(
        ("LOGIC", "HOLD"), logic_target) == frozenset(("LOGIC",))
    assert initialization_families(
        all_families, echo(delay=2)) == frozenset(all_families)
    # A user-selected logic-free physical bank is respected exactly.
    assert initialization_families(
        ("DELAY", "HOLD"), logic_target) == frozenset(
            ("DELAY", "HOLD"))


def test_focused_fnv_allele_mutation_is_a_bias_not_a_family_ban():
    from substrates.fnv.ga import _mutate_allele

    logic = BY_NAME["AND_LR_TO_D"].id
    delay = BY_NAME["DELAY1_L_TO_R"].id
    genome = Genome([Chromosome([
        FunctionalGene(self_in=logic, self_out=logic)
    ], telomere=1)])
    locus = ((0, 0),)
    with mock.patch(
            "substrates.fnv.ga.random.random",
            side_effect=[0.0, 0.0, 0.0, 0.0]):
        _mutate_allele(
            genome, frozenset(FAMILIES), locus, locus, ("LOGIC",))
    assert BY_ID[genome.chromosomes[0].genes[0].self_out].family == "LOGIC"

    genome.chromosomes[0].genes[0].self_out = logic
    with (
        mock.patch(
            "substrates.fnv.ga.random.random",
            side_effect=[0.0, 0.0, 1.0, 1.0]),
        mock.patch(
            "substrates.fnv.ga.random_component_id", return_value=delay),
    ):
        _mutate_allele(
            genome, frozenset(FAMILIES), locus, (), ("LOGIC",))
    assert genome.chromosomes[0].genes[0].self_out == delay


def test_fnv_crossover_can_preserve_whole_developmental_modules():
    from substrates.fnv.ga import crossover_functional

    def chromosome(tag, state):
        return Chromosome([
            FunctionalGene(self_in=state, self_out=state)
        ], tag=tag, telomere=2)

    logic = BY_NAME["AND_LR_TO_D"].id
    delay = BY_NAME["DELAY1_L_TO_R"].id
    left = Genome([chromosome(1, logic), chromosome(2, logic)])
    right = Genome([chromosome(3, delay), chromosome(4, delay)])
    with (
        mock.patch("substrates.fnv.ga.random.random", return_value=0.5),
        mock.patch("substrates.fnv.ga.random.randint", return_value=1),
        mock.patch("substrates.fnv.ga.random.sample", return_value=[0]),
    ):
        child = crossover_functional(left, right)
    assert child.chromosomes[0] == right.chromosomes[0]
    assert child.chromosomes[0] is not right.chromosomes[0]
    assert child.chromosomes[1] == left.chromosomes[1]


def test_fnv_preserves_a_target_blind_morphology_reserve():
    from substrates.fnv.ga import next_population

    population = [Genome([Chromosome([
        FunctionalGene(self_out=BY_NAME["AND_LR_TO_D"].id)
    ], tag=index, telomere=2)]) for index in range(10)]
    for index, genome in enumerate(population):
        genome._topology_rank = (index, index, index, 0, 0, 0, 0)
    children = next_population(
        population, [0.5] * len(population),
        make_genome=lambda: Genome([Chromosome([FunctionalGene()])]),
        case_vecs=[(0.5,)] * len(population), mean_mutations=0.0,
        selection="lexicase", chromosome_count=1,
        recombination=False, families=("LOGIC",))
    assert children[0] is not population[-1]
    assert children[0].chromosomes[0].tag == population[-1].chromosomes[0].tag


def test_fnv_logic_behavior_mutations_keep_the_same_physical_pins():
    from substrates.fnv.catalogue import behavior_component_ids

    source = BY_NAME["AND_LR_TO_D"]
    alternatives = [BY_ID[index]
                    for index in behavior_component_ids(source.id)]
    assert {entry.behavior for entry in alternatives} >= {"OR", "XOR"}
    assert all(entry.inputs == source.inputs for entry in alternatives)
    assert all(entry.outputs == source.outputs for entry in alternatives)


def test_fnv_gene_duplication_diverges_context_and_behavior_immediately():
    from substrates.fnv.ga import _diverged_duplicate

    source = FunctionalGene(
        ctx_l=BY_NAME["AND_LR_TO_D"].id,
        self_out=BY_NAME["AND_LR_TO_D"].id)
    random.seed(94)
    duplicate = _diverged_duplicate(
        source, frozenset(("LOGIC", "DELAY")), ("LOGIC", "DELAY"))
    assert duplicate is not source
    assert tuple(getattr(duplicate, field) for field in
                 ("ctx_l", "ctx_r", "ctx_d", "self_in")) != tuple(
                     getattr(source, field) for field in
                     ("ctx_l", "ctx_r", "ctx_d", "self_in"))
    assert duplicate.self_out != source.self_out


def test_fnv_observed_context_mutation_connects_to_occupied_sides():
    from substrates.fnv.ga import _contextual_component_id

    source = BY_NAME["DELAY1_D_TO_LR"].id
    random.seed(104)
    for _ in range(30):
        state = BY_ID[_contextual_component_id(
            source, source, 0, 0, ("LOGIC", "DELAY"))]
        assert set(state.inputs) == {"L", "R"}
        assert state.family == "LOGIC"
        assert state.outputs == ("D",)

    for _ in range(30):
        state = BY_ID[_contextual_component_id(
            source, 0, 0, 0, ("LOGIC", "DELAY"))]
        assert state.family == "DELAY"
        assert state.inputs == ("L",)
        assert "L" not in state.outputs


def test_context_growth_counts_only_physically_driven_facing_wires():
    from substrates.fnv.ga import _driven_context_directions

    drives_toward_receiver = BY_NAME["DELAY1_D_TO_LR"].id
    drives_away_from_receiver = BY_NAME["HOLD1_L_TO_R"].id
    assert _driven_context_directions(
        drives_toward_receiver, 0, 0) == frozenset(("L",))
    assert _driven_context_directions(
        drives_away_from_receiver, 0, 0) == frozenset()
    # Runtime input membership overrides the printed seed component's ports;
    # all fixed seed identities therefore advertise full source fanout during
    # development too.
    assert _driven_context_directions(
        BY_NAME["DELAY1_L_TO_RD"].id, 0, 0) == frozenset(("L",))


def test_fnv_plateau_rescue_enumerates_expressed_gate_neighbours():
    from substrates.fnv.ga import genome_signature, plateau_rescue_candidates

    state = BY_NAME["AND_LR_TO_D"].id
    genome = Genome([Chromosome([
        FunctionalGene(self_in=state, self_out=state)
    ], telomere=2)])
    with mock.patch(
            "substrates.fnv.growth.active_gene_loci_and_contexts",
            return_value=(((0, 0),), ())):
        random.seed(106)
        candidates = plateau_rescue_candidates(
            genome, limit=12, max_telomere=4,
            families=("LOGIC", "DELAY"), focus_families=("LOGIC",))
    assert candidates
    assert len(candidates) <= 12
    assert len({genome_signature(candidate)
                for candidate in candidates}) == len(candidates)
    replacements = {
        candidate.chromosomes[0].genes[0].self_out
        for candidate in candidates}
    assert replacements.intersection(
        {entry.id for entry in COMPONENTS
         if entry.family == "LOGIC" and entry.behavior in {"OR", "XOR"}})
    assert all(candidate.chromosomes[0].genes[0].self_out == 0
               or BY_ID[candidate.chromosomes[0].genes[0].self_out].family
               in {"LOGIC", "DELAY"}
               for candidate in candidates)


def test_fitted_readout_is_reused_without_refitting():
    from substrates.nervous.evaluation import fit_readout, score_frozen
    from substrates.nervous.targets import echo

    child_state = BY_NAME["DELAY1_R_TO_L"].id
    genome = Genome([Chromosome([
        FunctionalGene(
            ctx_l=0, ctx_r=SOURCE_STATE, ctx_d=0,
            self_in=0, self_out=child_state),
        FunctionalGene(
            ctx_l=0, ctx_r=SOURCE_STATE, ctx_d=0,
            self_in=child_state, self_out=child_state),
    ], telomere=2)], input_layout=((0, 0),))
    target = echo(delay=2)
    fitted = fit_readout(genome, target, backend="fnv")
    assert fitted is not None
    assert fitted.backend == "fnv"
    assert fitted.inputs == ((0, 0),)
    assert score_frozen(genome, target, fitted) == fitted.training_score


def test_fnv_interactive_temporal_playback_matches_scored_waveforms():
    from substrates.fnv.evaluation import prepare_functional
    from substrates.fnv.playback import (
        FunctionalPlayer, prepare_functional_playback)
    from substrates.nervous.playback import pulses_from_trial
    from substrates.nervous.targets import echo

    child_state = BY_NAME["DELAY1_R_TO_L"].id
    genome = Genome([Chromosome([
        FunctionalGene(
            ctx_l=0, ctx_r=SOURCE_STATE, ctx_d=0,
            self_in=0, self_out=child_state),
        FunctionalGene(
            ctx_l=0, ctx_r=SOURCE_STATE, ctx_d=0,
            self_in=child_state, self_out=child_state),
    ], telomere=2)], input_layout=((0, 0),))
    target = echo(delay=2)
    scored = prepare_functional(genome, target)
    playback = prepare_functional_playback(genome, target)
    assert playback is not None
    grid, inputs, outputs, horizon = playback
    assert (grid, inputs, outputs) == scored[:3]

    for trial_index in range(len(target.trials)):
        player = FunctionalPlayer(
            grid, inputs, outputs.values(), horizon=horizon,
            max_events=target.max_events)
        lanes = pulses_from_trial(target, len(inputs), trial_index)
        player.set_schedule({
            cell: list(lanes[index])
            for index, cell in enumerate(inputs)
        })
        player.sim.advance_to(horizon)
        for role, cell in outputs.items():
            observed = [
                tuple(interval)
                for interval in player.sim.pulse_intervals[cell]
            ]
            assert observed == scored[3].intervals[role][trial_index]


def test_fnv_interactive_logic_cases_use_the_exact_fitness_hold_window():
    from substrates.fnv.evaluation import prepare_functional
    from substrates.fnv.playback import (
        FunctionalPlayer, functional_case_pulses,
        prepare_functional_playback)
    from substrates.snn.targets import gate_target

    target = gate_target("AND", grid_size=5)
    random.seed(73)
    for _ in range(40):
        genome = random_functional_genome(
            2, max_telomere=3, n_inputs=target.n_inputs,
            families=("LOGIC", "DELAY"))
        scored = prepare_functional(genome, target)
        if scored is None:
            continue
        grid, inputs, outputs, horizon = prepare_functional_playback(
            genome, target)
        assert (grid, inputs, outputs) == scored[:3]
        for case_index, (bits, _expected) in enumerate(target.cases):
            lanes = functional_case_pulses(
                target, len(inputs), horizon, case_index)
            for bit, lane in zip(bits, lanes):
                assert lane == ([(0.0, horizon)] if bit else [])
            player = FunctionalPlayer(
                grid, inputs, outputs.values(), horizon=horizon,
                max_events=getattr(target, "max_events", 2048))
            player.set_schedule({
                cell: list(lanes[index])
                for index, cell in enumerate(inputs)
            })
            player.sim.advance_to(horizon)
            for terminal in target.outputs:
                # Combinational fitness scores the settled level at the end of
                # the hold window. Reconvergent real-delay paths may glitch
                # earlier, so ``ever`` is intentionally not the contract.
                assert int(player.sim.levels[outputs[terminal.role]]) == int(
                    scored[3][case_index]["acts"][terminal.role])
        return
    raise AssertionError("no random FNV genome produced a playable circuit")


def test_parallel_population_evaluation_returns_case_vectors():
    from substrates.fnv.ga import eval_batch_cases
    from substrates.snn.targets import gate_target
    random.seed(19)
    population = [
        random_functional_genome(
            2, max_telomere=3, families=("LOGIC", "DELAY"))
        for _ in range(3)
    ]
    cache = {}
    target = gate_target("AND", grid_size=5)
    fitnesses, cases = eval_batch_cases(
        population, target, cache=cache)
    assert len(fitnesses) == len(cases) == 3
    assert all(0.0 <= fitness <= 1.0 for fitness in fitnesses)
    assert all(case_vector is not None for case_vector in cases)
    assert all(
        isinstance(genome._topology_score, float)
        and genome._topology_score >= 0.0
        for genome in population)
    assert all(len(genome._topology_rank) == 7 for genome in population)
    assert all(len(genome._behavior_diagnostic) == 4
               for genome in population)
    expected_topology = [
        genome._topology_score for genome in population]
    for genome in population:
        genome._topology_score = -1.0
    eval_batch_cases(population, target, cache=cache)
    assert [
        genome._topology_score for genome in population
    ] == expected_topology


def test_fnv_behavior_repertoire_is_diagnostic_not_a_selection_tier():
    from substrates.fnv.ga import rank_key

    left = Genome([Chromosome([FunctionalGene()])])
    right = Genome([Chromosome([FunctionalGene()])])
    shared_topology = (3, 2, 2, 1, 1, 0, 0)
    left._topology_rank = shared_topology
    right._topology_rank = shared_topology
    left._behavior_diagnostic = (0, 0, 0, 0)
    right._behavior_diagnostic = (99, 99, 99, 99)

    assert rank_key(left, 0.5) == rank_key(right, 0.5)


def test_constructive_fnv_resolves_named_dependencies_not_gene_order():
    from substrates.fnv.genome import (
        CONSTRUCTIVE_ENCODING, BranchRef, PlacementGene)

    first = PlacementGene(
        10, BY_NAME["DELAY1_R_TO_LD"].id,
        (BranchRef(-1, "R"),), 10)
    second = PlacementGene(
        11, BY_NAME["DELAY1_L_TO_R"].id,
        (BranchRef(10, "L"),), 10)
    forward = Genome(
        [Chromosome([first, second], telomere=1)],
        input_layout=((0, 0),), encoding=CONSTRUCTIVE_ENCODING,
        next_gene_id=12)
    reversed_genes = Genome(
        [Chromosome([second, first], telomere=1)],
        input_layout=((0, 0),), encoding=CONSTRUCTIVE_ENCODING,
        next_gene_id=12)

    expected = {
        (0, 0): SOURCE_STATE,
        (1, 0): first.component_id,
        (2, 0): second.component_id,
    }
    assert grow_functional(forward, forward.input_layout) == expected
    assert grow_functional(reversed_genes, reversed_genes.input_layout) == expected


def test_constructive_fnv_collision_fails_only_that_branch():
    from substrates.fnv.construction import develop_constructive
    from substrates.fnv.genome import (
        CONSTRUCTIVE_ENCODING, BranchRef, PlacementGene)

    winner = PlacementGene(
        1, BY_NAME["DELAY1_R_TO_LD"].id,
        (BranchRef(-1, "R"),), 1)
    collision = PlacementGene(
        2, BY_NAME["HOLD1_R_TO_L"].id,
        (BranchRef(-1, "R"),), 2)
    independent = PlacementGene(
        3, BY_NAME["DELAY1_R_TO_LD"].id,
        (BranchRef(-2, "R"),), 3)
    genome = Genome(
        [Chromosome([collision, independent, winner], telomere=1)],
        input_layout=((0, 0), (0, 2)),
        encoding=CONSTRUCTIVE_ENCODING, next_gene_id=4)

    trace = develop_constructive(genome, genome.input_layout)
    assert trace.grid[(1, 0)] == winner.component_id
    assert trace.grid[(1, 2)] == independent.component_id
    assert trace.active_ids == frozenset((1, 3))
    assert 2 in trace.dormant_ids


def test_constructive_fanout_promotion_preserves_the_existing_branch():
    from substrates.fnv.construction import develop_constructive
    from substrates.fnv.construction_ga import (
        _fanout_options, _install_fanout_option, placement_genes)
    from substrates.fnv.genome import (
        CONSTRUCTIVE_ENCODING, BranchRef, PlacementGene)

    gate = PlacementGene(
        1, BY_NAME["AND_RD_TO_L"].id,
        (BranchRef(-1, "R"), BranchRef(-2, "D")), 1)
    consumer = PlacementGene(
        2, BY_NAME["DELAY1_L_TO_R"].id,
        (BranchRef(1, "L"),), 2)
    descendant = PlacementGene(
        3, BY_NAME["DELAY1_R_TO_L"].id,
        (BranchRef(2, "R"),), 2)
    independent = PlacementGene(
        4, BY_NAME["DELAY1_R_TO_LD"].id,
        (BranchRef(-3, "R"),), 4)
    genome = Genome(
        [Chromosome([gate, consumer, descendant, independent], telomere=1)],
        input_layout=((0, 0), (1, -1), (0, 2)),
        encoding=CONSTRUCTIVE_ENCODING, next_gene_id=5)

    option = next(option for option in _fanout_options(
        genome, genome.input_layout, ("LOGIC", "DELAY"))
                  if option[0] == 2)
    assert _install_fanout_option(genome, genome.input_layout, option)
    genes = {gene.gene_id: gene for gene in placement_genes(genome)}
    assert set(genes) == {1, 2, 3, 4}
    replacement = genes[2]
    assert BY_ID[replacement.component_id].family == "DELAY"
    assert len(BY_ID[replacement.component_id].outputs) == 2
    assert BY_ID[replacement.component_id].duration == 1
    assert replacement.inputs == consumer.inputs
    trace = develop_constructive(genome, genome.input_layout)
    assert trace.active_ids == frozenset((1, 2, 3, 4))


def test_constructive_cascade_mutation_changes_two_pin_compatible_gates():
    from substrates.fnv.construction import develop_constructive
    from substrates.fnv.construction_ga import (
        _install_logic_cascade_option, _logic_cascade_options,
        placement_genes)
    from substrates.fnv.genome import (
        CONSTRUCTIVE_ENCODING, BranchRef, PlacementGene)

    upstream = PlacementGene(
        1, BY_NAME["AND_LR_TO_D"].id,
        (BranchRef(-2, "L"), BranchRef(-1, "R")), 1)
    fanout = PlacementGene(
        2, BY_NAME["DELAY1_D_TO_LR"].id,
        (BranchRef(1, "D"),), 1)
    downstream = PlacementGene(
        3, BY_NAME["XOR_RD_TO_L"].id,
        (BranchRef(2, "R"), BranchRef(-3, "D")), 3)
    genome = Genome(
        [Chromosome([upstream, fanout, downstream], telomere=1)],
        input_layout=((0, 0), (2, 0), (2, -2)),
        encoding=CONSTRUCTIVE_ENCODING, next_gene_id=4)

    desired = (
        1, BY_NAME["OR_LR_TO_D"].id,
        3, BY_NAME["AND_RD_TO_L"].id)
    options = _logic_cascade_options(
        genome, genome.input_layout, ("LOGIC", "DELAY"))
    assert desired in options
    assert _install_logic_cascade_option(
        genome, genome.input_layout, desired)
    genes = {gene.gene_id: gene for gene in placement_genes(genome)}
    assert genes[1].component_id == BY_NAME["OR_LR_TO_D"].id
    assert genes[2] == fanout
    assert genes[3].component_id == BY_NAME["AND_RD_TO_L"].id
    assert develop_constructive(
        genome, genome.input_layout).active_ids == frozenset((1, 2, 3))


def test_constructive_plateau_rescue_enumerates_feasible_live_tip_joins():
    from substrates.fnv.construction import develop_constructive
    from substrates.fnv.construction_ga import (
        plateau_candidates_constructive, placement_genes)
    from substrates.fnv.genome import CONSTRUCTIVE_ENCODING

    # Both external source pads face the same empty cell.  Every physical
    # two-input LOGIC variant for those pins must be offered before random
    # plateau proposals, even though there are many unary frontier choices.
    genome = Genome(
        [Chromosome([], telomere=1)],
        input_layout=((-1, 0), (1, 0)),
        encoding=CONSTRUCTIVE_ENCODING, next_gene_id=1)
    candidates = plateau_candidates_constructive(
        genome, limit=5, families=("LOGIC", "DELAY"))

    assert len(candidates) == 5
    component_ids = set()
    for candidate in candidates:
        genes = placement_genes(candidate)
        assert len(genes) == 2
        assert BY_ID[genes[0].component_id].family == "LOGIC"
        assert len(BY_ID[genes[0].component_id].inputs) == 2
        assert BY_ID[genes[1].component_id].family == "DELAY"
        assert len(BY_ID[genes[1].component_id].outputs) == 2
        assert genes[1].inputs[0].node_id == genes[0].gene_id
        assert develop_constructive(
            candidate, candidate.input_layout).coordinates[1] == (0, 0)
        component_ids.add(genes[0].component_id)
    assert len(component_ids) == 5


def test_constructive_bridge_is_an_explicit_target_blind_component_path():
    from substrates.fnv.construction import develop_constructive
    from substrates.fnv.construction_ga import (
        _bridge_live_tips, placement_genes)
    from substrates.fnv.genome import CONSTRUCTIVE_ENCODING

    # These source pads have no common neighbouring cell.  A bridge mutation
    # must encode the route as ordinary numbered DELAY node(s) followed by one
    # ordinary numbered two-input LOGIC node.
    genome = Genome(
        [Chromosome([], telomere=1)],
        input_layout=((0, 0), (3, 0)),
        encoding=CONSTRUCTIVE_ENCODING, next_gene_id=1)
    random.seed(401)
    assert _bridge_live_tips(
        genome, genome.input_layout, ("LOGIC", "DELAY"))

    genes = placement_genes(genome)
    logic = [gene for gene in genes
             if BY_ID[gene.component_id].family == "LOGIC"]
    delays = [gene for gene in genes
              if BY_ID[gene.component_id].family == "DELAY"]
    assert len(logic) == 1
    assert len(delays) == len(genes) - 1
    assert len(BY_ID[logic[0].component_id].inputs) == 2
    assert len(BY_ID[genes[-1].component_id].outputs) == 2
    assert genes[-1].inputs[0].node_id == logic[0].gene_id
    trace = develop_constructive(genome, genome.input_layout)
    assert trace.active_ids == frozenset(gene.gene_id for gene in genes)


def test_constructive_checkpoint_keeps_labels_and_legacy_v2_still_loads():
    from substrates.fnv.genome import is_constructive

    fresh = random_functional_genome(
        2, families=("LOGIC", "DELAY"), n_inputs=3)
    document = genome_to_dict(fresh, "fnv")
    assert isinstance(document["chromosomes"][0]["genes"][0], dict)
    restored = genome_from_dict(document, "fnv")
    assert restored == fresh
    assert is_constructive(restored)

    legacy = Genome([Chromosome([
        FunctionalGene(self_in=0, self_out=SOURCE_STATE)
    ], telomere=2)], input_layout=((0, 0),))
    old_document = genome_to_dict(legacy, "fnv")
    assert old_document["development_version"] == 2
    old_restored = genome_from_dict(old_document, "fnv")
    assert old_restored == legacy
    assert not is_constructive(old_restored)


def test_incomplete_fnv_circuit_pads_every_lexicase_contract_case():
    from substrates.fnv.evaluation import evaluate_functional_full
    from substrates.fnv.ga import _lexicase
    from substrates.nervous.scoring import contract_case_count
    from substrates.nervous.targets import TEMPORAL_TARGETS

    target = TEMPORAL_TARGETS["Period stepper"]
    genome = Genome([Chromosome([FunctionalGene()], telomere=1)])
    fitness, cases = evaluate_functional_full(genome, target)
    assert fitness == 0.0
    assert cases == (0.0,) * contract_case_count(target)
    assert _lexicase([genome, genome], [cases, cases]) is genome


def test_runtime_controller_runs_and_saves_an_fnv_generation():
    import queue
    import tempfile
    import threading

    from runtime.checkpoint import load_checkpoint
    from runtime.config import GAConfig
    from runtime.controller import LATEST_POPULATION_NAME, run_evolution
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import gate_target

    messages = queue.Queue()
    config = RunConfig(
        ga=GAConfig(
            mean_mutations=2.0, chromosome_count=2, max_telomere=3),
        fnv=FNVConfig(("LOGIC", "DELAY")),
    )
    with tempfile.TemporaryDirectory() as results:
        run_evolution(
            1, 3, 2, 1,
            periodic_combinational_target(gate_target("AND", grid_size=5)),
            None,
            messages, threading.Event(), base_seed=9, backend="fnv",
            run_config=config, results_dir=results)
        population_path = os.path.join(results, LATEST_POPULATION_NAME)
        assert os.path.exists(population_path)
        saved = load_checkpoint(population_path)
        assert all(
            genome.input_layout is not None
            and len(genome.input_layout) == 2
            for genome in saved["genomes"])
    received = []
    while not messages.empty():
        received.append(messages.get())
    done = [message for message in received if message[0] == "done"]
    errors = [message for message in received if message[0] == "error"]
    certified = [
        message for message in received if message[0] == "certified"]
    assert len(done) == 1
    assert done[0][1] is not None
    assert len(certified) == 1
    assert "verdict" in certified[0][1]
    assert not errors


if __name__ == "__main__":
    for _name, _test in sorted(globals().copy().items()):
        if _name.startswith("test_") and callable(_test):
            _test()
            print("PASS", _name)
