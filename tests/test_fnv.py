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
    assert rank_key(small, 0.75) == rank_key(large, 0.75)

    large._topology_score = 4.0
    assert rank_key(large, 0.75) > rank_key(small, 0.75)
    assert _lexicase(
        [small, large], [(0.5, 0.5), (0.5, 0.5)]) is large
    small._topology_score = 100.0
    assert rank_key(large, 0.76) > rank_key(small, 0.75)


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
    from substrates.fnv.growth import active_gene_loci

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


def test_fnv_public_surface_has_no_inverse_development():
    import substrates.fnv as fnv

    assert not hasattr(fnv, "grid_to_genome_functional")
    assert not hasattr(fnv, "random_scaffold_genome")
    assert not hasattr(fnv, "phenotype_local_variants")


def test_checkpoint_roundtrip_checks_the_catalogue_hash():
    import tempfile
    from runtime.checkpoint import load_checkpoint, save_checkpoint
    from substrates.snn.targets import gate_target

    random.seed(7)
    genome = random_functional_genome(2, n_inputs=2)
    data = genome_to_dict(genome, "fnv")
    assert data["catalogue_hash"] == CATALOGUE_HASH
    assert data["development_version"] == 2
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
    expected_topology = [
        genome._topology_score for genome in population]
    for genome in population:
        genome._topology_score = -1.0
    eval_batch_cases(population, target, cache=cache)
    assert [
        genome._topology_score for genome in population
    ] == expected_topology


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
