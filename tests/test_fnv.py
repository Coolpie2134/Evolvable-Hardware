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
from substrates.fnv.ga import (
    _contract_input_dependencies, mutate_functional, mutate_input_layout,
    rank_key)
from substrates.fnv.genome import (
    Genome, Chromosome, functional_input_positions, random_functional_genome,
)
from substrates.fnv.construction import grow_functional
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


def test_fnv_selection_does_not_reward_an_equivalent_longer_chain():
    from substrates.fnv.evaluation import functional_topology
    from substrates.fnv.ga import _topology_tuple

    short = {
        (0, 0): SOURCE_STATE,
        (1, 0): BY_NAME["DELAY1_R_TO_L"].id,
    }
    long = {
        **short,
        (2, 0): BY_NAME["DELAY1_L_TO_R"].id,
    }
    short_topology = functional_topology(
        short, [(0, 0)], output_positions={"out": (1, 0)})
    long_topology = functional_topology(
        long, [(0, 0)], output_positions={"out": (2, 0)})
    assert long_topology.reachable_nodes > short_topology.reachable_nodes
    assert _topology_tuple(long_topology) == _topology_tuple(short_topology)


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


def test_fnv_topology_credits_reverse_grown_output_cones_before_completion():
    from substrates.fnv.evaluation import functional_topology

    root = (0, 0)
    left, right = (-1, 0), (1, 0)
    grid = {
        left: SOURCE_STATE,
        right: SOURCE_STATE,
        root: BY_NAME["AND_LR_TO_D"].id,
    }
    topology = functional_topology(
        grid, [left, right], output_positions={"sum": root})
    assert topology.live_output_roots == 1
    assert topology.output_cone_nodes == 1
    assert topology.output_cone_edges == 2
    assert topology.output_integrating_nodes == 1
    assert topology.min_output_input_convergence == 2
    assert topology.total_output_input_connections == 2
    assert topology.min_output_input_edges == 2
    assert topology.total_output_input_edges == 2
    assert topology.min_output_branch_input_edges == 1
    assert topology.total_output_branch_input_edges == 2
    assert topology.min_output_branch_input_convergence == 1
    assert topology.total_output_branch_input_convergence == 2

    # A second unwritten role is explicitly priced as an incomplete circuit.
    missing = functional_topology(
        grid, [left, right],
        output_positions={"sum": root, "carry": (8, 8)})
    assert missing.live_output_roots == 1
    assert missing.min_output_input_convergence == 0


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
    from substrates.fnv.genome import OUT_STATE, PAD_STATE, ControlGene

    for chromosome in genome.chromosomes:
        for gene in chromosome.genes:
            if isinstance(gene, ControlGene):
                continue          # reach and lifespan, not component states
            for field in ("ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"):
                state = getattr(gene, field)
                # PAD is a reserved context value, not a component, and only a
                # context field may carry it.
                if state == PAD_STATE:
                    assert field in ("ctx_l", "ctx_r", "ctx_d")
                    continue
                if state == OUT_STATE:
                    assert field == "self_in"
                    continue
                assert state == 0 or BY_ID[state].family in families


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


def test_fnv_public_surface_has_no_inverse_development():
    import substrates.fnv as fnv

    assert not hasattr(fnv, "grid_to_genome_functional")
    assert not hasattr(fnv, "random_scaffold_genome")
    assert not hasattr(fnv, "phenotype_local_variants")


def test_run_config_roundtrips_family_bank_selection():
    # A pre-readout checkpoint keeps the fitted-probe behavior it was saved
    # under even though fresh runs now default to genetic output sites.
    config = RunConfig(fnv=FNVConfig(("LOGIC", "TOGGLE"), "fitted"))
    loaded = RunConfig.from_dict({
        "ga": {},
        "pulse": {},
        "fnv": {"families": ["LOGIC", "TOGGLE"]},
    })
    assert config.fnv == loaded.fnv
    _assert_raises(ValueError, lambda: FNVConfig(()))
    _assert_raises(ValueError, lambda: FNVConfig(("FREE_OSC",)))


def test_fnv_readout_mode_validates_and_roundtrips():
    config = RunConfig(fnv=FNVConfig(("LOGIC", "DELAY"), "genetic"))
    loaded = RunConfig.from_dict({
        "ga": {}, "pulse": {},
        "fnv": {"families": ["LOGIC", "DELAY"],
                "readout_mode": "genetic"},
    })
    assert loaded.fnv == config.fnv
    assert FNVConfig().readout_mode == "genetic"
    assert RunConfig.from_dict({"ga": {}, "pulse": {}}).fnv.readout_mode == "fitted"
    _assert_raises(
        ValueError, lambda: FNVConfig(("LOGIC",), "best-looking-cell"))


def test_static_role_dependencies_follow_logical_input_identity_not_row_bits():
    from substrates.snn.targets import TARGETS

    target = TARGETS["2-bit adder"]
    assert _contract_input_dependencies(target, 0) == (0, 2)
    assert _contract_input_dependencies(target, 1) == (0, 1, 2, 3)
    assert _contract_input_dependencies(target, 2) == (0, 1, 2, 3)


def test_dead_genetic_output_is_silent_but_keeps_its_role_site():
    from substrates.fnv.evaluation import prepare_functional
    from substrates.snn.targets import gate_target

    target = gate_target("AND", grid_size=5)
    roles = tuple(terminal.role for terminal in target.outputs)
    genome = random_functional_genome(
        2, n_inputs=target.n_inputs, output_roles=roles,
        families=("LOGIC", "DELAY"))
    for chromosome in genome.chromosomes:
        chromosome.genes = []
        chromosome.split = 0
    target._fnv_readout_mode = "genetic"
    prepared = prepare_functional(genome, target)
    assert prepared is not None
    _grid, _inputs, outputs, cases = prepared
    assert outputs == dict(genome.output_layout)
    assert all(case["acts"][roles[0]] == 0 for case in cases)

    target._fnv_readout_mode = "fitted"
    assert prepare_functional(genome, target) is None


def test_fnv_development_ignores_target_io_coordinates():
    """Only logical role/count metadata enters FNV; target geometry does not."""
    import copy
    from substrates.fnv.evaluation import prepare_functional
    from substrates.snn.targets import gate_target

    target = gate_target("XOR", grid_size=7)
    roles = tuple(terminal.role for terminal in target.outputs)
    random.seed(188)
    genome = random_functional_genome(
        2, n_inputs=target.n_inputs, output_roles=roles,
        families=("LOGIC", "DELAY"))
    target._fnv_readout_mode = "genetic"
    moved = copy.deepcopy(target)
    moved.inputs = [(100 + index, -100) for index in range(moved.n_inputs)]
    for index, terminal in enumerate(moved.outputs):
        terminal.pos = (-100, 100 + index)
    moved._fnv_readout_mode = "genetic"

    original = prepare_functional(genome, target)
    relocated = prepare_functional(genome, moved)
    assert original is not None and relocated is not None
    assert original[:3] == relocated[:3]
    assert original[3] == relocated[3]


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


def test_fnv_logic_behavior_mutations_keep_the_same_physical_pins():
    from substrates.fnv.catalogue import behavior_component_ids

    source = BY_NAME["AND_LR_TO_D"]
    alternatives = [BY_ID[index]
                    for index in behavior_component_ids(source.id)]
    assert {entry.behavior for entry in alternatives} >= {"OR", "XOR"}
    assert all(set(entry.inputs) == set(source.inputs)
               for entry in alternatives)
    assert all(entry.outputs == source.outputs for entry in alternatives)


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
            families=("LOGIC", "DELAY"),
            output_roles=tuple(t.role for t in target.outputs))
        scored = prepare_functional(genome, target)
        if scored is None:
            continue
        grid, inputs, outputs, horizon, branches = prepare_functional_playback(
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
    target = gate_target("AND", grid_size=5)
    population = [
        random_functional_genome(
            2, max_telomere=3, families=("LOGIC", "DELAY"),
            output_roles=tuple(t.role for t in target.outputs))
        for _ in range(3)
    ]
    cache = {}
    fitnesses, cases = eval_batch_cases(
        population, target, cache=cache)
    assert len(fitnesses) == len(cases) == 3
    assert all(0.0 <= fitness <= 1.0 for fitness in fitnesses)
    assert all(case_vector is not None for case_vector in cases)
    assert all(
        isinstance(genome._topology_score, float)
        and genome._topology_score >= 0.0
        for genome in population)
    assert all(len(genome._topology_rank) == 25 for genome in population)
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
