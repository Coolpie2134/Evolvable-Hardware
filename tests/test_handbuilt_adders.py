"""Feasibility fixtures for the two hard Full Adder substrates."""
import random

from runtime.config import nv_run_config
from substrates.lut.branched import (
    develop_branched_lut, materialise_pads, output_root_sites,
    root_source_counts, table_family, table_support)
from substrates.lut.branched_ga import _nudge_cell, input_pads
from substrates.lut.ga import (
    _lut_structural_topology, _output_module_scores, evaluate_lut_full)
from substrates.lut.functions import FAMILY_TABLES
from substrates.lut.functions import INPUT_TABLES
from substrates.lut.lut import SEED_STATE
from substrates.lut.pulse import AsyncLutSim
from substrates.nervous.branched_ga import (
    _nudge_tile, input_pads as nervous_input_pads,
    plateau_rescue_candidates as nervous_plateau_rescue_candidates)
from substrates.nervous.branched import (
    develop_branched_hex,
    driven_roots as nervous_driven_roots,
    root_source_counts as nervous_root_source_counts)
from substrates.nervous.branched_synthesis import (
    synthesize_branched_full_adder as synthesize_nervous_full_adder)
from substrates.nervous.certification import certify
from substrates.nervous.ga import evaluate_nv_full
from substrates.nervous.hexgrid import ROUTING_HEX
from substrates.nervous.simulation import create_simulator
from substrates.nervous.targets import coincident_temporal_target
from substrates.nervous.temporal import nervous_topology
from substrates.nervous.tritile import (
    TRI_SEED_STATE, channel_configs, pack_channels)
from tools.probe_handbuilt_adders import (
    direct_tri_tile_capability, handbuilt_branched_lut_full_adder,
    handbuilt_nervous_full_adder)
from substrates.snn.targets import get_target


def test_live_branched_lut_encoding_can_grow_a_perfect_full_adder():
    result = handbuilt_branched_lut_full_adder()
    assert result['score'] == 1.0
    assert result['worst_case'] == 1.0
    assert result['cells'] == 6
    assert result['genes'] == 6
    pads = input_pads(result['genome'])
    body = develop_branched_lut(result['genome'], pads).grid
    grid = materialise_pads(body, pads)
    roots = output_root_sites(result['genome'], pads)
    assert root_source_counts(grid, pads, roots) == {1: 3, 2: 3}
    _fitness, _cases, prepared = evaluate_lut_full(
        result['genome'], result['target'], _return_prepared=True)
    assert _output_module_scores(_cases, result['target']) == (1.0, 1.0)
    topology = _lut_structural_topology(prepared)
    assert topology.integrating_nodes == 2


def test_paper_analog_nodes_can_compute_and_generalise_full_adder():
    result = handbuilt_nervous_full_adder()
    assert result['score'] == 1.0
    assert result['worst_case'] == 1.0
    assert not result['overflow']


def test_live_branched_nervous_genome_is_a_certified_full_adder():
    target = coincident_temporal_target(
        get_target('Full adder'), schedules=12)
    target.pulse_config = nv_run_config().pulse
    genome = synthesize_nervous_full_adder(
        target, chromosome_count=2, max_telomere=24)
    fitness, cases = evaluate_nv_full(genome, target)
    trace = develop_branched_hex(genome, nervous_input_pads(genome))

    assert fitness == 1.0 and min(cases) == 1.0
    assert len(genome.chromosomes) == 2
    assert len(trace.grid) == 44
    assert sorted(trace.owners.values()).count(1) == 23
    assert sorted(trace.owners.values()).count(2) == 21
    result = certify(genome, target, train=fitness, backend='nervous')
    assert result['verdict'] == 'CERTIFIED'
    assert result['holdouts'] == [1.0, 1.0, 1.0]


def test_nervous_plateau_rescue_is_normally_grown_and_scored():
    target = coincident_temporal_target(
        get_target('Full adder'), schedules=12)
    target.pulse_config = nv_run_config().pulse
    exemplar = synthesize_nervous_full_adder(
        target, chromosome_count=2, max_telomere=24)
    candidates = nervous_plateau_rescue_candidates(
        exemplar, target, limit=8, max_telomere=24)

    assert len(candidates) == 1
    assert evaluate_nv_full(candidates[0], target)[0] == 1.0


def test_sum_needs_a_network_while_carry_fits_one_tri_tile():
    capability = direct_tri_tile_capability()
    assert capability['Sum/XOR3'] is None
    assert capability['Carry/majority3'] == (4, 5, 6)


def test_declared_lut_output_is_observable_but_cannot_feed_another_output():
    """A genetic output root is a terminal, not a rewarded internal shortcut."""
    relay_w = (INPUT_TABLES['W'],) * 4
    grid = {
        (0, 0): SEED_STATE,
        (1, 0): relay_w,
        (2, 0): relay_w,
    }
    streams = [(0,), (1,), (1,), (1,), (1,), (0,), (0,)]

    probe_sim = AsyncLutSim(
        grid, input_nodes={(0, 0)}, output_nodes=set())
    probe = probe_sim.run_bits(streams, [(0, 0)], 9)
    sink_sim = AsyncLutSim(
        grid, input_nodes={(0, 0)}, output_nodes={(1, 0)})
    sink = sink_sim.run_bits(streams, [(0, 0)], 9)

    first = sink_sim._cidx[(1, 0)]
    second = sink_sim._cidx[(2, 0)]
    assert sink[:, first].any()              # still observable at its root
    assert probe[:, second].any()            # ordinary probes can be reused
    assert not sink[:, second].any()         # a declared terminal cannot drive
    assert root_source_counts(
        grid, [(0, 0)], {1: (1, 0), 2: (2, 0)}) == {1: 1, 2: 0}
    prepared = (grid, {'First': (1, 0), 'Second': (2, 0)}, None, [(0, 0)])
    assert _lut_structural_topology(prepared).reachable_nodes == 8


def test_declared_nervous_output_is_observable_but_cannot_feed_another_output():
    nodes = {'A', 'O1', 'O2'}
    routing = {node: (None, None, None, 'and') for node in nodes}
    sources = {
        'O1': ('A', 'A', None),
        'O2': ('O1', 'O1', None),
    }
    config = nv_run_config().pulse

    probe = create_simulator(
        nodes, routing, config=config, sources=sources,
        input_nodes={'A'}, output_nodes=set())
    probe.inject_pulse('A', 0.0, config.width)
    probe.advance_to(20.0)
    sink = create_simulator(
        nodes, routing, config=config, sources=sources,
        input_nodes={'A'}, output_nodes={'O1'})
    sink.inject_pulse('A', 0.0, config.width)
    sink.advance_to(20.0)

    assert sink.rise_times['O1']
    assert probe.rise_times['O2']
    assert not sink.rise_times['O2']

    # The target-blind final tie-break must see the same severed connection.
    lattice = {(0, 0): 1, (1, 0): 1, (2, 0): 1}
    lattice_routing = {
        (0, 0): (None, None, None, 'and'),
        (1, 0): ('R', 'R', None, 'and'),
        (2, 0): ('L', 'L', None, 'and'),
    }
    assert nervous_topology(
        lattice, lattice_routing, [(0, 0)]).reachable_nodes == 2
    assert nervous_topology(
        lattice, lattice_routing, [(0, 0)],
        output_nodes={(1, 0)}).reachable_nodes == 1

    tri_chain = {
        (0, 0): TRI_SEED_STATE,
        (1, 0): pack_channels(2, 2, 2),   # buffer the pad to the right
        (2, 0): pack_channels(3, 3, 3),   # would buffer root 1 to the left
    }
    roots = {1: (1, 0), 2: (2, 0)}
    assert nervous_root_source_counts(
        tri_chain, [(0, 0)], roots) == {1: 1, 2: 0}
    assert nervous_driven_roots(tri_chain, [(0, 0)], roots) == {1}


def test_lut_mutation_prefers_gate_refinement_over_catalogue_restart():
    xor2 = next(table for table in FAMILY_TABLES['XOR']
                if len(table_support(table)) == 2)
    local = 0
    for seed in range(200):
        random.seed(seed)
        changed = _nudge_cell((xor2, 0, 0, 0), ('UNRESTRICTED',))[0]
        local += (
            table_family(changed, ('UNRESTRICTED',)) == 'XOR'
            and len(table_support(changed).symmetric_difference(
                table_support(xor2))) <= 1)
    # A uniform draw from the 79-entry catalogue lands in this neighbourhood
    # only a few percent of the time.  The semantic mutation should make it the
    # majority move while retaining rarer family/topology jumps.
    assert local >= 100, local


def test_nervous_mutation_can_flip_and_to_or_without_rewiring_channel():
    base = pack_channels(4, 4, 4)  # three D&R coincidence channels
    same_wiring = 0
    for seed in range(200):
        random.seed(seed)
        changed = [config for config in channel_configs(_nudge_tile(base))
                   if config != 4]
        same_wiring += (
            len(changed) == 1
            and ROUTING_HEX[changed[0]][:3] == ROUTING_HEX[4][:3])
    assert same_wiring >= 100, same_wiring
