"""Regression tests for the paper's three-circuit nervous-network tile."""
from __future__ import annotations

import random
import os
import pickle
import tempfile
from types import SimpleNamespace

from evo_runtime.checkpoint import genome_from_dict, genome_to_dict, load_checkpoint
from nv_evo.genome import (MAX_STATE, Chromosome, Genome, HexGene,
                           random_hex_genome)
from nv_evo.hexgrid import (DIRECTIONS, GENOME_ENCODING, PAPER_ROUTING,
                            TILE_ENCODING, decode_tile_routing,
                            developmental_context,
                            directional_connections, facing_channel,
                            format_tile_state, hex_dirs, pack_tile_state,
                            parse_tile_state,
                            promote_legacy_state, tile_channel,
                            unpack_tile_state)
from nv_evo.nervous import SEED_STATE, _next_tile_state, interpret_nervous
from nv_evo.ga import _other_state
from nv_evo.simulation import create_simulator
from nv_evo.targets import OutputTerminal


def test_tile_state_is_three_independent_figure3_nibbles():
    state = pack_tile_state(1, 7, 15)
    assert unpack_tile_state(state) == (1, 7, 15)
    entries = decode_tile_routing(state)
    assert entries[0][:3] == ('D', 'D', None)
    assert entries[1][:3] == ('R', 'R', 'L')
    assert entries[2][:3] == ('L', 'R', 'D')
    assert all(entry[3] == 'and' for entry in entries)

    try:
        pack_tile_state(16, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range circuit selector was silently packed')


def test_tile_word_is_presented_as_three_figure3_configurations():
    state = pack_tile_state(3, 7, 15)
    assert format_tile_state(state) == '3/7/F'
    assert parse_tile_state('3/7/f') == state
    assert parse_tile_state('3, 7, F') == state


def test_drawable_connections_are_deduplicated_and_target_exact_circuits():
    reader = (0, 0)
    neighbours = set(hex_dirs(*reader).values())
    grid = {reader: pack_tile_state(2, 2, 2)}
    grid.update({pos: SEED_STATE for pos in neighbours})
    routing, _, _ = interpret_nervous(grid)

    connections = [connection for connection in
                   directional_connections(grid, routing)
                   if connection[1][:2] == reader]
    # State 2 is a one-nerve R buffer (R is fanned internally into E1+E2),
    # therefore it draws one green input per exact L/R/D destination circuit.
    assert len(connections) == 3
    assert {destination for _, destination, _ in connections} == {
        tile_channel(reader, direction) for direction in DIRECTIONS}
    assert all(role == 'exc' for _, _, role in connections)
    assert len(connections) == len(set(connections))


def test_each_inhibited_circuit_has_one_veto_not_a_double_inhibition():
    reader = (0, 0)
    neighbours = set(hex_dirs(*reader).values())
    grid = {reader: pack_tile_state(7, 7, 7)}
    grid.update({pos: SEED_STATE for pos in neighbours})
    routing, _, _ = interpret_nervous(grid)
    connections = [connection for connection in
                   directional_connections(grid, routing)
                   if connection[1][:2] == reader]
    for direction in DIRECTIONS:
        destination = tile_channel(reader, direction)
        incoming = [role for _, dest, role in connections if dest == destination]
        assert incoming.count('exc') == 1
        assert incoming.count('inh') == 1


def test_no_figure3_configuration_uses_one_nerve_as_excitation_and_inhibition():
    for e1, e2, i1, _op in PAPER_ROUTING:
        assert i1 is None or i1 not in {e1, e2}


def test_facing_channel_is_reciprocal_on_both_hex_parities():
    for pos in ((0, 0), (1, 0), (-2, 3), (4, -1)):
        for direction in DIRECTIONS:
            source = facing_channel(pos, direction)
            assert facing_channel(source[:2], source[2]) == tile_channel(pos, direction)


def test_one_programmed_output_does_not_clone_across_tile_channels():
    sensory, reader = (0, 0), (1, 0)
    # The reader's L output buffers its R input (the sensory tile). R/D are off.
    grid = {sensory: SEED_STATE, reader: pack_tile_state(2, 0, 0)}
    routing, _, _ = interpret_nervous(grid)
    sim = create_simulator(grid, routing)
    sim.inject_pulse(sensory, 0.0, 1.0)
    sim.advance_to(2.0)

    assert sim.rise_times[tile_channel(reader, 'L')] == [1.0]
    assert sim.rise_times[tile_channel(reader, 'R')] == []
    assert sim.rise_times[tile_channel(reader, 'D')] == []


def test_reader_uses_only_the_neighbours_facing_output_wire():
    sensory, reader = (0, 0), (1, 0)
    grid = {sensory: SEED_STATE, reader: pack_tile_state(2, 0, 0)}
    routing, _, _ = interpret_nervous(grid)

    wrong = create_simulator(grid, routing)
    wrong.inject_pulse(tile_channel(sensory, 'L'), 0.0, 1.0)
    wrong.advance_to(2.0)
    assert wrong.rise_times[tile_channel(reader, 'L')] == []

    facing = create_simulator(grid, routing)
    facing.inject_pulse(tile_channel(sensory, 'R'), 0.0, 1.0)
    facing.advance_to(2.0)
    assert facing.rise_times[tile_channel(reader, 'L')] == [1.0]


def test_output_placement_selects_an_exact_programmed_channel():
    sensory, reader = (0, 0), (1, 0)
    grid = {sensory: SEED_STATE, reader: pack_tile_state(2, 0, 0)}
    target = SimpleNamespace(
        inputs=[sensory], outputs=[OutputTerminal('Q', reader)],
        output_strategy='terminals', grid_size=3)
    _, _, outputs = interpret_nervous(grid, target)
    assert outputs == {'Q': tile_channel(reader, 'L')}


def test_state_tweak_changes_one_circuit_sram_bit_only():
    random.seed(19)
    state = 7
    for _ in range(30):
        changed = _other_state(state)
        delta = state ^ changed
        assert delta and delta & (delta - 1) == 0
        assert 0 <= changed < MAX_STATE


def test_random_genomes_contain_only_4bit_circuit_states():
    random.seed(23)
    assert MAX_STATE == 16
    for _ in range(20):
        genome = random_hex_genome(3)
        for chromosome in genome.chromosomes:
            for gene in chromosome.genes:
                values = (gene.ctx_l, gene.ctx_r, gene.ctx_d,
                          gene.self_in, gene.self_out)
                assert all(0 <= value < MAX_STATE for value in values)


def test_packed_tile_words_are_rejected_as_genome_alleles():
    try:
        HexGene(1, 2, 3, 0, pack_tile_state(1, 2, 3))
    except ValueError as exc:
        assert '0..15' in str(exc)
    else:
        raise AssertionError('packed phenotype word was accepted as a gene state')

    # Direct assignment remains possible for migration of old pickle objects,
    # but current checkpoint serialization must refuse to perpetuate it.
    gene = HexGene(1, 2, 3, 0, 4)
    gene.self_out = 1000
    genome = Genome([Chromosome(genes=[gene])])
    try:
        genome_to_dict(genome, 'nervous')
    except ValueError as exc:
        assert '4-bit' in str(exc)
    else:
        raise AssertionError('invalid nervous gene was serialized')


def test_one_gene_program_develops_lrd_circuits_independently():
    seed, reader = (0, 0), (1, 0)
    grid = {seed: SEED_STATE}
    assert developmental_context(grid, reader, 'L') == (1, 0, 0)
    assert developmental_context(grid, reader, 'R') == (0, 0, 1)
    assert developmental_context(grid, reader, 'D') == (0, 1, 0)

    genome = Genome([Chromosome(genes=[
        HexGene(1, 0, 0, 0, 3),
        HexGene(0, 0, 1, 0, 2),
        HexGene(0, 1, 0, 0, 1),
    ])])
    developed = _next_tile_state(genome, grid, reader, 0, {})
    assert unpack_tile_state(developed) == (3, 2, 1)


def test_directional_loop_rings_and_its_veto_stops_the_loop():
    left, right, reset = (0, 0), (1, 0), (2, 0)
    loop_state = pack_tile_state(0, 2, 0)  # R-out buffers the facing R input

    grid = {left: loop_state, right: loop_state}
    routing, _, _ = interpret_nervous(grid)
    sim = create_simulator(grid, routing, max_events=100)
    sim.inject_pulse(left, 0.0, 1.0)
    sim.advance_to(8.0)
    assert sim.rise_times[tile_channel(right, 'R')] == [1.0, 3.0, 5.0, 7.0]

    # State 7 is the same R buffer with its L input as an inhibitory veto.
    stopped_grid = {
        left: loop_state,
        right: pack_tile_state(0, 7, 0),
        reset: SEED_STATE,
    }
    stopped_routing, _, _ = interpret_nervous(stopped_grid)
    stopped = create_simulator(stopped_grid, stopped_routing, max_events=100)
    stopped.inject_pulse(left, 0.0, 1.0)
    stopped.inject_pulse(reset, 2.0, 1.0)
    stopped.advance_to(8.0)
    assert stopped.rise_times[tile_channel(right, 'R')] == [1.0]


def test_legacy_checkpoint_keeps_the_figure3_nibble_only():
    legacy = {
        'tag': 4,
        'gene_fields': ['ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'],
        'chromosomes': [{
            'tag': 1, 'split': 0, 'telomere': 3,
            'genes': [[1, 2, 3, 0, 17]],
        }],
    }
    genome = genome_from_dict(legacy, 'nervous')
    gene = genome.chromosomes[0].genes[0]
    assert (gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in, gene.self_out) == (
        1, 2, 3, 0, 1)


def test_unmarked_thousand_scale_genome_values_are_not_treated_as_legacy():
    invalid = {
        'tag': 4,
        'gene_fields': ['ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'],
        'chromosomes': [{
            'tag': 1, 'split': 0, 'telomere': 3,
            'genes': [[1, 2, 3, 0, 1000]],
        }],
    }
    try:
        genome_from_dict(invalid, 'nervous')
    except ValueError as exc:
        assert '5-bit' in str(exc)
    else:
        raise AssertionError('thousand-scale unmarked gene was accepted')


def test_new_checkpoint_encoding_round_trips_without_double_promotion():
    genome = Genome([Chromosome(
        genes=[HexGene(1, 7, 15, 3, 12)],
        split=0, tag=2, telomere=3)], tag=9)
    document = genome_to_dict(genome, 'nervous')
    assert document['state_encoding'] == GENOME_ENCODING
    assert all(0 <= value < 16
               for row in document['chromosomes'][0]['genes']
               for value in row)
    restored = genome_from_dict(document, 'nervous')
    gene = restored.chromosomes[0].genes[0]
    assert (gene.ctx_l, gene.ctx_r, gene.ctx_d,
            gene.self_in, gene.self_out) == (1, 7, 15, 3, 12)


def test_interim_packed_genome_expands_into_three_4bit_rules():
    packed = {
        'tag': 4,
        'state_encoding': TILE_ENCODING,
        'gene_fields': ['ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'],
        'chromosomes': [{
            'tag': 1, 'split': 1, 'telomere': 3,
            'genes': [
                [pack_tile_state(1, 2, 3), pack_tile_state(4, 5, 6),
                 pack_tile_state(7, 8, 9), pack_tile_state(0, 10, 11),
                 pack_tile_state(12, 13, 14)],
                [pack_tile_state(15, 14, 13), pack_tile_state(12, 11, 10),
                 pack_tile_state(9, 8, 7), pack_tile_state(6, 5, 4),
                 pack_tile_state(3, 2, 1)],
            ],
        }],
    }
    chromosome = genome_from_dict(packed, 'nervous').chromosomes[0]
    assert len(chromosome.genes) == 6
    assert chromosome.split == 3
    first_three = [
        (gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in, gene.self_out)
        for gene in chromosome.genes[:3]
    ]
    assert first_three == [
        (1, 4, 7, 0, 12),
        (2, 5, 8, 10, 13),
        (3, 6, 9, 11, 14),
    ]
    assert all(0 <= value < 16 for gene in chromosome.genes
               for value in (gene.ctx_l, gene.ctx_r, gene.ctx_d,
                             gene.self_in, gene.self_out))


def test_legacy_pickle_checkpoint_is_promoted_once():
    legacy_gene = HexGene(1, 2, 3, 0, 1)
    # Simulate an object pickled before HexGene enforced the 4-bit boundary.
    legacy_gene.self_out = 17
    old = Genome([Chromosome(
        genes=[legacy_gene], split=0, tag=2, telomere=3)], tag=9)
    state = {'backend': 'nervous', 'best_genome': old,
             'grid': {(0, 0): 1}, 'out_pos': {'Q': (0, 0)}}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'legacy.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(state, handle)
        restored = load_checkpoint(path)
    gene = restored['best_genome'].chromosomes[0].genes[0]
    assert gene.ctx_l == 1
    assert gene.self_out == 1
    assert restored['grid'][(0, 0)] == promote_legacy_state(1)
    assert restored['out_pos']['Q'] == tile_channel((0, 0), 'L')
    assert restored['state_encoding'] == TILE_ENCODING
    assert restored['genome_state_encoding'] == GENOME_ENCODING
