"""Contracts for LUT I/O mode 2: alternating buses around the whole body."""
from __future__ import annotations

import dataclasses
import random

from runtime.checkpoint import genome_from_dict, genome_to_dict
from runtime.config import GAConfig, RunConfig
from substrates.lut import ga as lut_ga
from substrates.lut.genome import (
    Genome, lut_exterior_edges, lut_exterior_inputs,
    random_lut_genome,
)
from substrates.lut.pulse import AsyncLutSim
from substrates.nervous.targets import coincidence_detector


def _edge_genome(n_inputs=2):
    return random_lut_genome(2, n_inputs=n_inputs)


def test_exterior_edges_are_outer_faces_and_form_alternating_input_buses():
    # The missing centre is an enclosed hole, not an outside input surface.
    grid = {
        (x, y): (1, 1, 1, 1)
        for x in range(3) for y in range(3)
        if (x, y) != (1, 1)
    }
    edges = lut_exterior_edges(grid)
    assert edges
    assert all(cell in grid and source not in grid
               for source, cell, _ in edges)
    hole_faces = {
        ((1, 0), 'N'), ((0, 1), 'E'),
        ((2, 1), 'W'), ((1, 2), 'S'),
    }
    assert all((cell, direction) not in hole_faces
               for _, cell, direction in edges)

    genome = _edge_genome(4)
    positions, links = lut_exterior_inputs(genome, grid, 4)
    assert len(positions) == 4
    assert all(positions)
    flattened = tuple(source for group in positions for source in group)
    assert len(flattened) == len(set(flattened)) == len(edges)
    assert set(flattened) == set(links)
    assert all(position not in grid for position in flattened)
    assert max(map(len, positions)) - min(map(len, positions)) <= 1
    for index, (source, cell, direction) in enumerate(edges):
        assert source in positions[index % 4]
        assert links[source] == (cell, direction)


def test_one_input_owns_every_face_and_two_inputs_alternate():
    grid = {(x, y): (1, 1, 1, 1) for x in range(2) for y in range(2)}
    edges = lut_exterior_edges(grid)

    one, one_links = lut_exterior_inputs(_edge_genome(1), grid, 1)
    assert one == (tuple(source for source, _, _ in edges),)
    assert set(one[0]) == set(one_links)

    two, two_links = lut_exterior_inputs(_edge_genome(2), grid, 2)
    for index, (source, cell, direction) in enumerate(edges):
        assert source in two[index % 2]
        assert two_links[source] == (cell, direction)


def test_exterior_source_drives_only_the_facing_lut_input():
    # 0xAAAA is the truth table "N input bit": it is high exactly when index
    # bit 0 is high. Only an exterior source on the N face may activate it.
    cell = (0, 0)
    grid = {cell: (0xAAAA, 0, 0, 0)}
    north = (0.0, 0.65)
    east = (0.65, 0.0)

    sim = AsyncLutSim(
        grid, external_inputs={north: (cell, 'N')})
    levels = sim.run_bits([(1,), (0,), (0,)], [north], 3)
    assert levels[:, 0].tolist() == [0, 1, 0]
    assert north not in sim.out
    assert set(sim.activity_at()) == {cell}

    wrong_face = AsyncLutSim(
        grid, external_inputs={east: (cell, 'E')})
    levels = wrong_face.run_bits([(1,), (0,), (0,)], [east], 3)
    assert levels[:, 0].tolist() == [0, 0, 0]


def test_exterior_source_obeys_real_delay_and_is_not_a_body_terminal():
    cell = (0, 0)
    source = (0.0, 0.65)
    sim = AsyncLutSim(
        {cell: (0xAAAA, 0, 0, 0)},
        external_inputs={source: (cell, 'N')})
    sim.inject_pulse(source, 0.25, 2.0)
    sim.advance_to(1.24)
    assert sim.out[cell] == 0
    sim.advance_to(1.25)
    assert sim.out[cell] != 0
    assert sim.input_nodes == set()
    assert source not in sim.rise_times


def test_prepare_lut_grows_from_neutral_seed_and_passes_directional_links(
        monkeypatch):
    target = coincidence_detector()
    target.lut_io_mode = 'exterior_edges'
    genome = _edge_genome(target.n_inputs)
    body = {(0, 0): (1, 1, 1, 1)}
    observed = {}

    def fake_grow(_genome, seeds, grid_size, iters):
        observed['seeds'] = tuple(seeds)
        return body

    def fake_place(grid, in_pos, _target, source_nodes=None,
                   external_inputs=None):
        observed['inputs'] = tuple(in_pos)
        observed['sources'] = source_nodes
        observed['links'] = dict(external_inputs or {})
        return {target.outputs[0].role: (0, 0)}, object()

    monkeypatch.setattr(lut_ga, 'grow_lut', fake_grow)
    monkeypatch.setattr(lut_ga, 'place_outputs_by_trace', fake_place)
    prepared = lut_ga.prepare_lut(genome, target)
    assert prepared is not None
    assert observed['seeds'] == ((0, 0),)
    assert observed['sources'] == set()
    assert len(observed['inputs']) == target.n_inputs
    assert all(len(group) == 2 for group in observed['inputs'])
    flattened = {
        source for group in observed['inputs'] for source in group}
    assert flattened == set(observed['links'])
    assert all(source not in body for source in flattened)


def test_legacy_point_layout_round_trips_but_cannot_change_exterior_binding():
    left = random_lut_genome(2, n_inputs=3, edge_input_layout=True)
    right = lut_ga.clone_genome(left)
    right.edge_input_layout = tuple(
        (value + 12345) % (1 << 16) for value in right.edge_input_layout)
    clone = lut_ga.clone_genome(left)
    assert clone.edge_input_layout == left.edge_input_layout

    grid = {(x, y): (1, 1, 1, 1) for x in range(3) for y in range(2)}
    assert lut_exterior_inputs(left, grid, 3) == lut_exterior_inputs(
        right, grid, 3)
    assert lut_ga.genome_signature(left) == lut_ga.genome_signature(right)

    random.seed(71)
    child = lut_ga.mutate_lut(left, mean_mutations=3.0)
    assert child.edge_input_layout == left.edge_input_layout
    child_a, child_b = lut_ga.crossover_lut(left, right)
    assert child_a.edge_input_layout == left.edge_input_layout
    assert child_b.edge_input_layout == right.edge_input_layout

    restored = genome_from_dict(
        genome_to_dict(left, 'lut'), 'lut')
    assert restored.edge_input_layout == left.edge_input_layout
    assert lut_exterior_inputs(restored, grid, 3) == lut_exterior_inputs(
        _edge_genome(3), grid, 3)


def test_lut_io_mode_round_trips_and_rejects_unknown_values():
    config = RunConfig(ga=GAConfig(lut_io_mode='exterior_edges'))
    restored = RunConfig.from_dict(dataclasses.asdict(config))
    assert restored.ga.lut_io_mode == 'exterior_edges'
    try:
        GAConfig(lut_io_mode='inside_out')
    except ValueError as exc:
        assert 'lut_io_mode' in str(exc)
    else:
        raise AssertionError('unknown lut_io_mode must raise ValueError')
