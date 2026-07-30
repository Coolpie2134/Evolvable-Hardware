"""Native LUT I/O: evolved square-grid sources and fitted output probes."""
from __future__ import annotations

import random

from runtime.checkpoint import genome_from_dict, genome_to_dict
from substrates.lut import ga as lut_ga
from substrates.lut.ga import (
    clone_genome, crossover_lut, genome_signature, mutate_input_layout,
)
from substrates.lut.genome import (
    input_layout_domain, input_layout_radius, lut_input_positions,
    random_input_layout, random_lut_genome,
)
from substrates.nervous.targets import coincidence_detector


FALLBACK = ((5, 5), (6, 6), (7, 7))


def _genome(n_inputs=3, seed=1):
    random.seed(seed)
    return random_lut_genome(
        2, n_inputs=n_inputs, input_layout=True)


def test_fresh_layout_is_anchored_distinct_and_compact():
    for count in (0, 1, 2, 3, 6):
        random.seed(count)
        layout = random_input_layout(count)
        assert len(layout) == count
        assert len(set(layout)) == count
        if count:
            assert layout[0] == (0, 0)
        domain = set(input_layout_domain(input_layout_radius(20, count)))
        assert all(cell in domain for cell in layout)


def test_mutation_moves_one_non_anchor_pad_one_cardinal_edge():
    random.seed(11)
    for _ in range(50):
        genome = _genome(4, random.randrange(10_000))
        before = genome.input_layout
        if not mutate_input_layout(genome):
            continue
        after = genome.input_layout
        moved = [index for index in range(4)
                 if before[index] != after[index]]
        assert moved and len(moved) == 1
        index = moved[0]
        assert index != 0
        assert after[0] == (0, 0)
        assert len(set(after)) == len(after)
        assert (abs(after[index][0] - before[index][0])
                + abs(after[index][1] - before[index][1])) == 1


def test_clone_and_crossover_keep_layout_as_a_whole_module():
    parent_a = _genome(3, 13)
    parent_b = _genome(3, 17)
    parent_a.input_layout = ((0, 0), (1, 0), (2, 0))
    parent_b.input_layout = ((0, 0), (0, 1), (0, 2))
    assert clone_genome(parent_a).input_layout == parent_a.input_layout
    seen = set()
    random.seed(19)
    for _ in range(60):
        for child in crossover_lut(parent_a, parent_b):
            assert child.input_layout in (
                parent_a.input_layout, parent_b.input_layout)
            seen.add(child.input_layout)
    assert seen == {parent_a.input_layout, parent_b.input_layout}


def test_layout_is_part_of_the_fitness_cache_signature():
    left = _genome(3, 23)
    right = clone_genome(left)
    right.input_layout = ((0, 0), (2, 0), (0, 2))
    assert genome_signature(left) != genome_signature(right)


def test_lut_shares_the_global_probe_assignment_rule_with_nv_and_fnv():
    import substrates.fnv.evaluation as fnv_evaluation
    import substrates.nervous.temporal as nv_temporal
    from substrates.nervous.scoring import (
        behavior_representatives, best_distinct_assignment)

    assert lut_ga.behavior_representatives is behavior_representatives
    assert lut_ga.best_distinct_assignment is best_distinct_assignment
    assert fnv_evaluation.best_distinct_assignment is best_distinct_assignment
    assert nv_temporal.best_distinct_assignment is best_distinct_assignment


def test_invalid_layout_is_unbindable_and_never_repaired():
    genome = _genome(3, 29)
    for broken in (
            ((0, 0), (1, 0)),
            ((0, 0), (1, 0), (1, 0)),
            ((1, 0), (2, 0), (3, 0)),
            ((0, 0), (1, 0), (2,)),
            ((0, 0), (1, 0), 'x'),
    ):
        genome.input_layout = broken
        assert lut_input_positions(genome, FALLBACK) == ()
        assert genome.input_layout is broken
    legacy = random_lut_genome(2)
    assert lut_input_positions(legacy, FALLBACK) == FALLBACK


def test_layout_checkpoint_round_trip_and_legacy_fallback():
    genome = _genome(3, 31)
    restored = genome_from_dict(genome_to_dict(genome, 'lut'), 'lut')
    assert restored.input_layout == genome.input_layout

    document = genome_to_dict(genome, 'lut')
    document.pop('input_layout')
    legacy = genome_from_dict(document, 'lut')
    assert legacy.input_layout is None
    assert lut_input_positions(legacy, FALLBACK) == FALLBACK


def test_prepare_lut_uses_layout_as_germline_and_source_membership(monkeypatch):
    target = coincidence_detector()
    genome = _genome(target.n_inputs, 37)
    body = set(genome.input_layout) | {(4, 4), (5, 4)}
    observed = {}

    def fake_grow(_genome, seeds, grid_size, iters):
        observed['seeds'] = tuple(seeds)
        return {cell: (1, 1, 1, 1) for cell in body}

    def fake_place(grid, in_pos, _target, source_nodes=None):
        observed['inputs'] = tuple(in_pos)
        observed['sources'] = set(source_nodes or ())
        role = _target.outputs[0].role
        return {role: (4, 4)}, object()

    monkeypatch.setattr(lut_ga, 'grow_lut', fake_grow)
    monkeypatch.setattr(lut_ga, 'place_outputs_by_trace', fake_place)
    prepared = lut_ga.prepare_lut(genome, target)
    assert prepared is not None
    assert observed['seeds'] == genome.input_layout
    assert observed['inputs'] == genome.input_layout
    assert observed['sources'] == set(genome.input_layout)


def test_interactive_reuses_scorer_layout_probes_and_noninvasive_physics():
    from ui.interactive import prepare_lut_playback

    target = coincidence_detector()
    genome = _genome(target.n_inputs, 0)
    scored = lut_ga.prepare_lut(genome, target)
    assert scored is not None
    playback = prepare_lut_playback(genome, target)
    assert playback is not None
    grid, in_pos, out_pos, sources, sinks = playback
    assert grid == scored[0]
    assert out_pos == scored[1]
    assert tuple(in_pos) == genome.input_layout
    assert sources == set(genome.input_layout)
    assert sinks == set()
