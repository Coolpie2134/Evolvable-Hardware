"""Equivalence guards for optimized evaluation hot paths."""
from __future__ import annotations

import itertools
import random

import numpy as np


def _snn_reference_lookup(genome, context):
    sn, ss, se, sw, si = context
    if sn == ss == se == sw == si == 0:
        return 0
    popcount = tuple(bin(value).count("1") for value in range(16))
    best_out, best_distance = 0, 1 << 30
    for chromosome in genome.chromosomes:
        if getattr(chromosome, "wiring", False):
            continue
        for gene in chromosome.genes:
            distance = (
                popcount[(gene.state_n ^ sn) & 0xF]
                + popcount[(gene.state_s ^ ss) & 0xF]
                + popcount[(gene.state_e ^ se) & 0xF]
                + popcount[(gene.state_w ^ sw) & 0xF]
                + popcount[(gene.self_in ^ si) & 0xF]
            )
            if distance < best_distance:
                best_distance, best_out = distance, gene.self_out
    return best_out if best_out else 1


def test_snn_packed_lookup_matches_fieldwise_reference():
    from substrates.snn.genome import random_genome
    from substrates.snn.growth import _compile_lookup, _lookup_compiled

    random.seed(20260729)
    for _ in range(12):
        genome = random_genome(3)
        program = _compile_lookup(genome)
        for _ in range(80):
            context = tuple(random.randrange(16) for _ in range(5))
            assert _lookup_compiled(program, *context) == \
                _snn_reference_lookup(genome, context)


def test_lut_vectorized_steady_duty_matches_scalar_for_every_short_trace():
    from substrates.lut.ga import _steady_duties, _steady_duty

    traces = list(itertools.product((0, 1), repeat=9))
    matrix = np.asarray(traces, dtype=np.uint8).T
    actual = _steady_duties(matrix)
    expected = np.asarray([_steady_duty(trace) for trace in traces])
    assert np.array_equal(actual, expected)


def test_lut_batched_lattice_matches_independent_simulators():
    from substrates.lut.pulse import AsyncLutSim

    rng = random.Random(20260730)
    grid = {
        (x, y): tuple(rng.randrange(1 << 16) for _ in range(4))
        for x in range(3) for y in range(2)
    }
    streams = np.asarray([
        [[rng.randrange(2), rng.randrange(2)] for _ in range(14)]
        for _ in range(7)
    ], dtype=np.uint8)
    modes = [
        (
            ((0, 0), (0, 1)),
            {(0, 0), (0, 1)},
            {},
        ),
        (
            ((-0.65, 0.0), (2.65, 1.0)),
            set(),
            {
                (-0.65, 0.0): ((0, 0), 'W'),
                (2.65, 1.0): ((2, 1), 'E'),
            },
        ),
    ]
    for in_pos, source_nodes, external_inputs in modes:
        compiled = AsyncLutSim(
            grid, input_nodes=source_nodes, output_nodes={(1, 1)},
            external_inputs=external_inputs)
        actual = compiled.run_bits_batch_lattice(streams, in_pos)
        expected = np.stack([
            AsyncLutSim(
                grid, input_nodes=source_nodes, output_nodes={(1, 1)},
                external_inputs=external_inputs,
            ).run_bits(trial.tolist(), in_pos, len(trial))
            for trial in streams
        ])
        assert np.array_equal(actual, expected)


def test_lut_batched_case_duties_match_serial_reference():
    from substrates.lut.ga import (
        _all_case_duties, _all_cell_duties, _combinational_schedule)
    from substrates.lut.genome import (
        lut_exterior_inputs, random_lut_genome)
    from substrates.snn.targets import get_target

    rng = random.Random(20260731)
    grid = {
        (x, y): tuple(rng.randrange(1 << 16) for _ in range(4))
        for x in range(3) for y in range(3)
    }
    target = get_target('Half adder')
    schedule = _combinational_schedule(target)
    in_pos, external_inputs = lut_exterior_inputs(
        random_lut_genome(1), grid, target.n_inputs)
    case_inputs = [in_bits for in_bits, _ in target.cases]
    expected = [
        _all_cell_duties(
            grid, target, in_bits, schedule, in_pos=in_pos,
            source_nodes=set(), external_inputs=external_inputs)
        for in_bits in case_inputs
    ]
    actual = _all_case_duties(
        grid, target, case_inputs, schedule, in_pos=in_pos,
        source_nodes=set(), external_inputs=external_inputs)
    assert [tuple(row) for row in actual] == [
        tuple(row) for row in expected]
    for actual_case, expected_case in zip(actual, expected):
        assert actual_case.keys() == expected_case.keys()
        assert all(
            abs(actual_case[cell] - expected_case[cell]) < 1e-12
            for cell in actual_case)


def test_fnv_compiled_wiring_matches_fresh_wiring():
    from substrates.fnv.evaluation import run_functional_events
    from substrates.fnv.simulation import compile_functional_grid
    from substrates.fnv.catalogue import BY_NAME

    source = (0, 0)
    grid = {
        source: BY_NAME["DELAY1_D_TO_LR"].id,
        (1, 0): BY_NAME["DELAY1_R_TO_LD"].id,
        (2, 0): BY_NAME["HOLD1_R_TO_LD"].id,
    }
    streams = [[1], [1], [0], [0], [0]]
    fresh = run_functional_events(
        grid, [source], {}, streams, len(streams))
    compiled = run_functional_events(
        grid, [source], {}, streams, len(streams),
        _compiled=compile_functional_grid(grid, [source]))
    assert fresh == compiled


def test_nervous_output_fitting_stops_at_first_overflow(monkeypatch):
    """An invalid event storm must not run the target's remaining trials."""
    from substrates.nervous import temporal
    from substrates.nervous.scoring import PhysicalEvents
    from substrates.nervous.targets import TEMPORAL_TARGETS

    target = TEMPORAL_TARGETS['SR latch']
    grid = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}
    calls = []

    def overflow(*_args, **_kwargs):
        calls.append(1)
        return [], {}, PhysicalEvents(), True

    monkeypatch.setattr(temporal, 'input_cone',
                        lambda *_args: set(grid))
    monkeypatch.setattr(temporal, 'run_nervous_events', overflow)
    monkeypatch.setattr(
        temporal, '_score_output_candidate',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('overflow reached candidate fitting')))

    outputs, traces = temporal.place_outputs_by_trace(
        grid, {}, [(0, 0), (1, 0)], target)

    assert len(target.trials) > 1
    assert len(calls) == 1
    assert traces.overflow
    assert all(position is None for position in outputs.values())


def test_expected_target_windows_are_compiled_once():
    from substrates.nervous.scoring import _expected_windows

    expected = [None, 0, 0, 1, 1, None, 0]
    _expected_windows.cache_clear()
    first = _expected_windows(expected)
    second = _expected_windows(list(expected))

    assert first is second
    assert first == ((0, (1, 2)), (1, (3, 4)), (0, (6,)))
    assert _expected_windows.cache_info().hits == 1


def test_interval_reconstruction_matches_engine_half_tick_samples():
    """Fitness can skip full-grid snapshots without changing sampled state."""
    from substrates.nervous.hexgrid import ROUTING_HEX
    from substrates.nervous.pulse import PulseConfig
    from substrates.nervous.temporal import (
        _sample_intervals, run_nervous_events)
    from substrates.nervous.tritile import TRI_SEED_STATE

    source, body = (0, 0), (1, 0)
    streams = [(0,), (1,), (1,), (0,), (0,), (1,), (0,), (0,)]
    single_grid = {source: 1, body: 2}
    single_routing = {
        cell: ROUTING_HEX[state] for cell, state in single_grid.items()}
    cases = [
        (single_grid, single_routing, 'single', PulseConfig()),
        (single_grid, single_routing, 'single',
         PulseConfig(model='paper_analog')),
        ({source: TRI_SEED_STATE, body: TRI_SEED_STATE}, {}, 'tri3',
         PulseConfig(model='paper_analog')),
    ]

    for grid, routing, arch, config in cases:
        sampled, _, _, sampled_overflow = run_nervous_events(
            grid, routing, [source], {}, streams, len(streams), prune=False,
            sample=True, config=config, arch=arch,
            terminal_inputs={source})
        _, _, events, event_overflow = run_nervous_events(
            grid, routing, [source], {}, streams, len(streams), prune=False,
            sample=False, config=config, arch=arch,
            terminal_inputs={source})

        assert not sampled_overflow and not event_overflow
        for cell in grid:
            expected = tuple(state.get(cell, 0) for state in sampled)
            actual = _sample_intervals(
                events.intervals.get(cell, ()), len(streams))
            assert actual == expected, (arch, cell, actual, expected)
