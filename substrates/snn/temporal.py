"""Temporal observation adapter for the recurrent SNN architecture.

The LIF engine was already continuous-time and event-driven, but temporal
targets could not reach it through the normal evaluator. This module supplies
the missing substrate-to-observation adapter and uses the same executable
``BehaviorContract`` scorer as the nervous and LUT backends.

Combinational SNN checkpoints remain feed-forward. A temporal run uses
``Arch(recurrent=True)`` so adjacent processing neurons are reciprocally
connected and can form oscillators, latches and counters.
"""
from __future__ import annotations

import dataclasses
import math

from substrates.nervous.contracts import behavior_contract_lines
from substrates.nervous.io_placement import (
    bind_io, flat_inputs, flat_outputs, growth_seeds, input_groups,
    io_strategy, merge_intervals, output_groups)
from substrates.nervous.scoring import (
    TemporalTraces, _obs_len, _score_output_candidate, needs_samples,
    score_contract, score_report_lines)
from substrates.nervous.temporal import _output_candidates

from .growth import cell_io_tags, grow_snn
from .lif_sim import _run as run_lif_events
from .snn import DEFAULT_ARCH, interpret_grid
from .targets import CURRENT_HIGH


# Temporal contracts are expressed in seconds; the legacy LIF electrical model
# is parameterised in milliseconds.  Five physical milliseconds per contract
# second makes the shortest sustainable two-cell feedback loop a period-2
# contract oscillator, while preserving the neuron's audited constants.
LIF_MS_PER_SECOND = 4.8


def temporal_arch(arch=None):
    """Return an SNN electrical profile with reciprocal temporal wiring."""
    base = arch or DEFAULT_ARCH
    return base if base.recurrent else dataclasses.replace(base, recurrent=True)


def _stream_events(streams, lane, horizon):
    """Convert contiguous high runs into physical source pulses."""
    events = []
    tick = 0
    stop = min(int(horizon), len(streams))
    while tick < stop:
        if lane >= len(streams[tick]) or not streams[tick][lane]:
            tick += 1
            continue
        start = tick
        tick += 1
        while (tick < stop and lane < len(streams[tick])
               and streams[tick][lane]):
            tick += 1
        events.append((float(start), float(max(1, tick - start))))
    return events


def _trial_input_events(trial, n_inputs, horizon):
    physical = getattr(trial, "input_events", None)
    if physical is not None:
        return [
            [(float(start), float(width)) for start, width in
             (physical[lane] if lane < len(physical) else ())]
            for lane in range(n_inputs)
        ]
    return [_stream_events(trial.streams, lane, horizon)
            for lane in range(n_inputs)]


def _sample_spikes(times, horizon, width=1.0):
    """Sample point spikes as one-second output pulses at tick midpoints."""
    row = []
    ordered = sorted(float(value) for value in times)
    index = 0
    for tick in range(int(horizon)):
        probe = tick + 0.5
        while index < len(ordered) and ordered[index] + width <= probe:
            index += 1
        row.append(
            1 if index < len(ordered)
            and ordered[index] <= probe < ordered[index] + width else 0)
    return row


def _run_trials(neurons, synapses, in_pos, target):
    """Return per-trial samples, events and unit-width output intervals."""
    obs = _obs_len(target)
    by_pos = {(neuron.x, neuron.y): neuron.id for neuron in neurons}
    lanes = []
    for cells in input_groups(in_pos):
        ids = [by_pos[cell] for cell in cells if cell in by_pos]
        if len(ids) != len(cells):
            return None
        lanes.append(ids)

    need_samples = needs_samples(target)
    trial_samples, trial_events, trial_intervals, overflow = [], [], [], False
    for trial in target.trials:
        source_events = _trial_input_events(trial, len(lanes), obs)
        scheduled = [
            (start * LIF_MS_PER_SECOND, neuron_id, CURRENT_HIGH,
             width * LIF_MS_PER_SECOND)
            for lane, ids in zip(source_events, lanes)
            for start, width in lane
            for neuron_id in ids
        ]
        lif_run = run_lif_events(
            neurons, synapses, {}, input_events=scheduled,
            sim_time=float(obs) * LIF_MS_PER_SECOND,
            max_events=getattr(target, 'max_events', 2048))
        spikes = {} if lif_run is None else lif_run.spikes
        overflow = overflow or bool(
            lif_run is not None and lif_run.overflow)
        events = {
            (neuron.x, neuron.y): [
                float(when) / LIF_MS_PER_SECOND
                for when in spikes.get(neuron.id, ())]
            for neuron in neurons}
        intervals = {
            pos: [(start, min(float(obs), start + 1.0))
                  for start in values if start < obs]
            for pos, values in events.items()}
        samples = {
            pos: (_sample_spikes(values, obs) if need_samples else [])
            for pos, values in events.items()}
        trial_samples.append(samples)
        trial_events.append(events)
        trial_intervals.append(intervals)
    return trial_samples, trial_events, trial_intervals, overflow


def _fixed_output_traces(run, out_pos, target):
    trial_samples, trial_events, trial_intervals, overflow = run
    traces = TemporalTraces(overflow=overflow)
    for role, cells in output_groups(out_pos).items():
        merged = [
            merge_intervals([trial_intervals[ti].get(cell, ())
                             for cell in cells])
            for ti in range(len(target.trials))]
        traces.intervals[role] = merged
        traces.events[role] = [
            [start for start, _end in intervals] for intervals in merged]
        traces[role] = []
        for ti in range(len(target.trials)):
            rows = [trial_samples[ti].get(cell, ()) for cell in cells]
            traces[role].append([
                1 if any(index < len(row) and row[index] for row in rows) else 0
                for index in range(_obs_len(target))
            ] if needs_samples(target) else [])
    return traces


def _fit_outputs(grid, in_pos, run, target):
    trial_samples, trial_events, trial_intervals, overflow = run
    in_set = set(flat_inputs(in_pos))
    out_pos = {terminal.role: None for terminal in target.outputs}
    traces = TemporalTraces(overflow=overflow)
    used = set()
    for terminal in target.outputs:
        best = best_key = None
        for cell in _output_candidates(grid, in_set, terminal):
            if cell in used:
                continue
            sampled, events, intervals, expected = [], [], [], []
            for trial_index, trial in enumerate(target.trials):
                exp = trial.expected.get(terminal.role)
                if exp is None:
                    continue
                sampled.append(trial_samples[trial_index].get(cell, []))
                events.append(trial_events[trial_index].get(cell, []))
                intervals.append(trial_intervals[trial_index].get(cell, []))
                expected.append(exp)
            score, _alignment = _score_output_candidate(
                sampled, events, expected, terminal.role, target,
                intervals=intervals)
            distance = (abs(cell[0] - terminal.pos[0])
                        + abs(cell[1] - terminal.pos[1]))
            key = (-score, distance, cell)
            if best_key is None or key < best_key:
                best_key, best = key, cell
        if best is None:
            return None, traces
        used.add(best)
        out_pos[terminal.role] = best
        traces[terminal.role] = [
            trial_samples[index].get(best, [])
            for index in range(len(target.trials))]
        traces.events[terminal.role] = [
            list(trial_events[index].get(best, ()))
            for index in range(len(target.trials))]
        traces.intervals[terminal.role] = [
            list(trial_intervals[index].get(best, ()))
            for index in range(len(target.trials))]
    return out_pos, traces


def prepare_snn_temporal(genome, target, arch=None):
    """Grow, interpret and fit/read a recurrent temporal SNN."""
    strategy = io_strategy(target)
    grid = grow_snn(
        genome, seeds=growth_seeds(target, strategy, genome),
        grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return None

    if strategy == "fixed":
        in_pos = list(target.inputs)
        if any(pos not in grid for pos in in_pos):
            return None
        out_pos = None
    else:
        bound = bind_io(
            genome, grid, target, strategy, tags=cell_io_tags(genome, grid))
        if bound is None:
            return None
        in_pos, out_pos = bound

    active_arch = temporal_arch(arch)
    neurons, synapses = interpret_grid(
        grid, target=target, arch=active_arch,
        input_pos=flat_inputs(in_pos), output_pos=out_pos)
    run = _run_trials(neurons, synapses, in_pos, target)
    if run is None:
        return None
    if out_pos is None:
        out_pos, traces = _fit_outputs(grid, in_pos, run, target)
        if out_pos is None:
            return None
    else:
        traces = _fixed_output_traces(run, out_pos, target)
    return grid, neurons, synapses, in_pos, out_pos, traces, active_arch


def score_snn_temporal(genome, target, arch=None):
    prep = prepare_snn_temporal(genome, target, arch)
    return 0.0 if prep is None else score_contract(prep[5], target)[0]


def snn_temporal_report(target, genome=None, arch=None):
    lines = ["Target: %s   [temporal recurrent SNN]" % target.name]
    if target.description:
        lines += [""] + target.description.splitlines()
    prep = None if genome is None else prepare_snn_temporal(
        genome, target, arch)
    if genome is not None and prep is None:
        lines += ["", "(circuit incomplete — grew too little or inputs dead)"]
    traces = prep[5] if prep is not None else None
    out_pos = prep[4] if prep is not None else None
    _score, body = score_report_lines(target, traces, out_pos)
    return "\n".join(lines + [""] + body)
