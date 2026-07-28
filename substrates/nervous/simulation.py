"""
substrates/nervous/simulation.py — the single construction and schedule API for nervous
nets: the audited path for asynchronous robustness testing.

temporal.py drives circuits from tick-quantised streams (`start * TICK`, integer
edges). That is fine for evolution, but it keeps the whole experiment on an
integer timing lattice — so it cannot certify that behaviour is ASYNCHRONOUS
rather than merely clocked at TICK granularity. This module drives the same
`PulseSim` from schedules of arbitrary FLOAT event times, and provides the pure
schedule primitives the metamorphic synchrony audit and the robustness harness
are built on.

A `schedule` is a list (one entry per input, in `in_pos` order) of
`(start_time, width)` float pulse events.

The physical constants (DELAY/WIDTH/TICK/COINC) are read from the `pulse` module
DYNAMICALLY (not imported by value), so a physics sweep that rebinds
`pulse.DELAY` is honoured here too instead of using a stale copy.
"""
from __future__ import annotations

from . import pulse
from .pulse import PulseSim


def create_simulator(grid, routing, max_events=None, config=None, delays=None,
                     sources=None, input_nodes=None, output_nodes=None):
    """Construct the configured simulator used by every Nervous consumer.

    ``delays`` ({cell: delay}) is consumed only by width-preserving transport;
    its output width is derived dynamically from the incoming waveform.
    ``sources`` ({node: (s1, s2, si)}) pre-resolves feeder nodes for the
    tri-tile substrate; None keeps the single-circuit hex_dirs decode."""
    if max_events is None and config is not None:
        max_events = config.event_cap
    if config is not None and getattr(config, 'model', 'uniform') == 'paper_analog':
        # The analog Fig. 1 engine is a distinct simulator sharing the same
        # external surface. Map the run's propagation delay onto the node's
        # delay; coincidence window and output width are EMERGENT, so COINC and
        # output WIDTH are not mapped into the node. PulseConfig.width can still
        # be used upstream as the default EXTERNAL stimulus duration.
        from .analog import AnalogPulseSim, AnalogConfig
        acfg = AnalogConfig(threshold=config.analog_threshold,
                            step=config.analog_step,
                            tau_leak=config.analog_tau_leak,
                            hysteresis=config.analog_hysteresis,
                            delay_prop=config.delay,
                            event_cap=getattr(config, 'event_cap', 4096))
        return AnalogPulseSim(grid, routing, max_events=max_events,
                              config=acfg, sources=sources,
                              input_nodes=input_nodes,
                              output_nodes=output_nodes)
    return PulseSim(grid, routing, max_events=max_events, config=config,
                    delays=delays, sources=sources,
                    input_nodes=input_nodes, output_nodes=output_nodes)


def normalize(schedule):
    """The EFFECTIVE physical stimulus: per input, sort pulses and merge those
    that overlap in time (a wired-OR net that is re-driven while still high shows
    ONE leading edge, not two). Returns a schedule of non-overlapping pulses."""
    out = []
    for ev in schedule:
        merged = []
        for (t, w) in sorted(ev):
            end = t + w
            if merged and t <= merged[-1][1]:          # overlaps the previous run
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((t, end))
        out.append([(a, b - a) for (a, b) in merged])
    return out


def effective_edges(schedule):
    """Per-input leading-edge times AFTER wired-OR merging — the edge train the
    substrate actually receives, which is what state semantics (e.g. parity) must
    be computed from, not the pre-merge schedule."""
    return [[t for (t, _w) in ev] for ev in normalize(schedule)]


def streams_to_schedule(streams, n_inputs, T, config=None):
    """The nominal lattice schedule: each contiguous high run in the integer
    trial `streams` becomes one float pulse `(start*TICK, run_len*TICK)` — the
    same wired-OR edge decomposition temporal._inject_stream_edges uses, but
    returned as a transformable float schedule instead of injected in place."""
    schedule = [[] for _ in range(n_inputs)]
    stop = min(int(T), len(streams))
    for i in range(n_inputs):
        t = 0
        while t < stop:
            if i >= len(streams[t]) or not streams[t][i]:
                t += 1
                continue
            start = t
            t += 1
            while t < stop and i < len(streams[t]) and streams[t][i]:
                t += 1
            width = pulse.WIDTH if config is None else config.width
            schedule[i].append((float(start) * pulse.TICK,
                                max(width, (t - start) * pulse.TICK)))
    return schedule


def run_schedule(grid, routing, in_pos, schedule, horizon, out_cells=None,
                 max_events=None, config=None, delays=None,
                 return_intervals=False, arch='single',
                 terminal_inputs=None, terminal_outputs=None):
    """Inject `schedule` at float times and run the event-driven sim to
    `horizon`, enforcing `max_events` (overflow ⇒ the run is invalid, exactly as
    in scoring). Returns ({cell: [leading-edge times]}, overflow) for `out_cells`
    (all live cells if None) — continuous, no tick quantisation.

    ``delays`` drives evolved-delay, width-preserving transport. Set
    ``return_intervals`` to receive complete physical output intervals instead
    of leading edges."""
    from .io_placement import input_groups, flat_inputs
    if arch == 'tri3':
        from .tritile import TriSim
        sim = TriSim(grid, flat_inputs(in_pos), max_events=max_events,
                     config=config, outputs=terminal_outputs)
    elif arch == 'single':
        sim = create_simulator(grid, routing, max_events=max_events,
                               config=config, delays=delays,
                               input_nodes=terminal_inputs,
                               output_nodes=terminal_outputs)
    else:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    # ``in_pos`` may carry per-input attachment GROUPS (evolvable binding):
    # lane i's schedule injects at every attachment cell of input i.
    for i, cells in enumerate(input_groups(in_pos)):
        for (t, w) in schedule[i]:
            for cell in cells:
                sim.inject_pulse(cell, float(t), float(w))
    sim.advance_to(float(horizon))
    cells = list(grid) if out_cells is None else out_cells
    if return_intervals:
        return ({c: [tuple(interval) for interval in
                     sim.pulse_intervals.get(c, ())]
                 for c in cells}, sim.overflow)
    return {c: list(sim.rise_times.get(c, [])) for c in cells}, sim.overflow


# ── pure schedule transforms (perturbations + metamorphic relations) ────────────

def translate(schedule, delta):
    """Shift every input event by `delta`."""
    return [[(t + delta, w) for (t, w) in ev] for ev in schedule]


def jitter(schedule, eps, rng, floor=0.0):
    """Independent per-event timing jitter, drawn uniformly from [-eps, eps].
    Widths are unchanged; start times are clamped to `floor`."""
    return [[(max(floor, t + rng.uniform(-eps, eps)), w) for (t, w) in ev]
            for ev in schedule]


def scale(schedule, k):
    """Scale every event time and width by `k` (used with a matching scale of
    the physical constants to test scale covariance)."""
    return [[(t * k, w * k) for (t, w) in ev] for ev in schedule]
