"""
nv_evo/simulation.py — the single construction and schedule API for nervous
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
from .pulse import PulseSim, DirectionalPulseSim
from .hexgrid import is_directional_routing


def create_simulator(grid, routing, max_events=None, config=None):
    """Construct the paper-faithful simulator used by every Nervous consumer."""
    if max_events is None and config is not None:
        max_events = config.event_cap
    simulator = DirectionalPulseSim if is_directional_routing(routing) else PulseSim
    return simulator(grid, routing, max_events=max_events, config=config)


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
                 max_events=None, config=None):
    """Inject `schedule` at float times and run the event-driven sim to
    `horizon`, enforcing `max_events` (overflow ⇒ the run is invalid, exactly as
    in scoring). Returns ({cell: [leading-edge times]}, overflow) for `out_cells`
    (all live cells if None) — continuous, no tick quantisation."""
    sim = create_simulator(grid, routing, max_events=max_events, config=config)
    for i, cell in enumerate(in_pos):
        for (t, w) in schedule[i]:
            sim.inject_pulse(cell, float(t), float(w))
    sim.advance_to(float(horizon))
    cells = list(grid) if out_cells is None else out_cells
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


def rate_scale(schedule, k):
    """Compress/expand the inter-event SCHEDULE by `k` (time axis only), leaving
    pulse widths intact — models a faster/slower input rate."""
    return [[(t * k, w) for (t, w) in ev] for ev in schedule]


def horizon_for(schedule, base_T, margin=None):
    """A horizon covering every scheduled event plus a settling margin, so a
    delayed output's late edges are observed rather than clipped at the boundary."""
    last = max((t + w for ev in schedule for (t, w) in ev), default=0.0)
    return (max(float(base_T) * pulse.TICK, last)
            + (4.0 * pulse.DELAY if margin is None else margin))
