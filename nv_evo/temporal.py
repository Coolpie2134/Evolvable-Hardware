"""
nv_evo/temporal.py — temporal scoring and loop analysis over the pulse engine.

The nervous net is a temporal system: a pulse injected by an input circulates
around a loop of buffers (a stored bit = a circulating pulse) until inhibition
stops it. The dynamics themselves are the asynchronous edge-triggered pulse
simulation in nv_evo/pulse.py (PulseSim). Event/cadence fitness schedules input
edges directly in continuous time; only playback and persistence targets request
sampled tick states.

Core (shared by interactive playback and temporal scoring):
    run_nervous(grid, routing, in_pos, out_pos, streams, T) -> (states, traces)
    score_temporal(genome, ttarget)                 -> behavioural fitness [0,1]
    loop_profile(grid, routing, in_pos, out_pos)    -> feedback-loop stats

Loop analysis feeds the GA's fitness shaping (nv_evo/ga.py): memory needs a
directed cycle in the signal graph, and a cycle only matters if the inputs can
write into it and it can drive an output.

Point-event targets consume the raw leading-edge timestamps without sampling.
Cadence and period-stepper targets score behavioral invariants; sampled traces
remain for GUI playback and state/persistence regimes where a circulating pulse
represents a stored level rather than one prescribed spike train.

The target types (Trial, TemporalTarget, presets) live in nv_evo/targets.py;
they are re-exported here for back-compat with older imports and pickles.
"""
from __future__ import annotations
from bisect import bisect_left
import math

from .nervous import grow_nervous, interpret_nervous
from .hexgrid import hex_dirs
from . import pulse as pulse_engine
from .pulse import PulseSim, TICK
from .targets import (OutputTerminal, Trial, TemporalTarget, TEMPORAL_TARGETS,
                      sr_latch, toggle_ff, oscillator, echo)


class TemporalTraces(dict):
    """Sampled role traces plus the raw point events that produced them.

    It deliberately remains a ``dict`` so existing GUI/report/LUT callers keep
    working.  Event-aware scorers read ``events[role][trial]``; old scorers read
    the ordinary sampled lists.  ``overflow`` invalidates a genome whose event
    count exceeded the target's deterministic safety cap.
    """

    def __init__(self, *args, events=None, overflow=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events or {}
        self.overflow = bool(overflow)
        self._event_result = None
        self._cadence_result = None


# ── dynamics (asynchronous pulse engine, sampled per tick) ──────────────────────
# The actual dynamics are event-driven edge-triggered pulses (see pulse.py).
# Playback samples once per tick; event-semantic scoring reads raw timestamps.

def input_cone(grid, routing, in_pos):
    """The cells whose value can be influenced by an input — everything forward-
    reachable from the input seeds in the signal graph. On the nervous net there
    is no spontaneous activity, so every cell OUTSIDE this cone is silent for all
    time; simulating only the cone is exact and, on the unbounded field where
    growth may leave a large off-cone blob, far cheaper."""
    edges = signal_graph(grid, routing)          # u -> cells that read u
    seen  = set(p for p in in_pos if p in grid)
    stack = list(seen)
    while stack:
        for v in edges.get(stack.pop(), ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return seen


def _inject_stream_edges(sim, in_pos, streams, T):
    """Queue each contiguous high run as one physical wired-OR pulse."""
    stop = min(int(T), len(streams))
    for input_index, cell in enumerate(in_pos):
        tick = 0
        while tick < stop:
            if input_index >= len(streams[tick]) or not streams[tick][input_index]:
                tick += 1
                continue
            start = tick
            tick += 1
            while (tick < stop and input_index < len(streams[tick])
                   and streams[tick][input_index]):
                tick += 1
            duration = max(pulse_engine.WIDTH, (tick - start) * TICK)
            sim.inject_pulse(cell, start * TICK, duration)


def _run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True,
                 max_events=None, sample=True):
    """
    Run T ticks of the asynchronous pulse simulation. streams[t] = tuple of
    input bits (one per in_pos); a 0->1 transition injects a pulse edge onto
    that input cell's net (held 1s are one long pulse). out_pos is
    {role: (x,y)}. Returns (states, traces):
        states : list[T] of sampled {(x,y):0/1} activity maps
        traces : {role: [0/1 over T]}
    `prune` restricts the simulation to the input cone (exact for the nervous
    net); off-cone cells are absent from `states` and read as 0 by callers.
    """
    sub = grid
    if prune:
        cone = input_cone(grid, routing, in_pos)
        if len(cone) < len(grid):
            sub = {c: grid[c] for c in cone}
    sim    = PulseSim(sub, routing, max_events=max_events)
    if not sample:
        # Event/cadence fitness needs edges, not O(T*cells) display snapshots.
        _inject_stream_edges(sim, in_pos, streams, T)
        sim.advance_to(T * TICK)
        return [], {role: [] for role in out_pos}, sim.rise_times, sim.overflow
    states = []
    traces = {role: [] for role in out_pos}
    for t in range(T):
        # run past the end of the input streams with zero input (padding), so a
        # circuit that responds at a DELAY is still observed — its late events
        # fall in [len(streams), T) instead of off the end (see _obs_len).
        row = streams[t] if t < len(streams) else (0,) * len(in_pos)
        state = sim.step({in_pos[i]: row[i] for i in range(len(in_pos))})
        states.append(state)
        for role, p in out_pos.items():
            traces[role].append(state.get(p, 0) if p else 0)
        if sim.overflow:
            break
    # ``step`` stops at the midpoint used for display.  Flush the remaining
    # physical half-tick so scoring retains every edge in the requested horizon.
    if not sim.overflow:
        sim.advance_to(T * TICK)
    return states, traces, sim.rise_times, sim.overflow


def run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True):
    """Backward-compatible sampled playback API.

    Dynamics remain asynchronous; these tick samples are for display and legacy
    state-window targets.  Use :func:`run_nervous_events` for behavioral timing.
    """
    states, traces, _, _ = _run_nervous(
        grid, routing, in_pos, out_pos, streams, T, prune=prune)
    return states, traces


def run_nervous_events(grid, routing, in_pos, out_pos, streams, T, prune=True,
                       max_events=None, sample=True):
    """Run once and return ``(states, traces, rise_times, overflow)``.

    ``rise_times`` maps every simulated cell to its continuous leading-edge
    timestamps.  No tick quantisation is applied to this event record.
    """
    return _run_nervous(grid, routing, in_pos, out_pos, streams, T,
                        prune=prune, max_events=max_events, sample=sample)


# ── temporal scoring ───────────────────────────────────────────────────────────
# The expected trace decomposes into WINDOWS: maximal runs of a constant
# expected level (settle gaps of None separate them). Scoring is per-window,
# then balanced across the two levels, so every behavioural phase (off before
# set, hold after set, off after reset, ...) carries equal weight — a long hold
# can't drown out a missed reset, and a constant output caps at 0.5.
#
# A "store 1" window is scored PHASE-TOLERANTLY: nervous-net memory holds a bit
# as a pulse *circulating* in a loop, which reads as a ripple (e.g. 1010…) at
# any single cell — the honeycomb has no triangles, so you can't OR the phases
# back into a steady DC level. What matters is that the cell is *actively
# ringing* with no long silent gap, so a store-1 window scores by activity
# coverage (each tick counts if it or an immediate neighbour fires): a ripple
# scores 1.0, a single blip scores low, silence scores 0. A "store 0" window
# still demands true silence (exact), so a net that keeps ringing after reset
# is penalised. Isolated single-tick expectations (echo, oscillator) are
# length-1 windows, for which coverage reduces to exact per-tick matching.

def _expected_windows(exp):
    """[(level, [ticks])] — maximal constant-level runs of scored ticks."""
    wins, cur, lvl = [], [], None
    for t, e in enumerate(exp):
        if e is None:
            if cur:
                wins.append((lvl, cur))
            cur, lvl = [], None
        elif lvl is None or e == lvl:
            lvl = e
            cur.append(t)
        else:
            wins.append((lvl, cur))
            lvl, cur = e, [t]
    if cur:
        wins.append((lvl, cur))
    return wins


def _window_score(trace, lvl, ticks):
    """Score one constant-level window. lvl==0: fraction of silent ticks.
    lvl==1: activity coverage — fraction of ticks that fire or sit next to a
    firing tick OF THE SAME WINDOW (rewards sustained ringing, not a lone blip;
    echoes in the unscored settle gaps outside the window don't count)."""
    if lvl == 0:
        return sum(1 for t in ticks if not trace[t]) / len(ticks)
    tick_set = set(ticks)
    covered = sum(1 for t in ticks
                  if trace[t]
                  or (t - 1 in tick_set and trace[t - 1])
                  or (t + 1 in tick_set and trace[t + 1]))
    return covered / len(ticks)


def _role_trace_score(trace, exp):
    """Windowed, level-balanced score of one trace against one expected trace:
    mean per-window score within each level, averaged across levels. A constant
    output scores 0.5; a correctly ringing/quiet memory scores 1.0.
    (Retained for diagnostics; selection uses the precision/recall score below.)"""
    per_level = {0: [], 1: []}
    for lvl, ticks in _expected_windows(exp):
        ticks = [t for t in ticks if t < len(trace)]
        if ticks:
            per_level[lvl].append(_window_score(trace, lvl, ticks))
    parts = [sum(v) / len(v) for v in per_level.values() if v]
    return sum(parts) / len(parts) if parts else 0.0


# ── precision / recall scoring (the selection metric) ────────────────────────────
# Rather than scoring the huge, easy mass of expected-0 ticks (which lets a
# do-nothing output look good on rare-positive targets), score ONLY the highs:
#   recall    = expected-1 ticks that the output actually hits — punishes silence
#   precision = of the output's own pulses, the fraction that belong — punishes
#               always-high / spurious firing
# combined as their harmonic mean (F1). Two asymmetries make it fit this
# substrate: recall allows ±1 COVERAGE (memory is a circulating pulse that rings
# every other tick, so an expected-1 counts as hit if the cell fires on it or an
# adjacent tick), while precision is EXACT (a pulse landing on an expected-0 tick
# is a false positive) — without that, a dense oscillator target would let an
# always-high output through. Unscored (settle) ticks count for neither.

# Observe the output for longer than the expected window so a DELAYED circuit's
# late events are actually seen (else a positive shift pushes them off the end and
# they read as misses — the "delayed Q counted false" the user hit). The expected
# trace stays length T; the trace is run to _obs_len. Crucially, recall still
# counts ALL expected highs (never excludes any) — that is what stops a silent
# output from escaping via a huge shift that would slide every high out of view.
def _obs_len(ttarget):
    return 2 * ttarget.T


def _pr_counts(trace, exp, shift=0, tol=1):
    """Spike-event counts of an output trace vs an expected trace, under a latency
    `shift`: an expected event at tick e corresponds to an output event at e+shift
    (shift=0 is the identity — the ordinary aligned scoring).

    recall (±tol): each expected-1 is hit if the output fires within tol of e+shift
      (the ±1 ring tolerance — a stored bit rings every other tick). ALL expected
      highs are in the denominator (none excluded), so silence can't be shifted
      away — a delayed circuit is captured because the trace is observed past T.
    precision (shift-consistent): a pulse at tick a is scored iff its shift-mapped
      position a-shift lands in the expected trace's scored region, and it's good
      iff exp[a-shift]==1. So spurious pulses always cost; pulses past the window
      that map to no expected tick are unscored (beyond the target)."""
    n = len(exp)
    fires = set(t for t in range(len(trace)) if trace[t])
    exp_high = [t for t in range(n) if exp[t] == 1]
    tp_rec = sum(1 for e in exp_high
                 if any((e + shift + d) in fires for d in range(-tol, tol + 1)))
    scored_fires = [a for a in fires
                    if 0 <= a - shift < n and exp[a - shift] is not None]
    tp_prec = sum(1 for a in scored_fires if exp[a - shift] == 1)
    return tp_rec, len(exp_high), tp_prec, len(scored_fires)


def _f1(tp_rec, n_exp, tp_prec, n_act):
    rec  = tp_rec / n_exp  if n_exp else 1.0     # nothing to recall -> vacuously 1
    prec = tp_prec / n_act if n_act else 1.0     # fired nothing    -> vacuously 1
    return (2 * rec * prec / (rec + prec)) if (rec + prec) else 0.0


def _pr_score(trace, exp, shift=0):
    """Per-trace F1 of the highs (see _pr_counts). Silent -> 0 when anything is
    expected; always-high -> low; a correct ringing/quiet output -> 1."""
    return _f1(*_pr_counts(trace, exp, shift))


# ── latency-invariant scoring ────────────────────────────────────────────────────
# The ABSOLUTE input->output latency must NOT drive fitness: a circuit that
# produces the right spike pattern at a consistent but different delay than the
# target's arbitrarily-chosen one has captured the idea and should score the same.
# So the score is taken at the single best latency SHIFT — one value shared across
# every trial and role (a real circuit has one fixed propagation delay), found by
# maximising the pooled F1. This frees the absolute delay while still requiring
# CONSISTENT timing (one shift must fit all trials) and still penalising spurious
# firing (precision is anchored to the output's scored region — see _pr_counts).
# shift=0 is included, so the aligned score is a lower bound: nothing regresses.

def _pooled_f1(traces, ttarget, shift):
    tr = ne = tp = na = 0
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            seq = traces.get(role, [])
            if ti >= len(seq):
                continue
            a, b, c, d = _pr_counts(seq[ti], exp, shift)
            tr += a; ne += b; tp += c; na += d
    return _f1(tr, ne, tp, na)


def _cand_shifts(pairs, T, tol=1):
    """The only shifts worth trying: F1 is piecewise-constant in the shift and
    changes only where an output pulse aligns with an expected event, so the
    maximum is attained at some (pulse_tick - expected_tick) offset (± the ring
    tolerance), plus 0. For sparse spike targets this is far fewer than the whole
    range; for dense ones it dedups to at most the range (2T-1) anyway."""
    # positive shifts may run to the observed trace length (delayed events live
    # past T); negative shifts (faster than nominal) are bounded by T.
    max_len = max((len(tr) for tr, _ in pairs), default=T)
    lo, hi = -(T - 1), max(T - 1, max_len - 1)
    shifts = {0}
    for trace, exp in pairs:
        fires = [a for a in range(len(trace)) if trace[a]]
        highs = [e for e in range(len(exp)) if exp[e] == 1]
        for a in fires:
            for e in highs:
                for d in (-tol, 0, tol):
                    s = a - e + d
                    if lo <= s <= hi:
                        shifts.add(s)
    return shifts


def _target_pairs(traces, ttarget):
    pairs = []
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            seq = traces.get(role, [])
            if ti < len(seq):
                pairs.append((seq[ti], exp))
    return pairs


def _best_shift(traces, ttarget):
    """(best_shift, best_f1): the global latency offset maximising pooled F1."""
    best_s, best_f = 0, -1.0
    for s in _cand_shifts(_target_pairs(traces, ttarget), ttarget.T):
        f = _pooled_f1(traces, ttarget, s)
        if f > best_f:
            best_f, best_s = f, s
    return best_s, best_f


def _placement_score(cell_traces, exps, ttarget):
    """Best-shift pooled F1 for ONE candidate output cell (a single latency shift
    over its own traces) — the per-cell ranking used when PLACING outputs. This
    MUST be latency-invariant too: if placement ranked at a fixed alignment, which
    cell gets chosen would depend on the target's nominal latency, quietly
    reintroducing the timing-dependence the score removes (measured: it did).
    Same _cand_shifts trick as _best_shift keeps it cheap."""
    best = 0.0
    for s in _cand_shifts(list(zip(cell_traces, exps)), ttarget.T):
        tr = ne = tp = na = 0
        for trace, exp in zip(cell_traces, exps):
            a, b, c, d = _pr_counts(trace, exp, s)
            tr += a; ne += b; tp += c; na += d
        f = _f1(tr, ne, tp, na)
        if f > best:
            best = f
    return best


# ── raw point-event scoring ───────────────────────────────────────────────────

def sampled_events(trace):
    """Leading-edge times recovered from a clocked/sample trace.

    This is the LUT adapter.  The nervous backend supplies physical timestamps
    directly and therefore never needs this quantising fallback.
    """
    return [float(t) for t, value in enumerate(trace)
            if value and (t == 0 or not trace[t - 1])]


def _role_events(traces, role, trial_index):
    event_map = getattr(traces, 'events', {})
    seqs = event_map.get(role, ())
    if trial_index < len(seqs):
        return list(seqs[trial_index])
    dense = traces.get(role, ())
    return sampled_events(dense[trial_index]) if trial_index < len(dense) else []


def _expected_events(trial, role):
    explicit = getattr(trial, 'expected_events', {}).get(role)
    if explicit is not None:
        return [float(t) for t in explicit]
    # Legacy event targets used isolated expected-high samples.  Treat each one
    # as a point; state/cadence targets never enter this scorer.
    return [float(t) for t, value in enumerate(trial.expected.get(role, ()))
            if value == 1]


def _scored_ranges(exp):
    """Half-open tick ranges whose expectation is not ``None``."""
    ranges = []
    start = None
    for tick, value in enumerate(exp):
        if value is not None and start is None:
            start = tick
        elif value is None and start is not None:
            ranges.append((float(start), float(tick)))
            start = None
    if start is not None:
        ranges.append((float(start), float(len(exp))))
    return tuple(ranges)


def _event_counts(actual, expected, exp, shift=0.0, tolerance=0.5,
                  presorted=False, scored_ranges=None):
    """One-to-one ordered event matches under one global latency offset.

    The hot path receives pre-sorted tuples and performs a zero-allocation
    two-pointer scan.  ``presorted=False`` keeps the helper safe for diagnostics
    and external callers.
    """
    if not presorted:
        actual = tuple(sorted(float(a) for a in actual))
        expected = tuple(sorted(float(e) for e in expected))
    if scored_ranges is None:
        scored_ranges = _scored_ranges(exp)
    # Precision's denominator is a range count in C rather than a Python scan
    # over every produced edge for every candidate shift.
    tick_eps = 1e-9
    n_actual = sum(
        (bisect_left(actual, hi + shift - tick_eps)
         - bisect_left(actual, lo + shift - tick_eps))
        for lo, hi in scored_ranges)
    cursor = matches = 0
    limit = tolerance + 1e-9
    for wanted in expected:
        lo = wanted + shift - limit
        hi = wanted + shift + limit
        index = bisect_left(actual, lo, cursor)
        while index < len(actual) and actual[index] <= hi:
            mapped = actual[index] - shift
            tick = int(math.floor(mapped + 1e-9))
            if 0 <= tick < len(exp) and exp[tick] is not None:
                matches += 1
                cursor = index + 1
                break
            index += 1
        else:
            # Later expected events cannot use an earlier actual edge.
            cursor = max(cursor, index)
    return matches, len(expected), matches, n_actual


def _event_pairs(traces, ttarget):
    pairs = []
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            frozen_exp = tuple(exp)
            pairs.append((tuple(sorted(_role_events(traces, role, ti))),
                          tuple(sorted(_expected_events(trial, role))),
                          frozen_exp, _scored_ranges(frozen_exp)))
    return pairs


def _event_candidate_shifts(pairs, ttarget):
    """All exact/boundary alignments that can change tolerant matching."""
    tol = float(getattr(ttarget, 'event_tolerance', 0.5))
    limit = float(getattr(ttarget, 'event_max_shift', ttarget.T))
    shifts = {0.0}
    for actual, expected, _, _ in pairs:
        for a in actual:
            for e in expected:
                centre = float(a) - float(e)
                for value in (centre - tol, centre, centre + tol):
                    if -limit <= value <= limit:
                        shifts.add(round(value, 9))
    # Deterministic ties: smallest magnitude, with causal/non-negative first.
    return sorted(shifts, key=lambda s: (abs(s), s < 0.0, s))


def _pooled_event_f1(pairs, tolerance, shift):
    tr = ne = tp = na = 0
    for actual, expected, exp, ranges in pairs:
        a, b, c, d = _event_counts(
            actual, expected, exp, shift, tolerance, presorted=True,
            scored_ranges=ranges)
        tr += a; ne += b; tp += c; na += d
    return _f1(tr, ne, tp, na)


def _best_event_shift(traces, ttarget):
    cached = getattr(traces, '_event_result', None)
    if cached is not None:
        return cached
    if getattr(traces, 'overflow', False):
        return 0.0, 0.0
    pairs = _event_pairs(traces, ttarget)
    tolerance = float(getattr(ttarget, 'event_tolerance', 0.5))
    best_shift, best_score = 0.0, -1.0
    for shift in _event_candidate_shifts(pairs, ttarget):
        score = _pooled_event_f1(pairs, tolerance, shift)
        if score > best_score + 1e-12:
            best_shift, best_score = shift, score
    return best_shift, max(0.0, best_score)


def _event_case_score(traces, ttarget, trial_index, role, shift):
    if getattr(traces, 'overflow', False):
        return 0.0
    trial = ttarget.trials[trial_index]
    exp = trial.expected[role]
    counts = _event_counts(
        _role_events(traces, role, trial_index),
        _expected_events(trial, role), exp, shift,
        float(getattr(ttarget, 'event_tolerance', 0.5)))
    return _f1(*counts)


def event_score(traces, ttarget, shift=None):
    """Exact sparse point-event F1 under one shared continuous-time latency."""
    if getattr(traces, 'overflow', False):
        return 0.0
    if shift is None:
        return _best_event_shift(traces, ttarget)[1]
    cached = getattr(traces, '_event_result', None)
    if cached is not None and abs(float(shift) - cached[0]) <= 1e-12:
        return cached[1]
    pairs = _event_pairs(traces, ttarget)
    return _pooled_event_f1(
        pairs, float(getattr(ttarget, 'event_tolerance', 0.5)), shift)


# ── cadence semantics for autonomous oscillators/patterns ────────────────────

def _input_edges(streams, input_index=0):
    return [float(t) for t, row in enumerate(streams)
            if input_index < len(row) and row[input_index]
            and (t == 0 or input_index >= len(streams[t - 1])
                 or not streams[t - 1][input_index])]


def _display_time(value):
    """Compact, stable timestamp text for reports (3 instead of 3.0)."""
    value = float(value)
    return str(int(value)) if value.is_integer() else ('%.3f' % value).rstrip('0').rstrip('.')


def event_list_summary(events):
    """Readable timestamp list without Python string quotes around each value."""
    return '[%s]' % ', '.join(_display_time(event) for event in events)


def trial_input_summary(trial, n_inputs, names=None):
    """Describe stimulus edges, not every sampled high in a held pulse."""
    names = list(names or [chr(65 + i) for i in range(n_inputs)])
    parts = []
    for i in range(n_inputs):
        label = names[i] if i < len(names) else 'I%d' % i
        edges = ', '.join(_display_time(t) for t in _input_edges(trial.streams, i))
        parts.append('%s@[%s]' % (label, edges))
    return '  '.join(parts)


def expected_window_summary(expected):
    """Compress a sampled expectation into readable half-open time ranges."""
    if not expected:
        return '(empty)'
    labels = {None: 'ignore', 0: 'quiet', 1: 'active'}
    parts = []
    start = 0
    current = expected[0]
    sentinel = object()
    for end in range(1, len(expected) + 1):
        value = expected[end] if end < len(expected) else sentinel
        if value != current:
            parts.append('%s[%d:%d)' % (labels.get(current, str(current)), start, end))
            start, current = end, value
    return ', '.join(parts)


def _cadence_trial_score(events, trial, target, latency):
    kicks = _input_edges(trial.streams)
    if not kicks:
        return 0.0
    kick = kicks[0]
    start = kick + latency + float(getattr(target, 'cadence_settle', 5.0))
    end = float(target.T) + latency
    before = [e for e in events if e < kick - 1e-9]
    steady = [e for e in events if start <= e < end]
    minimum = int(getattr(target, 'cadence_min_events', 4))
    if len(steady) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(steady, steady[1:])]
    period = float(getattr(target, 'cadence_period', 0.0))
    tol = float(getattr(target, 'cadence_tolerance', 0.5))
    regular = sum(1 for gap in gaps if abs(gap - period) <= tol + 1e-9) / len(gaps)
    count = min(1.0, len(steady) / max(1, minimum))
    required_span = max(period, end - start - period)
    coverage = min(1.0, (steady[-1] - steady[0]) / required_span)
    quiet = 1.0 / (1.0 + len(before))
    return quiet * regular * count * coverage


def cadence_score(traces, ttarget):
    """Score a sustained rhythm independent of absolute output phase."""
    cached = getattr(traces, '_cadence_result', None)
    if cached is not None:
        return cached
    if getattr(traces, 'overflow', False):
        return 0.0, tuple(0.0 for _ in ttarget.trials)
    limit = float(getattr(ttarget, 'event_max_shift', 12.0))
    candidates = {0.0}
    for ti, trial in enumerate(ttarget.trials):
        kicks = _input_edges(trial.streams)
        if not kicks:
            continue
        for role in trial.expected:
            for event in _role_events(traces, role, ti):
                latency = (event - kicks[0]
                           - float(getattr(ttarget, 'cadence_settle', 5.0)))
                if 0.0 <= latency <= limit:
                    candidates.add(round(latency, 9))
    best_score, best_cases = -1.0, ()
    for latency in sorted(candidates):
        cases = []
        for ti, trial in enumerate(ttarget.trials):
            role_scores = [_cadence_trial_score(
                _role_events(traces, role, ti), trial, ttarget, latency)
                for role in trial.expected]
            cases.append(sum(role_scores) / len(role_scores) if role_scores else 0.0)
        score = sum(cases) / len(cases) if cases else 0.0
        if score > best_score + 1e-12:
            best_score, best_cases = score, tuple(cases)
    return max(0.0, best_score), best_cases


# ── semantic period-stepper scoring ───────────────────────────────────────────

def _pulse_events(trace, lo, hi):
    """Leading edges of high runs in [lo, hi).  A LUT can hold a bit high for
    several samples; that is still one pulse event for cadence measurement."""
    lo, hi = max(0, lo), min(len(trace), hi)
    return [t for t in range(lo, hi)
            if trace[t] and (t == 0 or not trace[t - 1])]


def _stepper_epoch(trace, lo, hi, target):
    """Return (quality, measured_period) for one stable command epoch.

    Quality requires enough pulses, regular gaps, and coverage across the
    dwell.  The phase is deliberately free: a cadence is a relation between
    pulses, not a requirement that a control edge arrive at one magic phase.
    """
    events = _pulse_events(trace, lo, hi)
    if len(events) < getattr(target, 'stepper_min_events', 4):
        return 0.0, None
    gaps = [b - a for a, b in zip(events, events[1:])]
    if not gaps:
        return 0.0, None
    p = int(round(sorted(gaps)[len(gaps) // 2]))
    if not (getattr(target, 'stepper_min_period', 2)
            <= p <= getattr(target, 'stepper_max_period', 6)):
        return 0.0, None
    # ±1 tick absorbs sampling/ringing phase without allowing a fundamentally
    # different cadence to masquerade as regular.
    regular = sum(1 for gap in gaps if abs(gap - p) <= 1) / len(gaps)
    dwell = hi - lo
    required_span = max(1, dwell - 2 * p)
    coverage = min(1.0, (events[-1] - events[0]) / required_span)
    return regular * coverage, p


def _stepper_trial_score(trace, streams, target, shift):
    """Score one command stream at a fixed causal input→output latency."""
    commands = [t for t, row in enumerate(streams)
                if row and row[0] and (t == 0 or not streams[t - 1][0])]
    if not commands:
        return 0.0
    # Starting before the first command is not a controlled oscillator.
    pre = _pulse_events(trace, 0, commands[0] + shift)
    pre_score = 1.0 / (1.0 + len(pre))
    epochs, periods = [], []
    settle = getattr(target, 'stepper_settle', 2)
    for i, command in enumerate(commands):
        lo = command + shift + settle
        hi = ((commands[i + 1] + shift) if i + 1 < len(commands)
              else target.T + shift)
        quality, period = _stepper_epoch(trace, lo, hi, target)
        epochs.append(quality)
        periods.append(period)
    cadence = sum(epochs) / len(epochs) if epochs else 0.0
    # Every subsequent command must make the sustained cadence slower.  A
    # period-2 oscillator that ignores commands therefore gets zero relation
    # credit even though it matches some of the slower epoch's raw pulses.
    relations = [1.0 if a is not None and b is not None and b > a else 0.0
                 for a, b in zip(periods, periods[1:])]
    relation = sum(relations) / len(relations) if relations else 1.0
    return pre_score * cadence * relation


def period_stepper_score(traces, target):
    """(score, per-trial scores) for the cadence-stepper target.

    A single bounded, causal latency is selected for all trials.  This retains
    the useful latency freedom of temporal scoring without allowing a giant
    negative shift to hide an early command epoch.
    """
    best_score, best_cases = -1.0, ()
    for shift in range(getattr(target, 'stepper_max_delay', 8) + 1):
        cases = []
        for ti, trial in enumerate(target.trials):
            role_scores = []
            for role in trial.expected:
                seq = traces.get(role, [])
                role_scores.append(_stepper_trial_score(
                    seq[ti] if ti < len(seq) else [], trial.streams,
                    target, shift))
            cases.append(sum(role_scores) / len(role_scores) if role_scores else 0.0)
        score = sum(cases) / len(cases) if cases else 0.0
        if score > best_score:
            best_score, best_cases = score, tuple(cases)
    return best_score, best_cases


# Selection metric. The fitness question is "did the network produce the correct
# SPIKE EVENTS?", not "was the output level right at every tick?" — so the metric
# is the precision/recall (F1) of the output spikes above:
#   recall    — expected spikes the output actually produced (missing ones cost)
#   precision — of the spikes it produced, the fraction that were expected
#               (extra, unexpected spikes cost)
# A do-nothing output recalls nothing, so it scores 0 the moment any spike is
# expected — silence is no longer rewarded. A fire-constantly output has terrible
# precision. Only the correct spikes at the correct ticks (and nowhere else) reach
# 1.0. The other two metrics are kept for diagnostics / experiments only:
#   'balanced' — mean of pooled expected-0 / expected-1 window scores (this is the
#                one that rewarded silence: a constant output scores 0.5)
#   'blend'    — average of f1 and balanced (the former default; diluted f1 with
#                balanced's partial credit for doing nothing)
METRIC = 'f1'


def _trace_metric(trace, exp, metric=None):
    m = metric or METRIC
    if m == 'f1':
        return _pr_score(trace, exp)
    if m == 'balanced':
        return _role_trace_score(trace, exp)
    return 0.5 * (_role_trace_score(trace, exp) + _pr_score(trace, exp))


# The output is read at a cell NEAR the target's terminal, not anywhere in the
# organism. On the unbounded field a net can be hundreds of cells, and scoring
# every one both costs O(cells) per genome and lets a lucky far-off cell inflate
# the fitness without a real mechanism (the output location then jitters between
# genomes, wrecking the gradient). Restricting to the OUT_RADIUS-nearest cells
# to the terminal keeps placement cheap and the fitness signal about a fixed,
# local output — the terminal is a designated read-out point, after all.
OUT_RADIUS = 12


def _score_output_candidate(sampled, events, expected, role, target,
                            overflow=False):
    """Score one prospective output and return ``(score, reusable result)``."""
    if not sampled:
        return 0.0, None
    mode = getattr(target, 'score_mode', 'trace')
    if mode == 'period_stepper':
        return period_stepper_score({role: sampled}, target)[0], None
    bundle = TemporalTraces({role: sampled}, events={role: events},
                            overflow=overflow)
    if mode == 'events':
        result = _best_event_shift(bundle, target)
        return result[1], result
    if mode == 'cadence':
        result = cadence_score(bundle, target)
        return result[0], result
    return _placement_score(sampled, expected, target), None


def _output_candidates(grid, in_set, term):
    tx, ty = term.pos
    cands = [c for c in grid if c not in in_set]
    cands.sort(key=lambda c: (abs(c[0] - tx) + abs(c[1] - ty), c))
    return cands[:OUT_RADIUS]


def place_outputs_by_trace(grid, routing, in_pos, ttarget):
    """Assign each output role the live non-input cell — among those nearest its
    terminal (see OUT_RADIUS) — whose activity trace best matches the expected
    trace across ALL trials (ties broken by distance to the terminal, then cell
    order). Evolution builds the computation and lands it near the read-out.

    Returns (out_pos {role: (x,y)|None}, traces {role: [trace per trial]}).
    """
    in_set = set(in_pos)
    out_pos = {term.role: None for term in ttarget.outputs}
    traces  = TemporalTraces()
    if len(grid) <= len(in_set):
        return out_pos, traces
    # one dynamics run per trial, observed to _obs_len (past T) so a delayed
    # output's late events are captured; every cell's trace falls out of the states
    obs = _obs_len(ttarget)
    mode = getattr(ttarget, 'score_mode', 'trace')
    need_samples = mode not in ('events', 'cadence')
    # Topology is identical across trials.  Computing the exact reachable cone
    # once avoids rebuilding the signal graph for every stimulus schedule.
    cone = input_cone(grid, routing, in_pos)
    sub = grid if len(cone) == len(grid) else {c: grid[c] for c in cone}
    runs = [run_nervous_events(
                sub, routing, in_pos, {}, tr.streams, obs, prune=False,
                max_events=getattr(ttarget, 'max_events', 2048),
                sample=need_samples)
            for tr in ttarget.trials]
    trial_states = [run[0] for run in runs]
    trial_events = [run[2] for run in runs]
    traces.overflow = any(run[3] for run in runs)
    used = set()
    for term in ttarget.outputs:
        best, best_key, best_aux = None, None, None
        score_cache = {}
        for c in _output_candidates(grid, in_set, term):
            if c in used:
                continue
            ctr, cevents, cexp = [], [], []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                ctr.append([trial_states[ti][t].get(c, 0) for t in range(obs)]
                           if need_samples else [])
                cevents.append(list(trial_events[ti].get(c, ())))
                cexp.append(exp)
            if not ctr:
                s, aux = 0.0, None
            else:
                source = cevents if mode in ('events', 'cadence') else ctr
                signature = tuple(tuple(seq) for seq in source)
                cached = score_cache.get(signature)
                s, aux = cached if cached is not None else (None, None)
            if ctr and s is None:
                s, aux = _score_output_candidate(
                    ctr, cevents, cexp, term.role, ttarget, traces.overflow)
            if ctr:
                score_cache[signature] = (s, aux)
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best, best_aux = key, c, aux
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        traces[term.role] = [
            [trial_states[ti][t].get(best, 0) for t in range(obs)]
            if need_samples else []
            for ti in range(len(ttarget.trials))]
        traces.events[term.role] = [list(trial_events[ti].get(best, ()))
                                    for ti in range(len(ttarget.trials))]
        if len(ttarget.outputs) == 1 and mode == 'events':
            traces._event_result = best_aux
        elif len(ttarget.outputs) == 1 and mode == 'cadence':
            traces._cadence_result = best_aux
    return out_pos, traces


def prepare_net(genome, ttarget):
    """Grow + interpret a genome for a temporal target, placing outputs by
    trace match. Returns (grid, routing, in_pos, out_pos, traces) — where
    traces[role][i] is the chosen cell's trace in trial i — or None if the net
    is unusable (too small, no candidate output cells, or an input seed dead)."""
    grid = grow_nervous(genome, seeds=tuple(ttarget.inputs),
                        grid_size=ttarget.grid_size, iters=ttarget.iters)
    if len(grid) <= ttarget.n_inputs:
        return None
    routing, in_pos, _ = interpret_nervous(grid, ttarget)
    if any(p not in grid for p in in_pos):
        return None
    out_pos, traces = place_outputs_by_trace(grid, routing, in_pos, ttarget)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, routing, in_pos, out_pos, traces


def windowed_score(traces, ttarget, metric=None, shift=None):
    """Selection fitness core, pooled globally over every (trial, role) under
    METRIC (or an explicit `metric`):
      f1       — spike-event precision/recall (default; silent = 0, always-high low)
      balanced — pooled expected-0 vs expected-1 window scores, averaged (diagnostic)
      blend    — mean of the two (diagnostic)
    The f1 score is LATENCY-INVARIANT: taken at the best global input->output shift
    (see _best_shift), so the absolute delay doesn't matter, only consistent timing
    and no spurious firing. `shift` skips the search (pass the value from
    _best_shift when it's already known). 1.0 iff every expected spike is produced
    (ringing allowed, any consistent latency) and nothing fires where it shouldn't."""
    mode = getattr(ttarget, 'score_mode', 'trace')
    if mode == 'events':
        return event_score(traces, ttarget, shift=shift)
    if mode == 'cadence':
        return cadence_score(traces, ttarget)[0]
    if mode == 'period_stepper':
        return period_stepper_score(traces, ttarget)[0]
    m = metric or METRIC
    if m in ('f1', 'blend'):
        f1 = (_pooled_f1(traces, ttarget, shift) if shift is not None
              else _best_shift(traces, ttarget)[1])
    else:
        f1 = 0.0
    if m == 'f1':
        return f1
    per_level = {0: [], 1: []}
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            if ti >= len(tr):
                continue
            trace = tr[ti]
            for lvl, ticks in _expected_windows(exp):
                ticks = [t for t in ticks if t < len(trace)]
                if ticks:
                    per_level[lvl].append(_window_score(trace, lvl, ticks))
    parts = [sum(v) / len(v) for v in per_level.values() if v]
    bal = sum(parts) / len(parts) if parts else 0.0
    if m == 'balanced':
        return bal
    return 0.5 * (bal + f1)


def score_temporal_bundle(traces, ttarget):
    """Return ``(scalar, per-trial-role cases, alignment)`` for any score mode.

    Nervous and LUT evolution share this dispatcher so scalar fitness and
    lexicase case vectors cannot drift into subtly different semantics.
    ``alignment`` is the selected global event/trace shift, or ``None`` for
    invariant scorers that do not use one.
    """
    n_cases = sum(len(trial.expected) for trial in ttarget.trials)
    if getattr(traces, 'overflow', False):
        return 0.0, (0.0,) * n_cases, None
    mode = getattr(ttarget, 'score_mode', 'trace')
    if mode == 'period_stepper':
        score, cases = period_stepper_score(traces, ttarget)
        return score, tuple(cases), None
    if mode == 'cadence':
        score, cases = cadence_score(traces, ttarget)
        return score, tuple(cases), None
    if mode == 'events':
        shift, score = _best_event_shift(traces, ttarget)
        cases = tuple(
            _event_case_score(traces, ttarget, ti, role, shift)
            for ti, trial in enumerate(ttarget.trials)
            for role in trial.expected)
        return score, cases, shift

    shift, _ = _best_shift(traces, ttarget)
    cases = []
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            role_traces = traces.get(role, ())
            if ti >= len(role_traces):
                cases.append(0.0)
            elif METRIC == 'f1':
                cases.append(_pr_score(role_traces[ti], exp, shift))
            else:
                cases.append(_trace_metric(role_traces[ti], exp))
    return windowed_score(traces, ttarget, shift=shift), tuple(cases), shift


def exact_tick_accuracy(traces, ttarget):
    """Plain fraction of scored ticks matched exactly (no phase tolerance).
    Diagnostic only — a working circulating-pulse latch ripples, so this reads
    below 1.0 even when the memory is behaviourally correct."""
    correct = total = 0
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            tr_i = tr[ti] if ti < len(tr) else []
            for t in range(min(ttarget.T, len(exp), len(tr_i))):
                if exp[t] is None:
                    continue
                total += 1
                if tr_i[t] == exp[t]:
                    correct += 1
    return correct / total if total else 0.0


def score_temporal(genome, ttarget):
    """Behavioural fitness in [0,1] using the target's declared score mode."""
    prep = prepare_net(genome, ttarget)
    if prep is None:
        return 0.0
    _, _, _, _, traces = prep
    return score_temporal_bundle(traces, ttarget)[0]


def temporal_report(ttarget, genome=None):
    """Human-readable explanation of a temporal target for the GUI: what the
    circuit must do, its stimulus tests, and — when a genome is given — the
    evolved net's actual behaviour with a score for each test."""
    lines = ['Target: %s   [temporal nervous net]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    if getattr(ttarget, 'score_mode', 'trace') == 'period_stepper':
        lines += ['', 'Backend detail: nervous-net transition phase may be sub-tick.']
        if genome is None:
            return '\n'.join(lines + ['', '(run the GA or Load Saved to inspect a circuit)'])
        prep = prepare_net(genome, ttarget)
        if prep is None:
            return '\n'.join(lines + ['', '(circuit incomplete - grew too little or inputs dead)'])
        _, _, _, out_pos, traces = prep
        total, cases = period_stepper_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos[term.role]))
        for ti, (trial, score) in enumerate(zip(ttarget.trials, cases), 1):
            lines.append('Test %d: %s  cadence score %.3f %s' % (
                ti, trial_input_summary(trial, ttarget.n_inputs), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines.append('')
        lines.append('=> cadence-stepper score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else ''))
        return '\n'.join(lines)
    if getattr(ttarget, 'score_mode', 'trace') == 'cadence':
        lines += ['', 'Backend detail: cadence uses raw nervous-net edge timestamps.']
        if genome is None:
            return '\n'.join(lines + ['', '(run the GA or Load Saved to inspect a circuit)'])
        prep = prepare_net(genome, ttarget)
        if prep is None:
            return '\n'.join(lines + ['', '(circuit incomplete - grew too little or inputs dead)'])
        _, _, _, out_pos, traces = prep
        total, cases = cadence_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos[term.role]))
        for ti, score in enumerate(cases):
            events = _role_events(traces, ttarget.outputs[0].role, ti)
            lines.append('Test %d: %s  output edges@%s  cadence score %.3f %s' % (
                ti + 1, trial_input_summary(ttarget.trials[ti], ttarget.n_inputs),
                event_list_summary(events), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return '\n'.join(lines)
    if getattr(ttarget, 'score_mode', 'trace') == 'events':
        lines += ['', 'Backend detail: raw nervous-net timestamps retain sub-tick edges.']
        prep = None if genome is None else prepare_net(genome, ttarget)
        traces = prep[4] if prep is not None else None
        best_s = _best_event_shift(traces, ttarget)[0] if traces is not None else 0.0
        if prep is not None:
            for term in ttarget.outputs:
                lines.append("out '%s' read at %s" % (term.role, prep[3][term.role]))
            lines.append('measured output latency offset: %+.3f' % best_s)
        for ti, trial in enumerate(ttarget.trials):
            lines.append('')
            for role in trial.expected:
                expected = _expected_events(trial, role)
                lines.append('Test %d: %s' % (
                    ti + 1, trial_input_summary(trial, ttarget.n_inputs)))
                lines.append('  expect %s edges@%s' % (
                    role, event_list_summary(expected)))
                if traces is not None:
                    actual = _role_events(traces, role, ti)
                    score = _event_case_score(traces, ttarget, ti, role, best_s)
                    lines.append('        actual edges@%s  F1 %.3f %s' % (
                        event_list_summary(actual), score,
                        'PASS' if score >= 0.999 else 'FAIL'))
        if traces is not None:
            total = event_score(traces, ttarget, shift=best_s)
            lines += ['', '=> event score %.4f%s' % (
                total, '   SOLVED' if total >= 0.999 else '')]
        else:
            lines += ['', '(run the GA or Load Saved to inspect raw events)']
        return '\n'.join(lines)
    lines += ['',
              'Scored as persistent behaviour in active and quiet windows.',
              'A stored 1 may ring (1010...) because a nervous node is refractory;',
              'the scorer therefore tolerates one tick of ring phase.',
              'One shared input-to-output latency is used across every test/output.',
              'The exact per-tick value is diagnostic only and ignores ring tolerance.']

    prep = traces = None
    best_s = 0
    if genome is not None:
        prep = prepare_net(genome, ttarget)
        if prep is None:
            lines += ['', '(circuit incomplete — grew too little or inputs dead)']
        else:
            _, _, _, out_pos, traces = prep
            best_s = _best_shift(traces, ttarget)[0]
            for t in ttarget.outputs:
                lines.append("out '%s' read at %s" % (t.role, out_pos[t.role]))
            lines.append('measured output latency offset: %+d tick(s)' % best_s)

    for ti, trial in enumerate(ttarget.trials):
        lines += ['', 'Test %d: %s' % (
            ti + 1, trial_input_summary(trial, ttarget.n_inputs))]
        for role, exp in trial.expected.items():
            lines.append('  expect %s%s' % (
                expected_window_summary(exp),
                ' (%s)' % role if len(trial.expected) > 1 else ''))
            if traces is not None:
                tr = traces.get(role, [])
                tr_i = tr[ti] if ti < len(tr) else []
                # per-trial F1 at the genome's GLOBAL latency shift (so the display
                # matches the latency-invariant score, not a fixed alignment)
                tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp, best_s)
                s = _f1(tp_rec, n_exp, tp_prec, n_act)
                lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                             % (''.join(str(v) for v in tr_i), s,
                                'PASS' if s >= 0.999 else 'FAIL',
                                tp_rec, n_exp, tp_prec, n_act))
    if traces is not None:
        total = windowed_score(traces, ttarget, shift=best_s)
        lines += ['', '=> behavioural score %.4f%s   (exact per-tick %.4f)'
                  % (total, '   SOLVED' if total >= 0.999 else '',
                     exact_tick_accuracy(traces, ttarget))]
    else:
        lines += ['', '(run the GA or Load Saved to see the evolved traces here;',
                  ' drive it live in the Interactive tab with Step / Run)']
    return '\n'.join(lines)


# ── feedback-loop analysis (for the GA's loop-aware shaping) ────────────────────

def signal_graph(grid, routing):
    """Directed signal graph as {node: set(readers)}: u -> v iff node v's
    routing reads neighbour u (excitatory or inhibitory)."""
    edges = {c: set() for c in grid}
    for v, entry in routing.items():
        e1, e2, i1 = entry[0], entry[1], entry[2]
        nb = hex_dirs(*v)
        for d in (e1, e2, i1):
            if d is None:
                continue
            u = nb[d]
            if u in edges:
                edges[u].add(v)
    return edges


def _reachable(edges, sources):
    seen, stack = set(), [s for s in sources if s in edges]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(edges[u])
    return seen


def cycle_nodes(edges):
    """Nodes that lie on a directed cycle (can reach themselves). Grids are
    tiny (<= ~50 nodes), so per-node reachability is plenty fast."""
    on_cycle = set()
    for s in edges:
        seen, stack = set(), list(edges[s])
        while stack:
            u = stack.pop()
            if u == s:
                on_cycle.add(s)
                break
            if u in seen or u not in edges:
                continue
            seen.add(u)
            stack.extend(edges[u])
    return on_cycle


def loop_profile(grid, routing, in_pos, out_pos):
    """Feedback-loop stats of a grown net:
        n_cycle    — nodes on any directed cycle
        n_relevant — cycle nodes both writable from an input and driving an output
    A latch is exactly a relevant cycle: inputs can set/clear the circulating
    value and the output can read it."""
    edges = signal_graph(grid, routing)
    cyc   = cycle_nodes(edges)
    if not cyc:
        return {'n_cycle': 0, 'n_relevant': 0}
    from_in = _reachable(edges, in_pos)
    rev = {c: set() for c in edges}
    for u, vs in edges.items():
        for v in vs:
            rev[v].add(u)
    to_out = _reachable(rev, [p for p in out_pos.values() if p])
    return {'n_cycle': len(cyc), 'n_relevant': len(cyc & from_in & to_out)}
