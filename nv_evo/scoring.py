"""
nv_evo/scoring.py — THE scoring contract: every fitness number in the project
comes from this module.

Seven historical score modes collapse into two semantic families, declared in
one registry (``RELATIONS``) that every dispatch site reads:

Family "match" — compare the output against per-trial expectations, all
projections of one canonical observable (the output's physical intervals):
  * rises      (mode 'events')   — leading-edge F1 under one fitted global
                                   shift (``event_tolerance``, physical time).
  * intervals  (mode 'waveform') — rise AND fall matched (width-sensitive
                                   contracts; ``waveform_tolerance``).
  * coverage   (modes 'trace', 'retention', 'sr_retention') — sustained
                 activity over expected spans. Tick-domain coverage scores
                 hold targets with a substrate-declared gap (``hold_tol``:
                 1 = nv circulating-pulse ring, 0 = LUT level hold);
                 float-domain coverage (the retention scorers) judges
                 commanded (state, start, end) intervals on raw rise trains.

Family "rhythm" — measure an invariant rather than match a reference; phase is
legitimately free and event counts unbounded, so forcing these through the
matcher would create degeneracy, not remove it:
  * free_rhythm      (mode 'cadence')        — sustained period regularity.
  * commanded_rhythm (mode 'period_stepper') — command-stepped cadence.

The mode STRINGS are permanent aliases (checkpoints and saved targets carry
them); the registry is the single authority on what each means. Alignment
discipline is shared: one global shift fitted on training data, freezable via
``score_temporal_bundle(..., alignment=...)`` for held-out scoring.

Simulation, growth, and placement live in temporal.py / persistence.py; this
module is pure scoring math over prepared observables.
"""
from __future__ import annotations
from bisect import bisect_left
import math

from . import pulse
from . import pulse as pulse_engine


# ── the relation registry (single source of truth for mode semantics) ────────

class RelationSpec:
    """What a score-mode string means. ``family``/``relation`` name the
    semantics; ``observable`` names what the harness must collect ('samples',
    'rises', 'intervals', or 'schedule' for the genome-level retention
    pipeline); ``evaluator`` names the evaluation pipeline ('bundle' = scored
    from a TemporalTraces bundle here, otherwise the persistence.py entry
    point of that name)."""

    __slots__ = ('family', 'relation', 'observable', 'evaluator')

    def __init__(self, family, relation, observable, evaluator):
        self.family = family
        self.relation = relation
        self.observable = observable
        self.evaluator = evaluator


RELATIONS = {
    'events': RelationSpec('match', 'rises', 'rises', 'bundle'),
    'trace': RelationSpec('match', 'coverage', 'samples', 'bundle'),
    'waveform': RelationSpec('match', 'intervals', 'intervals', 'bundle'),
    'retention': RelationSpec('match', 'coverage', 'schedule', 'retention'),
    'sr_retention': RelationSpec('match', 'coverage', 'schedule',
                                 'sr_retention'),
    'cadence': RelationSpec('rhythm', 'free_rhythm', 'rises', 'bundle'),
    'period_stepper': RelationSpec('rhythm', 'commanded_rhythm', 'samples',
                                   'bundle'),
}


def relation_spec(target):
    """The registry entry for a target's declared score mode."""
    mode = getattr(target, 'score_mode', 'trace')
    try:
        return RELATIONS[mode]
    except KeyError:
        raise ValueError('unknown score mode: %r' % (mode,))


def needs_samples(target):
    """Whether scoring this target consumes per-tick samples (as opposed to
    raw edge timestamps / intervals). Drives what the simulation collects."""
    return relation_spec(target).observable == 'samples'


class PhysicalEvents(dict):
    """Leading-edge map carrying the matching complete physical intervals."""

    def __init__(self, *args, intervals=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervals = intervals or {}


class TemporalTraces(dict):
    """Sampled role traces plus the raw point events that produced them.

    It deliberately remains a ``dict`` so existing GUI/report/LUT callers keep
    working.  Event-aware scorers read ``events[role][trial]``; old scorers read
    the ordinary sampled lists.  ``overflow`` invalidates a genome whose event
    count exceeded the target's deterministic safety cap.
    """

    def __init__(self, *args, events=None, intervals=None, overflow=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events or {}
        self.intervals = intervals or {}
        self.overflow = bool(overflow)
        self._event_result = None
        self._cadence_result = None
        self._stepper_result = None
        self._waveform_result = None


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


def _pr_score(trace, exp, shift=0, tol=1):
    """Per-trace F1 of the highs (see _pr_counts). Silent -> 0 when anything is
    expected; always-high -> low; a correct ringing/quiet output -> 1. `tol` is
    the recall coverage window: 1 for the nervous net (a stored bit RINGS 1010,
    so ±1 counts it as held), 0 for the LUT array (a latch can hold a steady
    level, so a hold must be genuinely high on every tick — a 2-3 tick burst is
    not a 5-tick hold)."""
    return _f1(*_pr_counts(trace, exp, shift, tol))


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

def _pooled_f1(traces, ttarget, shift, tol=1):
    tr = ne = tp = na = 0
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            seq = traces.get(role, [])
            if ti >= len(seq):
                continue
            a, b, c, d = _pr_counts(seq[ti], exp, shift, tol)
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


def _best_shift(traces, ttarget, tol=1):
    """(best_shift, best_f1): the global latency offset maximising pooled F1."""
    best_s, best_f = 0, -1.0
    for s in _cand_shifts(_target_pairs(traces, ttarget), ttarget.T, tol):
        f = _pooled_f1(traces, ttarget, s, tol)
        if f > best_f:
            best_f, best_s = f, s
    return best_s, best_f


def _placement_score(cell_traces, exps, ttarget, tol=1):
    """Best-shift pooled F1 for ONE candidate output cell (a single latency shift
    over its own traces) — the per-cell ranking used when PLACING outputs. This
    MUST be latency-invariant too: if placement ranked at a fixed alignment, which
    cell gets chosen would depend on the target's nominal latency, quietly
    reintroducing the timing-dependence the score removes (measured: it did).
    Same _cand_shifts trick as _best_shift keeps it cheap. `tol` matches the
    substrate (0 = strict LUT hold, 1 = nervous-net ring) so placement picks the
    same cell the final score rewards."""
    best = 0.0
    for s in _cand_shifts(list(zip(cell_traces, exps)), ttarget.T, tol):
        tr = ne = tp = na = 0
        for trace, exp in zip(cell_traces, exps):
            a, b, c, d = _pr_counts(trace, exp, s, tol)
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
    if not getattr(ttarget, 'fit_latency', True):
        return [0.0]
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


# ── complete physical-waveform scoring ───────────────────────────────────────

def _role_intervals(traces, role, trial_index):
    seqs = getattr(traces, 'intervals', {}).get(role, ())
    if trial_index >= len(seqs):
        return []
    return [(float(start), float(end)) for start, end in seqs[trial_index]
            if math.isfinite(start) and end > start]


def _interval_case_score(actual, expected, tolerance=0.25):
    """Ordered full-waveform similarity with exact 1.0 only at both boundaries.

    Start and width are scored separately. Extra/missing intervals enlarge the
    denominator, preventing a correct first pulse from hiding waveform clutter.
    The exponential boundary gradient remains useful to evolution without
    granting a perfect score to an incorrect duration.
    """
    actual = sorted((float(a), float(b)) for a, b in actual)
    expected = sorted((float(a), float(b)) for a, b in expected)
    count = max(len(actual), len(expected))
    if not count:
        return 1.0
    scale = max(1e-9, float(tolerance))
    score = 0.0
    for (a0, a1), (e0, e1) in zip(actual, expected):
        start_quality = math.exp(-abs(a0 - e0) / scale)
        width_quality = math.exp(-abs((a1 - a0) - (e1 - e0)) / scale)
        score += 0.5 * (start_quality + width_quality)
    return score / count


def _waveform_expected(ttarget, trial, role):
    contract = getattr(ttarget, 'waveform_contract', '')
    config = getattr(ttarget, 'pulse_config', None) or pulse_engine.PulseConfig()
    expected = trial.expected_intervals.get(role, ())
    if contract and trial.input_events:
        delay = config.delay * float(ttarget.waveform_delay_multiplier)
        if contract == 'width_sum':
            lanes = trial.input_events
            if len(lanes) >= 2 and lanes[0] and lanes[1]:
                pulses = [lanes[0][0], lanes[1][0]]
                anchor = max(start for start, _ in pulses)
                width = sum(width for _, width in pulses)
                expected = [(anchor + delay, anchor + delay + width)]
            else:
                expected = []
        elif contract == 'odd_selector':
            source = sorted(trial.input_events[0])
            expected = [(start + delay, start + delay + width)
                        for index, (start, width) in enumerate(source)
                        if index % 2 == 0]
        elif contract == 'preserve':       # compatibility with brief V3 targets
            source = trial.input_events[0]
            expected = [(start + delay, start + delay + width)
                        for start, width in source]
        else:
            source = trial.input_events[0]
            width = config.width * float(ttarget.waveform_width_multiplier)
            expected = [(start + delay, start + delay + width)
                        for start, _ in source]
    return expected


def _waveform_at_shift(traces, ttarget, shift):
    cases = []
    tolerance = float(getattr(ttarget, 'waveform_tolerance', 0.25))
    for ti, trial in enumerate(ttarget.trials):
        for role in trial.expected:
            expected = [(start + shift, end + shift)
                        for start, end in _waveform_expected(
                            ttarget, trial, role)]
            cases.append(_interval_case_score(
                _role_intervals(traces, role, ti),
                expected, tolerance))
    return ((sum(cases) / len(cases)) if cases else 0.0, tuple(cases))


def _best_waveform_shift(traces, ttarget):
    cached = getattr(traces, '_waveform_result', None)
    if cached is not None:
        return cached
    if not getattr(ttarget, 'fit_latency', True):
        score, cases = _waveform_at_shift(traces, ttarget, 0.0)
        return 0.0, score, cases
    limit = float(getattr(ttarget, 'event_max_shift', ttarget.T))
    candidates = {0.0}
    for ti, trial in enumerate(ttarget.trials):
        for role in trial.expected:
            actual = _role_intervals(traces, role, ti)
            expected = _waveform_expected(ttarget, trial, role)
            for a_start, _ in actual:
                for e_start, _ in expected:
                    shift = a_start - e_start
                    if -limit <= shift <= limit:
                        candidates.add(round(shift, 9))
    best = (0.0, -1.0, ())
    for shift in sorted(candidates, key=lambda value: (abs(value), value < 0.0)):
        score, cases = _waveform_at_shift(traces, ttarget, shift)
        if score > best[1] + 1e-12:
            best = (shift, score, cases)
    traces._waveform_result = best
    return best


def waveform_score(traces, ttarget, shift=None):
    """Score complete intervals under one shared response-latency offset."""
    if shift is None:
        _, score, cases = _best_waveform_shift(traces, ttarget)
        return score, cases
    return _waveform_at_shift(traces, ttarget, float(shift))


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
        physical = getattr(trial, 'input_events', None)
        times = ([event[0] for event in physical[i]]
                 if physical is not None and i < len(physical)
                 else _input_edges(trial.streams, i))
        edges = ', '.join(_display_time(t) for t in times)
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


def _cadence_at_latency(traces, ttarget, latency):
    cases = []
    for ti, trial in enumerate(ttarget.trials):
        role_scores = [_cadence_trial_score(
            _role_events(traces, role, ti), trial, ttarget, latency)
            for role in trial.expected]
        cases.append(sum(role_scores) / len(role_scores) if role_scores else 0.0)
    score = sum(cases) / len(cases) if cases else 0.0
    return score, tuple(cases)


def _best_cadence_latency(traces, ttarget):
    cached = getattr(traces, '_cadence_result', None)
    if cached is not None:
        return cached
    if getattr(traces, 'overflow', False):
        return 0.0, 0.0, tuple(0.0 for _ in ttarget.trials)
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
    best_latency, best_score, best_cases = 0.0, -1.0, ()
    for latency in sorted(candidates):
        score, cases = _cadence_at_latency(traces, ttarget, latency)
        if score > best_score + 1e-12:
            best_latency, best_score, best_cases = latency, score, cases
    result = best_latency, max(0.0, best_score), best_cases
    if hasattr(traces, '_cadence_result'):
        traces._cadence_result = result
    return result


def cadence_score(traces, ttarget, latency=None):
    """Score sustained rhythm, optionally at one pre-fitted startup latency."""
    if getattr(traces, 'overflow', False):
        return 0.0, tuple(0.0 for _ in ttarget.trials)
    if latency is not None:
        return _cadence_at_latency(traces, ttarget, float(latency))
    _, score, cases = _best_cadence_latency(traces, ttarget)
    return score, cases


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


def _stepper_at_shift(traces, target, shift):
    cases = []
    for ti, trial in enumerate(target.trials):
        role_scores = []
        for role in trial.expected:
            seq = traces.get(role, [])
            role_scores.append(_stepper_trial_score(
                seq[ti] if ti < len(seq) else [], trial.streams,
                target, int(shift)))
        cases.append(sum(role_scores) / len(role_scores) if role_scores else 0.0)
    score = sum(cases) / len(cases) if cases else 0.0
    return score, tuple(cases)


def _best_stepper_shift(traces, target):
    cached = getattr(traces, '_stepper_result', None)
    if cached is not None:
        return cached
    best_shift, best_score, best_cases = 0, -1.0, ()
    for shift in range(getattr(target, 'stepper_max_delay', 8) + 1):
        score, cases = _stepper_at_shift(traces, target, shift)
        if score > best_score:
            best_shift, best_score, best_cases = shift, score, cases
    result = best_shift, best_score, best_cases
    if hasattr(traces, '_stepper_result'):
        traces._stepper_result = result
    return result


def period_stepper_score(traces, target, shift=None):
    """(score, per-trial scores) for the cadence-stepper target.

    A single bounded, causal latency is selected for all trials.  This retains
    the useful latency freedom of temporal scoring without allowing a giant
    negative shift to hide an early command epoch.
    """
    if shift is not None:
        return _stepper_at_shift(traces, target, int(shift))
    _, score, cases = _best_stepper_shift(traces, target)
    return score, cases


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


def _score_output_candidate(sampled, events, expected, role, target,
                            overflow=False, tol=1, intervals=None):
    """Score one prospective output and return ``(score, reusable result)``.
    `tol` = the substrate's hold coverage (0 strict for LUT, 1 ring for nv)."""
    if not sampled:
        return 0.0, None
    mode = getattr(target, 'score_mode', 'trace')
    if mode == 'period_stepper':
        return period_stepper_score({role: sampled}, target)[0], None
    intervals = intervals or [[] for _ in sampled]
    bundle = TemporalTraces({role: sampled}, events={role: events},
                            intervals={role: intervals},
                            overflow=overflow)
    if mode == 'events':
        result = _best_event_shift(bundle, target)
        return result[1], result
    if mode == 'cadence':
        result = _best_cadence_latency(bundle, target)
        return result[1], result
    if mode == 'waveform':
        return waveform_score(bundle, target)[0], None
    return _placement_score(sampled, expected, target, tol), None


def windowed_score(traces, ttarget, metric=None, shift=None, tol=None):
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
    if mode == 'waveform':
        return waveform_score(traces, ttarget, shift=shift)[0]
    if mode == 'cadence':
        return cadence_score(traces, ttarget)[0]
    if mode == 'period_stepper':
        return period_stepper_score(traces, ttarget)[0]
    if tol is None:
        tol = getattr(traces, 'hold_tol', 1)   # 0 = strict LUT hold, 1 = nv ring
    m = metric or METRIC
    if m in ('f1', 'blend'):
        f1 = (_pooled_f1(traces, ttarget, shift, tol) if shift is not None
              else _best_shift(traces, ttarget, tol)[1])
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


_REFIT_ALIGNMENT = object()


def score_temporal_bundle(traces, ttarget, alignment=_REFIT_ALIGNMENT):
    """Return ``(scalar, per-trial-role cases, alignment)`` for any score mode.

    Nervous and LUT evolution share this dispatcher so scalar fitness and
    lexicase case vectors cannot drift into subtly different semantics.
    By default the best global alignment is fitted. Passing an explicit
    ``alignment`` freezes that training choice for validation data. The return
    value always includes the alignment actually used.
    """
    n_cases = sum(len(trial.expected) for trial in ttarget.trials)
    if getattr(traces, 'overflow', False):
        used = None if alignment is _REFIT_ALIGNMENT else alignment
        return 0.0, (0.0,) * n_cases, used
    mode = getattr(ttarget, 'score_mode', 'trace')
    if mode == 'waveform':
        if alignment is _REFIT_ALIGNMENT:
            used, score, cases = _best_waveform_shift(traces, ttarget)
        else:
            used = float(alignment)
            score, cases = waveform_score(traces, ttarget, shift=used)
        return score, tuple(cases), used
    if mode == 'period_stepper':
        if alignment is _REFIT_ALIGNMENT:
            used, score, cases = _best_stepper_shift(traces, ttarget)
        else:
            used = int(alignment)
            score, cases = period_stepper_score(traces, ttarget, shift=used)
        return score, tuple(cases), used
    if mode == 'cadence':
        if alignment is _REFIT_ALIGNMENT:
            used, score, cases = _best_cadence_latency(traces, ttarget)
        else:
            used = float(alignment)
            score, cases = cadence_score(traces, ttarget, latency=used)
        return score, tuple(cases), used
    if mode == 'events':
        if alignment is _REFIT_ALIGNMENT:
            shift, score = _best_event_shift(traces, ttarget)
        else:
            shift = float(alignment)
            score = event_score(traces, ttarget, shift=shift)
        cases = tuple(
            _event_case_score(traces, ttarget, ti, role, shift)
            for ti, trial in enumerate(ttarget.trials)
            for role in trial.expected)
        return score, cases, shift

    tol = getattr(traces, 'hold_tol', 1)       # 0 = strict LUT hold, 1 = nv ring
    shift = (_best_shift(traces, ttarget, tol)[0]
             if alignment is _REFIT_ALIGNMENT else int(alignment))
    cases = []
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            role_traces = traces.get(role, ())
            if ti >= len(role_traces):
                cases.append(0.0)
            elif METRIC == 'f1':
                cases.append(_pr_score(role_traces[ti], exp, shift, tol))
            else:
                cases.append(_trace_metric(role_traces[ti], exp))
    return windowed_score(traces, ttarget, shift=shift, tol=tol), tuple(cases), shift


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


# ── float-time coverage: the retention scorers (formerly nv_evo/robustness.py) ─────

# ── predeclared scoring constants (a state-1 hold rings with period ~D+W) ────────
# The nervous-net memory holds a bit as a pulse CIRCULATING (reads as ~50%-duty
# ringing), and a 1->0 flip leaves a short decay tail before the ring is vetoed.
# So we (a) drop a transition guard band, (b) exempt a leading decay/ramp window,
# (c) skip intervals too short to judge after those, and (d) score by sustained
# occupancy: a held bit rings (coverage present, no long silent gap); a cleared
# bit stays quiet (low coverage after the decay window).
GUARD_FRAC   = 1.0     # transition guard band  = GUARD_FRAC * DELAY (each side)
LEAD_FRAC    = 1.0     # leading decay/ramp exemption = LEAD_FRAC * (D+W)
MININT_FRAC  = 1.0     # need interior >= MININT_FRAC*(D+W) to judge, else skip
RING_COV     = 0.30    # state-1 interior coverage giving full credit (ring ~0.5)
QUIET_COV    = 0.30    # state-0 interior coverage tolerated before score 0
GAP_TOL_FRAC = 2.0     # a state-1 silent gap beyond (D+W)+this*(D+W) => bit lost
PASS         = 0.90    # a case passes iff worst-interval score >= PASS


def _cov_gap(pulses, lo, hi):
    """Covered length and largest uncovered gap of `pulses` within [lo, hi]."""
    segs = []
    for (s, e) in pulses:
        s2, e2 = (s if s > lo else lo), (e if e < hi else hi)
        if e2 > s2:
            segs.append((s2, e2))
    segs.sort()
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], e if e > merged[-1][1] else merged[-1][1])
        else:
            merged.append((s, e))
    covered = sum(e - s for s, e in merged)
    cur, max_gap = lo, 0.0
    for s, e in merged:
        if s - cur > max_gap:
            max_gap = s - cur
        cur = e
    if hi - cur > max_gap:
        max_gap = hi - cur
    return covered, max_gap


def parity_intervals(edges, horizon):
    """Commanded (state, start, end) intervals from effective input edges: state
    starts 0 and flips at each edge (toggle / T-flip-flop semantics)."""
    bounds = [0.0] + sorted(edges) + [float(horizon)]
    return [(i % 2, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def sr_intervals(set_edges, reset_edges, horizon):
    """SR-latch command intervals from effective float edges.

    Reset wins an exact-time tie, matching the nervous net's inhibitory veto.
    Repeated Set or Reset edges leave the commanded state unchanged.
    """
    grouped = {}
    for t in set_edges:
        grouped.setdefault(float(t), [False, False])[0] = True
    for t in reset_edges:
        grouped.setdefault(float(t), [False, False])[1] = True
    state, start, out = 0, 0.0, []
    for t in sorted(grouped):
        if t < 0.0 or t > horizon:
            continue
        has_set, has_reset = grouped[t]
        new_state = 0 if has_reset else (1 if has_set else state)
        if new_state != state:
            if t > start:
                out.append((state, start, t))
            state, start = new_state, t
    if horizon > start:
        out.append((state, start, float(horizon)))
    return out


def score_interval_graded(rise, state, start, end, offset):
    """Graded score for one independently selectable SR behavior case.

    Active intervals receive the fraction of consecutive windows retained
    before the first loss. Quiet intervals receive mean quietness, which gives
    Reset a gradient as ringing is progressively suppressed. Final claims use
    :func:`score_retention`, not this training-only gradient.
    """
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead = GUARD_FRAC * D, LEAD_FRAC * (D + W)
    lo = start + offset + guard + (0.0 if state == 1 else lead)
    hi = end + offset - guard
    win = 2.0 * (D + W)
    if hi - lo < D + W:
        return 0.0
    pulses = [(t, t + W) for t in sorted(rise)]
    scores, t = [], lo
    while t + 1e-9 < hi:
        wh = min(t + win, hi)
        if wh - t >= D + W:
            cov, _ = _cov_gap(pulses, t, wh)
            frac = cov / (wh - t)
            scores.append(min(1.0, frac / RING_COV) if state == 1
                          else max(0.0, 1.0 - frac / QUIET_COV))
        t += win
    if not scores:
        return 0.0
    if state == 0:
        return sum(scores) / len(scores)
    retained = 0.0
    for score in scores:
        if score < PASS:
            retained += score / PASS
            break
        retained += 1.0
    return retained / len(scores)


def score_reset_influence(rise, reset_time, offset):
    """Behavioral gradient for Reset: reward reduced activity after its arrival.

    A candidate gets no credit unless it was active before Reset. This remains
    a curriculum aid only; the full phase requires actual sustained quiet.
    """
    W, D = pulse.WIDTH, pulse.DELAY
    span = 4.0 * (D + W)
    arrival = reset_time + offset
    pulses = [(t, t + W) for t in sorted(rise)]
    pre, _ = _cov_gap(pulses, max(0.0, arrival - span), arrival - D)
    post_lo = arrival + LEAD_FRAC * (D + W)
    post, _ = _cov_gap(pulses, post_lo, post_lo + span)
    pre_frac = pre / max(D + W, span - D)
    post_frac = post / span
    if pre_frac <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (pre_frac - post_frac) / pre_frac))


def score_state_intervals(rise, intervals, offset):
    """Worst-interval semantic score, or None if no interval is long enough to
    judge. `intervals` are (state,a,b) in INPUT time; `offset` maps input time to
    output time (frozen). A state-1 interval must ring (sustained coverage, no
    silent gap beyond tolerance); a state-0 interval must stay quiet after its
    decay window."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    gap_lost = (D + W) + GAP_TOL_FRAC * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        # ringing starts immediately at the offset (no ramp), so state-1 gets only
        # the transition guard; a cleared bit's ring takes a moment to die, so
        # state-0 additionally exempts a leading decay window.
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:                # too short to judge after exemptions
            continue
        cov, gap = _cov_gap(pulses, lo, hi)
        frac = cov / (hi - lo)
        if state == 1:                      # held bit: ringing, no long silence
            s = min(1.0, frac / RING_COV)
            s = min(s, 1.0 - min(1.0, max(0.0, gap - (D + W)) / (gap_lost - (D + W))))
        else:                               # cleared bit: quiet
            s = max(0.0, 1.0 - frac / QUIET_COV)
        if s < worst:
            worst = s
        scored += 1
    return worst if scored else None


def _windows_worst(pulses, lo, hi, state):
    """Slide a window across [lo, hi]; return the WORST window's correctness. A
    held (state-1) interval must stay active in EVERY window (a sustained ring,
    not a burst that dies); a cleared (state-0) interval must stay quiet in every
    window. This is what turns 'held for a moment' into 'retained across the gap'."""
    W, D = pulse.WIDTH, pulse.DELAY
    win = 2.0 * (D + W)
    worst, t = 1.0, lo
    while t + 1e-9 < hi:
        wh = min(t + win, hi)
        if wh - t >= (D + W):                 # judge only near-full windows
            cov, _ = _cov_gap(pulses, t, wh)
            frac = cov / (wh - t)
            s = min(1.0, frac / RING_COV) if state == 1 else max(0.0, 1.0 - frac / QUIET_COV)
            if s < worst:
                worst = s
        t += win
    return worst


def score_retention(rise, intervals, offset):
    """Worst-window-worst-interval retention score (or None if unjudgeable).
    Unlike score_state_intervals (which checks a hold over one window), this
    requires the commanded state to be sustained across the WHOLE interval — so a
    finite ring burst that decays before the next command fails at the horizon
    where it dies."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:
            continue
        s = _windows_worst(pulses, lo, hi, state)
        if s < worst:
            worst = s
        scored += 1
    return worst if scored else None


def score_retention_graded(rise, intervals, offset):
    """Gradient version of score_retention for EVOLUTION: each interval is scored
    by the FRACTION of its windows correct BEFORE the first sustained failure
    (0..1), worst interval. Strict score_retention has no gradient (0 until a bit
    holds the WHOLE horizon); this rewards holding longer, so selection can climb
    from a short burst toward a sustained loop. Use the strict version for claims."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    win = 2.0 * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:
            continue
        total, correct, failed, t = 0, 0, False, lo
        while t + 1e-9 < hi:
            wh = min(t + win, hi)
            if wh - t >= (D + W):
                total += 1
                cov, _ = _cov_gap(pulses, t, wh)
                frac = cov / (wh - t)
                s = min(1.0, frac / RING_COV) if state == 1 else max(0.0, 1.0 - frac / QUIET_COV)
                if not failed and s >= PASS:
                    correct += 1
                else:
                    failed = True
            t += win
        frac_ret = (correct / total) if total else 0.0
        if frac_ret < worst:
            worst = frac_ret
        scored += 1
    return worst if scored else None


# ── unified score report (one body; nv / LUT / Designer wrap it) ─────────────
# Formerly three near-identical implementations (temporal_report, lut_report,
# designer._traces_report), each with its own copy of the mode branch. The
# per-backend voice lives in a small notes dict; the structure lives here once.

NV_REPORT_NOTES = {
    'waveform': 'Backend detail: scoring compares physical rise/fall intervals.',
    'period_stepper': 'Backend detail: nervous-net transition phase may be sub-second.',
    'cadence': 'Backend detail: cadence uses raw nervous-net edge timestamps.',
    'events': 'Backend detail: raw nervous-net timestamps retain sub-second edges.',
    'trace_preamble': (
        'Scored as persistent behaviour in active and quiet windows.',
        'A stored 1 may ring (1010...) because a nervous node is refractory;',
        'the scorer therefore tolerates one second of ring phase.',
        'One shared input-to-output latency is used across every test/output.',
        'The exact per-second value is diagnostic only and ignores ring tolerance.'),
}

LUT_REPORT_NOTES = {
    'period_stepper': 'Backend detail: LUT cadence transitions may occur at sub-second times.',
    'cadence': 'Backend detail: cadence uses raw LUT wire edge timestamps.',
    'events': 'Backend detail: raw LUT wire timestamps retain sub-second edges.',
    'trace_preamble': (
        'Scored as persistent behaviour in active and quiet windows.',
        'A one-second phase tolerance allows recurrent outputs to ring.',
        'One shared input-to-output latency is used across every test/output.',
        'The exact per-second value is diagnostic only and ignores tolerance.'),
}


def score_report_lines(ttarget, traces, out_pos, notes=None):
    """(total_score_or_None, body lines) for a prepared bundle, any mode.

    ``traces`` may be None for the expectation-only preview (waveform/events/
    trace modes); rhythm modes need real output and their wrappers early-return
    before calling this. ``notes`` carries the per-backend detail phrases; None
    (the Designer) skips them."""
    mode = getattr(ttarget, 'score_mode', 'trace')
    notes = notes or {}
    lines = []

    def out_lines():
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (
                term.role, None if out_pos is None else out_pos.get(term.role)))

    if mode == 'waveform':
        if notes.get('waveform'):
            lines += ['', notes['waveform']]
        shift, total = 0.0, None
        if traces is not None:
            shift, total, _ = _best_waveform_shift(traces, ttarget)
            out_lines()
            lines.append('fitted shared response latency: %s'
                         % _display_time(shift))
        for ti, trial in enumerate(ttarget.trials):
            lines.append('')
            for role in trial.expected:
                expected = [(start + shift, end + shift)
                            for start, end in _waveform_expected(
                                ttarget, trial, role)]
                lines.append('Test %d: expect %s intervals %s' %
                             (ti + 1, role, expected))
                if traces is not None:
                    actual = _role_intervals(traces, role, ti)
                    score = _interval_case_score(
                        actual, expected,
                        getattr(ttarget, 'waveform_tolerance', 0.25))
                    lines.append('        actual %s  waveform score %.3f %s' %
                                 (actual, score,
                                  'PASS' if score >= 0.999 else 'FAIL'))
        if traces is not None:
            lines += ['', '=> waveform score %.4f%s' %
                      (total, '   SOLVED' if total >= 0.999 else '')]
        return total, lines

    if mode == 'period_stepper':
        if notes.get('period_stepper'):
            lines += ['', notes['period_stepper']]
        total, cases = period_stepper_score(traces, ttarget)
        out_lines()
        for ti, (trial, score) in enumerate(zip(ttarget.trials, cases), 1):
            lines.append('Test %d: %s  cadence score %.3f %s' % (
                ti, trial_input_summary(trial, ttarget.n_inputs), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence-stepper score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return total, lines

    if mode == 'cadence':
        if notes.get('cadence'):
            lines += ['', notes['cadence']]
        total, cases = cadence_score(traces, ttarget)
        out_lines()
        for ti, score in enumerate(cases):
            events = _role_events(traces, ttarget.outputs[0].role, ti)
            lines.append('Test %d: %s  output edges@%s  cadence score %.3f %s' % (
                ti + 1, trial_input_summary(ttarget.trials[ti], ttarget.n_inputs),
                event_list_summary(events), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return total, lines

    if mode == 'events':
        if notes.get('events'):
            lines += ['', notes['events']]
        best_s = _best_event_shift(traces, ttarget)[0] if traces is not None else 0.0
        if traces is not None:
            out_lines()
            if getattr(ttarget, 'fit_latency', True):
                lines.append('measured output latency offset: %+.3f seconds'
                             % best_s)
            else:
                lines.append('fixed timing: required delay %g seconds; no offset fitted'
                             % getattr(ttarget, 'latency', 0))
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
        total = None
        if traces is not None:
            total = event_score(traces, ttarget, shift=best_s)
            lines += ['', '=> event score %.4f%s' % (
                total, '   SOLVED' if total >= 0.999 else '')]
        return total, lines

    # trace / coverage
    preamble = notes.get('trace_preamble')
    if preamble:
        lines += [''] + list(preamble)
    best_s = 0
    if traces is not None:
        best_s = _best_shift(traces, ttarget)[0]
        out_lines()
        lines.append('measured output latency offset: %+d second(s)' % best_s)
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
                tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp, best_s)
                s = _f1(tp_rec, n_exp, tp_prec, n_act)
                lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                             % (''.join(str(v) for v in tr_i), s,
                                'PASS' if s >= 0.999 else 'FAIL',
                                tp_rec, n_exp, tp_prec, n_act))
    total = None
    if traces is not None:
        total = windowed_score(traces, ttarget, shift=best_s)
        lines += ['', '=> behavioural score %.4f%s   (exact per-second %.4f)'
                  % (total, '   SOLVED' if total >= 0.999 else '',
                     exact_tick_accuracy(traces, ttarget))]
    return total, lines
