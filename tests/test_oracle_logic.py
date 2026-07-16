"""
tests/test_oracle_logic.py — the reference state machines that DEFINE the
input-driven temporal targets. These are the ground truth every evolved circuit
is scored against, so a change to one silently redefines the goal; pin them.

Fast and pure (no growth, no simulation, no multiprocessing).

Run under pytest, or standalone:  py tests/test_oracle_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import (TEMPORAL_TARGETS,                      # noqa: E402
                    periodic_combinational_target)
from nv_evo.oracle import (make_c_element, orc_sr_latch,   # noqa: E402
                           orc_toggle, make_pulse_doubler,
                           orc_period_doubler, orc_period_tripler, ORACLE_SPECS,
                           make_refractory_filter,
                           make_a_first_rendezvous,
                           make_collision_serializer,
                           make_watchdog, make_a_parity_query,
                           make_a_mod3_query, make_a_batch_parity_query)
from nv_evo.temporal import TemporalTraces, event_score     # noqa: E402
from snn_evo.targets import gate_target                     # noqa: E402


def _trace(fn, seq):
    st, out = None, []
    for inb in seq:
        ob, st = fn(inb, st)
        out.append(ob[0])
    return out


def test_a_parity_query_counts_a_and_b_only_reads_state():
    f = make_a_parity_query()
    seq = [
        (0, 1),                    # zero A: even, no Q
        (1, 0), (0, 1), (0, 1),  # one A: repeated B queries both emit
        (1, 0), (0, 1),          # two A: even, no Q
        (1, 0), (1, 0), (0, 1),  # four A: still even
        (1, 0), (0, 1),          # five A: odd, emit
    ]
    assert _trace(f, seq) == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_a_parity_query_target_is_registered_and_labels_every_b_query():
    display = 'A-count parity queried by B'
    spec = 'A parity query (oracle)'
    assert display in TEMPORAL_TARGETS
    assert spec in ORACLE_SPECS
    target = ORACLE_SPECS[spec](seed=515)
    assert target.score_mode == 'events'
    assert len(target.inputs) == 2
    for trial in target.trials:
        parity = 0
        expected = []
        for tick, (a_edge, b_edge) in enumerate(trial.streams):
            if a_edge:
                parity ^= 1
            if b_edge and parity:
                expected.append(float(tick + target.latency))
        assert trial.expected_events['Q'] == expected

    fresh = ORACLE_SPECS[spec](seed=616)
    assert [trial.streams for trial in target.trials] != \
        [trial.streams for trial in fresh.trials]


def test_related_a_count_query_state_machines():
    mod3 = make_a_mod3_query()
    # B at counts 0/1/2 is quiet; count 3 and a repeated query both emit;
    # count 4 is quiet; count 6 emits again.
    mod3_seq = [(0, 1), (1, 0), (0, 1), (1, 0), (0, 1),
                (1, 0), (0, 1), (0, 1), (1, 0), (0, 1),
                (1, 0), (1, 0), (0, 1)]
    assert _trace(mod3, mod3_seq) == \
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]

    batch = make_a_batch_parity_query()
    # Each B queries and clears: empty=0, one A=1, empty=0, two A=0,
    # three A=1, then the immediately repeated empty query is 0.
    batch_seq = [(0, 1), (1, 0), (0, 1), (0, 1),
                 (1, 0), (1, 0), (0, 1),
                 (1, 0), (1, 0), (1, 0), (0, 1), (0, 1)]
    assert _trace(batch, batch_seq) == \
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_a_count_query_targets_use_mixed_pulse_lengths():
    pairs = {
        'A-count parity queried by B': 'A parity query (oracle)',
        'A-count multiple-of-3 queried by B': 'A modulo-3 query (oracle)',
        'Odd A batch closed by B': 'A batch parity query (oracle)',
    }
    for display, spec in pairs.items():
        assert display in TEMPORAL_TARGETS
        assert spec in ORACLE_SPECS
        target = ORACLE_SPECS[spec](seed=717)
        widths = {
            width
            for trial in target.trials
            for lane in trial.input_events
            for _, width in lane
        }
        assert widths == {0.5, 0.75, 1.0, 1.25, 1.75, 2.25}
        # Physical pulse durations must not alter the reference edge labels.
        for trial in target.trials:
            assert trial.input_events is not None
            for lane_index, lane in enumerate(trial.input_events):
                assert [start for start, _ in lane] == [
                    float(tick) for tick, row in enumerate(trial.streams)
                    if row[lane_index]]


def test_binary_truth_tables_become_phase_locked_periodic_targets():
    target = periodic_combinational_target(gate_target('AND'))
    assert target.temporal
    assert target.score_mode == 'events'
    assert target.supported_backends == ('nervous', 'lut')
    assert len(target.trials) == 4  # two row orders at two phases
    assert 'tick' not in target.description.lower()

    # Every lane's complete truth-table waveform repeats exactly each cycle.
    cycle = 4 * 4  # four AND rows, four seconds per row
    for trial in target.trials:
        for lane in range(target.n_inputs):
            events = [second for second, row in enumerate(trial.streams)
                      if row[lane]]
            first_cycle = [event for event in events if event < events[0] + cycle]
            assert events[:len(first_cycle)] == first_cycle
            assert events[len(first_cycle):2 * len(first_cycle)] == [
                event + cycle for event in first_cycle]

    role = target.outputs[0].role
    expected = [trial.expected_events[role] for trial in target.trials]
    exact = TemporalTraces(
        {role: [[] for _ in target.trials]}, events={role: expected})
    assert event_score(exact, target) == 1.0

    # One autonomous oscillator cannot satisfy all rows/phases of an AND gate.
    oscillator = [[4.0 + 4.0 * index for index in range(5)]
                  for _ in target.trials]
    free_running = TemporalTraces(
        {role: [[] for _ in target.trials]}, events={role: oscillator})
    assert event_score(free_running, target) < 1.0


def test_c_element_is_a_rendezvous():
    """Emit only after BOTH inputs have produced an edge (either order), then
    rearm; a lone input edge must never emit."""
    f = make_c_element()
    # A@0 then B@2 -> emit@2; A&B@4 -> emit@4; lone A@6 -> no emit
    assert _trace(f, [(1, 0), (0, 0), (0, 1), (0, 0), (1, 1), (0, 0), (1, 0)]) \
        == [0, 0, 1, 0, 1, 0, 0]
    # B-before-A also completes the rendezvous (order-independent)
    assert _trace(f, [(0, 1), (1, 0)]) == [0, 1]
    # lone edges on either input never emit
    assert _trace(f, [(1, 0)] * 5) == [0] * 5
    assert _trace(f, [(0, 1)] * 5) == [0] * 5
    # simultaneous edges emit immediately
    assert _trace(f, [(1, 1)]) == [1]


def test_sr_latch_set_reset_hold():
    """Set on A, reset on B, hold otherwise."""
    assert _trace(orc_sr_latch, [(1, 0), (0, 0), (0, 0)]) == [1, 1, 1]   # set holds
    assert _trace(orc_sr_latch, [(1, 0), (0, 1), (0, 0)]) == [1, 0, 0]   # reset clears
    assert _trace(orc_sr_latch, [(0, 0), (0, 0)]) == [0, 0]              # never set


def test_toggle_parity():
    """Each input edge flips the stored bit."""
    assert _trace(orc_toggle, [(1,), (0,), (1,), (0,), (1,)]) == [1, 1, 0, 0, 1]


def test_pulse_doubler_widths():
    """A pulse held x ticks yields one contiguous 2x-tick output hold starting
    with it; quiet input stays quiet; a pulse merging into the tail banks debt
    (total output = 2 x total input ticks)."""
    f = make_pulse_doubler()
    for w in (1, 2, 4):
        seq = [(0,)] * 3 + [(1,)] * w + [(0,)] * (3 * w)
        out = _trace(f, seq)
        run = sum(out)
        first = out.index(1)
        assert run == 2 * w and first == 3, (w, out)
        assert all(out[first + k] for k in range(run)), "hold must be contiguous"
    assert sum(_trace(f, [(0,)] * 8)) == 0
    # overlap: 2 in-ticks then 1 more during the tail -> 6 contiguous out-ticks
    out = _trace(f, [(1,), (1,), (0,), (1,)] + [(0,)] * 8)
    assert sum(out) == 6 and out[:6] == [1] * 6, out


def test_period_doubler_halves_edge_rate():
    """Emit on the 1st, 3rd, 5th... input edge: a period-p input train yields a
    period-2p output train; no input edge, no output."""
    for p in (2, 3, 4):
        seq = [(1 if t % p == 0 else 0,) for t in range(4 * p + 1)]
        out = _trace(orc_period_doubler, seq)
        fired = [t for t, v in enumerate(out) if v]
        assert fired == [0, 2 * p, 4 * p], (p, fired)   # intervals exactly 2p
    assert sum(_trace(orc_period_doubler, [(0,)] * 8)) == 0


def test_period_doubler_bank_mixes_periods():
    """The registry bank must mix MULTIPLE input periods (a fixed free-running
    cadence fits only one) and include a silent guard trial (kills oscillators).
    Period 1 must NOT appear: a pulse every tick wired-OR merges into one held
    level — physically it carries no period."""
    t = TEMPORAL_TARGETS['Period doubler (2x)']
    assert t.score_mode == 'events'
    periods, silent = set(), 0
    for tr in t.trials:
        ticks = [i for i, s in enumerate(tr.streams) if s[0]]
        if not ticks:
            silent += 1
            continue
        gaps = {b - a for a, b in zip(ticks, ticks[1:])}
        assert len(gaps) == 1, "each trial is one periodic train"
        periods.add(gaps.pop())
    assert len(periods) >= 2 and 1 not in periods, periods
    assert silent >= 1
    assert 'Period doubler (oracle)' in ORACLE_SPECS
    assert 'Pulse doubler (oracle)' in ORACLE_SPECS   # width variant kept available


def test_period_tripler_has_three_times_the_input_period():
    for period in (2, 3, 4):
        seq = [(1 if tick % period == 0 else 0,)
               for tick in range(6 * period + 1)]
        fired = [tick for tick, value in enumerate(
            _trace(orc_period_tripler, seq)) if value]
        assert fired == [0, 3 * period, 6 * period], (period, fired)
    assert sum(_trace(orc_period_tripler, [(0,)] * 10)) == 0


def test_period_tripler_bank_mixes_periods_and_has_silent_guard():
    target = TEMPORAL_TARGETS['Period tripler (3x)']
    assert target.score_mode == 'events'
    periods, silent = set(), 0
    for trial in target.trials:
        ticks = [tick for tick, bits in enumerate(trial.streams) if bits[0]]
        if not ticks:
            silent += 1
            assert not trial.expected_events['Q']
            continue
        gaps = {right - left for left, right in zip(ticks, ticks[1:])}
        assert len(gaps) == 1
        period = gaps.pop()
        periods.add(period)
        output = trial.expected_events['Q']
        assert all(right - left == 3 * period
                   for left, right in zip(output, output[1:]))
    assert len(periods) >= 2 and silent >= 1
    assert 'Period tripler (oracle)' in ORACLE_SPECS


def test_period_halver_measures_then_emits_at_half_period():
    target = TEMPORAL_TARGETS['Period halver (1/2x)']
    assert target.score_mode == 'events'
    periods, silent = set(), 0
    for trial in target.trials:
        ticks = [tick for tick, bits in enumerate(trial.streams) if bits[0]]
        output = trial.expected_events['Q']
        if not ticks:
            silent += 1
            assert not output
            continue
        input_gaps = {right - left for left, right in zip(ticks, ticks[1:])}
        assert len(input_gaps) == 1
        period = input_gaps.pop()
        periods.add(period)
        assert period >= 4 and period % 2 == 0
        assert output[0] == ticks[1] + target.latency
        assert all(right - left == period / 2
                   for left, right in zip(output, output[1:]))
    assert len(periods) >= 2 and silent >= 1
    assert 'Period halver (oracle)' in ORACLE_SPECS


def test_temporal_sum_encodes_delta_a_plus_delta_b():
    name = 'Temporal sum (ΔA + ΔB)'
    target = TEMPORAL_TARGETS[name]
    assert target.score_mode == 'events' and target.n_inputs == 2
    sums, positive, guards = set(), 0, 0
    for trial in target.trials:
        a_ticks = [tick for tick, bits in enumerate(trial.streams) if bits[0]]
        b_ticks = [tick for tick, bits in enumerate(trial.streams) if bits[1]]
        output = trial.expected_events['Q']
        if len(a_ticks) == len(b_ticks) == 2:
            gap_a = a_ticks[1] - a_ticks[0]
            gap_b = b_ticks[1] - b_ticks[0]
            assert len(output) == 2
            assert output[1] - output[0] == gap_a + gap_b
            assert output[0] == max(a_ticks[1], b_ticks[1]) + target.latency
            sums.add(gap_a + gap_b)
            positive += 1
        else:
            assert not output
            guards += 1
    assert positive >= 8 and guards >= 4 and len(sums) >= 4
    assert 'Temporal sum (oracle)' in ORACLE_SPECS
    fresh = ORACLE_SPECS['Temporal sum (oracle)'](seed=404)
    assert [trial.streams for trial in target.trials] != [
        trial.streams for trial in fresh.trials]


def test_new_interval_targets_reject_direct_wires_and_fixed_oscillators():
    names = (
        'Period tripler (3x)',
        'Period halver (1/2x)',
        'Temporal sum (ΔA + ΔB)',
    )
    for name in names:
        target = TEMPORAL_TARGETS[name]
        direct = [[
            float(tick + target.latency)
            for tick, bits in enumerate(trial.streams)
            if bits[0] and tick + target.latency < target.T
        ] for trial in target.trials]
        oscillator = [[float(tick) for tick in range(2, target.T, 4)]
                      for _trial in target.trials]

        def score(events):
            traces = TemporalTraces(
                {'Q': [[] for _trial in target.trials]},
                events={'Q': events})
            return event_score(traces, target)

        assert score(direct) < 0.75, (name, score(direct))
        assert score(oscillator) < 0.70, (name, score(oscillator))


def test_pair_gap_two_widths_is_physical_and_relative():
    """The new target uses float pulses and labels only exact 2w edge gaps."""
    display = 'Pair detection gap (2x pulse width)'
    spec = 'Pair gap 2x width (oracle)'
    assert display in TEMPORAL_TARGETS and spec in ORACLE_SPECS
    target = ORACLE_SPECS[spec](seed=515, pulse_width=0.75)
    # both asynchronous backends may run this continuous-time target; clocked
    # backends (snn) stay excluded so the fractional phases are never quantised
    assert set(target.supported_backends) == {'nervous', 'lut'}
    assert target.score_mode == 'events'
    assert any(start != int(start)
               for trial in target.trials
               for start, _width in trial.input_events[0])

    positives = wrong_gap_negatives = lone_or_silent = 0
    for trial in target.trials:
        events = trial.input_events[0]
        starts = [start for start, width in events if abs(width - 0.75) < 1e-12]
        expected = trial.expected_events['Q']
        for event in expected:
            completion = event - target.latency
            assert any(abs(start - completion) < 1e-9 for start in starts)
            assert any(abs(start - (completion - 1.5)) < 1e-9 for start in starts)
        if expected:
            positives += 1
        elif len(starts) >= 2:
            wrong_gap_negatives += 1
        else:
            lone_or_silent += 1
    assert positives >= 6
    assert wrong_gap_negatives >= 3
    assert lone_or_silent >= 2
    fresh = ORACLE_SPECS[spec](seed=616, pulse_width=0.75)
    assert [trial.input_events for trial in target.trials] != [
        trial.input_events for trial in fresh.trials]


def test_c_element_registered_as_target():
    """The C-element is reachable from the GUI/registry and the holdout spec."""
    assert 'C-element (2-in join)' in TEMPORAL_TARGETS
    assert 'C-element (oracle)' in ORACLE_SPECS
    t = TEMPORAL_TARGETS['C-element (2-in join)']
    assert t.score_mode == 'events'
    assert len(t.inputs) == 2
    # every trial with a positive expectation must have at least one output event
    assert any(1 in [x for x in tr.expected['Q'] if x is not None] for tr in t.trials)


def test_c_element_bank_requires_both_inputs():
    """The target must reject A-only, B-only, wired-OR, and autonomous cheats."""
    t = ORACLE_SPECS['C-element (oracle)'](seed=404)
    kinds = []
    for trial in t.trials:
        has_a = any(row[0] for row in trial.streams)
        has_b = any(row[1] for row in trial.streams)
        kinds.append((has_a, has_b, len(trial.expected_events['Q'])))
    assert (True, False, 0) in kinds
    assert (False, True, 0) in kinds
    assert (False, False, 0) in kinds
    mixed = [kind for kind in kinds if kind[0] and kind[1]]
    assert len(mixed) == 10 and all(n_events == 3 for _, _, n_events in mixed)

    def input_cheat(mode):
        event_lists = []
        for trial in t.trials:
            a = {float(tick) for tick, row in enumerate(trial.streams) if row[0]}
            b = {float(tick) for tick, row in enumerate(trial.streams) if row[1]}
            event_lists.append(sorted(a if mode == 'A' else b if mode == 'B'
                                      else a | b))
        traces = TemporalTraces({'Q': [[] for _ in t.trials]},
                                events={'Q': event_lists})
        return event_score(traces, t)

    # A global latency shift is deliberately still free, but no direct input
    # echo or wired-OR response may approach the certification threshold.
    assert input_cheat('A') < 0.70
    assert input_cheat('B') < 0.70
    assert input_cheat('OR') < 0.70

    best_autonomous = 0.0
    for period in range(1, 11):
        for phase in range(period):
            events = [[float(tick) for tick in range(phase, t.T, period)]
                      for _ in t.trials]
            traces = TemporalTraces({'Q': [[] for _ in t.trials]},
                                    events={'Q': events})
            best_autonomous = max(best_autonomous, event_score(traces, t))
    assert best_autonomous < 0.55


def test_c_element_bank_changes_with_seed_without_late_truncation():
    """Held-out timings vary, and every completed round remains observable."""
    first = ORACLE_SPECS['C-element (oracle)'](seed=11)
    second = ORACLE_SPECS['C-element (oracle)'](seed=22)
    assert [trial.streams for trial in first.trials] != [trial.streams for trial in second.trials]
    for target in (first, second):
        mixed_counts = []
        for trial in target.trials:
            has_a = any(row[0] for row in trial.streams)
            has_b = any(row[1] for row in trial.streams)
            if has_a and has_b:
                mixed_counts.append(len(trial.expected_events['Q']))
            assert all(tick + target.latency < target.T
                       for tick, row in enumerate(trial.streams) if any(row))
        assert mixed_counts == [3] * 10


def test_refractory_filter_dead_time_boundary():
    """An accepted event blocks exactly the next three ticks, then re-arms."""
    f = make_refractory_filter(3)
    # t=0 fires; t=2 is blocked; t=4 is the earliest accepted event.
    assert _trace(f, [(1,), (0,), (1,), (0,), (1,), (0,)]) \
        == [1, 0, 0, 0, 1, 0]
    assert _trace(f, [(0,)] * 8) == [0] * 8


def test_a_first_rendezvous_order_ties_and_rearm():
    """Only A-first rounds emit; B-first and ties are consumed silently."""
    f = make_a_first_rendezvous()
    seq = [
        (1, 0), (1, 0), (0, 0), (0, 1),  # A first, repeat ignored -> emit
        (0, 1), (0, 0), (1, 0),          # B first -> silent
        (1, 1),                            # tie -> silent and immediately rearm
        (1, 0), (0, 1),                   # A first again -> emit
    ]
    assert _trace(f, seq) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def test_collision_serializer_preserves_tokens():
    """A collision becomes two spaced events; isolated inputs stay one-for-one."""
    f = make_collision_serializer(2)
    seq = [(1, 1), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)]
    out = _trace(f, seq)
    assert out == [1, 0, 1, 0, 1, 0, 0]
    assert sum(out) == sum(a + b for a, b in seq)


def test_watchdog_deadline_rearm_and_never_armed_silence():
    """Deadline heartbeats win; quiet alarms once; later heartbeat re-arms."""
    f = make_watchdog(5)
    seq = [
        (1,), (0,), (0,), (0,), (0,), (1,),  # heartbeat at deadline cancels
        (0,), (0,), (0,), (0,), (0,),        # then five quiet ticks -> alarm
        (0,), (1,),                           # stay disarmed, then re-arm
        (0,), (0,), (0,), (0,), (0,),        # another five quiet -> alarm
    ]
    assert _trace(f, seq) == [0] * 10 + [1, 0, 0] + [0] * 4 + [1]
    assert _trace(f, [(0,)] * 12) == [0] * 12


def test_new_async_oracles_are_registered_and_seeded():
    """All presets reach the GUI/certifier and fresh seeds vary their timings."""
    pairs = {
        'Refractory filter (3 seconds)': 'Refractory filter (oracle)',
        'A-first rendezvous': 'A-first rendezvous (oracle)',
        'Collision serializer (2-to-1)': 'Collision serializer (oracle)',
        'Watchdog timeout (5 seconds)': 'Watchdog timeout (oracle)',
    }
    for display_name, spec_name in pairs.items():
        assert display_name in TEMPORAL_TARGETS
        assert spec_name in ORACLE_SPECS
        registered = TEMPORAL_TARGETS[display_name]
        fresh_a = ORACLE_SPECS[spec_name](seed=101)
        fresh_b = ORACLE_SPECS[spec_name](seed=202)
        assert registered.score_mode == fresh_a.score_mode == 'events'
        assert registered.n_outputs == fresh_a.n_outputs == 1
        assert len(registered.inputs) == len(fresh_a.inputs)
        assert [tr.streams for tr in fresh_a.trials] != [tr.streams for tr in fresh_b.trials]

    serializer = ORACLE_SPECS['Collision serializer (oracle)'](seed=303)
    for trial in serializer.trials:
        n_input_events = sum(sum(bits) for bits in trial.streams)
        assert len(trial.expected_events['Q']) == n_input_events


def test_registered_target_copy_uses_seconds_not_ticks():
    """User-facing temporal names and descriptions use the shared time unit."""
    for name, target in TEMPORAL_TARGETS.items():
        assert 'tick' not in name.lower(), name
        assert 'tick' not in target.description.lower(), (name, target.description)
    assert 'One-shot (5 seconds)' in TEMPORAL_TARGETS
    assert 'Refractory filter (3 seconds)' in TEMPORAL_TARGETS
    assert 'Watchdog timeout (5 seconds)' in TEMPORAL_TARGETS


def test_echo_delay_three_requires_absolute_timing():
    """Echo is a precision delay, so a fitted shift cannot rescue a wire."""
    target = TEMPORAL_TARGETS['Echo (delay 3)']
    assert target.score_mode == 'events'
    assert target.latency == 3
    assert target.fit_latency is False
    expected = [trial.expected_events['Q'] for trial in target.trials]

    exact = TemporalTraces(
        {'Q': [[] for _ in target.trials]}, events={'Q': expected})
    assert event_score(exact, target) == 1.0

    direct_events = []
    for trial in target.trials:
        direct_events.append([
            float(second) for second, row in enumerate(trial.streams)
            if row[0] and (second == 0 or not trial.streams[second - 1][0])])
    direct = TemporalTraces(
        {'Q': [[] for _ in target.trials]}, events={'Q': direct_events})
    assert event_score(direct, target) < 0.25

    for delta in (-1.0, 1.0):
        mistimed = TemporalTraces(
            {'Q': [[] for _ in target.trials]},
            events={'Q': [[event + delta for event in events]
                           for events in expected]})
        assert event_score(mistimed, target) < 1.0


def test_memory_targets_leave_time_to_observe_each_state():
    """Commands cannot arrive before ringing/quiet output is distinguishable."""
    minimum_gaps = {
        'SR latch': 10,
        'Toggle flip-flop': 10,
        'Gated oscillator': 12,
        'Resettable toggle': 10,
    }
    for name, minimum in minimum_gaps.items():
        target = TEMPORAL_TARGETS[name]
        for trial in target.trials:
            event_times = [second for second, row in enumerate(trial.streams)
                           if any(row)]
            assert all(b - a >= minimum
                       for a, b in zip(event_times, event_times[1:])), \
                (name, event_times)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except Exception as e:                     # noqa: BLE001
            print("%s  %s: %s" % ("FAIL" if isinstance(e, AssertionError)
                                  else "ERROR", fn.__name__, e))
    print("\n%d/%d oracle-logic tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
