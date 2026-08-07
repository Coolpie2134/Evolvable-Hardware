"""
tests/test_oracle_logic.py - the reference state machines that DEFINE the
input-driven temporal targets. These are the ground truth every evolved circuit
is scored against, so a change to one silently redefines the goal; pin them.

Fast and pure (no growth, no simulation, no multiprocessing).

Run under pytest, or standalone:  py tests/test_oracle_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous import (TEMPORAL_TARGETS,                      # noqa: E402
                    periodic_combinational_target)
from substrates.nervous.oracle import (make_c_element, orc_sr_latch,   # noqa: E402
                           orc_toggle, make_pulse_doubler,
                           orc_period_doubler, orc_period_tripler, ORACLE_SPECS,
                           make_refractory_filter,
                           make_a_first_rendezvous,
                           make_collision_serializer,
                           make_watchdog, make_a_parity_query,
                           make_a_mod3_query, make_a_batch_parity_query)
from substrates.nervous.temporal import TemporalTraces, event_score     # noqa: E402
from substrates.nervous.scoring import (contract_relations,             # noqa: E402
                            combinational_level_traces, score_contract)
from substrates.snn.targets import gate_target, get_target         # noqa: E402


def _trace(fn, seq):
    st, out = None, []
    for inb in seq:
        ob, st = fn(inb, st)
        out.append(ob[0])
    return out


def _is_event_target(target):
    return contract_relations(target) == ('event_correspondence',)


def _row_onsets(trial):
    """Ticks where a periodic combinational row RISES.

    Rows are presented as held levels, so a row occupies many active ticks and
    only the first of a contiguous run starts a case (see
    targets.periodic_combinational_target)."""
    return [tick for tick, row in enumerate(trial.streams)
            if any(row) and not (tick and any(trial.streams[tick - 1]))]


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
    assert _is_event_target(target)
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
    # A truth table is a LEVEL relation, not an edge relation - the same
    # contract FNV's settled-level readout has always been scored against.
    assert contract_relations(target) == ('combinational_level',)
    assert target.supported_backends == ('nervous', 'lut')
    assert len(target.trials) == 4  # two row orders at two phases
    assert 'tick' not in target.description.lower()

    # Cases are presented in widely-spaced, isolated windows so one test cannot
    # contaminate the next: the onset-to-onset spacing is many times the grid's
    # few-tick settling transient, not the old cramped 4-second packing.
    spacings = []
    for trial in target.trials:
        onsets = sorted(_row_onsets(trial))
        spacings += [b - a for a, b in zip(onsets, onsets[1:])]
    spacing = min(spacings)
    assert spacing >= 3 * target.grid_size
    cycle = 4 * spacing  # four AND rows, one window each

    # Every lane's complete truth-table waveform repeats exactly each cycle.
    for trial in target.trials:
        for lane in range(target.n_inputs):
            events = [second for second, row in enumerate(trial.streams)
                      if row[lane]]
            first_cycle = [event for event in events if event < events[0] + cycle]
            assert events[:len(first_cycle)] == first_cycle
            assert events[len(first_cycle):2 * len(first_cycle)] == [
                event + cycle for event in first_cycle]

    role = target.outputs[0].role
    exact = combinational_level_traces(
        target, lambda _role, bits: all(bits))
    assert score_contract(exact, target)[0] == 1.0

    # Holding the right rows is the whole answer; merely TWITCHING inside them
    # is a partial one, which is the difference between this contract and the
    # twin's. A one-tick assertion in the middle of the read window is worth its
    # duty, no more.
    def held(width):
        return TemporalTraces(
            {role: [[] for _ in target.trials]},
            events={role: [[low for low, _high in trial.expected_intervals[role]]
                           for trial in target.trials]},
            intervals={role: [[(low, low + width)
                               for low, _high in trial.expected_intervals[role]]
                              for trial in target.trials]})

    assert 0.5 < score_contract(held(1.0), target)[0] < 1.0
    assert score_contract(held(1.0), target)[0] < score_contract(
        held(3.0), target)[0] < 1.0

    # An edge at the old point-event latency has come and gone long before the
    # level is read, so it earns nothing on the rows that must assert.
    stale = TemporalTraces(
        {role: [[] for _ in target.trials]},
        events={role: [list(trial.expected_events[role])
                       for trial in target.trials]},
        intervals={role: [[(start, start + 1.0)
                           for start in trial.expected_events[role]]
                          for trial in target.trials]})
    assert score_contract(stale, target)[0] <= 0.5 + 1e-9

    # One autonomous oscillator cannot satisfy all rows/phases of an AND gate.
    oscillator = [[4.0 + 4.0 * index for index in range(5)]
                  for _ in target.trials]
    free_running = TemporalTraces(
        {role: [[] for _ in target.trials]}, events={role: oscillator})
    assert score_contract(free_running, target)[0] < 1.0


def test_multi_output_gate_cannot_coast_on_the_easy_output():
    """A half adder has an easy output (carry = AND) and a hard one (sum = XOR).
    A plain mean over outputs let a circuit solve only the carry and blanket the
    sum for (1.0 + 0.5)/2 = 0.75 - a comfortable plateau with no gradient toward
    the hard bit. Mean-AND-worst aggregation must (a) penalise that solve well
    below 0.75, (b) make the WEAK output dominate, and (c) still award 1.0 only
    when BOTH outputs are perfect."""
    from substrates.nervous.scoring import TemporalTraces as Traces, score_contract
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import get_target

    target = periodic_combinational_target(get_target('Half adder'))
    roles = [terminal.role for terminal in target.outputs]
    assert len(roles) == 2

    truth = {role: dict(zip(
        [bits for bits, _ in target.combinational_cases],
        [out[index] for _, out in target.combinational_cases]))
        for index, role in enumerate(roles)}

    def score(strong_only=None):
        """strong_only=None -> both outputs hold the truth; else that role holds
        the truth and the other blankets (asserts in every case window)."""
        traces = combinational_level_traces(
            target,
            lambda role, bits: (bool(truth[role].get(bits))
                                if strong_only is None or role == strong_only
                                else True))
        return score_contract(traces, target)[0]

    both_perfect = score(None)
    carry_only = score(strong_only=roles[0])
    sum_only = score(strong_only=roles[1])

    assert both_perfect >= 0.999                       # (c) all outputs perfect
    # (a) solving one output and blanketing the other is penalised past the old
    # 0.75 mean, toward the ~0.625 the mean-AND-worst blend gives.
    assert carry_only < 0.70 and sum_only < 0.70
    # (b) the score is dragged toward the weak (blanketed, ~0.5) output, not the
    # perfect one - i.e. the hard output carries the weight.
    assert carry_only < 0.5 + 0.5 * (1.0 - 0.5)         # < midpoint of 0.5..1.0


def test_lopsided_truth_tables_do_not_reward_indiscriminate_firing():
    """Firing on every case window must cap at 0.5, however lopsided the table.

    NAND puts a 1 on three of its four rows. Under pooled event F1 an output
    that simply echoed the case-valid strobe scored recall 1.0, precision 0.75
    -> 0.857, and evolution parked there instead of computing anything. The
    static truth-table path has always balanced expected-1 against expected-0
    rows for exactly this reason; the periodic encoding must too.
    """
    from substrates.nervous.scoring import score_contract

    def score(target, decide):
        return score_contract(
            combinational_level_traces(target, decide), target)[0]

    def truth_of(target):
        roles = [terminal.role for terminal in target.outputs]
        table = {
            role: {bits: out[index]
                   for bits, out in target.combinational_cases}
            for index, role in enumerate(roles)}
        return lambda role, bits: bool(table[role].get(bits))

    # EVERY combinational target, not a hand-picked few. This list used to name
    # NAND/NOR/XNOR/Majority-3 only, and OR - whose sole 0 is the dropped
    # all-zero row - scored a perfect 1.0 for blanket firing underneath it.
    from substrates.snn.targets import TARGETS

    checked = 0
    for name, base in TARGETS.items():
        if getattr(base, 'temporal', False) or not getattr(base, 'cases', ()):
            continue
        target = periodic_combinational_target(base)
        assert score(target, truth_of(target)) >= 0.999, name
        blanket = score(target, lambda role, bits: True)
        assert blanket <= 0.5 + 1e-9, (
            '%s rewards indiscriminate firing: %.4f' % (name, blanket))
        silent = score(target, lambda role, bits: False)
        assert silent <= 0.5 + 1e-9, (
            '%s rewards silence: %.4f' % (name, silent))
        checked += 1
    assert checked >= 15, 'expected the whole combinational suite, got %d' % checked


def test_combinational_rows_are_held_levels_and_temporal_twins_are_edges():
    """The one difference that makes the two encodings different problems.

    A combinational function is a function of a level PRESENT while the circuit
    computes: FNV holds its inputs for the whole settling horizon, SNN holds an
    input current for the whole run, and the native nervous/LUT static scorers
    hold their case levels. The periodic wrapper - the presentation nervous and
    LUT actually evolve against - must hold too, or it is posing a memory task
    (remember the row, then compute it) that duplicates its own `(temporal)`
    twin. The twin, conversely, must stay edge-timed.
    """
    from substrates.nervous.simulation import streams_to_schedule
    from substrates.nervous.targets import coincident_temporal_target
    from substrates.nervous.pulse import TICK

    for name in ('AND', 'XOR'):
        base = gate_target(name)
        held = periodic_combinational_target(base)

        # The hold outlasts a crossing of the grid, so a level is still applied
        # when the far side of the array settles ...
        assert held.combinational_hold >= 2 * held.grid_size
        for trial in held.trials:
            for tick in _row_onsets(trial):
                lanes = [lane for lane, bit in enumerate(trial.streams[tick])
                         if bit]
                assert lanes, 'a presented row must drive at least one lane'
                for lane in lanes:
                    assert all(trial.streams[tick + step][lane]
                               for step in range(held.combinational_hold))

        # ... and every row falls silent again before the next one rises, so a
        # shared lane cannot merge two rows into one uninterrupted level.
        for trial in held.trials:
            onsets = _row_onsets(trial)
            for tick, following in zip(onsets, onsets[1:]):
                assert following - tick >= held.combinational_hold + 2
                assert not any(trial.streams[tick + held.combinational_hold])

        # What the substrate physically receives is ONE long pulse per row, not
        # a burst of re-triggering edges.
        for trial in held.trials:
            schedule = streams_to_schedule(
                trial.streams, held.n_inputs, held.T)
            for lane in schedule:
                for _, width in lane:
                    assert abs(width - held.combinational_hold * TICK) < 1e-9

        # The edge-timed twin keeps the point-event presentation.
        twin = coincident_temporal_target(base)
        assert not getattr(twin, 'combinational_hold', 0)
        for trial in twin.trials:
            schedule = streams_to_schedule(trial.streams, twin.n_inputs, twin.T)
            for lane in schedule:
                for _, width in lane:
                    assert width <= TICK + 1e-9


def test_every_periodic_truth_table_receives_an_explicit_zero_row_window():
    """Every row is scored; only zero rows that must signal need a strobe."""
    ordinary = periodic_combinational_target(get_target('Full adder'))
    assert ordinary.combinational_data_inputs == 3
    assert ordinary.n_inputs == 3
    assert not ordinary.combinational_strobe
    assert all(len(trial.case_windows) == 16 for trial in ordinary.trials)

    decoder = periodic_combinational_target(get_target('2-to-4 decoder'))
    assert decoder.combinational_data_inputs == 2
    assert decoder.combinational_strobe
    assert decoder.n_inputs == 3
    assert 'case-valid lane' in decoder.description
    for trial in decoder.trials:
        # The strobe is HELD like every other lane, so count row onsets, not
        # the ticks the level occupies.
        marked_zero_rows = [
            tick for tick in _row_onsets(trial)
            if trial.streams[tick] == (0, 0, 1)]
        assert len(marked_zero_rows) == 2
        expected = set(trial.expected_events['D0'])
        assert all(
            float(tick + decoder.latency) in expected
            for tick in marked_zero_rows)

    comparator = periodic_combinational_target(get_target('2-bit comparator'))
    assert comparator.combinational_data_inputs == 4
    assert comparator.combinational_strobe
    assert comparator.n_inputs == 5


def test_all_zero_rows_are_presented_even_when_their_level_is_redundant():
    """OR's sole 0 sits on the all-zero row, which silence cannot present.

    Without a strobe every PRESENTED OR window expects 1, level balancing has
    one group to average, and a constant output scores a flawless 1.0 - OR
    "solved" that way in the contract benchmark. The strobe rule is therefore
    not 'the zero row must fire' but 'presenting the zero row changes what is
    measurable', and a low zero row can be just as load-bearing as a high one.
    """
    target = periodic_combinational_target(gate_target('OR'))
    assert target.combinational_strobe
    assert target.combinational_data_inputs == 2
    assert target.n_inputs == 3

    role = target.outputs[0].role
    for trial in target.trials:
        expected = set(trial.expected_events[role])
        onsets = _row_onsets(trial)
        assert len(onsets) == 8, 'all four rows must be presented, twice each'
        levels = {any(abs(e - (tick + target.latency)) < 1e-9 for e in expected)
                  for tick in onsets}
        assert levels == {False, True}, (
            'OR still has no negative case: levels=%s' % (levels,))

    # AND's zero level is already represented elsewhere, so no physical strobe
    # is needed; 00 still owns an explicit scheduled scoring window.
    and_target = periodic_combinational_target(gate_target('AND'))
    assert not and_target.combinational_strobe
    assert all(
        len(trial.case_windows) == 8
        for trial in and_target.trials)


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
    level - physically it carries no period."""
    t = TEMPORAL_TARGETS['Period doubler (2x)']
    assert _is_event_target(t)
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
    assert _is_event_target(target)
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
    assert _is_event_target(target)
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
    name = 'Temporal sum (deltaA + deltaB)'
    target = TEMPORAL_TARGETS[name]
    assert _is_event_target(target) and target.n_inputs == 2
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
        'Temporal sum (deltaA + deltaB)',
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
    assert _is_event_target(target)
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
    assert _is_event_target(t)
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


def test_stimuli_actually_present_the_clause_that_needs_memory():
    """Each of these targets was once satisfiable by a stateless delay line, not
    because the scorer was wrong but because its own stimulus never posed the
    part that needs state. These assertions are on the SCHEDULES, so they fail
    if a future retune quietly removes the hard cases again.
    """
    from substrates.nervous.targets import TEMPORAL_TARGETS

    # One-shot: a trigger arriving mid-pulse must be swallowed. With the old
    # min_gap = width + 4 no trigger ever landed inside an active interval.
    one_shot = TEMPORAL_TARGETS['One-shot (12 seconds)']
    width, inside = 12, 0
    for trial in one_shot.trials:
        ticks = [t for t, row in enumerate(trial.streams) if row[0]]
        inside += sum(1 for a, b in zip(ticks, ticks[1:]) if b - a <= width)
    assert inside >= 5, 'only %d retriggers land inside an active window' % inside

    # SR latch: hold duration must be set by Reset, not by a fixed decay. With
    # every hold in the band 10..14 one burst length fitted all of them.
    latch = TEMPORAL_TARGETS['SR latch']
    holds = []
    for trial in latch.trials:
        sets = [t for t, row in enumerate(trial.streams) if row[0]]
        resets = [t for t, row in enumerate(trial.streams) if row[1]]
        for at in sets:
            after = [r for r in resets if r > at]
            holds.append((after[0] if after else latch.T) - at)
    assert holds and max(holds) >= 3 * min(holds), (
        'hold durations span only %d..%d - a fixed-length burst fits them all'
        % (min(holds), max(holds)))

    # Collision serializer: the queue must actually back up, and every token
    # must still emerge inside the horizon.
    serializer = TEMPORAL_TARGETS['Collision serializer (2-to-1)']
    tokens = sum(sum(row) for trial in serializer.trials
                 for row in trial.streams)
    events = sum(1 for trial in serializer.trials
                 for value in trial.expected['Q'] if value == 1)
    assert tokens == events, (
        '%d tokens in but %d events out - the horizon is dropping tokens'
        % (tokens, events))
    ticks = sum(len({t for t, row in enumerate(trial.streams) if any(row)})
                for trial in serializer.trials)
    assert tokens >= 1.35 * ticks, (
        'only %.2f tokens per input tick - too few collisions to punish a '
        'wired-OR' % (tokens / ticks))


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
        assert _is_event_target(registered) and _is_event_target(fresh_a)
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
    assert 'One-shot (12 seconds)' in TEMPORAL_TARGETS
    assert 'Refractory filter (3 seconds)' in TEMPORAL_TARGETS
    assert 'Watchdog timeout (5 seconds)' in TEMPORAL_TARGETS


def test_one_shot_hold_outlasts_a_single_pulse():
    """A bare delay line must NOT be a perfect one-shot.

    This degeneracy has appeared twice. The first time the hold was 3 ticks and
    the old +/-1 ring tolerance let one pulse cover it; widening to 5 fixed that
    scorer but not the next one. Under the current state contract an active
    window is credited when its longest silence is within
    allowed_gap = 2*(delay + pulse width) = 4, and a lone pulse at the centre of
    a d-tick window leaves silences of (d - 1)/2 - so d must exceed 9 or an echo
    scores 1.000 again. Pin the property, not the number.
    """
    from substrates.nervous import pulse
    from substrates.nervous.scoring import TemporalTraces as Traces, score_contract
    from substrates.nervous.scoring import _expected_windows

    target = TEMPORAL_TARGETS['One-shot (12 seconds)']
    role = target.outputs[0].role
    allowed_gap = 2.0 * (pulse.DELAY + pulse.WIDTH)
    # Holds that run into the end of the horizon are legitimately truncated, so
    # judge the full-length ones: those are what must outlast a single pulse.
    holds = [len(ticks) for trial in target.trials
             for state, ticks in _expected_windows(trial.expected[role])
             if state == 1]
    assert holds, 'one-shot must command active windows'
    full = max(holds)
    assert full > 2 * allowed_gap + pulse.WIDTH, (
        'even a full hold is short enough for one pulse to cover: %d' % full)
    assert sum(h == full for h in holds) > len(holds) // 2, (
        'most holds should be full length, got %r' % (holds,))

    def score(make_events):
        data, events = {role: []}, {role: []}
        for trial in target.trials:
            fires = [tick for tick, row in enumerate(trial.streams) if row[0]]
            times = sorted({float(t) for t in make_events(fires, trial)
                            if 0 <= t < target.T})
            trace = [0.0] * target.T
            for t in times:
                trace[int(t)] = 1.0
            data[role].append(trace)
            events[role].append(times)
        return score_contract(Traces(data, events=events), target)[0]

    truth = score(lambda fires, trial: [
        tick for tick, value in enumerate(trial.expected[role]) if value == 1])
    assert truth >= 0.999, 'the oracle trace itself must score 1.0, got %r' % truth

    # A ring is how this substrate actually holds a bit; it must stay perfect.
    ring = score(lambda fires, trial: [
        tick for tick, value in enumerate(trial.expected[role])
        if value == 1 and tick % 2 == 0])
    assert ring >= 0.999, 'a circulating hold must still score 1.0, got %r' % ring

    for offset in (1, 2, len(target.trials) // 2, 8):
        echo = score(lambda fires, trial, d=offset: [f + d for f in fires])
        assert echo < 0.95, (
            'a bare echo at offset %d scores %.4f - one-shot is degenerate again'
            % (offset, echo))


def test_echo_delay_three_requires_absolute_timing():
    """Echo is a precision delay, so a fitted shift cannot rescue a wire."""
    target = TEMPORAL_TARGETS['Echo (delay 3)']
    assert _is_event_target(target)
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


# -- coincident-edge temporal twins of the combinational tables ----------------
# These are DERIVED from the truth tables, so a change to the builder silently
# redefines fifteen goals at once. Pin the encoding, not one hand-written case.

def _temporal_twins():
    from substrates.nervous.targets import TEMPORAL_TARGETS as registry
    return {name: target for name, target in registry.items()
            if name.endswith('(temporal)')}


def test_every_combinational_table_has_a_temporal_twin():
    from substrates.snn.targets import TARGETS
    twins = _temporal_twins()
    assert len(twins) == len(TARGETS)
    for base in TARGETS:
        assert '%s (temporal)' % base in twins


def test_temporal_twin_events_reproduce_the_truth_table():
    """Every presented row must expect exactly the row's asserted output bits."""
    from substrates.snn.targets import TARGETS
    for base, combinational in TARGETS.items():
        target = _temporal_twins()['%s (temporal)' % base]
        table = {tuple(i): tuple(o) for i, o in combinational.cases}
        data = combinational.n_inputs
        strobed = target.n_inputs > data
        roles = [terminal.role for terminal in target.outputs]
        for trial in target.trials:
            fired = {role: {int(t) for t in trial.expected_events[role]}
                     for role in roles}
            for tick, bits in enumerate(trial.streams):
                if not any(bits):
                    continue
                if strobed and not bits[data]:
                    continue
                expected = table[tuple(bits[:data])]
                for index, role in enumerate(roles):
                    assert ((tick + 1) in fired[role]) == bool(expected[index]), (
                        '%s: row %s role %s' % (base, bits[:data], role))


def test_temporal_twin_strobes_exactly_when_the_zero_row_carries_evidence():
    from substrates.snn.targets import TARGETS
    for base, combinational in TARGETS.items():
        target = _temporal_twins()['%s (temporal)' % base]
        extra = target.n_inputs - combinational.n_inputs
        assert extra in (0, 1)
        zero = next((o for i, o in combinational.cases if not any(i)),
                    (0,) * len(combinational.outputs))
        # A zero row that must FIRE is unrepresentable without the strobe:
        # silence cannot carry it on a quiescent substrate.
        if any(zero):
            assert extra == 1, base


def test_temporal_twin_keeps_blanket_firing_well_below_a_solve():
    """A circuit that fires on every presentation must not look like a solver."""
    for name, target in _temporal_twins().items():
        for trial in target.trials:
            presented = sum(1 for bits in trial.streams if any(bits))
            slots = presented * len(target.outputs)
            asserted = sum(len(v) for v in trial.expected_events.values())
            # One-to-one event F1 for a blanket responder is 2f/(1+f).
            fraction = asserted / float(slots)
            assert 2 * fraction / (1 + fraction) < 0.8, (
                '%s: blanket firing scores too well' % name)


def test_temporal_twin_row_orders_differ_so_a_fixed_rhythm_cannot_pass():
    """Rows land on the same tick grid, so the ORDER is what must vary."""
    for name, target in _temporal_twins().items():
        patterns = {tuple(tuple(bits) for bits in trial.streams if any(bits))
                    for trial in target.trials}
        assert len(patterns) > 1, name


def test_temporal_twins_are_deterministic_across_rebuilds():
    from substrates.nervous.targets import coincident_temporal_target
    from substrates.snn.targets import TARGETS
    for base, combinational in TARGETS.items():
        a = coincident_temporal_target(combinational)
        b = coincident_temporal_target(combinational)
        assert a.T == b.T and len(a.trials) == len(b.trials), base
        for left, right in zip(a.trials, b.trials):
            assert left.streams == right.streams, base
            assert left.expected_events == right.expected_events, base


def test_multi_output_spike_target_scores_each_role_separately():
    from substrates.nervous.targets import spike_target
    target = spike_target(
        'two-role probe',
        [({0: [3]}, {'X': [4], 'Y': []}),
         ({0: [9]}, {'X': [], 'Y': [10]})],
        T=14, n_inputs=1,
        outputs=[('X', (4, 1)), ('Y', (4, 3))])
    assert [t.role for t in target.outputs] == ['X', 'Y']
    # A role omitted from a case is silent-and-scored, not unscored: firing
    # there has to cost something.
    assert target.trials[0].expected_events['Y'] == []
    assert target.trials[0].expected['Y'][10] == 0
    assert target.trials[1].expected_events['Y'] == [10.0]


def test_multi_output_spike_target_rejects_an_unknown_role():
    from substrates.nervous.targets import spike_target
    try:
        spike_target('bad', [({0: [1]}, {'Nope': [2]})], T=6, n_inputs=1,
                     outputs=[('X', (2, 2))])
    except ValueError as exc:
        assert 'Nope' in str(exc)
    else:
        raise AssertionError('unknown output role must raise')


def test_temporal_twins_register_under_either_import_order():
    """The twins are derived from snn.targets, which imports this package back.

    Whichever module is imported first reaches the other half-built, so the
    registration runs from both ends. A regression here does not show up in the
    normal suite - it only bites a tool that imports snn.targets first, which is
    exactly how it escaped once already. Use subprocesses so each order starts
    from a clean interpreter.
    """
    import subprocess
    count = ('sum(1 for n in T if n.endswith("(temporal)"))')
    orders = {
        'snn first': (
            'import substrates.snn.targets;'
            'from substrates.nervous.targets import TEMPORAL_TARGETS as T;'
            'print(%s)' % count),
        'nervous first': (
            'from substrates.nervous.targets import TEMPORAL_TARGETS as T;'
            'import substrates.snn.targets;'
            'print(%s)' % count),
    }
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for label, program in orders.items():
        done = subprocess.run([sys.executable, '-c', program], cwd=root,
                              capture_output=True, text=True)
        assert done.returncode == 0, '%s: %s' % (label, done.stderr[-400:])
        assert int(done.stdout.strip()) == 15, (
            '%s registered %s twins' % (label, done.stdout.strip()))
