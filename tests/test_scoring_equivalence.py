"""Contract-v1 regression and anti-cheat tests.

The old golden file intentionally pinned seven mode-specific scorers. Contract
v1 is a semantic change, so the gate now pins the invariants that matter:
every target is declarative, all backends enter one evaluator, phase does not
change memory quality, and common shortcuts cannot reach perfect fitness.
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.contracts import (behavior_contract_badge,              # noqa: E402
                              behavior_contract_text, logic_contract,
                              state_contract, toggle_contract)
from substrates.nervous.targets import (OutputTerminal, TEMPORAL_TARGETS,       # noqa: E402
                                        TemporalTarget, Trial)
from substrates.nervous.scoring import (TemporalTraces, contract_relations,     # noqa: E402
                            contract_case_count, needs_samples,
                            combinational_level_traces,
                            score_contract, score_report_lines,
                            transition_score, _expected_events,
                            _expected_state_changes, _expected_windows,
                            _input_edges, _waveform_expected,
                            _min_waveform_shift)
from substrates.nervous.targets import periodic_combinational_target            # noqa: E402
from substrates.snn.targets import gate_target, get_target                 # noqa: E402


RELATIONS = {
    'event_correspondence', 'logical_state', 'pulse_intervals',
    'sustained_cadence', 'commanded_cadence', 'bounded_state',
    'transition_correspondence',
}


def test_every_registered_target_has_an_executable_contract():
    for name, target in TEMPORAL_TARGETS.items():
        assert target.contract.constraints, name
        assert set(contract_relations(target)) <= RELATIONS, name
        assert not hasattr(target, 'score_mode'), name
    assert logic_contract().constraints[0].relation == 'truth_table'


def test_contract_presentation_is_generated_from_executable_data():
    target = TEMPORAL_TARGETS['Oscillator']
    text = behavior_contract_text(target)
    assert 'Behavior Contract v1' in text
    assert 'score_contract (shared by every backend)' in text
    assert 'Sustained cadence' in text
    assert 'required period=' in text
    assert 'mean + worst restriction' in text
    assert 'Sustained cadence' in behavior_contract_badge(target)

    target.contract.constraints[0].parameters['presentation_probe'] = 17
    try:
        assert 'presentation probe=17' in behavior_contract_text(target)
    finally:
        del target.contract.constraints[0].parameters['presentation_probe']


def test_observation_collection_is_derived_from_contract():
    state = next(t for t in TEMPORAL_TARGETS.values()
                 if 'logical_state' in contract_relations(t))
    events = next(t for t in TEMPORAL_TARGETS.values()
                  if 'event_correspondence' in contract_relations(t))
    assert needs_samples(state)
    assert not needs_samples(events)


def test_logic_contract_rejects_constant_shortcuts():
    target = gate_target('XOR')
    perfect = [[bits[0] ^ bits[1]] for bits, _ in target.cases]
    assert score_contract(perfect, target)[0] == 1.0
    assert score_contract([[0]] * len(target.cases), target)[0] < 1.0
    assert score_contract([[1]] * len(target.cases), target)[0] < 1.0


def test_lopsided_logic_gate_gives_a_constant_no_edge():
    # The degeneracy the old flat-mean aggregation hid: AND is 1 in only one of
    # four rows, so always-0 used to score 0.75 - a do-nothing circuit looked
    # three-quarters solved. Balanced per-output aggregation pins any constant
    # to 0.5 (chance) regardless of how lopsided the truth table is, so the
    # gradient points at the function, not at the majority output value.
    for name in ('AND', 'NOR', 'OR', 'NAND'):
        target = gate_target(name)
        n = len(target.cases)
        assert score_contract([[0]] * n, target)[0] == 0.5, name
        assert score_contract([[1]] * n, target)[0] == 0.5, name
        perfect = [[out[0]] for _, out in target.cases]
        assert score_contract(perfect, target)[0] == 1.0, name


def test_logic_contract_rewards_partial_correctness_monotonically():
    # Fixing one more row must never lower the score - evolution needs the
    # gradient. AND: rows sorted so the single expected-1 row is corrected last.
    target = gate_target('AND')
    expected = [out[0] for _, out in target.cases]
    order = sorted(range(len(expected)), key=lambda i: expected[i])
    prev = -1.0
    obs = [[1 - e] for e in expected]                 # every row wrong
    for i in order:
        obs[i] = [expected[i]]                         # correct one more row
        score = score_contract(obs, target)[0]
        assert score >= prev - 1e-12, (i, score, prev)
        prev = score
    assert prev == 1.0


def test_multi_output_logic_ranks_balanced_progress_above_easy_output_only():
    """The same 12/16 correct cells can describe very different progress.

    Solving Carry while leaving Sum at chance must not tie a Full Adder that is
    75% correct on both required outputs.  This was the misleading 0.75 shelf
    produced by the old plain output mean.
    """
    target = get_target('Full adder')
    expected = [list(out_bits) for _, out_bits in target.cases]

    easy_only = [row[:] for row in expected]
    for row in easy_only:
        row[0] = 0

    balanced = [row[:] for row in expected]
    for output in range(2):
        zero_row = next(
            index for index, row in enumerate(expected)
            if row[output] == 0)
        one_row = next(
            index for index, row in enumerate(expected)
            if row[output] == 1)
        balanced[zero_row][output] = 1
        balanced[one_row][output] = 0

    easy_score, easy_cases, _ = score_contract(easy_only, target)
    balanced_score, balanced_cases, _ = score_contract(balanced, target)
    assert sum(easy_cases) == sum(balanced_cases) == 12
    assert easy_score == 0.625
    assert balanced_score == 0.75
    assert balanced_score > easy_score


def test_event_contract_requires_one_to_one_edges_and_silence():
    target = TEMPORAL_TARGETS['Coincidence (2-in)']
    perfect = TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        events={'Q': [list(tr.expected_events['Q'])
                      for tr in target.trials]})
    assert score_contract(perfect, target)[0] == 1.0

    silent = TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        events={'Q': [[] for _ in target.trials]})
    always = TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        events={'Q': [[float(t) for t in range(target.T)]
                      for _ in target.trials]})
    assert score_contract(silent, target)[0] < 1.0
    assert score_contract(always, target)[0] < 1.0


def test_temporal_logic_balances_outputs_in_scalar_and_alignment():
    """An event-rich Sum arm must not hide a completely dead Carry arm."""
    target = TEMPORAL_TARGETS['Full adder (temporal)']
    roles = [terminal.role for terminal in target.outputs]

    easy_only = TemporalTraces(
        {role: [[] for _trial in target.trials] for role in roles},
        events={
            roles[0]: [list(trial.expected_events[roles[0]])
                       for trial in target.trials],
            roles[1]: [[] for _trial in target.trials],
        })
    easy_score, easy_cases, _ = score_contract(easy_only, target)
    assert easy_score == 0.25

    balanced = TemporalTraces(
        {role: [[] for _trial in target.trials] for role in roles},
        events={
            role: [list(trial.expected_events[role][::2])
                   for trial in target.trials]
            for role in roles})
    balanced_score, balanced_cases, _ = score_contract(balanced, target)
    assert balanced_score > easy_score

    by_role = {role: [] for role in roles}
    case_index = 0
    for trial in target.trials:
        for role in trial.expected:
            by_role[role].append(balanced_cases[case_index])
            case_index += 1
    output_scores = [sum(by_role[role]) / len(by_role[role]) for role in roles]
    assert abs(balanced_score - 0.5 * (
        sum(output_scores) / len(output_scores) + min(output_scores))) < 1e-12


def test_periodic_truth_table_reports_the_exact_row_level_selection_score():
    target = periodic_combinational_target(get_target('Half adder'))
    assert not target.combinational_strobe
    assert target.n_inputs == target.combinational_data_inputs
    assert all(
        len(trial.case_windows)
        == len(target.combinational_cases) * 2
        for trial in target.trials)
    # A perfect circuit HOLDS each row's answer through the settled read window.
    roles = [terminal.role for terminal in target.outputs]
    table = {role: {bits: out[index] for bits, out in target.combinational_cases}
             for index, role in enumerate(roles)}
    traces = combinational_level_traces(
        target, lambda role, bits: bool(table[role].get(bits)))
    score, cases, _ = score_contract(traces, target)
    reported, lines = score_report_lines(target, traces, {
        terminal.role: [terminal.pos] for terminal in target.outputs})
    assert score == reported == 1.0
    assert len(cases) == contract_case_count(target)
    assert len(cases) == (
        len(target.trials)
        * len(target.combinational_cases)
        * 2  # repeats
        * len(target.outputs))
    text = '\n'.join(lines)
    assert 'Held combinational level' in text
    assert 'held truth-table levels' in text
    assert 'one-to-one timed events' not in text
    assert 'row 00' in text and 'row 11' in text
    assert 'held=100%' in text


def _ring_intervals(target, phase, period=4.0, width=1.0):
    result = []
    for trial in target.trials:
        intervals = []
        for state, ticks in _expected_windows(trial.expected['Q']):
            if state != 1 or len(ticks) <= period:
                continue
            start, end = float(min(ticks)), float(max(ticks) + 1)
            time = start + float(phase)
            while time < end:
                intervals.append((time, min(time + width, end)))
                time += period
        result.append(intervals)
    return result


def test_state_contract_is_phase_invariant_for_the_same_ring():
    # Retention remains phase-invariant by itself. The full Toggle contract now
    # also times each logical transition, so varying the response phase while
    # freezing alignment is intentionally observable there.
    target = dataclasses.replace(
        TEMPORAL_TARGETS['Toggle flip-flop'],
        contract=state_contract(timed_transitions=False))
    scores = []
    for phase in (0.0, 0.5, 1.0, 1.5, 2.0):
        traces = TemporalTraces(
            {'Q': [[] for _ in target.trials]},
            events={'Q': [[a for a, _ in seq]
                          for seq in _ring_intervals(target, phase)]},
            intervals={'Q': _ring_intervals(target, phase)})
        scores.append(score_contract(traces, target, alignment=0)[0])
    assert max(scores) - min(scores) < 1e-12


def test_state_contract_does_not_penalize_an_honest_ring_on_reset():
    """A pulse already in flight may land once after the state is cleared.

    Sweep both smaller and larger loops. Transition correspondence and
    retention must agree about the demonstrated lap; otherwise the former
    mistakes ordinary circulation for repeated logical activations.
    """
    names = (
        'One-shot (12 seconds)',
        'Resettable toggle',
        'SR latch',
        'Toggle flip-flop',
    )
    for name in names:
        target = TEMPORAL_TARGETS[name]
        roles = [terminal.role for terminal in target.outputs]
        for period in (2.0, 3.0, 4.0, 5.0, 6.0):
            per_role = {}
            for role in roles:
                trials = []
                for trial in target.trials:
                    intervals = []
                    for state, ticks in _expected_windows(
                            trial.expected.get(role, ())):
                        if state != 1 or not ticks:
                            continue
                        start = float(min(ticks))
                        end = float(max(ticks) + 1)
                        when = start
                        while when < end:
                            intervals.append((when, when + 1.0))
                            when += period
                        if when < float(target.T):
                            intervals.append((
                                when, min(when + 1.0, float(target.T))))
                    trials.append(intervals)
                per_role[role] = trials
            traces = TemporalTraces(
                {role: [[] for _ in target.trials] for role in roles},
                events={
                    role: [[start for start, _ in intervals]
                           for intervals in per_role[role]]
                    for role in roles},
                intervals=per_role)
            score, _, _ = score_contract(traces, target)
            assert score == 1.0, (name, period, score)


def test_state_contract_rejects_silence_and_permanent_activity():
    target = TEMPORAL_TARGETS['Toggle flip-flop']
    silent = TemporalTraces(
        {'Q': [[0] * (2 * target.T) for _ in target.trials]},
        events={'Q': [[] for _ in target.trials]},
        intervals={'Q': [[] for _ in target.trials]})
    high = TemporalTraces(
        {'Q': [[1] * (2 * target.T) for _ in target.trials]},
        intervals={'Q': [[(0.0, float(2 * target.T))]
                         for _ in target.trials]})
    assert score_contract(silent, target)[0] < 1.0
    assert score_contract(high, target)[0] < 1.0


def _strict_toggle_trace(changes, horizon):
    """Held-level trace whose Q flips at each continuous-time boundary."""
    samples = []
    level = 0
    change_set = set(changes)
    for tick in range(horizon):
        if tick in change_set:
            level ^= 1
        samples.append(level)
    intervals, start = [], None
    for tick, value in enumerate(samples + [0]):
        if value and start is None:
            start = float(tick)
        elif not value and start is not None:
            intervals.append((start, float(tick)))
            start = None
    traces = TemporalTraces(
        {'Q': [samples]},
        events={'Q': [[start for start, _ in intervals]]},
        intervals={'Q': [intervals]})
    traces.hold_tol = 0
    return traces


def test_toggle_requires_one_fitted_latency_but_gives_timing_partial_credit():
    # A flips at 1, 5, 9. With the nominal two-tick response, Q must flip at
    # 3, 7, 11. A uniformly slower circuit is equally valid; per-edge jitter is
    # not, because no single physical response latency explains it.
    T = 14
    input_edges = (1, 5, 9)
    streams = [
        (1 if tick in input_edges else 0,)
        for tick in range(T)]
    expected = [None, None, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    target = TemporalTarget(
        'Common-latency toggle',
        [(0, 0)],
        [OutputTerminal('Q', (1, 0))],
        T,
        [Trial(streams, {'Q': expected})],
        contract=toggle_contract(),
        latency=2)

    exact = _strict_toggle_trace((3, 7, 11), 2 * T)
    uniformly_slow = _strict_toggle_trace((4, 8, 12), 2 * T)
    jittered = _strict_toggle_trace((4, 6, 10), 2 * T)

    exact_score, _, exact_shift = score_contract(exact, target)
    slow_score, _, slow_shift = score_contract(uniformly_slow, target)
    jittered_score, _, _ = score_contract(jittered, target)
    jittered_transition, _, fitted_latency = transition_score(
        jittered, target)

    assert exact_score == 1.0
    assert exact_shift == 0.0
    assert slow_score == 1.0
    assert slow_shift == 1.0
    assert 0.0 < jittered_transition < 1.0
    assert 0.0 < jittered_score < 1.0
    assert fitted_latency == 2.0


def _state_target_trace(target, shift=0.0, jitter=False, strict=True):
    samples = {terminal.role: [] for terminal in target.outputs}
    events = {terminal.role: [] for terminal in target.outputs}
    intervals = {terminal.role: [] for terminal in target.outputs}
    jitter_pending = bool(jitter)
    horizon = 2 * target.T
    for trial in target.trials:
        for role in samples:
            changes = [
                time + shift for time in _expected_state_changes(
                    trial.expected.get(role, [0] * target.T))]
            if jitter_pending and changes:
                # Exceed the nervous ring's legal phase window as well as the
                # strict LUT timing tolerance.
                changes[0] += 6.0
                jitter_pending = False
            values, level = [], 0
            change_set = set(changes)
            for tick in range(horizon):
                if float(tick) in change_set:
                    level ^= 1
                values.append(level)
            spans, start = [], None
            for tick, value in enumerate(values + [0]):
                if value and start is None:
                    start = float(tick)
                elif not value and start is not None:
                    spans.append((start, float(tick)))
                    start = None
            samples[role].append(values)
            events[role].append([start for start, _ in spans])
            intervals[role].append(spans)
    traces = TemporalTraces(
        samples, events=events, intervals=intervals)
    traces.hold_tol = 0 if strict else 1
    return traces


def _event_target_trace(target, shift=0.0, jitter=False):
    roles = [terminal.role for terminal in target.outputs]
    events = {role: [] for role in roles}
    jitter_pending = bool(jitter)
    for trial in target.trials:
        for role in roles:
            sequence = [
                float(time) + shift
                for time in _expected_events(trial, role)]
            if jitter_pending and sequence:
                sequence[0] += 2.0
                jitter_pending = False
            events[role].append(sequence)
    return TemporalTraces(
        {role: [[] for _ in target.trials] for role in roles},
        events=events)


def test_every_registered_event_target_uses_one_alignment():
    targets = {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('event_correspondence',)
        and not getattr(target, 'combinational_cases', ())}
    assert targets
    for name, target in targets.items():
        exact = score_contract(_event_target_trace(target), target)[0]
        shifted = score_contract(
            _event_target_trace(target, shift=3.0), target)[0]
        jittered = score_contract(
            _event_target_trace(target, jitter=True), target)[0]
        assert exact == 1.0, (name, exact)
        if target.fit_latency:
            assert shifted == 1.0, (name, shifted)
            assert 0.0 < jittered < 1.0, (name, jittered)
        else:
            # Echo is deliberately an absolute-delay task.
            assert shifted < 1.0, (name, shifted)


def _event_target_trace_offsets(target, offsets, spurious=()):
    """Trace built from the expected edges, with a PER-ROLE time offset."""
    roles = [terminal.role for terminal in target.outputs]
    events = {role: [] for role in roles}
    for index, trial in enumerate(target.trials):
        for role in roles:
            sequence = [float(time) + offsets.get(role, 0.0)
                        for time in _expected_events(trial, role)]
            if index == 0:
                sequence += [float(time) for time in spurious]
            events[role].append(sorted(sequence))
    return TemporalTraces(
        {role: [[] for _ in target.trials] for role in roles},
        events=events)


def _fitted_event_targets():
    return {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('event_correspondence',)
        and target.fit_latency}


def test_event_targets_declare_the_latency_baked_into_their_edges():
    """Without this metadata a causal floor cannot even be expressed.

    Every one of these targets generates its expected edges at
    ``cause + latency``, so a target reporting latency 0 while expecting a
    response a second later is simply lying about itself - and the scorer,
    trusting it, would allow a full second of acausal alignment.
    """
    for name, target in _fitted_event_targets().items():
        first = min((time for trial in target.trials
                     for role in trial.expected
                     for time in _expected_events(trial, role)), default=None)
        if first is None:
            continue
        assert target.latency > 0, (name, target.latency)


def test_a_response_cannot_be_fitted_earlier_than_its_own_cause():
    """The fitted shift is a latency, not a licence to answer in advance.

    This is the flaw that let a Watchdog alarm score 1.0 while firing before
    the quiet interval it detects had finished elapsing: the trace was simply
    slid backwards until it lined up. A circuit may be FASTER than the nominal
    latency - down to answering simultaneously with its cause - and no faster.
    """
    checked = 0
    for name, target in _fitted_event_targets().items():
        roles = [terminal.role for terminal in target.outputs]
        tolerance = float(target.event_tolerance)
        # One tolerance width past the causal floor: unmistakably acausal.
        early = -(float(target.latency) + 2.0 * tolerance)
        offsets = {role: early for role in roles}
        assert score_contract(
            _event_target_trace_offsets(target, {}), target)[0] == 1.0, name
        acausal = score_contract(
            _event_target_trace_offsets(target, offsets), target)[0]
        assert acausal < 1.0, (name, acausal)
        checked += 1
    assert checked


def test_a_spurious_startup_edge_is_not_free():
    """Precision must count what the circuit emits before the scored window.

    The leading unscored region is startup grace for the trace view. It was
    also, accidentally, a place to dump edges for free - an extra event at t=0
    left every delayed event target scoring a perfect 1.0.
    """
    for name, target in _fitted_event_targets().items():
        cluttered = score_contract(
            _event_target_trace_offsets(target, {}, spurious=(0.0,)), target)[0]
        assert cluttered < 1.0, (name, cluttered)


def test_multi_output_targets_tolerate_one_tick_of_skew_between_outputs():
    """Different pads sit at different path lengths; that is routing, not logic.

    A full adder whose carry lands a tick after its sum has the logic right.
    Two ticks apart is a different claim, and still costs score.
    """
    targets = {name: target for name, target in _fitted_event_targets().items()
               if len(target.outputs) > 1}
    assert targets
    for name, target in targets.items():
        roles = [terminal.role for terminal in target.outputs]
        skewed = score_contract(
            _event_target_trace_offsets(target, {roles[1]: 1.0}), target)[0]
        wide = score_contract(
            _event_target_trace_offsets(target, {roles[1]: 2.0}), target)[0]
        assert skewed == 1.0, (name, skewed)
        assert wide < 1.0, (name, wide)


def _waveform_target_trace(target, shift=0.0, jitter=False):
    roles = [terminal.role for terminal in target.outputs]
    events = {role: [] for role in roles}
    intervals = {role: [] for role in roles}
    jitter_pending = bool(jitter)
    for trial in target.trials:
        for role in roles:
            spans = [
                (start + shift, end + shift)
                for start, end in _waveform_expected(
                    target, trial, role)]
            if jitter_pending and spans:
                start, end = spans[0]
                spans[0] = (start + 2.0, end + 2.0)
                jitter_pending = False
            intervals[role].append(spans)
            events[role].append([start for start, _ in spans])
    return TemporalTraces(
        {role: [[] for _ in target.trials] for role in roles},
        events=events, intervals=intervals)


def test_every_registered_waveform_target_uses_one_alignment():
    targets = {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('pulse_intervals',)}
    assert targets
    for name, target in targets.items():
        exact = score_contract(_waveform_target_trace(target), target)[0]
        shifted = score_contract(
            _waveform_target_trace(target, shift=3.0), target)[0]
        jittered = score_contract(
            _waveform_target_trace(target, jitter=True), target)[0]
        assert exact == 1.0, (name, exact)
        assert shifted == 1.0, (name, shifted)
        assert 0.0 < jittered < 1.0, (name, jittered)


def test_a_waveform_cannot_be_fitted_earlier_than_its_own_cause():
    """The same causal floor, for complete-interval scoring.

    An "Odd pulse selector" trace whose first interval began a second BEFORE
    the input pulse it selects used to score 1.0, because interval alignment
    searched arbitrarily far backwards. A response is allowed to be as fast as
    zero delay, and no faster.
    """
    targets = {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('pulse_intervals',)}
    assert targets
    for name, target in targets.items():
        floor = _min_waveform_shift(target)
        assert floor < 0.0, (name, floor)
        # At the floor the circuit answers simultaneously with its input:
        # unrealistically quick, but not acausal, so it still scores.
        assert score_contract(
            _waveform_target_trace(target, shift=floor), target)[0] == 1.0, name
        acausal = score_contract(
            _waveform_target_trace(target, shift=floor - 1.0), target)[0]
        assert acausal < 1.0, (name, acausal)


def _cadence_target_trace(target, latency=0.0, jitter=False):
    roles = [terminal.role for terminal in target.outputs]
    events = {role: [] for role in roles}
    jitter_pending = bool(jitter)
    for trial in target.trials:
        kick = _input_edges(trial.streams)[0]
        start = kick + latency + target.cadence_settle
        end = target.T + latency
        sequence, time = [], start
        while time < end - 1e-9:
            sequence.append(float(time))
            time += target.cadence_period
        if jitter_pending and len(sequence) > 2:
            sequence[1] += 1.0
            jitter_pending = False
        for role in roles:
            events[role].append(list(sequence))
    return TemporalTraces(
        {role: [[] for _ in target.trials] for role in roles},
        events=events)


def test_every_registered_sustained_cadence_uses_one_start_latency():
    targets = {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('sustained_cadence',)}
    assert targets
    for name, target in targets.items():
        exact = score_contract(_cadence_target_trace(target), target)[0]
        shifted = score_contract(
            _cadence_target_trace(target, latency=3.0), target)[0]
        jittered = score_contract(
            _cadence_target_trace(target, jitter=True), target)[0]
        assert exact == 1.0, (name, exact)
        assert shifted == 1.0, (name, shifted)
        assert 0.0 < jittered < 1.0, (name, jittered)


def test_every_registered_commanded_cadence_honors_each_command():
    targets = {
        name: target for name, target in TEMPORAL_TARGETS.items()
        if contract_relations(target) == ('commanded_cadence',)}
    assert targets
    for name, target in targets.items():
        role = target.outputs[0].role
        exact = TemporalTraces({
            role: [
                [1 if value == 1 else 0 for value in trial.expected[role]]
                + [0] * target.T
                for trial in target.trials]})
        ignored_commands = []
        for trial in target.trials:
            commands = _input_edges(trial.streams)
            start = int(commands[0] + 2)
            ignored_commands.append([
                1 if tick >= start and (tick - start) % 2 == 0 else 0
                for tick in range(2 * target.T)])
        assert score_contract(exact, target)[0] == 1.0, name
        assert score_contract(
            TemporalTraces({role: ignored_commands}), target)[0] < 1.0, name


def test_interval_contract_rejects_right_rises_with_wrong_widths():
    target = TEMPORAL_TARGETS['Pulse width sum (A+B)']
    perfect_intervals = {
        'Q': [list(tr.expected_intervals['Q']) for tr in target.trials]}
    perfect = TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        events={'Q': [[a for a, _ in seq]
                      for seq in perfect_intervals['Q']]},
        intervals=perfect_intervals)
    assert score_contract(perfect, target)[0] == 1.0
    wrong = TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        intervals={'Q': [[(a, a + 0.1) for a, _ in seq]
                         for seq in perfect_intervals['Q']]})
    assert score_contract(wrong, target)[0] < 1.0


def test_no_target_carries_a_hand_written_scoring_description():
    """The contract is the only statement of how a target is scored.

    Targets used to embed a 'Scoring:' paragraph chosen when the target was
    written. After the contract rewrite the GUI printed that stale prose
    directly above the contract that actually scored - two descriptions, one of
    them wrong. describe_target() now emits Goal/Tests only.
    """
    for name, target in TEMPORAL_TARGETS.items():
        description = getattr(target, 'description', '')
        assert 'Scoring:' not in description, name
        if description:
            assert description.startswith('Goal:'), name


def test_temporal_report_states_the_contract_exactly_once():
    from substrates.nervous.temporal import temporal_report
    for name, target in TEMPORAL_TARGETS.items():
        report = temporal_report(target)
        assert report.count('Behavior Contract v1') == 1, name
        assert 'score_contract (shared by every backend)' in report, name
