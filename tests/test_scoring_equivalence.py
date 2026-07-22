"""Contract-v1 regression and anti-cheat tests.

The old golden file intentionally pinned seven mode-specific scorers. Contract
v1 is a semantic change, so the gate now pins the invariants that matter:
every target is declarative, all backends enter one evaluator, phase does not
change memory quality, and common shortcuts cannot reach perfect fitness.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo.contracts import (behavior_contract_badge,              # noqa: E402
                              behavior_contract_text, logic_contract)
from nv_evo.targets import TEMPORAL_TARGETS                         # noqa: E402
from nv_evo.scoring import (TemporalTraces, contract_relations,     # noqa: E402
                            needs_samples, score_contract,
                            _expected_windows)
from snn_evo.targets import gate_target                             # noqa: E402


RELATIONS = {
    'event_correspondence', 'logical_state', 'pulse_intervals',
    'sustained_cadence', 'commanded_cadence', 'bounded_state',
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
    # four rows, so always-0 used to score 0.75 — a do-nothing circuit looked
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
    # Fixing one more row must never lower the score — evolution needs the
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
    target = TEMPORAL_TARGETS['Toggle flip-flop']
    scores = []
    for phase in (0.0, 0.5, 1.0, 1.5, 2.0):
        traces = TemporalTraces(
            {'Q': [[] for _ in target.trials]},
            events={'Q': [[a for a, _ in seq]
                          for seq in _ring_intervals(target, phase)]},
            intervals={'Q': _ring_intervals(target, phase)})
        scores.append(score_contract(traces, target, alignment=0)[0])
    assert max(scores) - min(scores) < 1e-12


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
    directly above the contract that actually scored — two descriptions, one of
    them wrong. describe_target() now emits Goal/Tests only.
    """
    for name, target in TEMPORAL_TARGETS.items():
        description = getattr(target, 'description', '')
        assert 'Scoring:' not in description, name
        if description:
            assert description.startswith('Goal:'), name


def test_temporal_report_states_the_contract_exactly_once():
    from nv_evo.temporal import temporal_report
    for name, target in TEMPORAL_TARGETS.items():
        report = temporal_report(target)
        assert report.count('Behavior Contract v1') == 1, name
        assert 'score_contract (shared by every backend)' in report, name
