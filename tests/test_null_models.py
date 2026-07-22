"""
tests/test_null_models.py — the null-model gauntlet.

A behavioral contract is only worth its name if trivial NON-solutions fail it.
This project has been bitten twice by the opposite: One-shot(3) once scored a
perfect 1.0 for a single-pulse echo, and the contract-v1 rewrite briefly left
Gated oscillator satisfied equally by silence, by one pulse, and by a correct
oscillator — all three at 1.0.

The contract tests assert `score < 1.0` for a couple of hand-picked cheats.
That bar is too low to catch either failure's near neighbours: a cheat sitting
at 0.97 passes it. This module instead sweeps EVERY registered target against
EVERY cheat and demands a real margin, so a newly added target cannot quietly
be degenerate and a scoring edit cannot quietly make an old one degenerate.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo.contracts import event_contract, state_contract          # noqa: E402
from nv_evo.targets import TEMPORAL_TARGETS                          # noqa: E402
from nv_evo.scoring import (TemporalTraces, score_contract,          # noqa: E402
                            contract_relations, _state_case_score)

# No trivial output may reach this score on any target. The worst real cheat
# measured across the suite sits near 0.77 (a free-running ring on the period
# dividers, which genuinely produces some of the demanded edges), so 0.85
# leaves headroom without excusing a degenerate target.
CHEAT_CEILING = 0.85

KINDS = ('silence', 'always_on', 'single_pulse', 'free_ring', 'passthrough')


def _trace_len(trial, role, target):
    expected = trial.expected.get(role)
    return len(expected) if expected else target.T


def _cheat_bundle(target, kind, period=2):
    """One trivial non-solution, expressed in every observable a contract may
    read (samples, rise events, and pulse intervals) so the bundle is valid
    whichever relation the target declares."""
    samples, events, intervals = {}, {}, {}
    for term in target.outputs:
        role = term.role
        rows, rises, spans = [], [], []
        for trial in target.trials:
            n = _trace_len(trial, role, target)
            if kind == 'silence':
                row = [0] * n
            elif kind == 'always_on':
                row = [1] * n
            elif kind == 'single_pulse':
                row = [1 if t == 2 else 0 for t in range(n)]
            elif kind == 'free_ring':
                row = [1 if t % period == 0 else 0 for t in range(n)]
            elif kind == 'passthrough':
                stream = trial.streams[0] if trial.streams else ()
                row = [1 if (t < len(stream) and stream[t]) else 0
                       for t in range(n)]
            else:
                raise ValueError(kind)
            times = [float(t) for t, v in enumerate(row) if v]
            rows.append(row)
            rises.append(times)
            spans.append([(t, t + 1.0) for t in times])
        samples[role], events[role], intervals[role] = rows, rises, spans
    return TemporalTraces(samples, events=events, intervals=intervals)


def _worst_cheat(target):
    """(best score any cheat achieves, which cheat it was)."""
    best, best_kind = -1.0, None
    for kind in KINDS:
        score = score_contract(_cheat_bundle(target, kind), target)[0]
        if score > best:
            best, best_kind = score, kind
    return best, best_kind


def test_no_trivial_output_reaches_high_fitness_on_any_target():
    offenders = []
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        score, kind = _worst_cheat(target)
        if score >= CHEAT_CEILING:
            offenders.append('%s: %s scores %.3f' % (name, kind, score))
    assert not offenders, (
        'trivial non-solutions reach >= %.2f on %d target(s):\n  %s'
        % (CHEAT_CEILING, len(offenders), '\n  '.join(offenders)))


def test_silence_is_never_a_perfect_solution():
    # The sharpest single case: a dead circuit must never certify.
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        score = score_contract(_cheat_bundle(target, 'silence'), target)[0]
        assert score < 1.0, '%s: silence scores %.6f' % (name, score)


def test_gated_oscillator_discriminates_silence_from_oscillation():
    # Regression pin. This target's active epochs are single ticks of a
    # period-2 cadence, so a state contract drops every one of them and scores
    # the case on its quiet epochs alone — silence, one pulse and a correct
    # oscillator all reached 1.0. START fixes the phase, so the commanded train
    # is fully determined and event correspondence is the contract that fits.
    target = TEMPORAL_TARGETS['Gated oscillator']
    assert 'event_correspondence' in contract_relations(target)

    perfect = _cheat_bundle(target, 'silence')       # shape, then fill it in
    for term in target.outputs:
        role = term.role
        rows = [[1 if v == 1 else 0 for v in trial.expected[role]]
                for trial in target.trials]
        perfect[role] = rows
        perfect.events[role] = [[float(t) for t, v in enumerate(r) if v]
                                for r in rows]
        perfect.intervals[role] = [[(float(t), float(t) + 1.0)
                                    for t, v in enumerate(r) if v]
                                   for r in rows]
    assert score_contract(perfect, target)[0] == 1.0
    assert score_contract(_cheat_bundle(target, 'silence'), target)[0] == 0.0
    assert score_contract(_cheat_bundle(target, 'single_pulse'),
                          target)[0] < 0.5


def test_state_relation_refuses_a_case_it_cannot_judge():
    """A case commanding activity in epochs too short to judge scores 0, not 1.

    Without this the quiet epochs are scored alone and silence is perfect.
    """
    class _Trial:
        expected = {'Q': [0, 0, 1, 0, 0, 1, 0, 0]}   # 1-tick active epochs

    class _Target:
        temporal = True
        trials = (_Trial(),)
        outputs = ()
        pulse_config = None

    silent = TemporalTraces({'Q': [[0] * 8]},
                            events={'Q': [[]]}, intervals={'Q': [[]]})
    value = _state_case_score(silent, _Target(), _Trial(), 'Q', 0, 0)
    assert value == 0.0, 'unjudgeable active epochs scored %.6f' % value


def test_every_temporal_target_declares_a_known_relation():
    # A target whose relation set is empty would score vacuously.
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        assert contract_relations(target), name
        assert target.contract.constraints, name


def test_contract_defaults_do_not_silently_pick_the_state_relation():
    """oracle_target() falls back to state_contract() when a builder passes no
    contract. That default is the risky one — it is the relation that can drop
    obligations — so this pins the two builders apart rather than letting a new
    target inherit it by omission."""
    assert state_contract().constraints[0].relation == 'logical_state'
    assert event_contract().constraints[0].relation == 'event_correspondence'
