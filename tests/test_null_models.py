"""
tests/test_null_models.py - the null-model gauntlet.

A behavioral contract is only worth its name if trivial NON-solutions fail it.
This project has been bitten twice by the opposite: One-shot(3) once scored a
perfect 1.0 for a single-pulse echo, and the contract-v1 rewrite briefly left
Gated oscillator satisfied equally by silence, by one pulse, and by a correct
oscillator - all three at 1.0.

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

from substrates.nervous.contracts import event_contract, state_contract          # noqa: E402
from substrates.nervous.targets import (TEMPORAL_TARGETS,                        # noqa: E402
                            periodic_combinational_target)
from substrates.snn.targets import TARGETS                                  # noqa: E402
from substrates.nervous.scoring import (TemporalTraces, score_contract,          # noqa: E402
                            contract_relations, _state_case_score)

# No trivial output may reach this score on any target. The worst real cheat
# across the suite is a bare WIRE on the Refractory filter at 0.837 (it echoes
# every input edge, and the filter's job is only to suppress the close ones),
# with Veto gate next at 0.800. 0.85 clears both without excusing a degenerate
# target. This bound used to read "near 0.77 (a free-running ring)" because the
# wire cheat was inert - see _cheat_bundle's passthrough note.
CHEAT_CEILING = 0.85

# 'passthrough' is swept over every input lane and a few delays (see
# _wire_variants): a wire that happens to be plugged into the wrong lane is not
# evidence that the target resists wires.
KINDS = ('silence', 'always_on', 'single_pulse', 'free_ring', 'passthrough',
         'case_blanket', 'wire_burst')


def _trace_len(trial, role, target):
    expected = trial.expected.get(role)
    return len(expected) if expected else target.T


def _cheat_bundle(target, kind, period=2, lane=0, delay=0, taps=2, spacing=2):
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
                # A BARE WIRE: input lane `lane` echoed across time, optionally
                # delayed. This previously read trial.streams[0] - the tick-0
                # input ROW - and indexed it by t, so the "wire" went dead after
                # t == n_inputs and scored ~0 on nearly every target. The single
                # most important null model was therefore never actually tested;
                # a real wire scores up to 0.837, not the 0.0 it used to report.
                row = [0] * n
                for tick, stream in enumerate(trial.streams):
                    if lane < len(stream) and stream[lane] and 0 <= tick + delay < n:
                        row[tick + delay] = 1
            elif kind == 'wire_burst':
                # A wire and `taps - 1` delayed copies of itself: the same input
                # edge re-emitted `taps` times, `spacing` apart. On the nervous
                # substrate this is just two or three paths of different length
                # from one input - no feedback, no stored bit, about as cheap as
                # a circuit gets. It belongs here because a level-holding target
                # cannot tell a short burst from a thin ring: One-shot scored a
                # perfect 1.000 for taps=2, spacing=3, which centred itself in
                # the 12-tick window at shift -6 and split it into gaps of
                # 4/2/4 while sitting silent for 10 of the 12 ticks. The plain
                # wire above misses this entirely - it emits ONE edge per input.
                row = [0] * n
                for tick, stream in enumerate(trial.streams):
                    if lane < len(stream) and stream[lane]:
                        for tap in range(taps):
                            at = tick + delay + tap * spacing
                            if 0 <= at < n:
                                row[at] = 1
            elif kind == 'case_blanket':
                # One pulse per input onset: the natural cheat for the periodic
                # combinational encoding, where each truth-table row is an
                # isolated window. None of the dense per-tick cheats above can
                # express it (they either flood a window or miss it), which is
                # how OR - every presented window expecting 1 - passed this
                # gauntlet at 0.10 while a blanket circuit scored a clean 1.0.
                row = [0] * n
                for tick, stream in enumerate(trial.streams):
                    if any(stream) and 0 <= tick + delay < n:
                        row[tick + delay] = 1
            else:
                raise ValueError(kind)
            times = [float(t) for t, v in enumerate(row) if v]
            rows.append(row)
            rises.append(times)
            spans.append([(t, t + 1.0) for t in times])
        samples[role], events[role], intervals[role] = rows, rises, spans
    return TemporalTraces(samples, events=events, intervals=intervals)


def _wire_variants(target, kind):
    """(lane, delay) pairs to try for the input-driven cheats.

    A contract fits its own propagation offset, so a delayed wire is not a
    different cheat in principle - but the fit is bounded, and a cheat that
    lands one tick outside it would read as 'the target resists wires' when it
    does not. Sweeping a few offsets removes that false negative.
    """
    if kind == 'passthrough':
        lanes = range(max(1, int(getattr(target, 'n_inputs', 1))))
    elif kind == 'wire_burst':
        lanes = range(max(1, int(getattr(target, 'n_inputs', 1))))
    elif kind == 'case_blanket':
        lanes = (0,)
    else:
        return ({},)
    latency = int(getattr(target, 'latency', 0) or 0)
    delays = sorted({0, 1, 2, 3, latency})
    if kind != 'wire_burst':
        return tuple({'lane': lane, 'delay': delay}
                     for lane in lanes for delay in delays)
    # Tap count and spacing both matter: the burst wins by tiling the commanded
    # window with gaps no longer than the scorer's circulation budget, so the
    # combination that fits a given window length is the one to find.
    return tuple({'lane': lane, 'delay': delay, 'taps': taps,
                  'spacing': spacing}
                 for lane in lanes for delay in delays
                 for taps in (2, 3, 4) for spacing in (1, 2, 3, 4))


def _worst_cheat(target):
    """(best score any cheat achieves, which cheat it was)."""
    best, best_kind = -1.0, None
    for kind in KINDS:
        for variant in _wire_variants(target, kind):
            score = score_contract(
                _cheat_bundle(target, kind, **variant), target)[0]
            if score > best:
                best, best_kind = score, kind
    return best, best_kind


def _every_scored_target():
    """(name, target) for everything a backend is actually scored against.

    TEMPORAL_TARGETS alone is not that set. The nervous and LUT backends reach
    the combinational truth tables through periodic_combinational_target, and
    those wrapped targets live in no registry this module used to sweep - which
    is why OR could expect a 1 in every presented window, score a perfect 1.0
    for blanket firing, and still pass the gauntlet.
    """
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        yield name, target
    for name, base in sorted(TARGETS.items()):
        if getattr(base, 'temporal', False) or not getattr(base, 'cases', ()):
            continue
        yield '%s (combinational)' % name, periodic_combinational_target(base)


# Targets known to sit above CHEAT_CEILING, with the reason and a tight bound.
# An entry is an admission that the target measures its hard part weakly - NOT
# permission to drift, since exceeding the bound still fails.
KNOWN_WEAK = {
    # The wired-OR that used to sit at 0.860 here is fixed: the stimulus now
    # runs the queue to varying depth, and a wired-OR scores 0.670 (blanket
    # 0.760). What binds instead is `wire_burst` - a multi-tap delay line, a
    # cheat this module did not own until it was added to catch One-shot.
    #
    # It is not fixable by making the stimulus denser still. Under pooled event
    # F1 an under-emitting wire and an over-emitting k-tap line trade off
    # against each other, and the best either can be pushed to is ~0.83 at any
    # collision fraction; the remaining gap to the ceiling is timing, and past
    # this point tightening the stimulus is fitting the target to the cheat
    # list rather than measuring anything. Recorded as measured.
    'Collision serializer (2-to-1)': 0.87,
}

# A different admission from KNOWN_WEAK, and it must not be confused with one.
# These targets are not measured weakly - they are fully satisfiable WITHOUT
# state, so a high trivial score is the correct answer rather than a leak. A
# 3-tap delay line really does emit three pulses per edge; a 2-tap one really
# does halve a period. Changing the stimulus cannot alter that, because it
# follows from what the target asks for. They are listed so nobody cites a
# solve here as evidence that anything was evolved, or that memory was needed.
FEEDFORWARD_BY_CONSTRUCTION = {
    'Burst x3': 1.00,
    'Period halver (1/2x)': 0.90,
}


def _recorded_bound(name):
    if name in KNOWN_WEAK:
        return KNOWN_WEAK[name]
    return FEEDFORWARD_BY_CONSTRUCTION.get(name)


def test_no_trivial_output_reaches_high_fitness_on_any_target():
    offenders, regressed = [], []
    for name, target in _every_scored_target():
        score, kind = _worst_cheat(target)
        bound = _recorded_bound(name)
        if bound is not None:
            if score > bound + 1e-9:
                regressed.append(
                    '%s: %s scores %.3f, above its recorded %.2f'
                    % (name, kind, score, bound))
        elif score >= CHEAT_CEILING:
            offenders.append('%s: %s scores %.3f' % (name, kind, score))
    assert not offenders, (
        'trivial non-solutions reach >= %.2f on %d target(s):\n  %s'
        % (CHEAT_CEILING, len(offenders), '\n  '.join(offenders)))
    assert not regressed, (
        'known-weak target(s) got weaker:\n  %s' % '\n  '.join(regressed))


def test_known_weak_targets_are_still_weak_and_still_needed():
    """A stale exception is worse than none - it hides a target that got fixed."""
    scored = dict(_every_scored_target())
    for registry, label in ((KNOWN_WEAK, 'KNOWN_WEAK'),
                            (FEEDFORWARD_BY_CONSTRUCTION,
                             'FEEDFORWARD_BY_CONSTRUCTION')):
        for name in registry:
            assert name in scored, (
                '%s names a target that no longer exists: %s' % (label, name))
            score, _kind = _worst_cheat(scored[name])
            assert score >= CHEAT_CEILING, (
                '%s no longer needs its exception (worst cheat %.3f < %.2f) - '
                'delete the %s entry' % (name, score, CHEAT_CEILING, label))


def test_a_delay_line_is_neither_a_one_shot_nor_a_latch():
    """Regression pin for two targets that certified a stateless delay line.

    Both scored a PERFECT 1.000, and both were stimulus faults rather than
    scoring faults - the clause that needs memory was never presented.

    One-shot: triggers were spaced min_gap = width + 4 apart, so no trigger ever
    landed inside an active interval and make_one_shot's `rem == 0` guard - the
    only thing there that cannot be built from delays - was never exercised.

    SR latch: every hold interval fell in the band 10..14 ticks, so one fixed
    burst length fitted all of them, and a 4-tap line driven by Set alone, with
    Reset wired to nothing, matched every trial including the Set->Reset->Set
    guardrail. Its burst simply died where the Reset happened to be.
    """
    one_shot = TEMPORAL_TARGETS['One-shot (12 seconds)']
    score = score_contract(
        _cheat_bundle(one_shot, 'wire_burst', lane=0, delay=0, taps=2,
                      spacing=3), one_shot)[0]
    assert score < CHEAT_CEILING, 'one-shot delay line scores %.3f' % score

    latch = TEMPORAL_TARGETS['SR latch']
    score = score_contract(
        _cheat_bundle(latch, 'wire_burst', lane=0, delay=0, taps=4,
                      spacing=3), latch)[0]
    assert score < CHEAT_CEILING, 'reset-blind latch scores %.3f' % score


def test_silence_is_never_a_perfect_solution():
    # The sharpest single case: a dead circuit must never certify.
    for name, target in _every_scored_target():
        score = score_contract(_cheat_bundle(target, 'silence'), target)[0]
        assert score < 1.0, '%s: silence scores %.6f' % (name, score)


def test_gated_oscillator_discriminates_silence_from_oscillation():
    # Regression pin. This target's active epochs are single ticks of a
    # period-2 cadence, so a state contract drops every one of them and scores
    # the case on its quiet epochs alone - silence, one pulse and a correct
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
    contract. That default is the risky one - it is the relation that can drop
    obligations - so this pins the two builders apart rather than letting a new
    target inherit it by omission."""
    assert tuple(
        clause.relation for clause in state_contract().constraints
    ) == ('transition_correspondence', 'logical_state')
    assert event_contract().constraints[0].relation == 'event_correspondence'
