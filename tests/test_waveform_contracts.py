"""Shared waveform contracts, playback stimuli, and certification physics."""
import random
from types import SimpleNamespace
from unittest import mock
from substrates.nervous.pulse import PulseConfig
from substrates.nervous.ga import evaluate_nv_full
from substrates.nervous.targets import TEMPORAL_TARGETS
from substrates.nervous.temporal import score_contract, TemporalTraces, _waveform_expected
from runtime.checkpoint import _target_to_dict, _target_from_dict
TOL = 1e-9

def test_temporal_fitness_stays_equal_to_the_declared_contract_score():
    """Structural feedback is a rank tie-break, never unreported fitness."""
    target = SimpleNamespace(
        temporal=True, combinational_cases=(), trials=())
    traces = SimpleNamespace(overflow=False)
    genome = SimpleNamespace(
        chromosomes=[], state_delays=None, arch='single',
        routing_patches=[])
    with mock.patch(
            'substrates.nervous.ga.prepare_net',
            return_value=(
                {(0, 0): 1}, {}, [[(0, 0)]],
                {'sum': [(0, 0)], 'carry': [(0, 0)]}, traces)), \
            mock.patch(
            'substrates.nervous.ga.score_contract',
                return_value=(0.75, (0.75,), None)):
        score, cases = evaluate_nv_full(genome, target)
    assert score == 0.75
    assert cases == (0.75,)

def test_waveform_target_checkpoint_round_trip():
    target = TEMPORAL_TARGETS['Odd pulse selector']
    restored = _target_from_dict(_target_to_dict(target))
    assert [c.relation for c in restored.contract.constraints] == \
        ['pulse_intervals']
    assert restored.waveform_contract == 'odd_selector'
    assert restored.trials[0].expected_intervals == \
        target.trials[0].expected_intervals

def _perfect_waveform_traces(target):
    intervals = [_waveform_expected(target, trial, 'Q')
                 for trial in target.trials]
    return TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        intervals={'Q': intervals})

def test_odd_selector_passes_odd_indexed_pulses_with_their_widths():
    target = TEMPORAL_TARGETS['Odd pulse selector']
    target.pulse_config = PulseConfig(model='paper_analog')
    assert target.waveform_contract == 'odd_selector'
    # six plain banks (0..5 pulses) + three adversarial-gap banks (2, 3, 4)
    assert ([len(trial.input_events[0]) for trial in target.trials]
            == list(range(6)) + [2, 3, 4])
    for trial in target.trials:
        source = sorted(trial.input_events[0])
        expected = _waveform_expected(target, trial, 'Q')
        assert len(expected) == (len(source) + 1) // 2
        for interval, pulse in zip(expected, source[::2]):
            assert abs(interval[0] - (pulse[0] + 1.0)) <= TOL
            assert abs((interval[1] - interval[0]) - pulse[1]) <= TOL
    assert score_contract(_perfect_waveform_traces(target), target)[0] == 1.0

def test_odd_selector_rejects_fixed_dead_time_filters():
    """REGRESSION: the original odd-selector banks accepted a parity-free
    refractory filter - one fixed dead time (D ~= 4.1) reproduced every
    schedule and scored a perfect 1.0 without counting anything. The
    adversarial gap banks must hold every fixed dead time (start- OR
    end-referenced, swept finely) below the 0.90 certification bar, on the
    training seed and on fresh held-out seeds, while the true index counter
    still scores exactly 1.0."""
    from substrates.nervous.oracle import ORACLE_SPECS
    from substrates.nervous.temporal import TemporalTraces, waveform_score
    latency = 1.0

    def traces_for(target, select):
        rows = []
        for trial in target.trials:
            events = sorted(trial.input_events[0])
            rows.append([(start + latency, start + latency + width)
                         for start, width in select(events)])
        return TemporalTraces({'Q': [[] for _ in target.trials]},
                              intervals={'Q': rows})

    def refractory(events, dead_time, from_end):
        out, ref = [], None
        for start, width in events:
            if ref is None or start >= ref + dead_time:
                out.append((start, width))
                ref = start + width if from_end else start
        return out

    spec = ORACLE_SPECS['Odd pulse selector (oracle)']
    for seed_index, target in enumerate(
            (TEMPORAL_TARGETS['Odd pulse selector'], spec(seed=1), spec(seed=2))):
        assert waveform_score(
            traces_for(target, lambda ev: ev[::2]), target)[0] == 1.0
        step = 0.025 if seed_index == 0 else 0.1
        best = 0.0
        for from_end in (False, True):
            dead_time = 0.0
            while dead_time <= 12.0:
                traces = traces_for(
                    target, lambda ev, d=dead_time, e=from_end: refractory(ev, d, e))
                best = max(best, waveform_score(traces, target)[0])
                dead_time += step
        assert best < 0.90, 'refractory filter scored %.4f (seed %d)' % (
            best, seed_index)

def test_playback_preset_preserves_physical_widths():
    from substrates.nervous.playback import pulses_from_trial

    class Trial:
        pass

    class Target:
        pass

    explicit_trial = Trial()
    explicit_trial.input_events = [[(1.25, 0.75)], [(2.0, 1.5)]]
    explicit_trial.streams = []
    explicit = Target()
    explicit.trials = [explicit_trial]
    assert pulses_from_trial(explicit, 2) == [
        [(1.25, 0.75)], [(2.0, 1.5)]]

    stream_trial = Trial()
    stream_trial.input_events = None
    stream_trial.streams = [(1,), (1,), (1,), (0,)]
    streamed = Target()
    streamed.trials = [stream_trial]
    streamed.pulse_config = PulseConfig(
        model='paper_analog', width=0.75)
    assert pulses_from_trial(streamed, 1) == [[(0.0, 3.0)]]

def test_carry_physics_copies_run_config():
    """REGRESSION: certification must carry the run's physics onto fresh spec
    targets, else a non-default model is validated under uniform physics."""
    from substrates.nervous.certification import carry_physics

    class T:
        pass
    src = T()
    src.pulse_config = PulseConfig(
        model='paper_analog', delay=0.4, width=0.7, coincidence=0.2)
    dst = carry_physics(src, T())
    assert dst.pulse_config == src.pulse_config
    # a source without physics leaves the destination untouched (no crash)
    clean = carry_physics(T(), T())
    assert not hasattr(clean, 'pulse_config')

def test_mix_event_widths_never_reaches_the_next_same_lane_event():
    """Widened oracle pulses must fall strictly clear of the next event on the
    same lane: any overlap merges two labelled stimulus events into ONE physical
    edge (wired-OR / drive union), making the expected output unreachable in
    every model."""
    from substrates.nervous.oracle import _mix_event_widths, _WIDTH_CLEARANCE

    class Bag:
        pass

    trial = Bag()
    trial.streams = [(1,), (0,), (1,), (1,)]        # gaps 2 then 1
    target = Bag()
    target.n_inputs = 1
    target.name = 'tight bank'
    target.trials = [trial]
    _mix_event_widths(target, random.Random(0))
    events = trial.input_events[0]
    assert [start for start, _ in events] == [0.0, 2.0, 3.0]
    for (start, width), (nxt, _) in zip(events, events[1:]):
        assert start + width <= nxt - _WIDTH_CLEARANCE + TOL

    # today's real banks space same-lane events >= 3 apart, so the clamp is a
    # no-op there: every width still comes straight from the mixing palette
    palette = {0.5, 0.75, 1.0, 1.25, 1.75, 2.25}
    parity = TEMPORAL_TARGETS['A-count parity queried by B']
    for trial in parity.trials:
        for lane in trial.input_events:
            for _, width in lane:
                assert width in palette

def test_gui_categories_group_combinational_and_pulse_width_targets():
    """The picker folders: truth tables (raw and periodic-wrapped) live under
    'Combinational logic' instead of scattering into 'Timed events'; the
    duration-semantics targets get their own 'Pulse width & duration' folder;
    checkpointed targets keep their explicit category."""
    from ui.target_ui import target_category, CATEGORY_ORDER
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import TARGETS

    assert 'Combinational logic' in CATEGORY_ORDER
    assert 'Pulse width & duration' in CATEGORY_ORDER
    for name, raw in TARGETS.items():
        assert target_category(name, raw) == 'Combinational logic'
        wrapped = periodic_combinational_target(raw)
        assert target_category(name, wrapped) == 'Combinational logic'
    assert target_category(
        'Odd pulse selector', TEMPORAL_TARGETS['Odd pulse selector']) \
        == 'Pulse width & duration'
    # semantics-derived folders are unchanged for everything else
    assert target_category('SR latch', TEMPORAL_TARGETS['SR latch']) \
        == 'Memory & state'
    assert target_category('Echo (delay 3)', TEMPORAL_TARGETS['Echo (delay 3)']) \
        == 'Timed events'
    restored = _target_from_dict(_target_to_dict(
        TEMPORAL_TARGETS['Odd pulse selector']))
    assert restored.category == 'Pulse width & duration'
