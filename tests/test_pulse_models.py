"""
tests/test_pulse_models.py — the nervous-net node-timing models.

The paper's node regenerates ONE fixed pulse width after ONE fixed delay, so an
input pulse's LENGTH is discarded past the first node (see the discussion in
nv_evo/pulse.py). One variant restores width as a usable degree of freedom:

  * 'uniform'       — the paper (fixed width + delay). Must be BYTE-IDENTICAL
                      to the engine before these variants existed, whether or
                      not a config is passed. The "keep the architecture the
                      same" leg.
  * 'pulse_delay'   — historical API identifier for width-preserving transport:
                      both pulse edges move by the same evolved node delay, so
                      [t,t+w) becomes [t+d_node,t+d_node+w).

(A third model made the emitted width a per-node-type genome vector; width
evolution has been retired and its tests removed.)

Run under pytest, or standalone:  py tests/test_pulse_models.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo.pulse import PulseSim, PulseConfig, TICK          # noqa: E402
from nv_evo.hexgrid import hex_dirs                           # noqa: E402
from nv_evo import random_hex_genome                          # noqa: E402
from nv_evo.ga import (evaluate_nv_full, genome_signature,    # noqa: E402
                       mutate_nv, _mutate_state_delay)
from nv_evo.nervous import (node_delays, grow_nervous,  # noqa: E402
                            interpret_nervous)
from nv_evo.genome import (default_state_delays,  # noqa: E402
                           MAX_STATE)
from nv_evo.targets import TEMPORAL_TARGETS                   # noqa: E402
from nv_evo.temporal import (score_temporal_bundle, TemporalTraces,  # noqa: E402
                             _waveform_expected)
from evo_runtime.checkpoint import (genome_to_dict, genome_from_dict,  # noqa: E402
                                    _target_to_dict, _target_from_dict)

TOL = 1e-9


# ── a hand-built 2-cell buffer chain A -> B ──────────────────────────────────────

def _buffer_pair():
    a = (0, 0)
    b = hex_dirs(*a)['R']
    back = [d for d in ('L', 'R', 'D') if hex_dirs(*b)[d] == a][0]
    grid = {a: 1, b: 1}
    routing = {a: (None, None, None, 'and'), b: (back, back, None, 'and')}
    return grid, routing, a, b


def _buffer_chain():
    a = (0, 0)
    b = hex_dirs(*a)['R']
    c = next(pos for pos in hex_dirs(*b).values() if pos != a)
    back_b = [d for d in ('L', 'R', 'D') if hex_dirs(*b)[d] == a][0]
    back_c = [d for d in ('L', 'R', 'D') if hex_dirs(*c)[d] == b][0]
    grid = {a: 1, b: 1, c: 1}
    routing = {
        a: (None, None, None, 'and'),
        b: (back_b, back_b, None, 'and'),
        c: (back_c, back_c, None, 'and'),
    }
    return grid, routing, a, b, c


def _two_input_node(op='and', inhibit=False):
    v = (0, 0)
    neighbours = hex_dirs(*v)
    a, b, i = neighbours['L'], neighbours['R'], neighbours['D']
    grid = {a: 1, b: 1, v: 1}
    routing = {
        a: (None, None, None, 'and'),
        b: (None, None, None, 'and'),
        v: ('L', 'R', 'D' if inhibit else None, op),
    }
    if inhibit:
        grid[i] = 1
        routing[i] = (None, None, None, 'and')
    return grid, routing, a, b, i, v


def _inhibited_buffer():
    v = (0, 0)
    neighbours = hex_dirs(*v)
    source, inhibitor = neighbours['L'], neighbours['D']
    grid = {source: 1, inhibitor: 1, v: 1}
    routing = {
        source: (None, None, None, 'and'),
        inhibitor: (None, None, None, 'and'),
        v: ('L', 'L', 'D', 'and'),
    }
    return grid, routing, source, inhibitor, v


# ── V1: uniform is unchanged ─────────────────────────────────────────────────────

def test_uniform_is_the_legacy_default():
    """Default config is the legacy 'uniform' abstraction; a buffer fires once at delay=1,
    width=1 regardless of input pulse length (widths are regenerated)."""
    grid, routing, a, b = _buffer_pair()
    assert PulseConfig().model == 'uniform'
    for w in (1.0, 2.0, 3.0):
        sim = PulseSim(grid, routing, config=PulseConfig())
        sim.inject_pulse(a, 0.0, w)
        sim.advance_to(30.0)
        assert sim.rise_times[b] == [1.0], "uniform delay must be fixed at 1"
        assert abs((sim.pulse_until[b] - sim.pulse_start[b]) - 1.0) <= TOL









# ── V3: evolved-delay width preservation (legacy id: pulse_delay) ─────────────

def test_pulse_delay_neutral_delay_preserves_width():
    """A neutral delay vector uses PulseConfig.delay and preserves width."""
    grid, routing, a, b = _buffer_pair()
    for width in (0.5, 1.0, 2.75):
        sim = PulseSim(
            grid, routing,
            config=PulseConfig(model='pulse_delay', delay=0.75))
        sim.inject_pulse(a, 0.2, width)
        sim.advance_to(20.0)
        assert sim.rise_times[a] == [0.2]
        assert sim.rise_times[b] == [0.95]
        assert abs(sim.pulse_start[b] - 0.95) <= TOL
        assert abs(sim.pulse_until[b] - (0.95 + width)) <= TOL


def test_pulse_delay_uses_config_delay():
    """Without a delay vector, the third model uses the configured base delay."""
    grid, routing, a, b = _buffer_pair()
    for delay in (0.25, 1.0, 2.5):
        sim = PulseSim(
            grid, routing,
            config=PulseConfig(model='pulse_delay', delay=delay))
        sim.inject_pulse(a, 0.4, 1.7)
        sim.advance_to(20.0)
        assert abs(sim.rise_times[b][0] - (0.4 + delay)) <= TOL
        assert abs(sim.pulse_until[b] - (0.4 + delay + 1.7)) <= TOL


def test_pulse_delay_uses_evolved_node_delay_for_both_edges():
    grid, routing, a, b = _buffer_pair()
    for width in (0.5, 1.75, 3.0):
        sim = PulseSim(
            grid, routing, config=PulseConfig(model='pulse_delay', delay=1.0),
            delays={b: 1.5})
        sim.inject_pulse(a, 0.25, width)
        sim.advance_to(20.0)
        assert sim.rise_times[b] == [1.75]
        assert sim.pulse_intervals[b] == [[1.75, 1.75 + width]]


def test_node_delays_maps_states_and_is_model_gated():
    random.seed(31)
    g = random_hex_genome(2)
    t = TEMPORAL_TARGETS['Echo (delay 3)']
    grid = grow_nervous(g, seeds=tuple(t.inputs))
    cfg = PulseConfig(model='pulse_delay', delay=2.0)
    assert node_delays(g, grid, cfg) is None
    g.state_delays = default_state_delays()
    g.state_delays[1] = 1.5
    assert node_delays(g, grid, PulseConfig()) is None
    assert node_delays(g, grid, PulseConfig(model='uniform')) is None
    delays = node_delays(g, grid, cfg)
    for pos, state in grid.items():
        assert abs(delays[pos] - 2.0 * g.state_delays[state & 0x1F]) <= TOL


def test_delay_mutation_changes_signature_and_is_model_selective():
    random.seed(37)
    g = random_hex_genome(2)
    sig0 = genome_signature(g)
    _mutate_state_delay(g)
    assert g.state_delays is not None and len(g.state_delays) == MAX_STATE
    assert genome_signature(g) != sig0
    assert any(abs(value - 1.0) > TOL for value in g.state_delays)

    neutral = random_hex_genome(2)
    for _ in range(30):
        neutral = mutate_nv(neutral, mean_mutations=3.0)
    assert neutral.state_delays is None
    for _ in range(30):
        neutral = mutate_nv(neutral, mean_mutations=3.0, evolve_delay=True)
        if neutral.state_delays is not None:
            break
    assert neutral.state_delays is not None


def test_evolved_delay_checkpoint_round_trip():
    g = random_hex_genome(2)
    g.state_delays = default_state_delays()
    g.state_delays[7] = 1.5
    restored = genome_from_dict(genome_to_dict(g, 'nervous'), 'nervous')
    assert restored.state_delays == g.state_delays
    assert genome_signature(restored) == genome_signature(g)


def test_waveform_target_checkpoint_round_trip():
    target = TEMPORAL_TARGETS['Pulse width sum (A+B)']
    restored = _target_from_dict(_target_to_dict(target))
    assert restored.score_mode == 'waveform'
    assert restored.waveform_contract == 'width_sum'
    assert restored.trials[0].expected_intervals == \
        target.trials[0].expected_intervals


def _perfect_waveform_traces(target):
    intervals = [_waveform_expected(target, trial, 'Q')
                 for trial in target.trials]
    return TemporalTraces(
        {'Q': [[] for _ in target.trials]},
        intervals={'Q': intervals})


def test_width_sum_target_uses_both_input_durations():
    target = TEMPORAL_TARGETS['Pulse width sum (A+B)']
    target.pulse_config = PulseConfig(model='pulse_delay')
    assert target.waveform_contract == 'width_sum'
    assert len(target.inputs) == 2
    positive = 0
    for trial in target.trials:
        expected = _waveform_expected(target, trial, 'Q')
        if all(trial.input_events):
            positive += 1
            (start, end), = expected
            pulses = [lane[0] for lane in trial.input_events]
            assert abs((end - start) - sum(width for _, width in pulses)) <= TOL
            assert abs(start - (max(t for t, _ in pulses) + 1.0)) <= TOL
        else:
            assert expected == []
    assert positive == 6
    assert score_temporal_bundle(_perfect_waveform_traces(target), target)[0] == 1.0


def test_waveform_targets_fit_one_shared_latency_but_not_width_errors():
    target = TEMPORAL_TARGETS['Pulse width sum (A+B)']
    target.pulse_config = PulseConfig(model='pulse_delay')
    shifted = _perfect_waveform_traces(target)
    latency = 2.3
    shifted.intervals['Q'] = [
        [(start + latency, end + latency) for start, end in trial]
        for trial in shifted.intervals['Q']]

    score, _, fitted = score_temporal_bundle(shifted, target)
    assert abs(score - 1.0) <= TOL
    assert abs(fitted - latency) <= TOL
    frozen_wrong, _, used = score_temporal_bundle(
        shifted, target, alignment=0.0)
    assert frozen_wrong < 1.0
    assert used == 0.0

    start, end = shifted.intervals['Q'][0][0]
    shifted.intervals['Q'][0][0] = (start, end - 0.5)
    shifted._waveform_result = None
    # A fitted latency cannot conceal an incorrect output duration.
    assert score_temporal_bundle(shifted, target)[0] < 1.0

    wrong = _perfect_waveform_traces(target)
    wrong.intervals['Q'][0][0] = (
        wrong.intervals['Q'][0][0][0],
        wrong.intervals['Q'][0][0][1] - 0.5)
    assert score_temporal_bundle(wrong, target)[0] < 1.0


def test_odd_selector_passes_odd_indexed_pulses_with_their_widths():
    target = TEMPORAL_TARGETS['Odd pulse selector']
    target.pulse_config = PulseConfig(model='pulse_delay')
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
    assert score_temporal_bundle(_perfect_waveform_traces(target), target)[0] == 1.0


def test_odd_selector_rejects_fixed_dead_time_filters():
    """REGRESSION: the original odd-selector banks accepted a parity-free
    refractory filter — one fixed dead time (D ~= 4.1) reproduced every
    schedule and scored a perfect 1.0 without counting anything. The
    adversarial gap banks must hold every fixed dead time (start- OR
    end-referenced, swept finely) below the 0.90 certification bar, on the
    training seed and on fresh held-out seeds, while the true index counter
    still scores exactly 1.0."""
    from nv_evo.oracle import ORACLE_SPECS
    from nv_evo.temporal import TemporalTraces, waveform_score
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


def test_removed_node_contract_targets_are_not_registered():
    for name in ('Node: fixed-width regeneration',
                 'Node: evolved-width regeneration',
                 'Node: preserve width + evolve delay'):
        assert name not in TEMPORAL_TARGETS


def test_pulse_delay_base_width_reduces_to_uniform():
    """At neutral delay and base width, preserved transport equals uniform."""
    grid, routing, a, b = _buffer_pair()
    cfg = dict(delay=0.6, width=1.4)
    uniform = PulseSim(grid, routing, config=PulseConfig(**cfg))
    uniform.inject_pulse(a, 0.3, 1.4); uniform.advance_to(20.0)
    preserved = PulseSim(
        grid, routing, config=PulseConfig(model='pulse_delay', **cfg))
    preserved.inject_pulse(a, 0.3, 1.4); preserved.advance_to(20.0)
    assert uniform.rise_times[b] == preserved.rise_times[b]
    assert uniform.pulse_start[b] == preserved.pulse_start[b]
    assert uniform.pulse_until[b] == preserved.pulse_until[b]


def test_pulse_delay_late_extension_propagates_causally():
    """A re-drive after the output rise extends its delayed fall, not its rise."""
    grid, routing, a, b = _buffer_pair()
    sim = PulseSim(
        grid, routing, config=PulseConfig(model='pulse_delay', delay=1.0))
    sim.inject_pulse(a, 0.0, 1.0)
    sim.advance_to(0.25)
    sim.inject_pulse(a, 0.5, 2.5)                  # aggregate source is [0,3)
    sim.advance_to(10.0)
    assert sim.rise_times[a] == [0.0]
    assert sim.rise_times[b] == [1.0]
    assert abs(sim.pulse_start[b] - 1.0) <= TOL
    assert abs(sim.pulse_until[b] - 4.0) <= TOL


def test_pulse_delay_extension_survives_multiple_hops():
    grid, routing, a, b, c = _buffer_chain()
    sim = PulseSim(
        grid, routing, config=PulseConfig(model='pulse_delay', delay=1.0))
    sim.inject_pulse(a, 0.0, 1.0)
    sim.advance_to(0.25)
    sim.inject_pulse(a, 0.5, 2.5)
    sim.advance_to(10.0)
    assert sim.rise_times[b] == [1.0]
    assert sim.rise_times[c] == [2.0]
    assert (sim.pulse_start[b], sim.pulse_until[b]) == (1.0, 4.0)
    assert (sim.pulse_start[c], sim.pulse_until[c]) == (2.0, 5.0)


def test_pulse_delay_overlap_and_touching_are_one_waveform():
    grid, routing, a, b = _buffer_pair()
    cfg = PulseConfig(model='pulse_delay', delay=0.6)
    cases = (
        ((0.0, 2.0), (1.0, 3.0)),               # overlap -> [0,4)
        ((0.0, 1.0), (1.0, 2.0)),               # touching -> [0,3)
    )
    for events in cases:
        sim = PulseSim(grid, routing, config=cfg)
        for start, width in events:
            sim.inject_pulse(a, start, width)
        sim.advance_to(10.0)
        end = max(start + width for start, width in events)
        assert sim.rise_times[a] == [0.0]
        assert sim.rise_times[b] == [0.6]
        assert abs(sim.pulse_until[b] - (end + 0.6)) <= TOL


def test_pulse_delay_positive_gap_remains_two_pulses():
    grid, routing, a, b = _buffer_pair()
    sim = PulseSim(
        grid, routing, config=PulseConfig(model='pulse_delay', delay=1.0))
    sim.inject_pulse(a, 0.0, 1.0)
    sim.inject_pulse(a, 3.0, 1.0)
    sim.advance_to(10.0)
    assert sim.rise_times[a] == [0.0, 3.0]
    assert sim.rise_times[b] == [1.0, 4.0]


def test_pulse_delay_incremental_step_matches_preinjected_waveform():
    grid, routing, a, b = _buffer_pair()
    cfg = PulseConfig(model='pulse_delay', delay=1.0)
    incremental = PulseSim(grid, routing, config=cfg)
    samples = [incremental.step({a: bit})[b]
               for bit in (1, 1, 1, 0, 0, 0)]
    incremental.advance_to(6.0)

    explicit = PulseSim(grid, routing, config=cfg)
    explicit.inject_pulse(a, 0.0, 3.0)
    explicit_samples = []
    for tick in range(6):
        explicit.advance_to(tick + 0.5)
        explicit_samples.append(explicit.activity_at(tick + 0.5)[b])
    explicit.advance_to(6.0)

    assert samples == explicit_samples == [0, 1, 1, 1, 0, 0]
    assert incremental.rise_times == explicit.rise_times
    assert (incremental.pulse_start[b], incremental.pulse_until[b]) == (1.0, 4.0)


def test_pulse_delay_simultaneous_or_unions_widths_order_independently():
    grid, routing, a, b, _, out = _two_input_node(op='or')
    cfg = PulseConfig(model='pulse_delay', delay=1.0)

    def run(order):
        sim = PulseSim(grid, routing, config=cfg)
        for cell, width in order:
            sim.inject_pulse(cell, 2.0, width)
        sim.advance_to(10.0)
        return sim

    first = run(((a, 2.0), (b, 3.0)))
    second = run(((b, 3.0), (a, 2.0)))
    assert first.rise_times[out] == second.rise_times[out] == [3.0]
    assert first.pulse_until[out] == second.pulse_until[out] == 6.0


def test_pulse_delay_simultaneous_coincidence_preserves_overlap():
    grid, routing, a, b, _, out = _two_input_node(op='and')
    cfg = PulseConfig(model='pulse_delay', delay=1.0)

    def run(order):
        sim = PulseSim(grid, routing, config=cfg)
        for cell, width in order:
            sim.inject_pulse(cell, 2.0, width)
        sim.advance_to(10.0)
        return sim

    first = run(((a, 2.0), (b, 3.0)))
    second = run(((b, 3.0), (a, 2.0)))
    assert first.rise_times[out] == second.rise_times[out] == [3.0]
    assert first.pulse_until[out] == second.pulse_until[out] == 5.0


def test_pulse_delay_simultaneous_inhibition_uses_final_batch_level():
    grid, routing, source, inhibitor, out = _inhibited_buffer()
    cfg = PulseConfig(model='pulse_delay', delay=1.0)

    vetoed = PulseSim(grid, routing, config=cfg)
    vetoed.inject_pulse(source, 1.0, 1.0)
    vetoed.inject_pulse(inhibitor, 1.0, 1.0)
    vetoed.advance_to(5.0)
    assert vetoed.rise_times[out] == []

    released = PulseSim(grid, routing, config=cfg)
    released.inject_pulse(inhibitor, 0.0, 1.0)   # low at the half-open end t=1
    released.inject_pulse(source, 1.0, 1.0)
    released.advance_to(5.0)
    assert released.rise_times[out] == [2.0]

    touching = PulseSim(grid, routing, config=cfg)
    touching.inject_pulse(inhibitor, 0.0, 1.0)
    touching.inject_pulse(inhibitor, 1.0, 1.0)   # still continuously high
    touching.inject_pulse(source, 1.0, 1.0)
    touching.advance_to(5.0)
    assert touching.rise_times[out] == []


def test_pulse_delay_consistent_across_scoring_paths():
    """Sampled and event paths see the same transported waveform."""
    from nv_evo.temporal import _run_nervous
    grid, routing, a, b = _buffer_pair()
    cfg = PulseConfig(model='pulse_delay', delay=1.0)
    streams = [(1,), (1,), (1,)] + [(0,)] * 10
    states, _, rise_sampled, _ = _run_nervous(
        grid, routing, [a], {'Q': b}, streams, 16, prune=True, config=cfg,
        sample=True)
    _, _, rise_event, _ = _run_nervous(
        grid, routing, [a], {'Q': b}, streams, 16, prune=True, config=cfg,
        sample=False)
    assert rise_sampled.get(b) == rise_event.get(b) == [1.0]
    assert [states[t].get(b, 0) for t in range(5)] == [0, 1, 1, 1, 0]


def test_playback_preset_preserves_physical_widths():
    from nv_evo.playback import pulses_from_trial

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
        model='pulse_delay', width=0.75)
    assert pulses_from_trial(streamed, 1) == [[(0.0, 3.0)]]


def test_run_nervous_uniform_unaffected_by_preinjection():
    """The pre-injection rewrite must not change uniform: a held input still
    fires the buffer at the fixed delay (1), width regenerated."""
    from nv_evo.temporal import _run_nervous
    grid, routing, a, b = _buffer_pair()
    streams = [(1,), (1,), (1,)] + [(0,)] * 10
    _, traces, rise, _ = _run_nervous(
        grid, routing, [a], {'Q': b}, streams, 16, prune=True,
        config=PulseConfig(), sample=True)
    assert rise.get(b) == [1.0], "uniform delay must stay fixed at 1"


def test_carry_physics_copies_run_config():
    """REGRESSION: certification must carry the run's physics onto fresh spec
    targets, else a non-default model is validated under uniform physics."""
    from nv_evo.certification import carry_physics

    class T:
        pass
    src = T()
    src.pulse_config = PulseConfig(
        model='pulse_delay', delay=0.4, width=0.7, coincidence=0.2)
    dst = carry_physics(src, T())
    assert dst.pulse_config == src.pulse_config
    # a source without physics leaves the destination untouched (no crash)
    clean = carry_physics(T(), T())
    assert not hasattr(clean, 'pulse_config')


def test_config_validates_model_and_physical_parameters():
    for bad in ('nope', 'Uniform', ''):
        try:
            PulseConfig(model=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('bad model %r accepted' % bad)
    for values in ({'delay': float('nan')}, {'width': float('inf')},
                   {'coincidence': -0.1}):
        try:
            PulseConfig(**values)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid physical config accepted: %r' % values)


def test_run_config_migrates_retired_delay_gain():
    from evo_runtime.config import RunConfig

    migrated = RunConfig.from_dict({
        'pulse': {
            'model': 'pulse_delay', 'delay': 0.8, 'width': 1.2,
            'coincidence': 0.3, 'delay_gain': 4.0,
        }})
    assert migrated.pulse == PulseConfig(
        model='pulse_delay', delay=0.8, width=1.2, coincidence=0.3)
    assert migrated.ga.node_model == 'pulse_delay'
    assert not hasattr(migrated.pulse, 'delay_gain')


def test_pulse_delay_rejects_invalid_injected_intervals():
    grid, routing, a, _ = _buffer_pair()
    sim = PulseSim(grid, routing, config=PulseConfig(model='pulse_delay'))
    for start, width in ((0.0, 0.0), (0.0, -1.0),
                         (float('nan'), 1.0), (0.0, float('inf'))):
        try:
            sim.inject_pulse(a, start, width)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid pulse accepted: %r' % ((start, width),))


# ── cross-model comparability guards ─────────────────────────────────────────────


def test_ga_config_timing_toggles_default_to_model_pairing():
    from evo_runtime.config import GAConfig
    assert GAConfig().timing_mutations() is False
    assert GAConfig(node_model='pulse_delay').timing_mutations() is True
    # the fixed-delay ablation: width-preserving transport, no delay mutation
    fixed = GAConfig(node_model='pulse_delay', evolve_delay=False)
    assert fixed.timing_mutations() is False
    # a mutation whose vector the engine ignores must be rejected, not evolved
    for bad in (dict(node_model='uniform', evolve_delay=True),
                dict(node_model='paper_analog', evolve_delay=True)):
        try:
            GAConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid timing toggle accepted: %r' % bad)


def test_retired_width_evolution_is_gone_but_old_checkpoints_still_load():
    """Width evolution was removed from the substrate. Its node model must no
    longer be constructible, and a checkpoint saved under it must migrate onto
    the paper's fixed-width node rather than fail to open."""
    from evo_runtime.config import GAConfig, RunConfig
    for bad in ({'node_model': 'evolved_width'},):
        try:
            GAConfig(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError('retired width model accepted: %r' % bad)
    try:
        PulseConfig(model='evolved_width')
    except ValueError:
        pass
    else:
        raise AssertionError('retired width model accepted by PulseConfig')

    migrated = RunConfig.from_dict({
        'ga': {'node_model': 'evolved_width', 'evolve_width': True},
        'pulse': {'model': 'evolved_width', 'width': 1.5}})
    assert migrated.ga.node_model == 'uniform'
    assert migrated.pulse.model == 'uniform'
    assert migrated.pulse.width == 1.5          # run physics is retained
    assert not hasattr(migrated.ga, 'evolve_width')


def test_timing_toggles_round_trip_run_config():
    import dataclasses
    from evo_runtime.config import RunConfig
    config = RunConfig.from_dict({
        'pulse': {'model': 'pulse_delay'},
        'ga': {'node_model': 'pulse_delay', 'evolve_delay': False}})
    assert config.ga.timing_mutations() is False
    rebuilt = RunConfig.from_dict(dataclasses.asdict(config))
    assert rebuilt.ga.evolve_delay is False
    # configs and checkpoints without the toggle keep the model pairing
    legacy = RunConfig.from_dict({'pulse': {'model': 'pulse_delay'}})
    assert legacy.ga.timing_mutations() is True


def test_fixed_delay_ablation_never_grows_a_delay_vector():
    """Reproduction under node_model='pulse_delay' with evolve_delay=False (the
    width-preserving/fixed-delay ablation) must never mutate delays, while the
    paired default still can."""
    from evo_runtime.config import GAConfig
    from nv_evo.ga import next_population

    def offspring(config, seed):
        random.seed(seed)
        population = [random_hex_genome(2) for _ in range(6)]
        fitnesses = [i / 6 for i in range(6)]
        out = []
        for _ in range(30):
            population = next_population(
                population, fitnesses,
                make_genome=lambda: random_hex_genome(2),
                mean_mutations=3.0, ga_config=config)
            out.extend(population)
        return out

    paired = GAConfig(node_model='pulse_delay', chromosome_count=2)
    assert any(g.state_delays is not None for g in offspring(paired, 61))
    fixed = GAConfig(node_model='pulse_delay', evolve_delay=False,
                     chromosome_count=2)
    assert all(g.state_delays is None for g in offspring(fixed, 61))


def test_waveform_targets_declare_width_preserving_model():
    """Waveform contracts need input-dependent output durations, which only
    width-preserving transport can emit — they declare supported_models so a
    'uniform' run is filtered out instead of silently capping
    below 1.0. Event/trace targets stay open to every model."""
    for name in ('Pulse width sum (A+B)', 'Odd pulse selector'):
        assert tuple(TEMPORAL_TARGETS[name].supported_models) == ('pulse_delay',)
    for name in ('SR latch', 'Toggle flip-flop', 'Echo (delay 3)',
                 'A-count parity queried by B'):
        assert not TEMPORAL_TARGETS[name].supported_models
    restored = _target_from_dict(_target_to_dict(
        TEMPORAL_TARGETS['Pulse width sum (A+B)']))
    assert tuple(restored.supported_models) == ('pulse_delay',)


def test_mix_event_widths_never_reaches_the_next_same_lane_event():
    """Widened oracle pulses must fall strictly clear of the next event on the
    same lane: any overlap merges two labelled stimulus events into ONE physical
    edge (wired-OR / drive union), making the expected output unreachable in
    every model."""
    from nv_evo.oracle import _mix_event_widths, _WIDTH_CLEARANCE

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
    from target_ui import target_category, CATEGORY_ORDER
    from nv_evo.targets import periodic_combinational_target
    from snn_evo.targets import TARGETS

    assert 'Combinational logic' in CATEGORY_ORDER
    assert 'Pulse width & duration' in CATEGORY_ORDER
    for name, raw in TARGETS.items():
        assert target_category(name, raw) == 'Combinational logic'
        wrapped = periodic_combinational_target(raw)
        assert target_category(name, wrapped) == 'Combinational logic'
    for name in ('Pulse width sum (A+B)', 'Odd pulse selector',
                 'Pair detection gap (2x pulse width)'):
        assert target_category(name, TEMPORAL_TARGETS[name]) \
            == 'Pulse width & duration'
    # semantics-derived folders are unchanged for everything else
    assert target_category('SR latch', TEMPORAL_TARGETS['SR latch']) \
        == 'Memory & state'
    assert target_category('Echo (delay 3)', TEMPORAL_TARGETS['Echo (delay 3)']) \
        == 'Timed events'
    restored = _target_from_dict(_target_to_dict(
        TEMPORAL_TARGETS['Pulse width sum (A+B)']))
    assert restored.category == 'Pulse width & duration'


# ── standalone runner ────────────────────────────────────────────────────────────

def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except AssertionError as e:
            print("FAIL  %s\n      %s" % (fn.__name__, e))
        except Exception as e:                     # noqa: BLE001
            print("ERROR %s\n      %r" % (fn.__name__, e))
    print("\n%d/%d pulse-model tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
