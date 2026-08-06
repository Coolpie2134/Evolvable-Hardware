"""
tests/test_engine_semantics.py - deterministic event-pulse engine semantics.

Geometry-free checks of the invariants the asynchronous model rests on, so a
refactor of substrates.nervous.pulse / substrates.nervous.simulation cannot silently change them:

  * WIRED-OR - a net re-driven while still high shows ONE leading edge, not two
    (inputs are pulse injections onto a shared net, not level clamps).
  * HELD LEVEL = ONE EDGE - a contiguous high run in a trial stream is one long
    pulse (one edge), so a stored bit is a single circulating pulse, not a train.
  * PULSE WIDTH - a driven wire is high for exactly [t, t+width).

Run under pytest, or standalone:  py tests/test_engine_semantics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous import pulse, simulation as sim                # noqa: E402
from substrates.nervous.pulse import PulseSim, PulseConfig             # noqa: E402
from substrates.nervous.hexgrid import ROUTING_HEX                     # noqa: E402
from substrates.nervous.temporal import run_nervous_events             # noqa: E402

TOL = 1e-9


def test_normalize_merges_overlapping_pulses():
    """Two pulses that overlap in time collapse to one (wired-OR); disjoint
    pulses stay separate."""
    # overlap: [1,3) and [2,5) -> [1,5)
    merged = sim.normalize([[(1.0, 2.0), (2.0, 3.0)]])
    assert merged == [[(1.0, 4.0)]], merged
    # touching at the boundary counts as overlapping: [1,3) and [3,4) -> [1,4)
    merged = sim.normalize([[(1.0, 2.0), (3.0, 1.0)]])
    assert merged == [[(1.0, 3.0)]], merged
    # disjoint with a gap stays two pulses
    merged = sim.normalize([[(1.0, 1.0), (5.0, 1.0)]])
    assert merged == [[(1.0, 1.0), (5.0, 1.0)]], merged


def test_effective_edges_are_post_merge_leading_edges():
    """The edge train the substrate receives is the leading edge of each merged
    run - an overlapping re-drive does NOT add an edge."""
    edges = sim.effective_edges([[(1.0, 2.0), (2.0, 3.0), (10.0, 1.0)]])
    assert edges == [[1.0, 10.0]], edges


def test_held_level_is_one_pulse():
    """A contiguous high run in a stream becomes ONE pulse (one edge) - the basis
    of circulating-pulse memory: a stored bit is a single pulse, not a train."""
    # stream: input 0 high on ticks 2,3,4 (a held level), then low
    streams = [[0], [0], [1], [1], [1], [0], [0]]
    sched = sim.streams_to_schedule(streams, n_inputs=1, T=7)
    assert len(sched[0]) == 1, sched            # one pulse, not three
    start, width = sched[0][0]
    assert abs(start - 2.0 * pulse.TICK) <= TOL
    assert width >= 3.0 * pulse.TICK - TOL      # spans the whole held run
    assert sim.effective_edges(sched) == [[2.0 * pulse.TICK]]


def test_pulse_width_high_window():
    """An injected pulse of width w makes its wire high for [t, t+w) and low
    after, on a single isolated cell."""
    grid = {(0, 0): 0}
    routing = {(0, 0): ROUTING_HEX[0]}
    s = PulseSim(grid, routing, config=PulseConfig(width=2.0))
    s.inject_pulse((0, 0), 5.0, 2.0)
    s.advance_to(20.0)
    assert s.activity_at(5.5) == {(0, 0): 1}    # inside the pulse
    assert s.activity_at(6.9) == {(0, 0): 1}
    assert s.activity_at(7.1) == {(0, 0): 0}    # after it ends
    assert s.rise_times[(0, 0)] == [5.0]        # exactly one leading edge


def test_wired_or_extend_no_second_edge():
    """Re-driving a wire while it is still high EXTENDS the pulse without a new
    leading edge (wired-OR), so `rise_times` records a single edge."""
    grid = {(0, 0): 0}
    routing = {(0, 0): ROUTING_HEX[0]}
    s = PulseSim(grid, routing, config=PulseConfig(width=3.0))
    s.inject_pulse((0, 0), 0.0, 3.0)            # high on [0, 3)
    s.inject_pulse((0, 0), 2.0, 3.0)            # re-driven while high -> [0, 5)
    s.advance_to(20.0)
    assert s.rise_times[(0, 0)] == [0.0], s.rise_times[(0, 0)]
    assert s.activity_at(4.5) == {(0, 0): 1}    # extended past the first end
    assert s.activity_at(5.1) == {(0, 0): 0}


def test_terminal_input_is_source_only():
    """A terminal input drives its reader but cannot be fired by that reader."""
    source, body = (0, 0), (1, 0)
    grid = {source: 2, body: 2}
    routing = {cell: ROUTING_HEX[state] for cell, state in grid.items()}

    reverse = PulseSim(
        grid, routing, config=PulseConfig(), input_nodes={source})
    reverse.inject_pulse(body, 0.0, 1.0)
    reverse.advance_to(4.0)
    assert reverse.rise_times[source] == []

    forward = PulseSim(
        grid, routing, config=PulseConfig(), input_nodes={source})
    forward.inject_pulse(source, 0.0, 1.0)
    forward.advance_to(4.0)
    assert forward.rise_times[body] == [1.0]


def test_terminal_output_is_observable_sink_only():
    """A sink can fire from the body, but no downstream node can read it."""
    source, sink, downstream = (0, 0), (1, 0), (2, 0)
    grid = {source: 0, sink: 2, downstream: 3}
    routing = {cell: ROUTING_HEX[state] for cell, state in grid.items()}
    terminal = PulseSim(
        grid, routing, config=PulseConfig(), input_nodes={source},
        output_nodes={sink})
    terminal.inject_pulse(source, 0.0, 1.0)
    terminal.advance_to(5.0)
    assert terminal.rise_times[sink] == [1.0]
    assert terminal.rise_times[downstream] == []


def test_explicit_fractional_trial_events_bypass_tick_injection():
    """A physical target event at 1.37 propagates from that exact time."""
    grid = {(0, 0): 0, (1, 0): 2}
    routing = {(0, 0): ROUTING_HEX[0], (1, 0): ROUTING_HEX[2]}
    _, _, rises, overflow = run_nervous_events(
        grid, routing, [(0, 0)], {'Q': (1, 0)}, [(0,)] * 6, 6,
        sample=False, input_events=[[(1.37, 1.0)]])
    assert not overflow
    assert rises[(0, 0)] == [1.37]
    assert rises[(1, 0)] == [2.37]


def test_snn_temporal_source_edges_are_ideal_and_not_refractory_limited():
    """Every external edge enters an SNN input port at its exact float time."""
    from substrates.snn.lif_sim import simulate_events
    from substrates.snn.snn import Neuron

    source = Neuron(
        id=0, x=0, y=0, state=1, vth=0.9, tau=15.0,
        excit=True, is_input=True)
    spikes = simulate_events(
        [source], [],
        [(1.37, 0, 1.0, 0.25), (2.62, 0, 1.0, 0.25)],
        sim_time=5.0)
    assert spikes[0] == [1.37, 2.62], spikes[0]


def test_snn_temporal_adapter_converts_seconds_to_lif_milliseconds():
    from substrates.snn.temporal import LIF_MS_PER_SECOND, _run_trials
    from substrates.snn.snn import Neuron
    from substrates.nervous.targets import OutputTerminal, TemporalTarget, Trial

    target = TemporalTarget(
        'SNN units probe', [(0, 0)], [OutputTerminal('Q', (1, 0))], 4,
        [Trial([(0,), (1,), (0,), (0,)], {'Q': [0, 1, 0, 0]})],
        grid_size=3)
    source = Neuron(
        id=0, x=0, y=0, state=1, vth=0.9, tau=15.0,
        excit=True, is_input=True)
    run = _run_trials([source], [], [(0, 0)], target)
    assert LIF_MS_PER_SECOND == 4.8
    assert run[1][0][(0, 0)] == [1.0]


def test_lut_temporal_output_fitting_uses_complete_intervals():
    """A duration target must select by rise *and fall*, not rise alone."""
    import numpy as np
    import substrates.lut.ga as lut_ga
    from substrates.nervous.contracts import interval_contract
    from substrates.nervous.targets import OutputTerminal, TemporalTarget, Trial

    target = TemporalTarget(
        'interval probe', [(0, 0)], [OutputTerminal('Q', (2, 0))], 6,
        [Trial([(0,)] * 6, {'Q': [0] * 6},
               expected_intervals={'Q': [(2.0, 4.0)]})],
        grid_size=5, contract=interval_contract(0.1))
    grid = {(0, 0): (0, 0, 0, 0),
            (1, 0): (0, 0, 0, 0),
            (2, 0): (0, 0, 0, 0)}

    class _Sim:
        _cidx = {(0, 0): 0, (1, 0): 1, (2, 0): 2}

    original = lut_ga._run_lut_trials

    def fake_trials(_grid, _inputs, _target, watch):
        events = {cell: ([2.0] if cell != (0, 0) else []) for cell in watch}
        intervals = {
            cell: ([(2.0, 4.0)] if cell == (1, 0)
                   else [(2.0, 3.0)] if cell == (2, 0) else [])
            for cell in watch}
        return (_Sim(), [np.zeros((18, 3), dtype=np.int8)],
                [events], [intervals], False)

    lut_ga._run_lut_trials = fake_trials
    try:
        out_pos, traces = lut_ga.place_outputs_by_trace(
            grid, [(0, 0)], target)
    finally:
        lut_ga._run_lut_trials = original
    assert out_pos['Q'] == (1, 0), out_pos
    assert traces.intervals['Q'] == [[(2.0, 4.0)]]


def test_temporal_snn_architecture_adds_explicit_recurrent_edges():
    """Recurrence is opt-in and leaves the combinational SNN graph unchanged."""
    from substrates.snn.snn import Arch, interpret_grid

    grid = {(0, 0): 1, (1, 0): 1}
    _neurons, feedforward = interpret_grid(grid, arch=Arch())
    _neurons, recurrent = interpret_grid(
        grid, arch=Arch(recurrent=True))
    assert {(edge.pre, edge.post) for edge in feedforward} == {(0, 1)}
    assert {(edge.pre, edge.post) for edge in recurrent} == {(0, 1), (1, 0)}


def test_recurrent_snn_can_sustain_a_physical_pulse_loop():
    """A slow two-neuron feedback loop persists after one source kick."""
    from substrates.snn.lif_sim import simulate_events
    from substrates.snn.snn import Neuron, Synapse

    neurons = [
        Neuron(0, 0, 0, 1, 0.3, 8.0, True, is_input=True),
        # vth=.9/tau=8 makes one hop take almost the full five-second EPSC;
        # the returning edge therefore arrives after the previous drive falls
        # and the neuron has re-armed.
        Neuron(1, 1, 0, 3, 0.9, 8.0, True),
        Neuron(2, 2, 0, 3, 0.9, 8.0, True),
    ]
    synapses = [
        Synapse(0, 1, 2.0), Synapse(1, 2, 2.0), Synapse(2, 1, 2.0)]
    spikes = simulate_events(
        neurons, synapses, [(1.0, 0, 4.0, 1.0)], sim_time=40.0)
    assert len(spikes[1]) >= 4 and spikes[1][-1] > 30.0, spikes


def test_recurrent_snn_loop_solves_period_two_oscillator_contract():
    """A designed physical witness separates expressibility from GA search."""
    from substrates.nervous.scoring import TemporalTraces, score_contract
    from substrates.nervous.targets import TEMPORAL_TARGETS
    from substrates.snn.lif_sim import _run
    from substrates.snn.snn import Neuron, Synapse
    from substrates.snn.temporal import LIF_MS_PER_SECOND

    target = TEMPORAL_TARGETS['Oscillator']
    neurons = [
        Neuron(0, 0, 0, 1, 0.3, 8.0, True, is_input=True),
        Neuron(1, 1, 0, 3, 0.9, 8.0, True),
        Neuron(2, 2, 0, 3, 0.9, 8.0, True),
    ]
    synapses = [
        Synapse(0, 1, 2.0), Synapse(1, 2, 2.0), Synapse(2, 1, 2.0)]
    traces = TemporalTraces()
    traces['Q'], traces.events['Q'], traces.intervals['Q'] = [], [], []
    for trial in target.trials:
        kick = next(tick for tick, row in enumerate(trial.streams) if row[0])
        run = _run(
            neurons, synapses, {},
            input_events=[(
                kick * LIF_MS_PER_SECOND, 0, 4.0, LIF_MS_PER_SECOND)],
            sim_time=target.T * LIF_MS_PER_SECOND, max_events=2048)
        events = [
            when / LIF_MS_PER_SECOND for when in run.spikes[1]]
        traces['Q'].append([])
        traces.events['Q'].append(events)
        traces.intervals['Q'].append([
            (when, min(float(target.T), when + 1.0)) for when in events])
    assert abs(score_contract(traces, target)[0] - 1.0) <= TOL


def test_recurrent_snn_event_cap_stops_runaway_feedback():
    """A pathological feedback circuit terminates and marks its trace invalid."""
    from substrates.snn.lif_sim import _run
    from substrates.snn.snn import Neuron, Synapse

    neurons = [
        Neuron(0, 0, 0, 1, 0.3, 8.0, True, is_input=True),
        Neuron(1, 1, 0, 3, 0.9, 8.0, True),
        Neuron(2, 2, 0, 3, 0.9, 8.0, True),
    ]
    synapses = [
        Synapse(0, 1, 2.0), Synapse(1, 2, 2.0), Synapse(2, 1, 2.0)]
    run = _run(
        neurons, synapses, {},
        input_events=[(1.0, 0, 4.0, 1.0)],
        sim_time=1000.0, max_events=5)
    assert run.overflow and run.event_count == 6


def test_temporal_snn_interactive_replays_seconds_timeline():
    """Interactive playback uses the same seconds conversion as fitness."""
    from ui.interactive import InteractiveTab
    from substrates.nervous.targets import TEMPORAL_TARGETS
    from substrates.snn.snn import Neuron, Synapse

    target = TEMPORAL_TARGETS['Oscillator']
    tab = InteractiveTab.__new__(InteractiveTab)
    tab._circuit = {'target': target}
    tab._neurons = [
        Neuron(0, 0, 0, 1, 0.3, 8.0, True, is_input=True),
        Neuron(1, 1, 0, 3, 0.9, 8.0, True),
        Neuron(2, 2, 0, 3, 0.9, 8.0, True),
    ]
    tab._synapses = [
        Synapse(0, 1, 2.0), Synapse(1, 2, 2.0), Synapse(2, 1, 2.0)]
    tab._in_pos = [(0, 0)]
    tab._snn_horizon = float(target.T)

    class _Editor:
        @staticmethod
        def schedule(_inputs):
            return {(0, 0): [(3.0, 1.0)]}

    tab._editor = _Editor()
    tab._snn_prepare_temporal()
    assert tab._snn_spikes[0] == [3.0]
    assert len(tab._snn_spikes[1]) >= 5
    assert tab._snn_times[-1] == float(target.T)
    assert not tab._snn_overflow


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
    print("\n%d/%d engine-semantics tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
