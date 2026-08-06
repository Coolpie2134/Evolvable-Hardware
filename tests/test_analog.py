"""
tests/test_analog.py - the analog Fig. 1 node engine (substrates/nervous/analog.py).

The digital engine hard-codes coincidence width, output width and refractory as
independent constants. The analog node DERIVES them from one charge/leak/
comparator mechanism. These tests pin the emergent behaviours that were the
whole point of the model - each is a property the digital engine cannot exhibit
from its own parameters:

  * a BUFFER (both terminals on one source) fires on a SINGLE edge;
  * a COINCIDENCE node (two different sources) does NOT fire on one edge, but
    DOES when both arrive close together, and does NOT when they are far apart
    (the window emerges from the leak, not a COINC knob);
  * dense/overlapping input STRETCHES the output pulse (paralyzable node);
  * an active inhibitory input vetoes an otherwise-firing coincidence;
  * determinism and the AnalogConfig physical-consistency guards.

Run under pytest, or standalone:  py tests/test_analog.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.analog import AnalogPulseSim, AnalogConfig            # noqa: E402
from substrates.nervous.pulse import PulseConfig                              # noqa: E402
from substrates.nervous.simulation import create_simulator                    # noqa: E402


# Build tiny explicit graphs with pre-resolved sources so the geometry is exact.
# Node 'y' is the circuit under test; 'a','b' are input wires; 'i' inhibitory.

def _buffer_graph():
    grid = {'a': 0, 'y': 0}
    routing = {'y': ('E', 'E', None, 'and'), 'a': (None, None, None, 'and')}
    sources = {'y': ('a', 'a', None), 'a': (None, None, None)}
    return grid, routing, sources


def _coincidence_graph():
    grid = {'a': 0, 'b': 0, 'y': 0}
    routing = {'y': ('E1', 'E2', None, 'and'),
               'a': (None, None, None, 'and'), 'b': (None, None, None, 'and')}
    sources = {'y': ('a', 'b', None), 'a': (None,) * 3, 'b': (None,) * 3}
    return grid, routing, sources


def _inhibited_coincidence_graph():
    grid = {'a': 0, 'b': 0, 'i': 0, 'y': 0}
    routing = {'y': ('E1', 'E2', 'I', 'and')}
    routing.update({c: (None, None, None, 'and') for c in ('a', 'b', 'i')})
    sources = {'y': ('a', 'b', 'i'), 'a': (None,) * 3, 'b': (None,) * 3,
               'i': (None,) * 3}
    return grid, routing, sources


def _sim(grid, routing, sources, config=None):
    return AnalogPulseSim(grid, routing, config=config or AnalogConfig(),
                          sources=sources, inputs=[c for c in grid if c != 'y'])


def test_buffer_fires_on_single_edge():
    grid, routing, sources = _buffer_graph()
    sim = _sim(grid, routing, sources)
    sim.inject_pulse('a', 0.0, 1.0)
    sim.advance_to(20.0)
    assert sim.rise_times['y'], 'buffer did not fire on a single edge'


def test_coincidence_needs_two_edges():
    grid, routing, sources = _coincidence_graph()
    sim = _sim(grid, routing, sources)
    sim.inject_pulse('a', 0.0, 1.0)         # only ONE excitatory line
    sim.advance_to(20.0)
    assert not sim.rise_times['y'], 'coincidence fired on a single input'


def test_coincidence_fires_when_edges_are_close():
    grid, routing, sources = _coincidence_graph()
    sim = _sim(grid, routing, sources)
    sim.inject_pulse('a', 0.0, 1.0)
    sim.inject_pulse('b', 0.05, 1.0)        # within a leak time-constant
    sim.advance_to(20.0)
    assert sim.rise_times['y'], 'coincidence did not fire on two close edges'


def test_coincidence_window_emerges_from_leak():
    grid, routing, sources = _coincidence_graph()
    cfg = AnalogConfig()
    sim = _sim(grid, routing, sources, cfg)
    # Second edge far enough that the first has leaked back above the point
    # where a single further step can cross threshold: no fire. Choose a gap of
    # several tau so the first step's residue is negligible.
    sim.inject_pulse('a', 0.0, 1.0)
    sim.inject_pulse('b', 6.0 * cfg.tau_leak, 1.0)
    sim.advance_to(30.0)
    assert not sim.rise_times['y'], 'coincidence fired on widely separated edges'


def test_dense_input_stretches_output_pulse():
    grid, routing, sources = _buffer_graph()
    cfg = AnalogConfig()
    # A single short edge -> baseline output width. (A held/overlapping pulse is
    # ONE edge by wired-OR, so use short non-overlapping pulses to get two edges.)
    s1 = _sim(grid, routing, sources, cfg)
    s1.inject_pulse('a', 0.0, 0.1)
    s1.advance_to(40.0)
    iv1 = s1.pulse_intervals['y'][0]
    w1 = iv1[1] - iv1[0]
    # A second edge while the comparator is still tripped pushes the node deeper,
    # so it takes longer to leak back to threshold: the pulse STRETCHES.
    s2 = _sim(grid, routing, sources, cfg)
    s2.inject_pulse('a', 0.0, 0.1)
    s2.inject_pulse('a', 0.15, 0.1)
    s2.advance_to(40.0)
    assert len(s2.pulse_intervals['y']) == 1, 'expected one stretched pulse, not a retrigger'
    iv2 = s2.pulse_intervals['y'][0]
    w2 = iv2[1] - iv2[0]
    assert w2 > w1 + 1e-6, 'dense input did not stretch the pulse (%.3f vs %.3f)' % (w2, w1)


def test_inhibition_vetoes_coincidence():
    grid, routing, sources = _inhibited_coincidence_graph()
    sim = _sim(grid, routing, sources)
    # hold the inhibitory wire high across the coincidence
    sim.inject_pulse('i', 0.0, 5.0)
    sim.inject_pulse('a', 1.0, 1.0)
    sim.inject_pulse('b', 1.05, 1.0)
    sim.advance_to(20.0)
    assert not sim.rise_times['y'], 'inhibition failed to veto a coincidence'


def test_inhibition_released_allows_fire():
    grid, routing, sources = _inhibited_coincidence_graph()
    sim = _sim(grid, routing, sources)
    sim.inject_pulse('i', 0.0, 1.0)          # veto ends before the pair
    sim.inject_pulse('a', 3.0, 1.0)
    sim.inject_pulse('b', 3.05, 1.0)
    sim.advance_to(20.0)
    assert sim.rise_times['y'], 'coincidence blocked even after inhibition ended'


def test_determinism():
    grid, routing, sources = _coincidence_graph()
    outs = []
    for _ in range(3):
        sim = _sim(grid, routing, sources)
        sim.inject_pulse('a', 0.0, 1.0)
        sim.inject_pulse('b', 0.1, 1.0)
        sim.advance_to(20.0)
        outs.append(tuple(sim.rise_times['y']))
    assert outs[0] == outs[1] == outs[2]


def test_config_rejects_unphysical_step():
    # step must be < gap (one edge can't fire) and 2*step > gap (a pair can).
    ok = False
    try:
        AnalogConfig(step=0.6)               # 0.6 > gap 0.5 -> one edge fires
    except ValueError:
        ok = True
    assert ok, 'AnalogConfig accepted a step that lets one edge fire a coincidence'
    ok = False
    try:
        AnalogConfig(step=0.2)               # 2*0.2 = 0.4 < gap 0.5 -> pair can't fire
    except ValueError:
        ok = True
    assert ok, 'AnalogConfig accepted a step too small for a pair to fire'


def test_run_config_reproduces_analog_constants():
    grid, routing, sources = _buffer_graph()
    cfg = PulseConfig(model='paper_analog', delay=1.25,
                      analog_threshold=0.45, analog_step=0.31,
                      analog_tau_leak=1.7, analog_hysteresis=0.09)
    sim = create_simulator(grid, routing, config=cfg, sources=sources)
    assert isinstance(sim, AnalogPulseSim)
    assert sim.config.delay_prop == 1.25
    assert sim.config.threshold == 0.45
    assert sim.config.step == 0.31
    assert sim.config.tau_leak == 1.7
    assert sim.config.hysteresis == 0.09


def test_discrete_step_surface_matches_held_input_semantics():
    grid, routing, sources = _buffer_graph()
    sim = _sim(grid, routing, sources)
    for _ in range(5):
        sim.step({'a': 1})
    sim.advance_to(20.0)
    assert len(sim.rise_times['a']) == 1
    assert len(sim.rise_times['y']) == 1


# -- standalone runner ------------------------------------------------------------

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
    print("\n%d/%d analog-node tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())


def test_analog_node_honours_the_or_routing_op():
    """An OR routing must fire from EITHER excitatory input under analog physics.

    The engine decides firing by summed charge, so it originally read only
    (e1, e2, i1) and dropped the op. Every OR routing therefore behaved as its
    AND twin: half the state alphabet was inert, and the tri tile - whose
    channels are the same alphabet - could not sustain a circulating pulse at
    all (Oscillator scored 0 under paper_analog while the digital engines
    solved it). OR is wired as the buffer's own trick: each input couples to
    both terminals, so one edge delivers 2*step.
    """
    from substrates.nervous.analog import AnalogPulseSim

    grid = {(0, 0): 1, (0, 1): 1, (1, 0): 1}
    src = {(1, 0): ((0, 0), (0, 1), None)}       # two distinct excitatory feeds

    def fired(op):
        routing = {(1, 0): ('L', 'R', None, op)}
        sim = AnalogPulseSim(grid, routing, sources=src)
        sim.inject_pulse((0, 0), 0.0)            # ONE of the two inputs only
        sim.advance_to(20.0)
        return bool(sim.rise_times.get((1, 0)))

    assert not fired('and'), 'coincidence fired on a single input'
    assert fired('or'), 'OR routing failed to fire on a single input'
