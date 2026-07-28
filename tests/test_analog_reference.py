"""
tests/test_analog_reference.py — validate the analog engine against an
INDEPENDENT circuit reference (the reference plan's point 6).

substrates/nervous/analog.py is event-driven and solves the node's leak recovery and its
threshold crossings ANALYTICALLY (no time step). This suite cross-checks that
analytical engine against a naive fixed-Δt numerical integration of the same
physical node — a completely separate implementation — over the sweeps the plan
calls for: E1/E2 separation, repeated/dense pulses, output width vs the leak,
and recovery after firing. Agreement to a few Δt confirms the fast engine is a
faithful solution of the circuit ODE, not a re-encoding of the same shortcuts.

(A transistor-level SPICE deck would be the ultimate reference; this Δt
integrator is the portable stand-in — it needs no external simulator and pins
the same emergent quantities.)

Run under pytest, or standalone:  py tests/test_analog_reference.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.analog import AnalogPulseSim, AnalogConfig            # noqa: E402


def integrate_reference(terminal_edges, cfg, horizon, dt=2e-4):
    """Independent numerical model of ONE Fig. 1 node.

    ``terminal_edges`` = list of (time, n_terminals): each event steps the node
    down by n_terminals * step. Returns the output wire intervals (comparator
    intervals shifted by the propagation delay), matching analog.py's semantics:
    trip when v < threshold (and armed), then release and re-arm when v recovers
    to threshold + hysteresis.
    """
    rest, thr, step = cfg.rest, cfg.threshold, cfg.step
    tau, hyst, delay = cfg.tau_leak, cfg.hysteresis, cfg.delay_prop
    # bucket edge step-counts by Δt index
    steps = {}
    for (t, n) in terminal_edges:
        k = int(round(t / dt))
        steps[k] = steps.get(k, 0) + n
    v, tripped, armed, start = rest, False, True, None
    out, decay = [], math.exp(-dt / tau)
    n_steps = int(round(horizon / dt))
    for k in range(n_steps + 1):
        if k in steps:
            v -= step * steps[k]
        if (not tripped) and armed and v < thr:
            tripped, start = True, k * dt
        elif tripped and v >= thr + hyst:
            tripped = False
            out.append((start, k * dt))
            armed = True
        v = rest + (v - rest) * decay            # leak toward rest
    if tripped:
        out.append((start, horizon))
    return [(s + delay, e + delay) for (s, e) in out]


def _buffer_engine_intervals(input_edges, cfg, horizon):
    """Drive a single buffer node (both terminals on input 'a') and return its
    output intervals. Each input edge is one source edge = two terminal steps."""
    grid = {'a': 0, 'y': 0}
    routing = {'y': ('E', 'E', None, 'and'), 'a': (None, None, None, 'and')}
    sources = {'y': ('a', 'a', None), 'a': (None, None, None)}
    sim = AnalogPulseSim(grid, routing, config=cfg, sources=sources, inputs=['a'])
    w = 0.05                                     # short, non-overlapping edges
    for t in input_edges:
        sim.inject_pulse('a', t, w)
    sim.advance_to(horizon)
    return [tuple(iv) for iv in sim.pulse_intervals['y']]


def _assert_intervals_close(got, ref, tol):
    assert len(got) == len(ref), 'interval COUNT differs: engine %d vs ref %d' % (
        len(got), len(ref))
    for (gs, ge), (rs, re) in zip(got, ref):
        assert abs(gs - rs) <= tol, 'rise %.4f vs %.4f' % (gs, rs)
        assert abs(ge - re) <= tol, 'fall %.4f vs %.4f' % (ge, re)


def test_single_edge_width_matches_reference():
    cfg = AnalogConfig()
    got = _buffer_engine_intervals([0.0], cfg, 30.0)
    ref = integrate_reference([(0.0, 2)], cfg, 30.0)
    _assert_intervals_close(got, ref, tol=2e-3)


def test_dense_burst_width_matches_reference():
    cfg = AnalogConfig()
    edges = [0.0, 0.15, 0.30]                    # three quick edges -> one wide pulse
    got = _buffer_engine_intervals(edges, cfg, 30.0)
    ref = integrate_reference([(t, 2) for t in edges], cfg, 30.0)
    _assert_intervals_close(got, ref, tol=3e-3)


def test_recovery_then_second_pulse_matches_reference():
    cfg = AnalogConfig()
    # two edges far apart -> two separate output pulses (recovery + re-arm)
    edges = [0.0, 8.0]
    got = _buffer_engine_intervals(edges, cfg, 40.0)
    ref = integrate_reference([(t, 2) for t in edges], cfg, 40.0)
    assert len(got) == 2, 'expected two distinct pulses'
    _assert_intervals_close(got, ref, tol=3e-3)


def test_coincidence_separation_sweep_matches_reference():
    """E1/E2 separation sweep: for each gap, whether the coincidence FIRES must
    agree between the engine and the reference (the window is emergent)."""
    cfg = AnalogConfig()
    grid = {'a': 0, 'b': 0, 'y': 0}
    routing = {'y': ('E1', 'E2', None, 'and'),
               'a': (None, None, None, 'and'), 'b': (None, None, None, 'and')}
    sources = {'y': ('a', 'b', None), 'a': (None,) * 3, 'b': (None,) * 3}
    for gap in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        sim = AnalogPulseSim(grid, routing, config=cfg, sources=sources,
                             inputs=['a', 'b'])
        sim.inject_pulse('a', 0.0, 0.05)
        sim.inject_pulse('b', gap, 0.05)
        sim.advance_to(30.0)
        engine_fired = bool(sim.pulse_intervals['y'])
        ref = integrate_reference([(0.0, 1), (gap, 1)], cfg, 30.0)
        ref_fired = bool(ref)
        assert engine_fired == ref_fired, \
            'gap %.2f: engine fired=%s, reference fired=%s' % (
                gap, engine_fired, ref_fired)


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
    print("\n%d/%d analog-reference tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
