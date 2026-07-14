"""
tests/test_engine_semantics.py — paper-faithful pulse-engine semantics.

Geometry-free checks of the invariants the asynchronous model rests on, so a
refactor of nv_evo.pulse / nv_evo.simulation cannot silently change them:

  * WIRED-OR — a net re-driven while still high shows ONE leading edge, not two
    (inputs are pulse injections onto a shared net, not level clamps).
  * HELD LEVEL = ONE EDGE — a contiguous high run in a trial stream is one long
    pulse (one edge), so a stored bit is a single circulating pulse, not a train.
  * PULSE WIDTH — a driven wire is high for exactly [t, t+width).

Run under pytest, or standalone:  py tests/test_engine_semantics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import pulse, simulation as sim                # noqa: E402
from nv_evo.pulse import PulseSim, PulseConfig             # noqa: E402
from nv_evo.hexgrid import ROUTING_HEX                     # noqa: E402
from nv_evo.temporal import run_nervous_events             # noqa: E402

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
    run — an overlapping re-drive does NOT add an edge."""
    edges = sim.effective_edges([[(1.0, 2.0), (2.0, 3.0), (10.0, 1.0)]])
    assert edges == [[1.0, 10.0]], edges


def test_held_level_is_one_pulse():
    """A contiguous high run in a stream becomes ONE pulse (one edge) — the basis
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
