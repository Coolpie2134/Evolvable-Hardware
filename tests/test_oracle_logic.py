"""
tests/test_oracle_logic.py — the reference state machines that DEFINE the
input-driven temporal targets. These are the ground truth every evolved circuit
is scored against, so a change to one silently redefines the goal; pin them.

Fast and pure (no growth, no simulation, no multiprocessing).

Run under pytest, or standalone:  py tests/test_oracle_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import TEMPORAL_TARGETS                        # noqa: E402
from nv_evo.oracle import (make_c_element, orc_sr_latch,   # noqa: E402
                           orc_toggle, make_pulse_doubler,
                           orc_period_doubler, ORACLE_SPECS,
                           make_refractory_filter,
                           make_a_first_rendezvous,
                           make_collision_serializer,
                           make_watchdog)


def _trace(fn, seq):
    st, out = None, []
    for inb in seq:
        ob, st = fn(inb, st)
        out.append(ob[0])
    return out


def test_c_element_is_a_rendezvous():
    """Emit only after BOTH inputs have produced an edge (either order), then
    rearm; a lone input edge must never emit."""
    f = make_c_element()
    # A@0 then B@2 -> emit@2; A&B@4 -> emit@4; lone A@6 -> no emit
    assert _trace(f, [(1, 0), (0, 0), (0, 1), (0, 0), (1, 1), (0, 0), (1, 0)]) \
        == [0, 0, 1, 0, 1, 0, 0]
    # B-before-A also completes the rendezvous (order-independent)
    assert _trace(f, [(0, 1), (1, 0)]) == [0, 1]
    # lone edges on either input never emit
    assert _trace(f, [(1, 0)] * 5) == [0] * 5
    assert _trace(f, [(0, 1)] * 5) == [0] * 5
    # simultaneous edges emit immediately
    assert _trace(f, [(1, 1)]) == [1]


def test_sr_latch_set_reset_hold():
    """Set on A, reset on B, hold otherwise."""
    assert _trace(orc_sr_latch, [(1, 0), (0, 0), (0, 0)]) == [1, 1, 1]   # set holds
    assert _trace(orc_sr_latch, [(1, 0), (0, 1), (0, 0)]) == [1, 0, 0]   # reset clears
    assert _trace(orc_sr_latch, [(0, 0), (0, 0)]) == [0, 0]              # never set


def test_toggle_parity():
    """Each input edge flips the stored bit."""
    assert _trace(orc_toggle, [(1,), (0,), (1,), (0,), (1,)]) == [1, 1, 0, 0, 1]


def test_pulse_doubler_widths():
    """A pulse held x ticks yields one contiguous 2x-tick output hold starting
    with it; quiet input stays quiet; a pulse merging into the tail banks debt
    (total output = 2 x total input ticks)."""
    f = make_pulse_doubler()
    for w in (1, 2, 4):
        seq = [(0,)] * 3 + [(1,)] * w + [(0,)] * (3 * w)
        out = _trace(f, seq)
        run = sum(out)
        first = out.index(1)
        assert run == 2 * w and first == 3, (w, out)
        assert all(out[first + k] for k in range(run)), "hold must be contiguous"
    assert sum(_trace(f, [(0,)] * 8)) == 0
    # overlap: 2 in-ticks then 1 more during the tail -> 6 contiguous out-ticks
    out = _trace(f, [(1,), (1,), (0,), (1,)] + [(0,)] * 8)
    assert sum(out) == 6 and out[:6] == [1] * 6, out


def test_period_doubler_halves_edge_rate():
    """Emit on the 1st, 3rd, 5th... input edge: a period-p input train yields a
    period-2p output train; no input edge, no output."""
    for p in (2, 3, 4):
        seq = [(1 if t % p == 0 else 0,) for t in range(4 * p + 1)]
        out = _trace(orc_period_doubler, seq)
        fired = [t for t, v in enumerate(out) if v]
        assert fired == [0, 2 * p, 4 * p], (p, fired)   # intervals exactly 2p
    assert sum(_trace(orc_period_doubler, [(0,)] * 8)) == 0


def test_period_doubler_bank_mixes_periods():
    """The registry bank must mix MULTIPLE input periods (a fixed free-running
    cadence fits only one) and include a silent guard trial (kills oscillators).
    Period 1 must NOT appear: a pulse every tick wired-OR merges into one held
    level — physically it carries no period."""
    t = TEMPORAL_TARGETS['Period doubler (2x)']
    assert t.score_mode == 'events'
    periods, silent = set(), 0
    for tr in t.trials:
        ticks = [i for i, s in enumerate(tr.streams) if s[0]]
        if not ticks:
            silent += 1
            continue
        gaps = {b - a for a, b in zip(ticks, ticks[1:])}
        assert len(gaps) == 1, "each trial is one periodic train"
        periods.add(gaps.pop())
    assert len(periods) >= 2 and 1 not in periods, periods
    assert silent >= 1
    assert 'Period doubler (oracle)' in ORACLE_SPECS
    assert 'Pulse doubler (oracle)' in ORACLE_SPECS   # width variant kept available


def test_c_element_registered_as_target():
    """The C-element is reachable from the GUI/registry and the holdout spec."""
    assert 'C-element (2-in join)' in TEMPORAL_TARGETS
    assert 'C-element (oracle)' in ORACLE_SPECS
    t = TEMPORAL_TARGETS['C-element (2-in join)']
    assert t.score_mode == 'events'
    assert len(t.inputs) == 2
    # every trial with a positive expectation must have at least one output event
    assert any(1 in [x for x in tr.expected['Q'] if x is not None] for tr in t.trials)


def test_refractory_filter_dead_time_boundary():
    """An accepted event blocks exactly the next three ticks, then re-arms."""
    f = make_refractory_filter(3)
    # t=0 fires; t=2 is blocked; t=4 is the earliest accepted event.
    assert _trace(f, [(1,), (0,), (1,), (0,), (1,), (0,)]) \
        == [1, 0, 0, 0, 1, 0]
    assert _trace(f, [(0,)] * 8) == [0] * 8


def test_a_first_rendezvous_order_ties_and_rearm():
    """Only A-first rounds emit; B-first and ties are consumed silently."""
    f = make_a_first_rendezvous()
    seq = [
        (1, 0), (1, 0), (0, 0), (0, 1),  # A first, repeat ignored -> emit
        (0, 1), (0, 0), (1, 0),          # B first -> silent
        (1, 1),                            # tie -> silent and immediately rearm
        (1, 0), (0, 1),                   # A first again -> emit
    ]
    assert _trace(f, seq) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def test_collision_serializer_preserves_tokens():
    """A collision becomes two spaced events; isolated inputs stay one-for-one."""
    f = make_collision_serializer(2)
    seq = [(1, 1), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)]
    out = _trace(f, seq)
    assert out == [1, 0, 1, 0, 1, 0, 0]
    assert sum(out) == sum(a + b for a, b in seq)


def test_watchdog_deadline_rearm_and_never_armed_silence():
    """Deadline heartbeats win; quiet alarms once; later heartbeat re-arms."""
    f = make_watchdog(5)
    seq = [
        (1,), (0,), (0,), (0,), (0,), (1,),  # heartbeat at deadline cancels
        (0,), (0,), (0,), (0,), (0,),        # then five quiet ticks -> alarm
        (0,), (1,),                           # stay disarmed, then re-arm
        (0,), (0,), (0,), (0,), (0,),        # another five quiet -> alarm
    ]
    assert _trace(f, seq) == [0] * 10 + [1, 0, 0] + [0] * 4 + [1]
    assert _trace(f, [(0,)] * 12) == [0] * 12


def test_new_async_oracles_are_registered_and_seeded():
    """All presets reach the GUI/certifier and fresh seeds vary their timings."""
    pairs = {
        'Refractory filter (3 ticks)': 'Refractory filter (oracle)',
        'A-first rendezvous': 'A-first rendezvous (oracle)',
        'Collision serializer (2-to-1)': 'Collision serializer (oracle)',
        'Watchdog timeout (5 ticks)': 'Watchdog timeout (oracle)',
    }
    for display_name, spec_name in pairs.items():
        assert display_name in TEMPORAL_TARGETS
        assert spec_name in ORACLE_SPECS
        registered = TEMPORAL_TARGETS[display_name]
        fresh_a = ORACLE_SPECS[spec_name](seed=101)
        fresh_b = ORACLE_SPECS[spec_name](seed=202)
        assert registered.score_mode == fresh_a.score_mode == 'events'
        assert registered.n_outputs == fresh_a.n_outputs == 1
        assert len(registered.inputs) == len(fresh_a.inputs)
        assert [tr.streams for tr in fresh_a.trials] != [tr.streams for tr in fresh_b.trials]

    serializer = ORACLE_SPECS['Collision serializer (oracle)'](seed=303)
    for trial in serializer.trials:
        n_input_events = sum(sum(bits) for bits in trial.streams)
        assert len(trial.expected_events['Q']) == n_input_events


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except Exception as e:                     # noqa: BLE001
            print("%s  %s: %s" % ("FAIL" if isinstance(e, AssertionError)
                                  else "ERROR", fn.__name__, e))
    print("\n%d/%d oracle-logic tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
