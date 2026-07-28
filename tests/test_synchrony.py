"""
tests/test_synchrony.py — metamorphic "no hidden clock" audit for the nervous
net's asynchronous pulse engine.

The project's north-star claim is that the substrate is *genuinely asynchronous*
(continuous-time, event-driven), not merely a synchronous machine clocked at TICK
granularity. A single lucky trace cannot establish that; a clock cheat has bitten
this project before ("combinational scoring was phantom / fixed-tick"). These
tests exercise the engine (substrates.nervous.pulse.PulseSim, driven through the audited
float path in substrates.nervous.simulation) with metamorphic relations that a truly
time-invariant system must satisfy and a tick-quantized one cannot:

  * TRANSLATION — shifting all inputs by an arbitrary (sub-tick) delta shifts all
    outputs by exactly that delta. This is the core clock-freedom test: a clocked
    engine quantizes a 0.37-unit shift to 0 or 1 tick and breaks the relation.
  * SCALE — scaling the schedule AND the physical constants by k scales every
    output time by k. There is no absolute timescale baked into the engine.
  * DETERMINISM — identical stimulus gives byte-identical output.
  * EVENT-ORDER INDEPENDENCE — injecting the same pulses in a different submission
    order gives identical output; concurrency is resolved by event time, not by
    the order work was queued.
  * NO SPONTANEOUS ACTIVITY — with no input the array is inert (the paper's "all
    switches start at zero, which prevents any signals from propagating").

Run under pytest, or standalone:  py tests/test_synchrony.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous import random_hex_genome                       # noqa: E402
from substrates.nervous import pulse, simulation as sim                # noqa: E402
from substrates.nervous.pulse import PulseConfig                       # noqa: E402
from substrates.nervous.nervous import grow_nervous, interpret_nervous # noqa: E402

SEEDS = ((0, 0), (2, 0))
TOL = 1e-6


# ── net fixtures (grown once, reused) ────────────────────────────────────────────

def _grow_active(search_seed, terminating):
    """Grow a small random net that produces output activity. When `terminating`
    the output must die out well before the horizon (last event < 25) so a shift
    or scale of a few units cannot clip an event at the horizon boundary — the
    counts then match exactly. Returns (routing, in_pos, schedule)."""
    random.seed(search_seed)
    sched = [[(1.0, 1.0), (5.0, 1.0)], [(3.0, 1.0)]]
    for _ in range(6000):
        grid = grow_nervous(random_hex_genome(2), seeds=SEEDS,
                            grid_size=6, iters=12)
        if len(grid) < 4:
            continue
        routing, _, _ = interpret_nervous(grid)
        in_pos = list(SEEDS)
        out, overflow = sim.run_schedule(grid, routing, in_pos, sched, 60.0)
        n = sum(len(v) for v in out.values())
        last = max((t for ts in out.values() for t in ts), default=0.0)
        if overflow or n < 4:
            continue
        if terminating and last >= 25.0:
            continue
        return grid, routing, in_pos, sched
    raise AssertionError("no active net found (search seed %d)" % search_seed)


_TERMINATING = _grow_active(1, terminating=True)
_OSCILLATING = _grow_active(1, terminating=False)


# ── comparison helpers ───────────────────────────────────────────────────────────

def _run_sched(net, sched, horizon, config=None):
    grid, routing, in_pos, _ = net
    trace, overflow = sim.run_schedule(grid, routing, in_pos, sched, horizon,
                                       config=config)
    assert not overflow
    return trace


def _assert_affine(base, other, shift=0.0, k=1.0, tol=TOL):
    """Every cell's event train in `other` equals `base` mapped t -> k*t+shift."""
    assert set(base) == set(other), "different cells produced output"
    for cell in base:
        a, b = base[cell], other[cell]
        assert len(a) == len(b), (
            "cell %r event count %d != %d (a clock/quantization would do this)"
            % (cell, len(a), len(b)))
        for x, y in zip(a, b):
            assert abs(k * x + shift - y) <= tol, (
                "cell %r: expected %.9f, got %.9f" % (cell, k * x + shift, y))


# ── tests ────────────────────────────────────────────────────────────────────────

def test_translation_invariance_subtick():
    """Shifting every input by a sub-tick delta shifts every output by exactly
    that delta — the engine has no tick grid to snap to."""
    net = _TERMINATING
    base = _run_sched(net, net[3], 60.0)
    for delta in (0.37, -0.13, 0.001, 1.0, 2.5, 3.14159):
        shifted = sim.translate(net[3], delta)
        out = _run_sched(net, shifted, 60.0 + max(0.0, delta))
        _assert_affine(base, out, shift=delta)


def test_scale_covariance():
    """Scaling the schedule and the physical constants (delay/width/coincidence)
    by k scales every output time by k — no absolute timescale is baked in."""
    net = _TERMINATING
    base = _run_sched(net, net[3], 60.0)
    for k in (2.0, 0.5, 3.7, 0.25):
        cfg = PulseConfig(delay=pulse.DELAY * k, width=pulse.WIDTH * k,
                          coincidence=pulse.COINC * k)
        out = _run_sched(net, sim.scale(net[3], k), 60.0 * k, config=cfg)
        _assert_affine(base, out, k=k, tol=1e-5)


def test_determinism():
    """Identical stimulus produces byte-identical output."""
    net = _TERMINATING
    a = _run_sched(net, net[3], 60.0)
    b = _run_sched(net, net[3], 60.0)
    _assert_affine(a, b)


def test_no_spontaneous_activity():
    """With no input the array stays inert — the no-spontaneous-activity
    invariant that makes the nervous net (unlike the LUT array) quiescent."""
    grid, routing, in_pos, _ = _OSCILLATING
    empty = [[] for _ in in_pos]
    trace, overflow = sim.run_schedule(grid, routing, in_pos, empty, 100.0)
    assert not overflow
    assert sum(len(v) for v in trace.values()) == 0, "fired with no input"


def test_event_order_independence():
    """Injecting the same pulses in a shuffled submission order yields identical
    output: the event heap resolves concurrency by TIME, not by queue order."""
    grid, routing, in_pos, sched = _TERMINATING
    events = [(in_pos[i], t, w) for i, ev in enumerate(sched) for (t, w) in ev]

    def run_in_order(order):
        s = sim.create_simulator(grid, routing)
        for cell, t, w in order:
            s.inject_pulse(cell, t, w)
        s.advance_to(80.0)
        return {c: list(s.rise_times.get(c, [])) for c in grid}

    base = run_in_order(events)
    rng = random.Random(0)
    for _ in range(5):
        shuffled = events[:]
        rng.shuffle(shuffled)
        _assert_affine(base, run_in_order(shuffled))


def test_translation_on_oscillator_by_ordinal():
    """Even a self-sustaining (free-oscillating) net translates cleanly: the
    k-th event on each cell shifts by exactly delta. Compared by ordinal so the
    horizon boundary (which only affects the final event) can't cause a spurious
    mismatch."""
    grid, routing, in_pos, sched = _OSCILLATING
    delta = 0.37
    base = _run_sched(_OSCILLATING, sched, 80.0)
    out = _run_sched(_OSCILLATING, sim.translate(sched, delta), 80.0 + delta)
    fired = 0
    for cell in base:
        a, b = base[cell], out[cell]          # silent cells have empty trains
        for x, y in zip(a, b):                # compare by ordinal (horizon-safe)
            assert abs(x + delta - y) <= TOL, (
                "oscillator cell %r not translation-invariant: %.9f vs %.9f"
                % (cell, x + delta, y))
        if a:
            fired += 1
    assert fired >= 1, "oscillator produced no output to compare"


# ── standalone runner (pytest not required) ──────────────────────────────────────

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
    print("\n%d/%d synchrony tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
