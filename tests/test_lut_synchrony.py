"""
tests/test_lut_synchrony.py — metamorphic "no hidden clock" audit for the LUT
array's asynchronous engine (lut_evo.pulse.AsyncLutSim), plus the lattice
quantization contract against the synchronous reference engine.

The nervous net earned its "genuinely asynchronous" claim through the
metamorphic relations in tests/test_synchrony.py. Moving the LUT substrate to
the same footing demands the same audit, adapted to LUT physics:

  * QUANTIZATION — with delay == TICK and stimuli on the integer tick lattice,
    the asynchronous engine reproduces the synchronous latched engine
    (lut.LutSim) bit for bit, including spontaneous power-on activity: the old
    engine is the quantization of the new one.
  * TRANSLATION — shifting all inputs by an arbitrary (sub-tick) delta shifts
    all output edges by exactly that delta. LUT arrays are usually
    spontaneously active (which breaks translation by construction — the
    power-on transient is anchored at t = 0), so this uses a quiescent AND
    terminating organism: eastward shift-register lines (E-table 0xFF00,
    index-0 bit clear), whose event trains end well before the horizon.
  * SCALE — scaling the schedule AND the gate delay by k scales every output
    edge by k. No absolute timescale is baked in.
  * DETERMINISM / EVENT-ORDER INDEPENDENCE — identical stimulus gives
    byte-identical edges regardless of injection submission order.
  * INERTIAL FILTERING — an input blip shorter than the gate delay does not
    propagate: a real gate's output node cannot follow it. This is the LUT
    analogue of the nervous node's refractory hysteresis, and exactly the
    behaviour a clocked engine cannot express (it would either quantise the
    blip away or stretch it to a full tick).
  * SPONTANEITY IS HONEST — a lookup table with its index-0 bit set fires at
    power-on with no input, at t = 0.0 exactly (real LUT physics, the one
    deliberate contrast with the nervous net's quiescence invariant).

Run under pytest, or standalone:  py tests/test_lut_synchrony.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lut_evo.genome import Genome, random_lut_chromosome        # noqa: E402
from lut_evo.lut import grow_lut, LutSim, SEED_STATE            # noqa: E402
from lut_evo.pulse import AsyncLutSim, LutConfig                # noqa: E402

TOL = 1e-9
SEEDS = ((0, 0), (2, 0))


# ── fixtures ─────────────────────────────────────────────────────────────────────

def _relay_patch():
    """A quiescent organism: a 3-wide bar of the paper's relay cells (0xFFFE
    per direction — high for any live neighbour input, index-0 bit clear, so
    no self-starting). NB once excited it sustains a parity blink forever
    (the square lattice is bipartite), so it fits quiescence checks but not
    horizon-sensitive event-count comparisons."""
    return {(x, y): SEED_STATE for x in range(6) for y in range(3)}


# E-facing table 0xFF00 = "output east iff my west input bit is set" (bit 3 of
# the LUT index), all other directions dead: a pure eastward shift register.
_SHIFT = (0, 0, 0xFF00, 0)


def _shift_lines():
    """A quiescent, TERMINATING organism: two parallel eastward shift-register
    lines. A pulse injected at a west end marches east one gate delay per cell
    and falls off the far end, so every event train is finite and ends well
    before the test horizon — exact event-count comparisons can't be broken by
    the horizon boundary (the same trick as the nervous audit's 'terminating'
    fixture)."""
    return {(x, y): _SHIFT for x in range(8) for y in range(2)}


def _run_events(grid, schedule, horizon, config=None):
    """Inject a float schedule [(cell, t, width), ...] and return every wire's
    leading-edge train."""
    sim = AsyncLutSim(grid, config=config)
    for cell, t, w in schedule:
        sim.inject_pulse(cell, t, w)
    sim.advance_to(float(horizon))
    assert not sim.overflow
    return {c: list(ts) for c, ts in sim.rise_times.items()}


def _assert_affine(base, other, shift=0.0, k=1.0, tol=TOL):
    assert set(base) == set(other), "different cells produced output"
    for cell in base:
        a, b = base[cell], other[cell]
        assert len(a) == len(b), (
            "cell %r event count %d != %d (a clock/quantization would do this)"
            % (cell, len(a), len(b)))
        for x, y in zip(a, b):
            assert abs(k * x + shift - y) <= tol, (
                "cell %r: expected %.9f, got %.9f" % (cell, k * x + shift, y))


_SCHED = [((0, 1), 1.0, 1.0), ((0, 1), 5.0, 1.0), ((0, 0), 3.0, 1.0)]


# ── tests ────────────────────────────────────────────────────────────────────────

def test_lattice_quantization_of_sync_engine():
    """delay == TICK + integer stimuli => bit-identical to LutSim, spontaneous
    power-on activity included (random LUT organisms nearly always have it).
    Both engine paths are audited: the vectorised lattice fast path AND the
    general event loop (forced by clearing the pristine flag) must agree with
    the synchronous reference — and with each other, edge times included."""
    random.seed(11)
    checked = 0
    while checked < 12:
        g = Genome(chromosomes=[random_lut_chromosome() for _ in range(2)],
                   tag=0.5)
        grid = grow_lut(g, seeds=SEEDS, grid_size=7, iters=20)
        if len(grid) < 3:
            continue
        streams = [tuple(random.randint(0, 1) for _ in SEEDS)
                   for _ in range(18)]
        Ba = LutSim(grid).run_bits(streams, list(SEEDS), 26)
        fast = AsyncLutSim(grid)
        Bf = fast.run_bits(streams, list(SEEDS), 26)
        slow = AsyncLutSim(grid)
        slow._pristine = False               # force the general event loop
        Bs = slow.run_bits(streams, list(SEEDS), 26)
        assert not fast.overflow and not slow.overflow
        assert np.array_equal(Ba, Bf), "fast path != sync on the tick lattice"
        assert np.array_equal(Ba, Bs), "event loop != sync on the tick lattice"
        assert fast.rise_times == slow.rise_times, "paths disagree on edges"
        assert fast.ever == slow.ever
        checked += 1


def test_translation_invariance_subtick():
    """Shifting every input by a sub-tick delta shifts every output edge by
    exactly that delta — the engine has no tick grid to snap to."""
    grid = _shift_lines()
    base = _run_events(grid, _SCHED, 40.0)
    assert sum(len(v) for v in base.values()) >= 4, "shift lines stayed silent"
    assert max(t for ts in base.values() for t in ts) < 25.0, (
        "fixture must terminate well before the horizon")
    for delta in (0.37, -0.13, 0.001, 1.0, 2.5, 3.14159):
        shifted = [(c, t + delta, w) for (c, t, w) in _SCHED]
        out = _run_events(grid, shifted, 40.0 + max(0.0, delta))
        _assert_affine(base, out, shift=delta)


def test_scale_covariance():
    """Scaling the schedule and the gate delay by k scales every output edge
    time by k — no absolute timescale is baked into the engine."""
    grid = _shift_lines()
    base = _run_events(grid, _SCHED, 40.0)
    for k in (2.0, 0.5, 3.7, 0.25):
        scaled = [(c, t * k, w * k) for (c, t, w) in _SCHED]
        out = _run_events(grid, scaled, 40.0 * k, config=LutConfig(delay=k))
        _assert_affine(base, out, k=k, tol=1e-6)


def test_determinism_and_order_independence():
    """Identical stimulus gives byte-identical edges; injections submitted in
    a shuffled order resolve by TIME, not by queue order."""
    grid = _shift_lines()
    base = _run_events(grid, _SCHED, 40.0)
    _assert_affine(base, _run_events(grid, _SCHED, 40.0))
    rng = random.Random(0)
    for _ in range(5):
        shuffled = list(_SCHED)
        rng.shuffle(shuffled)
        _assert_affine(base, _run_events(grid, shuffled, 40.0))


def test_inertial_blip_filtering():
    """An input blip shorter than the gate delay reaches the driven wire (the
    injection is physical) but does NOT propagate through any cell: with the
    inertial delay model the re-evaluation at the blip's trailing edge
    supersedes the pending response to its leading edge."""
    grid = _shift_lines()
    blip = [((0, 1), 2.0, 0.4)]                  # width 0.4 < delay 1.0
    out = _run_events(grid, blip, 30.0)
    assert out[(0, 1)] == [2.0], "the driven wire itself must show the blip"
    downstream = {c: ts for c, ts in out.items() if c != (0, 1)}
    assert sum(len(v) for v in downstream.values()) == 0, (
        "a sub-delay blip propagated: inertial filtering is broken")
    # the same pulse at a full delay width does propagate
    out = _run_events(grid, [((0, 1), 2.0, 1.0)], 30.0)
    assert sum(len(v) for c, v in out.items() if c != (0, 1)) > 0


def test_spontaneous_power_on_is_honest():
    """A table with its index-0 bit set fires with no input at exactly t=0 —
    real LUT physics, deliberately unlike the quiescent nervous net."""
    grid = {(0, 0): (0xFFFF, 0, 0, 0)}          # index-0 bit set on N
    sim = AsyncLutSim(grid)
    sim.advance_to(20.0)
    assert sim.rise_times[(0, 0)] == [0.0]
    # ... and the relay patch (index-0 bit clear) is inert without input
    quiet = AsyncLutSim(_relay_patch())
    quiet.advance_to(20.0)
    assert sum(len(v) for v in quiet.rise_times.values()) == 0


def test_lut_player_matches_direct_run():
    """The GUI playback path (lut_evo.playback.LutPlayer — the LUT twin of
    NervousPlayer, driven from the same pulse timeline) reproduces a direct
    engine run exactly: same edges whether the schedule is played through the
    dt-stepped cursor or injected and advanced in one go."""
    from lut_evo.playback import LutPlayer
    grid = _shift_lines()
    direct = _run_events(grid, _SCHED, 40.0)

    player = LutPlayer(grid, horizon=40.0)
    sched = {}
    for cell, t, w in _SCHED:
        sched.setdefault(cell, []).append((t, w))
    player.set_schedule(sched)
    while not player.at_end():
        player.step()
    played = {c: list(player.events_upto(c)) for c in grid}
    _assert_affine(direct, played)
    assert not player.overflow
    # activity at the cursor is the wires' current levels (a dict over cells)
    act = player.activity()
    assert set(act) == set(grid)


def test_float_time_target_runs_on_lut_backend():
    """The continuous-time pair target (fractional input_events) is now open
    to the LUT backend and scores deterministically without quantising."""
    from nv_evo.oracle import ORACLE_SPECS
    from lut_evo.ga import evaluate_lut_full, make_seed_genome
    target = ORACLE_SPECS['Pair gap 2x width (oracle)'](seed=99,
                                                        pulse_width=0.75)
    assert 'lut' in target.supported_backends
    random.seed(5)
    g = make_seed_genome(2)
    f1, c1 = evaluate_lut_full(g, target)
    f2, c2 = evaluate_lut_full(g, target)
    assert 0.0 <= f1 <= 1.0
    assert f1 == f2 and c1 == c2


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
    print("\n%d/%d LUT synchrony tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
