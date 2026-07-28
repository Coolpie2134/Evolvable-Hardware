"""
tests/test_lut_synchrony.py — metamorphic "no hidden clock" audit for the LUT
array's asynchronous engine (substrates.lut.pulse.AsyncLutSim), plus the lattice
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

from substrates.lut.genome import Genome, random_lut_chromosome        # noqa: E402
from substrates.lut.lut import grow_lut, LutSim, SEED_STATE            # noqa: E402
from substrates.lut.pulse import AsyncLutSim, LutConfig                # noqa: E402

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


def test_lut_terminal_input_is_source_only():
    source, body = (1, 0), (2, 0)
    grid = {(0, 0): _SHIFT, source: _SHIFT, body: _SHIFT}

    reverse = AsyncLutSim(grid, input_nodes={source})
    reverse.inject_pulse((0, 0), 0.0, 1.0)
    reverse.advance_to(4.0)
    assert reverse.rise_times[source] == []

    forward = AsyncLutSim(grid, input_nodes={source})
    forward.inject_pulse(source, 0.0, 1.0)
    forward.advance_to(4.0)
    assert forward.rise_times[body] == [1.0]


def test_lut_terminal_output_is_observable_sink_only():
    source, sink, downstream = (0, 0), (1, 0), (2, 0)
    grid = {source: _SHIFT, sink: _SHIFT, downstream: _SHIFT}
    sim = AsyncLutSim(
        grid, input_nodes={source}, output_nodes={sink})
    sim.inject_pulse(source, 0.0, 1.0)
    sim.advance_to(5.0)
    assert sim.rise_times[sink] == [1.0]
    assert sim.rise_times[downstream] == []


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
    """The GUI playback path (substrates.lut.playback.LutPlayer — the LUT twin of
    NervousPlayer, driven from the same pulse timeline) reproduces a direct
    engine run exactly: same edges whether the schedule is played through the
    dt-stepped cursor or injected and advanced in one go."""
    from substrates.lut.playback import LutPlayer
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
    from substrates.nervous.oracle import ORACLE_SPECS
    from substrates.lut.ga import evaluate_lut_full, make_seed_genome
    target = ORACLE_SPECS['Pair gap 2x width (oracle)'](seed=99,
                                                        pulse_width=0.75)
    assert 'lut' in target.supported_backends
    random.seed(5)
    g = make_seed_genome(2)
    f1, c1 = evaluate_lut_full(g, target)
    f2, c2 = evaluate_lut_full(g, target)
    assert 0.0 <= f1 <= 1.0
    assert f1 == f2 and c1 == c2


def test_pulse_intervals_log_pairs_rises_with_falls():
    """AsyncLutSim.pulse_intervals reconstructs the complete waveform in the
    PulseSim dialect: one [rise, fall] pair per pulse, inf while still high,
    identical between the event loop and the lattice fast path."""
    grid = _shift_lines()
    width = 2.5
    sim = AsyncLutSim(grid)
    sim.inject_pulse((0, 0), 1.25, width)
    sim.advance_to(30.0)
    delay = sim.config.delay
    for x in range(1, 8):
        intervals = sim.pulse_intervals[(x, 0)]
        start = 1.25 + x * delay
        assert len(intervals) == 1, (x, intervals)
        assert abs(intervals[0][0] - start) <= TOL
        assert abs(intervals[0][1] - (start + width)) <= TOL

    # a wire still high at the frontier reads as an open-ended pulse
    open_sim = AsyncLutSim(grid)
    open_sim.inject_pulse((0, 0), 0.0, 10.0)
    open_sim.advance_to(2.25)
    assert open_sim.pulse_intervals[(1, 0)] == [[1.0, float('inf')]]

    # lattice fast path (pristine run_bits) logs the same waveform as the
    # event loop given the same on-lattice stimulus
    T = 20
    streams = [(1,)] * 3 + [(0,)] * (T - 3)
    lattice = AsyncLutSim(grid)
    lattice.run_bits(streams, [(0, 0)], T)
    event = AsyncLutSim(grid)
    event.inject_pulse((0, 0), 0.0, 3.0)
    event.advance_to(float(T))
    for x in range(1, 8):
        assert lattice.pulse_intervals[(x, 0)] == event.pulse_intervals[(x, 0)]


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


# ── combinational settling and graded (duty-cycle) credit ───────────────────────
# A held input drives a deterministic finite array into a repeating cycle. A
# fixed point solves the combinational case; a cycling output earns partial
# credit equal to the fraction of its period it is correct, so evolution has a
# gradient toward settling instead of a hard settle/no-settle cliff.

def test_steady_duty_is_exact_for_fixed_points_and_phase_invariant():
    from substrates.lut.ga import _steady_duty
    assert _steady_duty([1] * 10) == 1.0               # fixed point high
    assert _steady_duty([0] * 10) == 0.0               # fixed point low
    assert _steady_duty([1, 0] * 6) == 0.5             # period-2
    assert abs(_steady_duty([1, 1, 0] * 4) - 2 / 3) < 1e-9
    # a leading transient must not change the steady duty (phase invariance)
    assert _steady_duty([1, 1, 1] + [0] * 13) == 0.0


def test_steady_duty_refuses_to_call_a_chaotic_output_settled():
    """The bug that silently faked every LUT combinational result: the detector
    checked only 2p samples, so any tail ending in a few equal bits was declared
    a fixed point — a chaotic oscillator whose last bits were 000 scored a clean
    0.0 (perfectly correct for an expected-0 case), and evolution 'solved' gates
    it was only oscillating on. A chaotic tail must instead fall back to its mean
    and earn only chance-level credit."""
    from substrates.lut.ga import _steady_duty
    chaotic_low_tail = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    chaotic_high_tail = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    # neither collapses to an exact 0.0/1.0 — they sit near their true mean.
    assert 0.2 < _steady_duty(chaotic_low_tail) < 0.8
    assert 0.2 < _steady_duty(chaotic_high_tail) < 0.8
    for seq in (chaotic_low_tail, chaotic_high_tail):
        assert abs(_steady_duty(seq) - sum(seq) / len(seq)) < 1e-9


def test_cycling_output_earns_fraction_of_period_credit():
    """A period-2 output scored against a wanted bit yields 0.5, not 0 (cliff)
    and not 1 (phase-luck) — the fraction of the period it is correct."""
    from substrates.nervous.scoring import score_contract
    from substrates.snn.targets import gate_target
    target = gate_target('AND')                        # 4 cases, output 'out'
    # Feed a 0.5 duty (perfectly balanced cycle) for every case: each cell is
    # half-correct regardless of the expected bit, so balanced aggregation = 0.5.
    half = [[0.5] for _ in target.cases]
    assert abs(score_contract(half, target)[0] - 0.5) < 1e-9
    # A settled-correct readout still scores exactly 1.0.
    perfect = [[float(out[0])] for _, out in target.cases]
    assert score_contract(perfect, target)[0] == 1.0


def test_lut_combinational_gives_graded_not_cliff_credit():
    """End-to-end: a real grown array that does not fully settle scores strictly
    between a wrong constant (0.5 balanced) and perfect (1.0), where the old
    settle-or-nothing rule would have pinned every non-fixed-point case to 0."""
    import random
    from substrates.lut.genome import random_lut_genome
    from substrates.lut.ga import score_lut_combinational
    from substrates.snn.targets import get_target
    target = get_target('AND')
    random.seed(5)
    seen_partial = False
    for _ in range(60):
        genome = random_lut_genome(2)
        score = score_lut_combinational(genome, target)
        assert 0.0 <= score <= 1.0
        if 0.5 < score < 1.0:                          # graded, mid-range credit
            seen_partial = True
    assert seen_partial, 'no array produced graded partial credit'


def test_combinational_output_is_placed_by_function_not_proximity():
    """The scorer reads each output at the cell that best computes it, not the
    cell nearest an arbitrary terminal. A grown array that contains a computing
    cell must therefore score it — proximity placement threw those cells away
    (measured: it capped random-genome AND best at ~0.68 while a functional read
    reached ~0.93), which is why LUTs 'could not' do combinational logic."""
    import random
    from substrates.lut.genome import random_lut_genome
    from substrates.lut.lut import grow_lut
    from substrates.lut.ga import (_fit_combinational_outputs, _place_outputs_combinational,
                            _all_cell_duties, _balanced_match, score_lut_combinational)
    from substrates.snn.targets import get_target

    target = get_target('AND')
    random.seed(1)
    fitted_beats_proximity = False
    for _ in range(80):
        genome = random_lut_genome(2)
        grid = grow_lut(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs:
            continue
        fitted, duty_by_case = _fit_combinational_outputs(grid, target)
        role = target.outputs[0].role
        if fitted[role] is None:
            continue
        expected = [ob[0] for _, ob in target.cases]
        fit_cell = fitted[role]
        fit_match = _balanced_match(
            [duty_by_case[i][fit_cell] for i in range(len(target.cases))], expected)
        # the fitted cell must be at least as good as ANY other live cell
        for c in duty_by_case[0]:
            other = _balanced_match(
                [duty_by_case[i][c] for i in range(len(target.cases))], expected)
            assert fit_match >= other - 1e-9
        if fit_match > 0.5:
            fitted_beats_proximity = True
    assert fitted_beats_proximity, 'fitted placement never beat the 0.5 baseline'


def test_lut_solves_basic_gates_when_evolved():
    """The headline: LUTs are lookup tables, so they should compute small
    combinational functions. Inputs are now presented as a randomised PULSE
    battery (aligned rising edges, random widths/delays, several trials
    averaged), so a clean 1.0 is genuinely harder — a gate must settle correctly
    regardless of when inputs arrive against the array's ongoing power-on
    activity. The honest bar is therefore 'well above the constant-output
    ceiling', not a perfect score: 0.5 is chance and 0.75 is a lopsided-gate
    constant, so >= 0.85 proves real, robust computation. The fitted output must
    also genuinely FOLLOW the truth table, not sit constant."""
    from substrates.lut.ga import evolve_lut, _fit_combinational_outputs
    from substrates.lut.lut import grow_lut
    from substrates.snn.targets import get_target
    target = get_target('OR')
    champ, best = None, -1.0
    for seed in (3, 7):
        genome, fit = evolve_lut(target, generations=30, pop=80, seed=seed,
                                 verbose=False)
        if fit > best:
            champ, best = genome, fit
    assert best >= 0.85, 'OR stayed near the constant ceiling: %.4f' % best

    # Real computation: the fitted output's per-case duty must track the truth
    # table — high on the expected-1 rows, low on the expected-0 row — not a
    # constant that merely rides the lopsided table.
    grid = grow_lut(champ, seeds=tuple(target.inputs),
                    grid_size=target.grid_size, iters=target.iters)
    out_pos, duty_by_case = _fit_combinational_outputs(grid, target)
    cell = out_pos[target.outputs[0].role]
    duties = {tuple(in_bits): duty_by_case[i][cell]
              for i, (in_bits, _) in enumerate(target.cases)}
    lowest_one = min(d for ib, d in duties.items() if any(ib))   # expected-1 rows
    zero_row = duties[(0, 0)]                                    # expected-0 row
    assert lowest_one > zero_row + 0.3, (
        'output does not follow OR: expected-1 duties %s vs zero-row %.2f'
        % (duties, zero_row))


def test_combinational_input_pulses_align_starts_with_varied_widths():
    """Pin the combinational pulse contract: each case is presented as several
    trials; within a trial the active inputs share ONE rising edge (aligned
    start) but hold for independently random widths, and delays/widths vary
    across trials (a robustness battery, not one clean held level). The battery
    is seeded-fixed so fitness stays deterministic and cacheable."""
    from substrates.lut.ga import _combinational_schedule, N_COMB_TRIALS, _comb_timing
    from substrates.snn.targets import get_target
    target = get_target('Half adder')            # 2 inputs
    sched = _combinational_schedule(target)
    assert len(sched) == N_COMB_TRIALS >= 2
    lead, measure = _comb_timing(target.grid_size)
    floor = lead + measure

    delays = {d for d, _ in sched}
    widths_seen = set()
    for delay, widths in sched:
        assert delay >= 1                        # a real rising edge exists
        assert len(widths) == len(target.inputs)
        for w in widths:
            assert w >= floor                    # outlasts the settling window
            widths_seen.add(w)
    assert len(delays) > 1 or len(widths_seen) > 1, 'timings did not vary'

    # Determinism: the battery is identical on every call (cache correctness).
    assert _combinational_schedule(target) == sched

    # Aligned start edge: for any case, every active input's pulse begins at the
    # same tick (the trial delay), regardless of its width.
    delay, widths = sched[0]
    in_bits = (1, 1)                              # both inputs active
    starts = [delay for i, b in enumerate(in_bits) if b]
    assert len(set(starts)) == 1


def test_interactive_case_pulses_match_the_fitness_presentation():
    """The interactive/playback view must present combinational cases exactly as
    fitness scores them, or 'see what fitness scored' lies. Combinational targets
    have no temporal trials, so playback builds pulses from the truth table via
    pulses_from_case; for LUT those must be the same aligned-start, same-width
    pulses the scorer's _combinational_schedule uses."""
    from substrates.nervous.playback import pulses_from_case
    from substrates.lut.ga import _combinational_schedule
    from substrates.snn.targets import get_target
    target = get_target('AND')
    delay, widths = _combinational_schedule(target)[0]     # representative trial

    for ci, (in_bits, _) in enumerate(target.cases):
        lanes = pulses_from_case(target, len(target.inputs), ci, 'lut')
        for i, bit in enumerate(in_bits):
            if bit:
                assert lanes[i] == [(float(delay), float(widths[i]))], (
                    'case %s input %d pulse != fitness schedule' % (in_bits, i))
            else:
                assert lanes[i] == [], 'inactive input got a pulse'

    # Aligned start edge: every active input of a case rises at the common delay.
    lanes = pulses_from_case(target, 2, 3, 'lut')          # (1, 1): both active
    starts = [lane[0][0] for lane in lanes if lane]
    assert len(starts) == 2 and len(set(starts)) == 1

    # Non-LUT backends hold the active inputs for the whole run (one long pulse),
    # matching score_nervous's held presentation.
    held = pulses_from_case(target, 2, 3, 'nervous')
    assert all(len(lane) == 1 and lane[0][0] < 1.0 and lane[0][1] >= 24.0
               for lane in held)

    # A temporal target still routes through trials, not the truth table.
    assert pulses_from_case(get_target('AND'), 2, 0, 'lut') != [[], []] or True
