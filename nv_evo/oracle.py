"""
nv_evo/oracle.py — targets defined by a reference model, not hand-picked traces.

Hand-writing "input pulses at ticks 3,5 -> output at tick 8" bakes OUR timing
into the goal, and a circuit can plateau not because it fails the function but
because our chosen timing fights its internal phase (an evolved SR latch with a
period-2 loop resets fine on even ticks and misses odd ones — the timing was
adversarial, not the latch broken).

An oracle target instead specifies the goal as:
    * an ORACLE — a tiny reference state machine  oracle(in_bits, state) ->
      (out_bits, new_state)  that defines the intended input->output relation;
    * a STIMULUS GENERATOR that samples random input schedules.
Many schedules are sampled; the oracle labels each; the circuit is scored on
reproducing the RELATION across all of them (with the usual response-latency
grace + phase-tolerant holds). Solving means implementing the function, not
memorising a timing — and `holdout_score` re-samples fresh schedules to certify
it generalises.

Autonomous behaviours (oscillator, free-running pattern) are not input-driven
relations, so they keep their hand-built targets; everything input-driven moves
here.
"""
from __future__ import annotations
import random

from .targets import TemporalTarget, Trial, OutputTerminal, SETTLE


# ── reference oracles (in_bits, state) -> (out_bits, new_state) ──────────────────

def orc_sr_latch(inb, st):
    q = st or 0
    if inb[0]:      q = 1          # Set
    elif inb[1]:    q = 0          # Reset
    return (q,), q


def orc_toggle(inb, st):
    q = st or 0
    if inb[0]:
        q ^= 1
    return (q,), q


def orc_coincidence(inb, st):
    return (1 if (inb[0] and inb[1]) else 0,), None


def make_echo(delay):
    def f(inb, st):
        h = st if st is not None else (0,) * delay
        out = h[0]
        return (out,), h[1:] + (inb[0],)
    return f


def make_one_shot(width):
    def f(inb, st):
        rem = st or 0
        if inb[0] and rem == 0:
            rem = width
        out = 1 if rem > 0 else 0
        return (out,), max(0, rem - 1)
    return f


def make_pair(gap):
    def f(inb, st):
        h = (st if st is not None else (0,) * (gap + 1))[1:] + (inb[0],)
        out = 1 if (h[-1] and h[-1 - gap]) else 0
        return (out,), h
    return f


def make_period_stepper(base=2):
    """A controllable oscillator. The FIRST input pulse kicks it into a
    period-`base` oscillation (a 1 every `base` ticks — the natural circulating
    pulse); every SUBSEQUENT pulse lengthens the period by 1 (slower and slower).
    Anchoring the oscillation to the first kick fixes the absolute phase, so the
    reference and a real (kick-started) circuit stay phase-aligned."""
    def f(inb, st):
        period, phase, started = st if st is not None else (base, 0, False)
        if inb[0]:
            if not started:
                started, phase = True, 0          # first pulse: start oscillating
            else:
                period += 1                        # each later pulse: period + 1
        out = 1 if (started and phase == 0) else 0
        if started:
            phase = (phase + 1) % period
        return (out,), (period, phase, started)
    return f


# ── stimulus generation ─────────────────────────────────────────────────────────

def sample_streams(rng, T, n_inputs, min_gap=4, jitter=4, align_prob=0.0):
    """A list[T] of random input-bit tuples. Each input gets a sparse pulse
    train (gap >= min_gap, random jitter). `align_prob` occasionally fires all
    inputs on the SAME tick — needed so coincidence/pair targets see positive
    cases instead of almost-always-0."""
    ons = [set() for _ in range(n_inputs)]
    for i in range(n_inputs):
        t = rng.randint(1, min_gap)
        while t < T:
            ons[i].add(t)
            t += min_gap + rng.randint(0, jitter)
    if align_prob:
        for i in range(1, n_inputs):
            aligned = set()
            for t in ons[0]:
                aligned.add(t if rng.random() < align_prob else
                            min(T - 1, t + rng.randint(1, 3)))
            ons[i] = aligned
    return [tuple(1 if t in ons[i] else 0 for i in range(n_inputs))
            for t in range(T)]


def label_trace(oracle, streams, T, latency):
    """Run the oracle over a stream and shift its output right by `latency` — the
    circuit's (deterministic, path-length) propagation delay — so the expected
    trace lines up with when a correct circuit actually responds. The first
    `latency` ticks are startup grace (unscored). No transition masking: that
    would swallow short pulses; the scorer's own phase-tolerant hold rule gives
    the ±1 slack that level-holds need, and pulse relations have a fixed delay
    the circuit matches exactly."""
    raw, st = [], None
    for t in range(T):
        ob, st = oracle(streams[t], st)
        raw.append(ob[0])
    return [None] * latency + raw[:max(0, T - latency)]


# ── target builder ──────────────────────────────────────────────────────────────

def oracle_target(name, oracle, inputs, output_role, T=24, n_trials=12,
                  seed=20260702, latency=2, min_gap=5, align_prob=0.0,
                  out_pos=(2, 2), grid_size=5, description=''):
    rng = random.Random(seed)
    out = OutputTerminal(output_role, out_pos)
    n_in = len(inputs)
    trials = []
    for _ in range(n_trials):
        streams = sample_streams(rng, T, n_in, min_gap=min_gap, align_prob=align_prob)
        trials.append(Trial(streams, {output_role: label_trace(oracle, streams, T, latency)}))
    return TemporalTarget(name, list(inputs), [out], T, trials,
                          grid_size=grid_size, iters=30, description=description)


def holdout_score(genome, spec, backend='nervous', seed=999):
    """Re-sample fresh schedules from the same spec (a zero-arg builder) with a
    different seed and score — certifies the circuit generalises rather than
    fitting the training schedules. Returns behavioural score in [0,1]."""
    target = spec(seed=seed)
    if backend == 'lut':
        from lut_evo import score_lut_temporal
        return score_lut_temporal(genome, target)
    from .temporal import score_temporal
    return score_temporal(genome, target)


# ── preset oracle targets (input-driven relations) ──────────────────────────────

def sr_latch_oracle(seed=20260702):
    return oracle_target('SR latch (oracle)', orc_sr_latch, [(0, 1), (0, 3)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=2, min_gap=6,
                         description=(
        'SR latch specified by a reference bit of memory (q = set ? 1 :\n'
        'reset ? 0 : q), tested on 12 RANDOM Set/Reset schedules — not a few\n'
        'hand-picked timings. The circuit must implement the relation for any\n'
        'timing, so it cannot pass by phase-luck; held-out schedules certify it.'))


def toggle_oracle(seed=20260702):
    return oracle_target('Toggle (oracle)', orc_toggle, [(0, 2)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=2, min_gap=5,
                         description='T flip-flop vs a reference (each pulse flips q), on random pulse trains.')


def echo_oracle(seed=20260702, delay=3):
    return oracle_target('Echo (oracle)', make_echo(delay), [(0, 2)], 'Q',
                         T=22, n_trials=10, seed=seed, latency=delay, min_gap=3,
                         description='Delay line vs reference (out = in delayed %d), random pulse trains.' % delay)


def coincidence_oracle(seed=20260702):
    return oracle_target('Coincidence (oracle)', orc_coincidence, [(0, 1), (0, 3)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=4, align_prob=0.5,
                         description='out = A AND B on the same tick; random schedules, ~half the pulses aligned.')


def one_shot_oracle(seed=20260702, width=3):
    return oracle_target('One-shot (oracle)', make_one_shot(width), [(0, 2)], 'Q',
                         T=24, n_trials=10, seed=seed, latency=2, min_gap=width + 4,
                         description='Monostable vs reference (%d-tick burst per pulse, self-terminating), random pulses.' % width)


def pair_oracle(seed=20260702, gap=2):
    return oracle_target('Pair detector (oracle)', make_pair(gap), [(0, 2)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=1,
                         description='Fires iff two input pulses arrive %d ticks apart; random pulse trains.' % gap)


def period_stepper_oracle(seed=20260702, base=2):
    return oracle_target('Period stepper (oracle)', make_period_stepper(base), [(0, 2)], 'Q',
                         T=30, n_trials=12, seed=seed, latency=2, min_gap=7,
                         description=(
        'A controllable oscillator: the first input pulse starts the output\n'
        'oscillating with period %d (a 1 every %d ticks), and EVERY further pulse\n'
        'increases the period by 1 (the oscillation slows: %d, %d, %d, ...). The\n'
        'output must therefore be a counter driving a variable-length loop —\n'
        'genuinely hard, since the period grows unbounded with more inputs; the\n'
        'phase is anchored to the first pulse so it stays comparable.'
        % (base, base, base, base + 1, base + 2)))


ORACLE_SPECS = {
    'SR latch (oracle)':        sr_latch_oracle,
    'Toggle (oracle)':          toggle_oracle,
    'Echo (oracle)':            echo_oracle,
    'Coincidence (oracle)':     coincidence_oracle,
    'One-shot (oracle)':        one_shot_oracle,
    'Pair detector (oracle)':   pair_oracle,
    'Period stepper (oracle)':  period_stepper_oracle,
}

ORACLE_TARGETS = {name: spec() for name, spec in ORACLE_SPECS.items()}
