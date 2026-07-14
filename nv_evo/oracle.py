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

from .targets import (TemporalTarget, Trial, OutputTerminal, describe_target,
                      EVENT_SCORING, PERSISTENCE_SCORING)


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


def make_gated_oscillator():
    """Run/stop memory: input A (START) injects a pulse into a period-2 loop so
    the output oscillates; input B (STOP) is an inhibitory input that drains it.
    This is the paper's core memory image made input-driven — "a pulse circulates
    a loop of buffers until stopped by an inhibitory input" (§3). Period 2 is even,
    so the bipartite honeycomb reaches it on the cheap one-pulse route; the LUT
    recurrent CA holds the same two-state cycle. START fixes the phase, so the
    ordinary latency-invariant F1 scores it — no special mode needed. STOP wins a
    tie so a simultaneous START/STOP leaves it drained, matching an inhibitory
    veto overriding excitation on the same tick."""
    def f(inb, st):
        running, phase = st if st is not None else (False, 0)
        if inb[0]:                     # START: kick the loop
            running, phase = True, 0
        if inb[1]:                     # STOP: inhibitory drain (dominant)
            running = False
        out = 1 if (running and phase == 0) else 0
        if running:
            phase ^= 1
        return (out,), (running, phase)
    return f


def make_resettable_toggle():
    """T flip-flop with an asynchronous clear: input A flips the stored bit, input
    B forces it to 0. A toggle loop guarded by an inhibitory reset line — memory
    plus a veto, both primitives the substrate already has. B wins a tie so a
    simultaneous flip+clear clears."""
    def f(inb, st):
        q = st or 0
        if inb[0]:
            q ^= 1
        if inb[1]:
            q = 0
        return (q,), q
    return f


def make_pulse_doubler():
    """Pulse-width doubler: an input pulse held for x ticks produces an output
    pulse held for 2x ticks (starting with the input). The circuit must MEASURE
    the input's duration, not just respond to its edge — a fixed delay-line
    cheat (out = in OR delay_k(in), width x+k) only doubles the single width
    x = k, so the trial bank mixes several widths to force real measurement.

    As a state machine: while the input is high, output high and bank one tick
    of 'debt'; after it falls, keep the output high until the debt is repaid —
    x during + x after = 2x total, contiguous. A pulse arriving during the tail
    merges (output stays high, debt accumulates), conserving total output = 2 x
    total input ticks."""
    def f(inb, st):
        debt = st or 0
        if inb[0]:
            return (1,), debt + 1              # high: emit + bank one tail tick
        if debt > 0:
            return (1,), debt - 1              # tail: emit while repaying
        return (0,), 0
    return f


def orc_period_doubler(inb, st):
    """Divide-by-2 over edges: emit on the 1st, 3rd, 5th, ... input pulse, so a
    periodic input train of period p yields a periodic output of period 2p —
    the output "doubles the period" (halves the rate). Edge-native: the input
    information is inter-edge INTERVALS, exactly what an asynchronous substrate
    computes with (a held level would be one edge and carry no period at all)."""
    parity = st or 0
    if inb[0]:
        return (1 if parity == 0 else 0,), parity ^ 1
    return (0,), parity


def make_c_element():
    """Muller C-element in transition signalling — a 2-input rendezvous / join.

    Emit an output event only once BOTH inputs have produced an edge, in EITHER
    order, then rearm and wait for the next pair. A lone edge on one input must
    NOT emit; the element has to REMEMBER that the first input arrived while it
    waits for the second — so this is a stored-state element, the asynchronous
    handshake keystone (the C-element is what joins the two rails of a
    micropipeline). The stored state is which inputs have arrived this round.

    (The textbook level-mode C-element — output high while both inputs are high,
    holding on disagreement — is not a natural fit for an edge-coincidence
    substrate that has no level-AND; transition signalling is the faithful
    encoding and is the same device in event form.)"""
    def f(inb, st):
        seen_a, seen_b = st if st is not None else (False, False)
        seen_a = seen_a or bool(inb[0])
        seen_b = seen_b or bool(inb[1])
        if seen_a and seen_b:                  # rendezvous complete
            return (1,), (False, False)        # emit one event, then rearm
        return (0,), (seen_a, seen_b)
    return f


def make_period_stepper(base=2, step=2, max_period=6):
    """Build a bounded, re-phased cadence controller.

    The first command starts a period-``base`` oscillator.  Every later command
    advances it by ``step`` (up to ``max_period``) and restarts phase.  This is
    a finite, physically meaningful version of period stepping: it asks for a
    slower sustained cadence without pretending an in-flight pulse instantly
    had the new cadence all along.
    """
    def f(inb, st):
        period, phase, started = st if st is not None else (base, 0, False)
        if inb[0]:
            if not started:
                started = True
                period = base
            else:
                period = min(max_period, period + step)
            phase = 0
        out = 1 if started and phase == 0 else 0
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
            # The hardware sees a leading edge, not a logical sample.  Leave at
            # least one low tick between events; consecutive 1 samples would be
            # one extended pulse and must not be labelled as two stimuli.
            t += max(2, min_gap + rng.randint(0, jitter))
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
                  out_pos=(2, 2), grid_size=5, description='',
                  score_mode='trace'):
    rng = random.Random(seed)
    out = OutputTerminal(output_role, out_pos)
    n_in = len(inputs)
    trials = []
    for _ in range(n_trials):
        streams = sample_streams(rng, T, n_in, min_gap=min_gap, align_prob=align_prob)
        exp = label_trace(oracle, streams, T, latency)
        events = ([float(t) for t, value in enumerate(exp) if value == 1]
                  if score_mode == 'events' else [])
        trials.append(Trial(streams, {output_role: exp},
                            {output_role: events} if score_mode == 'events' else {}))
    return TemporalTarget(name, list(inputs), [out], T, trials,
                          grid_size=grid_size, iters=30, description=description,
                          score_mode=score_mode, latency=latency)


def holdout_score(genome, spec, backend='nervous', seed=999, fitted=None):
    """Score fresh schedules with training readout and alignment frozen.
    The spec's default target supplies training schedules; ``seed`` supplies
    validation schedules. Pass ``fitted`` to reuse one training fit."""
    from .evaluation import fit_readout, score_frozen
    if fitted is None:
        fitted = fit_readout(genome, spec(), backend=backend)
    if fitted is None:
        return 0.0
    if fitted.backend != backend:
        raise ValueError('fitted readout backend does not match holdout backend')
    return score_frozen(genome, spec(seed=seed), fitted)


# ── preset oracle targets (input-driven relations) ──────────────────────────────

def sr_latch_oracle(seed=20260702):
    target = oracle_target('SR latch (oracle)', orc_sr_latch, [(0, 1), (0, 3)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=2, min_gap=6,
                         description=describe_target(
        'Input A sets one stored bit; input B resets it; otherwise Q retains its '
        'previous state.', PERSISTENCE_SCORING,
        'Twelve seeded random schedules plus explicit never-set and '
        'Set→Reset→Set tests exercise storage, clearing, and reloading.'))
    # Persistence guardrails: silence without Set, and a Set->Reset->Set trial
    # that requires the same circuit to store, clear, and store again.  A
    # one-shot response to Set or an unconditional ring fails these contrasts.
    silent = [(0, 0)] * target.T
    cycle = [(0, 0)] * target.T
    cycle[3] = (1, 0)
    cycle[11] = (0, 1)
    cycle[18] = (1, 0)
    target.trials.extend([
        Trial(silent, {'Q': label_trace(orc_sr_latch, silent, target.T, 2)}),
        Trial(cycle, {'Q': label_trace(orc_sr_latch, cycle, target.T, 2)}),
    ])
    return target


def toggle_oracle(seed=20260702):
    return oracle_target('Toggle (oracle)', orc_toggle, [(0, 2)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=2, min_gap=5,
                         description=describe_target(
        'Each input edge flips the stored output state.', PERSISTENCE_SCORING,
        'Twelve seeded random pulse trains vary phase and spacing.'))


def echo_oracle(seed=20260702, delay=3):
    return oracle_target('Echo (oracle)', make_echo(delay), [(0, 2)], 'Q',
                         T=22, n_trials=10, seed=seed, latency=delay, min_gap=3,
                         score_mode='events',
                         description=describe_target(
        'Reproduce the input-edge train while preserving its intervals.',
        EVENT_SCORING,
        'Ten seeded random schedules use a nominal %d-tick reference delay; '
        'absolute evolved latency is free.' % delay))


def coincidence_oracle(seed=20260702):
    return oracle_target('Coincidence (oracle)', orc_coincidence, [(0, 1), (0, 3)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=4, align_prob=0.5,
                         score_mode='events',
                         description=describe_target(
        'Emit Q only when A and B arrive together.', EVENT_SCORING,
        'Twelve seeded schedules contain approximately equal coincident and '
        'offset input events.'))


def one_shot_oracle(seed=20260702, width=5):
    # width MUST exceed 3: the persistence scorer's +/-1 ring tolerance lets a
    # SINGLE output pulse cover a 3-tick active window (a pulse at the centre hits
    # all three expected ticks), so a plain delay-line/echo scored a perfect
    # one-shot. A 5-tick window cannot be covered by one pulse, so it forces a
    # genuine self-terminating BURST (a ~10101 ring that stops) — the actual
    # monostable behaviour. (Verified: width 3 solves with single pulses; width 5
    # solves with sustained bursts.)
    return oracle_target('One-shot (oracle)', make_one_shot(width), [(0, 2)], 'Q',
                         T=28, n_trials=10, seed=seed, latency=2, min_gap=width + 4,
                         description=describe_target(
        'Each input edge starts a %d-tick active interval that self-terminates.'
        % width, PERSISTENCE_SCORING,
        'Ten seeded random schedules space triggers far enough to verify '
        'termination and re-arming.'))


def pair_oracle(seed=20260702, gap=2):
    return oracle_target('Pair detector (oracle)', make_pair(gap), [(0, 2)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=1,
                         score_mode='events',
                         description=describe_target(
        'Emit Q when two input edges are separated by exactly %d ticks.' % gap,
        EVENT_SCORING,
        'Twelve seeded random trains mix valid pairs, wrong gaps, and isolated '
        'events.'))


def period_stepper_oracle(seed=20260702, base=2):
    """Finite period-stepper trials with long, independently scored dwells.

    The stepper is bounded (2 -> 4 -> 6) and re-phased at each command: it asks
    the circuit to make its cadence slower in steps, rather than to reproduce
    one arbitrary transient phase from an unbounded abstract counter.
    """
    T, rng = 108, random.Random(seed)
    oracle = make_period_stepper(base=base, step=2, max_period=6)
    # One start-only trial establishes the base cadence.  Single-step trials add
    # one command (2 -> 4); double-step trials add two (2 -> 4 -> 6).  Every
    # dwell gets >= ~26 ticks so several complete cycles stay visible at each
    # cadence, including the slow period-6 tail.
    pulse_sets = [[rng.randint(3, 8)]]
    for _ in range(2):
        start = rng.randint(3, 8)
        pulse_sets.append([start, start + rng.randint(26, 31)])
    for _ in range(2):
        start = rng.randint(3, 8)
        g1 = rng.randint(26, 31)
        g2 = rng.randint(32, 37)
        pulse_sets.append([start, start + g1, start + g1 + g2])
    trials = []
    for commands in pulse_sets:
        streams = [(1 if t in commands else 0,) for t in range(T)]
        trials.append(Trial(streams, {'Q': label_trace(oracle, streams, T, 2)}))
    return TemporalTarget(
        'Period stepper (oracle)', [(0, 2)], [OutputTerminal('Q', (2, 2))],
        T, trials, grid_size=5, iters=30,
        description=describe_target(
            'The first command starts period 2; later commands step the '
            'sustained cadence to periods 4 and 6.',
            'Score regular coverage within each command-delimited dwell and '
            'require every later period to be strictly longer; phase is free.',
            'Five long schedules cover start-only, one-step, and two-step '
            'operation with enough time to observe each settled cadence.'),
        score_mode='period_stepper', stepper_min_period=2,
        # A command can meet a circulating pulse at any phase.  Its short
        # switchover is intentionally unscored; only the settled cadence matters.
        stepper_max_period=6, stepper_settle=6, stepper_min_events=4,
        stepper_max_delay=8)


def gated_oscillator_oracle(seed=20260702):
    return oracle_target('Gated oscillator (oracle)', make_gated_oscillator(),
                         [(0, 1), (0, 3)], 'Q',
                         T=32, n_trials=12, seed=seed, latency=2, min_gap=8,
                         description=describe_target(
        'Input A starts a period-2 output cadence; input B stops it. Q is quiet '
        'outside the commanded run interval.', PERSISTENCE_SCORING,
        'Twelve seeded random A/B schedules verify start, sustained running, '
        'stop, and later restart.'))


def resettable_toggle_oracle(seed=20260702):
    return oracle_target('Resettable toggle (oracle)', make_resettable_toggle(),
                         [(0, 1), (0, 3)], 'Q',
                         T=26, n_trials=12, seed=seed, latency=2, min_gap=5,
                         description=describe_target(
        'Input A flips the stored bit; input B clears it to 0 and dominates a '
        'simultaneous A+B event.', PERSISTENCE_SCORING,
        'Twelve seeded random schedules exercise toggling, clearing, and '
        'post-clear recovery.'))


def pulse_doubler_oracle(seed=20260702, widths=(1, 2, 3)):
    """Pulse-width doubler trials: held input runs of MIXED widths x, each
    expecting a contiguous 2x-tick output hold. Mixing widths in one bank is the
    anti-cheat: a fixed-delay response (width x+k) or a fixed-length one-shot
    fits at most one width, so only genuine width measurement scores across all
    trials. Runs are spaced so each 2x tail completes plus quiet margin before
    the next pulse; a silent trial guards the quiet side."""
    T, latency, rng = 40, 1, random.Random(seed)
    oracle = make_pulse_doubler()
    trials = []

    def add(runs):
        ons = set()
        for (start, width) in runs:
            for k in range(width):
                ons.add(start + k)
        streams = [(1 if t in ons else 0,) for t in range(T)]
        trials.append(Trial(streams, {'Q': label_trace(oracle, streams, T, latency)}))

    for w in widths:                              # clean single-pulse measurements
        for _ in range(2):
            add([(rng.randint(2, 8), w)])
    for _ in range(3):                            # two pulses, mixed widths
        w1, w2 = rng.choice(widths), rng.choice(widths)
        s1 = rng.randint(2, 6)
        s2 = s1 + 3 * w1 + rng.randint(3, 6)      # after w1's 2x tail + quiet
        if s2 + 3 * w2 + latency + 2 < T:
            add([(s1, w1), (s2, w2)])
        else:
            add([(s1, w1)])
    add([])                                       # never-pulsed: Q must stay quiet
    return TemporalTarget(
        'Pulse doubler (oracle)', [(0, 2)], [OutputTerminal('Q', (2, 2))],
        T, trials, grid_size=5, iters=30,
        description=describe_target(
            'Stretch each input pulse to twice its width: a pulse held x ticks '
            'yields one contiguous 2x-tick output hold starting with it. The '
            'bank mixes widths x in %s, so a fixed delay or fixed-length '
            'one-shot cannot fit them all - the circuit must measure x.'
            % (tuple(widths),), PERSISTENCE_SCORING,
            'Seeded single- and double-pulse schedules across every width plus '
            'a silent guard trial.'),
        score_mode='trace', latency=latency)


def period_doubler_oracle(seed=20260702, periods=(2, 3, 4)):
    """Period-doubler trials: periodic input trains of MIXED periods p, each
    expecting an output train of period 2p (every 2nd input edge). Mixing
    periods is the anti-cheat — a free-running oscillator or any fixed-cadence
    responder fits at most one input rate — and the silent guard trial kills
    oscillators outright (no input => no output). Phases vary per trial so a
    phase-locked fake can't memorise tick positions.

    Period 1 is deliberately EXCLUDED: a pulse every tick wired-OR merges into
    one held level (one edge), physically indistinguishable from constant
    input on both substrates — it carries no period."""
    T, latency, rng = 30, 1, random.Random(seed)
    trials = []

    def add(ticks):
        on = set(ticks)
        streams = [(1 if t in on else 0,) for t in range(T)]
        exp = label_trace(orc_period_doubler, streams, T, latency)
        events = [float(t) for t, v in enumerate(exp) if v == 1]
        trials.append(Trial(streams, {'Q': exp}, {'Q': events}))

    for p in periods:
        for _ in range(3):                        # 3 phase-shifted trials per period
            phase = rng.randint(1, p + 2)
            add([phase + k * p for k in range((T - 2 - phase) // p + 1)])
    add([])                                       # silent guard: no input, no output
    return TemporalTarget(
        'Period doubler (oracle)', [(0, 2)], [OutputTerminal('Q', (2, 2))],
        T, trials, grid_size=5, iters=30,
        description=describe_target(
            'Double the period of the input train: for a periodic input of '
            'period p, emit every 2nd input edge so the output is periodic at '
            '2p. The bank mixes periods p in %s at varying phases, so a fixed '
            'free-running cadence cannot fit them all, and a silent trial '
            'forbids output without input.' % (tuple(periods),), EVENT_SCORING,
            'Three phase-varied periodic schedules per period plus a silent '
            'guard trial.'),
        score_mode='events', latency=latency)


def c_element_oracle(seed=20260702):
    return oracle_target('C-element (oracle)', make_c_element(), [(0, 1), (0, 3)], 'Q',
                         T=28, n_trials=12, seed=seed, latency=2, min_gap=5,
                         align_prob=0.3, score_mode='events',
                         description=describe_target(
        'Transition-signalling Muller C-element (2-input rendezvous/join): emit Q '
        'once BOTH inputs have produced an edge, in either order, then rearm. '
        'Remembering the first arrival while waiting for the second is the stored '
        'state — the asynchronous handshake keystone.', EVENT_SCORING,
        'Twelve seeded random A/B schedules interleave the two inputs in both '
        'orders, including lone edges that must not emit until the partner arrives.'))


ORACLE_SPECS = {
    'SR latch (oracle)':          sr_latch_oracle,
    'C-element (oracle)':         c_element_oracle,
    'Toggle (oracle)':            toggle_oracle,
    'Echo (oracle)':              echo_oracle,
    'Coincidence (oracle)':       coincidence_oracle,
    'One-shot (oracle)':          one_shot_oracle,
    'Pulse doubler (oracle)':     pulse_doubler_oracle,
    'Period doubler (oracle)':    period_doubler_oracle,
    'Pair detector (oracle)':     pair_oracle,
    'Period stepper (oracle)':    period_stepper_oracle,
    'Gated oscillator (oracle)':  gated_oscillator_oracle,
    'Resettable toggle (oracle)': resettable_toggle_oracle,
}

ORACLE_TARGETS = {name: spec() for name, spec in ORACLE_SPECS.items()}
