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
import math
import random

from .targets import (TemporalTarget, Trial, OutputTerminal,
                      describe_target)
from .contracts import (cadence_step_contract, event_contract,
                        interval_contract, state_contract)


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


def orc_period_tripler(inb, st):
    """Divide the input edge rate by three: emit on edge 1, 4, 7, ... ."""
    phase = st or 0
    if inb[0]:
        return (1 if phase == 0 else 0,), (phase + 1) % 3
    return (0,), phase


def make_a_parity_query():
    """Count A edges modulo two; B queries the retained parity.

    B does not clear the count, so repeated B queries return the same answer
    until another A edge changes parity. If A and B coincide, A is counted
    before that B query.
    """
    def f(inb, st):
        odd = bool(st)
        if inb[0]:
            odd = not odd
        return (1 if inb[1] and odd else 0,), odd
    return f


def make_a_mod3_query():
    """B emits when at least three A edges have occurred and count(A) % 3 = 0."""
    def f(inb, st):
        count = st or 0
        if inb[0]:
            count += 1
        return (1 if inb[1] and count > 0 and count % 3 == 0 else 0,), count
    return f


def make_a_batch_parity_query():
    """B queries whether the A count since the previous B is odd, then clears."""
    def f(inb, st):
        odd = bool(st)
        if inb[0]:
            odd = not odd
        output = 1 if inb[1] and odd else 0
        return (output,), False if inb[1] else odd
    return f


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


def make_refractory_filter(dead_time=3):
    """Pass one input event, then suppress events during a finite dead time.

    This is an event-rate limiter rather than a one-shot: the accepted output is
    one point event, not a held interval.  An event accepted at tick ``t`` blocks
    ticks ``t+1`` through ``t+dead_time`` and the filter can fire again at
    ``t+dead_time+1``.
    """
    dead_time = int(dead_time)
    if dead_time < 1:
        raise ValueError('dead_time must be positive')

    def f(inb, st):
        cooldown = int(st or 0)
        if cooldown > 0:
            return (0,), cooldown - 1
        if inb[0]:
            return (1,), dead_time
        return (0,), 0
    return f


def make_a_first_rendezvous():
    """Order-sensitive two-input rendezvous with repeated re-arming rounds.

    The first arrival is remembered until the other input completes the round.
    Emit one event only when A arrived first; B-first and simultaneous-tie rounds
    complete silently.  Same-channel repeats do not change the remembered winner.
    Unlike the fixed-gap A->B target, any positive gap is accepted.
    """
    def f(inb, st):
        first = int(st or 0)             # 0 idle, 1 A-first, 2 B-first
        a, b = bool(inb[0]), bool(inb[1])
        if first == 0:
            if a and b:                  # simultaneous tie: consume the round
                return (0,), 0
            if a:
                return (0,), 1
            if b:
                return (0,), 2
            return (0,), 0
        if first == 1 and b:
            return (1,), 0               # A won; B closes the round
        if first == 2 and a:
            return (0,), 0               # B won; A closes silently
        return (0,), first               # wait; ignore same-channel repeats
    return f


def make_collision_serializer(spacing=2):
    """Serialize event multiplicity from two inputs onto one output wire.

    Every A or B event contributes one token.  At most one output event is
    emitted at a time, with ``spacing`` ticks between emissions.  Therefore an
    isolated input produces one event while simultaneous A+B produces two
    separated events instead of losing one in a wired-OR collision.
    """
    spacing = int(spacing)
    if spacing < 1:
        raise ValueError('spacing must be positive')

    def f(inb, st):
        cooldown, queued = st if st is not None else (0, 0)
        cooldown, queued = int(cooldown), int(queued)
        queued += int(bool(inb[0])) + int(bool(inb[1]))
        if cooldown > 0:
            cooldown -= 1
        if cooldown == 0 and queued > 0:
            return (1,), (spacing, queued - 1)
        return (0,), (cooldown, queued)
    return f


def make_watchdog(timeout=5):
    """Emit once when an armed heartbeat input stays quiet for ``timeout`` ticks.

    The first heartbeat arms the watchdog and every later heartbeat restarts its
    timer.  A heartbeat exactly on the deadline wins and prevents an alarm.  Once
    an alarm fires the watchdog disarms; a later heartbeat arms a fresh round.
    Never-armed silence therefore remains silent.
    """
    timeout = int(timeout)
    if timeout < 1:
        raise ValueError('timeout must be positive')

    def f(inb, st):
        if inb[0]:                       # heartbeat wins on the deadline tick
            return (0,), 0
        if st is None:                   # never armed / already timed out
            return (0,), None
        quiet = int(st) + 1
        if quiet >= timeout:
            return (1,), None            # one alarm, then disarm
        return (0,), quiet
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

def sample_streams(rng, T, n_inputs, min_gap=4, jitter=4, align_prob=0.0,
                   global_gap=False):
    """A list[T] of random input-bit tuples. Each input gets a sparse pulse
    train (gap >= min_gap, random jitter). `align_prob` occasionally fires all
    inputs on the SAME tick — needed so coincidence/pair targets see positive
    cases instead of almost-always-0."""
    ons = [set() for _ in range(n_inputs)]
    if global_gap:
        # Memory outputs need an observable interval after each command. Build
        # one globally spaced event train and distribute it across the inputs;
        # independent per-input trains can interleave only a second apart even
        # when each individual lane has a large min_gap.
        order = list(range(n_inputs))
        rng.shuffle(order)
        event_index = 0
        t = rng.randint(1, min_gap)
        while t < T:
            ons[order[event_index % n_inputs]].add(t)
            event_index += 1
            if event_index % n_inputs == 0:
                rng.shuffle(order)
            t += max(2, min_gap + rng.randint(0, jitter))
        return [tuple(1 if t in ons[i] else 0 for i in range(n_inputs))
                for t in range(T)]
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
                  contract=None, global_gap=False):
    contract = contract or state_contract()
    event_semantics = any(c.relation == 'event_correspondence'
                          for c in contract.constraints)
    rng = random.Random(seed)
    out = OutputTerminal(output_role, out_pos)
    n_in = len(inputs)
    trials = []
    for _ in range(n_trials):
        streams = sample_streams(
            rng, T, n_in, min_gap=min_gap, align_prob=align_prob,
            global_gap=global_gap)
        exp = label_trace(oracle, streams, T, latency)
        events = ([float(t) for t, value in enumerate(exp) if value == 1]
                  if event_semantics else [])
        trials.append(Trial(streams, {output_role: exp},
                            {output_role: events} if event_semantics else {}))
    return TemporalTarget(name, list(inputs), [out], T, trials,
                          grid_size=grid_size, iters=30, description=description,
                          contract=contract,
                          latency=latency)


def _event_bank_target(name, oracle, inputs, pulse_banks, T, latency,
                       description, out_pos=(2, 2), grid_size=5):
    """Build a single-output event target from seeded, explicit pulse banks.

    ``pulse_banks`` is a sequence of ``{input_index: iterable[tick]}`` mappings.
    Keeping the schedule generator separate from the state machine guarantees
    positive, negative, boundary, re-arm, and silence cases while still allowing
    ``spec(seed=...)`` to produce genuinely fresh held-out timings.
    """
    trials = []
    n_inputs = len(inputs)
    for pulses in pulse_banks:
        ons = {index: set(map(int, ticks)) for index, ticks in pulses.items()}
        streams = [tuple(1 if tick in ons.get(index, ()) else 0
                         for index in range(n_inputs))
                   for tick in range(T)]
        exp = label_trace(oracle, streams, T, latency)
        events = [float(tick) for tick, value in enumerate(exp) if value == 1]
        trials.append(Trial(streams, {'Q': exp}, {'Q': events}))
    return TemporalTarget(
        name, list(inputs), [OutputTerminal('Q', out_pos)], T, trials,
        grid_size=grid_size, iters=30, description=description,
        contract=event_contract(), latency=latency)


# A widened pulse must FALL strictly before the next event on the same lane
# rises: at zero clearance the models disagree (the legacy engine emits a new
# edge at a touch, width-preserving transport unions touching drives into one
# waveform), and any overlap merges two labelled stimulus events into ONE
# physical edge — unreachable expected output in every model.
_WIDTH_CLEARANCE = 0.5


def _mix_event_widths(target, rng):
    """Give every event-bank input varied physical widths without moving edges.

    Widths are clamped so a widened pulse can never reach the next event on its
    lane (see _WIDTH_CLEARANCE). Today's banks keep same-lane spacing >= 3 so
    the clamp is a no-op; it exists so a future tighter bank cannot silently
    erase input edges the reference oracle already counted.
    """
    widths = (0.5, 0.75, 1.0, 1.25, 1.75, 2.25)
    offset = rng.randrange(len(widths))
    serial = 0
    for trial in target.trials:
        lanes = []
        for input_index in range(target.n_inputs):
            ticks = [tick for tick, row in enumerate(trial.streams)
                     if row[input_index]]
            events = []
            for position, tick in enumerate(ticks):
                width = widths[(offset + serial) % len(widths)]
                serial += 1
                if position + 1 < len(ticks):
                    gap = float(ticks[position + 1] - tick)
                    width = min(width, gap - _WIDTH_CLEARANCE)
                if width <= 0:
                    raise ValueError(
                        'event bank spacing too tight to widen pulses on lane '
                        '%d of %r' % (input_index, target.name))
                events.append((float(tick), width))
            lanes.append(events)
        trial.input_events = lanes
    return target


def _explicit_event_trial(T, n_inputs, pulses, output_events):
    """Build one dense-display/raw-event trial from integer event times."""
    ons = {index: set(map(int, ticks)) for index, ticks in pulses.items()}
    streams = [tuple(1 if tick in ons.get(index, ()) else 0
                     for index in range(n_inputs))
               for tick in range(T)]
    events = sorted(float(tick) for tick in output_events if 0 <= tick < T)
    event_set = set(events)
    expected = [1 if float(tick) in event_set else 0 for tick in range(T)]
    return Trial(streams, {'Q': expected}, {'Q': events})


def holdout_score(genome, spec, backend='nervous', seed=999, fitted=None,
                  physics_from=None):
    """Score fresh schedules with training readout and alignment frozen.
    The spec's default target supplies training schedules; ``seed`` supplies
    validation schedules. Pass ``fitted`` to reuse one training fit.

    ``physics_from`` is the training target carrying the run's physics config
    (pulse/lut); it is copied onto the freshly built spec targets so validation
    runs under the SAME node-timing model as training (else a non-default model
    would be scored under the default uniform physics — see certification.py)."""
    from .evaluation import fit_readout, score_frozen
    from .certification import carry_physics

    def build(**kwargs):
        target = spec(**kwargs)
        return target if physics_from is None else carry_physics(physics_from, target)

    if fitted is None:
        fitted = fit_readout(genome, build(), backend=backend)
    if fitted is None:
        return 0.0
    if fitted.backend != backend:
        raise ValueError('fitted readout backend does not match holdout backend')
    return score_frozen(genome, build(seed=seed), fitted)


# ── preset oracle targets (input-driven relations) ──────────────────────────────

def sr_latch_oracle(seed=20260702):
    target = oracle_target('SR latch (oracle)', orc_sr_latch, [(0, 1), (0, 3)], 'Q',
                         T=40, n_trials=12, seed=seed, latency=2, min_gap=10,
                         global_gap=True,
                         description=describe_target(
        'Input A sets one stored bit; input B resets it; otherwise Q retains its '
        'previous state.',
        'Twelve seeded random schedules plus explicit never-set and '
        'Set→Reset→Set tests exercise storage, clearing, and reloading.'))
    # Persistence guardrails: silence without Set, and a Set->Reset->Set trial
    # that requires the same circuit to store, clear, and store again.  A
    # one-shot response to Set or an unconditional ring fails these contrasts.
    silent = [(0, 0)] * target.T
    cycle = [(0, 0)] * target.T
    cycle[3] = (1, 0)
    cycle[16] = (0, 1)
    cycle[29] = (1, 0)
    target.trials.extend([
        Trial(silent, {'Q': label_trace(orc_sr_latch, silent, target.T, 2)}),
        Trial(cycle, {'Q': label_trace(orc_sr_latch, cycle, target.T, 2)}),
    ])
    return target


def toggle_oracle(seed=20260702):
    return oracle_target('Toggle (oracle)', orc_toggle, [(0, 2)], 'Q',
                         T=40, n_trials=12, seed=seed, latency=2, min_gap=10,
                         global_gap=True,
                         description=describe_target(
        'Each input edge flips the stored output state.',
        'Twelve seeded random pulse trains vary phase and spacing.'))


def echo_oracle(seed=20260702, delay=3):
    return oracle_target('Echo (oracle)', make_echo(delay), [(0, 2)], 'Q',
                         T=22, n_trials=10, seed=seed, latency=delay, min_gap=3,
                         contract=event_contract(fit_latency=False),
                         description=describe_target(
        'Reproduce every input edge exactly %d seconds later.' % delay,
        'Ten seeded schedules vary phase and spacing. A direct input-to-output '
        'connection fails because no additional latency offset is fitted.'))


def coincidence_oracle(seed=20260702):
    return oracle_target('Coincidence (oracle)', orc_coincidence, [(0, 1), (0, 3)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=4, align_prob=0.5,
                         contract=event_contract(),
                         description=describe_target(
        'Emit Q only when A and B arrive together.',
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
        'Each input edge starts a %d-second active interval that self-terminates.'
        % width,
        'Ten seeded random schedules space triggers far enough to verify '
        'termination and re-arming.'))


def pair_oracle(seed=20260702, gap=2):
    return oracle_target('Pair detector (oracle)', make_pair(gap), [(0, 2)], 'Q',
                         T=24, n_trials=12, seed=seed, latency=3, min_gap=1,
                         contract=event_contract(),
                         description=describe_target(
        'Emit Q when two input edges are separated by exactly %d seconds.' % gap,
        'Twelve seeded random trains mix valid pairs, wrong gaps, and isolated '
        'events.'))


def pair_two_widths_oracle(seed=20260702, pulse_width=None):
    """Physical-time pair detector with gap measured relative to input width.

    ``gap`` means leading-edge separation, matching the existing pair detector's
    convention. Every explicit input pulse is ``pulse_width`` wide; only a second
    leading edge exactly ``2 * pulse_width`` after the preceding edge emits Q.
    Fractional phases keep this target off the integer stimulus lattice.
    """
    if pulse_width is None:
        from .pulse import WIDTH
        pulse_width = WIDTH
    width = float(pulse_width)
    if width <= 0:
        raise ValueError('pulse_width must be positive')
    rng = random.Random(seed)
    # (edge positions in width units, positions that complete valid pairs)
    patterns = [
        ([0.0, 2.0], [2.0]),
        ([0.0, 2.0], [2.0]),
        ([0.0, 1.5], []),
        ([0.0, 2.5], []),
        ([0.0, 3.0], []),
        ([0.0], []),
        ([], []),
        ([0.0, 2.0, 7.0, 9.0], [2.0, 9.0]),
        ([0.0, 2.0, 4.0], [2.0, 4.0]),
        ([0.0, 2.0, 5.0], [2.0]),
        ([0.0, 3.0, 5.0], [5.0]),
        ([0.0, 2.5, 4.5], [4.5]),
    ]
    rng.shuffle(patterns)
    latency = 1.0
    trials_data = []
    last_time = 0.0
    for positions, completions in patterns:
        phase = rng.uniform(1.25, 3.75) * width
        starts = [phase + position * width for position in positions]
        expected = [phase + position * width + latency
                    for position in completions]
        last_time = max([last_time] + starts + expected)
        trials_data.append((starts, expected))

    horizon = max(24, int(math.ceil(last_time + 6.0 * width + latency)))
    trials = []
    for starts, expected_events in trials_data:
        # Both asynchronous backends consume input_events directly at their
        # real fractional times. A zero stream remains as an honest fallback:
        # a clocked backend cannot silently claim it ran this continuous-time
        # target by rounding the fractional phases.
        streams = [(0,)] * horizon
        exp = [0] * horizon
        trials.append(Trial(
            streams, {'Q': exp}, {'Q': expected_events},
            input_events=[[(start, width) for start in starts]]))

    target = TemporalTarget(
        'Pair gap 2x width (oracle)', [(0, 2)],
        [OutputTerminal('Q', (2, 2))], horizon, trials,
        grid_size=5, iters=30,
        contract=event_contract(0.15 * width,
                                max(12.0, 12.0 * width)),
        latency=latency,
        supported_backends=('nervous', 'lut'),
        category='Pulse width & duration',
        description=describe_target(
            'Emit Q when two physical input-pulse leading edges are separated by '
            'exactly twice the input pulse width (2w).',
            'Twelve seeded fractional-phase schedules include exact 2w pairs, '
            '1.5w/2.5w/3w wrong gaps, chains, mixed valid/invalid gaps, a lone '
            'pulse, and silence. Input pulse width w is %.3g seconds.' % width))
    return target


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
            'Five long schedules cover start-only, one-step, and two-step '
            'operation with enough time to observe each settled cadence.'),
        contract=cadence_step_contract(
            min_period=2,
        # A command can meet a circulating pulse at any phase.  Its short
        # switchover is intentionally unscored; only the settled cadence matters.
            max_period=6, settle=6, min_events=4, max_delay=8))


def gated_oscillator_oracle(seed=20260702):
    # START fixes the phase, so the commanded output is a fully determined spike
    # train and event correspondence scores it (see make_gated_oscillator). It
    # must NOT fall through to the default state contract: this target's active
    # epochs are single ticks of a period-2 cadence, every one of them shorter
    # than a legal circulation gap, so a state contract drops all of them and
    # scores the case on its quiet epochs alone — under which silence, a single
    # pulse and a correct oscillator are indistinguishable at 1.0.
    return oracle_target('Gated oscillator (oracle)', make_gated_oscillator(),
                         [(0, 1), (0, 3)], 'Q',
                         T=44, n_trials=12, seed=seed, latency=2, min_gap=12,
                         global_gap=True, contract=event_contract(),
                         description=describe_target(
        'Input A starts a period-2 output cadence; input B stops it. Q is quiet '
        'outside the commanded run interval.',
        'Twelve seeded random A/B schedules verify start, sustained running, '
        'stop, and later restart.'))


def resettable_toggle_oracle(seed=20260702):
    return oracle_target('Resettable toggle (oracle)', make_resettable_toggle(),
                         [(0, 1), (0, 3)], 'Q',
                         T=44, n_trials=12, seed=seed, latency=2, min_gap=10,
                         global_gap=True,
                         description=describe_target(
        'Input A flips the stored bit; input B clears it to 0 and dominates a '
        'simultaneous A+B event.',
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
            'Stretch each input pulse to twice its width: a pulse held x seconds '
            'yields one contiguous 2x-second output hold starting with it. The '
            'bank mixes widths x in %s, so a fixed delay or fixed-length '
            'one-shot cannot fit them all - the circuit must measure x.'
            % (tuple(widths),),
            'Seeded single- and double-pulse schedules across every width plus '
            'a silent guard trial.'),
        contract=state_contract(), latency=latency,
        category='Pulse width & duration')


def pulse_width_sum_oracle(seed=20260716):
    """Emit one Q pulse whose duration is width(A) + width(B).

    One shared output-latency offset is fitted across the whole bank. Fitness is
    therefore about accumulating the two durations into Q, not choosing one
    topology-specific absolute response time.
    """
    rng = random.Random(seed)
    pairs = ((0.5, 1.0), (1.5, 0.75), (2.5, 1.25),
             (0.75, 2.25), (1.0, 3.0), (2.0, 0.5))
    trials = []
    for index, (width_a, width_b) in enumerate(pairs):
        phase = rng.uniform(1.0, 2.5)
        offset = (0.0, 0.4, 1.1)[index % 3]
        events = [[(phase, width_a)], [(phase + offset, width_b)]]
        start = max(phase, phase + offset) + 1.0
        expected = [(start, start + width_a + width_b)]
        trials.append(Trial(
            [(0, 0)] * 24, {'Q': [0] * 24}, input_events=events,
            expected_intervals={'Q': expected}))
    # Missing-input and silence guards: addition requires one pulse from both.
    trials.extend([
        Trial([(0, 0)] * 24, {'Q': [0] * 24},
              input_events=[[(2.0, 1.5)], []],
              expected_intervals={'Q': []}),
        Trial([(0, 0)] * 24, {'Q': [0] * 24},
              input_events=[[], [(2.0, 1.5)]],
              expected_intervals={'Q': []}),
        Trial([(0, 0)] * 24, {'Q': [0] * 24}, input_events=[[], []],
              expected_intervals={'Q': []}),
    ])
    return TemporalTarget(
        'Pulse width sum (oracle)', [(0, 1), (0, 3)],
        [OutputTerminal('Q', (2, 2))], 24, trials,
        grid_size=5, iters=30, contract=interval_contract(0.20),
        supported_backends=('nervous',),
        supported_models=('pulse_delay',),
        category='Pulse width & duration',
        waveform_contract='width_sum',
        description=describe_target(
            'For one A pulse and one B pulse, emit one Q pulse whose '
            'width equals width(A) + width(B).',
            'Six fractional-phase width pairs vary both widths and overlap; '
            'A-only, B-only, and silent trials guard false output.'))


def odd_pulse_selector_oracle(seed=20260716):
    """Pass the 1st, 3rd, 5th... A pulse, preserving each selected width."""
    rng = random.Random(seed)
    width_banks = (
        (), (0.5,), (1.75, 0.75), (2.5, 0.5, 1.25),
        (0.75, 2.0, 0.5, 1.5), (1.0, 2.25, 0.75, 1.75, 0.5),
    )
    trials = []

    def add(events):
        expected = [(start + 1.0, start + 1.0 + width)
                    for index, (start, width) in enumerate(events)
                    if index % 2 == 0]
        last = max((start + width for start, width in events), default=0.0)
        T = max(24, int(math.ceil(last + 4.0)))
        trials.append(Trial(
            [(0,)] * T, {'Q': [0] * T}, input_events=[events],
            expected_intervals={'Q': expected}))

    for widths in width_banks:
        cursor = rng.uniform(1.0, 2.0)
        events = []
        for width in widths:
            events.append((cursor, width))
            cursor += width + rng.uniform(1.0, 1.75)
        add(events)

    # ANTI-TIMING banks. The plain banks alone do NOT enforce index counting:
    # every start-to-start gap there is shorter than any two chained gaps, so a
    # parity-free REFRACTORY FILTER with one fixed dead time reproduced every
    # schedule exactly (measured: D ~= 4.1 scored 1.0). These three schedules
    # make every fixed dead time fail on SEVERAL trials at once: 2nd pulses
    # must be SUPPRESSED across long >= 5.75s gaps (any filter still passing
    # there needs D beyond them), while 3rd pulses must be PASSED only ~3s or
    # ~1.5s after the previous accepted one (needs D below that, start- or
    # end-referenced). Only counting the input index satisfies all banks; the
    # best fixed dead time now measures ~0.87, safely below the 0.90
    # certification bar (see tests/test_pulse_models.py).
    start = rng.uniform(1.0, 2.0)
    add([(start, 0.5),
         (start + rng.uniform(5.75, 7.0), rng.choice((0.75, 1.0, 1.25)))])
    start = rng.uniform(1.0, 2.0)
    first_gap, second_gap = rng.uniform(1.3, 1.6), rng.uniform(1.3, 1.6)
    add([(start, 0.5), (start + first_gap, 0.5),
         (start + first_gap + second_gap, 0.75)])
    start = rng.uniform(1.0, 2.0)
    long_a, long_b = rng.uniform(5.75, 7.0), rng.uniform(5.75, 7.0)
    quick = rng.uniform(1.3, 1.6)
    add([(start, 0.5), (start + long_a, 0.75),
         (start + long_a + quick, 0.75),
         (start + long_a + quick + long_b, 0.5)])

    horizon = max(len(trial.streams) for trial in trials)
    for trial in trials:
        trial.streams.extend([(0,)] * (horizon - len(trial.streams)))
        trial.expected['Q'].extend([0] * (horizon - len(trial.expected['Q'])))
    return TemporalTarget(
        'Odd pulse selector (oracle)', [(0, 2)],
        [OutputTerminal('Q', (2, 2))], horizon, trials,
        grid_size=5, iters=30, contract=interval_contract(0.20),
        supported_backends=('nervous',),
        supported_models=('pulse_delay',),
        category='Pulse width & duration',
        waveform_contract='odd_selector',
        description=describe_target(
            'Pass A pulses numbered 1, 3, 5, ... to Q, suppress pulses numbered '
            '2, 4, 6, ... and preserve every passed pulse width.',
            'Banks contain zero through five pulses with mixed widths, fractional '
            'phases, and varied gaps; the empty bank guards autonomous output. '
            'Two adversarial schedules (a suppressed pulse after a long gap, an '
            'accepted third pulse after quick ones) defeat fixed dead-time '
            'refractory filters, so only counting the pulse index passes.'))


def a_parity_query_oracle(seed=20260716):
    """B emits Q exactly when the cumulative number of A edges is odd."""
    rng = random.Random(seed)
    banks = []
    # Each mixed schedule queries initial even parity, queries odd parity twice
    # (proving B does not reset it), returns to even, adds two more A edges
    # (still even), then returns to odd. Only the timings vary across trials.
    operations = ('B', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B')
    for _ in range(10):
        cursor = rng.randint(2, 3)
        a_ticks, b_ticks = [], []
        for operation in operations:
            (a_ticks if operation == 'A' else b_ticks).append(cursor)
            cursor += rng.randint(3, 4)
        banks.append({0: a_ticks, 1: b_ticks})
    guard = rng.randint(2, 4)
    banks.extend([
        {0: [guard, guard + 4, guard + 8, guard + 13]},
        {1: [guard, guard + 4, guard + 9, guard + 14]},
        {},
    ])
    target = _event_bank_target(
        'A parity query (oracle)', make_a_parity_query(),
        [(0, 1), (0, 3)], banks, T=48, latency=2,
        description=describe_target(
            'Count input A events cumulatively. Whenever B fires, emit Q if '
            'the number of A events seen so far is odd; remain silent if it is '
            'even.',
            'Ten seeded schedules query both parities, repeat B without changing '
            'the count, and add consecutive A events. A-only, B-only, and silent '
            'guards reject direct echoes and autonomous output. B does not reset '
            'the retained parity. Pulse widths vary from 0.5 to 2.25 seconds; '
            'each pulse counts once regardless of its duration.'))
    return _mix_event_widths(target, rng)


def a_mod3_query_oracle(seed=20260716):
    """B queries whether the cumulative A count is a positive multiple of 3."""
    rng = random.Random(seed)
    operations = ('B', 'A', 'B', 'A', 'B', 'A', 'B', 'B',
                  'A', 'B', 'A', 'A', 'B')
    banks = []
    for _ in range(10):
        cursor = rng.randint(2, 3)
        a_ticks, b_ticks = [], []
        for operation in operations:
            (a_ticks if operation == 'A' else b_ticks).append(cursor)
            cursor += rng.randint(3, 4)
        banks.append({0: a_ticks, 1: b_ticks})
    guard = rng.randint(2, 4)
    banks.extend([
        {0: [guard, guard + 4, guard + 8, guard + 13, guard + 17]},
        {1: [guard, guard + 4, guard + 9, guard + 14]},
        {},
    ])
    target = _event_bank_target(
        'A modulo-3 query (oracle)', make_a_mod3_query(),
        [(0, 1), (0, 3)], banks, T=56, latency=2,
        description=describe_target(
            'Count A events cumulatively. Emit Q when B fires and the positive '
            'A count is divisible by three; otherwise remain silent.',
            'Schedules query counts 0, 1, 2, 3, 4, and 6, including repeated B '
            'queries at count 3. A-only, B-only, and silent guards reject echoes '
            'and autonomous output. Mixed 0.5-to-2.25-second pulse widths ensure '
            'the circuit counts edges rather than high-time.'))
    return _mix_event_widths(target, rng)


def a_batch_parity_query_oracle(seed=20260716):
    """B reports odd A parity since the preceding B and starts a new batch."""
    rng = random.Random(seed)
    operations = ('B', 'A', 'B', 'B', 'A', 'A', 'B',
                  'A', 'A', 'A', 'B', 'B')
    banks = []
    for _ in range(10):
        cursor = rng.randint(2, 3)
        a_ticks, b_ticks = [], []
        for operation in operations:
            (a_ticks if operation == 'A' else b_ticks).append(cursor)
            cursor += rng.randint(3, 4)
        banks.append({0: a_ticks, 1: b_ticks})
    guard = rng.randint(2, 4)
    banks.extend([
        {0: [guard, guard + 4, guard + 8, guard + 13]},
        {1: [guard, guard + 4, guard + 9, guard + 14]},
        {},
    ])
    target = _event_bank_target(
        'A batch parity query (oracle)', make_a_batch_parity_query(),
        [(0, 1), (0, 3)], banks, T=52, latency=2,
        description=describe_target(
            'Count A events since the previous B. When B fires, emit Q if that '
            'batch count is odd, then clear the parity and begin a new batch.',
            'Schedules query empty, one-A, two-A, and three-A batches, with '
            'back-to-back empty queries. A-only, B-only, and silent guards plus '
            'mixed 0.5-to-2.25-second widths test edge counting, clearing, and '
            'retained parity.'))
    return _mix_event_widths(target, rng)


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
            'forbids output without input.' % (tuple(periods),),
            'Three phase-varied periodic schedules per period plus a silent '
            'guard trial.'),
        contract=event_contract(), latency=latency)


def period_tripler_oracle(seed=20260702, periods=(2, 3, 4)):
    """Mixed-period edge trains whose output interval must be exactly 3p."""
    periods = tuple(int(period) for period in periods)
    if not periods or any(period < 2 for period in periods):
        raise ValueError('period tripler requires input periods of at least 2')
    T, latency, rng = 42, 1, random.Random(seed)
    trials = []

    def add(ticks):
        on = set(ticks)
        streams = [(1 if tick in on else 0,) for tick in range(T)]
        exp = label_trace(orc_period_tripler, streams, T, latency)
        events = [float(tick) for tick, value in enumerate(exp) if value == 1]
        trials.append(Trial(streams, {'Q': exp}, {'Q': events}))

    for period in periods:
        for _ in range(3):
            phase = rng.randint(1, period + 2)
            add([phase + index * period
                 for index in range((T - 2 - phase) // period + 1)])
    add([])
    return TemporalTarget(
        'Period tripler (oracle)', [(0, 2)], [OutputTerminal('Q', (2, 2))],
        T, trials, grid_size=5, iters=30,
        description=describe_target(
            'Triple the input period: for a periodic input of period p, emit '
            'every third input edge so consecutive Q events are separated by '
            '3p seconds.',
            'Three phase-varied schedules for each input period in %s plus a '
            'silent guard. Mixed periods prevent a fixed oscillator from '
            'matching the bank.' % (periods,)),
        contract=event_contract(), latency=latency)


def period_halver_oracle(seed=20260702, periods=(4, 6, 8)):
    """Measure p, then insert a midpoint edge to produce output period p/2.

    The first input interval is a causal measurement window and produces no
    scored output. From the second edge onward, Q fires at each input edge and
    halfway to the next one using the measured period. Period 2 is excluded:
    halving it would create an edge every second, which wired-OR represents as
    one held level rather than a periodic edge train.
    """
    periods = tuple(int(period) for period in periods)
    if (not periods
            or any(period < 4 or period % 2 for period in periods)):
        raise ValueError('period halver requires even input periods >= 4')
    T, latency, rng = 50, 1, random.Random(seed)
    trials = []
    for period in periods:
        for _ in range(3):
            phase = rng.randint(1, period + 1)
            ticks = [phase + index * period
                     for index in range((T - 3 - phase) // period + 1)]
            output = []
            for edge in ticks[1:]:              # first interval measures p
                output.extend((edge + latency,
                               edge + period // 2 + latency))
            trials.append(_explicit_event_trial(
                T, 1, {0: ticks}, output))
    trials.append(_explicit_event_trial(T, 1, {}, []))
    return TemporalTarget(
        'Period halver (oracle)', [(0, 2)], [OutputTerminal('Q', (2, 2))],
        T, trials, grid_size=5, iters=30,
        description=describe_target(
            'Halve the input period: after measuring one complete interval p, '
            'emit Q at the input cadence and at each midpoint so consecutive '
            'Q events are p/2 seconds apart.',
            'Three phase-varied schedules for each even period in %s plus a '
            'silent guard. The mixed periods require measurement rather than '
            'a fixed-rate oscillator.' % (periods,)),
        contract=event_contract(), latency=latency)


def temporal_sum_oracle(seed=20260702):
    """Encode ΔA + ΔB as the interval between two output events.

    Each positive trial supplies exactly two A events and two B events. Once
    both input intervals are complete, Q emits a start event and a finish event
    separated by their sum. Incomplete and silent trials require no output.
    """
    T, latency, rng = 48, 1, random.Random(seed)
    interval_pairs = [
        (2, 3), (3, 4), (4, 2), (5, 3), (2, 6),
        (4, 5), (3, 6), (5, 2), (6, 4),
    ]
    rng.shuffle(interval_pairs)
    trials = []
    for gap_a, gap_b in interval_pairs:
        start_a = rng.randint(2, 7)
        start_b = rng.randint(2, 7)
        a_ticks = [start_a, start_a + gap_a]
        b_ticks = [start_b, start_b + gap_b]
        complete = max(a_ticks[-1], b_ticks[-1])
        first_q = complete + latency
        trials.append(_explicit_event_trial(
            T, 2, {0: a_ticks, 1: b_ticks},
            [first_q, first_q + gap_a + gap_b]))

    # Neither lane alone, one unmeasurable event per lane, nor silence defines
    # two intervals, so all of these must remain quiet.
    trials.extend([
        _explicit_event_trial(T, 2, {0: [3, 7]}, []),
        _explicit_event_trial(T, 2, {1: [4, 9]}, []),
        _explicit_event_trial(T, 2, {0: [3], 1: [6]}, []),
        _explicit_event_trial(T, 2, {}, []),
    ])
    return TemporalTarget(
        'Temporal sum (oracle)', [(0, 1), (0, 3)],
        [OutputTerminal('Q', (2, 2))], T, trials,
        grid_size=5, iters=30,
        description=describe_target(
            'Measure the interval ΔA between two A events and ΔB between two B '
            'events, then emit two Q events separated by ΔA + ΔB seconds.',
            'Nine seeded schedules vary both intervals and lane ordering. '
            'A-only, B-only, incomplete, and silent guards forbid direct '
            'connections and fixed bursts.'),
        contract=event_contract(), latency=latency)


def c_element_oracle(seed=20260702):
    """Balanced rendezvous schedules that require influence from both inputs.

    The generic aligned-stream sampler is deliberately not used here. It makes
    every B edge a same-or-later copy of A, which lets a direct B-to-Q path imitate
    the rendezvous without remembering A. Every mixed trial below contains an
    A-first round, a B-first round, a simultaneous round, and an incomplete final
    round. Dedicated A-only, B-only, and silent guards penalise either-input
    echoes, wired OR, and spontaneous oscillation.
    """
    rng = random.Random(seed)
    banks = []
    for trial_index in range(10):
        a_ticks, b_ticks = set(), set()
        cursor = rng.randint(2, 3)
        rounds = ['A', 'B', 'tie']
        rng.shuffle(rounds)
        for kind in rounds:
            if kind == 'tie':
                a_ticks.add(cursor)
                b_ticks.add(cursor)
                close = cursor
            else:
                gap = rng.randint(2, 5)
                first, second = ((a_ticks, b_ticks) if kind == 'A'
                                 else (b_ticks, a_ticks))
                first.add(cursor)
                # Repeats from the first input must not create Q before the
                # other side arrives. Keep them strictly inside this round.
                if gap >= 4 and rng.random() < 0.65:
                    first.add(cursor + rng.randint(2, gap - 1))
                close = cursor + gap
                second.add(close)
            cursor = close + rng.randint(3, 4)

        # Leave one side pending at the end. Alternating the side prevents an
        # A-only or B-only readout from being favoured by pooled event scoring.
        pending = a_ticks if trial_index % 2 == 0 else b_ticks
        pending.add(cursor)
        if rng.random() < 0.5:
            pending.add(cursor + 2)
        banks.append({0: sorted(a_ticks), 1: sorted(b_ticks)})

    guard_start = rng.randint(3, 5)
    banks.extend([
        {0: [guard_start, guard_start + 4, guard_start + 9]},
        {1: [guard_start + 1, guard_start + 6, guard_start + 10]},
        {},
    ])
    return _event_bank_target(
        'C-element (oracle)', make_c_element(), [(0, 1), (0, 3)], banks,
        T=36, latency=2,
        description=describe_target(
            'Transition-signalling Muller C-element (2-input rendezvous/join): '
            'emit Q once BOTH inputs have produced an edge, in either order, '
            'then rearm. Remembering the first arrival while waiting for the '
            'second is the stored state — the asynchronous handshake keystone.',
            'Ten seeded mixed schedules each contain A-first, B-first, '
            'simultaneous, repeated-first, incomplete, and re-arm cases. A-only, '
            'B-only, and silent guards forbid single-input echoes, wired OR, and '
            'free oscillation.'))


def refractory_filter_oracle(seed=20260702, dead_time=3):
    """Seeded burst banks straddling the dead-time acceptance boundary."""
    rng = random.Random(seed)
    banks = []
    for _ in range(10):
        tick = rng.randint(2, 4)
        pulses = [tick]
        # Every trial contains a blocked short gap, an accepted boundary/long
        # gap, and two independently sampled gaps.  Pulses stay one tick wide.
        gaps = [rng.choice((2, 3)), rng.choice((4, 5, 6)),
                rng.choice((2, 3, 4, 5, 6)), rng.choice((3, 4, 5, 7))]
        if rng.random() < 0.5:
            gaps[0], gaps[1] = gaps[1], gaps[0]
        for gap in gaps:
            tick += gap
            if tick < 31:
                pulses.append(tick)
        banks.append({0: pulses})
    banks.append({})                         # never stimulated: must stay quiet
    return _event_bank_target(
        'Refractory filter (oracle)', make_refractory_filter(dead_time),
        [(0, 2)], banks, T=34, latency=1,
        description=describe_target(
            'Pass the first input event, suppress events for %d seconds, then '
            're-arm and pass the next eligible event.' % dead_time,
            'Ten seeded burst schedules mix blocked gaps below the dead time, '
            'the exact re-arm boundary, longer accepted gaps, and a silent guard.'))


def a_first_rendezvous_oracle(seed=20260702):
    """Balanced, variable-gap A-first/B-first/tie rounds with re-arming."""
    rng = random.Random(seed)
    banks = []
    for _ in range(12):
        a_ticks, b_ticks = set(), set()
        cursor = rng.randint(2, 4)
        rounds = ['A', 'B', 'tie']
        rng.shuffle(rounds)
        for kind in rounds:
            gap = rng.randint(2, 6)
            if kind == 'A':
                a_ticks.add(cursor); b_ticks.add(cursor + gap)
                if gap >= 4 and rng.random() < 0.5:
                    a_ticks.add(cursor + 2)       # same-side distractor
            elif kind == 'B':
                b_ticks.add(cursor); a_ticks.add(cursor + gap)
                if gap >= 4 and rng.random() < 0.5:
                    b_ticks.add(cursor + 2)
            else:
                a_ticks.add(cursor); b_ticks.add(cursor)
            cursor += gap + rng.randint(3, 5)
        # An incomplete final request checks that merely seeing A is not enough.
        if cursor < 39 and rng.random() < 0.5:
            (a_ticks if rng.random() < 0.5 else b_ticks).add(cursor)
        banks.append({0: sorted(a_ticks), 1: sorted(b_ticks)})
    return _event_bank_target(
        'A-first rendezvous (oracle)', make_a_first_rendezvous(),
        [(0, 1), (0, 3)], banks, T=42, latency=2,
        description=describe_target(
            'Treat each A/B pair as a race: emit Q when A arrived first and B '
            'completes the round; B-first and simultaneous ties stay silent.',
            'Twelve seeded schedules mix both orders, variable gaps, ties, '
            'same-channel distractors, incomplete rounds, and repeated re-arming.'))


def collision_serializer_oracle(seed=20260702, spacing=2):
    """Banks where singles remain single and A+B collisions become two events."""
    rng = random.Random(seed)
    banks = []
    for _ in range(10):
        a_ticks, b_ticks = set(), set()
        cursor = rng.randint(2, 4)
        episodes = ['A', 'B', 'AB', rng.choice(('A', 'B', 'AB'))]
        rng.shuffle(episodes)
        for kind in episodes:
            if kind in ('A', 'AB'):
                a_ticks.add(cursor)
            if kind in ('B', 'AB'):
                b_ticks.add(cursor)
            cursor += spacing + rng.randint(4, 6)  # let the serializer drain
        banks.append({0: sorted(a_ticks), 1: sorted(b_ticks)})
    banks.append({})                               # no spontaneous output
    return _event_bank_target(
        'Collision serializer (oracle)', make_collision_serializer(spacing),
        [(0, 1), (0, 3)], banks, T=40, latency=1,
        description=describe_target(
            'Merge A and B onto Q without losing event count: an isolated input '
            'makes one Q event, while simultaneous A+B is serialized into two '
            'Q events separated by %d seconds.' % spacing,
            'Ten seeded episode banks mix A-only, B-only, and collision events '
            'at varied phases, plus a silent guard; every input token must emerge.'))


def watchdog_oracle(seed=20260702, timeout=5):
    """Heartbeat banks covering safe, deadline, late, re-arm, and silent cases."""
    rng = random.Random(seed)
    banks = []
    # Exact-deadline heartbeats cancel the alarm and exercise the off-by-one.
    start = rng.randint(2, 4)
    banks.append({0: list(range(start, 38, timeout))})
    for _ in range(10):
        tick = rng.randint(2, 4)
        pulses = [tick]
        # Guaranteed safe/deadline gaps and late gaps; shuffle their order so
        # held-out seeds change the history without dropping boundary coverage.
        gaps = [rng.choice((3, 4, 5)), rng.choice((6, 7, 8)),
                rng.choice((3, 5)), rng.choice((6, 8, 9))]
        rng.shuffle(gaps)
        for gap in gaps:
            tick += gap
            if tick < 37:
                pulses.append(tick)
        banks.append({0: pulses})
    banks.append({})                               # never armed: no alarm
    return _event_bank_target(
        'Watchdog timeout (oracle)', make_watchdog(timeout),
        [(0, 2)], banks, T=42, latency=1,
        description=describe_target(
            'After the first heartbeat, emit one alarm if no new heartbeat '
            'arrives for %d seconds. A deadline heartbeat cancels the alarm; after '
            'an alarm, a later heartbeat re-arms the watchdog.' % timeout,
            'Seeded schedules mix safe, exact-deadline, and late heartbeat gaps, '
            'multiple timeout/re-arm rounds, and never-armed silence.'))


ORACLE_SPECS = {
    'Pulse width sum (oracle)':    pulse_width_sum_oracle,
    'Odd pulse selector (oracle)': odd_pulse_selector_oracle,
    'A parity query (oracle)':     a_parity_query_oracle,
    'A modulo-3 query (oracle)':   a_mod3_query_oracle,
    'A batch parity query (oracle)': a_batch_parity_query_oracle,
    'SR latch (oracle)':          sr_latch_oracle,
    'C-element (oracle)':         c_element_oracle,
    'Refractory filter (oracle)': refractory_filter_oracle,
    'A-first rendezvous (oracle)': a_first_rendezvous_oracle,
    'Collision serializer (oracle)': collision_serializer_oracle,
    'Watchdog timeout (oracle)':   watchdog_oracle,
    'Toggle (oracle)':            toggle_oracle,
    'Echo (oracle)':              echo_oracle,
    'Coincidence (oracle)':       coincidence_oracle,
    'One-shot (oracle)':          one_shot_oracle,
    'Pulse doubler (oracle)':     pulse_doubler_oracle,
    'Period doubler (oracle)':    period_doubler_oracle,
    'Period tripler (oracle)':    period_tripler_oracle,
    'Period halver (oracle)':     period_halver_oracle,
    'Temporal sum (oracle)':      temporal_sum_oracle,
    'Pair detector (oracle)':     pair_oracle,
    'Pair gap 2x width (oracle)': pair_two_widths_oracle,
    'Period stepper (oracle)':    period_stepper_oracle,
    'Gated oscillator (oracle)':  gated_oscillator_oracle,
    'Resettable toggle (oracle)': resettable_toggle_oracle,
}

ORACLE_TARGETS = {name: spec() for name, spec in ORACLE_SPECS.items()}
