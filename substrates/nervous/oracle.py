"""
substrates/nervous/oracle.py - targets defined by a reference model, not hand-picked traces.

Hand-writing "input pulses at ticks 3,5 -> output at tick 8" bakes OUR timing
into the goal, and a circuit can plateau not because it fails the function but
because our chosen timing fights its internal phase (an evolved SR latch with a
period-2 loop resets fine on even ticks and misses odd ones - the timing was
adversarial, not the latch broken).

An oracle target instead specifies the goal as:
    * an ORACLE - a tiny reference state machine  oracle(in_bits, state) ->
      (out_bits, new_state)  that defines the intended input->output relation;
    * a STIMULUS GENERATOR that samples random input schedules.
Many schedules are sampled; the oracle labels each; the circuit is scored on
reproducing the RELATION across all of them (with the usual response-latency
grace + phase-tolerant holds). Solving means implementing the function, not
memorising a timing - and `holdout_score` re-samples fresh schedules to certify
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
                        toggle_contract,
                        interval_contract, state_contract)


# -- reference oracles (in_bits, state) -> (out_bits, new_state) ------------------

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
    This is the paper's core memory image made input-driven - "a pulse circulates
    a loop of buffers until stopped by an inhibitory input" (section 3). Period 2 is even,
    so the bipartite honeycomb reaches it on the cheap one-pulse route; the LUT
    recurrent CA holds the same two-state cycle. START fixes the phase, so the
    ordinary latency-invariant F1 scores it - no special mode needed. STOP wins a
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
    B forces it to 0. A toggle loop guarded by an inhibitory reset line - memory
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
    the input's duration, not just respond to its edge - a fixed delay-line
    cheat (out = in OR delay_k(in), width x+k) only doubles the single width
    x = k, so the trial bank mixes several widths to force real measurement.

    As a state machine: while the input is high, output high and bank one tick
    of 'debt'; after it falls, keep the output high until the debt is repaid -
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
    periodic input train of period p yields a periodic output of period 2p -
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
    """Muller C-element in transition signalling - a 2-input rendezvous / join.

    Emit an output event only once BOTH inputs have produced an edge, in EITHER
    order, then rearm and wait for the next pair. A lone edge on one input must
    NOT emit; the element has to REMEMBER that the first input arrived while it
    waits for the second - so this is a stored-state element, the asynchronous
    handshake keystone (the C-element is what joins the two rails of a
    micropipeline). The stored state is which inputs have arrived this round.

    (The textbook level-mode C-element - output high while both inputs are high,
    holding on disagreement - is not a natural fit for an edge-coincidence
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


# -- stimulus generation ---------------------------------------------------------

def sample_streams(rng, T, n_inputs, min_gap=4, jitter=4, align_prob=0.0,
                   global_gap=False):
    """A list[T] of random input-bit tuples. Each input gets a sparse pulse
    train (gap >= min_gap, random jitter). `align_prob` occasionally fires all
    inputs on the SAME tick - needed so coincidence/pair targets see positive
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
    """Run the oracle over a stream and shift its output right by `latency` - the
    circuit's (deterministic, path-length) propagation delay - so the expected
    trace lines up with when a correct circuit actually responds. The first
    `latency` ticks are startup grace (unscored). No transition masking: that
    would swallow short pulses; the scorer's own phase-tolerant hold rule gives
    the +/-1 slack that level-holds need, and pulse relations have a fixed delay
    the circuit matches exactly."""
    raw, st = [], None
    for t in range(T):
        ob, st = oracle(streams[t], st)
        raw.append(ob[0])
    return [None] * latency + raw[:max(0, T - latency)]


# -- target builder --------------------------------------------------------------

def oracle_target(name, oracle, inputs, output_role, T=24, n_trials=12,
                  seed=20260702, latency=2, min_gap=5, jitter=4, align_prob=0.0,
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
            rng, T, n_in, min_gap=min_gap, jitter=jitter, align_prob=align_prob,
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
# physical edge - unreachable expected output in every model.
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
    would be scored under the default uniform physics - see certification.py)."""
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


# -- preset oracle targets (input-driven relations) ------------------------------

def sr_latch_oracle(seed=20260702):
    # min_gap 10 with the default jitter put EVERY hold interval in the band
    # 10..14, and a stored bit whose duration never varies is not a stored bit:
    # a fixed-length burst fits all of them at once. Measured on the old
    # spacing, a 4-tap delay line driven by Set alone -- Reset not connected to
    # anything -- scored a perfect 1.000, because at alignment -2 the burst's
    # natural death (4 taps x 3 ticks) landed where the Reset happened to be.
    # The spread has to come from the jitter, not from a lower floor: commands
    # closer than ~10 ticks leave too little time to tell a ringing output from
    # a quiet one (test_memory_targets_leave_time_to_observe_each_state), so
    # min_gap stays at 10 and the range widens upward to 10..30. That is enough
    # that no single burst length can both cover the long holds and fall silent
    # through the short ones.
    target = oracle_target('SR latch (oracle)', orc_sr_latch, [(0, 1), (0, 3)], 'Q',
                         T=40, n_trials=12, seed=seed, latency=2, min_gap=10,
                         jitter=20, global_gap=True,
                         description=describe_target(
        'Input A sets one stored bit; input B resets it; otherwise Q retains its '
        'previous state.',
        'Twelve seeded random schedules spanning a wide range of hold durations, '
        'plus explicit never-set, short-hold, long-hold and Set->Reset->Set tests, '
        'exercise storage, clearing, and reloading.'))
    # Persistence guardrails: silence without Set, a Set->Reset->Set trial that
    # requires the same circuit to store, clear, and store again, and a pair
    # that contrasts the shortest hold with the longest. A one-shot response to
    # Set, an unconditional ring, or any fixed-duration burst fails these.
    silent = [(0, 0)] * target.T
    cycle = [(0, 0)] * target.T
    cycle[3] = (1, 0)
    cycle[16] = (0, 1)
    cycle[29] = (1, 0)
    short = [(0, 0)] * target.T
    short[4] = (1, 0)           # shortest observable hold: a long burst
    short[14] = (0, 1)          # overruns it into the cleared interval
    short[26] = (1, 0)
    short[36] = (0, 1)
    long_hold = [(0, 0)] * target.T
    long_hold[4] = (1, 0)       # never reset: must still be set at the horizon
    target.trials.extend([
        Trial(streams, {'Q': label_trace(orc_sr_latch, streams, target.T, 2)})
        for streams in (silent, cycle, short, long_hold)
    ])
    # A LUT level latch has one state stage and one feedback stage.  A command
    # that vanishes after a single lattice tick disappears on the same wave the
    # feedback first arrives, producing a forced phase swap rather than either
    # stable fixed point.  Keep each logical command high for the minimum two
    # stages.  This is still ONE leading-edge event to the nervous substrate,
    # so its command semantics are unchanged; it merely gives the level device
    # a physically meaningful input aperture.
    for trial in target.trials:
        # Keep the logical stream as one command tick so schedule audits and
        # the oracle still see exactly one command.  The physical schedule is
        # where that level receives its two-stage aperture.
        trial.input_events = [
            [(float(tick), 2.0) for tick, row in enumerate(trial.streams)
             if row[lane] and (
                 tick == 0 or not trial.streams[tick - 1][lane])]
            for lane in range(2)]
    return target


def toggle_oracle(seed=20260702):
    return oracle_target('Toggle (oracle)', orc_toggle, [(0, 2)], 'Q',
                         T=40, n_trials=12, seed=seed, latency=2, min_gap=10,
                         global_gap=True,
                         contract=toggle_contract(),
                         description=describe_target(
        'Each input edge flips the stored output state.',
        'Twelve seeded random pulse trains vary phase and spacing.'))


def echo_oracle(seed=20260702, delay=3):
    return oracle_target('Echo (oracle)', make_echo(delay), [(0, 2)], 'Q',
                         # ``make_echo`` already shifts the state-machine
                         # output by ``delay`` samples. Adding it again in
                         # label_trace judged a stated 3-second echo at 6.
                         T=22, n_trials=10, seed=seed, latency=0, min_gap=3,
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


def one_shot_oracle(seed=20260702, width=12):
    # The hold MUST outlast what one pulse can cover, or a bare delay line is a
    # perfect one-shot. The bound comes from the state contract, not from taste:
    # scoring._state_case_score credits an active window whose longest silence is
    # within allowed_gap = 2*(delay + width_pulse) = 4, and a single pulse sitting
    # at the CENTRE of a d-tick window leaves silences of (d - width_pulse)/2. So
    # one pulse is refuted only when d > 2*allowed_gap + width_pulse = 9.
    #
    # The old width of 5 was chosen against the +/-1 ring tolerance of the earlier
    # scorer and no longer binds: measured on the current code, a plain input echo
    # scored a perfect 1.000 here, and every "solved" One-shot row in the audit was
    # that artifact. 12 clears the bound with margin and still self-terminates well
    # inside the horizon.
    #
    # Widening alone does not make the target need state, and the old spacing
    # (min_gap = width + 4, jitter 4 -> gaps 16..20) guaranteed every trigger
    # landed in an already-quiet output. The `rem == 0` guard in make_one_shot --
    # the ONE clause here that cannot be built without memory -- was therefore
    # never exercised by a single trial, and a 3-tap delay line scored 1.000
    # honestly rather than by a scoring artifact. Gaps of 6..20 straddle the
    # width so roughly half the triggers now arrive mid-pulse and must be
    # swallowed without extending or restarting the interval; a feedforward tap
    # answers them and spills past the window into the quiet epoch.
    return oracle_target('One-shot (oracle)', make_one_shot(width), [(0, 2)], 'Q',
                         T=52, n_trials=10, seed=seed, latency=2, min_gap=6,
                         jitter=14,
                         description=describe_target(
        'Each input edge starts a %d-second active interval that self-terminates. '
        'A trigger arriving while the interval is still active is ignored: it '
        'neither extends nor restarts it.' % width,
        'Ten seeded random schedules straddle the pulse width, so triggers both '
        'inside and outside the active interval verify suppression, termination '
        'and re-arming.'))


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
    # scores the case on its quiet epochs alone - under which silence, a single
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
    periods is the anti-cheat - a free-running oscillator or any fixed-cadence
    responder fits at most one input rate - and the silent guard trial kills
    oscillators outright (no input => no output). Phases vary per trial so a
    phase-locked fake can't memorise tick positions.

    Period 1 is deliberately EXCLUDED: a pulse every tick wired-OR merges into
    one held level (one edge), physically indistinguishable from constant
    input on both substrates - it carries no period."""
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
    """Encode deltaA + deltaB as the interval between two output events.

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
            'Measure the interval deltaA between two A events and deltaB between two B '
            'events, then emit two Q events separated by deltaA + deltaB seconds.',
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
            'second is the stored state - the asynchronous handshake keystone.',
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
    # Collisions used to be a third of the episodes and every episode was given
    # enough room to drain, so the queue never held more than one token and the
    # output grid never left the input grid. Against that, a wired-OR -- which
    # is what this substrate does for free at any merge -- was right about every
    # isolated event and lost only the second pulse of each collision: F1 0.86,
    # i.e. the collision handling that IS the target was worth 0.14 of its score.
    #
    # Counting alone cannot fix that. A wired-OR under-emits and a fixed k-tap
    # delay line over-emits, so trading one off against the other bottoms out
    # near F1 0.83 whatever the collision fraction. What separates a serializer
    # from BOTH is that it is not a shift-invariant filter: when tokens arrive
    # faster than `spacing` the queue backs up and the output grid detaches from
    # the input grid, and it re-attaches whenever the queue drains. So the banks
    # below interleave three regimes -- isolated singles (queue empty, output
    # rides the input), collisions (depth 2), and irregular bursts whose
    # inter-arrival times vary INSIDE the burst (depth 3-5, output phase set by
    # the backlog rather than by any input edge). A uniform burst would not do:
    # tokens arriving at a constant rate produce a constant-rate output, which is
    # exactly a multi-tap wire.
    rng = random.Random(seed)
    banks = []
    for _ in range(12):
        a_ticks, b_ticks = set(), set()
        cursor = rng.randint(2, 4)
        episodes = ['A', 'B', 'AB', 'AB', 'burst', 'burst',
                    rng.choice(('A', 'B', 'AB'))]
        rng.shuffle(episodes)
        for kind in episodes:
            if kind == 'burst':
                # 3-5 tokens at irregular spacing. Same-lane events stay >= 2
                # ticks apart so two stimuli never merge into one wide pulse.
                arrivals = [0]
                # Backlog DEPTH is the discriminator, so vary it widely: a
                # delay line has one fixed tap count and can only match one
                # depth, while a queue is indifferent to how deep it gets.
                for _step in range(rng.choice((1, 2, 5, 7))):
                    arrivals.append(arrivals[-1] + rng.choice((1, 2, 3)))
                # Alternate which lane leads the burst. With A always leading,
                # every backlog began on an A edge and a multi-tap delay line
                # watching lane A alone tracked the whole output.
                lead = rng.choice((0, 1))
                added = 0
                for index, offset in enumerate(arrivals):
                    before = offset - arrivals[index - 1] if index else 99
                    after = (arrivals[index + 1] - offset
                             if index + 1 < len(arrivals) else 99)
                    # Consecutive arrivals alternate lanes, so a lane repeats
                    # only every other arrival and its gap is at least 1+1=2.
                    # Doubling onto BOTH lanes is allowed only where both
                    # neighbours are >= 2 away, for the same reason.
                    both = before >= 2 and after >= 2 and (
                        index == 0 or rng.random() < 0.7)
                    if both or (index + lead) % 2 == 0:
                        a_ticks.add(cursor + offset)
                        added += 1
                    if both or (index + lead) % 2 == 1:
                        b_ticks.add(cursor + offset)
                        added += 1
                # Every queued token still owes `spacing` ticks of service, so
                # the headroom has to scale with the BACKLOG, not with the burst
                # width. Fixed headroom let a deep burst run past the horizon
                # and silently drop tokens the description promises will emerge.
                cursor += arrivals[-1] + added * spacing + rng.randint(2, 4)
                continue
            if kind in ('A', 'AB'):
                a_ticks.add(cursor)
            if kind in ('B', 'AB'):
                b_ticks.add(cursor)
            cursor += spacing + rng.randint(4, 6)  # let the serializer drain
        banks.append({0: sorted(a_ticks), 1: sorted(b_ticks)})
    banks.append({})                               # no spontaneous output
    return _event_bank_target(
        'Collision serializer (oracle)', make_collision_serializer(spacing),
        [(0, 1), (0, 3)], banks, T=112, latency=1,
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


# -- substrate-characteristic targets -------------------------------------------
# One target per substrate, each chosen to sit where that substrate's physics is
# a genuine advantage rather than an obstacle, and each harder than the existing
# members of its family.

def make_gap_bandpass(low=2, high=4):
    """Fire only when B follows A after a gap INSIDE a band. NERVOUS.

    Why this substrate: the answer is a function of nothing but elapsed time
    between two edges, and the circuit is judged on a band with BOTH a floor and
    a ceiling. A too-early B must be rejected as firmly as a too-late one, so a
    solver cannot be a plain coincidence detector (that gets the ceiling for
    free and fails the floor) nor a plain delay line (the reverse). It needs an
    internal delay of its own, opened by A, plus something that closes the
    window again - which on this hardware is exactly what the inhibitory input
    is for. No clock is involved anywhere; this is the asynchronous timing
    discrimination the substrate exists to do.

    Harder than 'Pair detector (gap 2)': that accepts one exact spacing on one
    input, where this spans a band across two inputs and must reject on both
    sides of it.
    """
    low, high = int(low), int(high)

    def f(inb, st):
        since = st
        # AGE FIRST, then judge. Ageing after the check made ``since`` report
        # one tick less than the elapsed time, so this device silently
        # implemented the band [low+1, high+1] while claiming [low, high] - and
        # the mis-specification was invisible until a hand-written "gap in band"
        # reference disagreed with the oracle on every single trial.
        if since is not None:
            since = since + 1 if since < high + 3 else None
        fire = 0
        # B is judged against the window opened by the LAST A. A simultaneous
        # A+B is judged against the PREVIOUS A, then opens a fresh window, so it
        # is never a gap of zero.
        if inb[1] and since is not None and low <= since <= high:
            fire = 1
        if inb[0]:
            since = 0
        return (fire,), since
    return f


def make_resettable_divider(divisor=4):
    """Emit one event per N input events; a second input clears the count. FNV.

    Why this substrate: N=4 needs TWO bits of state plus a decode plus a reset
    path, and FNV is the encoding that can NAME parts individually - a genome
    here places a toggle, another toggle, a join and a reset route as distinct
    components, which is precisely what a context-addressed genome cannot do
    (identical neighbourhoods are forced to become identical cells). It is the
    compositional case the FNV catalogue was assembled for: TOGGLE for the
    counter bits, LOGIC for the decode, DELAY to line them up.

    Harder than 'Resettable toggle' (one bit) and than 'Divide-by-3' (no reset):
    this needs modular counting AND asynchronous clearing in the same circuit,
    and a reset arriving mid-count must not emit.
    """
    divisor = int(divisor)
    if divisor < 2:
        raise ValueError('divisor must be at least 2')

    def f(inb, st):
        count = int(st or 0)
        if inb[1]:                       # reset clears, and never emits
            return (0,), 0
        if inb[0]:
            count += 1
            if count >= divisor:
                return (1,), 0
        return (0,), count
    return f


def orc_d_latch(inb, st):
    """Transparent D latch: Q follows D while Enable is high, holds when low. LUT.

    Why this substrate: this is a LEVEL device, not an edge device. While Enable
    is high the output must track D continuously - not respond to its edges -
    and when Enable falls the output must sit at whatever D last was. The LUT
    array is level logic that holds natively, so this is its home ground; the
    nervous net cannot express it at all, because a capacitively-coupled node
    has no stable high (see nv-held-level-not-a-signal) and 'follow this level'
    is not a signal it can read.

    Harder than 'SR latch': set/reset are two edge commands, where this has to
    distinguish a data line from a control line and be transparent to one of
    them only while the other permits it.
    """
    q = int(st or 0)
    if inb[1]:
        q = 1 if inb[0] else 0
    return (q,), q


def _bandpass_streams(rng, T, low, high, n_trials):
    """Schedules with a BALANCED spread of A->B gaps.

    The first version drew random pulse trains, which produced gaps 2 and 3
    thirty-odd times each but the two gaps that DEFINE the band - the accept
    edge at ``high`` and the reject edge at ``high + 1`` - only eight or nine
    times between them. A circuit could then score well by handling the common
    gaps and ignoring the edges, and that is exactly what evolution did: 1800
    generations reached 0.95 on training and 0.03 to 0.57 held out, memorising
    timings rather than learning the band.

    So every gap from 1 to ``high + 3`` gets equal billing, at varied phases and
    in varied order, with several pairs per trial.
    """
    span = list(range(1, high + 4))
    streams = []
    for index in range(n_trials):
        rows = [(0, 0)] * T
        order = list(span)
        rng.shuffle(order)
        tick = 2 + (index % 3)
        for gap in order:
            if tick + gap >= T - 1:
                break
            rows[tick] = (1, rows[tick][1])
            rows[tick + gap] = (rows[tick + gap][0], 1)
            # Clear of the next pair so one pair's B cannot be read against the
            # next pair's A.
            tick += gap + rng.randint(high + 2, high + 5)
        streams.append(rows)
    return streams


def gap_bandpass_oracle(seed=20260817, low=2, high=4):
    oracle = make_gap_bandpass(low, high)
    rng = random.Random(seed)
    T = 64
    trials = []
    for stream in _bandpass_streams(rng, T, low, high, 18):
        expected = label_trace(oracle, stream, T, 2)
        trials.append(Trial(
            stream, {'Q': expected},
            {'Q': [float(t) for t, v in enumerate(expected) if v == 1]}))
    # Degenerate inputs, which no balanced schedule guarantees.
    for name in ('lone_a', 'lone_b', 'together'):
        stream = [(0, 0)] * T
        for tick in (6, 20, 34, 48):
            if name == 'lone_a':
                stream[tick] = (1, 0)
            elif name == 'lone_b':
                stream[tick] = (0, 1)
            else:
                stream[tick] = (1, 1)
        expected = label_trace(oracle, stream, T, 2)
        trials.append(Trial(
            stream, {'Q': expected},
            {'Q': [float(t) for t, v in enumerate(expected) if v == 1]}))
    target = TemporalTarget(
        'Gap band-pass (oracle)', [(0, 1), (0, 3)],
        [OutputTerminal('Q', (2, 2))], T, trials, grid_size=5, iters=30,
        contract=event_contract(fit_latency=False), latency=2,
        category='Pulse width & duration',
        description=describe_target(
            'Emit one event when input B arrives %d to %d seconds after input '
            'A, and stay silent when it arrives sooner, later, or alone.'
            % (low, high),
            'Eighteen schedules give every gap from 1 to %d equal billing at '
            'varied phases and orders, so the accept edge at %d and the reject '
            'edge at %d are as well represented as the easy middle of the '
            'band. Lone-A, lone-B and simultaneous arrivals are tested '
            'explicitly.' % (high + 3, high, high + 1)))
    return target


def _divider_streams(rng, T, n_trials, divisor):
    """Dense count events with OCCASIONAL resets.

    ``sample_streams`` gives every input the same pulse rate, which is fatal
    here: with A and B equally likely, a run of ``divisor`` counts survives to
    completion about (1/2)**divisor of the time, so the schedule emits almost
    nothing. Measured on the first version - 2 expected events across 17 trials,
    15 of them completely silent - a target with no positive evidence in it.

    So the count input runs densely and the reset arrives every few counts,
    which is the duty cycle the device is actually for.
    """
    streams = []
    for index in range(n_trials):
        rows = [(0, 0)] * T
        spacing = 3 + (index % 3)
        tick = rng.randint(2, 4)
        counts = []
        while tick < T - 1:
            counts.append(tick)
            tick += spacing + rng.randint(0, 1)
        for tick in counts:
            rows[tick] = (1, 0)
        if index % 3 != 2 and len(counts) > divisor:
            pos = rng.randrange(divisor // 2, len(counts) - 1)
            reset_tick = counts[pos] + 1
            if 0 <= reset_tick < T and rows[reset_tick] == (0, 0):
                rows[reset_tick] = (0, 1)
        streams.append(rows)
    return streams


def resettable_divider_oracle(seed=20260817, divisor=4):
    oracle = make_resettable_divider(divisor)
    rng = random.Random(seed)
    T = 64
    trials = [
        Trial(stream, {'Q': label_trace(oracle, stream, T, 2)},
              {'Q': [float(t) for t, v in enumerate(
                  label_trace(oracle, stream, T, 2)) if v == 1]})
        for stream in _divider_streams(rng, T, 14, divisor)]
    target = TemporalTarget(
        'Resettable divide-by-%d (oracle)' % divisor, [(0, 1), (0, 3)],
        [OutputTerminal('Q', (2, 2))], T, trials, grid_size=5, iters=30,
        contract=event_contract(fit_latency=False), latency=2,
        category='Memory & state',
        description=describe_target(
            'Emit one event for every %d events on input A; an event on input '
            'B clears the running count without emitting.' % divisor,
            'Fourteen seeded schedules with dense counts and occasional '
            'resets, plus explicit runs of exactly %d, one short of %d, a '
            'reset landing mid-count, back-to-back full cycles, and '
            'reset-only silence.' % (divisor, divisor)))
    extra = []
    # Exactly one full cycle, then one short of a second: emits once only.
    run = [(0, 0)] * target.T
    for index, tick in enumerate(range(4, 4 + 3 * (2 * divisor - 1), 3)):
        if tick < target.T:
            run[tick] = (1, 0)
    extra.append(run)
    # A reset arriving mid-count must discard the partial count entirely.
    mid = [(0, 0)] * target.T
    for tick in (4, 7, 10):
        mid[tick] = (1, 0)
    mid[13] = (0, 1)
    for tick in range(16, 16 + 3 * divisor, 3):
        if tick < target.T:
            mid[tick] = (1, 0)
    extra.append(mid)
    # Resets alone: never emit.
    resets = [(0, 0)] * target.T
    for tick in (5, 15, 25, 35):
        resets[tick] = (0, 1)
    extra.append(resets)
    for stream in extra:
        expected = label_trace(oracle, stream, target.T, 2)
        target.trials.append(Trial(
            stream, {'Q': expected},
            {'Q': [float(t) for t, v in enumerate(expected) if v == 1]}))
    return target


def _d_latch_streams(rng, T, n_trials):
    """Held D levels with Enable windows - a level stimulus, not a pulse train.

    ``sample_streams`` emits single-tick pulses, which is the wrong stimulus for
    a level device: a transparent latch has to see D SUSTAINED across an enable
    window, and be tested on whether it keeps that value after the window
    closes. So this builds its own schedule.
    """
    streams = []
    for _ in range(n_trials):
        rows = [(0, 0)] * T
        data, tick = rng.randint(0, 1), 0
        segments = []
        while tick < T:
            span = rng.randint(4, 9)
            segments.append((tick, min(T, tick + span), data))
            data ^= 1 if rng.random() < 0.65 else 0
            tick += span
        enables = []
        tick = rng.randint(2, 5)
        while tick < T - 3:
            width = rng.randint(2, 4)
            enables.append((tick, min(T - 1, tick + width)))
            tick += width + rng.randint(3, 8)
        for start, end, value in segments:
            for t in range(start, end):
                rows[t] = (value, 0)
        for start, end in enables:
            for t in range(start, end):
                rows[t] = (rows[t][0], 1)
        # A physical transparent latch cannot sample a data transition at the
        # exact instant Enable falls.  The LUT implementation has two causal
        # stages (state and feedback), so give D the corresponding two-tick
        # setup/hold aperture around every closing edge.  Previously almost
        # every generated trial changed D at or next to that edge; an otherwise
        # exact latch then scored 0.555 because the oracle demanded zero-time
        # capture, while the same circuit scores 1.0 once this ordinary timing
        # requirement is respected.
        for _start, end in enables:
            stable_from = max(0, end - 2)
            held_data = rows[stable_from][0]
            for t in range(stable_from, min(T, end + 2)):
                rows[t] = (held_data, rows[t][1])
        streams.append(rows)
    return streams


def d_latch_oracle(seed=20260817):
    rng = random.Random(seed)
    T = 56
    inputs = [(0, 1), (0, 3)]
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for stream in _d_latch_streams(rng, T, 12):
        trials.append(Trial(
            stream, {'Q': label_trace(orc_d_latch, stream, T, 2)}))
    # Retention is the whole point: D changes AFTER the window closes and Q must
    # ignore it. A latch that is permanently transparent passes everything else
    # and fails exactly here.
    hold = [(0, 0)] * T
    for t in range(4, 9):
        hold[t] = (1, 1 if t < 7 else 0)      # load 1, then enable drops
    for t in range(9, 30):
        hold[t] = (0, 0)                       # D goes low; Q must stay high
    for t in range(30, 34):
        hold[t] = (0, 1)                       # re-enable: now Q follows to 0
    trials.append(Trial(hold, {'Q': label_trace(orc_d_latch, hold, T, 2)}))
    never = [(1, 0)] * T                       # D high, never enabled: stay low
    trials.append(Trial(never, {'Q': label_trace(orc_d_latch, never, T, 2)}))
    return TemporalTarget(
        'Gated D latch (oracle)', inputs, [out], T, trials,
        grid_size=5, iters=30, contract=state_contract(), latency=2,
        # LEVEL device: the nervous net's capacitively-coupled node cannot read
        # a held input at all (see nv-held-level-not-a-signal), so this is a LUT
        # target by construction rather than by preference.
        supported_backends=('lut',),
        category='Memory & state',
        description=describe_target(
            'Output Q follows input D while input E is high, and holds its last '
            'value when E is low.',
            'Twelve seeded schedules of held D levels and enable windows, plus '
            'an explicit retention test where D changes after the window closes '
            'and Q must ignore it, and a never-enabled trial that must stay low.'))


# -- rhythmic and pattern targets ------------------------------------------------
# One per substrate, each sited so the substrate's own physics is the reason the
# task is natural rather than incidental, and each legible when watched.

def label_multi(oracle, streams, T, latency, roles):
    """``label_trace`` for a MULTI-OUTPUT oracle.

    ``label_trace`` keeps only ``ob[0]``, silently discarding every output past
    the first. These targets are about the RELATIONSHIP between several
    outputs, so the whole tuple has to survive.
    """
    raw, state = [], None
    for tick in range(T):
        observed, state = oracle(streams[tick], state)
        raw.append(tuple(observed))
    shifted = [None] * latency + raw[:max(0, T - latency)]
    return {role: [None if row is None else int(row[index]) for row in shifted]
            for index, role in enumerate(roles)}


def make_rhythm_cascade():
    """Divide one clock three ways at once: every pulse, every 2nd, every 4th.

    FNV, and visually the point: three outputs beating at nested rates is a
    rhythm you can watch, not a truth table. Structurally it is a TOGGLE chain -
    the family that ships 18 catalogue entries and, until the constructive draw
    was unlocked, was expressed in 0.0% of grown bodies.

    Unlike the nervous ring, the three outputs need genuinely DIFFERENT
    structure (a tap, one toggle, two toggles), which is what a genome that
    names its parts individually is supposed to be good at.
    """
    def f(inb, st):
        count = int(st or 0)
        if not inb[0]:
            return (0, 0, 0), count
        count += 1
        return (1,
                1 if count % 2 == 0 else 0,
                1 if count % 4 == 0 else 0), count
    return f


def make_ring_pattern(period=3, phases=3):
    """One kick, then a travelling wave around N outputs, indefinitely.

    NERVOUS, and it asks for the exact behaviour that has been sabotaging every
    other target on this substrate. Measured: a two-cell nervous loop given ONE
    input edge fires every 2 ticks indefinitely, and emits the identical train
    whether the input is held 5 ticks or 40 - it latches on the first edge and
    cannot be switched off. For a latch that is a liability; for "pulse once,
    then pulsate" it is the mechanism.

    The outputs are taps on one circulating structure rather than three
    separate computations, which also side-steps the open multi-output defect:
    an organism that gets the ring turning gets every phase at once.
    """
    def f(inb, st):
        started, tick = st if st is not None else (False, 0)
        if inb[0]:
            started, tick = True, 0
        elif started:
            tick += 1
        out = [0] * phases
        if started and tick % period == 0:
            out[(tick // period) % phases] = 1
        return tuple(out), (started, tick)
    return f


def make_count_to(limit=4):
    """Count input pulses; hold the output HIGH once the count reaches ``limit``.

    LUT. Each pulse adds one, and the readout is a HELD LEVEL - exactly what
    this substrate does natively and what the nervous net physically cannot do
    at all, its node being capacitively coupled with no stable high. One
    output, so it does not depend on the unresolved multi-output defect.
    """
    def f(inb, st):
        count = int(st or 0)
        if inb[0]:
            count += 1
        return (1 if count >= limit else 0,), count
    return f


def rhythm_cascade_oracle(seed=20260818):
    # THE SEED MUST DO SOMETHING. `holdout_score` builds its validation set by
    # calling spec(seed=...), so a spec that ignores its seed hands back the
    # TRAINING schedules and certification becomes a no-op - held-out equals
    # train by construction, and a memorising circuit certifies. The first
    # version of this target did exactly that and reported "generalises
    # exactly" on every seed, which meant nothing at all.
    rng = random.Random(seed)
    T, latency = 72, 2
    roles = ('R1', 'R2', 'R4')
    outs = [OutputTerminal('R1', (2, 2)), OutputTerminal('R2', (2, 3)),
            OutputTerminal('R4', (3, 2))]
    oracle = make_rhythm_cascade()
    schedules = []
    # Steady clocks at several rates and phases. A cascade must divide whatever
    # it is handed, so the rate has to vary or a fixed delay chain would pass.
    periods = rng.sample([3, 4, 5, 6, 7, 8], 4)
    for period in periods:
        for phase in (rng.randint(1, 3), rng.randint(4, 6)):
            schedules.append([(1,) if t >= phase and (t - phase) % period == 0
                              else (0,) for t in range(T)])
    # An interrupted clock: the division must survive a gap without resetting.
    gap_period = rng.choice((4, 5))
    stop = rng.randint(26, 34)
    resume = stop + rng.randint(12, 18)
    rows = [(0,)] * T
    for t in list(range(3, stop, gap_period)) + list(
            range(resume, T - 1, gap_period)):
        rows[t] = (1,)
    schedules.append(rows)
    schedules.append([(0,)] * T)                  # silence
    built = []
    for rows in schedules:
        expected = label_multi(oracle, rows, T, latency, roles)
        built.append(Trial(
            rows, expected,
            {role: [float(t) for t, v in enumerate(expected[role]) if v == 1]
             for role in roles}))
    return TemporalTarget(
        'Rhythm cascade (oracle)', [(0, 2)], outs, T, built,
        grid_size=7, iters=30, contract=event_contract(fit_latency=False),
        latency=latency, category='Rhythm & cadence',
        description=describe_target(
            'Divide one input clock three ways at once: R1 on every pulse, R2 '
            'on every second pulse, R4 on every fourth.',
            'Eight steady clocks spanning four rates and two phases, plus an '
            'interrupted clock whose division must survive the gap, plus '
            'silence. A fixed delay chain cannot pass, because the rate '
            'varies between trials.'))


def ring_pattern_oracle(seed=20260818, period=3, phases=3):
    # See rhythm_cascade_oracle: a spec that ignores its seed makes its own
    # held-out set a copy of its training set.
    rng = random.Random(seed)
    T, latency = 64, 2
    roles = tuple('P%d' % (index + 1) for index in range(phases))
    outs = [OutputTerminal(role, (2, 2 + index))
            for index, role in enumerate(roles)]
    oracle = make_ring_pattern(period, phases)
    schedules = []
    # One kick, at a different tick each time. Nothing else ever arrives, so a
    # circuit that needs continued input scores nothing. The kick ticks are
    # drawn from the seed and SPREAD WIDELY: clustering them (the first version
    # used 3, 5, 8, 11, 14) lets a free-running oscillator that ignores the
    # input line up with all of them at once, which is phase-locking to the
    # clock rather than being started by the kick.
    kicks = sorted(rng.sample(range(2, T // 2), 5))
    for kick in kicks:
        rows = [(0,)] * T
        rows[kick] = (1,)
        schedules.append(rows)
    schedules.append([(0,)] * T)                  # never kicked: stay silent
    restart = [(0,)] * T                          # kicked twice: wave restarts
    first = rng.randint(3, 8)
    restart[first] = (1,)
    restart[first + rng.randint(24, 32)] = (1,)
    schedules.append(restart)
    built = []
    for rows in schedules:
        expected = label_multi(oracle, rows, T, latency, roles)
        built.append(Trial(
            rows, expected,
            {role: [float(t) for t, v in enumerate(expected[role]) if v == 1]
             for role in roles}))
    return TemporalTarget(
        'Ring pattern (oracle)', [(0, 2)], outs, T, built,
        grid_size=7, iters=30, contract=event_contract(fit_latency=False),
        latency=latency, category='Rhythm & cadence',
        description=describe_target(
            'A single input pulse starts a travelling wave: P1, P2 and P3 fire '
            'in turn, %d seconds apart, and keep going with no further input.'
            % period,
            'Five trials kick at different ticks, one never kicks at all and '
            'must stay silent, and one kicks twice so the wave restarts from '
            'the first phase.'))


def count_to_oracle(seed=20260818, limit=4):
    rng = random.Random(seed)
    T, latency = 56, 2
    oracle = make_count_to(limit)
    schedules = []
    for index in range(10):
        rows = [(0,)] * T
        spacing = 3 + (index % 4)
        tick = 2 + (index % 3)
        while tick < T - 1:
            rows[tick] = (1,)
            tick += spacing + rng.randint(0, 1)
        schedules.append(rows)
    # One short of the limit: must NEVER go high. This is the trial that a
    # circuit merely responding to input, rather than counting, fails.
    short = [(0,)] * T
    for step in range(limit - 1):
        short[4 + step * 5] = (1,)
    schedules.append(short)
    exact = [(0,)] * T                            # reaches it exactly, late
    for step in range(limit):
        exact[6 + step * 6] = (1,)
    schedules.append(exact)
    schedules.append([(0,)] * T)                  # silent: never high
    built = [Trial(rows, {'Q': label_trace(oracle, rows, T, latency)})
             for rows in schedules]
    return TemporalTarget(
        'Count to %d (oracle)' % limit, [(0, 2)],
        [OutputTerminal('Q', (2, 2))], T, built, grid_size=5, iters=30,
        contract=state_contract(), latency=latency,
        # LEVEL readout: the nervous node cannot hold one at all.
        supported_backends=('lut',),
        category='Memory & state',
        description=describe_target(
            'Each input pulse adds one to a stored count; the output goes high '
            'once the count reaches %d, and stays high.' % limit,
            'Ten seeded pulse trains at varied spacing, plus a run stopping '
            'one short of the limit that must never go high, a run reaching it '
            'exactly, and silence.'))


ORACLE_SPECS = {
    'Rhythm cascade (oracle)':    rhythm_cascade_oracle,
    'Ring pattern (oracle)':      ring_pattern_oracle,
    'Count to 4 (oracle)':        count_to_oracle,
    'Gap band-pass (oracle)':     gap_bandpass_oracle,
    'Resettable divide-by-4 (oracle)': resettable_divider_oracle,
    'Gated D latch (oracle)':     d_latch_oracle,
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
