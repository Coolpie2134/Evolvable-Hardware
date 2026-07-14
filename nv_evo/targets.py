"""
nv_evo/targets.py — temporal targets for the hex nervous net.

A TemporalTarget scores behavior over time instead of a truth table.  Each
Trial drives a stimulus stream and may define raw expected point events,
sampled state/persistence windows, or a target-specific cadence invariant.

Every preset carries SEVERAL trials with different pulse timings. A net that
merely matches one fixed schedule (a lucky delay chain) fails the shifted
trials; only genuine state — a loop holding a circulating value — passes all
of them. That is what makes these targets select for memory.

The nervous backend also runs the combinational targets registered in
snn_evo.targets (gates, adders, custom tables); those are plain data objects
passed in by the GUI, so nothing here needs to import them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Pos = Tuple[int, int]


@dataclass
class OutputTerminal:
    role: str            # unique label, e.g. "Q"
    pos:  Pos            # terminal cell the output is read nearest to


@dataclass
class Trial:
    streams:  List[Tuple[int, ...]]                 # streams[t] = input bits at tick t
    expected: Dict[str, List[Optional[int]]]        # role -> expected bit per tick (None = don't score)
    # Optional ground-truth point events.  Event-semantic targets populate this
    # directly so consecutive events are not confused with one held-high run.
    expected_events: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class TemporalTarget:
    name:            str
    inputs:          List[Pos]
    outputs:         List[OutputTerminal]
    T:               int
    trials:          List[Trial]
    grid_size:       int = 7
    iters:           int = 30      # safety CAP — growth stops at its attractor
    output_strategy: str = "terminals"
    temporal:        bool = True          # marker so the GUI/GA can dispatch
    description:     str = ''             # human explanation, shown in the GUI
    # Scoring is semantic: ``events`` matches point events, ``cadence`` measures
    # a free-running rhythm, legacy ``trace`` retains persistence-window
    # semantics, and ``period_stepper`` measures its command/cadence invariant.
    score_mode:      str = 'trace'
    event_tolerance: float = 0.5
    event_max_shift: float = 12.0
    max_events:      int = 2048
    # Nominal input->output latency baked into the expected traces (oracle
    # `latency`). Kept as explicit metadata so semantic scoring does not have to
    # rediscover it; the fitted alignment is a SEPARATE, additional offset.
    latency:         int = 0
    cadence_period:  float = 0.0
    cadence_tolerance: float = 0.5
    cadence_settle:  float = 5.0
    cadence_min_events: int = 4
    stepper_min_period: int = 2
    stepper_max_period: int = 6
    stepper_settle: int = 2
    stepper_min_events: int = 4
    stepper_max_delay: int = 8

    @property
    def n_inputs(self):  return len(self.inputs)
    @property
    def n_outputs(self): return len(self.outputs)
    # so interactive playback can drive it like any nervous target
    @property
    def cases(self):     return []
    high = 1.0


# ── trace-building helpers ──────────────────────────────────────────────────────

# Unscored grace ticks after an input event: the output has this long to
# respond (and settle) before it is scored, so a valid circuit is not required
# to react at one exact tick — a Set that takes 2 ticks and a Reset that takes
# 2-5 are all accepted. Too small over-penalises legitimate response-latency
# variation (measured: raising 3 -> 5 lifted the SR-latch ceiling ~0.96 -> ~0.98
# by no longer marking slightly-slow resets wrong); too large leaves too few
# scored ticks. 5 is the balance for the T=20..24 traces here.
SETTLE = 5


def describe_target(goal, scoring, tests):
    """Consistent three-part description shown by every GUI/report."""
    return 'Goal: %s\nScoring: %s\nTests: %s' % (
        goal.strip(), scoring.strip(), tests.strip())


EVENT_SCORING = (
    'Match output edges one-to-one. One latency offset is shared '
    'by every test; missing and extra edges both reduce fitness.')
CADENCE_SCORING = (
    'Measure sustained inter-edge cadence and coverage. Startup latency and '
    'absolute phase are free; pre-trigger activity is penalized.')
PERSISTENCE_SCORING = (
    'Compare active and quiet persistence windows after a settling interval. '
    'Recurrent ringing is a valid stored 1.')


def _pulse_streams(T, n_inputs, pulses):
    """streams[t] for `pulses` = {input_index: [ticks it is 1]}."""
    ons = {i: set(ts) for i, ts in pulses.items()}
    return [tuple(1 if t in ons.get(i, ()) else 0 for i in range(n_inputs))
            for t in range(T)]


def _hold_trace(T, events):
    """Expected trace from `events` = [(tick, level)]: after each event's tick
    + SETTLE the output must hold `level` until the next event. Ticks inside a
    settle window are None (unscored)."""
    exp = [None] * T
    events = sorted(events) + [(T, None)]
    for (tick, level), (nxt, _) in zip(events, events[1:]):
        start = 0 if tick == 0 else tick + SETTLE
        for t in range(start, min(nxt, T)):
            exp[t] = level
    return exp


# ── general spike-event target builder ──────────────────────────────────────────

def spike_target(name, cases, T, n_inputs=None, output_role='Q', latency=1,
                 inputs=None, out_pos=(2, 2), grid_size=5, iters=30,
                 description=''):
    """Describe a temporal function purely as SPIKE EVENTS — the easy path to a
    new target. Each case in `cases` is ``(input_spikes, output_spikes)``:

        input_spikes  — {input_index: [ticks]}, or a list-of-lists (one ticks
                        list per input), giving the ticks each input pulses on.
        output_spikes — the list of ticks the output should fire on.

    Expected outputs are stored as point-event timestamps. A net that stays
    silent is penalised for every missing event and a net that fires spuriously
    is penalised for every extra one (one-to-one event F1). The first `latency`
    ticks remain unscored startup grace for compatibility with the trace view.

    `latency` is now just a nominal minimum-causal offset (default 1), NOT a delay
    the circuit must match: scoring is latency-invariant (nv_evo.temporal, best
    global continuous-time shift), so a circuit that produces the right spikes
    at ANY consistent delay scores the same. Describe the RELATIVE event
    structure; the absolute input->output delay is free.

    Example — a coincidence detector (fires one tick after A and B coincide)::

        spike_target('Coincidence', [
            ({0: [20],     1: [20]},     [21]),   # A & B on tick 20 -> spike @21
            ({0: [5],      1: [12]},     []),     # never coincide   -> silent
            ({0: [8],      1: []},       []),     # lone A           -> silent
            ({0: [3, 15],  1: [3, 15]},  [4, 16]),# coincide twice   -> two spikes
        ], T=24, n_inputs=2)
    """
    def _norm(input_spikes):
        if isinstance(input_spikes, dict):
            return {int(i): list(ts) for i, ts in input_spikes.items()}
        return {i: list(ts) for i, ts in enumerate(input_spikes)}

    norm_cases = [(_norm(isp), set(osp)) for isp, osp in cases]
    if n_inputs is None:
        n_inputs = max([0] + [i + 1 for isp, _ in norm_cases for i in isp])
    if inputs is None:
        inputs = [(0, min(grid_size - 1, 1 + 2 * i)) for i in range(n_inputs)]
    out = OutputTerminal(output_role, out_pos)
    trials = []
    for pulse_dict, out_set in norm_cases:
        streams = _pulse_streams(T, n_inputs, pulse_dict)
        exp = [1 if t in out_set else (None if t < latency else 0)
               for t in range(T)]
        trials.append(Trial(streams, {output_role: exp},
                            {output_role: sorted(float(t) for t in out_set)}))
    return TemporalTarget(name, list(inputs), [out], T, trials,
                          grid_size=grid_size, iters=iters,
                          score_mode='events', event_tolerance=0.5,
                          description=description or describe_target(
        'Produce exactly the requested output-edge pattern.', EVENT_SCORING,
        'Each supplied case contributes its input schedule, required edges, and '
        'the surrounding silence window.'))


# ── preset temporal targets ─────────────────────────────────────────────────────
# Inputs/outputs kept close together (nervous nets need short signal paths).

# Trial banks are deliberately DIVERSE in pulse count, spacing and gap parity:
# with only a couple of schedules the GA memorises the timings (measured: a
# "solved" toggle locked up on pulse gaps it never saw). A net can only score
# 1.0 across the whole bank by implementing the actual rule.

def sr_latch(grid_size=5):
    """Set/Reset latch: a Set pulse drives Q to 1 (and it holds); a Reset pulse
    drives it to 0. Five trials with shifted timings and mixed set->reset gap
    parities — including a long hold with no reset — so only a real, timing-
    independent latch scores 1.0."""
    T = 20
    Set, Reset = (0, 1), (0, 3)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for set_t, reset_t in ((2, 10), (5, 14), (3, 9), (5, 12), (2, None)):
        pulses = {0: [set_t]}
        events = [(0, 0), (set_t, 1)]
        if reset_t is not None:
            pulses[1] = [reset_t]
            events.append((reset_t, 0))
        trials.append(Trial(_pulse_streams(T, 2, pulses),
                            {'Q': _hold_trace(T, events)}))
    return TemporalTarget('SR latch', [Set, Reset], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          description=describe_target(
        'Input A sets one stored bit; input B resets it. Q must remain in the '
        'selected state without continued input.', PERSISTENCE_SCORING,
        'Five Set/Reset schedules vary event phase and hold length, including a '
        'Set-only test that must retain the bit through the horizon.'))


def _toggle_trial(T, pulses):
    events, q = [(0, 0)], 0
    for p in pulses:
        q ^= 1
        events.append((p, q))
    return Trial(_pulse_streams(T, 1, {0: pulses}), {'Q': _hold_trace(T, events)})


def toggle_ff(grid_size=5):
    """T flip-flop: each input pulse flips the output (period-2 memory).
    Six trials spanning 2/3-pulse schedules with odd AND even gaps at varied
    phases — a phase-locked ring that only toggles for one spacing fails."""
    T = 24
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    banks = ([3, 9, 14], [4, 12], [2, 7, 12], [3, 10, 15], [5, 11], [2, 8, 15])
    return TemporalTarget('Toggle flip-flop', [In], [out], T,
                          [_toggle_trial(T, p) for p in banks],
                          grid_size=grid_size, iters=30,
                          description=describe_target(
        'Each input edge flips the stored bit: quiet to active, then active to '
        'quiet.', PERSISTENCE_SCORING,
        'Six pulse trains vary phase, count, and odd/even spacing so a '
        'phase-locked shortcut cannot pass.'))


def oscillator(grid_size=5, period=2):
    """Kicked oscillator: a startup pulse injects a value into a loop, which then
    rings on its own (no input needed to sustain — but, correctly, an input IS
    needed to start it: nothing comes from nothing). Output should thereafter
    keep toggling; the exact phase is not scored (only that Q alternates and is
    never stuck), so any circulating loop of the right period qualifies.

    Two trials (kick at different ticks) so a fixed one-shot pulse chain that
    just happens to blip once can't pass — only a genuinely ringing loop does.
    """
    T = 20
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for kick in (0, 3, 5):
        streams = _pulse_streams(T, 1, {0: [kick]})
        # after the kick + settle, require a strict alternation of some phase.
        # score BOTH phases loosely by leaving the absolute phase free: we mark
        # expected as the alternation that starts 1 at the first scored tick;
        # the balanced GA scorer rewards matching the toggling, and a stuck
        # output (all 0 or all 1) can match at most half.
        start = kick + SETTLE + 1
        exp = [None] * T
        phase = 0
        for t in range(start, T):
            exp[t] = 1 if (phase // (period // 2)) % 2 == 0 else 0
            phase += 1
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('Oscillator (period %d)' % period, [In], [out], T,
                          trials, grid_size=grid_size, iters=30,
                          score_mode='cadence', cadence_period=float(period),
                          cadence_settle=float(SETTLE),
                          description=describe_target(
        'One input edge starts a free-running output cadence with period %d.'
        % period, CADENCE_SCORING,
        'Three kick times require silence before the trigger and sustained '
        'oscillation through the end; a finite burst fails.'))


def pattern_generator(grid_size=5, pattern=(1, 0, 0, 0)):
    """Kicked pattern generator (paper §3: "simple pattern generation circuits
    can be built from these circuits, connected in loops"): one kick pulse must
    start the output repeating `pattern` indefinitely — a loop whose length and
    loading encode the bit sequence.

    The honeycomb is BIPARTITE — every edge joins an (x+y)-even node to an odd
    one (see hexgrid.hex_dirs), so every cycle has even length and a single
    circulating pulse can only produce an EVEN period. An ODD-period pattern
    like 100 is therefore geometrically out of reach on the cheap one-pulse
    route: it would need a SECOND pulse injected half a loop away (output_period
    = loop_length / n_pulses), a conjunction the GA path dips through lower
    fitness to reach and empirically never crosses — neither a bigger grid nor
    200 generations moved it, it just parked on local optima (a one-pulse
    period-6 loop, F1 ~0.67; a 3-spike burst that then dies, F1 ~0.76). So the
    default pattern is 1000: period 4, a single pulse circulating a length-4
    loop read at one cell, which the parity-legal route reaches directly. The
    kick tick and absolute phase are free (phase/latency-invariant scoring)."""
    T = 28
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for kick in (2, 5, 7):
        streams = _pulse_streams(T, 1, {0: [kick]})
        exp = [None] * T
        for i, t in enumerate(range(kick + SETTLE + 1, T)):
            exp[t] = pattern[i % len(pattern)]
        trials.append(Trial(streams, {'Q': exp}))
    pat = ''.join(map(str, pattern))
    cyclic_rises = [i for i, bit in enumerate(pattern)
                    if bit and not pattern[i - 1]]
    # One pulse per cycle is a pure cadence invariant.  Patterns with several
    # uneven pulse offsets still need their richer phase-relative trace scorer.
    mode = 'cadence' if len(cyclic_rises) == 1 else 'trace'
    return TemporalTarget('Pattern (%s)' % pat, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          score_mode=mode, cadence_period=float(len(pattern)),
                          cadence_settle=float(SETTLE),
                          description=describe_target(
        'One input edge starts the repeating output pattern %s (period %d).'
        % (pat, len(pattern)),
        CADENCE_SCORING if mode == 'cadence' else
        'Compare the repeating relative pattern after startup; absolute phase '
        'and latency are free.',
        'Three kick times require pre-trigger silence and repetition through the '
        'full observation window.'))


def echo(grid_size=5, delay=3):
    """Echo: output reproduces the input pulse train `delay` ticks later — the
    simplest temporal target (a delay line), a stepping stone to real memory.
    Four pulse trains with varied spacing.  Distinct pulse edges are separated
    by at least one low tick; adjacent high samples would be one held pulse in
    the physical input model, not two events."""
    T = 20
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for pulses in ([2, 7, 9, 13], [4, 10, 16], [3, 5, 9, 14], [6, 12, 14]):
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None] * delay + [streams[t - delay][0] for t in range(delay, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(p + delay) for p in pulses
                                   if p + delay < T]}))
    return TemporalTarget('Echo (delay %d)' % delay, [In], [out], T, trials,
                          grid_size=grid_size, iters=30, score_mode='events',
                          description=describe_target(
        'Reproduce the input-edge train at Q while preserving every inter-edge '
        'interval.', EVENT_SCORING,
        'Four schedules vary pulse count and spacing. The nominal %d-tick '
        'label sets the reference trace; absolute circuit latency remains free.'
        % delay))


def coincidence_detector(grid_size=5, latency=1):
    """Two-input coincidence detector — the paper's marquee node capability
    lifted to a circuit: output pulses iff BOTH inputs pulse at the same tick.
    Trials mix simultaneous pairs (fire), pulses staggered by 1-2 ticks (must
    NOT fire — the async coincidence window is what enforces this physically),
    and lone single-channel pulses (must not fire)."""
    T = 20
    A, B = (0, 1), (0, 3)
    out  = OutputTerminal('Q', (2, 2))
    # (pulses_on_A, pulses_on_B) per trial
    banks = [
        ([3, 9, 14], [3, 9, 14]),          # all coincident -> fire each time
        ([3, 10],    [4, 10]),             # stagger 1 then coincide
        ([5, 12],    [7, 13]),             # never coincident
        ([2, 8, 13], [2, 9, 13]),          # coincide, miss, coincide
        ([4],        []),                  # lone pulses -> silence
        ([],         [6]),
    ]
    trials = []
    for pa, pb in banks:
        streams = _pulse_streams(T, 2, {0: pa, 1: pb})
        exp = [None] * latency + [
            1 if (streams[t - latency][0] and streams[t - latency][1]) else 0
            for t in range(latency, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(t + latency)
                                   for t in range(T - latency)
                                   if streams[t][0] and streams[t][1]]}))
    return TemporalTarget('Coincidence (2-in)', [A, B], [out], T, trials,
                          grid_size=grid_size, iters=30, score_mode='events',
                          description=describe_target(
        'Emit one Q edge only when inputs A and B arrive together.',
        EVENT_SCORING,
        'Six schedules mix coincident pairs, one- and two-tick offsets, and '
        'single-input events. Offset and lone events must remain silent.'))


def one_shot(grid_size=5, width=3, latency=3):
    """One-shot / monostable: each input pulse triggers a fixed `width`-tick
    burst at the output, which must then self-terminate — a loop that loads
    itself AND cuts itself off (delayed self-inhibition). Hold windows are
    phase-tolerant (a ringing burst counts); the silence after must be real.
    One unscored tick after each burst tolerates the turn-off transient."""
    T = 24
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for pulses in ([3], [2, 11], [5, 14], [4, 12, 20]):   # spacing >= width + 4
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None if t < latency else 0 for t in range(T)]
        for p in pulses:
            for t in range(p + latency, min(p + latency + width, T)):
                exp[t] = 1
            if p + latency + width < T:
                exp[p + latency + width] = None            # turn-off transient
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('One-shot (%d ticks)' % width, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          description=describe_target(
        'Each input edge starts a self-terminating active interval lasting %d '
        'ticks, after which Q must return quiet.' % width,
        PERSISTENCE_SCORING,
        'Four schedules include isolated and repeated triggers. Every interval '
        'must terminate and the circuit must re-arm for later input.'))


def pair_detector(grid_size=5, gap=2, latency=3):
    """Double-pulse detector: output fires iff two input pulses arrive exactly
    `gap` ticks apart (out at second pulse + latency) — a delay line feeding a
    coincidence node, timing used as computation. Wrong gaps and lone pulses
    must stay silent."""
    T = 22
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    banks = [
        [3, 5],                 # gap 2 -> fire
        [4, 8],                 # gap 4 -> silent
        [2, 4, 11, 13],         # two gap-2 pairs -> fire twice
        [6],                    # lone pulse -> silent
        [3, 4, 9, 11],          # gap 1 (no) then gap 2 (yes)
    ]
    trials = []
    for pulses in banks:
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None] * (latency + gap) + [
            1 if (streams[t - latency][0] and streams[t - latency - gap][0]) else 0
            for t in range(latency + gap, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(t + latency)
                                   for t in range(gap, T - latency)
                                   if streams[t][0] and streams[t - gap][0]]}))
    return TemporalTarget('Pair detector (gap %d)' % gap, [In], [out], T, trials,
                          grid_size=grid_size, iters=30, score_mode='events',
                          description=describe_target(
        'Emit Q when two input edges are separated by exactly %d ticks.' % gap,
        EVENT_SCORING,
        'Five schedules include valid pairs, wrong gaps, multiple pairs, and a '
        'single edge. Only correctly spaced pairs may produce output.'))


# ── more complex temporal functions (point-event targets; Nervous + LUT) ────────
# Built straight from spike_target: each is a handful of (input_spikes,
# output_spikes) cases, so the behaviour is transparent and easy to tweak. Cases
# mix positive and negative examples at varied timings so a net can't pass by
# memorising one schedule. These point-event targets penalise both missing and
# spurious output edges.

def temporal_xor(grid_size=5, latency=1):
    """Temporal XOR — the complement of coincidence: fire iff EXACTLY ONE of the
    two inputs pulses on a tick (both-or-neither -> silent). Needs each input to
    excite the output while the pair mutually inhibits."""
    T = 22
    return spike_target('Temporal XOR (2-in)', [
        ({0: [4],        1: []},        [4 + latency]),          # A only  -> fire
        ({0: [],         1: [7]},       [7 + latency]),          # B only  -> fire
        ({0: [11],       1: [11]},      []),                     # both    -> silent
        ({0: [3, 15],    1: [8, 15]},   [3 + latency, 8 + latency]),  # A,B,then both
        ({0: [5, 9],     1: [9, 18]},   [5 + latency, 18 + latency]), # overlap @9 -> silent
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Emit one Q edge when exactly one of A or B arrives; simultaneous A+B '
        'must cancel.', EVENT_SCORING,
        'Five schedules cover A-only, B-only, simultaneous inputs, and mixed '
        'trains with both positive and suppressed events.'))


def ordered_sequence(grid_size=5, gap=3, latency=1):
    """Ordered two-input sequence detector: fire only when A pulses and THEN B
    pulses exactly `gap` ticks later (B-before-A, wrong gaps and lone pulses stay
    silent). Order matters — a delay line on A must meet B at a coincidence node,
    so the reverse order misses."""
    T = 24
    return spike_target('Sequence A->B (gap %d)' % gap, [
        ({0: [4],        1: [4 + gap]},     [4 + gap + latency]),   # A then B  -> fire
        ({0: [10 + gap], 1: [10]},          []),                    # B then A  -> silent
        ({0: [6],        1: [6 + gap + 2]}, []),                    # gap wrong -> silent
        ({0: [3, 14],    1: [3 + gap, 14 + gap]},
                                       [3 + gap + latency, 14 + gap + latency]),
        ({0: [8],        1: []},            []),                    # lone A    -> silent
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Emit Q only for the ordered sequence A then B, separated by %d ticks.'
        % gap, EVENT_SCORING,
        'Five schedules include correct order, reverse order, wrong spacing, '
        'repeated valid sequences, and an incomplete sequence.'))


def veto_gate(grid_size=5, latency=1):
    """Inhibited echo: output echoes input A after `latency` ticks UNLESS input B
    pulses on the same tick, which vetoes that echo. The inhibitory routing used
    as a real gate — 'pass A, but B can suppress it'."""
    T = 22
    return spike_target('Veto gate (B blocks A)', [
        ({0: [3, 9, 15],  1: []},        [3 + latency, 9 + latency, 15 + latency]),
        ({0: [4, 10],     1: [10]},      [4 + latency]),          # B blocks the 2nd
        ({0: [6],         1: [6]},       []),                     # B blocks the only A
        ({0: [5, 12, 18], 1: [12]},      [5 + latency, 18 + latency]),
        ({0: [],          1: [7, 14]},   []),                     # B alone -> nothing
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Pass each A edge to Q unless B arrives simultaneously; B alone must '
        'produce nothing.', EVENT_SCORING,
        'Five schedules mix unopposed A, vetoed A+B, and B-only events.'))


def burst_generator(grid_size=5, n=3, spacing=2, latency=1):
    """Fan-out / burst: a single input kick produces a fixed BURST of `n` evenly
    spaced output spikes, then silence until the next kick. One edge in, several
    edges out — a delay-line tap or a short re-triggerable ring."""
    T = 22
    def burst(k):
        return [k + latency + i * spacing for i in range(n)]
    return spike_target('Burst x%d' % n, [
        ({0: [3]},         burst(3)),
        ({0: [5]},         burst(5)),
        ({0: [2, 13]},     burst(2) + burst(13)),   # re-triggers
        ({0: [4]},         burst(4)),
    ], T=T, n_inputs=1, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Convert each input edge into %d Q edges separated by %d ticks, then '
        'return to silence.' % (n, spacing), EVENT_SCORING,
        'Four schedules vary trigger time and include a two-trigger test that '
        'requires the generator to re-arm.'))


def divide_by_3(grid_size=5, latency=1):
    """Divide-by-3 counter: the output fires on every THIRD input pulse and stays
    silent on the other two — a modulo-3 counter, harder than the toggle (÷2)
    because it needs two bits of state, not one."""
    T = 26
    def every3(pulses):
        return [pulses[i] + latency for i in range(2, len(pulses), 3)]
    trains = [[3, 6, 9, 12, 15, 18], [2, 5, 8, 11, 14, 17, 20], [4, 8, 12, 16, 20]]
    return spike_target('Divide-by-3', [({0: tr}, every3(tr)) for tr in trains],
                        T=T, n_inputs=1, latency=latency, grid_size=grid_size,
                        description=describe_target(
        'Emit Q on every third input edge and remain silent on the other two.',
        EVENT_SCORING,
        'Three trains vary phase, spacing, and length so the circuit must retain '
        'the modulo-3 count rather than memorize one schedule.'))


# ── registry: one entry per function, best-measuring style (metric shootout) ───
# Judged on HELD-OUT schedules (fresh random timings), oracle-trained circuits
# generalise far better for most input-driven functions (echo hand-trained:
# held-out 0.55 vs oracle 1.00; latch 0.74 vs 0.93) — so those use the oracle
# spec. Coincidence measured BETTER hand-built (held-out f1 0.94 vs 0.68), so it
# stays hand-built. Oscillator / Pattern are autonomous behaviours (no
# input->output relation to sample) and stay hand-built by necessity.
TEMPORAL_TARGETS = {
    'Oscillator':            oscillator(),
    'Pattern (1000)':        pattern_generator(),
    'Coincidence (2-in)':    coincidence_detector(),
    'Temporal XOR (2-in)':   temporal_xor(),
    'Sequence A->B':         ordered_sequence(),
    'Veto gate':             veto_gate(),
    'Burst x3':              burst_generator(),
    'Divide-by-3':           divide_by_3(),
}

# Registry display name -> its reference-oracle spec name (in oracle.ORACLE_SPECS).
# Exposed so held-out certification can recover the reference state machine that
# defines a given target and re-sample fresh validation schedules from it.
ORACLE_KEY_TO_SPEC = {
    'SR latch':              'SR latch (oracle)',
    'C-element (2-in join)': 'C-element (oracle)',
    'Refractory filter (3 ticks)': 'Refractory filter (oracle)',
    'A-first rendezvous':     'A-first rendezvous (oracle)',
    'Collision serializer (2-to-1)': 'Collision serializer (oracle)',
    'Watchdog timeout (5 ticks)': 'Watchdog timeout (oracle)',
    'Toggle flip-flop':      'Toggle (oracle)',
    'Echo (delay 3)':        'Echo (oracle)',
    'One-shot (5 ticks)':    'One-shot (oracle)',
    'Period doubler (2x)':   'Period doubler (oracle)',
    'Pair detector (gap 2)': 'Pair detector (oracle)',
    'Period stepper':        'Period stepper (oracle)',
    'Gated oscillator':      'Gated oscillator (oracle)',
    'Resettable toggle':     'Resettable toggle (oracle)',
}


def _register_oracle_targets():
    import dataclasses
    from .oracle import ORACLE_SPECS
    for key, spec_name in ORACLE_KEY_TO_SPEC.items():
        t = ORACLE_SPECS[spec_name]()
        TEMPORAL_TARGETS[key] = dataclasses.replace(t, name=key)

_register_oracle_targets()
