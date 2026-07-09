"""
nv_evo/targets.py — temporal targets for the hex nervous net.

A TemporalTarget scores an output *trace* over time instead of a truth table:
each Trial drives the inputs with a tick-by-tick stream and scores the output
against an expected trace (None = don't score that tick, e.g. settle windows).

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

    Every tick from `latency` onward is scored: a spike is expected exactly at
    the output ticks and silence everywhere else. So a net that stays silent is
    penalised for every missing spike and a net that fires spuriously is
    penalised for every extra one (the spike-event F1 metric). The first
    `latency` ticks are unscored startup grace.

    `latency` is now just a nominal minimum-causal offset (default 1), NOT a delay
    the circuit must match: scoring is latency-invariant (nv_evo.temporal, best
    global shift), so a circuit that produces the right spikes at ANY consistent
    delay scores the same. Describe the RELATIVE spike structure; the absolute
    input->output delay is free.

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
        trials.append(Trial(streams, {output_role: exp}))
    return TemporalTarget(name, list(inputs), [out], T, trials,
                          grid_size=grid_size, iters=iters,
                          description=description or (
        'Custom spike-event target: fire at the specified output ticks and stay\n'
        'silent otherwise. Scored by spike-event precision/recall (F1).'))


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
                          grid_size=grid_size, iters=30, description=(
        'One bit of memory. A pulse on Set (input A) drives Q high and it must\n'
        'HOLD with no further input — the bit is stored as a pulse circulating\n'
        'in a loop, so sustained ringing counts as holding. A pulse on Reset\n'
        '(input B) must break the circulation and leave Q silent.\n'
        'Trials shift the Set/Reset timing (one has no Reset at all: hold\n'
        'forever), so only a real, timing-independent latch scores 1.0.'))


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
                          grid_size=grid_size, iters=30, description=(
        'Each input pulse FLIPS the stored bit: pulse -> Q rings, next pulse ->\n'
        'Q silent, and so on (a frequency divider). Needs a loop the input can\n'
        'both load and clear, depending on its current state.\n'
        'Six schedules with odd AND even pulse gaps at varied phases — a\n'
        'phase-locked ring that only toggles for one spacing fails.'))


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
                          trials, grid_size=grid_size, iters=30, description=(
        'A single kick pulse must start a free-running alternation: the pulse\n'
        'circulates a loop of buffers forever, read as Q toggling every tick.\n'
        'Correctly, NOTHING may happen before the kick (no input -> no output);\n'
        'trials kick at different ticks so a one-shot blip chain cannot pass.'))


def pattern_generator(grid_size=5, pattern=(1, 0, 0)):
    """Kicked pattern generator (paper §3: "simple pattern generation circuits
    can be built from these circuits, connected in loops"): one kick pulse must
    start the output repeating `pattern` indefinitely — a loop whose length and
    loading encode the bit sequence.

    Period 3 is deliberately KEPT though it is a hard, DECEPTIVE target. The
    lattice is bipartite, so a single pulse only circulates an even-length loop
    (period = loop length); an odd period is achievable only via the harder route
    output_period = loop_length / n_pulses — a length-6 loop carrying TWO pulses
    spaced 3 apart. The cheap one-pulse route parity forbids, so the GA gets stuck
    on strong LOCAL OPTIMA and never crosses to the real solution (measured, and
    neither a bigger grid nor 200 generations moves it):
      * a one-pulse period-6 loop  -> output 100000, F1 ~0.67
      * a transient 3-spike burst that then dies -> F1 ~0.76
    Reaching period 3 needs a fork that injects a SECOND pulse half a loop later
    into a sustaining loop — a conjunction the path to which dips through lower
    fitness. The fix is search-side (behavioural diversity / quality-diversity to
    keep the period-6 stepping stone AND explore off it), NOT a target change. The
    kick tick and absolute phase are free (phase/latency-invariant scoring)."""
    T = 24
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
    return TemporalTarget('Pattern (%s)' % pat, [In], [out], T, trials,
                          grid_size=grid_size, iters=30, description=(
        'Like the oscillator, but richer: after a single kick pulse the output\n'
        'must repeat the bit pattern %s forever (period %d) — a pulse circulating\n'
        'a loop of length %d read at one cell. The absolute phase/latency is free;\n'
        'only the repeating structure is scored. Nothing may happen before the\n'
        'kick; trials kick at different ticks.' % (pat, len(pattern), len(pattern))))


def echo(grid_size=5, delay=3):
    """Echo: output reproduces the input pulse train `delay` ticks later — the
    simplest temporal target (a delay line), a stepping stone to real memory.
    Four pulse trains with varied spacing (incl. adjacent pulses)."""
    T = 20
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for pulses in ([2, 7, 8, 13], [4, 10, 16], [3, 5, 9, 14], [6, 12, 13]):
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None] * delay + [streams[t - delay][0] for t in range(delay, T)]
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('Echo (delay %d)' % delay, [In], [out], T, trials,
                          grid_size=grid_size, iters=30, description=(
        'Output must reproduce the input pulse train EXACTLY %d ticks later —\n'
        'a pure delay line (each hop through a node costs one tick, so the\n'
        'signal path must be exactly %d hops). The simplest temporal target;\n'
        'trials use varied spacings including back-to-back pulses.' % (delay, delay)))


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
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('Coincidence (2-in)', [A, B], [out], T, trials,
                          grid_size=grid_size, iters=30, description=(
        'Output pulses if and only if BOTH inputs pulse on the same tick —\n'
        'the nervous node\'s marquee capability (E1 AND E2 edge coincidence).\n'
        'Pulses staggered by 1-2 ticks and lone single-input pulses must stay\n'
        'silent: the async coincidence window enforces this physically.'))


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
                          grid_size=grid_size, iters=30, description=(
        'A monostable: each input pulse triggers a burst of activity at Q that\n'
        'lasts %d ticks and must then SHUT ITSELF OFF — a loop that loads\n'
        'itself and, after a delay, inhibits itself. The first target where the\n'
        'inhibitory veto is constructive rather than just a reset line.\n'
        'Bursts count phase-tolerantly (ringing is fine); the silence after\n'
        'each burst must be real, and it must re-trigger for later pulses.' % width))


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
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('Pair detector (gap %d)' % gap, [In], [out], T, trials,
                          grid_size=grid_size, iters=30, description=(
        'Fires only when two input pulses arrive EXACTLY %d ticks apart: a\n'
        'delay line feeds one arm of a coincidence node and the direct path\n'
        'feeds the other, so the second pulse of a correctly-spaced pair meets\n'
        'the delayed first pulse — timing used as computation. Wrong gaps and\n'
        'lone pulses must stay silent.' % gap))


# ── more complex temporal functions (spike-event described; Nervous + LUT) ──────
# Built straight from spike_target: each is a handful of (input_spikes,
# output_spikes) cases, so the behaviour is transparent and easy to tweak. Cases
# mix positive and negative examples at varied timings so a net can't pass by
# memorising one schedule. Scored by the spike-event F1 metric like everything
# else (silence and spurious firing both cost).

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
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size, description=(
        'Fire iff EXACTLY ONE input pulses on a tick — temporal XOR, the mirror\n'
        'of the coincidence detector. A lone pulse on either input produces one\n'
        'output spike %d ticks later; two pulses on the SAME tick must cancel\n'
        '(silent). Each input excites the output while the two veto each other.'
        % latency))


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
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size, description=(
        'Fire only when input A pulses and THEN input B pulses exactly %d ticks\n'
        'later. ORDER matters: B-before-A, wrong gaps and lone pulses stay\n'
        'silent. A delay line on A must coincide with the direct B path — the\n'
        'reverse ordering never lines up.' % gap))


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
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size, description=(
        'Echo input A %d ticks later, UNLESS input B pulses on the same tick —\n'
        'then that echo is vetoed. B carries no signal of its own; it only\n'
        'suppresses A. The inhibitory (NOT) input used as a real gate.' % latency))


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
    ], T=T, n_inputs=1, latency=latency, grid_size=grid_size, description=(
        'A single input kick fans out into a BURST of %d output spikes spaced %d\n'
        'ticks apart, then silence until the next kick (which re-triggers it).\n'
        'One edge in, several out — a tapped delay line or short re-triggerable\n'
        'ring. Extra or missing spikes in the burst are both penalised.'
        % (n, spacing)))


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
                        description=(
        'Frequency divider: the output fires on every THIRD input pulse and\n'
        'stays silent on the other two (a modulo-3 counter). Harder than the\n'
        'toggle flip-flop (which is divide-by-2) — it needs two bits of state to\n'
        'count 0,1,2,0,1,2. Random-length trains stop it memorising one cadence.'))


# ── registry: one entry per function, best-measuring style (metric shootout) ───
# Judged on HELD-OUT schedules (fresh random timings), oracle-trained circuits
# generalise far better for most input-driven functions (echo hand-trained:
# held-out 0.55 vs oracle 1.00; latch 0.74 vs 0.93) — so those use the oracle
# spec. Coincidence measured BETTER hand-built (held-out f1 0.94 vs 0.68), so it
# stays hand-built. Oscillator / Pattern are autonomous behaviours (no
# input->output relation to sample) and stay hand-built by necessity.
TEMPORAL_TARGETS = {
    'Oscillator':            oscillator(),
    'Pattern (100)':         pattern_generator(),
    'Coincidence (2-in)':    coincidence_detector(),
    'Temporal XOR (2-in)':   temporal_xor(),
    'Sequence A->B':         ordered_sequence(),
    'Veto gate':             veto_gate(),
    'Burst x3':              burst_generator(),
    'Divide-by-3':           divide_by_3(),
}

def _register_oracle_targets():
    import dataclasses
    from .oracle import ORACLE_SPECS
    for key, spec_name in (('SR latch',              'SR latch (oracle)'),
                           ('Toggle flip-flop',      'Toggle (oracle)'),
                           ('Echo (delay 3)',        'Echo (oracle)'),
                           ('One-shot (3 ticks)',    'One-shot (oracle)'),
                           ('Pair detector (gap 2)', 'Pair detector (oracle)'),
                           ('Period stepper',        'Period stepper (oracle)')):
        t = ORACLE_SPECS[spec_name]()
        TEMPORAL_TARGETS[key] = dataclasses.replace(t, name=key)

_register_oracle_targets()
