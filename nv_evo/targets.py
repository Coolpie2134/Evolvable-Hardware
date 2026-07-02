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
    iters:           int = 10
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

SETTLE = 3      # unscored ticks after an input event (signal propagation time)


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
                          grid_size=grid_size, iters=10, description=(
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
                          grid_size=grid_size, iters=10, description=(
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
                          trials, grid_size=grid_size, iters=10, description=(
        'A single kick pulse must start a free-running alternation: the pulse\n'
        'circulates a loop of buffers forever, read as Q toggling every tick.\n'
        'Correctly, NOTHING may happen before the kick (no input -> no output);\n'
        'trials kick at different ticks so a one-shot blip chain cannot pass.'))


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
                          grid_size=grid_size, iters=10, description=(
        'Output must reproduce the input pulse train EXACTLY %d ticks later —\n'
        'a pure delay line (each hop through a node costs one tick, so the\n'
        'signal path must be exactly %d hops). The simplest temporal target;\n'
        'trials use varied spacings including back-to-back pulses.' % (delay, delay)))


def coincidence_detector(grid_size=5, latency=3):
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
                          grid_size=grid_size, iters=10, description=(
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
                          grid_size=grid_size, iters=10, description=(
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
                          grid_size=grid_size, iters=10, description=(
        'Fires only when two input pulses arrive EXACTLY %d ticks apart: a\n'
        'delay line feeds one arm of a coincidence node and the direct path\n'
        'feeds the other, so the second pulse of a correctly-spaced pair meets\n'
        'the delayed first pulse — timing used as computation. Wrong gaps and\n'
        'lone pulses must stay silent.' % gap))


TEMPORAL_TARGETS = {
    'SR latch':         sr_latch(),
    'Toggle flip-flop': toggle_ff(),
    'Oscillator':       oscillator(),
    'Echo (delay 3)':   echo(),
    'Coincidence (2-in)':    coincidence_detector(),
    'One-shot (3 ticks)':    one_shot(),
    'Pair detector (gap 2)': pair_detector(),
}
