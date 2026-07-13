"""
nv_evo/pulse.py — asynchronous, edge-triggered pulse dynamics (paper Fig. 1).

The nervous net is "a continuous-time, asynchronous system with binary-valued
output. It propagates a pulse arriving at its input with a fixed delay" (§3).
This module is an event-driven simulation of exactly that:

  * a WIRE (each cell's output net) is idle or carries a pulse [start, end);
  * all action is precipitated by a pulse EDGE (the pulse's leading edge —
    the paper's 1->0 transition; polarity is abstracted away);
  * a buffer node (E1==E2) triggers on a single edge of its one input wire;
    a coincidence node triggers only when edges arrive on BOTH excitatory
    wires within the coincidence window COINC ("neither input alone can
    trigger a response");
  * an active inhibitory input (its wire is pulsing at trigger time) prevents
    the response;
  * a triggered node emits a pulse of width WIDTH after a fixed delay DELAY,
    and is refractory until its own pulse ends (the comparator hysteresis —
    no chattering);
  * external inputs are INJECTED onto the input cell's net (wired-OR, like
    driving the physical perimeter wire). A held-high input is one long pulse
    — one edge — not a train of edges.

Hardware mapping is 1:1 with the paper's circuit: DELAY is the node's fixed
propagation delay, WIDTH the pulse width set by the leak transistor (the
"time constant"), COINC the integration window of the capacitively-coupled
excitatory inputs. Every leading edge is retained in ``rise_times`` for
continuous-time fitness; wires are also sampled once per TICK for playback and
legacy persistence-window targets.

With the default constants (DELAY = WIDTH = TICK, COINC < TICK) behaviour on
the integer tick lattice matches the earlier synchronous engine — that engine
was the quantization of this one. The constants can now be varied to study
sub-tick timing (unequal delays, edge alignment, incommensurate loop periods).
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass

from .hexgrid import hex_dirs

# Backward-compatible defaults. New runs carry an immutable PulseConfig instead
# of mutating these module values or relying on worker-process environments.
DELAY = 1.0   # fixed node propagation delay
WIDTH = 1.0   # output pulse width (leak rate)
COINC = 0.5   # excitatory coincidence window
TICK  = 1.0     # sampling period of the scoring layer (one target tick)


@dataclass(frozen=True)
class PulseConfig:
    """Immutable paper-model timing parameters carried with one run."""
    delay: float = DELAY
    width: float = WIDTH
    coincidence: float = COINC
    event_cap: int = 2048

    def __post_init__(self):
        if (self.delay <= 0 or self.width <= 0 or self.coincidence < 0
                or self.event_cap < 1):
            raise ValueError('delay/width/event_cap must be positive and coincidence non-negative')

_NEG = float('-inf')


class PulseSim:
    """Event-driven pulse simulation of one grown nervous net.

    Usage: sim = PulseSim(grid, routing); then once per tick
    ``state = sim.step({input_cell: bit, ...})`` which injects/extends input
    pulses, advances the event queue, and returns {cell: 0/1} — each wire
    sampled at the middle of the tick. ``sim.ever`` maps cells to whether
    their wire has pulsed at all (the combinational "did it fire" read-out).
    """

    def __init__(self, grid, routing, max_events=None, config=None):
        self.grid    = grid
        self.routing = routing
        self.config  = config or PulseConfig()
        # src[v] = (s1, s2, si): the cells feeding v's E1 / E2 / I1.
        # watch[u] = cells that read u on an excitatory input.
        self.src   = {}
        self.op    = {}                               # v -> 'and' | 'or'
        self.watch = {c: [] for c in grid}
        for v, entry in routing.items():
            if v not in grid:
                continue
            e1, e2, i1 = entry[0], entry[1], entry[2]
            self.op[v] = entry[3] if len(entry) > 3 else 'and'
            nb = hex_dirs(*v)
            s1 = nb[e1] if e1 is not None else None
            s2 = nb[e2] if e2 is not None else None
            si = nb[i1] if i1 is not None else None
            self.src[v] = (s1, s2, si)
            for s in {s1, s2}:
                if s is not None and s in self.watch:
                    self.watch[s].append(v)
        self.pulse_start = {c: _NEG for c in grid}
        self.pulse_until = {c: _NEG for c in grid}
        self.refr_until  = {c: _NEG for c in grid}
        self.last_edge   = {c: {} for c in grid}      # v -> {source: edge time}
        self.ever        = {c: 0 for c in grid}
        # Raw leading-edge timestamps are the substrate's behavioural output.
        # Tick samples remain a playback convenience, not the source of truth
        # for event-based fitness.
        self.rise_times  = {c: [] for c in grid}
        self.max_events  = None if max_events is None else max(1, int(max_events))
        self.event_count = 0
        self.overflow    = False
        self._heap = []                               # (time, seq, cell, pulse_end)
        self._seq  = 0
        self._tick = 0                                # next tick index
        self._prev = {}                               # input cell -> previous bit

    # ── wires and events ─────────────────────────────────────────────────────

    def _high(self, c, t):
        # a routed direction may point at a dead / off-grid cell: no wire,
        # never high (its excitatory edge never comes; its veto never fires)
        return self.pulse_start.get(c, _NEG) <= t < self.pulse_until.get(c, _NEG)

    def _push(self, t, cell, end):
        self._seq += 1
        heapq.heappush(self._heap, (t, self._seq, cell, end))

    def _run_until(self, t_end):
        """Process events chronologically. Events sharing a timestamp are
        applied in two phases — first every wire rise, then every edge
        notification — so e.g. an inhibitory pulse arriving simultaneously
        with the excitatory edges reliably vetoes them."""
        while self._heap and self._heap[0][0] <= t_end:
            t = self._heap[0][0]
            batch = []
            while self._heap and self._heap[0][0] == t:
                batch.append(heapq.heappop(self._heap))
            new_edges = []
            for (_, _, cell, end) in batch:
                if self._high(cell, t):                       # wired-OR merge:
                    if end > self.pulse_until[cell]:          # already high ->
                        self.pulse_until[cell] = end          # extend, no edge
                else:
                    if (self.max_events is not None
                            and self.event_count >= self.max_events):
                        # A pathological oscillator must not make one genome
                        # monopolise evaluation.  Mark the run invalid and stop
                        # deterministically; scorers turn overflow into zero.
                        self.overflow = True
                        self._heap.clear()
                        return
                    self.pulse_start[cell] = t
                    self.pulse_until[cell] = end
                    self.ever[cell] = 1
                    self.rise_times[cell].append(float(t))
                    self.event_count += 1
                    new_edges.append(cell)
            for u in new_edges:
                for v in self.watch[u]:
                    self._consider(v, u, t)

    def advance_to(self, when):
        """Process queued events through an absolute physical time.

        ``step`` intentionally stops at the mid-tick display sample.  Temporal
        evaluation calls this once at the end of a run so late half-tick edges
        are retained in ``rise_times`` instead of being silently discarded.
        """
        self._run_until(float(when))

    def activity_at(self, when):
        """Return the wire levels at an already-processed physical time."""
        when = float(when)
        return {cell: (1 if self._high(cell, when) else 0)
                for cell in self.grid}

    def inject_pulse(self, cell, t, width=None):
        """Inject one input pulse of ``width`` (default WIDTH) onto ``cell``'s net
        at absolute time ``t``, wired-OR with any existing pulse. This is the
        continuous-time analogue of ``step``'s per-tick input injection, used by
        the async Interactive/Designer playback to place edges at real (possibly
        sub-tick) times. Events are queued; call ``advance_to`` to process them."""
        if cell in self.grid:
            w = self.config.width if width is None else float(width)
            self._push(float(t), cell, float(t) + w)

    def _consider(self, v, u, t):
        """Cell v sees a pulse edge from neighbour u at time t."""
        if t < self.refr_until[v]:
            return
        s1, s2, si = self.src[v]
        if si is not None and self._high(si, t):              # inhibitory veto
            return
        if s1 == s2:                                          # buffer: one edge
            trig = (u == s1)
        elif self.op[v] == 'or':                              # OR: either edge
            trig = (u == s1 or u == s2)                       # activates the node
        else:                                                 # coincidence: both
            self.last_edge[v][u] = t                          # edges within COINC
            other = s2 if u == s1 else s1
            trig = (t - self.last_edge[v].get(other, _NEG)) <= self.config.coincidence
        if trig:
            self.refr_until[v] = t + self.config.delay + self.config.width
            self._push(t + self.config.delay, v,
                       t + self.config.delay + self.config.width)

    # ── the per-tick interface used by scoring / playback ────────────────────

    def step(self, input_vals):
        """Advance one tick. input_vals {cell: 0/1} drives the input nets:
        a 0->1 stream transition injects a pulse edge; consecutive 1s extend
        the same pulse (a held level is ONE long pulse, one edge). Returns
        {cell: 0/1} — every wire sampled mid-tick."""
        t0 = self._tick * TICK
        for c, b in input_vals.items():
            b = 1 if b else 0
            if b:
                if self._prev.get(c, 0) and self.pulse_until[c] >= t0:
                    self.pulse_until[c] = max(self.pulse_until[c], t0 + TICK)
                else:
                    self._push(t0, c, t0 + max(self.config.width, TICK))
            self._prev[c] = b
        sample_t = t0 + 0.5 * TICK
        self._run_until(sample_t)
        self._tick += 1
        return {c: (1 if self._high(c, sample_t) else 0) for c in self.grid}
