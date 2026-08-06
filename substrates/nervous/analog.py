"""
substrates/nervous/analog.py - the analog Fig. 1 node: charge, leak, comparator, hysteresis.

The default engine (pulse.py) is a digital abstraction of the paper's node: it
regenerates ONE fixed-width pulse after ONE fixed delay and treats the
coincidence window as a hard rectangle. The real circuit (Edwards EH'02, Fig. 1)
is analog: excitatory edges couple CAPACITIVELY onto a high-impedance comparator
node, a Vbp-controlled transistor LEAKS that node back toward Vdd, and the
comparator (with a little hysteresis) trips when the node crosses threshold. From
that single mechanism, three behaviours the digital engine hard-codes instead
EMERGE, and are now consequences of the same physical constants rather than
independent knobs (the reference plan's point 3):

  * coincidence  - one excitatory TERMINAL edge steps the node down by ``step``;
    with step < (Vdd - threshold) a single edge cannot trip it, but two within a
    leak time-constant sum past threshold. "Neither input alone can trigger a
    response." The window WIDTH is set by ``tau_leak`` and ``step``, not a COINC
    knob. A buffer ties BOTH terminals to one source, so a single source edge
    delivers 2*step and fires alone - exactly the paper's buffer-vs-coincidence
    distinction, falling straight out of the wiring.
  * output width - the comparator holds its output while the node sits below
    threshold; the leak sets how long that is. Wider/denser input pushes the node
    further down and STRETCHES the pulse (a following edge during the pulse
    extends it), so a node is paralyzable, unlike the digital fixed width.
  * recovery/refractory - a hysteretic comparator trips at ``threshold`` and
    releases/re-arms only at ``threshold+hysteresis``. The output pulse and
    recovery are one physical state transition, not a fixed pulse followed by
    a separate hidden lock-out.

Voltages are normalised: Vdd = ``rest`` = 1.0, Ground = 0.0. The node idles at
rest; edges step it DOWN; it recovers UP toward rest as
``v(t) = rest + (v0 - rest) * exp(-(t - t0) / tau_leak)``. Because steps are
instantaneous and recovery is monotone-up between them, a downward threshold
crossing can only happen AT an edge - so firing is decided at edge times and no
inter-event root-finding is needed. The output fall is the recovery time solved
analytically, rescheduled (inertial-delay style, like substrates/lut/pulse.py) whenever
a later edge extends the pulse.

Figure 1's I1 input controls the leak/bias branch. Without transistor parameters
the portable model cannot claim a calibrated analogue conductance, so active I1
is represented explicitly as a fast reset-to-Vdd clamp at an excitatory edge.
That is a qualitative circuit approximation, not a made-up graded voltage.

AnalogPulseSim mirrors PulseSim's external surface (inject_pulse / advance_to /
activity_at / rise_times / pulse_intervals / ever / overflow / config / grid),
so it is a drop-in for scoring and playback, and it composes with the tri-tile
substrate through the same pre-resolved ``sources`` hook.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .hexgrid import hex_dirs

_NEG = float('-inf')


@dataclass(frozen=True)
class AnalogConfig:
    """Physical constants of the Fig. 1 node (normalised Vdd = rest = 1.0).

    Coincidence tolerance, output width and recovery are DERIVED from these, not
    set independently - the reference plan's point 3. Defaults are chosen so a
    single excitatory edge (``step``) leaves the node above ``threshold`` while
    two (a buffer's 2*step, or two coincident lines) cross it. These are
    qualitative normalised defaults: the paper does not publish transistor
    sizes or a calibrated time/voltage scale."""
    rest: float = 1.0             # Vdd
    threshold: float = 0.5        # comparator trip level (below rest)
    step: float = 0.34            # capacitive down-step per excitatory terminal edge
    tau_leak: float = 1.10        # Vbp leak time constant (recovery toward rest)
    hysteresis: float = 0.08      # re-arm margin above threshold (anti-chatter)
    delay_prop: float = 1.0       # comparator + inverter propagation delay
    event_cap: int = 4096
    model: str = 'paper_analog'   # identifies this engine to create_simulator

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
                self.rest, self.threshold, self.step, self.tau_leak,
                self.hysteresis, self.delay_prop)):
            raise ValueError('analog constants must be finite')
        gap = self.rest - self.threshold
        if not (0 < self.threshold < self.rest):
            raise ValueError('threshold must lie strictly between 0 and rest')
        if not (self.step < gap):
            raise ValueError('step must be < (rest - threshold) so ONE excitatory '
                             'edge cannot fire a coincidence node (paper semantics)')
        if not (2 * self.step > gap):
            raise ValueError('2*step must exceed (rest - threshold) so a buffer / '
                             'a coincident pair CAN fire')
        for name in ('tau_leak', 'delay_prop'):
            if getattr(self, name) <= 0:
                raise ValueError('%s must be positive' % name)
        if self.hysteresis < 0 or self.event_cap < 1:
            raise ValueError('hysteresis must be non-negative; event_cap >= 1')
        if self.threshold + self.hysteresis >= self.rest:
            raise ValueError('threshold + hysteresis must remain below rest')

    # Duck-compatibility with PulseConfig so the shared scoring/injection code
    # (which reads config.width / config.delay) works unchanged. For the analog
    # node these are not independent knobs: 'width' is only the minimum injected
    # INPUT pulse width, and output width / coincidence are emergent.
    @property
    def width(self):
        return self.delay_prop

    @property
    def delay(self):
        return self.delay_prop

    @property
    def coincidence(self):
        return 0.0


class AnalogPulseSim:
    """Event-driven analog nervous-net node (see module docstring)."""

    def __init__(self, grid, routing, max_events=None, config=None,
                 sources=None, inputs=None, input_nodes=None,
                 output_nodes=None):
        self.grid    = grid
        self.routing = routing
        self.config  = config or AnalogConfig()
        self.max_events = (self.config.event_cap if max_events is None
                           else max(1, int(max_events)))
        self.event_count = 0
        self.overflow = False
        self._inputs = set(inputs or ()) | set(input_nodes or ())
        self._outputs = set(output_nodes or ())

        # src[v] = (s1, s2, si) feeder cells; watch[u] = cells reading u's wire.
        self.src   = {}
        self.watch = {c: [] for c in grid}
        self.op = {}
        for v, entry in routing.items():
            if v not in grid:
                continue
            if v in self._inputs:
                continue
            e1, e2, i1 = entry[0], entry[1], entry[2]
            # entry[3] is the routing op. 'or' means EITHER excitatory input
            # fires the node on its own; physically that is the buffer's wiring
            # generalised - each input is coupled to BOTH terminals, so one edge
            # delivers 2*step and crosses threshold. Ignoring it collapsed every
            # OR routing onto its AND twin, which made half the state alphabet
            # inert under this engine and left the tri tile unable to sustain a
            # circulating pulse.
            self.op[v] = entry[3] if len(entry) > 3 else 'and'
            if sources is not None:
                s1, s2, si = sources.get(v, (None, None, None))
            else:
                nb = hex_dirs(*v)
                s1 = nb[e1] if e1 is not None else None
                s2 = nb[e2] if e2 is not None else None
                si = nb[i1] if i1 is not None else None
            s1 = None if s1 in self._outputs else s1
            s2 = None if s2 in self._outputs else s2
            si = None if si in self._outputs else si
            self.src[v] = (s1, s2, si)
            for s in {s1, s2}:
                if s is not None and s in self.watch:
                    self.watch[s].append(v)

        # analog node state
        self.v       = {c: self.config.rest for c in grid}   # comparator-node voltage
        self.v_time  = {c: 0.0 for c in grid}                # last time v was refreshed
        self.armed   = {c: True for c in grid}               # past hysteresis, may fire
        self.out_high = {c: False for c in grid}             # comparator tripped
        self._fall_seq = {c: 0 for c in grid}                # inertial reschedule guard
        self._pending_end = {c: None for c in grid}          # wire end before it opens

        # wire intervals (for downstream stepping, inhibitory reads, scoring)
        self.pulse_start = {c: _NEG for c in grid}
        self.pulse_until = {c: _NEG for c in grid}
        self.ever        = {c: 0 for c in grid}
        self.rise_times  = {c: [] for c in grid}
        self.pulse_intervals = {c: [] for c in grid}

        self._heap = []             # (time, seq, kind, cell, arg)
        self._seq  = 0
        self._now  = 0.0
        self._tick = 0
        self._prev = {c: 0 for c in grid}

    # -- voltage helpers ------------------------------------------------------
    def _refresh(self, c, t):
        """Leak v[c] up toward rest to absolute time t."""
        dt = t - self.v_time[c]
        if dt > 0:
            rest = self.config.rest
            self.v[c] = rest + (self.v[c] - rest) * math.exp(-dt / self.config.tau_leak)
            self.v_time[c] = t

    def _recover_time(self, v_now, level):
        """deltat for v to leak from ``v_now`` up to ``level`` (both < rest)."""
        rest = self.config.rest
        num = rest - v_now
        den = rest - level
        if num <= den:                     # already at/above level
            return 0.0
        return self.config.tau_leak * math.log(num / den)

    def _high(self, c, t):
        return self.pulse_start.get(c, _NEG) <= t < self.pulse_until.get(c, _NEG)

    # -- event queue ----------------------------------------------------------
    def _push(self, t, kind, cell, arg=None):
        self._seq += 1
        heapq.heappush(self._heap, (float(t), self._seq, kind, cell, arg))

    def inject_pulse(self, cell, t, width=None):
        """Drive an external pulse onto an input wire (wired-OR). Its RISING edge
        couples into watchers, exactly like a neighbour's output edge."""
        if cell not in self.grid:
            return
        w = self.config.delay_prop if width is None else float(width)
        start = float(t)
        if not (math.isfinite(start) and math.isfinite(w) and w > 0):
            raise ValueError('pulse start must be finite and width positive')
        self._push(start, 'wire_rise', cell, start + w)

    def advance_to(self, when):
        self._run_until(float(when))

    def activity_at(self, when):
        when = float(when)
        return {c: (1 if self._high(c, when) else 0) for c in self.grid}

    def step(self, input_vals):
        """Advance one display tick for legacy truth-table callers.

        A held input remains one wired-OR interval, so only its initial edge is
        capacitively coupled into downstream nodes, just as in PulseSim.
        """
        from .pulse import TICK
        t0 = self._tick * TICK
        for cell, value in input_vals.items():
            high = 1 if value else 0
            if high:
                if (self._prev.get(cell, 0)
                        and self.pulse_until.get(cell, _NEG) >= t0):
                    end = max(self.pulse_until[cell], t0 + TICK)
                    self.pulse_until[cell] = end
                    if self.pulse_intervals.get(cell):
                        self.pulse_intervals[cell][-1][1] = float(end)
                else:
                    self.inject_pulse(cell, t0, max(self.config.width, TICK))
            self._prev[cell] = high
        sample_t = t0 + 0.5 * TICK
        self._run_until(sample_t)
        self._tick += 1
        return self.activity_at(sample_t)

    # -- the simulation core --------------------------------------------------
    # Two timelines per node: the COMPARATOR (node voltage, trips exactly at an
    # excitatory edge, releases at threshold+hysteresis) and the OUTPUT
    # WIRE, which is the comparator delayed by delay_prop. Downstream nodes are
    # coupled by the WIRE's rising edge, giving one propagation delay per hop.

    def _open_input_wire(self, cell, t, end):
        """An INPUT wire goes high over [t, end). Returns True if this is a NEW
        rise (so the caller couples watchers after the whole batch has opened)."""
        if self._high(cell, t):
            if end > self.pulse_until[cell]:
                self.pulse_until[cell] = end
                self.pulse_intervals[cell][-1][1] = float(end)
            return False
        return self._record_rise(cell, t, end)

    def _record_rise(self, cell, t, end):
        if self.event_count >= self.max_events:
            self.overflow = True
            self._heap.clear()
            return False
        self.pulse_start[cell] = t
        self.pulse_until[cell] = end
        self.ever[cell] = 1
        self.rise_times[cell].append(float(t))
        self.pulse_intervals[cell].append([float(t), float(end)])
        self.event_count += 1
        return True

    def _open_circuit_wire(self, v, t):
        """A circuit node's output wire rises at t with its pending end. Returns
        True on a genuine new rise."""
        end = self._pending_end[v]
        if end is None or end <= t:
            end = t + self.config.delay_prop
        if self._high(v, t):
            if end > self.pulse_until[v]:
                self.pulse_until[v] = end
                self.pulse_intervals[v][-1][1] = float(end)
            return False
        if not self._record_rise(v, t, end):
            return False
        return True

    def _excite(self, v, source, t):
        """A watched wire ``source`` rose at t: apply its capacitive step(s) to v
        and decide whether v fires (or extends an in-progress pulse)."""
        s1, s2, si = self.src[v]
        n_terminals = (1 if s1 == source else 0) + (1 if s2 == source else 0)
        if n_terminals == 0:
            return
        if self.op.get(v) == 'or':
            # OR wiring couples each excitatory input to both terminals, so a
            # lone edge delivers the same 2*step a buffer does. (A buffer is
            # already s1 == s2 and reaches 2 by counting.)
            n_terminals = 2
        cfg = self.config
        self._refresh(v, t)
        if si is not None and self._high(si, t):
            # Fig. 1's I1 controls the leak/bias branch.  With no transistor
            # parameters available, model it explicitly as a fast clamp to Vdd
            # at an excitatory edge rather than inventing a graded voltage.
            self.v[v] = cfg.rest
            self.v_time[v] = t
            return
        self.v[v] -= n_terminals * cfg.step
        if self.v[v] >= cfg.threshold:
            return                              # no crossing
        if self.out_high[v]:
            self._reschedule_release(v, t)      # paralyzable: extend the pulse
            return
        if not self.armed[v]:
            return                              # within hysteresis lock-out
        # fire: comparator trips at t; output wire rises at t + delay_prop
        self.armed[v] = False
        self.out_high[v] = True
        cfg = self.config
        d_rel = self._recover_time(
            self.v[v], cfg.threshold + cfg.hysteresis)
        self._pending_end[v] = (t + d_rel) + cfg.delay_prop
        self._fall_seq[v] += 1
        self._push(t + cfg.delay_prop, 'wire_rise', v, None)   # None => circuit
        self._push(t + d_rel, 'release', v, self._fall_seq[v])

    def _reschedule_release(self, v, t):
        """A fresh push while the comparator is tripped: recompute the release
        from the deeper voltage (inertial-delay stretch) and, if the wire is
        already open, extend its recorded interval."""
        self._refresh(v, t)
        cfg = self.config
        d_rel = self._recover_time(
            self.v[v], cfg.threshold + cfg.hysteresis)
        end = (t + d_rel) + cfg.delay_prop
        self._pending_end[v] = end
        self._fall_seq[v] += 1
        if self._high(v, t) and self.pulse_intervals[v] and end > self.pulse_until[v]:
            self.pulse_until[v] = end
            self.pulse_intervals[v][-1][1] = float(end)
        self._push(t + d_rel, 'release', v, self._fall_seq[v])

    def _release(self, v, t, seq):
        if seq != self._fall_seq[v] or not self.out_high[v]:
            return                              # superseded by a stretch
        cfg = self.config
        self._refresh(v, t)
        # A true hysteretic comparator releases and re-arms at the upper
        # threshold.  There is no hidden post-pulse lockout state.
        self.out_high[v] = False
        self.armed[v] = True

    def _run_until(self, t_end):
        while self._heap and self._heap[0][0] <= t_end:
            t = self._heap[0][0]
            batch = []
            while self._heap and self._heap[0][0] == t:
                batch.append(heapq.heappop(self._heap))
            # Two-phase: OPEN every wire that rises at this instant before any
            # excitation is judged, so simultaneous coincident edges are seen
            # together and a simultaneous inhibitory rise vetoes reliably. Safe
            # because delay_prop > 0 makes every secondary rise strictly future.
            new_rises = []
            for (_, _, kind, cell, arg) in batch:
                if kind == 'wire_rise':
                    opened = (self._open_circuit_wire(cell, t) if arg is None
                              else self._open_input_wire(cell, t, arg))
                    if opened:
                        new_rises.append(cell)
                if self.overflow:
                    return
            for cell in new_rises:
                for w in self.watch[cell]:
                    self._excite(w, cell, t)
                if self.overflow:
                    return
            for (_, _, kind, cell, arg) in batch:
                if kind == 'release':
                    self._release(cell, t, arg)
            self._now = t
