"""Continuous-time, event-driven leaky-integrate-and-fire simulation.

The previous implementation advanced every neuron on a global 0.1 ms loop.
That was convenient for plotting but made the computational model effectively
clocked.  This module instead advances directly from one physical event to the
next: an input-current change, synaptic pulse edge, refractory release, or
membrane threshold crossing.  ``DT`` remains only as the sampling interval for
the GUI trace arrays; it is not part of circuit evaluation.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


# ``DT`` is a display/sample cadence retained for the GUI and old callers.  The
# solver itself is continuous-time and never iterates over these samples.
DT           = 0.1    # ms visualisation sample spacing
SIM_TIME     = 20.0   # ms evaluation horizon
V_REST       = 0.0
V_RESET      = 0.0
R_M          = 1.0    # membrane resistance (GOhm)
T_REFRAC     = 2.0    # refractory period (ms)
EPSC_DUR     = 5.0    # rectangular synaptic-current duration (ms)
# The old timestep model delivered a spike's EPSC on the following 0.1 ms step.
# Keeping that causal delay preserves its intended propagation ordering without
# imposing a clock on the new solver.
SYN_DELAY    = DT

N_STEPS      = int(SIM_TIME / DT)       # trace compatibility only
REFRAC_STEPS = int(T_REFRAC / DT)       # trace compatibility only
EPSC_STEPS   = int(EPSC_DUR / DT)       # trace compatibility only

_EPS = 1e-9
_CURRENT, _THRESHOLD = 0, 1


class _EventLIF:
    """One continuous-time run of a small LIF circuit.

    Synapses emit rectangular current pulses.  Between event times, each
    non-refractory membrane has a closed-form exponential solution, so no
    numerical integration step or global update clock is required.
    """

    def __init__(self, neurons, synapses, input_currents, input_events=None,
                 input_pulses=None, sim_time=SIM_TIME, max_events=None):
        self.neurons = neurons
        self.n       = len(neurons)
        self.sim_time = float(sim_time)
        self.tau     = np.array([nu.tau for nu in neurons], dtype=np.float64)
        self.vth     = np.array([nu.vth for nu in neurons], dtype=np.float64)
        self.v       = np.full(self.n, V_REST, dtype=np.float64)
        self.i_syn   = np.zeros(self.n, dtype=np.float64)
        self.i_ext   = np.zeros(self.n, dtype=np.float64)
        self.refr_until = np.full(self.n, -math.inf, dtype=np.float64)
        # Edge-triggered re-arm: one contiguous stimulus can produce at most
        # one output pulse.  A new pulse is permitted only after the drive has
        # fallen away, which is what turns a LIF cell into an asynchronous
        # signal relay rather than a rate-coded oscillator.
        self.armed   = np.ones(self.n, dtype=bool)
        self.version = np.zeros(self.n, dtype=np.int64)
        self.t       = 0.0
        self.heap    = []
        self._seq    = 0
        self.spikes  = {i: [] for i in range(self.n)}
        self.max_events = (
            None if max_events is None else max(1, int(max_events)))
        self.event_count = 0
        self.overflow = False

        self.input_ids = {i for i, nu in enumerate(neurons) if nu.is_input}
        self.outgoing = [[] for _ in range(self.n)]
        for syn in synapses:
            if 0 <= syn.pre < self.n and 0 <= syn.post < self.n:
                # Input cells are ideal source ports.  Letting a grown circuit
                # drive an input port is exactly how A could spuriously light B.
                if syn.post in self.input_ids:
                    continue
                # Per-synapse delay is optional for old pickle-compatible
                # Synapse objects; new circuit wiring uses the causal default.
                delay = max(0.0, float(getattr(syn, 'delay', SYN_DELAY)))
                self.outgoing[syn.pre].append((syn.post, float(syn.weight), delay))

        for nid, current in (input_currents or {}).items():
            if 0 <= nid < self.n:
                if nid not in self.input_ids:
                    self.i_ext[nid] = float(current)

        # Segments are snapshots immediately after every state-changing time.
        # They let simulate_trace sample an event-driven run for the GUI without
        # reintroducing fixed-step dynamics.
        self.segments = []
        self._record_segment()

        for nid in input_pulses or ():
            if nid in self.input_ids:
                self._push(0.0, _CURRENT, 'input_pulse', nid)

        # Optional pulse inputs make the engine usable by temporal SNN targets.
        # Each entry is (time_ms, neuron_id, current, duration_ms). Events on an
        # input neuron are ideal source edges: amplitude/width describe the
        # external waveform but the circuit receives its leading edge exactly.
        # Events on ordinary neurons retain the analog-current interpretation.
        for event in input_events or ():
            time, nid, current, duration = event
            if 0 <= nid < self.n and duration > 0:
                if nid in self.input_ids:
                    self._push(float(time), _CURRENT, 'input_pulse', nid)
                else:
                    self._push(float(time), _CURRENT, 'external', nid,
                               float(current))
                    self._push(float(time) + float(duration), _CURRENT,
                               'external', nid, -float(current))

        for nid in range(self.n):
            self._schedule_threshold(nid)

    def _push(self, when, priority, kind, nid, value=None):
        if when >= self.sim_time - _EPS:
            return
        self._seq += 1
        heapq.heappush(self.heap, (max(self.t, when), priority, self._seq,
                                   kind, nid, value))

    def _advance_to(self, when):
        """Analytically advance all membranes to ``when``.

        A refractory release is itself queued as an event, so a membrane is
        either refractory for this entire segment or integrates for all of it.
        """
        dt = when - self.t
        if dt <= _EPS:
            self.t = when
            return
        active = ((self.refr_until <= self.t + _EPS) & self.armed)
        if np.any(active):
            inf = V_REST + R_M * (self.i_ext[active] + self.i_syn[active])
            decay = np.exp(-dt / self.tau[active])
            self.v[active] = inf + (self.v[active] - inf) * decay
        self.v[~active] = V_RESET
        self.t = when

    def _threshold_time(self, nid):
        if not self.armed[nid]:
            return None
        if self.refr_until[nid] > self.t + _EPS:
            return None
        v = self.v[nid]
        threshold = self.vth[nid]
        if v >= threshold - _EPS:
            return self.t
        inf = V_REST + R_M * (self.i_ext[nid] + self.i_syn[nid])
        if inf <= threshold + _EPS:
            return None
        ratio = (threshold - inf) / (v - inf)
        if ratio <= 0.0:
            return self.t
        if ratio >= 1.0:
            return None
        return self.t - self.tau[nid] * math.log(ratio)

    def _schedule_threshold(self, nid):
        crossing = self._threshold_time(nid)
        if crossing is None or crossing >= self.sim_time - _EPS:
            return
        self._push(crossing, _THRESHOLD, 'threshold', nid,
                   int(self.version[nid]))

    def _fire(self, nid):
        if not self._record_spike(nid):
            return
        self.v[nid] = V_RESET
        self.refr_until[nid] = self.t + T_REFRAC
        self.armed[nid] = False
        self.version[nid] += 1
        self._push(self.refr_until[nid], _CURRENT, 'ready', nid)
        for post, weight, delay in self.outgoing[nid]:
            arrival = self.t + delay
            self._push(arrival, _CURRENT, 'synaptic', post, weight)
            self._push(arrival + EPSC_DUR, _CURRENT, 'synaptic', post, -weight)

    def _emit_source(self, nid):
        """Emit one ideal external edge without neuronal refractory.

        Input cells are physical ports, not LIF processing elements. Every
        distinct leading edge presented by the environment must enter the
        circuit, including edges separated by less than ``T_REFRAC``.
        """
        if not self._record_spike(nid):
            return
        for post, weight, delay in self.outgoing[nid]:
            arrival = self.t + delay
            self._push(arrival, _CURRENT, 'synaptic', post, weight)
            self._push(arrival + EPSC_DUR, _CURRENT, 'synaptic', post, -weight)

    def _record_spike(self, nid):
        self.spikes[nid].append(float(self.t))
        self.event_count += 1
        if (self.max_events is not None
                and self.event_count > self.max_events):
            self.overflow = True
            self.heap.clear()
            return False
        return True

    def _rearm_if_quiet(self, nid):
        if self.i_ext[nid] + self.i_syn[nid] <= _EPS:
            self.armed[nid] = True

    def _handle(self, kind, nid, value):
        if kind == 'synaptic':
            self.i_syn[nid] += value
            self.version[nid] += 1
            self._rearm_if_quiet(nid)
            self._schedule_threshold(nid)
        elif kind == 'external':
            self.i_ext[nid] += value
            self.version[nid] += 1
            self._rearm_if_quiet(nid)
            self._schedule_threshold(nid)
        elif kind == 'ready':
            # The membrane stayed at reset through refractory; from this point
            # onward its extant currents can charge it again.
            self.v[nid] = V_RESET
            self.version[nid] += 1
            self._rearm_if_quiet(nid)
            self._schedule_threshold(nid)
        elif kind == 'input_pulse':
            # Source ports emit an ideal, one-shot edge.  They do not need a
            # held bias current, never accept synaptic feedback, and are not
            # subject to a processing neuron's refractory period.
            self._emit_source(nid)
        elif kind == 'threshold':
            if value != int(self.version[nid]):
                return                         # stale prediction after an event
            if self.refr_until[nid] > self.t + _EPS:
                return
            if self.v[nid] < self.vth[nid] - 1e-7:
                return
            self._fire(nid)

    def _record_segment(self):
        snap = (self.t, self.v.copy(), self.i_syn.copy(), self.i_ext.copy(),
                self.refr_until.copy(), self.armed.copy())
        if self.segments and abs(self.segments[-1][0] - self.t) <= _EPS:
            self.segments[-1] = snap
        else:
            self.segments.append(snap)

    def run(self):
        while self.heap and not self.overflow:
            when = self.heap[0][0]
            if when >= self.sim_time - _EPS:
                break
            self._advance_to(when)
            # Consume every event caused at this same physical instant.  With
            # the causal default SYN_DELAY these are independent changes, but
            # this also gives deterministic behaviour to a custom zero-delay
            # synapse.
            while self.heap and abs(self.heap[0][0] - when) <= _EPS:
                _, _, _, kind, nid, value = heapq.heappop(self.heap)
                self._handle(kind, nid, value)
            self._record_segment()
        self._advance_to(self.sim_time)
        self._record_segment()
        return self

    def sample_voltage(self, times):
        """Sample the continuous run for display; this does not affect events."""
        out = np.zeros((len(times), self.n), dtype=np.float32)
        seg_i = 0
        for row, when in enumerate(times):
            while (seg_i + 1 < len(self.segments)
                   and self.segments[seg_i + 1][0] <= when + _EPS):
                seg_i += 1
            start, v, i_syn, i_ext, refr_until, armed = self.segments[seg_i]
            dt = max(0.0, when - start)
            sample = v.copy()
            active = (refr_until <= start + _EPS) & armed
            if np.any(active) and dt > 0.0:
                inf = V_REST + R_M * (i_ext[active] + i_syn[active])
                decay = np.exp(-dt / self.tau[active])
                sample[active] = inf + (sample[active] - inf) * decay
            sample[~active] = V_RESET
            out[row] = sample
        return out


def _run(neurons, synapses, input_currents, input_events=None,
         input_pulses=None, sim_time=SIM_TIME, max_events=None):
    if not neurons:
        return None
    return _EventLIF(neurons, synapses, input_currents, input_events,
                     input_pulses, sim_time, max_events).run()


def simulate(neurons, synapses, input_currents, sim_time=SIM_TIME):
    """Return ``{neuron_id: [continuous spike times in ms]}``.

    A logical high on an input port becomes one source pulse at t=0, rather
    than a 20 ms held current.  This preserves the public truth-table API while
    giving it asynchronous pulse semantics: one input event can propagate to
    an output without repeatedly re-triggering every cell.
    """
    pulses = [nid for nid, current in (input_currents or {}).items()
              if current and 0 <= nid < len(neurons) and neurons[nid].is_input]
    run = _run(neurons, synapses, input_currents, input_pulses=pulses,
               sim_time=sim_time)
    return {} if run is None else run.spikes


def simulate_events(neurons, synapses, input_events, sim_time=SIM_TIME):
    """Run pulse-driven external inputs without a global simulation clock.

    ``input_events`` entries are ``(time_ms, neuron_id, current, duration_ms)``.
    This is the asynchronous-facing API for temporal SNN targets; the existing
    ``simulate`` function remains the static-current wrapper used by truth
    tables.
    """
    run = _run(neurons, synapses, {}, input_events=input_events,
               sim_time=sim_time)
    return {} if run is None else run.spikes


def simulate_trace(neurons, synapses, input_currents, sim_time=SIM_TIME,
                   sample_dt=DT):
    """Event-driven simulation plus sampled GUI traces.

    The returned voltage history is sampled after the fact at ``sample_dt``;
    it is not used to advance the circuit.  Existing callers retain their
    ``(times, V_hist, spikes, fired_hist)`` contract.
    """
    n_steps = int(sim_time / sample_dt)
    times = [k * sample_dt for k in range(n_steps)]
    pulses = [nid for nid, current in (input_currents or {}).items()
              if current and 0 <= nid < len(neurons) and neurons[nid].is_input]
    run = _run(neurons, synapses, input_currents, input_pulses=pulses,
               sim_time=sim_time)
    if run is None:
        return (times, np.zeros((n_steps, 0), np.float32), {},
                np.zeros((n_steps, 0), bool))
    v_hist = run.sample_voltage(times)
    fired = np.zeros((n_steps, len(neurons)), dtype=bool)
    for nid, spike_times in run.spikes.items():
        for when in spike_times:
            sample = min(n_steps - 1, max(0, int(round(when / sample_dt))))
            fired[sample, nid] = True
            # Preserve the visible pre-reset peak expected by the old UI.
            v_hist[sample, nid] = max(v_hist[sample, nid], run.vth[nid] + 0.1)
    return times, v_hist, run.spikes, fired
