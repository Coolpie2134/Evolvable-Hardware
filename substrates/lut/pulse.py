"""
substrates/lut/pulse.py — asynchronous, event-driven dynamics for the LUT array.

This puts the LUT substrate on the same physical footing as the nervous net
(substrates/nervous/pulse.py): a continuous-time, event-driven system with no global clock.
The two substrates keep their distinct characters —

  * a nervous WIRE carries PULSES (edge-triggered nodes, refractory);
  * a LUT WIRE carries LEVELS (each cell is a logic element: its four output
    bits are a lookup of the four neighbour bits facing it — a latch can hold
    a steady level, so held state is a level, not a circulating pulse);

— but both are now simulated as physical hardware: every cell has one fixed
propagation delay, all action is precipitated by a wire LEVEL CHANGE, and
inputs are wired-OR injections onto the input cells' nets at arbitrary
(possibly sub-tick) real times.

Delay model: INERTIAL, the standard asynchronous-logic model. When any of a
cell's input wires changes at time t, the cell re-evaluates its lookup on the
CURRENT input levels and schedules that value to appear on its output at
t + delay, superseding any not-yet-applied earlier evaluation (an input blip
shorter than the delay is filtered, exactly like a real gate whose output
node cannot follow it). A cell whose re-evaluation equals its present output
cancels its pending change.

Power-on: all wires start at level 0, and every cell evaluates that all-zero
neighbourhood — a lookup table with its index-0 bit set therefore fires
SPONTANEOUSLY (the paper's "most circuits produce immediate sustained
activity"). This is real LUT physics and the one deliberate contrast with the
nervous net's no-spontaneous-activity invariant; the transient is scheduled
at t = 0 so it lands on the same time base as the first synchronous update.

With the default delay == TICK and stimuli on the integer tick lattice the
behaviour is bit-identical to the synchronous latched engine (lut.LutSim):
that engine is the quantization of this one, just as the old synchronous
nervous engine was the quantization of PulseSim. Off-lattice stimuli and
scaled delays exercise genuine continuous time (see tests/test_lut_synchrony).

Implementation: with one uniform delay, level changes cluster into WAVES
(times of the form input_edge + k*delay), so the simulation advances one
whole-grid numpy update per wave instead of per-cell Python events — the
event-driven semantics at (close to) the vectorised engine's speed. Waves on
a finite set of edge offsets are structurally bounded (no Zeno runs); a wave
cap plus the optional ``max_events`` rise cap keep a pathological genome from
monopolising evaluation, mirroring PulseSim's overflow contract.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from substrates.nervous.pulse import TICK


@dataclass(frozen=True)
class LutConfig:
    """Immutable physical timing parameters carried with one run.

    ``delay`` is each cell's propagation delay (the analogue of PulseConfig's
    node delay). There is no pulse width: LUT outputs are levels that persist
    until re-evaluated. ``wave_cap`` bounds the number of update waves in one
    run — far above any legitimate run, purely a runaway backstop.
    """
    delay:    float = 1.0
    wave_cap: int   = 200_000

    def __post_init__(self):
        if self.delay <= 0 or self.wave_cap < 1:
            raise ValueError('delay and wave_cap must be positive')


_INF = float('inf')


class AsyncLutSim:
    """Event-driven continuous-time simulation of a grown LUT array.

    Drop-in for the synchronous ``LutSim`` API (``step``, ``run_bits``,
    ``out``, ``ever``, ``reset``) and additionally speaks the PulseSim
    dialect: ``inject_pulse(cell, t, width)`` places a wired-OR level
    injection at any real time, ``advance_to(when)`` processes events through
    an absolute time, ``rise_times`` holds every wire's continuous 0->high
    leading-edge timestamps, and ``overflow`` marks a run that exceeded its
    event/wave budget (scorers turn overflow into zero).
    """

    def __init__(self, grid, config=None, max_events=None,
                 input_nodes=None, output_nodes=None):
        self.grid   = grid
        self.config = config or LutConfig()
        cells       = list(grid)
        self._cells = cells
        self.n      = n = len(cells)
        self._cidx  = {c: i for i, c in enumerate(cells)}
        self.input_nodes = set(input_nodes or ())
        self.output_nodes = set(output_nodes or ())
        self._input_mask = np.fromiter(
            (c in self.input_nodes for c in cells), dtype=bool, count=n)
        self._output_mask = np.fromiter(
            (c in self.output_nodes for c in cells), dtype=bool, count=n)
        if (self._input_mask & self._output_mask).any():
            raise ValueError('LUT terminal inputs and outputs must be distinct')
        self.max_events = None if max_events is None else max(1, int(max_events))
        # four LUT columns (same layout as LutSim; index n = padding sentinel)
        self._Ln = np.fromiter((grid[c][0] for c in cells), dtype=np.int64, count=n)
        self._Ls = np.fromiter((grid[c][1] for c in cells), dtype=np.int64, count=n)
        self._Le = np.fromiter((grid[c][2] for c in cells), dtype=np.int64, count=n)
        self._Lw = np.fromiter((grid[c][3] for c in cells), dtype=np.int64, count=n)

        def col(dx, dy):     # column of each cell's (dx,dy) neighbour (n = none)
            ci = self._cidx
            return np.fromiter((ci.get((c[0] + dx, c[1] + dy), n) for c in cells),
                               dtype=np.intp, count=n)
        self._nN, self._nS = col(0, 1), col(0, -1)
        self._nE, self._nW = col(1, 0), col(-1, 0)
        self.reset()

    # ── state / power-on ─────────────────────────────────────────────────────

    def reset(self):
        n = self.n
        self._logic   = np.zeros(n, dtype=np.int64)   # each cell's computed nibble
        self._inj     = np.zeros(n, dtype=np.int64)   # wired-OR injection depth
        self._injor   = np.zeros(n, dtype=np.int64)   # 0xF where depth > 0
        self._wirepad = np.zeros(n + 1, dtype=np.int64)  # levels (+ absent = 0)
        self._drivepad = np.zeros(n + 1, dtype=np.int64) # outward-visible levels
        self._pend_t  = np.full(n, _INF)              # inertial pending change
        self._pend_v  = np.zeros(n, dtype=np.int64)
        self._affbuf  = np.zeros(n + 1, dtype=bool)   # scratch reader mask
        self._edges   = []                            # heap: (t, seq, col, ±1)
        self._waves   = []                            # heap of pending-wave times
        self._wave_set = set()
        self._seq     = 0
        self._tick    = 0                             # next step() tick index
        self._ever    = np.zeros(n, dtype=bool)
        self._rise_log  = []                          # [(t, ndarray of columns)]
        self._fall_log  = []                          # [(t, ndarray of columns)]
        self._rise_dict = None
        self._interval_dict = None
        self._out_dict  = None
        self.event_count = 0
        self.wave_count  = 0
        self.overflow    = False
        self._pristine   = True      # no stimulus queued/processed yet
        # Power-on transient: every cell evaluates the all-zero neighbourhood.
        # LUTs with the index-0 bit set emit spontaneously; the change lands at
        # t = 0, matching the synchronous engine's first latched update.
        v = self._eval()
        live = v != 0
        if live.any():
            self._pend_v[live] = v[live]
            self._pend_t[live] = 0.0
            self._push_wave(0.0)

    def _eval(self):
        """Every cell's lookup of the CURRENT wire levels (one numpy pass).
        Nibble layout and index bits exactly as LutSim._advance."""
        pe = self._drivepad
        pe[:self.n] = self._wirepad[:self.n]
        pe[:self.n][self._output_mask] = 0
        idx = (((pe[self._nN] & 0x4) >> 2)         # from N -> bit0
               | ((pe[self._nS] & 0x8) >> 2)       # from S -> bit1
               | ((pe[self._nE] & 0x1) << 2)       # from E -> bit2
               | ((pe[self._nW] & 0x2) << 2))      # from W -> bit3
        out = ((((self._Ln >> idx) & 1) << 3)
               | (((self._Ls >> idx) & 1) << 2)
               | (((self._Le >> idx) & 1) << 1)
               | ((self._Lw >> idx) & 1))
        # Source terminals expose only external injection; their LUT cannot
        # consume neighbour state or generate a signal of its own.
        out[self._input_mask] = 0
        return out

    # ── event queue ──────────────────────────────────────────────────────────

    def _push_wave(self, t):
        if t not in self._wave_set:
            self._wave_set.add(t)
            heapq.heappush(self._waves, t)

    def inject_pulse(self, cell, t, width=None):
        """Drive ``cell``'s net high for [t, t+width) (wired-OR with anything
        else driving it), starting at any real time. Events are queued; call
        ``advance_to`` (or let ``step`` / ``run_bits`` advance) to process."""
        col = self._cidx.get(cell)
        if col is None or self._output_mask[col]:
            return
        self._pristine = False
        w = TICK if width is None else float(width)
        t = float(t)
        self._seq += 1
        heapq.heappush(self._edges, (t, self._seq, col, 1))
        self._seq += 1
        heapq.heappush(self._edges, (t + w, self._seq, col, -1))

    def _next_time(self):
        t = self._edges[0][0] if self._edges else _INF
        if self._waves and self._waves[0] < t:
            t = self._waves[0]
        return t

    def _run_until(self, t_end):
        while not self.overflow:
            t = self._next_time()
            if t > t_end:
                break
            self._process(t)

    def _process(self, t):
        """One wave: apply everything due at exactly time t, then re-evaluate
        the readers of every wire that changed (inertial delay — the newest
        evaluation supersedes any pending one)."""
        n = self.n
        wire = self._wirepad[:n]
        if self._waves and self._waves[0] == t:
            heapq.heappop(self._waves)
            self._wave_set.discard(t)
            due = self._pend_t == t
            if due.any():
                self._logic[due] = self._pend_v[due]
                self._pend_t[due] = _INF
        while self._edges and self._edges[0][0] == t:
            _, _, col, delta = heapq.heappop(self._edges)
            self._inj[col] += delta
            self._injor[col] = 0xF if self._inj[col] > 0 else 0
        new_wire = self._logic | self._injor
        changed = new_wire != wire
        if not changed.any():
            return
        rising = changed & (wire == 0)
        if rising.any():
            cols = np.nonzero(rising)[0]
            self._rise_log.append((t, cols))
            self._rise_dict = None
            self._interval_dict = None
            self._ever[cols] = True
            self.event_count += len(cols)
            if (self.max_events is not None
                    and self.event_count > self.max_events):
                self._overflow_now()
                return
        falling = changed & (wire != 0) & (new_wire == 0)
        if falling.any():
            # trailing edges complete the waveform log (pulse_intervals);
            # nibble-to-nibble changes that stay high are neither edge
            self._fall_log.append((t, np.nonzero(falling)[0]))
            self._interval_dict = None
        wire[...] = new_wire
        self._out_dict = None
        self.wave_count += 1
        if self.wave_count > self.config.wave_cap:
            self._overflow_now()
            return
        # The readers of a changed wire are its four grid neighbours (boolean
        # scatter into a scratch mask — cheaper than a sorted unique). On a
        # dense wave (a chaotic array flips most cells every delay) skip the
        # scatter and treat every cell as a reader: an unaffected cell's
        # lookup equals its current output, landing in the cancel branch —
        # EXCEPT a cell holding a pending update that is due later than this
        # wave (possible only when waves interleave within one delay, i.e.
        # off-lattice stimuli), whose pending must not be postponed; the
        # outstanding-pending check falls back to the exact mask then.
        # Sink changes are recorded and observable, but electrically invisible:
        # they have no readers to re-evaluate.
        drivers = changed & ~self._output_mask
        if not drivers.any():
            return
        ch = np.nonzero(drivers)[0]
        affm = self._affbuf
        if 4 * ch.size >= n and not (self._pend_t < _INF).any():
            aff = True
        else:
            affm[:] = False
            affm[self._nN[ch]] = True
            affm[self._nS[ch]] = True
            affm[self._nE[ch]] = True
            affm[self._nW[ch]] = True
            affm[n] = False                        # drop the off-grid sentinel
            aff = affm[:n]
            if not aff.any():
                return
        v = self._eval()
        needs = aff & (v != self._logic)
        self._pend_t[aff & ~needs] = _INF          # inertial cancellation
        if needs.any():
            self._pend_v[needs] = v[needs]
            t2 = t + self.config.delay
            self._pend_t[needs] = t2
            self._push_wave(t2)

    def _overflow_now(self):
        # A pathological genome must not monopolise evaluation: mark the run
        # invalid and stop deterministically (scorers turn overflow into zero).
        self.overflow = True
        self._edges = []
        self._waves = []
        self._wave_set.clear()

    # ── the PulseSim-dialect continuous-time interface ───────────────────────

    def advance_to(self, when):
        """Process queued events through an absolute physical time."""
        self._pristine = False
        self._run_until(float(when))

    def activity_at(self, _when=None):
        """Wire levels {cell: 0/1} at the current event frontier. LUT wires are
        levels, so the present state IS the state at any time processed since
        the last change; pass the frontier time you advanced to."""
        wire = self._wirepad[:self.n]
        return {self._cells[i]: (1 if wire[i] else 0) for i in range(self.n)}

    @property
    def rise_times(self):
        """{cell: [continuous leading-edge times]} — every wire's 0->high
        transitions, no tick quantisation (the substrate's behavioural output
        for event/cadence fitness). Built lazily; ``rise_trains`` is cheaper
        when only a few cells are wanted."""
        if self._rise_dict is None:
            d = {c: [] for c in self._cells}
            cells = self._cells
            for t, cols in self._rise_log:
                for i in cols:
                    d[cells[i]].append(t)
            self._rise_dict = d
        return self._rise_dict

    @property
    def pulse_intervals(self):
        """{cell: [[start, end]]} — the complete per-wire waveform log, in the
        PulseSim dialect: one [rise, fall] pair per pulse, ``end`` is
        float('inf') while the wire is still high. Level logic alternates
        rise/fall strictly per wire, so pairing the k-th rise with the k-th
        fall reconstructs the waveform exactly."""
        if self._interval_dict is None:
            rises = {c: [] for c in self._cells}
            falls = {c: [] for c in self._cells}
            cells = self._cells
            for t, cols in self._rise_log:
                for i in cols:
                    rises[cells[i]].append(t)
            for t, cols in self._fall_log:
                for i in cols:
                    falls[cells[i]].append(t)
            self._interval_dict = {
                c: [[start, falls[c][k] if k < len(falls[c]) else float('inf')]
                    for k, start in enumerate(rises[c])]
                for c in cells}
        return self._interval_dict

    def rise_trains(self, cells):
        """{cell: [leading-edge times]} for just the requested cells."""
        out = {c: [] for c in cells}
        wantmask = np.zeros(self.n, dtype=bool)
        wanted = [self._cidx[c] for c in cells if c in self._cidx]
        if not wanted:
            return out
        wantmask[wanted] = True
        names = self._cells
        for t, cols in self._rise_log:
            for i in cols[wantmask[cols]]:
                out[names[i]].append(t)
        return out

    # ── the LutSim-compatible per-tick interface ─────────────────────────────

    def step(self, input_vals):
        """Advance one tick. input_vals {cell: 0/1} drives the input nets as
        levels: a 1 holds the net high for this tick (consecutive 1s are one
        seamless level — the wired-OR merge leaves no dip and no extra edge).
        Returns {cell: 0/1}, every wire sampled mid-tick."""
        self._pristine = False
        t0 = self._tick * TICK
        for c, b in input_vals.items():
            if b:
                self.inject_pulse(c, t0, TICK)
        self._run_until(t0 + 0.5 * TICK)
        self._tick += 1
        wire = self._wirepad[:self.n]
        cells = self._cells
        return {cells[i]: (1 if wire[i] else 0) for i in range(self.n)}

    def run_bits(self, streams, in_pos, T):
        """Bulk scoring runner: T ticks driven by ``streams`` (rows of input
        bits for ``in_pos``; T past the end runs on zero input) returning a
        [T, ncells] uint8 mid-tick level matrix — same contract and, on the
        tick lattice with delay == TICK, the same bits as LutSim.run_bits.
        The trailing half tick is flushed so late edges are retained in
        ``rise_times``.

        On a pristine simulator with delay == TICK this takes the lattice
        fast path (see _run_bits_lattice); anything already queued or a
        non-TICK delay runs through the general event loop."""
        if self._pristine and self.config.delay == TICK and T > 0:
            B = self._run_bits_lattice(streams, in_pos, T)
        else:
            self._pristine = False
            in_cols = [self._cidx.get(p) for p in in_pos]
            ns = len(streams)
            B = np.zeros((T, self.n), dtype=np.uint8)
            wire = self._wirepad[:self.n]
            for t in range(T):
                if t < ns:
                    row = streams[t]
                    t0 = t * TICK
                    for i, c in enumerate(in_cols):
                        if (c is not None and not self._output_mask[c]
                                and i < len(row) and row[i]):
                            self._seq += 1
                            heapq.heappush(self._edges, (t0, self._seq, c, 1))
                            self._seq += 1
                            heapq.heappush(self._edges,
                                           (t0 + TICK, self._seq, c, -1))
                self._run_until((t + 0.5) * TICK)
                B[t] = wire != 0
                if self.overflow:
                    break
        if not self.overflow:
            self._run_until(T * TICK)
        return B

    def _run_bits_lattice(self, streams, in_pos, T):
        """The tick-lattice fast path. With delay == TICK and integer TICK-wide
        stimuli, the event loop reduces wave-for-wave to the synchronous
        latched update (the quantization contract audited against LutSim in
        tests/test_lut_synchrony.py), so run it as one vectorised loop: the
        asynchronous engine costs evolution almost nothing on lattice targets.
        Rise log, event counts and overflow behave identically; on exit the
        exact event-frontier state at (T - 0.5) is reconstructed — pending
        logic changes due at T and the final tick's still-live injection — so
        the ordinary event loop continues seamlessly (the caller's trailing
        flush to T processes them)."""
        n = self.n
        self._pristine = False
        # fold the power-on wave into the loop's first evaluation
        self._waves = []
        self._wave_set.clear()
        self._pend_t[:] = _INF
        in_cols = [self._cidx.get(p) for p in in_pos]
        ns = len(streams)
        B = np.zeros((T, n), dtype=np.uint8)
        wire = self._wirepad[:n]
        logic, injor, ever = self._logic, self._injor, self._ever
        last_inj = ()
        for t in range(T):
            v = self._eval()                   # logic_t = LUT(wire_{t-1})
            logic[...] = v
            injor[:] = 0
            last_inj = ()
            if t < ns:
                row = streams[t]
                cols = [c for i, c in enumerate(in_cols)
                        if (c is not None and not self._output_mask[c]
                            and i < len(row) and row[i])]
                if cols:
                    injor[cols] = 0xF
                    last_inj = cols
            new_wire = logic | injor
            rising = (wire == 0) & (new_wire != 0)
            falling = (wire != 0) & (new_wire == 0)
            wire[...] = new_wire
            B[t] = new_wire != 0
            if rising.any():
                rcols = np.nonzero(rising)[0]
                self._rise_log.append((t * TICK, rcols))
                ever[rcols] = True
                self.event_count += len(rcols)
                if (self.max_events is not None
                        and self.event_count > self.max_events):
                    self._overflow_now()
                    break
            if falling.any():
                self._fall_log.append((t * TICK, np.nonzero(falling)[0]))
                self._interval_dict = None
            self.wave_count += 1
            if self.wave_count > self.config.wave_cap:
                self._overflow_now()
                break
        self._rise_dict = None
        self._out_dict = None
        self._tick = T
        if not self.overflow:
            v = self._eval()
            needs = v != logic
            if needs.any():
                self._pend_v[needs] = v[needs]
                self._pend_t[needs] = T * TICK
                self._push_wave(T * TICK)
            for c in last_inj:
                self._inj[c] = 1
                injor[c] = 0xF
                self._seq += 1
                heapq.heappush(self._edges, (T * TICK, self._seq, c, -1))
        return B

    def run_input_events(self, input_events, in_pos, T, sample=True):
        """Drive an explicit floating-time stimulus schedule (one list of
        ``(start, width)`` pulses per input) and run to the T-tick horizon.
        Returns the [T, ncells] mid-tick sample matrix (zeros if ``sample`` is
        False — event fitness needs edges, not display snapshots)."""
        self._pristine = False
        for i, cell in enumerate(in_pos):
            events = input_events[i] if i < len(input_events) else ()
            for start, width in events:
                self.inject_pulse(cell, float(start), float(width))
        B = np.zeros((T if sample else 0, self.n), dtype=np.uint8)
        if sample:
            wire = self._wirepad[:self.n]
            for t in range(T):
                self._run_until((t + 0.5) * TICK)
                B[t] = wire != 0
                if self.overflow:
                    break
        if not self.overflow:
            self._run_until(T * TICK)
        return B

    @property
    def out(self):
        """{cell: nibble} — current wire levels (post-injection), the same
        emission map LutSim.out exposes for playback drawing."""
        if self._out_dict is None:
            wire = self._wirepad[:self.n]
            self._out_dict = {self._cells[i]: int(wire[i]) for i in range(self.n)}
        return self._out_dict

    @property
    def ever(self):
        e = self._ever
        return {self._cells[i]: (1 if e[i] else 0) for i in range(self.n)}
