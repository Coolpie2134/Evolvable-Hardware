"""
substrates/nervous/playback.py - asynchronous (continuous-time) playback and a clickable
pulse-timeline, shared by the Interactive and Designer substrate views.

Both asynchronous substrates - the nervous net (substrates/nervous/pulse.py) and the LUT
array (substrates/lut/pulse.py) - are continuous-time, event-driven elements whose
fitness reads real edge timestamps.  These helpers let the GUI DRIVE and SHOW
them in that same continuous time instead of a synchronous tick lattice:

  * AsyncPlayer drives any simulator speaking the pulse dialect
    (``inject_pulse`` / ``advance_to`` / ``activity_at`` / ``rise_times`` /
    ``overflow``): it injects pulse intervals at their real times and advances
    a physical-time cursor, reporting wire activity and the real leading-edge
    times of any cell. No GUI dependency.
  * NervousPlayer builds it over the configured digital or analog simulator -
    the same engine used by Nervous evolution. The LUT twin (substrates.lut.playback.LutPlayer)
    builds it over AsyncLutSim.
  * PulseLaneEditor binds a matplotlib axis to editable input pulses: click for
    the default width, drag for a custom width, or click one to remove it. It draws
    the lanes and a moving playback cursor.  Both tabs and both substrates reuse
    it, so the fiddly event wiring lives in one place.
"""
from __future__ import annotations

import math

from . import pulse as pulse_engine
from .simulation import create_simulator, streams_to_schedule

DEFAULT_DT      = 0.5     # playback resolution (sub-tick; WIDTH defaults to 1 TICK)
DEFAULT_HORIZON = 40.0    # physical-time window shown / played
PLAY_MAX_EVENTS = 20000   # runaway guard for a pathological oscillator


class AsyncPlayer:
    """Continuous-time playback over any pulse-dialect simulator (GUI-free,
    testable). Subclasses supply ``_make_sim()`` for their substrate."""

    def __init__(self, horizon=DEFAULT_HORIZON, dt=DEFAULT_DT,
                 pulse_width=None, default_width=None):
        self.horizon = float(horizon)
        self.dt      = float(dt)
        base = pulse_engine.WIDTH if default_width is None else default_width
        self.pulse_width = (float(base) if pulse_width is None
                            else float(pulse_width))
        self.schedule = {}                    # {input_cell: [(start, width)]}
        self.reset()

    def _make_sim(self):
        raise NotImplementedError

    def set_schedule(self, schedule):
        """Set ``{input_cell: pulses}``, where a pulse is ``time`` or
        ``(time, width)``. The tuple form exposes physical width coding while
        the scalar form keeps the Designer's quick-click default."""
        self.schedule = {}
        for cell, pulses in schedule.items():
            normalized = []
            for item in pulses:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    start, width = float(item[0]), float(item[1])
                else:
                    start, width = float(item), self.pulse_width
                normalized.append((start, width))
            self.schedule[cell] = sorted(normalized)
        self.reset()

    def reset(self):
        self.sim = self._make_sim()
        for cell, pulses in self.schedule.items():
            for start, width in pulses:
                self.sim.inject_pulse(cell, start, width)
        self.cursor = 0.0
        self.sim.advance_to(0.0)

    def step(self):
        """Advance the playback cursor by one dt of real time."""
        self.cursor = min(self.horizon, round(self.cursor + self.dt, 6))
        self.sim.advance_to(self.cursor)
        return self.cursor

    def at_end(self):
        return self.cursor >= self.horizon - 1e-9

    def activity(self):
        """{cell: 0/1} - which wires are high at the current cursor time."""
        return self.sim.activity_at(self.cursor)

    def events_upto(self, cell):
        """Real leading-edge times of ``cell`` seen so far (<= cursor)."""
        # Edges are recorded only when ``advance_to`` processes them, and the
        # player never advances beyond cursor. Avoid filtering/allocation here.
        return self.sim.rise_times.get(cell, ())

    @property
    def overflow(self):
        return self.sim.overflow


class NervousPlayer(AsyncPlayer):
    """Continuous-time playback of one grown nervous net."""

    def __init__(self, grid, routing, horizon=DEFAULT_HORIZON, dt=DEFAULT_DT,
                 pulse_width=None, max_events=PLAY_MAX_EVENTS, config=None,
                 delays=None, arch='single', inputs=None, outputs=None,
                 terminal_inputs=None):
        self.grid    = grid
        self.routing = routing
        self.config  = config
        self.max_events = max_events
        # Per-cell delays for width-preserving transport; that model copies the
        # incoming waveform's duration dynamically rather than regenerating it.
        self.delays  = delays
        self.arch    = arch
        self.inputs  = list(inputs or ())
        self.outputs = list(outputs or ())
        self.terminal_inputs = list(terminal_inputs or ())
        default_width = (pulse_engine.WIDTH if config is None else config.width)
        super().__init__(horizon=horizon, dt=dt, pulse_width=pulse_width,
                         default_width=default_width)

    def _make_sim(self):
        if self.arch == 'tri3':
            from .tritile import TriSim
            return TriSim(self.grid, self.inputs, config=self.config,
                          max_events=self.max_events, outputs=self.outputs)
        if self.arch != 'single':
            raise ValueError('unknown tile architecture: %r' % (self.arch,))
        return create_simulator(self.grid, self.routing,
                                max_events=self.max_events, config=self.config,
                                delays=self.delays,
                                input_nodes=self.terminal_inputs,
                                output_nodes=self.outputs)


# Display-only RC constants for capacitor-style playback (charge_levels).
# Fast charge while a wire pulses, slower visible discharge after it falls -
# these shape the ANIMATION only; the scored physics stays the binary pulse.
CHARGE_TAU    = 0.30
DISCHARGE_TAU = 0.90


def charge_levels(sim, t, charge_tau=CHARGE_TAU, discharge_tau=DISCHARGE_TAU):
    """{cell: 0..1} capacitor-style charge of every wire at time ``t``.

    Models each node as an RC follower of its own binary waveform: charge
    rises toward 1 while the wire is high (time constant ``charge_tau``) and
    decays toward 0 after it falls (``discharge_tau``), so playback shows
    nodes charging and discharging like the paper's capacitively-coupled
    hardware instead of snapping on/off. Purely cosmetic - reads the engine's
    ``pulse_intervals`` waveform log and never touches the simulation.
    Returns None when the simulator has no waveform log."""
    intervals = getattr(sim, 'pulse_intervals', None)
    if intervals is None:
        return None
    t = float(t)
    levels = {}
    for cell, pulses in intervals.items():
        charge, frontier = 0.0, None
        for start, end in pulses:
            start = float(start)
            if start > t:
                break
            if frontier is not None and start > frontier:
                charge *= math.exp(-(start - frontier) / discharge_tau)
            high_until = min(float(end), t)
            if high_until > start:
                charge = 1.0 + (charge - 1.0) * math.exp(
                    -(high_until - start) / charge_tau)
            frontier = high_until
        if frontier is not None and t > frontier:
            charge *= math.exp(-(t - frontier) / discharge_tau)
        levels[cell] = charge
    return levels


def pulses_from_trial(target, n_inputs, trial_index=0):
    """Physical input intervals from one stored trial (default: the first).

    ``trial_index`` selects which test case to load (clamped to the stored
    range), so playback UIs can step through the whole bank instead of only
    ever showing trial 0. Widths are retained so width-preserving playback
    receives the same waveform used by scoring rather than silently
    substituting the global default width.
    """
    out = [[] for _ in range(n_inputs)]
    trials = getattr(target, 'trials', None) if target is not None else None
    if not trials:
        return out
    trial = trials[max(0, min(int(trial_index), len(trials) - 1))]
    physical = getattr(trial, 'input_events', None)
    if physical is not None:
        for i in range(n_inputs):
            out[i] = ([(float(start), float(width)) for start, width in physical[i]]
                      if i < len(physical) else [])
        return out
    streams = trial.streams
    return streams_to_schedule(
        streams, n_inputs, len(streams),
        config=getattr(target, 'pulse_config', None))


def pulses_from_case(target, n_inputs, case_index=0, backend='lut',
                     trial_index=0):
    """Input pulse lanes for one COMBINATIONAL truth-table case, matching the
    presentation that backend's fitness actually scores.

    Combinational targets carry ``cases`` (a truth table), not temporal
    ``trials``, so a playback UI must use this rather than
    :func:`pulses_from_trial` (which would find no trials and show nothing) to
    reproduce what fitness tested. The LUT array scores each case as a battery of
    pulses that share a rising edge after a delay and hold for random per-input
    widths (see ``substrates.lut.ga._combinational_schedule``), reading the output while
    they are held; one representative trial of that battery is shown here. Every
    other backend holds the active inputs for the whole run (``score_nervous``
    semantics), shown as one long pulse."""
    out = [[] for _ in range(n_inputs)]
    cases = getattr(target, 'cases', None)
    if not cases:
        return out
    in_bits = cases[max(0, min(int(case_index), len(cases) - 1))][0]
    if backend == 'lut':
        from substrates.lut.ga import _combinational_schedule
        schedule = _combinational_schedule(target)
        delay, widths = schedule[max(0, min(int(trial_index),
                                            len(schedule) - 1))]
        for i, bit in enumerate(in_bits):
            if bit and i < n_inputs:
                out[i] = [(float(delay), float(widths[i]))]
    else:                                   # held for the whole run (one edge)
        horizon = float(max(24, getattr(target, 'T', 24) or 24))
        for i, bit in enumerate(in_bits):
            if bit and i < n_inputs:
                out[i] = [(0.5, horizon)]
    return out


class PulseLaneEditor:
    """A clickable per-input pulse timeline drawn on a matplotlib axis."""

    def __init__(self, ax, canvas, labels, horizon=DEFAULT_HORIZON, snap=0.5,
                 on_change=None, default_width=None):
        self.ax        = ax
        self.canvas    = canvas
        self.labels    = list(labels)
        self.horizon   = float(horizon)
        self.snap      = float(snap)
        self.on_change = on_change
        self.default_width = float(
            pulse_engine.WIDTH if default_width is None else default_width)
        self.pulses    = [[] for _ in self.labels]   # per-input (start, width)
        self.cursor    = 0.0
        self._cursor_artist = None
        self._drag = None
        self.cid = canvas.mpl_connect('button_press_event', self._on_click)
        self.release_cid = canvas.mpl_connect(
            'button_release_event', self._on_release)

    def schedule(self, in_cells):
        """Map lane pulses onto their grid cells: {cell: [(start, width)]}.

        ``in_cells`` may carry per-input attachment GROUPS (evolvable I/O
        binding): lane i's pulses are replicated onto every cell of group i,
        and a cell shared by several lanes receives their merged train (the
        pulse engines wired-OR overlapping injections). A flat cell list maps
        1:1 exactly as before."""
        from .io_placement import input_groups
        groups = input_groups(in_cells)
        sched = {}
        for i in range(min(len(groups), len(self.pulses))):
            for cell in groups[i]:
                sched.setdefault(cell, []).extend(self.pulses[i])
        return {cell: sorted(pulses) for cell, pulses in sched.items()}

    def set_pulses(self, pulses):
        self.pulses = []
        for row in pulses:
            normalized = []
            for item in row:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    normalized.append((float(item[0]), float(item[1])))
                else:
                    normalized.append((float(item), self.default_width))
            self.pulses.append(sorted(normalized))
        while len(self.pulses) < len(self.labels):
            self.pulses.append([])
        self.draw()

    def clear(self):
        self.pulses = [[] for _ in self.labels]
        self.draw()
        if self.on_change:
            self.on_change()

    def set_cursor(self, t, redraw=True):
        self.cursor = float(t)
        if self._cursor_artist is None:
            self.draw()
            return
        self._cursor_artist.set_xdata([self.cursor, self.cursor])
        if redraw:
            self.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or ev.ydata is None:
            return
        lane = int(round(ev.ydata))
        if not (0 <= lane < len(self.labels)):
            return
        t = max(0.0, round(ev.xdata / self.snap) * self.snap)
        if t >= self.horizon:
            return
        existing = next(((start, width) for start, width in self.pulses[lane]
                         if start - self.snap * 0.75 <= t
                         <= start + width + self.snap * 0.75), None)
        if existing is not None:
            self.pulses[lane].remove(existing)
            self.draw()
            if self.on_change:
                self.on_change()
            return
        self._drag = (lane, t)

    def _on_release(self, ev):
        if self._drag is None:
            return
        lane, start = self._drag
        self._drag = None
        end = start
        if ev.inaxes is self.ax and ev.xdata is not None and ev.ydata is not None:
            release_lane = int(round(ev.ydata))
            if release_lane == lane:
                end = max(0.0, round(ev.xdata / self.snap) * self.snap)
        width = (end - start if end > start + self.snap * 0.5
                 else self.default_width)
        width = max(self.snap, min(width, self.horizon - start))
        self.pulses[lane].append((start, width))
        self.pulses[lane].sort()
        self.draw()
        if self.on_change:
            self.on_change()

    def draw(self):
        ax = self.ax
        ax.clear()
        n = max(1, len(self.labels))
        ax.set_xlim(0, self.horizon)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(range(len(self.labels)))
        ax.set_yticklabels(self.labels, fontsize=8)
        ax.set_xlabel('continuous time (seconds)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title('input pulses - click for default width; drag for custom width',
                     fontsize=8)
        for lane in range(len(self.labels)):
            ax.axhline(lane, color='#e6e6e6', lw=0.6, zorder=0)
            for start, width in self.pulses[lane]:
                ax.plot([start, start + width], [lane, lane], lw=5,
                        solid_capstyle='butt', color='#b02020', zorder=3)
                ax.plot([start, start + width], [lane, lane], marker='|',
                        ls='none', ms=12, mew=1.5,
                        color='#7c1010', zorder=4)
        self._cursor_artist = ax.axvline(
            self.cursor, color='#e8a33d', lw=1.5, alpha=0.9, zorder=2)
        self.canvas.draw_idle()

    def disconnect(self):
        try:
            self.canvas.mpl_disconnect(self.cid)
            self.canvas.mpl_disconnect(self.release_cid)
        except Exception:
            pass
