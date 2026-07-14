"""
nv_evo/playback.py — asynchronous (continuous-time) playback and a clickable
pulse-timeline, shared by the Interactive and Designer substrate views.

Both asynchronous substrates — the nervous net (nv_evo/pulse.py) and the LUT
array (lut_evo/pulse.py) — are continuous-time, event-driven elements whose
fitness reads real edge timestamps.  These helpers let the GUI DRIVE and SHOW
them in that same continuous time instead of a synchronous tick lattice:

  * AsyncPlayer drives any simulator speaking the pulse dialect
    (``inject_pulse`` / ``advance_to`` / ``activity_at`` / ``rise_times`` /
    ``overflow``): it injects pulse intervals at their real times and advances
    a physical-time cursor, reporting wire activity and the real leading-edge
    times of any cell. No GUI dependency.
  * NervousPlayer builds it over the paper-faithful PulseSim — the same event
    engine used by Nervous evolution. The LUT twin (lut_evo.playback.LutPlayer)
    builds it over AsyncLutSim.
  * PulseLaneEditor binds a matplotlib axis to editable input pulses: click for
    the default width, drag for a custom width, or click one to remove it. It draws
    the lanes and a moving playback cursor.  Both tabs and both substrates reuse
    it, so the fiddly event wiring lives in one place.
"""
from __future__ import annotations

from . import pulse as pulse_engine
from .simulation import create_simulator

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
        """{cell: 0/1} — which wires are high at the current cursor time."""
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
                 pulse_width=None, max_events=PLAY_MAX_EVENTS, config=None):
        self.grid    = grid
        self.routing = routing
        self.config  = config
        self.max_events = max_events
        default_width = (pulse_engine.WIDTH if config is None else config.width)
        super().__init__(horizon=horizon, dt=dt, pulse_width=pulse_width,
                         default_width=default_width)

    def _make_sim(self):
        return create_simulator(self.grid, self.routing,
                                max_events=self.max_events, config=self.config)


def pulses_from_trial(target, n_inputs):
    """Leading-edge times of each input in the target's first trial — a sensible
    default schedule to prefill the pulse timeline with. Returns n_inputs lists."""
    out = [[] for _ in range(n_inputs)]
    trials = getattr(target, 'trials', None) if target is not None else None
    if not trials:
        return out
    physical = getattr(trials[0], 'input_events', None)
    if physical is not None:
        for i in range(n_inputs):
            out[i] = ([float(start) for start, _width in physical[i]]
                      if i < len(physical) else [])
        return out
    streams = trials[0].streams
    for i in range(n_inputs):
        out[i] = [float(t) for t in range(len(streams))
                  if i < len(streams[t]) and streams[t][i]
                  and (t == 0 or not streams[t - 1][i])]
    return out


def toggle_pulse(times, t, snap):
    """Add ``t`` to a sorted pulse list, or remove the nearby one if present.
    Pure helper so the click behaviour is testable without a canvas."""
    near = [p for p in times if abs(p - t) < snap * 0.75]
    if near:
        for p in near:
            times.remove(p)
    else:
        times.append(t)
        times.sort()
    return times


class PulseLaneEditor:
    """A clickable per-input pulse timeline drawn on a matplotlib axis."""

    def __init__(self, ax, canvas, labels, horizon=DEFAULT_HORIZON, snap=0.5,
                 on_change=None):
        self.ax        = ax
        self.canvas    = canvas
        self.labels    = list(labels)
        self.horizon   = float(horizon)
        self.snap      = float(snap)
        self.on_change = on_change
        self.pulses    = [[] for _ in self.labels]   # per-input (start, width)
        self.cursor    = 0.0
        self._cursor_artist = None
        self._drag = None
        self.cid = canvas.mpl_connect('button_press_event', self._on_click)
        self.release_cid = canvas.mpl_connect(
            'button_release_event', self._on_release)

    def schedule(self, in_cells):
        """Map lane pulses onto their grid cells: {cell: [times]}."""
        return {in_cells[i]: list(self.pulses[i])
                for i in range(min(len(in_cells), len(self.pulses)))}

    def set_pulses(self, pulses):
        self.pulses = []
        for row in pulses:
            normalized = []
            for item in row:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    normalized.append((float(item[0]), float(item[1])))
                else:
                    normalized.append((float(item), float(pulse_engine.WIDTH)))
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
                 else float(pulse_engine.WIDTH))
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
        ax.set_title('input pulses — click for default width; drag for custom width',
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
