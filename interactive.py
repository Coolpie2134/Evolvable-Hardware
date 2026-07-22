"""
interactive.py — an "Interactive / Test" tab for the Evolvable-Hardware GUI.

After a circuit is evolved (or loaded), this lets you drive its inputs and watch
the response. All three backends share Step / Run / Reset controls, while each
uses the input and time representation appropriate to its physics:

  * SNN      — LIF playback: the membrane potentials charge (grey → hot), neurons
               flash as they spike, and the signal wave propagates along the
               excitatory (green) / inhibitory (red) synapses. Alongside the
               network: a spike raster and the output neurons' membrane traces vs
               their fire thresholds. The 20 ms response is precomputed once
               (`simulate_trace`) and scrubbed frame by frame.
  * Nervous  — ASYNCHRONOUS continuous-time playback: place input pulses on a
               clickable timeline, then Step / Run in real (possibly sub-tick)
               time and watch pulses propagate with their actual delays, loops
               latch, and oscillators run. It uses the paper-faithful PulseSim,
               the same asynchronous event engine used by Nervous evolution.
  * LUT      — the SAME timeline and continuous-time playback, driving the
               asynchronous level-logic engine (AsyncLutSim, the same one LUT
               evolution scores with): watch the cells' four directional
               lookup outputs propagate with their real gate delay, levels
               instead of pulses.

Kept in its own module so app.py stays lean: app.py builds one tab and hands it a
callback (`app.current_circuit()`) returning the genome / target / backend / arch.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from snn_evo import grow_snn, interpret_grid, simulate_trace, draw_snn_net, N_STEPS, DT
from nv_evo import grow_nervous, interpret_nervous, place_outputs_by_trace
from nv_evo.viz import draw_hex_net
from nv_evo.contracts import behavior_contract_badge
from nv_evo.playback import (NervousPlayer, PulseLaneEditor, pulses_from_trial,
                             pulses_from_case, charge_levels)
from lut_evo import grow_lut
from lut_evo.playback import LutPlayer


from lut_evo import place_outputs_by_trace as lut_place_by_trace
from lut_evo.ga import _place_outputs_combinational as lut_place_combinational
from lut_evo.viz import draw_lut_net

# LIF playback advances this many 0.1 ms *display samples* per frame; the
# underlying LIF response is event-driven rather than stepped.
SNN_STRIDE = 5


class InteractiveTab:
    def __init__(self, parent, get_circuit):
        self.parent      = parent
        self.get_circuit = get_circuit         # () -> dict | None
        self._inputs     = []                  # list of (label, BooleanVar) — SNN only
        self._circuit    = None                # current loaded circuit context
        self._state      = {}                  # nervous activity map
        self._running    = False
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.parent, padding=(6, 4))
        top.pack(fill='x')
        ttk.Button(top, text='Load current solution', command=self.sync).pack(side='left')
        ttk.Label(top, text='   Inputs:').pack(side='left')
        self._inbar = ttk.Frame(top)
        self._inbar.pack(side='left')

        self._ctrl = ttk.Frame(self.parent, padding=(6, 0))
        self._ctrl.pack(fill='x')

        self._contract_var = tk.StringVar(
            value='Behavior Contract v1 will appear when a circuit is loaded.')
        ttk.Label(self.parent, textvariable=self._contract_var, anchor='w',
                  foreground='#245a8d', padding=(9, 1),
                  justify='left').pack(fill='x')

        self.fig = plt.figure(figsize=(9, 5))
        self.fig.patch.set_facecolor('#f5f5f5')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self._placeholder('Evolve or Load a solution, then click "Load current solution".')

        self._status = tk.StringVar(value='')
        ttk.Label(self.parent, textvariable=self._status, anchor='w',
                  relief='sunken', padding=(6, 2), wraplength=900,
                  justify='left').pack(fill='x', side='bottom', padx=4, pady=(0, 4))

    def _placeholder(self, msg):
        self.fig.clf()
        ax = self.fig.add_subplot(111); ax.axis('off')
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='#888')
        self.canvas.draw_idle()

    # ── load a circuit ────────────────────────────────────────────────────────────

    def sync(self):
        c = self.get_circuit()
        if not c or c.get('genome') is None:
            self._status.set('No solution yet — Run or Load Saved first.')
            return
        self._stop()
        self._circuit = c
        target, backend = c['target'], c['backend']
        self._backend = backend
        self._contract_var.set(behavior_contract_badge(target))

        # drop any prior pulse-timeline click handler before rebuilding
        if getattr(self, '_editor', None) is not None:
            self._editor.disconnect()
            self._editor = None

        # input bar: the nervous net is asynchronous, so it is driven by a
        # clickable pulse TIMELINE; LUT/SNN keep level toggles.
        for w in self._inbar.winfo_children():
            w.destroy()
        self._inputs = []
        self._case_cb = None
        # Temporal targets enumerate stored trials; combinational targets have no
        # trials, only a truth table (`cases`) presented as fitness-matched input
        # pulses. Either way the dropdown loads exactly what fitness scored.
        self._case_kind = ('trials' if getattr(target, 'trials', None)
                           else 'cases' if getattr(target, 'cases', None)
                           else None)
        if backend in ('nervous', 'lut'):
            # every stored test case is loadable, not just trial 0: pick one
            # from the dropdown to see exactly what fitness scored, then edit
            # the timeline freely (the box flips to '(custom schedule)').
            if self._case_kind:
                ttk.Label(self._inbar, text='Case:').pack(side='left',
                                                          padx=(4, 2))
                self._case_var = tk.StringVar()
                self._case_cb = ttk.Combobox(
                    self._inbar, textvariable=self._case_var, state='readonly',
                    width=34, values=self._case_labels(target))
                self._case_cb.current(0)
                self._case_cb.pack(side='left', padx=(0, 6))
                self._case_cb.bind('<<ComboboxSelected>>',
                                   self._on_case_selected)
            ttk.Label(self._inbar, text='pulses: click the timeline',
                      foreground='#888').pack(side='left', padx=4)
            ttk.Button(self._inbar, text='Clear', width=6,
                       command=self._nv_clear_pulses).pack(side='left', padx=3)
        else:
            for i in range(len(target.inputs)):
                v = tk.BooleanVar(value=False)
                lbl = chr(65 + i) if i < 26 else 'i%d' % i
                ttk.Checkbutton(self._inbar, text=lbl, variable=v,
                                command=self._on_toggle).pack(side='left', padx=2)
                self._inputs.append((lbl, v))

        # controls
        for w in self._ctrl.winfo_children():
            w.destroy()
        if backend == 'nervous':
            self._nv_arch = getattr(c['genome'], 'arch', 'single')
            self._grid = grow_nervous(c['genome'], seeds=tuple(target.inputs),
                                      grid_size=target.grid_size, iters=target.iters)
            self._routing, self._in_pos, self._out_pos = interpret_nervous(
                self._grid, target, arch=self._nv_arch)
            # Per-node delays feed BOTH the trace-matched placement (so the
            # highlighted output cell is the one fitness actually read) and the
            # playback engine below. node_delays returns None off the
            # width-preserving model, so this is uniform-safe.
            from nv_evo.nervous import node_delays
            pulse_config = getattr(target, 'pulse_config', None)
            self._nv_delays = (None if self._nv_arch == 'tri3' else
                               node_delays(c['genome'], self._grid, pulse_config))
            if getattr(target, 'temporal', False):
                # show the same output cell the fitness reads (trace-matched)
                self._out_pos, _ = place_outputs_by_trace(
                    self._grid, self._routing, self._in_pos, target,
                    delays=self._nv_delays,
                    arch=self._nv_arch)
            self._setup_async(target)
            self._playback_controls(
                '   (click the timeline to place input pulses; Step/Run in real time)')
            self._reset()
        elif backend == 'lut':                             # LUT array (async levels)
            self._grid = grow_lut(c['genome'], seeds=tuple(target.inputs),
                                  grid_size=target.grid_size, iters=target.iters)
            self._in_pos = list(target.inputs)
            if getattr(target, 'temporal', False):
                self._out_pos, _ = lut_place_by_trace(self._grid, self._in_pos, target)
            else:
                self._out_pos = lut_place_combinational(self._grid, target)
            self._setup_async(target)
            self._playback_controls(
                '   (click the timeline to place input pulses; Step/Run in real time)')
            self._reset()
        else:                                              # SNN — LIF playback
            self._grid = grow_snn(c['genome'], seeds=tuple(target.inputs),
                                  grid_size=target.grid_size, iters=target.iters)
            self._neurons, self._synapses = interpret_grid(
                self._grid, target=target, arch=c['arch'])
            self._in_ids = []
            for p in target.inputs:
                nn = next((m for m in self._neurons
                           if (m.x, m.y) == p and m.is_input), None)
                self._in_ids.append(nn.id if nn else None)
            self._playback_controls(
                '   (toggle an input to inject one source pulse; Step/Run watches its wave)')
            self._reset()
        self._status.set('Loaded %s [%s] — drive the inputs.' % (target.name, backend))

    def _playback_controls(self, hint):
        ttk.Button(self._ctrl, text='Step', command=self._step).pack(side='left', padx=3)
        self._run_btn = ttk.Button(self._ctrl, text='Run', command=self._toggle_run)
        self._run_btn.pack(side='left', padx=3)
        ttk.Button(self._ctrl, text='Reset', command=self._reset).pack(side='left', padx=3)
        ttk.Label(self._ctrl, text=hint, foreground='#888').pack(side='left', padx=6)

    # ── nervous + LUT: asynchronous continuous-time playback ──────────────────────

    def _setup_async(self, target):
        labels = [chr(65 + i) if i < 26 else 'i%d' % i
                  for i in range(len(self._in_pos))]
        horizon = float(max(24, getattr(target, 'T', 24) or 24))
        # Combinational LUT pulses arrive after a delay and are held for wide
        # widths (see _combinational_schedule); stretch the timeline so the whole
        # held window fitness reads is visible.
        if self._case_kind == 'cases' and self._backend == 'lut':
            from lut_evo.ga import _combinational_schedule
            schedule = _combinational_schedule(target)
            horizon = max(horizon,
                          max(d + max(ws) for d, ws in schedule) + 4.0)
        pulse_config = getattr(target, 'pulse_config', None)
        default_width = getattr(pulse_config, 'width', None)
        self.fig.clf()
        gs = self.fig.add_gridspec(3, 2, height_ratios=[0.55, 0.5, 0.5],
                                   width_ratios=[1.1, 1.0], hspace=0.85,
                                   wspace=0.25)
        self._ax_lanes = self.fig.add_subplot(gs[0, :])
        self._axg = self.fig.add_subplot(gs[1:, 0])
        self._axt = self.fig.add_subplot(gs[1, 1])
        self._axw = self.fig.add_subplot(gs[2, 1])
        self._editor = PulseLaneEditor(self._ax_lanes, self.canvas, labels,
                                       horizon=horizon, snap=0.5,
                                       on_change=self._nv_schedule_changed,
                                       default_width=default_width)
        self._editor.set_pulses(self._case_pulses(target, 0))
        if self._backend == 'nervous':
            config = pulse_config
            # sync() and shared with the trace-matched placement, so playback and
            # the highlighted output cell agree on the physics.
            self._player = NervousPlayer(
                self._grid, self._routing, horizon=horizon,
                max_events=getattr(target, 'max_events', 2048),
                config=config, delays=getattr(self, '_nv_delays', None),
                arch=getattr(self, '_nv_arch', 'single'), inputs=self._in_pos)
        else:                                  # LUT — same player, level engine
            self._player = LutPlayer(
                self._grid, horizon=horizon,
                config=getattr(target, 'lut_config', None))
        self._player.set_schedule(self._editor.schedule(self._in_pos))

    def _case_kind_for(self, target):
        """'trials' (temporal), 'cases' (combinational truth table), or None.
        Falls back to deriving from the target if the setup attribute is unset
        (e.g. a unit test constructing the tab directly)."""
        kind = getattr(self, '_case_kind', None)
        if kind is not None:
            return kind
        return ('trials' if getattr(target, 'trials', None)
                else 'cases' if getattr(target, 'cases', None) else None)

    def _case_pulses(self, target, index):
        """The input pulse lanes for case/trial ``index``, matching what fitness
        scores for this backend (temporal trial vs combinational truth table)."""
        n_inputs = len(self._in_pos)
        if self._case_kind_for(target) == 'cases':
            return pulses_from_case(target, n_inputs, index,
                                    getattr(self, '_backend', 'lut'))
        return pulses_from_trial(target, n_inputs, index)

    def _case_labels(self, target):
        """One dropdown row per stored test case: 'Case k/N: ...'."""
        n_inputs = len(target.inputs)
        if self._case_kind_for(target) == 'cases':
            labels = []
            for index, (in_bits, out_bits) in enumerate(target.cases):
                labels.append('Case %d/%d: in=%s -> %s'
                              % (index + 1, len(target.cases),
                                 ''.join(map(str, in_bits)),
                                 ''.join(map(str, out_bits))))
            return labels
        count = len(target.trials)
        labels = []
        for index in range(count):
            lanes = pulses_from_trial(target, n_inputs, index)
            parts = []
            for lane, events in enumerate(lanes):
                if not events:
                    continue
                name = chr(65 + lane) if lane < 26 else 'i%d' % lane
                times = ', '.join('%g' % round(start, 2)
                                  for start, _ in events)
                parts.append('%s[%s]' % (name, times))
            labels.append('Case %d/%d: %s'
                          % (index + 1, count,
                             '  '.join(parts) if parts else 'silent'))
        return labels

    def _on_case_selected(self, _evt=None):
        if not self._circuit or getattr(self, '_editor', None) is None:
            return
        index = self._case_cb.current()
        if index < 0:
            return
        self._loading_case = True
        try:
            self._editor.set_pulses(
                self._case_pulses(self._circuit['target'], index))
            self._nv_schedule_changed()   # set_schedule resets playback to t=0
        finally:
            self._loading_case = False

    def _nv_schedule_changed(self):
        self._stop()
        # A hand edit means the timeline no longer matches the selected case.
        if (getattr(self, '_case_cb', None) is not None
                and not getattr(self, '_loading_case', False)):
            self._case_var.set('(custom schedule)')
        self._player.set_schedule(self._editor.schedule(self._in_pos))
        self._draw_async()

    def _nv_clear_pulses(self):
        if getattr(self, '_editor', None) is not None:
            self._editor.clear()

    def _draw_async(self):
        target = self._circuit['target']
        # Update only the cursor artist; the final draw below paints all axes.
        self._editor.set_cursor(self._player.cursor, redraw=False)
        self._axg.clear()
        title = 't = %.1f%s' % (self._player.cursor,
                                '   (event cap hit)' if self._player.overflow
                                else '')
        if self._backend == 'nervous':
            # capacitor-style playback: nodes charge while pulsing and fade
            # after the pulse ends (display only — physics stays binary)
            activity = (charge_levels(self._player.sim, self._player.cursor)
                        or self._player.activity())
            draw_hex_net(self._axg, self._grid, target.grid_size,
                         routing=self._routing,
                         in_pos=self._in_pos, out_pos=self._out_pos,
                         activity=activity,
                         show_edges=(getattr(self, '_nv_arch', 'single') == 'single'),
                         arch=getattr(self, '_nv_arch', 'single'),
                         title=title)
        else:
            in_pos = [p for p in self._in_pos if p in self._grid]
            draw_lut_net(self._axg, self._grid, activity=self._player.nibbles(),
                         in_pos=in_pos, out_pos=self._out_pos, show_edges=True,
                         title=title)
        self._draw_event_strip(self._axt)
        self._draw_width_strip(self._axw)
        self.canvas.draw_idle()
        self._status.set('t = %.1f time units   output edges: %s' % (
            self._player.cursor, self._nv_output_summary()))

    def _draw_event_strip(self, axt):
        target = self._circuit['target']
        axt.clear()
        for k, term in enumerate(target.outputs):
            cell = self._out_pos.get(term.role)
            evs = self._player.events_upto(cell) if cell else []
            if evs:
                axt.vlines(evs, k - 0.4, k + 0.4, color='#1a6fd0', lw=1.6)
        axt.axvline(self._player.cursor, color='#e8a33d', lw=1.4, alpha=0.9)
        axt.set_xlim(0, self._player.horizon)
        axt.set_ylim(-0.6, max(1, len(target.outputs)) - 0.4)
        axt.set_yticks(range(len(target.outputs)))
        axt.set_yticklabels([t.role[:6] for t in target.outputs], fontsize=8)
        axt.set_title('output pulse edges (real time)', fontsize=9)
        # the width strip directly below carries the shared time axis label
        axt.tick_params(labelsize=7)

    def _output_spans(self, cell):
        """[(start, end, still_open)] pulses of one output wire, clipped to the
        playback cursor. Both engines expose ``pulse_intervals`` (PulseSim
        natively, AsyncLutSim via its rise/fall logs); an interval whose fall
        has not happened yet reads as open-ended at the cursor."""
        raw = getattr(self._player.sim, 'pulse_intervals', None)
        if raw is None or cell is None:
            return []
        cursor = self._player.cursor
        spans = []
        for start, end in raw.get(cell, ()):
            if start > cursor:
                continue
            spans.append((float(start), float(min(end, cursor)), end > cursor))
        return spans

    def _draw_width_strip(self, axw):
        """Second output view: the LEVEL waveform under the edge strip, each
        pulse drawn as a bar labelled with its physical width — so
        width-semantics targets (width sum, odd selector, doubler) can be read
        directly instead of inferred from edge spacing."""
        target = self._circuit['target']
        axw.clear()
        # A label is only drawn when it actually fits inside its bar, else the
        # text spills across neighbouring pulses. One character is roughly this
        # much of the time axis at the strip's font size.
        char_span = 0.012 * max(self._player.horizon, 1e-9)
        for k, term in enumerate(target.outputs):
            cell = self._out_pos.get(term.role)
            for start, end, open_ended in self._output_spans(cell):
                axw.broken_barh([(start, max(end - start, 0.05))],
                                (k - 0.35, 0.7),
                                facecolors='#7cb8f2' if open_ended else '#1a6fd0',
                                edgecolors='#0d4a94', linewidth=0.6)
                width = end - start
                label = ('%.2g…' % width) if open_ended else ('%.2g' % width)
                if width >= len(label) * char_span:
                    axw.text((start + end) / 2, k, label, ha='center',
                             va='center', fontsize=7, color='white')
        axw.axvline(self._player.cursor, color='#e8a33d', lw=1.4, alpha=0.9)
        axw.set_xlim(0, self._player.horizon)
        axw.set_ylim(-0.6, max(1, len(target.outputs)) - 0.4)
        axw.set_yticks(range(len(target.outputs)))
        axw.set_yticklabels([t.role[:6] for t in target.outputs], fontsize=8)
        axw.set_title('output pulse widths (level view)', fontsize=9)
        axw.set_xlabel('continuous time (tick units)', fontsize=8)
        axw.tick_params(labelsize=7)

    def _nv_output_summary(self):
        target = self._circuit['target']
        parts = []
        for term in target.outputs:
            cell = self._out_pos.get(term.role)
            n = len(self._player.events_upto(cell)) if cell else 0
            parts.append('%s=%d' % (term.role, n))
        return '  '.join(parts)

    def _in_bits(self):
        return tuple(1 if v.get() else 0 for _, v in self._inputs)

    def _on_toggle(self):
        # level toggles exist only for the SNN backend; the asynchronous
        # substrates are driven by the pulse timeline
        if not self._circuit:
            return
        if self._backend == 'snn':
            self._snn_prepare()         # inputs changed → recompute the LIF run
            self._cursor = 0
            self._draw()

    # ── playback (SNN LIF / nervous + LUT async) ──────────────────────────────────

    def _reset(self):
        self._stop()
        if self._backend == 'snn':
            self._snn_prepare()
            self._cursor = 0
            self._draw()
            return
        self._player.reset()                    # async continuous-time playback
        self._draw_async()

    def _step(self):
        if self._backend == 'snn':
            self._snn_step()
            return
        if self._running and self._player.at_end():
            self._player.reset()                # loop the run while Playing
        else:
            self._player.step()
        self._draw_async()

    def _draw(self):
        if self._backend == 'snn':
            self._draw_snn()
        else:
            self._draw_async()

    def _toggle_run(self):
        if self._running:
            self._stop()
        else:
            self._running = True
            self._run_btn.config(text='Pause')
            self._tick_loop()

    def _tick_loop(self):
        if not self._running:
            return
        self._step()
        interval = 90 if self._backend == 'snn' else 70
        self.parent.after(interval, self._tick_loop)

    def _stop(self):
        self._running = False
        if hasattr(self, '_run_btn'):
            try:
                self._run_btn.config(text='Run')
            except tk.TclError:
                pass

    def close(self):
        """Release playback callbacks and matplotlib event bindings."""
        self._stop()
        if getattr(self, '_editor', None) is not None:
            self._editor.disconnect()
            self._editor = None

    # ── SNN LIF playback (membrane charge → spikes over 20 ms) ─────────────────────

    def _snn_prepare(self):
        """Run the LIF sim for the current input bits and cache the full trace.
        The response is deterministic, so we compute it once and just scrub a
        playback cursor over it (like the nervous/LUT tick playback)."""
        target = self._circuit['target']
        bits = self._in_bits()
        # A logical high is interpreted by lif_sim as one source pulse.  Reusing
        # the target's high value keeps this view consistent with fitness/plots.
        currents = {self._in_ids[i]: (target.high if bits[i] else 0.0)
                    for i in range(len(self._in_ids)) if self._in_ids[i] is not None}
        (self._snn_times, self._snn_V,
         self._snn_spikes, self._snn_fired) = simulate_trace(
            self._neurons, self._synapses, currents)

    def _snn_step(self):
        nxt = self._cursor + SNN_STRIDE
        if nxt >= N_STEPS:
            nxt = 0 if self._running else N_STEPS - 1     # loop while Running
        self._cursor = nxt
        self._draw()

    def _draw_snn(self):
        cur   = self._cursor
        t_ms  = self._snn_times[cur]
        V     = self._snn_V[cur]
        lo    = max(0, cur - SNN_STRIDE + 1)              # flash = fired this frame
        fired = (self._snn_fired[lo:cur + 1].any(axis=0)
                 if self._snn_fired.shape[1] else None)

        self.fig.clf()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0],
                                   height_ratios=[1.0, 0.72], wspace=0.2, hspace=0.36)
        axg = self.fig.add_subplot(gs[:, 0])
        axr = self.fig.add_subplot(gs[0, 1])
        axv = self.fig.add_subplot(gs[1, 1])
        draw_snn_net(axg, self._neurons, self._synapses, v=V, fired=fired,
                     title='t = %4.1f ms   inputs = %s'
                           % (t_ms, ''.join(map(str, self._in_bits()))))
        self._draw_raster(axr, t_ms)
        self._draw_out_traces(axv, cur)
        self.canvas.draw_idle()
        self._status.set('t = %.1f ms    %s' % (t_ms, self._snn_outcome_text()))

    def _draw_raster(self, ax, t_ms):
        """All-neuron spike raster with a moving play cursor; inputs red, outputs blue."""
        ax.clear()
        ns = self._neurons
        for i, n in enumerate(ns):
            times = self._snn_spikes.get(n.id, [])
            if n.is_output:
                ax.axhspan(i - 0.5, i + 0.5, color='#1a6fd0', alpha=0.07)
            if times:
                col = ('#b02020' if n.is_input else
                       '#1a6fd0' if n.is_output else '#8a94a6')
                ax.vlines(times, i - 0.4, i + 0.4, color=col, lw=1.2)
        ax.axvline(t_ms, color='#e8a33d', lw=1.6, alpha=0.9)
        ax.set_xlim(0, self._snn_times[-1] + DT)
        ax.set_ylim(-0.6, max(1, len(ns)) - 0.4)
        ax.set_title('spike raster', fontsize=9)
        ax.set_xlabel('ms', fontsize=8); ax.set_yticks([])
        ax.tick_params(labelsize=7)

    def _draw_out_traces(self, ax, cur):
        """Output-neuron membrane potential up to the cursor, with the fire threshold."""
        ax.clear()
        outs = [n for n in self._neurons if n.is_output]
        t = self._snn_times[:cur + 1]
        vmax = 0.5
        for n in outs:
            ax.plot(t, self._snn_V[:cur + 1, n.id], lw=1.6,
                    label=(n.out_role or 'out')[:6])
            ax.axhline(n.vth, color='#cc4b37', lw=0.8, ls='--', alpha=0.5)
            vmax = max(vmax, float(n.vth) * 1.25)
        ax.set_xlim(0, self._snn_times[-1] + DT)
        ax.set_ylim(-0.04, vmax)
        ax.set_title('output membrane V  (— fire threshold)', fontsize=9)
        ax.set_xlabel('ms', fontsize=8); ax.tick_params(labelsize=7)
        if outs:
            ax.legend(fontsize=7, loc='upper right')
        else:
            ax.text(0.5, 0.5, '(no output neuron)', ha='center', va='center',
                    color='#999', transform=ax.transAxes)

    def _snn_outcome_text(self):
        parts = []
        for n in self._neurons:
            if n.is_output:
                nsp = len(self._snn_spikes.get(n.id, []))
                parts.append('%s→%s' % ((n.out_role or 'out'),
                                        'FIRES(1)' if nsp else 'silent(0)'))
        return '   '.join(parts) if parts else '(no output)'
