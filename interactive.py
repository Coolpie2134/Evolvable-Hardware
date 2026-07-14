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
from nv_evo.playback import NervousPlayer, PulseLaneEditor, pulses_from_trial
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

        # drop any prior pulse-timeline click handler before rebuilding
        if getattr(self, '_editor', None) is not None:
            self._editor.disconnect()
            self._editor = None

        # input bar: the nervous net is asynchronous, so it is driven by a
        # clickable pulse TIMELINE; LUT/SNN keep level toggles.
        for w in self._inbar.winfo_children():
            w.destroy()
        self._inputs = []
        if backend in ('nervous', 'lut'):
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
            self._grid = grow_nervous(c['genome'], seeds=tuple(target.inputs),
                                      grid_size=target.grid_size, iters=target.iters)
            self._routing, self._in_pos, self._out_pos = interpret_nervous(self._grid, target)
            if getattr(target, 'temporal', False):
                # show the same output cell the fitness reads (trace-matched)
                self._out_pos, _ = place_outputs_by_trace(
                    self._grid, self._routing, self._in_pos, target)
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
        self.fig.clf()
        gs = self.fig.add_gridspec(2, 2, height_ratios=[0.55, 1.0],
                                   width_ratios=[1.1, 1.0], hspace=0.6, wspace=0.25)
        self._ax_lanes = self.fig.add_subplot(gs[0, :])
        self._axg = self.fig.add_subplot(gs[1, 0])
        self._axt = self.fig.add_subplot(gs[1, 1])
        self._editor = PulseLaneEditor(self._ax_lanes, self.canvas, labels,
                                       horizon=horizon, snap=0.5,
                                       on_change=self._nv_schedule_changed)
        self._editor.set_pulses(pulses_from_trial(target, len(self._in_pos)))
        if self._backend == 'nervous':
            self._player = NervousPlayer(
                self._grid, self._routing, horizon=horizon,
                max_events=getattr(target, 'max_events', 2048),
                config=getattr(target, 'pulse_config', None))
        else:                                  # LUT — same player, level engine
            self._player = LutPlayer(
                self._grid, horizon=horizon,
                config=getattr(target, 'lut_config', None))
        self._player.set_schedule(self._editor.schedule(self._in_pos))

    def _nv_schedule_changed(self):
        self._stop()
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
            draw_hex_net(self._axg, self._grid, target.grid_size,
                         routing=self._routing,
                         in_pos=self._in_pos, out_pos=self._out_pos,
                         activity=self._player.activity(), show_edges=True,
                         title=title)
        else:
            in_pos = [p for p in self._in_pos if p in self._grid]
            draw_lut_net(self._axg, self._grid, activity=self._player.nibbles(),
                         in_pos=in_pos, out_pos=self._out_pos, show_edges=True,
                         title=title)
        self._draw_event_strip(self._axt)
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
        axt.set_xlabel('continuous time (tick units)', fontsize=8)
        axt.tick_params(labelsize=7)

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
