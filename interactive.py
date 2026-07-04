"""
interactive.py — an "Interactive / Test" tab for the Evolvable-Hardware GUI.

After a circuit is evolved (or loaded), this lets you drive the inputs by hand
with toggle buttons and watch the response:

  * SNN      — set inputs, run the LIF sim, read each output's spikes / bit.
  * Nervous  — pulse playback: toggle inputs, Step / Run / Reset through time and
               watch activity propagate, loops latch, and oscillators run (it uses
               nv_evo.temporal's stepping engine).

Kept in its own module so app.py stays lean: app.py builds one tab and hands it a
callback (`app.current_circuit()`) returning the genome / target / backend / arch.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from snn_evo import grow_snn, interpret_grid, simulate
from nv_evo import (grow_nervous, interpret_nervous, PulseSim,
                    place_outputs_by_trace)
from nv_evo.viz import draw_hex_net
from lut_evo import grow_lut, LutSim
from lut_evo import place_outputs_by_trace as lut_place_by_trace
from lut_evo.ga import _place_outputs_combinational as lut_place_combinational
from lut_evo.viz import draw_lut_net


class InteractiveTab:
    def __init__(self, parent, get_circuit):
        self.parent      = parent
        self.get_circuit = get_circuit         # () -> dict | None
        self._inputs     = []                  # list of (label, BooleanVar)
        self._circuit    = None                # current loaded circuit context
        self._state      = {}                  # nervous activity map
        self._trace      = {}                  # role -> [bits]
        self._tick       = 0
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
                  relief='sunken', padding=(6, 2)).pack(fill='x', side='bottom', padx=4, pady=(0, 4))

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

        # input toggles
        for w in self._inbar.winfo_children():
            w.destroy()
        self._inputs = []
        for i in range(len(target.inputs)):
            v = tk.BooleanVar(value=False)
            lbl = chr(65 + i) if i < 26 else 'i%d' % i
            ttk.Checkbutton(self._inbar, text=lbl, variable=v,
                            command=self._on_toggle).pack(side='left', padx=2)
            self._inputs.append((lbl, v))

        # controls
        for w in self._ctrl.winfo_children():
            w.destroy()
        if backend in ('nervous', 'lut'):
            if backend == 'nervous':
                self._grid = grow_nervous(c['genome'], seeds=tuple(target.inputs),
                                          grid_size=target.grid_size, iters=target.iters)
                self._routing, self._in_pos, self._out_pos = interpret_nervous(self._grid, target)
                if getattr(target, 'temporal', False):
                    # show the same output cell the fitness reads (trace-matched)
                    self._out_pos, _ = place_outputs_by_trace(
                        self._grid, self._routing, self._in_pos, target)
                hint = '   (toggle inputs, then Step/Run to watch pulses & memory)'
            else:                                             # LUT array
                self._grid = grow_lut(c['genome'], seeds=tuple(target.inputs),
                                      grid_size=target.grid_size, iters=target.iters)
                self._in_pos = list(target.inputs)
                if getattr(target, 'temporal', False):
                    self._out_pos, _ = lut_place_by_trace(self._grid, self._in_pos, target)
                else:
                    self._out_pos = lut_place_combinational(self._grid, target)
                hint = '   (toggle inputs, then Step/Run — latched synchronous array)'
            ttk.Button(self._ctrl, text='Step',  command=self._step).pack(side='left', padx=3)
            self._run_btn = ttk.Button(self._ctrl, text='Run', command=self._toggle_run)
            self._run_btn.pack(side='left', padx=3)
            ttk.Button(self._ctrl, text='Reset', command=self._reset).pack(side='left', padx=3)
            ttk.Label(self._ctrl, text=hint, foreground='#888').pack(side='left', padx=6)
            self._reset()
        else:
            ttk.Label(self._ctrl, text='(toggle inputs — the LIF circuit re-runs instantly)',
                      foreground='#888').pack(side='left', padx=6)
            self._eval_snn()
        self._status.set('Loaded %s [%s] — drive the inputs.' % (target.name, backend))

    def _in_bits(self):
        return tuple(1 if v.get() else 0 for _, v in self._inputs)

    def _on_toggle(self):
        if not self._circuit:
            return
        if self._backend in ('nervous', 'lut'):
            self._draw()                # show that inputs changed; takes effect next Step
        else:
            self._eval_snn()

    # ── nervous / LUT playback (one tick per Step) ────────────────────────────────

    def _reset(self):
        self._stop()
        if self._backend == 'lut':
            self._sim = LutSim(self._grid)
            self._nibbles = {}
        else:
            self._sim = PulseSim(self._grid, self._routing)
        self._state = {c: 0 for c in self._grid}
        self._trace = {t.role: [] for t in self._circuit['target'].outputs}
        self._tick = 0
        self._draw()

    def _step(self):
        bits = self._in_bits()
        invals = {self._in_pos[i]: bits[i] for i in range(len(self._in_pos))}
        self._state = self._sim.step(invals)
        if self._backend == 'lut':
            self._nibbles = dict(self._sim.out)     # 4-bit N/S/E/W emission map
        self._tick += 1
        for t in self._circuit['target'].outputs:
            p = self._out_pos.get(t.role)
            self._trace[t.role].append(self._state.get(p, 0) if p else 0)
            if len(self._trace[t.role]) > 60:
                self._trace[t.role] = self._trace[t.role][-60:]
        self._draw()

    def _draw(self):
        if self._backend == 'lut':
            self._draw_lut()
        else:
            self._draw_nervous()

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
        self.parent.after(180, self._tick_loop)

    def _stop(self):
        self._running = False
        if hasattr(self, '_run_btn'):
            try:
                self._run_btn.config(text='Run')
            except tk.TclError:
                pass

    def _draw_nervous(self):
        target = self._circuit['target']
        self.fig.clf()
        gs_spec = self.fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.25)
        axg = self.fig.add_subplot(gs_spec[0, 0])
        axt = self.fig.add_subplot(gs_spec[0, 1])
        draw_hex_net(axg, self._grid, target.grid_size, routing=self._routing,
                     in_pos=self._in_pos, out_pos=self._out_pos, activity=self._state,
                     show_edges=True,
                     title='tick %d   inputs=%s' % (self._tick, ''.join(map(str, self._in_bits()))))
        self._draw_trace_strip(axt, 'output pulse trains')
        self.canvas.draw_idle()

    def _draw_lut(self):
        target = self._circuit['target']
        self.fig.clf()
        gs_spec = self.fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.25)
        axg = self.fig.add_subplot(gs_spec[0, 0])
        axt = self.fig.add_subplot(gs_spec[0, 1])
        in_pos = [p for p in self._in_pos if p in self._grid]
        draw_lut_net(axg, self._grid, activity=self._nibbles, in_pos=in_pos,
                     out_pos=self._out_pos, show_edges=True,
                     title='tick %d   inputs=%s' % (self._tick, ''.join(map(str, self._in_bits()))))
        self._draw_trace_strip(axt, 'output bit trains')
        self.canvas.draw_idle()

    def _draw_trace_strip(self, axt, title):
        """Shared right-hand output-trace panel for nervous & LUT playback."""
        target = self._circuit['target']
        axt.set_title(title, fontsize=9)
        for k, t in enumerate(target.outputs):
            tr = self._trace.get(t.role, [])
            if tr:
                axt.step(range(len(tr)), [v + k * 1.4 for v in tr],
                         where='post', lw=1.4, label=t.role)
        axt.set_ylim(-0.3, 1.4 * max(1, len(target.outputs)))
        axt.set_xlabel('tick'); axt.set_yticks([])
        if any(self._trace.get(t.role) for t in target.outputs):
            axt.legend(fontsize=7, loc='upper right')

    # ── SNN instant evaluation ─────────────────────────────────────────────────────

    def _eval_snn(self):
        c = self._circuit
        target, arch = c['target'], c['arch']
        from snn_evo import CURRENT_HIGH
        grid = grow_snn(c['genome'], seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
        ns, ss = interpret_grid(grid, target=target, arch=arch)
        in_ids = []
        for p in target.inputs:
            n = next((nn for nn in ns if (nn.x, nn.y) == p and nn.is_input), None)
            in_ids.append(n.id if n else None)
        bits = self._in_bits()
        currents = {in_ids[i]: (CURRENT_HIGH if bits[i] else 0.0)
                    for i in range(len(in_ids)) if in_ids[i] is not None}
        sp = simulate(ns, ss, currents)

        self.fig.clf()
        ax = self.fig.add_subplot(111); ax.axis('off')
        lines = ['Inputs: ' + '  '.join('%s=%d' % (lbl, v.get()) for lbl, v in self._inputs), '']
        for term in target.outputs:
            o = next((nn for nn in ns if nn.is_output and nn.out_role == term.role), None)
            if o is None:
                lines.append('%s: (output neuron not found)' % term.role); continue
            nsp = len(sp.get(o.id, []))
            lines.append('%-8s pos=%s   spikes=%-3d   -> %s'
                         % (term.role, (o.x, o.y), nsp, 'FIRES (1)' if nsp >= 1 else 'silent (0)'))
        lines += ['', '(raw response to the literal input currents; complement-encoded',
                  ' outputs like carry read inverted — see the Voltage tab note)']
        ax.text(0.03, 0.95, '\n'.join(lines), va='top', ha='left',
                family='monospace', fontsize=11)
        self.canvas.draw_idle()
