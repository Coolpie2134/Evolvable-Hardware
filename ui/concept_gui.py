#!/usr/bin/env python3
"""
concept_gui.py - a GUI playground for the `experiments/concept/` proof-of-concept GAs.

Each sim runs its own faithful `main()` loop in a background thread, bridged to
the GUI (see experiments/concept/engines.py). Instead of the original ASCII output you get
live matplotlib visuals:

    - Fitness histogram   population fitness distribution (replaces the ASCII bars)
    - Best organism       its output grid, beside the target when the sim has one
    - Fitness over time    best + mean fitness per generation

Controls: pick a sim, set population size, then Run / Pause / Step / Reset.

The concept sims now share experiments/concept/common/terminal.py (platform-guarded), so
they import on Windows too - but they're designed for Linux; this GUI runs the
GA logic on any platform that can import them.

Usage:
    python -m ui.concept_gui
"""
import os, sys, queue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from experiments.concept.engines import available_engines, make_engine, CONFIGS, DEFAULT_SIM
from . import ui_compat


class ConceptGUI:
    def __init__(self, root):
        self.root = root
        root.title('Concept GA Playground')
        ui_compat.apply_theme(root)

        self.avail      = available_engines()
        self.engine     = None
        self._step_mode = False
        self._poll_job  = None
        self._best_hist = []
        self._mean_hist = []

        self._build_ui()
        ui_compat.fit_window(root, min_width=860, min_height=600)
        root.protocol('WM_DELETE_WINDOW', self._close)
        self._poll()

    # -- UI --------------------------------------------------------------------

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=(6, 4))
        ctrl.pack(fill='x', side='top')

        ttk.Label(ctrl, text='Simulation:').pack(side='left', padx=(2, 2))
        self._sim_var = tk.StringVar(value=DEFAULT_SIM)
        self._sim_cb  = ttk.Combobox(ctrl, textvariable=self._sim_var,
                                     values=list(CONFIGS), width=11, state='readonly')
        self._sim_cb.pack(side='left')
        self._sim_cb.bind('<<ComboboxSelected>>', self._on_sim_change)

        ttk.Label(ctrl, text='Population:').pack(side='left', padx=(8, 2))
        self._pop_var = tk.StringVar(value='150')
        ttk.Entry(ctrl, textvariable=self._pop_var, width=7).pack(side='left')

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=10)

        self._run_btn   = ttk.Button(ctrl, text='Run',   command=self._run)
        self._pause_btn = ttk.Button(ctrl, text='Pause', command=self._pause, state='disabled')
        self._step_btn  = ttk.Button(ctrl, text='Step',  command=self._step)
        self._reset_btn = ttk.Button(ctrl, text='Reset', command=self._reset)
        for b in (self._run_btn, self._pause_btn, self._step_btn, self._reset_btn):
            b.pack(side='left', padx=3)

        self.fig = plt.figure(figsize=(9.4, 5.2))
        self.fig.patch.set_facecolor('#f5f5f5')
        gs = self.fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                                   left=0.07, right=0.97, top=0.92, bottom=0.10)
        self.ax_hist = self.fig.add_subplot(gs[:, 0:2])
        self.ax_grid = self.fig.add_subplot(gs[0, 2])
        self.ax_line = self.fig.add_subplot(gs[1, 2])
        self._init_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        self._status = tk.StringVar()
        ttk.Label(self.root, textvariable=self._status, anchor='w',
                  relief='sunken', padding=(6, 2), wraplength=840,
                  justify='left').pack(fill='x', side='bottom',
                                                        padx=4, pady=(0, 4))
        self._on_sim_change()

    def _init_axes(self):
        self.ax_hist.set_title('Population fitness distribution', fontsize=10)
        self.ax_hist.set_xlabel('Fitness'); self.ax_hist.set_ylabel('# organisms')
        self.ax_grid.set_title('Best organism', fontsize=9)
        self.ax_grid.set_xticks([]); self.ax_grid.set_yticks([])
        self.ax_line.set_title('Fitness over time', fontsize=9)
        self.ax_line.set_xlabel('Generation'); self.ax_line.grid(True, alpha=0.3)

    # -- sim selection ---------------------------------------------------------

    def _on_sim_change(self, _evt=None):
        self._reset()
        name = self._sim_var.get()
        cfg, status = self.avail.get(name, (None, 'unknown'))
        ok = cfg is not None
        if ok:
            self._pop_var.set(str(cfg.get('popsize_default', 150)))
        for b in (self._run_btn, self._step_btn):
            b.config(state='normal' if ok else 'disabled')
        self._status.set('%s - %s%s' % (name, status,
                         '. Set Population and click Run or Step.' if ok else ''))

    # -- controls --------------------------------------------------------------

    def _build(self, paused):
        name = self._sim_var.get()
        cfg, _ = self.avail.get(name, (None, ''))
        if cfg is None:
            return False
        try:
            pop = int(self._pop_var.get())
            if pop < 2:
                raise ValueError
        except ValueError:
            self._status.set('Invalid population size (need an integer >= 2).')
            return False
        self._best_hist.clear(); self._mean_hist.clear()
        self.engine = make_engine(name)
        self.engine.start(popsize=pop, paused=paused)
        return True

    def _run(self):
        if self.engine is None:
            if not self._build(paused=False):
                return
        else:
            self.engine.resume()
        self._set_running(True)

    def _pause(self):
        if self.engine:
            self.engine.pause()
        self._set_running(False)

    def _step(self):
        if self.engine is None:
            if not self._build(paused=True):
                return
        self._step_mode = True
        self.engine.resume()
        self._set_running(True)

    def _reset(self):
        if self.engine:
            self.engine.stop()
        self.engine = None
        self._step_mode = False
        self._best_hist.clear(); self._mean_hist.clear()
        self._clear_displays()
        self._set_running(False)

    def _close(self):
        """Stop the worker before destroying Tk so no background thread leaks."""
        if self.engine:
            self.engine.stop()
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        ui_compat.cancel_after_callbacks(self.root)
        self.root.destroy()

    def _set_running(self, running):
        self._run_btn.config(state='disabled' if running else 'normal')
        self._pause_btn.config(state='normal' if running else 'disabled')
        self._step_btn.config(state='disabled' if running else 'normal')
        self._sim_cb.config(state='disabled' if running else 'readonly')

    # -- polling + drawing -----------------------------------------------------

    def _poll(self):
        latest = None
        got = 0
        if self.engine is not None and self.engine.q is not None:
            try:
                while True:
                    msg = self.engine.q.get_nowait()
                    if isinstance(msg, dict) and 'error' in msg:
                        self._status.set('Sim error - see console.')
                        print(msg['error'])
                        self._reset()
                        latest = None
                        break
                    latest = msg
                    got += 1
                    self._best_hist.append((msg['generation'], msg['best_fitness']))
                    self._mean_hist.append((msg['generation'], msg['mean_fitness']))
            except queue.Empty:
                pass
        if latest is not None:
            self._draw(latest)
            if self._step_mode:
                self._step_mode = False
                if self.engine:
                    self.engine.pause()
                self._set_running(False)
        if self.root.winfo_exists():
            self._poll_job = self.root.after(50, self._poll)

    def _board_image(self, s):
        best = np.array(s['best_grid'], dtype=float)
        tg = s.get('target_grid')
        if tg is not None:
            target = np.array(tg, dtype=float)
            gap = np.full((best.shape[0], 1), 0.5)
            return np.hstack([best, gap, target]), True
        return best, False

    def _draw(self, s):
        # histogram
        self.ax_hist.clear()
        fits = np.asarray(s['fitnesses'])
        top  = max(1, int(s['max_fitness']))
        counts = np.bincount(np.clip(fits, 0, top), minlength=top + 1)
        self.ax_hist.bar(np.arange(len(counts)), counts, color='#3b6fb0', width=0.9)
        self.ax_hist.set_title('Population fitness distribution - gen %d' % s['generation'],
                               fontsize=10)
        self.ax_hist.set_xlabel('Fitness'); self.ax_hist.set_ylabel('# organisms')
        self.ax_hist.set_xlim(-0.5, top + 0.5)

        # best organism (and target, if any)
        self.ax_grid.clear()
        img, has_target = self._board_image(s)
        self.ax_grid.imshow(img, cmap='Greys', vmin=0, vmax=1,
                            interpolation='nearest', aspect='auto')
        ttl = 'Best (fit=%d, stop=%d)' % (s['best_fitness'], s['best_stop'])
        if has_target:
            ttl += '   |  target'
        self.ax_grid.set_title(ttl, fontsize=8)
        self.ax_grid.set_xticks([]); self.ax_grid.set_yticks([])

        self._redraw_line()
        self.canvas.draw_idle()

        extra = '  '.join('%s=%s' % (k, v) for k, v in s.get('extra', {}).items())
        self._status.set('%s  gen=%d  best=%d  mean=%.2f  %s' %
                         (self._sim_var.get(), s['generation'],
                          s['best_fitness'], s['mean_fitness'], extra))

    def _redraw_line(self):
        self.ax_line.clear()
        self.ax_line.set_title('Fitness over time', fontsize=9)
        self.ax_line.set_xlabel('Generation'); self.ax_line.grid(True, alpha=0.3)
        if self._best_hist:
            bx, by = zip(*self._best_hist)
            mx, my = zip(*self._mean_hist)
            self.ax_line.plot(bx, by, 'b-', lw=1.3, label='best')
            self.ax_line.plot(mx, my, 'r--', lw=1.0, alpha=0.8, label='mean')
            self.ax_line.legend(fontsize=7, loc='upper left')

    def _clear_displays(self):
        for ax in (self.ax_hist, self.ax_grid, self.ax_line):
            ax.clear()
        self._init_axes()
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    ConceptGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
