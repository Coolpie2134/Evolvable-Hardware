"""
diversity_ui.py — the "Diversity" tab: what variety survives in a SOLVED
population.

The Evolution chart's spread series (population fitness sigma) is identically
zero once every genome scores 1.0, so it goes blind exactly where the question
gets interesting. This tab reads variety off STRUCTURE instead, via the
four-level collapse funnel in nv_evo/diversity.py, and optionally samples the
mutational neighbourhood.

Analysis grows every genome (and, for the behaviour level, runs it on an
off-spec probe bank), so it runs on a worker thread with progress reporting and
a working Stop; the UI thread only drains a queue.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from nv_evo import diversity as dv

POLL_MS = 120
DEFAULT_POPULATION = os.path.join('results', 'solver_generation.json')


class _Cancelled(Exception):
    """Raised inside the worker's progress callback to honour Stop."""


class DiversityTab:
    """Population-diversity view. ``get_population_path`` is an optional
    callback returning the app's most recently written population file."""

    def __init__(self, parent, get_population_path=None, mono='Courier New'):
        self.parent = parent
        self._get_population_path = get_population_path
        self._mono = mono
        self._path = None
        self._queue = queue.Queue()
        self._worker = None
        self._stop = threading.Event()
        self._after_id = None
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.parent, padding=(6, 4))
        top.pack(fill='x')
        self._load_btn = ttk.Button(top, text='Load population…',
                                    command=self.choose_population)
        self._load_btn.pack(side='left')
        self._run_btn = ttk.Button(top, text='Analyse', command=self.analyse)
        self._run_btn.pack(side='left', padx=(6, 0))
        self._stop_btn = ttk.Button(top, text='Stop', command=self.stop,
                                    state='disabled')
        self._stop_btn.pack(side='left', padx=(6, 0))

        self._robust_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Mutational robustness',
                        variable=self._robust_var).pack(side='left', padx=(12, 0))
        ttk.Label(top, text='Samples:').pack(side='left', padx=(6, 2))
        self._samples_var = tk.StringVar(value='8')
        ttk.Entry(top, textvariable=self._samples_var, width=4).pack(side='left')
        ttk.Label(top, text='Max genomes:').pack(side='left', padx=(10, 2))
        self._limit_var = tk.StringVar(value='60')
        ttk.Entry(top, textvariable=self._limit_var, width=5).pack(side='left')

        self._status = tk.StringVar(
            value='Load a solved population (results/solver_generation.json) '
                  'and press Analyse.')
        ttk.Label(self.parent, textvariable=self._status, anchor='w',
                  padding=(8, 2)).pack(fill='x')

        body = ttk.Frame(self.parent)
        body.pack(fill='both', expand=True)

        self._fig = plt.figure(figsize=(7.0, 5.2))
        self._fig.patch.set_facecolor('#f5f5f5')
        self._canvas = FigureCanvasTkAgg(self._fig, master=body)
        self._canvas.get_tk_widget().pack(side='left', fill='both', expand=True,
                                          padx=4, pady=4)
        self._placeholder('No population analysed yet.')

        panel = ttk.LabelFrame(body, text='Report', padding=6)
        panel.pack(side='right', fill='both', padx=(0, 6), pady=6)
        # The report is an ALIGNED table, so it must not word-wrap (wrap='none')
        # — which means it needs a horizontal scrollbar too, or long lines are
        # simply cut off. ScrolledText only supplies the vertical one, hence the
        # explicit grid of Text + both scrollbars.
        grid = ttk.Frame(panel)
        grid.pack(fill='both', expand=True)
        grid.rowconfigure(0, weight=1)
        grid.columnconfigure(0, weight=1)
        # 80 columns: the widest report line (the statistics header) is 77, so
        # the table fits without horizontal scrolling on a default window
        self._text = tk.Text(grid, width=80, height=24,
                             font=(self._mono, 9), state='disabled',
                             wrap='none')
        self._text.grid(row=0, column=0, sticky='nsew')
        yscroll = ttk.Scrollbar(grid, orient='vertical',
                                command=self._text.yview)
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll = ttk.Scrollbar(grid, orient='horizontal',
                                command=self._text.xview)
        xscroll.grid(row=1, column=0, sticky='ew')
        self._text.configure(yscrollcommand=yscroll.set,
                             xscrollcommand=xscroll.set)

    def _placeholder(self, message):
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=11,
                color='#888888')
        ax.axis('off')
        self._canvas.draw_idle()

    def _set_text(self, text):
        self._text.config(state='normal')
        self._text.delete('1.0', 'end')
        self._text.insert('end', text)
        self._text.config(state='disabled')

    # ── population selection ─────────────────────────────────────────────────

    def notify_population(self, path):
        """The app calls this when a run writes a fresh population file."""
        if path and os.path.exists(path):
            self._path = path
            self._status.set('Population ready: %s — press Analyse.'
                             % os.path.basename(path))

    def _resolve_path(self):
        if self._path and os.path.exists(self._path):
            return self._path
        if self._get_population_path is not None:
            candidate = self._get_population_path()
            if candidate and os.path.exists(candidate):
                return candidate
        return DEFAULT_POPULATION if os.path.exists(DEFAULT_POPULATION) else None

    def choose_population(self):
        initial = os.path.dirname(self._resolve_path() or DEFAULT_POPULATION)
        path = filedialog.askopenfilename(
            title='Load solved population',
            initialdir=initial or '.',
            filetypes=[('Population checkpoint', '*.json'), ('All files', '*.*')])
        if path:
            self._path = path
            self._status.set('Loaded %s — press Analyse.'
                             % os.path.basename(path))

    # ── analysis ─────────────────────────────────────────────────────────────

    def analyse(self):
        if self._worker is not None:
            return
        path = self._resolve_path()
        if path is None:
            self._status.set('No population file found. Run the GA to a solve '
                             '(it writes results/solver_generation.json), or '
                             'use Load population…')
            return
        try:
            limit = max(1, int(self._limit_var.get()))
            samples = max(1, int(self._samples_var.get()))
        except ValueError:
            self._status.set('Max genomes and samples must be whole numbers.')
            return

        self._stop.clear()
        self._run_btn.config(state='disabled')
        self._load_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self._status.set('Analysing %s …' % os.path.basename(path))
        self._worker = threading.Thread(
            target=self._work, daemon=True,
            args=(path, limit, samples, bool(self._robust_var.get())))
        self._worker.start()
        self._poll()

    def stop(self):
        self._stop.set()
        self._status.set('Stopping after the current genome…')
        self._stop_btn.config(state='disabled')

    def _work(self, path, limit, samples, want_robustness):
        """Worker thread: load, cluster, optionally sample robustness."""
        try:
            from evo_runtime.checkpoint import load_checkpoint
            state = load_checkpoint(path)
            if 'genomes' not in state:
                self._queue.put(('error', 'That file is a single-genome '
                                          'checkpoint; this view needs a '
                                          'population (solver_generation.json).'))
                return
            genomes = state['genomes'][:limit]
            target = state['target']
            backend = state['backend']
            config = state.get('run_config')
            valid = state.get('valid', 0.999)

            def funnel_progress(level, index, total):
                if self._stop.is_set():
                    raise _Cancelled
                self._queue.put(('progress', '%s: %d/%d' % (level, index + 1,
                                                            total)))

            report = dv.diversity_funnel(genomes, backend, target, config,
                                         on_progress=funnel_progress)
            self._queue.put(('funnel', report, len(genomes), target.name,
                             backend))

            robustness = None
            if want_robustness:
                def robust_progress(index, total, sample, n_samples):
                    if self._stop.is_set():
                        raise _Cancelled
                    self._queue.put(('progress', 'robustness: genome %d/%d, '
                                                 'mutant %d/%d'
                                     % (index + 1, total, sample + 1, n_samples)))

                robustness = dv.robustness(genomes, backend, target, config,
                                           samples=samples, valid=valid,
                                           on_progress=robust_progress)
            self._queue.put(('done', robustness))
        except _Cancelled:
            self._queue.put(('cancelled', None))
        except Exception as exc:                      # surface, never hang
            self._queue.put(('error', '%s: %s' % (type(exc).__name__, exc)))

    def _poll(self):
        report = None
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == 'progress':
                    self._status.set(message[1])
                elif kind == 'funnel':
                    _, report, n, name, backend = message
                    self._report = report
                    self._population = (n, name)
                    self._set_text(dv.format_report(
                        report, population=n, target_name=name))
                    self._draw(report, None)
                    self._status.set('%s — %d genomes, backend %s'
                                     % (name, n, backend))
                elif kind == 'cancelled':
                    self._status.set('Stopped.')
                    self._finish()
                    return
                elif kind == 'error':
                    self._set_text(message[1])
                    self._placeholder('Analysis failed.')
                    self._status.set(message[1])
                    self._finish()
                    return
                elif kind == 'done':
                    robustness = message[1]
                    current = getattr(self, '_report', None)
                    if robustness is not None and current is not None:
                        n, name = getattr(self, '_population', (None, None))
                        self._set_text('%s\n\n%s' % (
                            dv.format_report(current, population=n,
                                             target_name=name),
                            dv.format_robustness(robustness)))
                        self._draw(current, robustness)
                    self._status.set('Analysis complete.')
                    self._finish()
                    return
        except queue.Empty:
            pass
        self._after_id = self.parent.after(POLL_MS, self._poll)

    def _finish(self):
        self._worker = None
        self._after_id = None
        self._run_btn.config(state='normal')
        self._load_btn.config(state='normal')
        self._stop_btn.config(state='disabled')

    # ── plots ────────────────────────────────────────────────────────────────

    def _pie(self, ax, stats):
        """One level as a composition pie: every wedge is ONE cluster, sized by
        the share of the population that collapses into it. Many thin wedges =
        finely divided; one solid disc = monoculture.

        Deliberately no pooling of the small tail: merging 18 singletons into a
        single 72% slice draws exactly the picture the data denies — it reads
        as a dominant cluster when the truth is maximal fragmentation. Thin
        wedges are honest, so every cluster keeps its own.
        """
        label = dv.LEVEL_LABEL.get(stats.level, stats.level)
        sizes = list(stats.sizes)
        if not sizes:
            ax.axis('off')
            ax.set_title('%s\n(nothing measurable)' % label, fontsize=9)
            return
        colours = plt.get_cmap('tab20')([i % 20 for i in range(len(sizes))])
        # hairlines once the wedges get thin, or the borders eat the slices
        edge = 0.6 if len(sizes) <= 20 else (0.3 if len(sizes) <= 60 else 0.0)

        def autopct(percent):
            # only wedges with room for text get a label; the rest stay clean
            return ('%.0f%%' % percent) if percent >= 8.0 else ''

        ax.pie(sizes, colors=colours, startangle=90, counterclock=False,
               autopct=autopct, pctdistance=0.72,
               textprops={'fontsize': 7, 'color': '#222222'},
               wedgeprops={'linewidth': edge, 'edgecolor': 'white'})
        ax.set_aspect('equal')
        # three short lines, not one long one: the columns are narrow and a
        # single-line subtitle overruns into the neighbouring axes
        ax.set_title('%s\n%d groups, largest %.1f%%\neffective %.2f'
                     % (label, stats.distinct,
                        100.0 * stats.largest_share, stats.effective),
                     fontsize=8.5, linespacing=1.25)

    def _draw(self, report, robustness):
        """Four composition pies — one per level — showing what share of the
        population each distinct class accounts for. The robustness histogram
        joins them on the right when it was sampled."""
        self._fig.clf()
        # Constrained layout, not tight_layout: pie titles are two lines wide
        # and would otherwise be clipped by, or overlap, the neighbouring axes
        # when the pane is resized.
        self._fig.set_layout_engine('constrained')
        levels = list(report.levels)
        if robustness is None:
            grid = self._fig.add_gridspec(2, 2)
            pie_axes = [self._fig.add_subplot(grid[row, col])
                        for row in (0, 1) for col in (0, 1)]
            rax = None
        else:
            grid = self._fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.3))
            pie_axes = [self._fig.add_subplot(grid[row, col])
                        for row in (0, 1) for col in (0, 1)]
            rax = self._fig.add_subplot(grid[:, 2])
        for index, ax in enumerate(pie_axes):
            if index < len(levels):
                self._pie(ax, levels[index])
            else:
                ax.axis('off')

        total = levels[0].total if levels else 0
        self._fig.suptitle('Group composition by level   (n = %d genomes)'
                           % total, fontsize=10)

        if rax is not None:
            counts = robustness.histogram(10)
            centres = [(i + 0.5) / 10.0 for i in range(10)]
            rax.bar(centres, counts, width=0.09, color='#54A24B')
            rax.set_xlabel('Local robustness', fontsize=8)
            rax.set_ylabel('Genomes', fontsize=8, labelpad=2)
            rax.tick_params(labelsize=7)
            rax.yaxis.get_major_locator().set_params(integer=True)
            rax.set_title('Mutational robustness\nlocal %.2f   '
                          'novel-valid %.2f'
                          % (robustness.local, robustness.novel_valid),
                          fontsize=8.5)
        self._canvas.draw_idle()
