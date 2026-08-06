#!/usr/bin/env python3
"""
app.py - Evolvable Hardware: single-window GUI.

Self-contained merge of the former main.py (evolve + truth table),
plot_growth.py (circuit-growth snapshots) and plot_ha.py (membrane-voltage
traces), now generalised to any registered target function.

Pick a target from the dropdown (logic gates, half/full adder, multi-bit adder)
or build your own truth table with "Custom...". The GA evolves a grown SNN to
match it, across three tabs:

    Evolution        live fitness chart + truth table for the chosen target
    Circuit Growth   growth snapshots (seed + each iteration)
    Voltage Traces   membrane voltages per output, per input case

The GA runs in a background thread; progress is polled on the Tk main loop.
On completion the best genome (and its target) are saved to
results/best_genome.json; the two plot tabs can be exported to PNG.

Usage:
    python -m ui.app
"""
import sys, os, math, time, random, threading, queue, dataclasses
import multiprocessing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tkinter as tk
from tkinter import ttk, scrolledtext

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np

from substrates.snn import (grow_snn, grow_snn_snapshots, interpret_grid, simulate,
                     simulate_trace, circuit_summary,
                     TARGETS, DEFAULT_TARGET, get_target, truth_table_target,
                     Arch, DEFAULT_ARCH, temporal_arch,
                     prepare_snn_temporal, snn_temporal_report)
from substrates.snn.genome import GRID_SIZE
from substrates.snn.lif_sim import DT, SIM_TIME, N_STEPS
from substrates.nervous import (nervous_truth_table, grow_nervous_snapshots, interpret_nervous,
                    nervous_case_outputs,
                    ROUTING, temporal_report, periodic_combinational_target)
from substrates.nervous import TEMPORAL_TARGETS
from substrates.nervous.contracts import behavior_contract_lines
from substrates.nervous.viz import draw_hex_net
from substrates.lut.viz import draw_lut_net, draw_lut_table
from substrates.fnv import (
    COMPONENTS as FNV_COMPONENTS,
    functional_case_outputs, functional_input_positions, functional_report,
    grow_functional_snapshots, prepare_functional,
)
from substrates.fnv.viz import draw_functional_net
# Absolute package imports work both for the documented ``python -m ui.app``
# launch and when this file is invoked directly by an IDE/file association.
# ROOT is inserted above, so direct execution still resolves the package.
from ui.interactive import InteractiveTab
from ui.designer import DesignerTab, _Tip

#: Shown, disabled, for the substrates with one native I/O mechanism.
NATIVE_IO_LABEL = 'Evolved inputs / fitted outputs'
#: Shown when a legacy fixed-input nervous checkpoint is loaded.
LEGACY_IO_LABEL = 'Legacy fixed inputs / fitted outputs'
LUT_PAD_IO_LABEL = 'Evolved internal source pads'
LUT_EDGE_IO_LABEL = 'Alternating exterior perimeter buses'
from ui.diversity_ui import DiversityTab
from ui.target_ui import TargetPicker
from ui import ui_compat
from runtime.config import (FNVConfig, FNV_FAMILIES, GAConfig, RunConfig,
                                DEFAULT_EVALUATION_WORKERS,
                                LUT_FUNCTION_FAMILIES, NV_NEW_RUN_PROFILES,
                                MAX_EVALUATION_WORKERS,
                                MAX_CHROMOSOME_COUNT as MAX_CHROMS,
                                default_max_telomere)
from runtime.checkpoint import load_checkpoint, save_checkpoint
from runtime.controller import worker_entry as evolution_worker_entry

RESULTS_DIR = os.path.join(ROOT, 'results')
CKPT        = os.path.join(RESULTS_DIR, 'best_genome.json')
SOLVER_POP  = os.path.join(RESULTS_DIR, 'solver_generation.json')
LATEST_POP  = os.path.join(RESULTS_DIR, 'latest_population.json')
LEGACY_CKPT = os.path.join(RESULTS_DIR, 'best_genome.pkl')
os.makedirs(RESULTS_DIR, exist_ok=True)

V_REST          = 0.0
R_M             = 1.0
V_RESET         = 0.0
MAX_VOLT_CASES  = 8     # voltage tab caps rows for readability on big targets


# -- target-aware analysis helpers ---------------------------------------------

def _growth_seeds(target, genome=None):
    """Return the strategy's developmental origin for the SNN analysis path.

    Fixed mode uses declared pads, developmental spatial mode uses the
    genome's input anchors, and the other evolvable modes use a neutral centre.
    """
    from substrates.nervous.io_placement import growth_seeds, io_strategy
    return growth_seeds(target, io_strategy(target), genome)


def grow_for(genome, target):
    return grow_snn(genome, seeds=_growth_seeds(target, genome),
                    grid_size=target.grid_size, iters=target.iters)


def interpret_for(genome, target, arch):
    grid = grow_for(genome, target)
    if getattr(target, 'temporal', False):
        arch = temporal_arch(arch)
    from substrates.nervous.io_placement import io_strategy, bind_io, flat_inputs
    if io_strategy(target) != 'fixed':
        from substrates.snn.growth import cell_io_tags
        bound = bind_io(genome, grid, target,
                        tags=cell_io_tags(genome, grid))
        if bound is None:
            return grid, [], []
        in_pos, out_pos = bound
        neurons, synapses = interpret_grid(
            grid, target=target, arch=arch,
            input_pos=flat_inputs(in_pos), output_pos=out_pos)
    else:
        neurons, synapses = interpret_grid(grid, target=target, arch=arch)
    return grid, neurons, synapses


def _case_currents(target, in_bits, in_ids, complement):
    currents = {}
    for bit, ids in zip(in_bits, in_ids):
        ids = ids if isinstance(ids, (tuple, list)) else [ids]
        base = target.high if bit else 0.0
        level = (target.high - base) if complement else base
        for iid in ids:
            currents[iid] = max(currents.get(iid, 0.0), level)
    return currents


def simulate_vmem(neurons, synapses, input_currents, track_ids):
    """Sample voltage traces from the same event-driven LIF run as fitness."""
    _, v_hist, spikes, _ = simulate_trace(neurons, synapses, input_currents)
    return spikes, {nid: v_hist[:, nid].copy()
                    for nid in track_ids if 0 <= nid < len(neurons)}


def build_truth_table(genome, target, arch):
    """Return the truth-table report for `target` as a string."""
    grid, ns, ss = interpret_for(genome, target, arch)
    from substrates.nervous.io_placement import io_strategy, bind_io, input_groups
    if io_strategy(target) != 'fixed':
        from substrates.snn.growth import cell_io_tags
        bound = bind_io(genome, grid, target,
                        tags=cell_io_tags(genome, grid))
        groups = input_groups(bound[0]) if bound else []
    else:
        groups = input_groups(target.inputs)
    by_pos = {(n.x, n.y): n for n in ns if n.is_input}
    in_ids = [[by_pos[pos].id for pos in cells if pos in by_pos]
              for cells in groups]
    out_ids = []
    for term in target.outputs:
        out_ids.append([n.id for n in ns
                        if n.is_output and n.out_role == term.role])

    lines = (['Target: ' + target.name, ''] +
             behavior_contract_lines(target) + [
             '',
             'Goal: Produce the specified logical outputs for each input combination.',
             'Scoring: Each output is an exact expected-versus-observed check; all checks weigh equally.',
             'Tests: %d defined input combination%s.' % (
                 len(target.cases), '' if len(target.cases) == 1 else 's'),
             '',
             'Circuit: ' + circuit_summary(ns, ss)])
    for term, ids in zip(target.outputs, out_ids):
        if ids:
            o = ns[ids[0]]
            enc = []
            if term.complement_inputs: enc.append('complement-in')
            if term.invert_spike:      enc.append('invert')
            tag = ('  [%s]' % ', '.join(enc)) if enc else ''
            lines.append("  out '%s': %d site%s; first=(%d,%d) "
                         "vth=%.1f excit=%s%s" %
                         (term.role, len(ids), '' if len(ids) == 1 else 's',
                          o.x, o.y, o.vth, o.excit, tag))
        else:
            lines.append("  out '%s': (not found)" % term.role)

    if any(not ids for ids in in_ids) or any(not ids for ids in out_ids):
        lines.append('')
        lines.append('(circuit incomplete: some input/output neurons missing)')
        return '\n'.join(lines)

    in_hdr  = ' '.join('i%d' % i for i in range(len(target.inputs)))
    out_hdr = ' '.join('%s expected/actual' % t.role for t in target.outputs)
    lines.append('')
    lines.append('  %s | %s | result' % (in_hdr, out_hdr))
    lines.append('  ' + '-' * (len(in_hdr) + len(out_hdr) + 14))

    correct = total = 0
    encodings = {t.complement_inputs for t in target.outputs}
    for in_bits, out_bits in target.cases:
        sims = {c: simulate(ns, ss, _case_currents(target, in_bits, in_ids, c))
                for c in encodings}
        row_ok = True
        cells  = []
        for i, term in enumerate(target.outputs):
            sp    = sims[term.complement_inputs]
            fired = any(len(sp.get(output_id, [])) >= 1
                        for output_id in out_ids[i])
            act   = (0 if fired else 1) if term.invert_spike else (1 if fired else 0)
            ok    = act == out_bits[i]
            row_ok = row_ok and ok
            total += 1
            correct += 1 if ok else 0
            cells.append('%d/%d' % (out_bits[i], act))
        in_str  = ' '.join(str(b) for b in in_bits).ljust(len(in_hdr))
        out_str = ' '.join(c.ljust(len('%s expected/actual' % t.role))
                           for c, t in zip(cells, target.outputs))
        lines.append('  %s | %s | %s' % (in_str, out_str, 'PASS' if row_ok else 'FAIL'))

    fit = correct / total if total else 0.0
    lines.append('')
    lines.append('  => %d/%d checks  (fitness = %.4f)%s' %
                 (correct, total, fit, '   ALL PASS' if correct == total else ''))
    return '\n'.join(lines)


def grid_to_rgba(grid, grid_size, seed_set, output_pos):
    """Convert a grid dict to an RGBA image array (grid_size x grid_size x 4)."""
    img = np.ones((grid_size, grid_size, 4), dtype=float)
    for (x, y), state in grid.items():
        row, col = grid_size - 1 - y, x
        if (x, y) in seed_set:
            img[row, col] = [0.9, 0.1, 0.1, 1.0]            # red - seed
        elif (x, y) in output_pos:
            img[row, col] = [0.1, 0.8, 0.2, 1.0]            # green - output
        else:
            excit = not bool((state >> 3) & 0x1)
            alpha = 0.35 + 0.55 * (state & 0x7) / 7.0
            if excit:
                img[row, col] = [0.15, 0.35, 0.85, alpha]   # blue - excitatory
            else:
                img[row, col] = [0.90, 0.50, 0.10, alpha]   # orange - inhibitory
    return img


def _split_display(chromosome):
    count = len(chromosome.genes)
    if count == 0:
        return 'none'
    if count == 1:
        return 'gene fields'
    return str(max(1, min(int(chromosome.split), count - 1)))


def build_genome_text(genome, fitness=None, binding=None):
    """Render the genome as a readable, aligned chromosome/gene table.

    ``binding`` is an optional pre-rendered I/O-binding summary (see
    substrates/nervous/io_placement.describe_binding) appended as a footer, so a genome
    evolved under an evolvable io_placement strategy shows exactly which tag
    each port selected and where it attached."""
    chroms  = list(getattr(genome, 'chromosomes', []) or [])
    n_total = sum(len(c.genes) for c in chroms)
    # I/O tags only matter (and are only shown) when some gene carries one -
    # default genomes stay rendered exactly as before.
    show_iotags = any(
        getattr(c, 'wiring', False)
        or any(getattr(g, 'tag', 0) or getattr(g, 'io_selector', 0)
               for g in c.genes)
        for c in chroms)
    spatial_iotags = (
        binding is not None
        and 'io_placement=spatial_chromosome' in str(binding))
    L = []
    head = 'Genome  -  %d chromosome%s,  %d genes' % (
        len(chroms), '' if len(chroms) == 1 else 's', n_total)
    if fitness is not None:
        head += '   (fitness = %.4f)' % fitness
    L.append(head)
    L.append('')
    g0 = next((gene for chromosome in chroms
               for gene in chromosome.genes), None)
    constructive_fnv = g0 is not None and hasattr(g0, 'component_id')
    hexmode = g0 is not None and hasattr(g0, 'ctx_l')
    lutmode = g0 is not None and hasattr(g0, 'ctx_n')
    sides = ('named FNV output ports and dependency labels'
             if constructive_fnv else
             'rotated circuit context L/R/D (4-bit states 0-15)' if hexmode
             else 'the 4 sides N/E/S/W (each a 16-bit LUT)' if lutmode
             else 'the 4 sides N/E/S/W')
    element = 'core circuit' if hexmode else 'cell'
    if constructive_fnv:
        L.append('Each placement gene installs one permanent FNV component at')
        L.append('the empty honeycomb site faced by all of its named input ports.')
        L.append('IDs and output-port labels are stable: list order is not a cursor.')
        L.append('A collision suppresses only that placement; unrelated branches survive.')
    else:
        L.append('Each gene maps an expected neighbourhood (%s + the %s' %
                 (sides, element))
        L.append('itself) to an output state. During growth a %s adopts the output of the' %
                 element)
        L.append('gene whose pattern is closest (min Hamming distance) to its real neighbours.')
    if constructive_fnv:
        L.append('branch labels group connected blocks for duplication, rerouting, and crossover.')
    elif hexmode:
        L.append('Each rule develops one 4-bit core circuit and is applied independently')
        L.append("to a tile's L/R/D circuits in their rotated orientation. out = 0 switches")
        L.append('that circuit off; the phenotype tile disappears only when all three are off.')
    elif lutmode:
        L.append('Each 16-bit LUT is really a BOOLEAN FUNCTION of the four bits the cell')
        L.append('receives from its neighbours: shown here minimised as out = f(N,S,E,W)')
        L.append("(NOT  = not, - = and, + = or).  'out' is the LUT this gene installs; a")
        L.append("growth gene is one whose own context is empty (self = dead), the only")
        L.append('kind that can bring a dead cell to life.  out = 0 means that direction')
        L.append('stays dead.  Raw hex is kept in brackets for reference.')
    else:
        L.append("'until' = the gene only applies while the growth iteration <= that limit.")
    if constructive_fnv:
        L.append('Chromosomes are hereditary containers; crossover exchanges labelled branch blocks.')
    else:
        L.append('split = the between-gene crossover point; a one-gene chromosome')
        L.append('recombines the rule fields inside that gene instead.')
    if show_iotags:
        if spatial_iotags:
            L.append('On chromosome 3, io-tag/selector encode a normalised')
            L.append('(x,y) anchor. Each port attaches to the nearest free')
            L.append('living cell; coordinate mutations move the anchor locally.')
        else:
            L.append('io-tag = expression priority on a body gene (tag-rank mode),')
            L.append('or the desired CELL TYPE on chromosome 3 (wiring mode).')
            L.append('Each wiring gene selects exactly one physical cell; its')
            L.append('selector chooses among matching instances of the desired type.')
    L.append('')
    if not chroms:
        L.append('(empty genome)')
        return '\n'.join(L)

    iotag_hdr, iotag_sep = (
        (' | x/y  ', '+-------') if spatial_iotags
        else ((' | iotag', '+-------') if show_iotags else ('', '')))
    if constructive_fnv:
        hdr = '    #  | stable id | component | branch | named input ports'
        sep = '  -----+-----------+-----------+--------+------------------'
    elif hexmode:
        hdr = '    #  | ctx L ctx R ctx D | self |  out' + iotag_hdr
        sep = '  -----+----------------+------+------' + iotag_sep
    elif lutmode:
        hdr = sep = None                          # block layout instead of a row
    else:
        hdr = '    #  |   N    E    S    W  | self |  out | until' + iotag_hdr
        sep = '  -----+---------------------+------+------+-------' + iotag_sep
    for ci, c in enumerate(chroms):
        tel = getattr(c, 'telomere', None)
        L.append('Chromosome %s     tag=%-4d  split=%-11s%s   (%d genes)%s'
                 % (chr(ord('a') + ci), c.tag, _split_display(c),
                    ('  telomere=%d' % tel) if tel is not None else '',
                    len(c.genes),
                    '   [WIRING chromosome - I/O port map]'
                    if getattr(c, 'wiring', False) else ''))
        if hdr:
            L.append(hdr)
            L.append(sep)
        effective_split = (max(1, min(int(c.split), len(c.genes) - 1))
                           if len(c.genes) > 1 else 0)
        for gi, g in enumerate(c.genes):
            if effective_split == gi:
                L.append('       - - - - - - split - - - - - -')
            if show_iotags:
                iotag = (' | %5d/%d'
                         % (getattr(g, 'tag', 0),
                            getattr(g, 'io_selector', 0)))
            else:
                iotag = ''
            if constructive_fnv:
                from substrates.fnv.catalogue import BY_ID
                refs = ', '.join('%d:%s' % (ref.node_id, ref.direction)
                                 for ref in g.inputs)
                name = BY_ID[g.component_id].name
                L.append('  %4d | %9d | %4d %-15s | %6d | %s'
                         % (gi, g.gene_id, g.component_id, name,
                            g.branch_id, refs))
            elif hexmode:
                L.append('  %4d | %4d %4d %4d | %4d | %4d%s'
                         % (gi, g.ctx_l, g.ctx_r, g.ctx_d, g.self_in,
                            g.self_out, iotag))
            elif lutmode:
                L.extend(_lut_gene_lines(gi, g, show_iotags))
            else:
                L.append('  %4d | %4d %4d %4d %4d | %4d | %4d | %5d%s'
                         % (gi, g.state_n, g.state_e, g.state_s, g.state_w,
                            g.self_in, g.self_out, g.limit, iotag))
        L.append('')
    if binding:
        L.append('I/O binding (evolved):')
        L.extend('  ' + line for line in str(binding).splitlines())
        L.append('')
    return '\n'.join(L)


def _lut_gene_lines(gi, g, show_iotags=False):
    """Readable two-line view of one LUT gene: the boolean function the gene
    installs (out, decoded from its 16-bit table to logic over the neighbour
    inputs N/S/E/W) plus the raw hex of every table for reference. The tables
    themselves are drawn as truth grids on the Growth tab. See substrates/lut/boolfn.py."""
    from substrates.lut.boolfn import lut_sop, popcount
    kind = 'growth' if g.self_in == 0 else 'maint '
    iotag = (('   iotag=%d selector=%d'
              % (getattr(g, 'tag', 0), getattr(g, 'io_selector', 0)))
             if show_iotags else '')
    return [
        '  %3d %s  out = %s%s' % (gi, kind, lut_sop(g.self_out), iotag),
        '            hex  out=%04X (%d/16 on)   ctx N/E/S/W=%04X %04X %04X %04X   self=%04X'
        % (g.self_out, popcount(g.self_out),
           g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in),
    ]


# -- custom truth-table dialog -------------------------------------------------

class CustomTargetDialog(tk.Toplevel):
    """Build a Target from a hand-entered truth table."""

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self.title('Custom truth table')
        self.on_done = on_done
        self.result  = None
        self._rows   = []   # list of (in_bits, [out StringVars])
        self.transient(parent)
        self.grab_set()
        self.resizable(True, True)
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.bind('<Escape>', lambda _event: self.destroy())

        top = ttk.Frame(self, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text='Name:').grid(row=0, column=0, sticky='e', padx=2)
        self._name = tk.StringVar(value='Custom')
        ttk.Entry(top, textvariable=self._name, width=18).grid(row=0, column=1, padx=2)
        ttk.Label(top, text='Inputs:').grid(row=0, column=2, sticky='e', padx=2)
        self._nin = tk.IntVar(value=2)
        ttk.Spinbox(top, from_=1, to=6, width=4, textvariable=self._nin).grid(row=0, column=3, padx=2)
        ttk.Label(top, text='Outputs:').grid(row=0, column=4, sticky='e', padx=2)
        self._nout = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=1, to=6, width=4, textvariable=self._nout).grid(row=0, column=5, padx=2)
        ttk.Button(top, text='Build table', command=self._build).grid(row=0, column=6, padx=8)

        self._table_frame = ttk.Frame(self, padding=(8, 0))
        self._table_frame.pack(fill='both', expand=True)

        btns = ttk.Frame(self, padding=8)
        btns.pack(fill='x')
        ttk.Button(btns, text='Create', command=self._create).pack(side='right', padx=4)
        ttk.Button(btns, text='Cancel', command=self.destroy).pack(side='right')

        self._build()
        ui_compat.fit_window(self, min_width=560, min_height=420)

    def _build(self):
        for w in self._table_frame.winfo_children():
            w.destroy()
        self._rows = []
        try:
            n_in  = max(1, min(6, int(self._nin.get())))
            n_out = max(1, min(6, int(self._nout.get())))
        except (tk.TclError, ValueError):
            return
        self._nin.set(n_in)
        self._nout.set(n_out)
        self._shape = (n_in, n_out)
        hdr = ttk.Frame(self._table_frame); hdr.pack(anchor='w')
        for i in range(n_in):
            ttk.Label(hdr, text='i%d' % i, width=3).pack(side='left')
        ttk.Label(hdr, text='  ', width=2).pack(side='left')
        for o in range(n_out):
            ttk.Label(hdr, text='O%d' % o, width=4).pack(side='left')
        canvas = tk.Canvas(self._table_frame, height=min(360, 28 * (2 ** n_in)),
                           highlightthickness=0)
        scroll = ttk.Scrollbar(self._table_frame, orient='vertical', command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        for combo in range(2 ** n_in):
            in_bits = tuple((combo >> i) & 1 for i in range(n_in))
            rowf = ttk.Frame(inner); rowf.pack(anchor='w')
            for b in in_bits:
                ttk.Label(rowf, text=str(b), width=3).pack(side='left')
            ttk.Label(rowf, text='  ', width=2).pack(side='left')
            outs = []
            for _ in range(n_out):
                v = tk.StringVar(value='0')
                ttk.Spinbox(rowf, from_=0, to=1, width=4, textvariable=v).pack(side='left')
                outs.append(v)
            self._rows.append((in_bits, outs))

    def _create(self):
        try:
            n_in  = int(self._nin.get())
            n_out = int(self._nout.get())
        except (tk.TclError, ValueError):
            return
        if getattr(self, '_shape', None) != (n_in, n_out):
            messagebox.showinfo('Custom truth table',
                                'Click Build table after changing Inputs or Outputs.',
                                parent=self)
            return
        rows = []
        for in_bits, outs in self._rows:
            try:
                out_bits = tuple(1 if int(v.get()) else 0 for v in outs)
            except (tk.TclError, ValueError):
                out_bits = tuple(0 for _ in outs)
            rows.append((in_bits, out_bits))
        name  = self._name.get().strip() or 'Custom'
        roles = ['O%d' % o for o in range(n_out)]
        gsize = max(GRID_SIZE, n_in, n_out)
        target = truth_table_target(name, n_in, roles, rows, grid_size=gsize)
        self.result = target
        self.destroy()
        self.on_done(target)


# -- main application ----------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        root.title('Evolvable Hardware: Edwards Indirect Encoding')
        self.q            = queue.Queue()
        self._worker      = None
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        self._recombination_event = threading.Event()
        self._recombination_event.set()
        self._poll_job    = None
        self.best_genome  = None
        self.best_fitness = 0.0
        self._gen_history = []
        self._abs_gen     = 0
        self._custom      = {}     # name -> Target for user-built targets
        self._periodic_target_cache = {}
        self.target       = get_target(DEFAULT_TARGET)
        # what the display tabs currently reflect (set on Run / Load):
        self._disp_target  = self.target
        self._disp_arch    = DEFAULT_ARCH
        self._disp_backend = 'snn'
        # consistent ttk theme + resolved monospace font (Windows/Linux parity)
        ui_compat.apply_theme(root)
        self._mono = ui_compat.mono_family(root)
        self._build_ui()
        # Pin an explicit window geometry once everything is laid out. A toplevel
        # with a programmatic geometry keeps that size instead of auto-resizing
        # to its content, so switching notebook tabs no longer resizes the window
        # (an X11-visible bug; Windows tolerated the content-driven sizing).
        ui_compat.fit_window(root, min_width=960, min_height=680)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._poll()

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=(6, 4))
        ctrl.pack(fill='x', side='top')

        ttk.Label(ctrl, text='Model:').pack(side='left', padx=(2, 2))
        self._backend_var = tk.StringVar(value='SNN')
        self._telomere_values = {
            backend: str(default_max_telomere(backend))
            for backend in ('snn', 'nervous', 'lut', 'fnv')
        }
        self._telomere_backend = 'snn'
        self._backend_cb  = ttk.Combobox(ctrl, textvariable=self._backend_var,
                                         values=['SNN', 'Nervous', 'FNV', 'LUT'],
                                         width=9, state='readonly')
        self._backend_cb.pack(side='left')
        self._backend_cb.bind('<<ComboboxSelected>>', self._on_backend_change)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)

        ttk.Label(ctrl, text='Target:').pack(side='left', padx=(2, 2))
        self._target_var = tk.StringVar(value=DEFAULT_TARGET)
        self._target_picker = TargetPicker(
            ctrl, self._targets_for_backend(self._backend()), variable=self._target_var,
            command=self._on_target_change, target_width=31)
        self._target_picker.pack(side='left')
        self._target_cb = self._target_picker.target_cb
        ttk.Button(ctrl, text='Custom...', command=self._open_custom).pack(side='left', padx=3)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)

        self._run_btn  = ttk.Button(ctrl, text='Run',        command=self._start_ga)
        self._pause_btn = ttk.Button(ctrl, text='Pause', command=self._toggle_pause,
                                     state='disabled')
        self._stop_btn = ttk.Button(ctrl, text='Stop',       command=self._stop_ga, state='disabled')
        self._load_btn = ttk.Button(ctrl, text='Load Saved', command=self._load_saved)
        self._save_btn = ttk.Button(ctrl, text='Save PNGs',  command=self._save_pngs, state='disabled')
        for b in (self._run_btn, self._pause_btn, self._stop_btn,
                  self._load_btn, self._save_btn):
            b.pack(side='left', padx=3)
        self._recombination_var = tk.BooleanVar(value=True)
        self._recombination_chk = ttk.Checkbutton(
            ctrl, text='Recombine', variable=self._recombination_var,
            command=self._sync_recombination)
        self._recombination_chk.pack(side='left', padx=(6, 0))

        # Run settings get their own row so controls remain reachable on laptop
        # screens instead of forcing a >1200 px top bar.
        run_ctrl = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        run_ctrl.pack(fill='x', side='top')
        ttk.Label(run_ctrl, text='Run settings:').pack(side='left', padx=(2, 4))

        def lentry(parent, label, default, width=6):
            ttk.Label(parent, text=label).pack(side='left', padx=(6, 2))
            v = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=v, width=width).pack(side='left')
            return v

        self._pop_var   = lentry(run_ctrl, 'Population:', 50)
        # long single run instead of many short restarts, so slow steady progress
        # is visible (user-preferred; was Gens=30, Tries=20 which restarted every
        # 30 generations before any trend showed).
        self._gens_var  = lentry(run_ctrl, 'Generations:', 500)
        self._tries_var = lentry(run_ctrl, 'Restarts:', 1)
        self._seed_var  = lentry(run_ctrl, 'Seed:', 'random', width=7)
        self._workers_var = lentry(
            run_ctrl, 'Workers:', DEFAULT_EVALUATION_WORKERS, width=3)

        self._progress = ttk.Progressbar(run_ctrl, length=130, mode='determinate')
        self._progress.pack(side='right', padx=(8, 4), fill='x', expand=True)

        self._graded_var = tk.BooleanVar(value=False)
        self._graded_chk = ttk.Checkbutton(run_ctrl, text='Graded logic fitness',
                                           variable=self._graded_var)
        self._graded_chk.pack(side='left', padx=(10, 0))

        # -- second row: model-specific parameters --
        ctrl2 = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ctrl2.pack(fill='x', side='top')

        def aentry(parent, label, default, width=5, store=None,
                   label_store=None):
            label_widget = ttk.Label(parent, text=label)
            label_widget.pack(side='left', padx=(6, 2))
            if label_store is not None:
                label_store.append(label_widget)
            v = tk.StringVar(value=str(default))
            e = ttk.Entry(parent, textvariable=v, width=width)
            e.pack(side='left')
            if store is not None:
                store.append(e)                # so the widget can be enabled/disabled
            return v

        # substrate group (SNN only - hidden for nervous nets)
        self._arch_frame = ttk.Frame(ctrl2)
        self._arch_frame.pack(side='left')
        ttk.Label(self._arch_frame, text='Substrate:').pack(side='left', padx=(2, 0))
        self._syn_var  = aentry(self._arch_frame, 'Syn weight:', DEFAULT_ARCH.syn_weight)
        self._vmin_var = aentry(self._arch_frame, 'Vth min:',    DEFAULT_ARCH.vth_levels[0])
        self._vmax_var = aentry(self._arch_frame, 'Vth max:',    DEFAULT_ARCH.vth_levels[-1])
        self._cur_var  = aentry(self._arch_frame, 'Input I:',    self.target.high)
        self._arch_sep = ttk.Separator(ctrl2, orient='vertical')
        self._arch_sep.pack(side='left', fill='y', padx=8)

        # genome group. Grid / Iters were removed: growth is now self-limiting -
        # the nervous telomere is a Hayflick bound, so neither a grid clip nor an
        # iteration cap governs it (grid_size survives only as each target's I/O
        # layout scale, taken from the target itself). Only the genome's
        # chromosome count remains tunable here.
        self._layout_frame = ttk.Frame(ctrl2)
        self._layout_frame.pack(side='left')
        ttk.Label(self._layout_frame, text='Genome:').pack(side='left', padx=(2, 0))
        self._chroms_var = aentry(self._layout_frame, 'Chroms:', 2, width=4)
        # Telomere ceiling: caps how far a chromosome's telomere - the organism's
        # growth RADIUS - may drift via mutation (and bounds fresh-genome init).
        # Eval cost tracks cell count ~ radius^2, so this is the lever for run
        # speed vs how big nets may grow. LUT organisms default to 8 because
        # their dense ontogeny otherwise fills hundreds of cells; nervous nets
        # retain their existing default of 20.
        self._maxtel_var = aentry(
            self._layout_frame, 'Max telomere:',
            default_max_telomere('snn'), width=4)
        self._layout_reset_btn = ttk.Button(self._layout_frame, text='Reset', width=6,
                                             command=self._reset_arch)
        self._layout_reset_btn.pack(side='left', padx=8)

        self._model_note = ttk.Label(ctrl2, text='', foreground='#888888',
                                     wraplength=500, justify='left')
        self._model_note.pack(side='left', padx=6, fill='x', expand=True)
        self._model_note.bind('<Configure>', lambda event: self._model_note.configure(
            wraplength=max(140, event.width - 8)), add='+')

        # -- third row: GA + substrate-physics tuning (applied on Run) --
        from substrates.nervous.ga import (MEAN_MUTATIONS as _MM, IMMIGRANT_FRAC as _IM,
                               TOURNAMENT_K as _TK, MUT_DECAY as _AL)
        from runtime.mutation import DEFAULT_MUTATION_LIMIT as _ML
        from substrates.nervous.pulse import DELAY as _D, WIDTH as _W, COINC as _C
        self._tune_defaults = dict(mut=_MM, imm=_IM, tk=_TK, alpha=_AL,
                                   beta=1.0, limit=_ML, elite=5,
                                   delay=_D, width=_W, coinc=_C)
        ctrl3 = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ctrl3.pack(fill='x', side='top')

        ga_frame = ttk.Frame(ctrl3); ga_frame.pack(side='left')
        ttk.Label(ga_frame, text='GA:').pack(side='left', padx=(2, 0))
        # Nervous/FNV/LUT read these; SNN has its own fixed constants and ignores
        # them, so the whole row is disabled when SNN is selected (see
        # _reconfigure_for_backend). Widgets are collected so they can be toggled.
        self._ga_entries = []
        self._mut_var  = aentry(ga_frame, 'Mutations/child:', _MM, width=4, store=self._ga_entries)
        self._imm_var  = aentry(ga_frame, 'Immigrants:',      _IM, width=4, store=self._ga_entries)
        self._tourn_var = aentry(ga_frame, 'Tournament:',     _TK, width=3, store=self._ga_entries)
        # Size of the elite breeding pool. Reproduction does not copy these
        # verbatim; after a terminal solve, evaluated parents may survive the
        # parent+offspring consolidation step. 0 = use the whole population.
        self._elite_var = aentry(ga_frame, 'Elites:',         5,   width=3, store=self._ga_entries)
        # simulated-annealing decay: mutation rate *= alpha each generation (1 = off)
        self._alpha_var = aentry(ga_frame, 'Anneal alpha:',       _AL, width=6, store=self._ga_entries)
        # beta controls plateau reheating: 0 disables it, 1 keeps the tuned
        # behavior, and larger values raise mutation faster during stagnation.
        self._beta_var = aentry(ga_frame, 'Plateau beta:', 1.0,
                                width=4, store=self._ga_entries)
        self._mutation_limit_var = aentry(
            ga_frame, 'Mutation cap:', _ML, width=4, store=self._ga_entries)
        # epsilon-lexicase selection (Nervous/FNV/LUT contract cases): streams over the
        # whole population instead of tournament, bypassing the elite pool so it
        # can actually act. Off = tournament (the tuned default).
        self._lexicase_var = tk.BooleanVar(value=False)
        self._lexicase_chk = ttk.Checkbutton(ga_frame, text='epsilon-lexicase',
                                              variable=self._lexicase_var)
        self._lexicase_chk.pack(side='left', padx=(6, 0))
        self._ga_entries.append(self._lexicase_chk)
        ttk.Separator(ctrl3, orient='vertical').pack(side='left', fill='y', padx=8)

        # pulse-physics group (nervous net only)
        self._pulse_frame = ttk.Frame(ctrl3)
        self._pulse_frame.pack(side='left')
        ttk.Label(self._pulse_frame, text='Pulse:').pack(side='left', padx=(2, 0))
        self._pulse_entries = []
        self._pulse_labels = []
        self._delay_var = aentry(
            self._pulse_frame, 'Delay:', _D, width=4,
            store=self._pulse_entries, label_store=self._pulse_labels)
        self._width_var = aentry(
            self._pulse_frame, 'Width:', _W, width=4,
            store=self._pulse_entries, label_store=self._pulse_labels)
        self._coinc_var = aentry(
            self._pulse_frame, 'Coinc:', _C, width=4,
            store=self._pulse_entries, label_store=self._pulse_labels)
        # New NV runs expose only the current coherent substrate profiles.
        # ONE profile: the paper's three-circuit tile on the paper's Fig. 1
        # analog node. The single-tile and digital tri-circuit engines are
        # retired from new runs - measured worse on score, on solve rate, and
        # decisively on held-out certification (see runtime/config.py). They
        # remain in the codebase only as reference implementations for the
        # timing-model audits.
        ttk.Label(self._pulse_frame, text='NV profile:').pack(
            side='left', padx=(6, 2))
        self._NV_PROFILE_LABELS = {
            'Analog tri-circuit (3-output, paper Fig. 1 node)':
                NV_NEW_RUN_PROFILES['analog_tri'],
        }
        self._nv_profile_var = tk.StringVar(
            value='Analog tri-circuit (3-output, paper Fig. 1 node)')
        self._nv_profile_cb = ttk.Combobox(
            self._pulse_frame, textvariable=self._nv_profile_var, width=39,
            state='readonly', values=list(self._NV_PROFILE_LABELS))
        self._nv_profile_cb.pack(side='left')
        self._nv_profile_cb.bind('<<ComboboxSelected>>',
                                 self._on_nv_profile_change)
        # I/O description/selection. Nervous and FNV show their one native
        # evolved-pad/fitted-probe architecture, LUT chooses between internal
        # pads and exterior-edge drivers, and SNN exposes the compatible legacy
        # placement strategies. The frame remains visible for every backend.
        self._io_frame = ttk.Frame(ctrl3)
        self._io_frame.pack(side='left')
        ttk.Label(self._io_frame, text='I/O binding:').pack(
            side='left', padx=(6, 2))
        # Native-pad substrates use this widget as a description, not a choice.
        self._IO_PLACEMENT_LABELS = {
            'Fixed (original)':            'fixed',
            'Dedicated input/output nodes': 'terminal_nodes',
            'Evolvable: node-type rank':   'tag_rank',
            'Evolvable: wiring chromosome':   'wiring_chromosome',
            'Evolvable: spatial chromosome':  'spatial_chromosome',
        }
        self._LUT_IO_LABELS = {
            LUT_PAD_IO_LABEL: 'source_pads',
            LUT_EDGE_IO_LABEL: 'exterior_edges',
        }
        self._io_placement_var = tk.StringVar(value='Fixed (original)')
        self._io_placement_cb = ttk.Combobox(
            self._io_frame, textvariable=self._io_placement_var, width=31,
            state='readonly', values=list(self._IO_PLACEMENT_LABELS))
        self._io_placement_cb.pack(side='left')
        self._io_placement_cb.bind('<<ComboboxSelected>>',
                                   self._on_io_placement_change)
        self._pulse_sep = ttk.Separator(ctrl3, orient='vertical')
        self._pulse_sep.pack(side='left', fill='y', padx=8)
        self._tune_reset_btn = ttk.Button(ctrl3, text='Reset tuning', width=12,
                                           command=self._reset_tuning)
        self._tune_reset_btn.pack(side='left', padx=4)

        # Analog node constants (paper_analog profile only). Their own row so
        # the pulse row cannot overflow; packed/hidden before the notebook by
        # _sync_nv_profile_controls. Defaults are the frozen PulseConfig values,
        # so an untouched row reproduces the audited physics exactly.
        from substrates.nervous.pulse import PulseConfig as _PulseConfig
        _AC = _PulseConfig()
        self._analog_defaults = dict(vth=_AC.analog_threshold,
                                     step=_AC.analog_step,
                                     tau=_AC.analog_tau_leak,
                                     hyst=_AC.analog_hysteresis)
        self._analog_row = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ttk.Label(self._analog_row, text='Analog node:').pack(
            side='left', padx=(2, 0))
        self._analog_entries = []
        self._athr_var = aentry(self._analog_row, 'Vth:',
                                _AC.analog_threshold, width=5,
                                store=self._analog_entries)
        self._astep_var = aentry(self._analog_row, 'Step:',
                                 _AC.analog_step, width=5,
                                 store=self._analog_entries)
        self._atau_var = aentry(self._analog_row, 'Tau leak:',
                                _AC.analog_tau_leak, width=5,
                                store=self._analog_entries)
        self._ahyst_var = aentry(self._analog_row, 'Hysteresis:',
                                 _AC.analog_hysteresis, width=5,
                                 store=self._analog_entries)
        ttk.Label(self._analog_row,
                  text='(two edges must fire, one must not: '
                       '(1-Vth)/2 < Step < 1-Vth; Vth+Hysteresis < 1)').pack(
            side='left', padx=(8, 0))

        # FNV exposes component BANKS only.  Selecting a family enables every
        # permanent route/output-count/timing type in that family.
        self._fnv_row = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ttk.Label(self._fnv_row, text='FNV component families:').pack(
            side='left', padx=(2, 6))
        self._fnv_family_vars = {}
        self._fnv_family_checks = []
        family_labels = {
            'LOGIC': 'Logic',
            'DELAY': 'Delay',
            'NORMALIZER': 'Normalizer',
            'HOLD': 'Hold',
            'C_ELEMENT': 'C-element',
            'TOGGLE': 'Toggle',
            'GATED_OSCILLATOR': 'Gated oscillator',
        }
        for family in FNV_FAMILIES:
            variable = tk.BooleanVar(value=True)
            check = ttk.Checkbutton(
                self._fnv_row, text=family_labels[family], variable=variable)
            check.pack(side='left', padx=(0, 7))
            self._fnv_family_vars[family] = variable
            self._fnv_family_checks.append(check)
        ttk.Separator(self._fnv_row, orient='vertical').pack(
            side='left', fill='y', padx=(2, 8))
        self._fnv_dictionary_btn = ttk.Button(
            self._fnv_row, text='Node number dictionary',
            command=self._show_fnv_node_dictionary)
        self._fnv_dictionary_btn.pack(side='left')

        # LUT gate banks restrict executable truth tables while leaving the
        # developmental CAM context patterns fully expressive.
        self._lut_function_row = ttk.Frame(
            self.root, padding=(6, 0, 6, 4))
        ttk.Label(
            self._lut_function_row,
            text='LUT function banks:').pack(side='left', padx=(2, 6))
        self._lut_function_family_vars = {}
        self._lut_function_family_checks = []
        lut_family_labels = {
            'ROUTING': 'Routing',
            'AND': 'AND',
            'OR': 'OR',
            'XOR': 'XOR',
            'VETO': 'Veto',
            'THRESHOLD': 'Threshold',
            'MUX': 'Mux',
            'UNRESTRICTED': 'Arbitrary LUT',
        }
        for family in LUT_FUNCTION_FAMILIES:
            variable = tk.BooleanVar(value=(family == 'UNRESTRICTED'))
            check = ttk.Checkbutton(
                self._lut_function_row,
                text=lut_family_labels[family], variable=variable)
            check.pack(side='left', padx=(0, 8))
            self._lut_function_family_vars[family] = variable
            self._lut_function_family_checks.append(check)
        ttk.Label(
            self._lut_function_row,
            text='(OFF is always available)', foreground='#777777').pack(
                side='left', padx=(4, 0))

        # -- fourth row: local-minimum escape mechanisms (runtime/escape.py) --
        # EscapeConfig, new sessions, and Reset remain all-off. Every mechanism
        # applies to all four backends and both GA drive paths.
        from runtime.escape import EscapeConfig as _EC
        _ED = _EC()
        self._escape_defaults = dict(
            lifespan=False, stages=_ED.lifespan_checkpoints,
            crowding=False, window=_ED.crowding_window,
            reserve=_ED.crowding_fraction,
            drift=False, adaptive=False,
            rebirth=False, patience=_ED.rebirth_patience,
            fraction=_ED.rebirth_fraction,
            lineage_walk=False, lineage_fraction=_ED.lineage_walk_fraction,
            robust=False, jitter=_ED.robustness_jitter,
            islands=False, island_count=_ED.island_count,
            island_interval=_ED.island_migration_interval,
            downsample=_ED.lexicase_downsample)
        # Three sub-rows: all of these on one line overflows the window at its
        # natural width, and a control the user cannot see is a control that
        # silently does not exist.
        self._escape_row = ttk.Frame(self.root, padding=(6, 0, 6, 0))
        self._escape_row.pack(fill='x', side='top')
        esc1 = ttk.Frame(self._escape_row); esc1.pack(fill='x', side='top')
        esc2 = ttk.Frame(self._escape_row); esc2.pack(fill='x', side='top',
                                                      pady=(2, 0))
        esc3 = ttk.Frame(self._escape_row); esc3.pack(fill='x', side='top',
                                                      pady=(2, 4))
        ctrl4 = esc1
        ttk.Label(esc1, text='Escape:').pack(side='left', padx=(2, 0))
        ttk.Label(esc2, text='          ').pack(side='left', padx=(2, 0))
        self._escape_entries = []
        # Lifespan scoring and the robustness objective are measured through the
        # temporal scoring contract, and epsilon-lexicase needs per-case vectors -
        # none of which the SNN backend has. Those widgets are collected here so
        # _reconfigure_for_backend can disable exactly them rather than making
        # the whole row look inapplicable.
        self._escape_async_only = []

        def echeck(text, tip, store=None):
            var = tk.BooleanVar(value=False)
            widget = ttk.Checkbutton(ctrl4, text=text, variable=var)
            widget.pack(side='left', padx=(6, 0))
            self._escape_entries.append(widget)
            if store is not None:
                store.append(widget)
            _Tip(widget, tip)
            return var

        # Score the organism at several points along its DEVELOPMENT, not only
        # as a finished adult. Manufactures a gradient on flat plateaus; the
        # reported fitness is still the adult score.
        self._lifespan_var = echeck(
            'Lifespan',
            'Scores the circuit at several points during growth, not only '
            'once fully grown.\nThose partial scores become extra test cases '
            'and a tie-break, so a genome whose\nhalf-grown circuit already '
            'works a bit has somewhere to climb from.\nThe reported fitness '
            'is still the fully grown score.\nCosts about one extra '
            'evaluation per stage.',
            store=self._escape_async_only)
        self._lifespan_stages_var = aentry(
            ctrl4, 'stages:', _ED.lifespan_checkpoints, width=3,
            store=self._escape_entries)
        self._escape_async_only.append(self._escape_entries[-1])
        # Restricted tournament replacement.
        self._crowding_var = echeck(
            'Crowding',
            'Restricted tournament replacement. Each offspring competes '
            'against the most\ngenetically similar member of a random window '
            'rather than a random member.\nThat keeps specialists alive, '
            'since one can only be displaced by a better\nversion of itself.'
            '\n\nNothing in the crowded reserve is ever replaced by something '
            'worse, so that\nshare of the population can only improve. '
            '"reserve" sets how much of the\npopulation is crowded; the rest '
            'keeps ordinary generational replacement, which\nis where the '
            'exploratory turnover happens.\n\nAt 1.00 the mean rises smoothly '
            'and never dips. That preserves niches, but a\npopulation that '
            'cannot move downhill also cannot cross a valley.')
        self._crowding_window_var = aentry(
            ctrl4, 'window:', _ED.crowding_window, width=3,
            store=self._escape_entries)
        self._crowding_fraction_var = aentry(
            ctrl4, 'reserve:', _ED.crowding_fraction, width=4,
            store=self._escape_entries)
        self._drift_var = echeck(
            'Neutral drift',
            'Accepts equal-ranked challengers instead of requiring strict '
            'improvement.\nDrifting sideways across a plateau is how this '
            'substrate reaches new circuits.\nRequiring strict improvement '
            'freezes the archived best in place.')
        self._adaptive_mut_var = echeck(
            'Self-adaptive mutation',
            'Each individual carries its own mutation rate, inherited with a '
            'log-normal nudge.\nA stuck lineage heats up on its own while an '
            'improving lineage stays cool.\nThis acts per lineage, unlike the '
            'population-wide plateau reheat (beta).')
        # Second sub-row from here on (echeck/aentry both resolve ctrl4 at call
        # time, so rebinding it moves the remaining widgets).
        ctrl4 = esc2
        self._rebirth_var = echeck(
            'Rebirth',
            'On a stall, rebuilds part of the population from a spread of '
            'archived ancestors\nat a raised mutation rate. The idea is to '
            'back up to an earlier branch point\nand leave it in a different '
            'direction. Current elites are kept.')
        self._rebirth_patience_var = aentry(
            ctrl4, 'stall:', _ED.rebirth_patience, width=4,
            store=self._escape_entries)
        self._rebirth_fraction_var = aentry(
            ctrl4, 'frac:', _ED.rebirth_fraction, width=4,
            store=self._escape_entries)
        self._robust_var = echeck(
            'Robustness',
            'A second objective. Re-scores the circuit under jittered '
            'physics, takes the\nworst case, and ranks it strictly below '
            'correctness so it can never be traded\nagainst a correct answer. '
            'Among equally correct circuits it prefers the one\nthat tolerates '
            'more variation. Costs one extra evaluation per jitter sample.',
            store=self._escape_async_only)
        self._robust_jitter_var = aentry(
            ctrl4, 'jitter:', _ED.robustness_jitter, width=4,
            store=self._escape_entries)
        self._escape_async_only.append(self._escape_entries[-1])
        self._islands_var = echeck(
            'Islands',
            'Splits the population into groups that breed separately, each at '
            'its own\nmutation rate. Cold groups exploit while hot groups '
            'explore, at the same time,\ninstead of one population annealing '
            'between the two. Rare ring migration lets a\ndiscovery spread '
            'gradually rather than sweeping every group at once.\n\n'
            'The groups share one objective and differ only in how they '
            'search. Splitting\nthem by objective instead, one group per test '
            'case, was tried in this project\nand failed: they specialise into '
            'incompatible optima, and every migrant is a\nhybrid that fails on '
            'both sides.')
        self._island_count_var = aentry(
            ctrl4, 'demes:', _ED.island_count, width=3,
            store=self._escape_entries)
        self._island_interval_var = aentry(
            ctrl4, 'migrate:', _ED.island_migration_interval, width=4,
            store=self._escape_entries)
        self._downsample_var = aentry(
            ctrl4, 'Lexicase sample:', _ED.lexicase_downsample, width=4,
            store=self._escape_entries)
        self._escape_async_only.append(self._escape_entries[-1])
        _Tip(self._escape_entries[-1],
             'Fraction of test cases epsilon-lexicase uses each generation '
             '(1 = all of them).\nSampling fewer buys several times more '
             'generations at the same selection\nquality. The sample is '
             'redrawn every generation, so this also serves as a\nrotating '
             'stimulus set. Only has an effect when epsilon-lexicase is selected.')
        # A genuine valley needs a lineage that is allowed to be worse for more
        # than one generation. Keep this on its own row: unlike neutral drift
        # and high-rate rebirth, its guarantee is easy to state precisely.
        ctrl4 = esc3
        ttk.Label(esc3, text='          ').pack(side='left', padx=(2, 0))
        self._lineage_walk_var = echeck(
            'Lineage walk',
            'Reserves a small group for mutation-only random walks that '
            'ignore fitness.\nEach walker descends from its own previous '
            'state, so an intermediate that is\ntemporarily worse survives '
            'long enough to mutate again. Any walker that\nimproves is copied '
            'back into the ordinary breeding pool.\n\nThis knows nothing '
            'about the task and adds no extra evaluations. It spends the\n'
            'selected share of existing population slots on crossing valleys '
            'that take\nmore than one generation to cross.')
        self._lineage_fraction_var = aentry(
            ctrl4, 'share:', _ED.lineage_walk_fraction, width=4,
            store=self._escape_entries)
        self._escape_reset_btn = ttk.Button(
            ctrl4, text='Reset escape', width=13,
            command=self._reset_escape)
        self._escape_reset_btn.pack(side='left', padx=(10, 2))
        # Live telemetry: rebirth count, crowding replacements, mean adaptive
        # rate. Populated from the worker's 'escape' messages.
        self._escape_status = tk.StringVar(value='off')
        self._escape_status_label = ttk.Label(
            esc3, textvariable=self._escape_status, foreground='#777777',
            anchor='w')
        self._escape_status_label.pack(side='left', padx=(8, 0),
                                       fill='x', expand=True)

        # Fix a natural size on the notebook and stop it from resizing to fit
        # whichever tab is shown. On X11 a matplotlib canvas in a freshly-mapped
        # tab requests its figsize*dpi in pixels, and without this the whole
        # window would jump size every time you switched tabs. width/height give
        # the notebook a fixed request; the tab frames fill it via expand.
        self._nb = nb = ttk.Notebook(self.root, width=1020, height=540)
        nb.pack(fill='both', expand=True, padx=4, pady=(0, 2))
        self._build_evolve_tab(nb)
        self._build_growth_tab(nb)
        self._build_voltage_tab(nb)
        self._build_genome_tab(nb)
        self._interactive_frame = self._add_tab(nb, 'Interactive')
        self._interactive = InteractiveTab(
            self._interactive_frame, self.current_circuit)
        self._designer_frame = self._add_tab(nb, 'Designer')
        self._designer = DesignerTab(self._designer_frame,
                                     get_circuit=self.current_circuit)
        # Population diversity: only meaningful for the asynchronous backends
        # (it grows and probes circuits), so it is hidden for SNN alongside the
        # Designer. Analysis is opt-in - it is far too slow to run on every
        # redraw.
        self._diversity_frame = self._add_tab(nb, 'Diversity')
        # Updated after each run: unsuccessful/stopped runs point at their full
        # latest evaluated generation; successful post-solve diversification
        # points at the solver-only population instead.
        self._diversity_population_path = (
            LATEST_POP if os.path.exists(LATEST_POP) else SOLVER_POP)
        self._diversity = DiversityTab(
            self._diversity_frame,
            get_population_path=lambda: self._diversity_population_path,
            mono=self._mono)

        self._status = tk.StringVar(
            value='Ready: pick a model and target, set parameters, click Run (or Load Saved).')
        self._status_label = ttk.Label(
            self.root, textvariable=self._status, anchor='w',
            relief='sunken', padding=(6, 2), wraplength=1000, justify='left')
        self._status_label.pack(
            fill='x', side='bottom', padx=4, pady=(0, 4))

        # Text-bearing header/status widgets follow the actual window width.
        # Fixed wrap lengths left the model note in a narrow block at fullscreen.
        self.root.bind('<Configure>', self._resize_text_regions, add='+')

        self._reconfigure_for_backend()      # set initial model-specific layout
        self._on_target_change()             # render the initial contract immediately

    def _resize_text_regions(self, event):
        """Use available fullscreen width without forcing a larger base window."""
        if event.widget is not self.root:
            return
        self._status_label.configure(wraplength=max(600, event.width - 24))

    def _reconfigure_for_backend(self):
        """Show only the controls / tab labels relevant to the selected model."""
        backend = self._backend()
        if hasattr(self, '_io_placement_cb'):
            # LUT exposes its two physical source architectures. Nervous and
            # FNV retain one native source-pad architecture; SNN keeps the
            # legacy binding strategy choices.
            native_io = backend in ('fnv', 'nervous')
            if backend == 'lut':
                labels = list(self._LUT_IO_LABELS)
                if self._io_placement_var.get() not in labels:
                    self._io_placement_var.set(LUT_PAD_IO_LABEL)
            elif native_io:
                labels = [NATIVE_IO_LABEL]
                self._io_placement_var.set(NATIVE_IO_LABEL)
            else:
                labels = [
                    label for label, strategy
                    in self._IO_PLACEMENT_LABELS.items()
                    if backend != 'snn' or strategy != 'terminal_nodes']
            if (not native_io and backend != 'lut'
                    and backend == 'snn'
                    and self._selected_io_placement() == 'terminal_nodes'):
                self._io_placement_var.set('Fixed (original)')
            elif (not native_io and backend != 'lut'
                  and self._io_placement_var.get() not in labels):
                self._io_placement_var.set('Fixed (original)')
            self._io_placement_cb.configure(values=labels)
            self._io_placement_cb.configure(
                state=('disabled' if native_io else 'readonly'))
        self._sync_telomere_backend(backend)
        if backend != 'snn':
            self._arch_frame.pack_forget()
            self._arch_sep.pack_forget()
            self._graded_chk.state(['disabled'])
            if backend == 'nervous':
                tri = self._selected_tile_arch() == 'tri3'
                model = self._selected_node_model()[0]
                tile_note = ('three independent L/R/D Figure-3 circuits'
                             if tri else 'one legacy broadcast circuit')
                if model == 'paper_analog':
                    physics_note = (
                        'Analog charge/leak/comparator physics; output width '
                        'and coincidence are emergent. Width only sets the '
                        'default external stimulus length; Coinc is ignored.')
                elif model == 'uniform':
                    physics_note = (
                        "The paper's fixed digital abstraction: every node "
                        'regenerates a Width-long pulse after Delay; Coinc '
                        'sets the coincidence window.')
                else:
                    physics_note = (
                        'Both waveform edges are preserved and propagation '
                        'delay evolves by routing state.')
                note = ('Nervous net: HEX array; each tile has %s. %s Loops '
                        'circulate injected pulses as memory. Substrate '
                        '(Vth/Syn/Input) and Graded do not apply.'
                        % (tile_note, physics_note))
            else:
                note = ('LUT array: SQUARE array (each cell wired to 4 neighbours N/S/E/W), '
                        '4 directional 16-bit lookup tables per cell, asynchronous level logic '
                        '(paper Architecture 2 / sim6). Recurrent & dynamical: TEMPORAL '
                        'targets only (it cannot settle to combinational logic). '
                        'Substrate/Graded do not apply.')
            if backend == 'fnv':
                note = (
                    'Functional NV Net (FNV) - directed honeycomb hardware. '
                    'Each node is one fixed logic, timing, memory, or gated-'
                    'oscillator component; two independent antiparallel wires '
                    'share every edge. Component families are enabled only as '
                    'whole physical banks, with no evolvable node parameters.')
            self._model_note.config(text=note)
            self._set_tab_label(self._volt_tab, 'Activity')
        else:
            self._arch_frame.pack(side='left', before=self._layout_frame)
            self._arch_sep.pack(side='left', fill='y', padx=8, before=self._layout_frame)
            self._graded_chk.state(['!disabled'])
            self._model_note.config(text='SNN: leaky integrate-and-fire neurons.')
            self._set_tab_label(self._volt_tab, 'Voltage Traces')
        # pulse-physics knobs apply only to the nervous net's pulse engine.
        # Re-pack relative to the always-present I/O-binding frame (the
        # separator is itself hidden for other models, so it can't be a pack
        # anchor). Nervous, LUT, and FNV show their native evolved-pad/fitted-
        # probe policy as a disabled description; SNN keeps the strategy choice.
        if hasattr(self, '_pulse_frame'):
            if backend == 'nervous':
                self._pulse_frame.pack(side='left', before=self._io_frame)
                self._pulse_sep.pack(side='left', fill='y', padx=8,
                                     before=self._tune_reset_btn)
                self._sync_nv_profile_controls()   # analog row follows profile
            else:
                self._pulse_frame.pack_forget()
                if hasattr(self, '_analog_row'):
                    self._analog_row.pack_forget()
        if hasattr(self, '_fnv_row'):
            if backend == 'fnv':
                self._fnv_row.pack(
                    fill='x', side='top', before=self._escape_row)
            else:
                self._fnv_row.pack_forget()
        if hasattr(self, '_lut_function_row'):
            if backend == 'lut':
                self._lut_function_row.pack(
                    fill='x', side='top', before=self._escape_row)
            else:
                self._lut_function_row.pack_forget()
        # GA tuning (mutations / immigrants / tournament / elites / anneal / lexicase)
        # feeds the nervous, FNV and LUT GAs; SNN uses its own fixed constants
        # and ignores them - so disable the whole row for SNN rather than let it look
        # as if it applies.
        if hasattr(self, '_ga_entries'):
            st = 'disabled' if backend == 'snn' else 'normal'
            for w in self._ga_entries:
                try:
                    w.configure(state=st)
                except tk.TclError:
                    pass
        # The population-level escape mechanisms (crowding / neutral drift /
        # self-adaptive mutation / rebirth) work on all four backends. Lifespan
        # scoring, the robustness objective and lexicase downsampling all read
        # the temporal scoring contract's per-case vectors, which SNN has none
        # of - disable exactly those and leave the rest live.
        if hasattr(self, '_escape_async_only'):
            st = 'disabled' if backend in ('snn', 'fnv') else 'normal'
            if backend == 'fnv':
                self._lifespan_var.set(False)
                self._robust_var.set(False)
                self._downsample_var.set('1.0')
            for w in self._escape_async_only:
                try:
                    w.configure(state=st)
                except tk.TclError:
                    pass
        # the Designer edits grown hardware, which only the two asynchronous
        # substrates have - hide its tab for SNN runs, and keep its
        # architecture in lockstep with the Model selector otherwise
        if hasattr(self, '_designer'):
            try:
                if backend in ('snn', 'fnv'):
                    self._nb.hide(self._designer_frame)
                else:
                    self._nb.add(self._designer_frame)   # restore if hidden
                    self._designer.follow_backend(backend)
            except tk.TclError:
                pass
        # Diversity grows and probes circuits, so it too is nervous/LUT only.
        if hasattr(self, '_diversity_frame'):
            try:
                if backend in ('snn', 'fnv'):
                    self._nb.hide(self._diversity_frame)
                else:
                    self._nb.add(self._diversity_frame)
            except tk.TclError:
                pass
        self._show_interactive_tab()
        # dropdown offers only backend-valid targets (LUT hides combinational)
        if hasattr(self, '_target_cb'):
            self._refresh_target_list()

    def _set_tab_label(self, frame, text):
        try:
            self._nb.tab(frame, text=text)
        except Exception:
            pass

    def _show_fnv_node_dictionary(self):
        """Open the permanent FNV component-ID catalogue."""
        existing = getattr(self, '_fnv_dictionary_window', None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus_set()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        self._fnv_dictionary_window = window
        window.title('FNV node number dictionary')
        window.geometry('970x620')
        window.minsize(720, 360)
        window.transient(self.root)

        ttk.Label(
            window,
            text=('The number printed inside an FNV node is its permanent '
                  'physical component type. IDs are append-only.'),
            padding=(10, 9, 10, 5), anchor='w').pack(fill='x')

        frame = ttk.Frame(window, padding=(10, 0, 10, 10))
        frame.pack(fill='both', expand=True)
        columns = ('id', 'name', 'family', 'function', 'inputs', 'outputs',
                   'timing')
        table = ttk.Treeview(
            frame, columns=columns, show='headings', selectmode='browse')
        headings = {
            'id': 'ID',
            'name': 'Permanent component name',
            'family': 'Family',
            'function': 'Function',
            'inputs': 'Inputs',
            'outputs': 'Outputs',
            'timing': 'Fixed timing',
        }
        widths = {
            'id': 45, 'name': 260, 'family': 125, 'function': 115,
            'inputs': 65, 'outputs': 70, 'timing': 145,
        }
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(
                column, width=widths[column],
                anchor=('center' if column in {
                    'id', 'inputs', 'outputs'} else 'w'),
                stretch=(column == 'name'))
        for entry in FNV_COMPONENTS:
            if entry.behavior == 'GATED_OSCILLATOR':
                timing = 'high %d / low %d ticks' % (
                    entry.high_time, entry.low_time)
            elif entry.duration:
                timing = '%d tick%s' % (
                    entry.duration, '' if entry.duration == 1 else 's')
            elif entry.id:
                timing = 'delay %d tick%s' % (
                    entry.delay, '' if entry.delay == 1 else 's')
            else:
                timing = '-'
            table.insert('', 'end', values=(
                entry.id, entry.name, entry.family, entry.behavior,
                ','.join(entry.inputs) or '-', ','.join(entry.outputs) or '-',
                timing))

        scroll = ttk.Scrollbar(
            frame, orient='vertical', command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        table.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        def close_dictionary():
            self._fnv_dictionary_window = None
            window.destroy()

        window.protocol('WM_DELETE_WINDOW', close_dictionary)

    def _show_interactive_tab(self):
        """Keep playback reachable for every backend, including FNV.

        ``Notebook.add`` restores a tab previously hidden during an older
        backend selection without constructing a second InteractiveTab.
        """
        if not hasattr(self, '_interactive_frame'):
            return
        try:
            self._nb.add(self._interactive_frame)
        except tk.TclError:
            pass

    def _build_evolve_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text='Evolution')
        left = ttk.Frame(frame)
        left.pack(side='left', fill='both', expand=True)
        self._fit_fig, self._fit_ax = plt.subplots(figsize=(5, 3.8))
        self._fit_fig.patch.set_facecolor('#f5f5f5')
        self._fit_ax.set_xlabel('Generation (across all tries)')
        self._fit_ax.set_ylabel('Fitness')
        self._fit_ax.set_ylim(0, 1.05)
        self._fit_ax.set_xlim(0, 10)
        self._fit_ax.set_title('Fitness vs Generation', fontsize=10)
        self._fit_ax.grid(True, alpha=0.3)
        self._best_line, = self._fit_ax.plot([], [], 'b-',  lw=1.5, label='Best (all-time)')
        self._genbest_line, = self._fit_ax.plot([], [], color='#1ea64a', lw=1.0,
                                                label='Best new offspring', alpha=0.85)
        self._mean_line, = self._fit_ax.plot([], [], 'r--', lw=1.0, label='Mean', alpha=0.7)
        self._std_line, = self._fit_ax.plot(
            [], [], color='#e67e22', linestyle=':', lw=1.2,
            label='Population fitness sigma', alpha=0.9)
        self._mut_ax = self._fit_ax.twinx()
        self._mut_ax.set_ylabel('Effective mutation rate', color='#7d3c98')
        self._mut_ax.tick_params(axis='y', labelcolor='#7d3c98', labelsize=8)
        self._mut_ax.set_ylim(0, 8.4)
        self._mutation_line, = self._mut_ax.plot(
            [], [], color='#7d3c98', lw=1.0, alpha=0.8,
            label='Effective mutation')
        self._mutation_text = self._fit_ax.text(
            0.99, 0.02, 'Mutation: -', transform=self._fit_ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#7d3c98',
            bbox=dict(facecolor='white', edgecolor='#7d3c98', alpha=0.75,
                      boxstyle='round,pad=0.2'))
        chart_lines = [self._best_line, self._genbest_line, self._mean_line,
                       self._std_line, self._mutation_line]
        self._fit_ax.legend(chart_lines, [line.get_label() for line in chart_lines],
                            fontsize=8, loc='best')
        self._fit_fig.tight_layout()
        self._fit_canvas = FigureCanvasTkAgg(self._fit_fig, master=left)
        self._fit_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        # Before a run this shows the target's executable contract; afterwards
        # it includes actual observations and the score from the same evaluator
        # evolution used.
        self._tt_frame = ttk.LabelFrame(frame, text='Behavior Contract', padding=6)
        self._tt_frame.pack(side='right', fill='both', padx=(0, 6), pady=6)
        # wrap='word' so long prose lines flow onto the next line instead of
        # being clipped by the panel width; short aligned rows are unaffected.
        self._tt_text = scrolledtext.ScrolledText(
            self._tt_frame, width=62, height=22,
            font=(self._mono, 9), state='disabled', wrap='word')
        self._tt_text.pack(fill='both', expand=True)

    def _build_growth_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text='Circuit Growth')
        self._growth_fig = plt.figure(figsize=(9.5, 9.5))
        self._growth_fig.patch.set_facecolor('#f5f5f5')
        self._growth_canvas = FigureCanvasTkAgg(self._growth_fig, master=frame)
        self._growth_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self._draw_placeholder(self._growth_fig, self._growth_canvas,
                               'Run the GA or Load Saved to see circuit growth.')

    def _build_voltage_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text='Voltage Traces')
        self._volt_tab = frame
        self._volt_fig = plt.figure(figsize=(11, 7.5))
        self._volt_fig.patch.set_facecolor('#f5f5f5')
        self._volt_canvas = FigureCanvasTkAgg(self._volt_fig, master=frame)
        self._volt_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self._draw_placeholder(self._volt_fig, self._volt_canvas,
                               'Run the GA or Load Saved to see membrane voltages.')

    def _add_tab(self, nb, label):
        frame = ttk.Frame(nb)
        nb.add(frame, text=label)
        return frame

    def current_circuit(self):
        """For the Interactive tab: the most-recent best genome + its context."""
        if self.best_genome is None:
            return None
        return {'genome': self.best_genome, 'target': self._disp_target,
                'backend': self._disp_backend, 'arch': self._disp_arch}

    def _build_genome_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text='Genome')
        self._genome_fig = plt.figure(figsize=(11, 7.5))
        self._genome_fig.patch.set_facecolor('#f5f5f5')
        self._genome_canvas = FigureCanvasTkAgg(self._genome_fig, master=frame)
        self._genome_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self._draw_placeholder(self._genome_fig, self._genome_canvas,
                               'Run the GA or Load Saved to see the evolved genome.')

    @staticmethod
    def _draw_placeholder(fig, canvas, msg):
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11, color='#888888')
        ax.axis('off')
        canvas.draw_idle()

    # -- target selection ------------------------------------------------------

    def _all_targets(self):
        d = dict(TARGETS)
        d.update(TEMPORAL_TARGETS)
        d.update(self._custom)
        return d

    def _periodic_target(self, target):
        """Return one stable periodic wrapper per combinational target object."""
        key = id(target)
        cached = self._periodic_target_cache.get(key)
        if cached is None:
            cached = periodic_combinational_target(target)
            self._periodic_target_cache[key] = cached
        return cached

    def _targets_for_backend(self, backend):
        """Return static SNN targets or periodic asynchronous targets."""
        if backend in ('nervous', 'lut'):
            d = {
                name: (target if getattr(target, 'temporal', False)
                       else self._periodic_target(target))
                for name, target in self._all_targets().items()
            }
        else:
            d = self._all_targets()
        # model-restricted targets (supported_models) are physically
        # unattainable under other node-timing models - a waveform contract
        # needs width-preserving transport, so hide it rather than let a
        # 'uniform' run silently cap below 1.0. The node model only exists on
        # the nervous backend; elsewhere the restriction does not apply.
        node_model = (self._selected_node_model()[0]
                      if backend == 'nervous' else None)
        return {
            name: target for name, target in d.items()
            if (not getattr(target, 'supported_backends', ())
                or backend in target.supported_backends
                or (backend == 'fnv'
                    and any(candidate in target.supported_backends
                            for candidate in ('nervous', 'lut'))))
            if (node_model is None
                or not getattr(target, 'supported_models', ())
                or node_model in target.supported_models)
        }

    def _refresh_target_list(self):
        """Repopulate the dropdown for the current backend; if the current
        selection is no longer valid there, switch to the first available."""
        targets = self._targets_for_backend(self._backend())
        names = list(targets)
        self._target_picker.set_targets(targets)
        if names and self._target_var.get() not in names:
            self._target_picker.select(names[0], notify=True)

    def _on_target_change(self, _evt=None):
        name = self._target_var.get()
        self.target = self._targets_for_backend(self._backend()).get(
            name, get_target(DEFAULT_TARGET))
        self._cur_var.set(str(self.target.high))
        if getattr(self.target, 'temporal', False):
            self._reconfigure_for_backend()
            # show what this target IS right away in the Evolution tab's panel
            try:
                if self._backend() == 'fnv':
                    preview = functional_report(self.target)
                elif self._backend() == 'lut':
                    from substrates.lut import lut_report
                    preview = lut_report(self.target)
                elif self._backend() == 'snn':
                    preview = snn_temporal_report(self.target)
                else:
                    preview = temporal_report(self.target)
                self._set_tt(preview, title='Behavior Contract')
            except Exception:
                pass
            model = {
                'lut': 'continuous-time LUT array',
                'nervous': 'continuous-time nervous net',
                'fnv': 'continuous-time Functional NV Net',
                'snn': 'continuous-time recurrent SNN',
            }[self._backend()]
            self._status.set('Target: %s, %s; %d input%s, %d output%s, %d test seconds. '
                             'See Evolution for its executable contract and Interactive for playback.'
                             % (self.target.name, model, self.target.n_inputs,
                                '' if self.target.n_inputs == 1 else 's',
                                self.target.n_outputs,
                                '' if self.target.n_outputs == 1 else 's', self.target.T))
            return
        n_cases = len(self.target.cases)
        preview = ['Target: ' + self.target.name, '']
        preview += behavior_contract_lines(self.target)
        preview += ['', 'Tests: %d defined input combination%s.' %
                    (n_cases, '' if n_cases == 1 else 's'),
                    '', '(run the GA or Load Saved to inspect a circuit)']
        self._set_tt('\n'.join(preview), title='Behavior Contract')
        self._status.set('Target: %s, %d inputs, %d outputs, %d truth-table tests%s' % (
            self.target.name, self.target.n_inputs, self.target.n_outputs, n_cases,
            '   (large: evolution will be slow)' if n_cases > 32 else ''))

    def _effective_target(self, high, graded):
        """Apply the GUI's high/graded knobs to the selected target. Growth is no
        longer grid/iters-bounded (the nervous telomere is self-limiting), so the
        target keeps its own I/O layout: grid_size/iters are left untouched."""
        if getattr(self.target, 'temporal', False):
            # temporal targets have fixed close I/O and no high/graded fields
            return dataclasses.replace(self.target)
        return dataclasses.replace(self.target, high=high, graded=graded)

    def _reset_tuning(self):
        d = self._tune_defaults
        self._mut_var.set(str(d['mut']));   self._imm_var.set(str(d['imm']))
        self._tourn_var.set(str(d['tk']));  self._alpha_var.set(str(d['alpha']))
        self._beta_var.set(str(d['beta']))
        self._mutation_limit_var.set(str(d['limit']))
        self._elite_var.set(str(d['elite']))
        self._delay_var.set(str(d['delay']))
        self._width_var.set(str(d['width'])); self._coinc_var.set(str(d['coinc']))
        if hasattr(self, '_analog_defaults'):
            a = self._analog_defaults
            self._athr_var.set(str(a['vth']))
            self._astep_var.set(str(a['step']))
            self._atau_var.set(str(a['tau']))
            self._ahyst_var.set(str(a['hyst']))
        if hasattr(self, '_nv_profile_var'):
            self._nv_profile_var.set(
                'Analog tri-circuit (3-output, paper Fig. 1 node)')
            self._sync_nv_profile_controls()
        if hasattr(self, '_lexicase_var'):
            self._lexicase_var.set(False)    # tournament is the tuned default
        for variable in getattr(self, '_fnv_family_vars', {}).values():
            variable.set(True)
        if hasattr(self, '_lifespan_var'):
            self._reset_escape()
        if hasattr(self, '_target_picker'):
            self._reconfigure_for_backend()
            self._refresh_target_list()

    def _reset_escape(self):
        """Return every escape mechanism to off: the pre-escape behaviour."""
        d = self._escape_defaults
        self._lifespan_var.set(d['lifespan'])
        self._lifespan_stages_var.set(str(d['stages']))
        self._crowding_var.set(d['crowding'])
        self._crowding_window_var.set(str(d['window']))
        self._crowding_fraction_var.set(str(d['reserve']))
        self._drift_var.set(d['drift'])
        self._adaptive_mut_var.set(d['adaptive'])
        self._rebirth_var.set(d['rebirth'])
        self._rebirth_patience_var.set(str(d['patience']))
        self._rebirth_fraction_var.set(str(d['fraction']))
        self._lineage_walk_var.set(d['lineage_walk'])
        self._lineage_fraction_var.set(str(d['lineage_fraction']))
        self._robust_var.set(d['robust'])
        self._robust_jitter_var.set(str(d['jitter']))
        self._islands_var.set(d['islands'])
        self._island_count_var.set(str(d['island_count']))
        self._island_interval_var.set(str(d['island_interval']))
        self._downsample_var.set(str(d['downsample']))
        self._escape_status.set('off')

    def _read_escape_config(self):
        """Parse the escape row, or None when a field is invalid.

        EscapeConfig validates its own ranges, so a nonsensical entry surfaces
        as "invalid tuning" on Run rather than as a crashed worker thread.
        """
        from runtime.escape import EscapeConfig
        try:
            return EscapeConfig(
                lifespan_scoring=bool(self._lifespan_var.get()),
                lifespan_checkpoints=int(self._lifespan_stages_var.get()),
                crowding=bool(self._crowding_var.get()),
                crowding_window=int(self._crowding_window_var.get()),
                crowding_fraction=float(self._crowding_fraction_var.get()),
                neutral_drift=bool(self._drift_var.get()),
                self_adaptive_mutation=bool(self._adaptive_mut_var.get()),
                rebirth=bool(self._rebirth_var.get()),
                rebirth_patience=int(self._rebirth_patience_var.get()),
                rebirth_fraction=float(self._rebirth_fraction_var.get()),
                lineage_walk=bool(self._lineage_walk_var.get()),
                lineage_walk_fraction=float(
                    self._lineage_fraction_var.get()),
                robustness=bool(self._robust_var.get()),
                robustness_jitter=float(self._robust_jitter_var.get()),
                islands=bool(self._islands_var.get()),
                island_count=int(self._island_count_var.get()),
                island_migration_interval=int(
                    self._island_interval_var.get()),
                lexicase_downsample=float(self._downsample_var.get()))
        except (TypeError, ValueError):
            return None

    def _selected_fnv_families(self):
        if not hasattr(self, '_fnv_family_vars'):
            return FNV_FAMILIES
        return tuple(
            family for family in FNV_FAMILIES
            if self._fnv_family_vars[family].get())

    def _selected_lut_function_families(self):
        if not hasattr(self, '_lut_function_family_vars'):
            return ('UNRESTRICTED',)
        return tuple(
            family for family in LUT_FUNCTION_FAMILIES
            if self._lut_function_family_vars[family].get())

    def _read_run_config(self, chromosome_count=None):
        """Parse controls into an immutable, process-safe run configuration."""
        try:
            mut = float(self._mut_var.get()); imm = float(self._imm_var.get())
            tournament = int(self._tourn_var.get())
            alpha = float(self._alpha_var.get()); beta = float(self._beta_var.get())
            mutation_limit = float(self._mutation_limit_var.get())
            elite = int(self._elite_var.get())
            max_telomere = int(self._maxtel_var.get())
            evaluation_workers = int(self._workers_var.get())
            delay = float(self._delay_var.get()); width = float(self._width_var.get())
            coincidence = float(self._coinc_var.get())
            node_model, evolve_delay = self._selected_node_model()
            tile_arch = self._selected_tile_arch()
            # Analog constants are read only under the analog profile; other
            # profiles keep the frozen PulseConfig defaults untouched.
            analog_kwargs = {}
            if node_model == 'paper_analog':
                analog_kwargs = dict(
                    analog_threshold=float(self._athr_var.get()),
                    analog_step=float(self._astep_var.get()),
                    analog_tau_leak=float(self._atau_var.get()),
                    analog_hysteresis=float(self._ahyst_var.get()))
            if (mut < 0 or not (0 <= imm < 1) or tournament < 1
                    or not (0 < alpha <= 1) or not (0 <= beta <= 10)
                    or mutation_limit < 1
                    or elite < 0 or max_telomere < 2
                    or not 1 <= evaluation_workers <= MAX_EVALUATION_WORKERS
                    or not all(math.isfinite(value) for value in
                               (delay, width, coincidence))
                    or delay <= 0 or width <= 0 or coincidence < 0):
                raise ValueError
        except ValueError:
            return None
        escape = self._read_escape_config()
        if escape is None:
            return None
        from substrates.nervous.pulse import PulseConfig
        try:
            pulse_config = PulseConfig(delay=delay, width=width,
                                       coincidence=coincidence,
                                       model=node_model, **analog_kwargs)
        except ValueError:
            # PulseConfig enforces the analog coupling constraints (e.g. step
            # inside ((1-Vth)/2, 1-Vth)); report as invalid tuning, not a crash.
            return None
        try:
            fnv_config = FNVConfig(self._selected_fnv_families())
        except ValueError:
            return None
        lut_function_families = self._selected_lut_function_families()
        if not lut_function_families:
            return None
        try:
            return RunConfig(
                ga=GAConfig(
                    mean_mutations=mut, immigrant_fraction=imm,
                    mutation_limit=mutation_limit,
                    tournament_size=tournament, elite_count=elite,
                    mutation_decay=alpha,
                    stagnation_beta=beta,
                    selection=(
                        'lexicase' if bool(self._lexicase_var.get())
                        else 'tournament'),
                    recombination_enabled=bool(
                        self._recombination_var.get()),
                    max_telomere=max_telomere,
                    # The selected profile owns the architecture/physics
                    # pairing. node_model must still mirror pulse.model for
                    # worker processes.
                    node_model=node_model,
                    evolve_delay=evolve_delay,
                    tile_arch=tile_arch,
                    io_placement=self._selected_io_placement(),
                    lut_io_mode=self._selected_lut_io_mode(),
                    lut_function_families=lut_function_families,
                    chromosome_count=chromosome_count,
                    evaluation_workers=evaluation_workers,
                    escape=escape),
                pulse=pulse_config,
                fnv=fnv_config)
        except ValueError:
            return None

    def _backend(self):
        v = self._backend_var.get().lower()
        if v.startswith('fnv') or v.startswith('functional'):
            return 'fnv'
        if v.startswith('nerv'):
            return 'nervous'
        if v.startswith('lut'):
            return 'lut'
        return 'snn'

    def _selected_node_model(self):
        """(pulse model, evolve_delay) for the NV profile."""
        _, model, evolve_delay = self._selected_nv_profile()
        return model, evolve_delay

    def _selected_tile_arch(self):
        return self._selected_nv_profile()[0]

    def _selected_io_placement(self):
        """The I/O binding strategy ('fixed' when the control is absent).

        The descriptive labels shown for the substrates with one native I/O
        mechanism are not strategies, so they resolve to 'fixed': those runs
        carry an evolved input layout on the genome instead.
        """
        if not hasattr(self, '_io_placement_var'):
            return 'fixed'
        return self._IO_PLACEMENT_LABELS.get(
            self._io_placement_var.get(), 'fixed')

    def _selected_lut_io_mode(self):
        """The LUT physical source architecture; other backends ignore it."""
        if not hasattr(self, '_io_placement_var'):
            return 'source_pads'
        labels = getattr(self, '_LUT_IO_LABELS', {
            LUT_PAD_IO_LABEL: 'source_pads',
            LUT_EDGE_IO_LABEL: 'exterior_edges',
        })
        return labels.get(
            self._io_placement_var.get(), 'source_pads')

    def _on_io_placement_change(self, _evt=None):
        """Keep the genome controls runnable for the selected I/O strategy."""
        if self._selected_io_placement() not in (
                'wiring_chromosome', 'spatial_chromosome'):
            return
        try:
            chromosome_count = int(self._chroms_var.get())
        except (TypeError, ValueError):
            chromosome_count = 0
        if chromosome_count < 3:
            self._chroms_var.set('3')
            if hasattr(self, '_status'):
                self._status.set(
                    'Chroms raised to 3: chromosome 3 is the evolvable, '
                    'non-developmental I/O map.')

    def _selected_nv_profile(self):
        default = ('tri3', 'paper_analog', None)
        if not hasattr(self, '_nv_profile_var'):
            return default
        return self._NV_PROFILE_LABELS.get(
            self._nv_profile_var.get(), default)

    def _on_nv_profile_change(self, _evt=None):
        self._sync_nv_profile_controls()
        self._reconfigure_for_backend()
        self._refresh_target_list()

    def _sync_nv_profile_controls(self):
        """Keep labels and editability honest for the selected NV physics."""
        if not hasattr(self, '_pulse_entries'):
            return
        analog = self._selected_node_model()[0] == 'paper_analog'
        labels = (('Propagation delay:', 'Input width:', 'Coinc (emergent):')
                  if analog else ('Delay:', 'Width:', 'Coinc:'))
        for widget, text in zip(self._pulse_labels, labels):
            widget.configure(text=text)
        locked = bool(getattr(self, '_nv_controls_locked', False))
        for index, widget in enumerate(self._pulse_entries):
            widget.configure(state=('disabled'
                                    if locked or (analog and index == 2)
                                    else 'normal'))
        self._nv_profile_cb.configure(
            state='disabled' if locked else 'readonly')
        self._tune_reset_btn.configure(
            state='disabled' if locked else 'normal')
        if hasattr(self, '_analog_row'):
            for widget in self._analog_entries:
                widget.configure(state='disabled' if locked else 'normal')
            if (analog and hasattr(self, '_nb')
                    and self._backend() == 'nervous'):
                self._analog_row.pack(fill='x', side='top', before=self._nb)
            else:
                self._analog_row.pack_forget()

    def _set_nv_controls_locked(self, locked):
        self._nv_controls_locked = bool(locked)
        self._sync_nv_profile_controls()
        if hasattr(self, '_io_placement_cb'):
            self._io_placement_cb.configure(
                state=('disabled' if locked or self._backend() == 'fnv'
                       else 'readonly'))
        for widget in getattr(self, '_fnv_family_checks', ()):
            widget.configure(state='disabled' if locked else 'normal')
        for widget in getattr(self, '_lut_function_family_checks', ()):
            widget.configure(state='disabled' if locked else 'normal')

    def _sync_telomere_backend(self, backend=None):
        """Swap in the remembered growth ceiling for the selected backend.

        Fresh LUT runs start at 8 and the other models at 20. A value typed by
        the user is retained when switching away from a model and back.
        """
        backend = backend or self._backend()
        previous = getattr(self, '_telomere_backend', backend)
        if previous == backend or not hasattr(self, '_maxtel_var'):
            self._telomere_backend = backend
            return
        current = self._maxtel_var.get().strip()
        if current:
            self._telomere_values[previous] = current
        self._maxtel_var.set(self._telomere_values.get(
            backend, str(default_max_telomere(backend))))
        self._telomere_backend = backend

    def _on_backend_change(self, _evt=None):
        self._reconfigure_for_backend()
        # The same combinational name maps to static SNN data or a periodic
        # asynchronous wrapper, so refresh the selected object too.
        self._on_target_change()
        backend = self._backend()
        if backend == 'nervous':
            self._status.set('Model: Nervous net, coincidence + inhibition + loops; '
                             'best with small grids and close I/O.')
        elif backend == 'fnv':
            self._status.set(
                'Model: Functional NV Net - fixed physical functions on '
                'directed honeycomb wires; select whole component families.')
        elif backend == 'lut':
            self._status.set('Model: LUT array, square grid, 4 neighbours, four 16-bit '
                             'lookup tables per cell, asynchronous level logic (sim6 / Arch 2).')
        else:
            self._status.set('Model: SNN, leaky integrate-and-fire neurons.')

    def _read_arch(self):
        """Parse the substrate fields -> (Arch, input_high, ok)."""
        try:
            sw   = float(self._syn_var.get())
            vmin = float(self._vmin_var.get())
            vmax = float(self._vmax_var.get())
            high = float(self._cur_var.get())
            if sw <= 0 or high <= 0 or vmin < 0 or vmax < vmin:
                raise ValueError
        except ValueError:
            return None, None, False
        levels = tuple(round(vmin + (vmax - vmin) * i / 3.0, 4) for i in range(4))
        recurrent = (
            self._backend() == 'snn'
            and getattr(self.target, 'temporal', False))
        return Arch(syn_weight=sw, vth_levels=levels,
                    tau_levels=DEFAULT_ARCH.tau_levels,
                    recurrent=recurrent), high, True

    def _reset_arch(self):
        self._syn_var.set(str(DEFAULT_ARCH.syn_weight))
        self._vmin_var.set(str(DEFAULT_ARCH.vth_levels[0]))
        self._vmax_var.set(str(DEFAULT_ARCH.vth_levels[-1]))
        self._cur_var.set(str(self.target.high))
        self._chroms_var.set(
            '3' if self._selected_io_placement() in (
                'wiring_chromosome', 'spatial_chromosome')
            else '2')
        backend = self._backend()
        value = str(default_max_telomere(backend))
        self._maxtel_var.set(value)
        self._telomere_values[backend] = value

    def _open_custom(self):
        CustomTargetDialog(self.root, self._on_custom_built)

    def _on_custom_built(self, target):
        self._custom[target.name] = target
        self._target_picker.set_targets(self._targets_for_backend(self._backend()))
        self._target_picker.select(target.name, notify=True)

    # -- GA control ------------------------------------------------------------

    def _start_ga(self):
        selected = self._targets_for_backend(self._backend()).get(
            self._target_var.get().strip())
        if selected is None:
            self._status.set('Choose a target from the filtered list before running.')
            self._target_cb.focus_set()
            return
        if selected is not self.target:
            self._on_target_change()
        try:
            pop   = int(self._pop_var.get())
            gens  = int(self._gens_var.get())
            tries = int(self._tries_var.get())
            if pop < 2 or gens < 1 or tries < 1:
                raise ValueError
        except ValueError:
            self._status.set('Invalid parameters: Pop>=2, Gens>=1, Tries>=1 (integers).')
            return

        seed_txt = self._seed_var.get().strip().lower()
        if seed_txt in ('', 'random', 'rand', 'none'):
            # pick an explicit seed so the run is recorded and replayable
            base_seed = random.randrange(1, 2_000_000_000)
        else:
            try:
                base_seed = int(seed_txt)
            except ValueError:
                self._status.set("Invalid seed: use an integer or 'random'.")
                return
        self._active_seed = base_seed
        # show the actual seed in the field so it's visible and replayable
        # (type 'random' again to draw a fresh one)
        self._seed_var.set(str(base_seed))

        arch, high, ok = self._read_arch()
        if not ok:
            self._status.set('Invalid substrate: Syn weight>0, Input I>0, 0<=Vth min<=Vth max.')
            return

        try:
            n_chroms = int(self._chroms_var.get())
            if not 1 <= n_chroms <= MAX_CHROMS:
                raise ValueError
        except ValueError:
            self._status.set('Invalid genome: Chroms must be an integer from '
                             '1 to %d.' % MAX_CHROMS)
            return
        if (self._selected_io_placement() in (
                'wiring_chromosome', 'spatial_chromosome')
                and n_chroms < 3):
            self._status.set(
                'Chromosome-based I/O needs at least 3 chromosomes; '
                'chromosome 3 is the evolvable port map.')
            return

        run_config = self._read_run_config(chromosome_count=n_chroms)
        if run_config is None:
            self._status.set('Invalid tuning: Mutations>=0, 0<=Immigrants<1, '
                             'Tournament>=1, Elites>=0, 0<alpha<=1, 0<=beta<=10, '
                             'Mutation cap>=1, '
                             'Workers=1-%d, '
                             'Max telomere>=2, '
                             'at least one FNV family enabled, '
                             'Delay/Width>0, Coinc>=0; analog: 0<Vth<1, '
                             '(1-Vth)/2<Step<1-Vth, Tau>0, Hyst>=0, '
                             'Vth+Hyst<1.' % MAX_EVALUATION_WORKERS)
            return

        backend = self._backend()
        eff_target = self._effective_target(high, self._graded_var.get())
        setattr(eff_target, 'pulse_config', run_config.pulse)
        if backend == 'fnv':
            setattr(eff_target, '_fnv_families', run_config.fnv.families)
        # Mirror the evolvable I/O binding onto the display/scoring target so the
        # main-thread report and playback (which call prepare_net/prepare_lut/
        # interpret paths) bind ports the same way the run does. All backends
        # honour the strategy.
        setattr(eff_target, 'io_placement', run_config.ga.io_placement)
        if backend == 'lut':
            setattr(eff_target, 'lut_io_mode', run_config.ga.lut_io_mode)
        self._active_target  = eff_target
        self._active_arch    = arch
        self._active_chroms  = n_chroms
        self._active_backend = backend
        self._active_run_config = run_config
        self._disp_target    = eff_target
        self._disp_arch      = arch
        self._disp_backend   = backend

        self._gen_history.clear()
        self.best_genome = None
        self.best_fitness = 0.0
        self._abs_gen = 0
        self._n_unique_solvers = 0
        self._certification = None
        self._ga_error = None
        self._last_rebirth = None
        self._escape_status.set(run_config.ga.escape.summary())
        self._solver_save_error = None
        self._population_save_error = None
        self._stop_requested = False
        self._run_started = time.monotonic()
        self._work_phase = 'Starting worker processes'
        self._phase_detail = ''
        self._progress.configure(maximum=max(1, tries * (gens + 1)), value=0)
        self._best_line.set_data([], [])
        self._mean_line.set_data([], [])
        self._std_line.set_data([], [])
        self._genbest_line.set_data([], [])
        self._mutation_line.set_data([], [])
        self._mutation_text.set_text('Mutation: -')
        self._fit_ax.set_xlim(0, max(gens * tries, 10) + 1)
        self._fit_ax.set_title('Fitness vs Generation: %s' % self.target.name, fontsize=10)
        self._fit_canvas.draw_idle()
        self._set_tt('Evolving %s ...\n' % self.target.name)

        self._run_btn.config(state='disabled')
        self._pause_btn.config(state='normal', text='Pause')
        self._stop_btn.config(state='normal')
        self._load_btn.config(state='disabled')
        self._save_btn.config(state='disabled')
        self._target_picker.set_state('disabled')
        self._backend_cb.config(state='disabled')
        self._recombination_chk.config(state='disabled')
        self._set_nv_controls_locked(True)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._recombination_event = threading.Event()
        if run_config.ga.recombination_enabled:
            self._recombination_event.set()
        self._worker = threading.Thread(
            target=evolution_worker_entry,
            args=(gens, pop, n_chroms, tries, eff_target, arch, self.q, self._stop_event,
                  base_seed, backend, run_config, RESULTS_DIR, self._pause_event,
                  self._recombination_event),
            daemon=True)
        self._worker.start()
        backend_note = (('%s/%s/%s' %
                         (backend, run_config.ga.tile_arch,
                          run_config.pulse.model))
                        if backend == 'nervous' else backend)
        self._status.set('Evolving %s [%s] ...  pop=%d  gens=%d  tries=%d  seed=%d%s' %
                         (self.target.name, backend_note, pop, gens, tries, base_seed,
                          '  [graded]' if self._graded_var.get() else ''))

    def _sync_recombination(self):
        enabled = bool(self._recombination_var.get())
        if enabled:
            self._recombination_event.set()
        else:
            self._recombination_event.clear()
        if self._worker is not None:
            self._active_run_config = dataclasses.replace(
                self._active_run_config,
                ga=dataclasses.replace(
                    self._active_run_config.ga,
                    recombination_enabled=enabled))
            state = 'ON' if enabled else 'OFF'
            self._status.set(
                'Recombination %s: applies to the next offspring generation.'
                % state)

    def _toggle_pause(self):
        if self._worker is None:
            return
        if self._pause_event.is_set():
            self._sync_recombination()
            self._pause_event.clear()
            self._pause_btn.config(text='Pause')
            self._recombination_chk.config(state='disabled')
            self._work_phase = 'Resuming evolution'
            self._status.set(self._work_phase + '...')
        else:
            self._pause_event.set()
            self._pause_btn.config(text='Resume')
            self._recombination_chk.config(state='normal')
            self._work_phase = 'Pause requested: finishing current evaluation batch'
            self._status.set(self._work_phase + '...')

    def _stop_ga(self):
        self._stop_requested = True
        self._stop_event.set()
        self._pause_event.clear()
        self._pause_btn.config(state='disabled', text='Pause')
        self._stop_btn.config(state='disabled')
        self._work_phase = 'Stopping after the current evaluation batch'
        self._status.set(self._work_phase + '...')

    def _load_saved(self):
        path = CKPT if os.path.exists(CKPT) else LEGACY_CKPT
        if not os.path.exists(path):
            self._status.set('No saved genome at %s: run the GA first.' % CKPT)
            return
        try:
            state = load_checkpoint(path)
        except (OSError, ValueError) as exc:
            self._status.set('Could not load saved genome: %s' % exc)
            return
        if state.get('hand_built') and state.get('best_genome') is None:
            # a Designer phenotype with no genome: the main tabs all regrow from
            # DNA, so there is nothing to show here - open it in the Designer.
            self._status.set('Hand-built design (no genome): open it in the '
                             'Designer tab (Load design...).')
            return
        loaded_genome = state['best_genome']
        actual_chroms = len(getattr(loaded_genome, 'chromosomes', []) or [])
        if not 1 <= actual_chroms <= MAX_CHROMS:
            self._status.set(
                'Saved genome has %d chromosomes; this build supports 1-%d.' %
                (actual_chroms, MAX_CHROMS))
            return
        loaded_config = state.get('run_config') or RunConfig()
        if isinstance(loaded_config, dict):  # trusted legacy pickle migration
            try:
                loaded_config = RunConfig.from_dict(loaded_config)
            except ValueError as exc:
                self._status.set('Saved run configuration is invalid: %s' % exc)
                return
        configured_chroms = loaded_config.ga.chromosome_count
        loaded_backend = state.get('backend', 'snn')
        actual_tile_arch = getattr(loaded_genome, 'arch', 'single')
        chrom_warn = ''
        if configured_chroms is not None and configured_chroms != actual_chroms:
            chrom_warn = ('  : saved Chroms=%d disagreed with the genome; '
                          'using its actual count %d' %
                          (configured_chroms, actual_chroms))
        arch_warn = ''
        if (loaded_backend == 'nervous'
                and loaded_config.ga.tile_arch != actual_tile_arch):
            arch_warn = ('  : saved tile setting disagreed with the genome; '
                         'using %s' % actual_tile_arch)
        normalized_config = dataclasses.replace(
            loaded_config, ga=dataclasses.replace(
                loaded_config.ga, chromosome_count=actual_chroms,
                tile_arch=(actual_tile_arch if loaded_backend == 'nervous'
                           else loaded_config.ga.tile_arch)))

        self.best_genome  = loaded_genome
        self.best_fitness = state['best_fitness']
        # a Designer design with a genome AND hand edits: the main app regrows
        # from DNA, so it cannot reproduce the exact hand-edited grid - flag it
        # so the user knows to open it in the Designer for the precise circuit.
        hand_edited = bool(state.get('hand_built') and state.get('grid_edited'))
        saved_target = state.get('target')
        if saved_target is None:
            saved_target = get_target(state.get('target_name') or DEFAULT_TARGET)
        saved_arch = state.get('arch') or DEFAULT_ARCH
        self.target       = saved_target
        self._disp_target = saved_target
        self._disp_arch   = saved_arch
        self._graded_var.set(bool(getattr(saved_target, 'graded', False)))
        self._syn_var.set(str(saved_arch.syn_weight))
        self._vmin_var.set(str(saved_arch.vth_levels[0]))
        self._vmax_var.set(str(saved_arch.vth_levels[-1]))
        self._cur_var.set(str(saved_target.high))
        saved_seed = state.get('seed')
        if saved_seed is not None:
            self._seed_var.set(str(saved_seed))
            self._active_seed = saved_seed
        saved_backend = loaded_backend
        self._active_run_config = normalized_config
        if saved_backend == 'fnv':
            setattr(saved_target, '_fnv_families',
                    normalized_config.fnv.families)
        elif saved_backend == 'lut':
            setattr(
                saved_target, 'lut_io_mode',
                getattr(normalized_config.ga,
                        'lut_io_mode', 'source_pads'))
            setattr(
                saved_target, '_lut_function_families',
                normalized_config.ga.lut_function_families)
        self._beta_var.set(str(normalized_config.ga.stagnation_beta))
        self._mutation_limit_var.set(str(normalized_config.ga.mutation_limit))
        self._workers_var.set(str(normalized_config.ga.evaluation_workers))
        self._recombination_var.set(normalized_config.ga.recombination_enabled)
        self._delay_var.set(str(normalized_config.pulse.delay))
        self._width_var.set(str(normalized_config.pulse.width))
        self._coinc_var.set(str(normalized_config.pulse.coincidence))
        if hasattr(self, '_analog_defaults'):
            a = self._analog_defaults
            loaded_pulse = normalized_config.pulse
            self._athr_var.set(str(getattr(
                loaded_pulse, 'analog_threshold', a['vth'])))
            self._astep_var.set(str(getattr(
                loaded_pulse, 'analog_step', a['step'])))
            self._atau_var.set(str(getattr(
                loaded_pulse, 'analog_tau_leak', a['tau'])))
            self._ahyst_var.set(str(getattr(
                loaded_pulse, 'analog_hysteresis', a['hyst'])))
        saved_profile = (
            normalized_config.ga.tile_arch,
            normalized_config.pulse.model,
            getattr(normalized_config.ga, 'evolve_delay', None))
        profile_label = next(
            (label for label, profile in self._NV_PROFILE_LABELS.items()
             if profile == saved_profile), None)
        profile_warn = ''
        if profile_label is None:
            # A checkpoint saved under a retired engine. The dropdown can
            # only show the one live profile, so say plainly that the run
            # being displayed is not the one a fresh Run would use.
            profile_label = 'Analog tri-circuit (3-output, paper Fig. 1 node)'
            if loaded_backend == 'nervous':
                profile_warn = ('  : this checkpoint was saved under a '
                                'RETIRED NV engine; the shown profile is '
                                'the current one, not the saved one')
        self._nv_profile_var.set(profile_label)
        # Restore the I/O binding dropdown from the loaded run (old checkpoints
        # default to 'fixed').
        saved_io = getattr(normalized_config.ga, 'io_placement', 'fixed')
        io_label = next(
            (label for label, strat in self._IO_PLACEMENT_LABELS.items()
             if strat == saved_io), 'Fixed (original)')
        self._io_placement_var.set(io_label)
        for family, variable in getattr(
                self, '_fnv_family_vars', {}).items():
            variable.set(family in normalized_config.fnv.families)
        for family, variable in getattr(
                self, '_lut_function_family_vars', {}).items():
            variable.set(
                family in normalized_config.ga.lut_function_families)
        self._sync_nv_profile_controls()
        self._sync_recombination()
        self._chroms_var.set(str(actual_chroms))
        self._active_chroms = actual_chroms
        self._disp_backend   = saved_backend
        self._active_backend = saved_backend
        self._backend_var.set({
            'nervous': 'Nervous', 'fnv': 'FNV', 'lut': 'LUT'
        }.get(saved_backend, 'SNN'))
        if saved_target.name not in self._all_targets():
            self._custom[saved_target.name] = saved_target
        self._reconfigure_for_backend()      # filters the dropdown for the backend
        if saved_backend == 'lut':
            lut_label = next(
                (label for label, mode in self._LUT_IO_LABELS.items()
                 if mode == normalized_config.ga.lut_io_mode),
                LUT_PAD_IO_LABEL)
            self._io_placement_var.set(lut_label)
        # Exterior LUT buses are a fixed architecture over every exposed face;
        # unlike native source pads they require no per-genome layout field.
        native_layout = (
            ()
            if (saved_backend == 'lut'
                and normalized_config.ga.lut_io_mode == 'exterior_edges')
            else getattr(loaded_genome, 'input_layout', None))
        if (saved_backend in ('nervous', 'fnv', 'lut')
                and native_layout is None):
            legacy_label = (
                LEGACY_IO_LABEL if saved_io == 'fixed'
                else 'Legacy: %s' % io_label)
            self._io_placement_var.set(legacy_label)
            self._io_placement_cb.configure(
                values=[legacy_label], state='disabled')
        # A loaded checkpoint reflects its saved run, not the fresh-run default.
        saved_telomere = str(normalized_config.ga.max_telomere)
        self._maxtel_var.set(saved_telomere)
        self._telomere_values[saved_backend] = saved_telomere
        self._telomere_backend = saved_backend
        self._refresh_target_list()          # keep the backend filter (don't re-broaden;
                                             # LUT deliberately hides combinational targets)
        self._target_picker.select(saved_target.name)
        warn = ('  : hand-edited design: this view is REGROWN from the genome; '
                'open the Designer tab to see the exact edited circuit'
                if hand_edited else '')
        self._status.set('Loaded %s  fitness=%.4f  syn_w=%.2f%s' %
                         (saved_target.name, self.best_fitness,
                          saved_arch.syn_weight,
                          warn + chrom_warn + arch_warn + profile_warn))
        self._update_all(self.best_genome, self.best_fitness)
        self._save_btn.config(state='normal')

    # -- queue polling ---------------------------------------------------------

    def _redraw_fit_chart(self, max_pts=2000):
        """Push fitness and effective mutation history to the chart. The full
        history is kept, but each plotted line is reduced to at
        most ~`max_pts` points so redraw cost stays flat on 100k-generation runs.

        Reduction is a per-bucket MIN/MAX ENVELOPE per series, not a stride: the
        old `hist[::step]` sampling silently dropped single-generation features,
        so the spiky green gen-best line's dips vanished (and reappeared) as the
        stride changed with run length. Keeping each bucket's extremes means a
        one-generation dip or spike is always drawn, at any zoom-out level."""
        hist = self._gen_history
        if not hist:
            return
        series = ((self._best_line, 1), (self._mean_line, 2),
                  (self._genbest_line, 3), (self._mutation_line, 4),
                  (self._std_line, 5))
        if len(hist) <= max_pts:
            xs = [d[0] for d in hist]
            for line, col in series:
                line.set_data(xs, [d[col] for d in hist])
        else:
            size = math.ceil(len(hist) / (max_pts // 2))
            for line, col in series:
                xs, ys = [], []
                for b in range(0, len(hist), size):
                    chunk = hist[b:b + size]
                    lo = min(chunk, key=lambda d: d[col])
                    hi = max(chunk, key=lambda d: d[col])
                    for d in sorted({lo[0]: lo, hi[0]: hi}.values()):
                        xs.append(d[0]); ys.append(d[col])
                if xs[-1] != hist[-1][0]:          # always keep the latest point
                    xs.append(hist[-1][0]); ys.append(hist[-1][col])
                line.set_data(xs, ys)
        self._fit_ax.set_xlim(0, hist[-1][0] + 1)
        mutation_values = [row[4] for row in hist]
        self._mut_ax.set_ylim(0, max(1.05, max(mutation_values) * 1.08))
        self._mutation_text.set_text('Mutation: %.3f' % mutation_values[-1])
        self._fit_canvas.draw_idle()

    def _poll(self):
        last_gen = None                        # newest 'gen' this poll - redraw ONCE
        finished = False
        try:
            while True:
                msg  = self.q.get_nowait()
                kind = msg[0]
                if kind == 'gen':
                    (_, try_n, gen, best_f, mean_f, offspring_best,
                     mutation_rate, fitness_std) = msg
                    self._abs_gen += 1
                    self._gen_history.append(
                        (self._abs_gen, best_f, mean_f, offspring_best,
                         mutation_rate, fitness_std))
                    self._progress.configure(value=self._abs_gen)
                    last_gen = (try_n, gen, best_f, mean_f, offspring_best,
                                mutation_rate, fitness_std)
                elif kind == 'phase':
                    _, phase, current, total, amount = msg
                    self._work_phase = phase
                    if phase.startswith('Diversif') or phase.startswith('Preparing'):
                        self._phase_detail = ('round %d/%d, %d unique' %
                                              (current, total, amount))
                    else:
                        self._phase_detail = ('try %d, generation %d, population %d' %
                                              (current, total, amount))
                elif kind == 'paused':
                    paused = bool(msg[1])
                    self._work_phase = ('Paused between generations' if paused
                                        else 'Resuming evolution')
                    state = 'ON' if self._recombination_event.is_set() else 'OFF'
                    self._status.set('%s: recombination %s.' %
                                     (self._work_phase, state))
                elif kind == 'diverse':
                    _, n_unique, valid = msg
                    self._n_unique_solvers = n_unique
                    self._diversity_population_path = SOLVER_POP
                    self._status.set('Diversifying: built %d genotypically UNIQUE '
                                     'solvers (each >= %.3f) -> results/solver_generation.json'
                                     % (n_unique, valid))
                    # point the Diversity tab at the population just written;
                    # analysis stays opt-in (it grows every genome)
                    if getattr(self, '_diversity', None) is not None:
                        self._diversity.notify_population(SOLVER_POP)
                elif kind == 'solver_saved':
                    _, n_valid, valid, status, _path = msg
                    self._n_unique_solvers = n_valid
                    self._status.set(
                        '%s solver snapshot saved: %d valid genomes '
                        '(each >= %.3f) -> results/solver_generation.json'
                        % (status.capitalize(), n_valid, valid))
                    # Keep the Diversity tab on the full evaluated generation.
                    # Successful post-solve diversification has its own
                    # 'diverse' message, which deliberately selects the solver
                    # population instead.
                elif kind == 'population_saved':
                    _, count, status, path = msg
                    self._diversity_population_path = path
                    if getattr(self, '_diversity', None) is not None:
                        self._diversity.notify_population(path)
                    self._status.set(
                        '%s population saved for Diversity: %d genomes -> %s'
                        % (status.capitalize(), count,
                           os.path.basename(path)))
                elif kind == 'population_save_error':
                    _, tb = msg
                    lines = tb.strip().splitlines()
                    self._population_save_error = (
                        lines[-1] if lines else 'unknown save error')
                elif kind == 'solver_save_error':
                    _, tb = msg
                    lines = tb.strip().splitlines()
                    self._solver_save_error = (
                        lines[-1] if lines else 'unknown save error')
                elif kind == 'escape':
                    # Live escape telemetry, one message per generation while
                    # any mechanism is on. Rendered next to the controls that
                    # produced it so a run's behaviour is attributable.
                    stats = msg[1]
                    parts = [stats['summary']]
                    if stats['rebirths']:
                        last = getattr(self, '_last_rebirth', None)
                        detail = ('' if last is None else
                                  ' from gens %s' % ','.join(
                                      str(g) for g in last['ancestors']))
                        parts.append('rebirths %d (last gen %s%s)'
                                     % (stats['rebirths'],
                                        stats['last_rebirth_gen'], detail))
                    if stats['archive']:
                        parts.append('ancestors %d' % stats['archive'])
                    if stats['crowding_replacements']:
                        parts.append('crowd-replaced %d'
                                     % stats['crowding_replacements'])
                    if stats.get('migrations'):
                        parts.append('migrations %d' % stats['migrations'])
                    if stats.get('lineage_walk_steps'):
                        parts.append('walker steps %d'
                                     % stats['lineage_walk_steps'])
                    if stats.get('robust_blend'):
                        parts.append('worst-case %.0f%%'
                                     % (stats['robust_blend'] * 100))
                    if self._adaptive_mut_var.get():
                        parts.append('mean rate %.2f' % stats['mean_rate'])
                    self._escape_status.set('  |  '.join(parts))
                elif kind == 'rebirth':
                    # Stashed rather than written to the status bar: the status
                    # bar is rewritten with the generation line every poll, so
                    # a one-shot event posted there would never be readable.
                    self._last_rebirth = msg[1]
                elif kind == 'certified':
                    # Held-out verdict for the winning genome (CERTIFIED /
                    # OVERFIT / BELOW / SOLVED / PLATEAU / UNCERTIFIED).
                    self._certification = msg[1]
                elif kind == 'error':
                    _, tb = msg
                    lines = tb.strip().splitlines()
                    self._ga_error = lines[-1] if lines else 'unknown error'
                elif kind == 'best':
                    _, genome, fit = msg
                    self.best_genome  = genome
                    self.best_fitness = fit
                    # Coalesce the expensive full redraw: new bests arrive in
                    # bursts early on, and redrawing the growth/genome/trace
                    # panels each time makes the UI lag. Stash it and let the
                    # throttle below repaint at most a few times a second.
                    self._pending_best = (genome, fit)
                elif kind == 'done':
                    finished = True
                    _, genome, fit = msg
                    completion_error = None
                    if genome is not None and not self._ga_error:
                        self.best_genome  = genome
                        self.best_fitness = fit
                        save_target = getattr(self, '_active_target', self.target)
                        save_arch   = getattr(self, '_active_arch', DEFAULT_ARCH)
                        try:
                            save_checkpoint(
                                CKPT, genome, fit, save_target, save_arch,
                                getattr(self, '_active_seed', None),
                                getattr(self, '_active_backend', 'snn'),
                                getattr(self, '_active_run_config', None),
                                getattr(self, '_certification', None))
                            self._update_all(genome, fit)
                        except Exception as exc:
                            # Never strand the application in its locked
                            # running state because final persistence or a
                            # display refresh failed. The champion remains in
                            # memory and all controls are released below.
                            completion_error = '%s: %s' % (
                                type(exc).__name__, exc)
                        finally:
                            self._pending_best = None
                    self._run_btn.config(state='normal')
                    self._pause_btn.config(state='disabled', text='Pause')
                    self._stop_btn.config(state='disabled')
                    self._load_btn.config(state='normal')
                    self._target_picker.set_state('normal')
                    self._backend_cb.config(state='readonly')
                    self._recombination_chk.config(state='normal')
                    self._pause_event.clear()
                    self._save_btn.config(
                        state='normal' if self.best_genome else 'disabled')
                    self._progress.configure(value=self._progress.cget('maximum'))
                    self._worker = None
                    self._set_nv_controls_locked(False)
                    if completion_error:
                        self._status.set(
                            'Run finished, but final save/display failed: %s  '
                            '(controls re-enabled)' % completion_error)
                    elif getattr(self, '_ga_error', None):
                        self._status.set('GA stopped on error: %s  (controls re-enabled)'
                                         % self._ga_error)
                    else:
                        saved = ('  saved -> %s' % CKPT) if genome else ''
                        nu = getattr(self, '_n_unique_solvers', 0)
                        div = ('  |  %d valid solvers' % nu) if nu else ''
                        cert = getattr(self, '_certification', None)
                        cv = ('  |  %s' % cert['verdict']) if cert and cert.get('verdict') else ''
                        solver_error = getattr(self, '_solver_save_error', None)
                        sv = ('  |  solver snapshot NOT saved: %s' % solver_error
                              if solver_error else '')
                        population_error = getattr(
                            self, '_population_save_error', None)
                        pv = ('  |  diversity population NOT saved: %s'
                              % population_error if population_error else '')
                        outcome = 'Stopped' if getattr(self, '_stop_requested', False) else 'Done'
                        self._status.set('%s: %s fitness=%.4f  seed=%d (type it in Seed to replay)%s%s%s%s%s' %
                                         (outcome, self.target.name, fit,
                                          getattr(self, '_active_seed', 0),
                                          saved, div, cv, sv, pv))
        except queue.Empty:
            pass
        if last_gen is not None:               # redraw the fitness chart once per poll
            self._redraw_fit_chart()
        if last_gen is not None and not finished:
            (try_n, gen, best_f, mean_f, offspring_best,
             mutation_rate, fitness_std) = last_gen
            self._status.set('%s  seed=%d  try=%d  gen=%d  best=%.4f  offspring-best=%.4f  mean=%.4f  sigma=%.4f  mutation=%.3f' %
                             (self.target.name, getattr(self, '_active_seed', 0),
                              try_n, gen, best_f, offspring_best, mean_f,
                              fitness_std, mutation_rate))
        # Full circuit/traces rendering touches every hidden tab and can take
        # longer than an evaluation batch.  Keep the main loop responsive while
        # evolving; the final champion is rendered once on completion.
        if self._worker is not None and last_gen is None:
            elapsed = time.monotonic() - getattr(self, '_run_started', time.monotonic())
            phase = getattr(self, '_work_phase', 'Working')
            detail = getattr(self, '_phase_detail', '')
            self._status.set('%s... %s  elapsed %s' %
                             (phase, detail, self._format_elapsed(elapsed)))
        if self.root.winfo_exists():
            self._poll_job = self.root.after(60, self._poll)

    @staticmethod
    def _format_elapsed(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return '%d:%02d:%02d' % (hours, minutes, seconds)
        return '%d:%02d' % (minutes, seconds)

    def close(self):
        """Stop background work and scheduled callbacks before closing Tk."""
        self._stop_event.set()
        self._pause_event.clear()
        for tab in (getattr(self, '_interactive', None),
                    getattr(self, '_designer', None)):
            if tab is not None:
                tab.close()
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        ui_compat.cancel_after_callbacks(self.root)
        self.root.destroy()

    # -- display updates -------------------------------------------------------

    def _seed_tag(self):
        s = getattr(self, '_active_seed', None)
        return ('   seed=%d' % s) if s is not None else ''

    def _update_all(self, genome, fitness):
        self._update_truth_table(genome)
        self._draw_growth(genome, fitness)
        self._draw_voltages(genome)
        self._draw_genome(genome, fitness)

    def _set_tt(self, text, title=None):
        if title is not None and getattr(self, '_tt_frame', None) is not None:
            self._tt_frame.config(text=title)
        self._tt_text.config(state='normal')
        self._tt_text.delete('1.0', 'end')
        self._tt_text.insert('end', text)
        self._tt_text.config(state='disabled')

    def _update_truth_table(self, genome):
        temporal = getattr(self._disp_target, 'temporal', False)
        try:
            if self._disp_backend == 'fnv':
                text = functional_report(self._disp_target, genome)
            elif temporal:
                if self._disp_backend == 'lut':
                    from substrates.lut import lut_report
                    text = lut_report(self._disp_target, genome)
                elif self._disp_backend == 'snn':
                    text = snn_temporal_report(
                        self._disp_target, genome, self._disp_arch)
                else:
                    text = temporal_report(self._disp_target, genome)
            elif self._disp_backend == 'nervous':
                text = nervous_truth_table(genome, self._disp_target)
            elif self._disp_backend == 'lut':
                from substrates.lut import lut_truth_table
                text = lut_truth_table(genome, self._disp_target)
            else:
                text = build_truth_table(genome, self._disp_target, self._disp_arch)
        except Exception as exc:
            text = 'Error building report:\n' + str(exc)
        self._set_tt(text, title='Contract Score')

    def _draw_growth(self, genome, fitness):
        target = self._disp_target
        if self._disp_backend == 'fnv':
            self._draw_growth_fnv(genome, target, fitness)
            return
        if self._disp_backend == 'nervous':
            self._draw_growth_hex(genome, target, fitness)
            return
        if self._disp_backend == 'lut':
            self._draw_growth_lut(genome, target, fitness)
            return
        try:
            if getattr(target, 'temporal', False):
                prep = prepare_snn_temporal(
                    genome, target, self._disp_arch)
                if prep is None:
                    raise ValueError('incomplete temporal SNN')
                grid, ns_list, output_binding = prep[0], prep[1], prep[4]
                from substrates.nervous.io_placement import output_groups
                output_pos = {
                    cell: role
                    for role, cells in output_groups(output_binding).items()
                    for cell in cells}
            else:
                grid, ns_list, _ = interpret_for(
                    genome, target, self._disp_arch)
                output_pos = {
                    (n.x, n.y): n.out_role for n in ns_list if n.is_output}
            snapshots  = grow_snn_snapshots(
                genome, seeds=_growth_seeds(target, genome),
                                            grid_size=target.grid_size, iters=target.iters)
            rgba_fn = grid_to_rgba
        except Exception:
            return

        seed_set = set(target.inputs)
        gs       = target.grid_size
        n_panels = len(snapshots)
        total    = n_panels + 1                       # + legend cell
        ncols    = min(6, max(4, math.ceil(math.sqrt(total))))
        nrows    = math.ceil(total / ncols)

        self._growth_fig.clf()
        self._growth_fig.suptitle(
            '%s: Circuit Growth   (fitness=%.4f)%s' % (target.name, fitness, self._seed_tag()),
            fontsize=10, fontweight='bold', y=0.995)
        axes = self._growth_fig.subplots(
            nrows, ncols, squeeze=False,
            gridspec_kw={'hspace': 0.5, 'wspace': 0.15})
        flat = [ax for row in axes for ax in row]

        for idx, snap in enumerate(snapshots):
            ax  = flat[idx]
            img = rgba_fn(snap, gs, seed_set, output_pos)
            ax.imshow(img, origin='upper', aspect='equal', interpolation='nearest',
                      extent=[-0.5, gs - 0.5, -0.5, gs - 0.5])
            for i in range(gs + 1):
                ax.axhline(i - 0.5, color='#cccccc', lw=0.3)
                ax.axvline(i - 0.5, color='#cccccc', lw=0.3)
            ax.set_xlim(-0.5, gs - 0.5); ax.set_ylim(-0.5, gs - 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            label = 'Iter %d' % idx if idx > 0 else 'Seed'
            ax.set_title('%s  (%d)' % (label, len(snap)), fontsize=7, pad=2)
            for i, (sx, sy) in enumerate(target.inputs):
                ax.text(sx, gs - 1 - sy, chr(65 + i) if i < 26 else '*',
                        ha='center', va='center', fontsize=5,
                        color='white', fontweight='bold')
            for (ox, oy), role in output_pos.items():
                if (ox, oy) in snap:
                    ax.text(ox, gs - 1 - oy, role[:2],
                            ha='center', va='center', fontsize=4.5,
                            color='white', fontweight='bold')

        for idx in range(n_panels, len(flat)):
            flat[idx].set_visible(False)
        leg_ax = flat[n_panels]
        leg_ax.set_visible(True)
        leg_ax.axis('off')
        # only the SNN backend reaches here (nervous/lut return early above)
        patches = [
            mpatches.Patch(color=(0.9, 0.1, 0.1),    label='Input seed'),
            mpatches.Patch(color=(0.1, 0.8, 0.2),    label='Output neuron'),
            mpatches.Patch(color=(0.15, 0.35, 0.85), label='Excitatory'),
            mpatches.Patch(color=(0.90, 0.50, 0.10), label='Inhibitory'),
            mpatches.Patch(color=(1.0, 1.0, 1.0),    label='Empty', edgecolor='#aaa'),
        ]
        leg_ax.legend(handles=patches, loc='center', fontsize=7.5,
                      frameon=False, title='Node type', title_fontsize=8)
        self._growth_canvas.draw_idle()

    def _draw_growth_fnv(self, genome, target, fitness):
        try:
            inputs = functional_input_positions(genome, target.inputs)
            snapshots = grow_functional_snapshots(
                genome, seeds=inputs,
                grid_size=target.grid_size, iters=target.iters)
            prepared = prepare_functional(genome, target)
            outputs = prepared[2] if prepared is not None else {}
        except Exception:
            return
        fig = self._growth_fig
        fig.clf()
        fig.suptitle(
            '%s - Functional NV Net growth   (fitness=%.4f)%s' %
            (target.name, fitness, self._seed_tag()),
            fontsize=10, fontweight='bold', y=0.995)
        count = len(snapshots)
        columns = min(5, max(3, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        axes = fig.subplots(rows, columns, squeeze=False)
        flat = [axis for row in axes for axis in row]
        for index, snapshot in enumerate(snapshots):
            draw_functional_net(
                flat[index], snapshot, input_positions=inputs,
                output_positions=(outputs if index == count - 1 else {}),
                show_edges=(index == count - 1),
                title=('Seed (%d)' % len(snapshot) if index == 0 else
                       'Iter %d (%d)' % (index, len(snapshot))))
        for index in range(count, len(flat)):
            flat[index].set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._growth_canvas.draw_idle()

    def _draw_growth_hex(self, genome, target, fitness):
        """Growth tab for the hex nervous net: snapshots as honeycomb panels,
        the final one wired (excitatory green / inhibitory red)."""
        try:
            snaps = grow_nervous_snapshots(
                genome, seeds=_growth_seeds(target, genome),
                                           grid_size=target.grid_size, iters=target.iters)
            arch = getattr(genome, 'arch', 'single')
            routing, in_pos, out_pos = interpret_nervous(
                snaps[-1], target, arch=arch)
            # Honour an evolvable io_placement strategy: mark the tag-bound
            # cells the scorer actually drives/reads.
            from substrates.nervous.io_placement import io_strategy, bind_io, flat_inputs
            bound = None
            if io_strategy(target) != 'fixed':
                bound = bind_io(genome, snaps[-1], target)
            if bound is not None:
                in_pos, out_pos = flat_inputs(bound[0]), bound[1]
            elif getattr(target, 'temporal', False):
                from substrates.nervous import place_outputs_by_trace
                out_pos, _ = place_outputs_by_trace(
                    snaps[-1], routing, in_pos, target, arch=arch)
        except Exception:
            return
        fig = self._growth_fig
        fig.clf()
        fig.suptitle('%s: hex nervous-net growth   (fitness=%.4f)%s'
                     % (target.name, fitness, self._seed_tag()),
                     fontsize=10, fontweight='bold', y=0.995)
        gs    = target.grid_size
        n     = len(snaps)
        ncols = min(5, max(3, math.ceil(math.sqrt(n))))
        nrows = math.ceil(n / ncols)
        axes  = fig.subplots(nrows, ncols, squeeze=False)
        flat  = [a for row in axes for a in row]
        for idx, snap in enumerate(snaps):
            rt = ({} if arch == 'tri3' else
                  {p: ROUTING[s & 0x1F] for p, s in snap.items()})
            last = (idx == n - 1)
            draw_hex_net(flat[idx], snap, gs, routing=rt, in_pos=in_pos,
                         out_pos=(out_pos if last else {}), show_edges=last,
                         arch=arch,
                         title=('Iter %d (%d)' % (idx, len(snap))) if idx else 'Seed (%d)' % len(snap))
        for idx in range(n, len(flat)):
            flat[idx].set_visible(False)
        self._growth_canvas.draw_idle()

    def _draw_growth_lut(self, genome, target, fitness):
        """Growth tab for the LUT array: honeycomb-equivalent square panels drawn
        with the first-class renderer (per-side capability nibs + I/O markers),
        the final panel showing the mature organism, plus a legend."""
        try:
            from substrates.lut import (
                grow_lut_snapshots, place_outputs_by_trace,
                lut_growth_seeds)
            from substrates.lut.ga import _place_outputs_combinational
            snaps = grow_lut_snapshots(
                genome, seeds=lut_growth_seeds(
                    genome, target,
                    getattr(target, 'io_placement', 'fixed')),
                                       grid_size=target.grid_size, iters=target.iters)
        except Exception:
            return
        final  = snaps[-1] if snaps else {}
        from substrates.lut import (
            lut_input_positions, lut_exterior_inputs, lut_io_mode)
        exterior = lut_io_mode(target) == 'exterior_edges'
        evolved_layout = getattr(genome, 'input_layout', None) is not None
        if exterior:
            resolved_inputs, external_inputs = lut_exterior_inputs(
                genome, final, target.n_inputs)
        else:
            external_inputs = {}
            resolved_inputs = (
                lut_input_positions(genome, target.inputs)
                if evolved_layout else tuple(target.inputs))
        in_pos = (
            list(resolved_inputs) if exterior
            else [p for p in resolved_inputs if p in final])
        try:
            # Under an evolvable io_placement strategy the genome's tags choose
            # the ports - mark the SAME cells the scorer drives/reads.
            from substrates.nervous.io_placement import io_strategy, bind_io, flat_inputs
            bound = None
            if (not exterior and not evolved_layout
                    and io_strategy(target) != 'fixed'
                    and final):
                from substrates.lut.lut import cell_io_tags
                bound = bind_io(genome, final, target,
                                tags=cell_io_tags(genome, final))
            if bound is not None:
                in_pos, out_pos = flat_inputs(bound[0]), bound[1]
            elif getattr(target, 'temporal', False):
                out_pos, _ = place_outputs_by_trace(
                    final, list(resolved_inputs), target,
                    source_nodes=(
                        set() if exterior else
                        set(resolved_inputs) if evolved_layout else None),
                    external_inputs=external_inputs)
            else:
                out_pos = _place_outputs_combinational(
                    final, target, in_pos=list(resolved_inputs),
                    source_nodes=(
                        set() if exterior else
                        set(resolved_inputs) if evolved_layout else None),
                    external_inputs=external_inputs)
        except Exception:
            out_pos = {}
        fig = self._growth_fig
        fig.clf()
        fig.suptitle('%s: LUT array growth   (fitness=%.4f)%s'
                     % (target.name, fitness, self._seed_tag()),
                     fontsize=10, fontweight='bold', y=0.995)
        n = len(snaps)
        # The distinct lookup tables actually present in the mature organism,
        # most common first - shown below as REAL truth tables (green=1/white=0)
        # so the raw table content is visible, not just the wedge-colour hash.
        from collections import Counter
        from substrates.lut import lut_sop
        from substrates.lut.viz import _lut_color
        counts = Counter(v for st in final.values() for v in st if v)
        luts   = [v for v, _ in counts.most_common(8)]
        # Two STRUCTURED sections (a single uniform grid scattered the LUT tables
        # among the growth snapshots): growth stages + legend on top, the
        # distinct-LUT gallery in its own grid below - each section self-contained
        # so the tables no longer land in arbitrary leftover cells.
        snap_cells = n + 1                          # growth snapshots + one legend cell
        snap_cols  = min(6, max(2, snap_cells))
        snap_rows  = math.ceil(snap_cells / snap_cols)
        lut_cols   = min(6, max(1, len(luts)))
        lut_rows   = math.ceil(len(luts) / lut_cols) if luts else 0
        gs = fig.add_gridspec(2, 1, height_ratios=[snap_rows, max(1, lut_rows)],
                              hspace=0.35, top=0.92, bottom=0.05, left=0.04, right=0.97)
        gs_snap = gs[0].subgridspec(snap_rows, snap_cols, hspace=0.45, wspace=0.15)
        for idx, snap in enumerate(snaps):
            last = (idx == n - 1)
            ax = fig.add_subplot(gs_snap[idx // snap_cols, idx % snap_cols])
            draw_lut_net(ax, snap,
                         in_pos=(in_pos if (last or not exterior) else []),
                         out_pos=(out_pos if last else {}), show_edges=True,
                         external_inputs=(
                             external_inputs if last else None),
                         title=('Iter %d (%d)' % (idx, len(snap))) if idx
                               else 'Seed (%d)' % len(snap))
        leg = fig.add_subplot(gs_snap[n // snap_cols, n % snap_cols])
        leg.axis('off')
        patches = [
            mpatches.Patch(facecolor='white', edgecolor='#b02020', lw=2, label='Input seed'),
            mpatches.Patch(facecolor='white', edgecolor='#17902f', lw=2, label='Output cell'),
            mpatches.Patch(facecolor='#7ec8e3', edgecolor='#c3ccd6',
                           label='Cell = 4 directional LUTs'),
            mpatches.Patch(facecolor='#e0b0ff', edgecolor='#c3ccd6',
                           label="wedge colour = that LUT's state"),
            mpatches.Patch(facecolor=(0.95, 0.96, 0.97), edgecolor='#c3ccd6',
                           label='Dead direction (LUT = 0)'),
            mpatches.Patch(facecolor='#1ea64a', edgecolor='#b9c2cc',
                           label='tables below: green=1, white=0'),
        ]
        leg.legend(handles=patches, loc='center', fontsize=7, frameon=False,
                   title='LUT array: growth stages above,\ndistinct lookup tables below (Fig. 13)',
                   title_fontsize=7.5)
        if luts:
            gs_lut = gs[1].subgridspec(lut_rows, lut_cols, hspace=0.6, wspace=0.2)
            for k, v in enumerate(luts):
                sop = lut_sop(v)
                if len(sop) > 16:
                    sop = sop[:15] + '...'
                ax = fig.add_subplot(gs_lut[k // lut_cols, k % lut_cols])
                draw_lut_table(ax, v, swatch=_lut_color(v),
                               title='LUT %04X  x%d\n%s' % (v, counts[v], sop))
        self._growth_canvas.draw_idle()

    def _draw_lut_dynamics(self, genome):
        """Activity tab for a temporal LUT target: the paper's Fig. 14 'motion
        picture' as a filmstrip, the array's OUTPUT BITS (green=1 / red=0 per
        direction) at a spread of seconds through the first trial, so the running
        dynamics that ARE the LUT's computation are visible over time."""
        target = self._disp_target
        try:
            from substrates.lut import (
                AsyncLutSim, prepare_lut, lut_exterior_inputs, lut_io_mode)
            prepared = prepare_lut(genome, target)
            if prepared is None:
                raise ValueError
            grid, out_pos, _traces, bound_in = prepared
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: circuit incomplete, grew too little)')
            return
        trial   = target.trials[0]
        from substrates.nervous.io_placement import (
            input_groups, flat_inputs, terminal_node_sets)
        # Drive the SAME cells the scorer drives: the bound attachment groups
        # under an evolvable strategy (an input may fan out to several sites; a
        # shared site wired-ORs), the seed pads otherwise.
        groups = input_groups(bound_in)
        in_pos  = flat_inputs(groups)                  # terminal membership
        exterior_inputs = {}
        if lut_io_mode(target) == 'exterior_edges':
            _, exterior_inputs = lut_exterior_inputs(
                genome, grid, target.n_inputs)
            terminal_inputs, terminal_outputs = set(), set()
        elif getattr(genome, 'input_layout', None) is not None:
            terminal_inputs, terminal_outputs = set(in_pos), set()
        else:
            terminal_inputs, terminal_outputs = terminal_node_sets(
                target, groups, out_pos)
        sim = AsyncLutSim(
            grid, input_nodes=terminal_inputs,
            output_nodes=terminal_outputs,
            external_inputs=exterior_inputs)
        frames  = []                                   # (tick, nibble-map)
        physical = getattr(trial, 'input_events', None)
        if physical is not None:
            # float-time stimulus: inject the real (sub-tick) schedule and
            # sample the running levels mid-tick, as scoring does
            for i, cells in enumerate(groups):
                for start, width in (physical[i] if i < len(physical) else ()):
                    for cell in cells:
                        sim.inject_pulse(cell, start, width)
            for t in range(target.T):
                sim.advance_to(t + 0.5)
                frames.append((t, dict(sim.out)))
        else:
            for t in range(target.T):
                levels = {}
                for i, cells in enumerate(groups):
                    bit = trial.streams[t][i] if i < len(trial.streams[t]) else 0
                    for cell in cells:
                        levels[cell] = levels.get(cell, 0) | bit
                sim.step(levels)
                frames.append((t, dict(sim.out)))
        # show up to 8 evenly-spaced ticks
        k    = min(8, len(frames))
        idxs = [round(i * (len(frames) - 1) / max(1, k - 1)) for i in range(k)]
        self._volt_fig.clf()
        self._volt_fig.suptitle('%s: LUT output dynamics over seconds  '
                                '(green=1 / red=0, Fig. 14)%s'
                                % (target.name, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        ncols = min(4, k)
        nrows = math.ceil(k / ncols)
        axes  = self._volt_fig.subplots(nrows, ncols, squeeze=False)
        flat  = [a for row in axes for a in row]
        for j, fi in enumerate(idxs):
            tick, nib = frames[fi]
            draw_lut_net(flat[j], grid, activity=nib, in_pos=groups,
                         out_pos=out_pos, show_edges=True,
                         external_inputs=exterior_inputs,
                         title='second %d' % tick)
        for j in range(k, len(flat)):
            flat[j].set_visible(False)
        self._volt_canvas.draw_idle()

    def _draw_activity_lut(self, genome):
        """Activity tab for the LUT array. Temporal targets (the only kind LUT
        runs) show the running output dynamics; the combinational per-case view
        is kept defensively for any combinational target reaching here."""
        target = self._disp_target
        if getattr(target, 'temporal', False):
            self._draw_lut_dynamics(genome)
            return
        try:
            from substrates.lut import lut_case_outputs
            grid, out_pos, cases = lut_case_outputs(genome, target)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: cannot evaluate this circuit)')
            return
        if not cases:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: circuit incomplete, inputs/outputs missing)')
            return
        from substrates.lut import (
            lut_exterior_inputs, lut_io_mode)
        exterior_inputs = {}
        if lut_io_mode(target) == 'exterior_edges':
            in_pos, exterior_inputs = lut_exterior_inputs(
                genome, grid, target.n_inputs)
            in_pos = list(in_pos)
        elif getattr(genome, 'input_layout', None) is not None:
            from substrates.lut import lut_input_positions
            in_pos = [p for p in lut_input_positions(
                genome, target.inputs) if p in grid]
        else:
            in_pos = [p for p in target.inputs if p in grid]
        cases  = cases[:MAX_VOLT_CASES]
        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s: LUT array response per case%s%s'
                                % (target.name, extra, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        axes = self._volt_fig.subplots(1, len(cases), squeeze=False)[0]
        for ci, case in enumerate(cases):
            all_ok = all(case['acts'][t.role] == case['out_bits'][i]
                         for i, t in enumerate(target.outputs))
            res = '  '.join('%s %d/%s' % (t.role, case['out_bits'][i],
                            '?' if case['acts'][t.role] is None else case['acts'][t.role])
                            for i, t in enumerate(target.outputs))
            settled = '' if case.get('stable', True) else '  (unsettled)'
            draw_lut_net(axes[ci], grid, activity=case['node_nibbles'],
                         in_pos=in_pos, out_pos=out_pos, show_edges=True,
                         external_inputs=exterior_inputs,
                         title='in=%s\n%s  %s%s' % (''.join(map(str, case['in_bits'])),
                                                  res, 'OK' if all_ok else 'FAIL', settled))
            axes[ci].set_title(axes[ci].get_title(),
                               color='green' if all_ok else 'red', fontsize=7)
        self._volt_canvas.draw_idle()

    def _draw_voltages(self, genome):
        if self._disp_backend == 'fnv':
            self._draw_activity_fnv(genome)
            return
        if self._disp_backend == 'nervous':
            self._draw_activity(genome)
            return
        if self._disp_backend == 'lut':
            self._draw_activity_lut(genome)
            return
        target = self._disp_target
        if getattr(target, 'temporal', False):
            self._draw_placeholder(
                self._volt_fig, self._volt_canvas,
                'Temporal recurrent SNN: the fitted event/state traces are '
                'listed in the Contract Score panel; use Interactive to load '
                'or edit a complete trial timeline.')
            return
        try:
            grid, ns, ss = interpret_for(genome, target, self._disp_arch)
            from substrates.nervous.io_placement import (io_strategy, bind_io,
                                             input_groups)
            if io_strategy(target) != 'fixed':
                from substrates.snn.growth import cell_io_tags
                bound = bind_io(
                    genome, grid, target, tags=cell_io_tags(genome, grid))
                groups = input_groups(bound[0]) if bound else []
            else:
                groups = input_groups(target.inputs)
            by_pos = {(n.x, n.y): n for n in ns if n.is_input}
            in_ids = [[by_pos[pos].id for pos in cells if pos in by_pos]
                      for cells in groups]
            if any(not ids for ids in in_ids):
                raise ValueError
            out_neurons = []
            for term in target.outputs:
                members = [n for n in ns
                           if n.is_output and n.out_role == term.role]
                if not members:
                    raise ValueError
                out_neurons.append(members)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(circuit incomplete: no I/O neurons to trace)')
            return

        cases   = target.cases[:MAX_VOLT_CASES]
        n_rows  = len(cases)
        n_cols  = len(target.outputs)
        track   = [n.id for members in out_neurons for n in members]
        t       = np.arange(N_STEPS) * DT
        palette = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d cases)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s: Membrane Voltages%s%s' % (target.name, extra, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        gspec = gridspec.GridSpec(n_rows, n_cols, figure=self._volt_fig,
                                  hspace=0.6, wspace=0.35,
                                  top=0.90, bottom=0.07, left=0.09, right=0.97)

        for row, (in_bits, out_bits) in enumerate(cases):
            encodings = {term.complement_inputs for term in target.outputs}
            sims = {c: simulate_vmem(ns, ss,
                                     _case_currents(target, in_bits, in_ids, c), track)
                    for c in encodings}
            for col, term in enumerate(target.outputs):
                members    = out_neurons[col]
                neuron     = members[0]
                spk, vmem  = sims[term.complement_inputs]
                fired      = any(len(spk.get(n.id, [])) >= 1 for n in members)
                act        = (0 if fired else 1) if term.invert_spike else (1 if fired else 0)
                exp        = out_bits[col]
                color      = palette[col % len(palette)]
                ax = self._volt_fig.add_subplot(gspec[row, col])
                bus_v = np.max(
                    np.vstack([vmem[n.id] for n in members]), axis=0)
                ax.plot(t, bus_v, color=color, lw=1.1)
                ax.axhline(neuron.vth, color='gray', lw=0.8, ls='--', alpha=0.7)
                ax.set_xlim(0, SIM_TIME)
                ax.set_ylim(-0.15, neuron.vth + 0.35)
                ax.set_ylabel('V', fontsize=8)
                if row == 0:
                    notes = []
                    if term.complement_inputs:
                        notes.append('complement in')
                    if term.invert_spike:
                        notes.append('fires => 0')
                    note = ('  [%s]' % ', '.join(notes)) if notes else ''
                    ax.set_title("%s  %d site%s%s" % (
                                     term.role, len(members),
                                     '' if len(members) == 1 else 's', note),
                                 fontsize=8.5)
                ax.text(0.02, 0.88, 'in=%s' % ''.join(map(str, in_bits)),
                        transform=ax.transAxes, fontsize=7.5, va='top')
                # Make the spike->logic mapping explicit for any experimental
                # inverted output, so its result cannot look contradictory.
                fy  = 'Y' if fired else 'N'
                res = 'fired=%s -> %d  exp=%d  %s' % (fy, act, exp,
                                                          'OK' if act == exp else 'FAIL')
                ax.text(0.98, 0.88, res, transform=ax.transAxes,
                        fontsize=7.5, va='top', ha='right',
                        color='green' if act == exp else 'red', fontweight='bold')
                if row == n_rows - 1:
                    ax.set_xlabel('Time (ms)', fontsize=8)
        self._volt_canvas.draw_idle()

    def _draw_activity_fnv(self, genome):
        target = self._disp_target
        if getattr(target, 'temporal', False):
            self._draw_placeholder(
                self._volt_fig, self._volt_canvas,
                'Temporal FNV target - the fitted physical event/state traces '
                'are listed in Contract Score.')
            return
        try:
            grid, inputs, outputs, cases = functional_case_outputs(
                genome, target)
        except Exception:
            self._draw_placeholder(
                self._volt_fig, self._volt_canvas,
                '(FNV: cannot evaluate this circuit)')
            return
        if not cases:
            self._draw_placeholder(
                self._volt_fig, self._volt_canvas,
                '(FNV: circuit incomplete - inputs/outputs missing)')
            return
        cases = cases[:MAX_VOLT_CASES]
        self._volt_fig.clf()
        self._volt_fig.suptitle(
            '%s - Functional NV Net activity per case%s' %
            (target.name, self._seed_tag()),
            fontsize=10, fontweight='bold', y=0.99)
        axes = self._volt_fig.subplots(1, len(cases), squeeze=False)[0]
        for index, case in enumerate(cases):
            correct = all(
                case['acts'][terminal.role] == case['out_bits'][out_index]
                for out_index, terminal in enumerate(target.outputs))
            result = '  '.join(
                '%s %d/%d' % (
                    terminal.role, case['out_bits'][out_index],
                    case['acts'][terminal.role])
                for out_index, terminal in enumerate(target.outputs))
            draw_functional_net(
                axes[index], grid, input_positions=inputs,
                output_positions=outputs, activity=case['node_outputs'],
                show_edges=True,
                title='in=%s\n%s  %s' % (
                    ''.join(map(str, case['in_bits'])), result,
                    'OK' if correct else 'FAIL'))
            axes[index].set_title(
                axes[index].get_title(),
                color='green' if correct else 'red', fontsize=7)
        self._volt_canvas.draw_idle()

    def _draw_activity(self, genome):
        """Nervous-net version of the Voltage tab: node activity per input case."""
        target = self._disp_target
        if getattr(target, 'temporal', False):
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   'Temporal target: drive it over time in the Interactive tab '
                                   '(Step / Run).')
            return
        try:
            grid, in_pos, out_pos, cases = nervous_case_outputs(genome, target)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(nervous: cannot evaluate this circuit)')
            return
        if not cases:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(nervous: circuit incomplete, inputs/outputs missing)')
            return

        gs      = target.grid_size
        cases   = cases[:MAX_VOLT_CASES]
        arch = getattr(genome, 'arch', 'single')
        routing = interpret_nervous(grid, target, arch=arch)[0]
        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s: hex nervous-net activity per case%s%s'
                                % (target.name, extra, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        axes = self._volt_fig.subplots(1, len(cases), squeeze=False)[0]
        for ci, case in enumerate(cases):
            all_ok = all(case['acts'][t.role] == case['out_bits'][i]
                         for i, t in enumerate(target.outputs))
            res = '  '.join('%s %d/%d' % (t.role, case['out_bits'][i], case['acts'][t.role])
                            for i, t in enumerate(target.outputs))
            draw_hex_net(axes[ci], grid, gs, routing=routing, in_pos=in_pos, out_pos=out_pos,
                         activity=case['node_outputs'],
                         show_edges=(arch == 'single'),
                         arch=arch,
                         title='in=%s\n%s  %s' % (''.join(map(str, case['in_bits'])), res,
                                                  'OK' if all_ok else 'FAIL'))
            axes[ci].set_title(axes[ci].get_title(),
                               color='green' if all_ok else 'red', fontsize=7)
        self._volt_canvas.draw_idle()

    # -- genome viewer ---------------------------------------------------------

    # one colour per cell-state (categorical, so equal states read the same)
    _STATE_CMAP = plt.cm.tab20

    def _state_color(self, v):
        c = self._STATE_CMAP(v % 20)
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return c, ('white' if lum < 0.55 else '#222')

    def _draw_gene(self, ax, ox, oy, gene):
        """One gene as a card: a 3x3 neighbourhood pattern -> output chip."""
        # card
        ax.add_patch(FancyBboxPatch((ox - 0.25, oy - 0.45), 6.3, 3.9,
                     boxstyle='round,pad=0.02,rounding_size=0.2',
                     facecolor='white', edgecolor='#cdd6e2', lw=1.0, zorder=1))

        lut = hasattr(gene, 'ctx_n')                       # 16-bit LUT gene

        def chip(cx, cy, val, sz=0.92, fs=8.5):
            face, txt = self._state_color((val * 2654435761) % 20 if lut else val)
            ax.add_patch(FancyBboxPatch((cx, cy), sz, sz,
                         boxstyle='round,pad=0,rounding_size=0.14',
                         facecolor=face, edgecolor='#5b6b7d', lw=0.7, zorder=2))
            ax.text(cx + sz / 2, cy + sz / 2,
                    ('%04X' % val) if lut else str(val), ha='center', va='center',
                    fontsize=(fs * 0.55) if lut else fs,
                         fontweight='bold', color=txt, zorder=3)

        if hasattr(gene, 'component_id'):
            from substrates.fnv.catalogue import BY_ID
            entry = BY_ID[gene.component_id]
            refs = ', '.join('%d:%s' % (ref.node_id, ref.direction)
                             for ref in gene.inputs)
            ax.text(ox + 0.15, oy + 0.25, 'id %d   branch %d' % (
                gene.gene_id, gene.branch_id), fontsize=7.5,
                color='#556', zorder=3)
            chip(ox + 0.2, oy + 1.05, gene.component_id, sz=1.25, fs=10)
            ax.text(ox + 1.7, oy + 1.15, entry.name, fontsize=7.2,
                    fontweight='bold', color='#223', zorder=3)
            ax.text(ox + 1.7, oy + 1.75, 'inputs: ' + refs,
                    fontsize=7.0, color='#556', zorder=3)
            ax.text(ox + 1.7, oy + 2.35,
                    'outputs: ' + ','.join(entry.outputs),
                    fontsize=7.0, color='#556', zorder=3)
            return

        # neighbourhood (y grows downward; axis is inverted below)
        if hasattr(gene, 'ctx_l'):                         # hex gene: L / R / D
            chip(ox + 0, oy + 1, gene.ctx_l)               # L
            chip(ox + 1, oy + 1, gene.self_in)             # centre = expected self
            chip(ox + 2, oy + 1, gene.ctx_r)               # R
            chip(ox + 1, oy + 2, gene.ctx_d)               # D
        elif lut:                                          # LUT gene: N / E / S / W (hex)
            chip(ox + 1, oy + 0, gene.ctx_n)               # N
            chip(ox + 0, oy + 1, gene.ctx_w)               # W
            chip(ox + 1, oy + 1, gene.self_in)             # centre = expected self
            chip(ox + 2, oy + 1, gene.ctx_e)               # E
            chip(ox + 1, oy + 2, gene.ctx_s)               # S
        else:                                              # SNN gene: N / E / S / W plus
            chip(ox + 1, oy + 0, gene.state_n)             # N
            chip(ox + 0, oy + 1, gene.state_w)             # W
            chip(ox + 1, oy + 1, gene.self_in)             # centre = expected self
            chip(ox + 2, oy + 1, gene.state_e)             # E
            chip(ox + 1, oy + 2, gene.state_s)             # S
        # output chip (bigger), with an arrow from the pattern
        ax.add_patch(FancyArrowPatch((ox + 3.1, oy + 1.45), (ox + 4.05, oy + 1.45),
                     arrowstyle='-|>', mutation_scale=11, color='#445', lw=1.4, zorder=2))
        chip(ox + 4.2, oy + 0.85, gene.self_out, sz=1.25, fs=11)
        # growth-limit badge (top-right of card) - SNN genes only; the hex gene
        # has no per-gene iteration limit (paper-faithful associative memory)
        if hasattr(gene, 'limit'):
            ax.text(ox + 5.85, oy - 0.18, 'it<=%d' % gene.limit, ha='right', va='top',
                    fontsize=6.5, color='#888', zorder=3)
        # I/O-binding tag badge (bottom-right) - only when the gene carries one,
        # so default genomes draw exactly as before
        if getattr(gene, 'tag', 0):
            ax.text(ox + 5.85, oy + 3.05, 'io:%d' % gene.tag, ha='right', va='top',
                    fontsize=6.5, color='#a03070', zorder=3)

    def _io_binding_text(self, genome):
        """The evolved I/O binding summary for the display target, or None when
        the fixed strategy is active / anything is missing. Grows the organism
        with the backend's own growth + tag attribution so the summary names
        exactly the cells the scorer drives and reads."""
        target = getattr(self, '_disp_target', None)
        if genome is None or target is None:
            return None
        try:
            from substrates.nervous.io_placement import io_strategy, describe_binding
            if io_strategy(target) == 'fixed':
                return None
            backend = getattr(self, '_disp_backend', 'snn')
            if backend == 'nervous':
                from substrates.nervous.nervous import grow_nervous
                grid = grow_nervous(
                    genome, seeds=_growth_seeds(target, genome))
                tags = None                       # nervous attribution is the default
            elif backend == 'lut':
                from substrates.lut.lut import grow_lut, cell_io_tags
                grid = grow_lut(genome, seeds=_growth_seeds(target, genome),
                                grid_size=target.grid_size, iters=target.iters)
                tags = cell_io_tags(genome, grid)
            else:
                from substrates.snn.growth import grow_snn, cell_io_tags
                grid = grow_snn(genome, seeds=_growth_seeds(target, genome),
                                grid_size=target.grid_size)
                tags = cell_io_tags(genome, grid)
            return describe_binding(genome, grid, target, tags=tags)
        except Exception:
            return None

    def _draw_genome(self, genome, fitness):
        fig = self._genome_fig
        fig.clf()
        chroms = list(getattr(genome, 'chromosomes', []) or [])
        if not chroms:
            self._draw_placeholder(fig, self._genome_canvas, '(empty genome)')
            return
        n_total = sum(len(c.genes) for c in chroms)
        title = 'Genome :  %d chromosome%s,  %d genes' % (
            len(chroms), '' if len(chroms) == 1 else 's', n_total)
        if fitness is not None:
            title += '   (fitness = %.4f)' % fitness
        title += self._seed_tag()

        g0 = next((g for c in chroms for g in c.genes), None)
        constructive_fnv = g0 is not None and hasattr(g0, 'component_id')
        lutmode = g0 is not None and hasattr(g0, 'ctx_n')
        CARD_CAP = 24
        # A dense LUT genome (hundreds of ontogeny genes) can't be shown as one
        # card per gene - they overlap into noise. Show its VOCABULARY instead:
        # the distinct output lookup tables it installs, most-common first, the
        # way the Growth tab summarises the grown organism.
        if lutmode and n_total > CARD_CAP:
            self._draw_genome_lut_summary(fig, chroms, n_total, title)
            binding = self._io_binding_text(genome)
            if binding:
                fig.text(0.01, 0.02, binding, fontsize=7.5, color='#a03070',
                         va='bottom', family='monospace')
            self._genome_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.axis('off')

        CW, CH  = 7.0, 4.4        # card footprint (incl. spacing)
        per_row = 4
        y       = 0.0
        drawn   = 0
        truncated = False

        for ci, chrom in enumerate(chroms):
            tel = getattr(chrom, 'telomere', None)
            is_wiring = getattr(chrom, 'wiring', False)
            ax.text(0.0, y, "Chromosome %s   -   tag %d   -   split %s%s   -   %d genes%s"
                    % (chr(ord('a') + ci), chrom.tag, _split_display(chrom),
                       ('   -   telomere %d' % tel) if tel is not None else '',
                       len(chrom.genes),
                       '   -   WIRING (I/O port map)' if is_wiring else ''),
                    fontsize=9.5, fontweight='bold',
                    color='#a03070' if is_wiring else '#334', va='bottom')
            y += 1.0
            drawn_here = 0
            for gene in chrom.genes:
                if drawn >= CARD_CAP:
                    truncated = True
                    break
                # position within THIS chromosome's drawn cards
                self._draw_gene(ax, (drawn_here % per_row) * CW,
                                y + (drawn_here // per_row) * CH, gene)
                drawn += 1
                drawn_here += 1
            # advance by the rows actually drawn, not the full gene count - the
            # old code stretched the axis by hundreds of unused rows on a capped
            # genome, which (with aspect='equal') crushed the cards into overlap.
            y += (math.ceil(drawn_here / per_row) if drawn_here else 1) * CH + 0.7
            if truncated:
                ax.text(0.0, y, '...  %d more genes not shown (see the genome .txt export)'
                        % (n_total - drawn), fontsize=9, color='#a00', va='bottom')
                y += 1.0
                break

        _hex = bool(g0 is not None and hasattr(g0, 'ctx_l'))
        if constructive_fnv:
            _sides = 'stable source/output-port labels'
            _tail = '; dependency order places components and collisions fail locally.'
        else:
            _sides = ('rotated L/R/D circuit states (each 0-15)'
                      if _hex else 'N/E/S/W sides')
            _tail = ('.' if _hex else
                     ', active while iter <= limit.')
        explanation = (
            'card = one fixed component placement: its stable ID owns the '
            'listed output ports and its named inputs anchor it to existing '
            'physical branches%s' % _tail
            if constructive_fnv else
            'card = one gene: the cluster is the expected neighbourhood '
            '(%s + centre = self), the chip after -> is the output state; '
            'growth picks the gene closest (min Hamming distance) to a circuit%s'
            % (_sides, _tail))
        ax.text(0.0, y + 0.2, explanation,
                fontsize=7.5, color='#666', va='bottom', wrap=True)
        y += 1.0
        # Evolved I/O binding footer: which allele/anchor each port selected and
        # attached (only under an evolvable io_placement strategy).
        binding = self._io_binding_text(genome)
        if binding:
            ax.text(0.0, y + 0.4, binding, fontsize=8, color='#a03070',
                    va='top', family='monospace')
            y += 1.2 + 0.55 * binding.count('\n')

        ax.set_xlim(-0.4, per_row * CW)
        ax.set_ylim(0, y + 1.4)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        self._genome_canvas.draw_idle()

    def _draw_genome_lut_summary(self, fig, chroms, n_total, title):
        """Faithful per-CHROMOSOME view of a dense LUT genome: one row per
        chromosome showing its stats + the most-common output lookup tables IT
        installs (its own 'vocabulary'), as real truth grids with xcounts + SOP.
        Shows the genome's structure (which chromosome carries which LUTs) rather
        than a global merge or an unreadable wall of one card per gene."""
        from collections import Counter
        from substrates.lut import lut_sop
        from substrates.lut.viz import _lut_color
        K = 6                                     # top output LUTs shown per chromosome
        shown = chroms[:8]
        axes = fig.subplots(len(shown), K + 1, squeeze=False)
        fig.suptitle(title, fontsize=11, fontweight='bold')
        for r, c in enumerate(shown):
            counts = Counter(g.self_out for g in c.genes if g.self_out)
            n_growth = sum(1 for g in c.genes if getattr(g, 'self_in', 1) == 0)
            tel = getattr(c, 'telomere', None)
            is_wiring = getattr(c, 'wiring', False)
            lbl = axes[r][0]
            lbl.axis('off')
            lbl.text(0.0, 0.5, 'Chrom %s%s\n%d genes\n%d distinct\n%d growth%s' % (
                chr(ord('a') + r), ' [WIRING]' if is_wiring else '',
                len(c.genes), len(counts), n_growth,
                ('\ntel %d' % tel) if tel is not None else ''),
                va='center', ha='left', fontsize=8.5, family='monospace',
                color='#a03070' if is_wiring else 'black',
                transform=lbl.transAxes)
            top = counts.most_common(K)
            for k in range(K):
                ax = axes[r][k + 1]
                if k < len(top):
                    w, cnt = top[k]
                    sop = lut_sop(w)
                    if len(sop) > 14:
                        sop = sop[:13] + '...'
                    draw_lut_table(ax, w, swatch=_lut_color(w),
                                   title='%04X x%d\n%s' % (w, cnt, sop))
                else:
                    ax.set_visible(False)
        if len(chroms) > len(shown):
            fig.text(0.5, 0.005,
                     '...  %d more chromosomes: full detail in the genome .txt export'
                     % (len(chroms) - len(shown)),
                     ha='center', fontsize=8, color='#a00')
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    # -- save PNGs -------------------------------------------------------------

    def _save_pngs(self):
        if self.best_genome is None:
            self._status.set('No genome loaded: nothing to save.')
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        safe = ''.join(c if c.isalnum() else '_' for c in self._disp_target.name)
        ts   = time.strftime('%Y%m%d_%H%M%S')
        growth_png = os.path.join(RESULTS_DIR, 'growth_%s_%s.png'  % (safe, ts))
        volt_png   = os.path.join(RESULTS_DIR, 'voltage_%s_%s.png' % (safe, ts))
        genome_png = os.path.join(RESULTS_DIR, 'genome_%s_%s.png'  % (safe, ts))
        genome_txt = os.path.join(RESULTS_DIR, 'genome_%s_%s.txt'  % (safe, ts))
        self._growth_fig.savefig(growth_png, dpi=140, bbox_inches='tight', facecolor='white')
        self._volt_fig.savefig(volt_png,     dpi=130, bbox_inches='tight', facecolor='white')
        self._genome_fig.savefig(genome_png, dpi=130, bbox_inches='tight', facecolor='white')
        with open(genome_txt, 'w', encoding='utf-8') as f:
            seed = getattr(self, '_active_seed', None)
            if seed is not None:
                f.write('seed = %d\n\n' % seed)
            f.write(build_genome_text(self.best_genome, self.best_fitness,
                                      binding=self._io_binding_text(self.best_genome)))
        self._status.set('Saved growth / voltage / genome (png + txt) (%s) -> results/' % ts)


# -- entry point ---------------------------------------------------------------

def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
