#!/usr/bin/env python3
"""
app.py — Evolvable Hardware: single-window GUI.

Self-contained merge of the former main.py (evolve + truth table),
plot_growth.py (circuit-growth snapshots) and plot_ha.py (membrane-voltage
traces), now generalised to any registered target function.

Pick a target from the dropdown (logic gates, half/full adder, multi-bit adder)
or build your own truth table with "Custom…". The GA evolves a grown SNN to
match it, across three tabs:

    Evolution        live fitness chart + truth table for the chosen target
    Circuit Growth   growth snapshots (seed + each iteration)
    Voltage Traces   membrane voltages per output, per input case

The GA runs in a background thread; progress is polled on the Tk main loop.
On completion the best genome (and its target) are saved to
results/best_genome.pkl; the two plot tabs can be exported to PNG.

Usage:
    python app.py
"""
import sys, os, math, time, random, copy, pickle, threading, queue, dataclasses
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
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

from snn_evo import (grow_snn, grow_snn_snapshots, interpret_grid, simulate,
                     circuit_summary, score, random_genome,
                     TARGETS, DEFAULT_TARGET, get_target, truth_table_target,
                     Arch, DEFAULT_ARCH)
from snn_evo.genome import GRID_SIZE, MAX_STATE
from snn_evo.ga import _eval_batch, next_population
from snn_evo.lif_sim import DT, SIM_TIME, N_STEPS, REFRAC_STEPS, EPSC_STEPS
from nv_evo import (nervous_truth_table, grow_nervous_snapshots, interpret_nervous,
                    nervous_case_outputs, circuit_summary_nervous, ROUTING,
                    temporal_report)
from nv_evo import TEMPORAL_TARGETS
from nv_evo.viz import draw_hex_net
from lut_evo.viz import draw_lut_net, draw_lut_table
from interactive import InteractiveTab
import ui_compat

RESULTS_DIR = os.path.join(ROOT, 'results')
CKPT        = os.path.join(RESULTS_DIR, 'best_genome.pkl')
os.makedirs(RESULTS_DIR, exist_ok=True)

V_REST          = 0.0
R_M             = 1.0
V_RESET         = 0.0
MAX_VOLT_CASES  = 8     # voltage tab caps rows for readability on big targets


# ── target-aware analysis helpers ─────────────────────────────────────────────

def grow_for(genome, target):
    return grow_snn(genome, seeds=tuple(target.inputs),
                    grid_size=target.grid_size, iters=target.iters)


def interpret_for(genome, target, arch):
    grid = grow_for(genome, target)
    neurons, synapses = interpret_grid(grid, target=target, arch=arch)
    return grid, neurons, synapses


def _case_currents(target, in_bits, in_ids, complement):
    currents = {}
    for bit, iid in zip(in_bits, in_ids):
        base = target.high if bit else 0.0
        currents[iid] = (target.high - base) if complement else base
    return currents


def simulate_vmem(neurons, synapses, input_currents, track_ids):
    """Like lif_sim.simulate() but also records per-step membrane voltage."""
    n   = len(neurons)
    TAU = np.array([nu.tau for nu in neurons], dtype=np.float32)
    VTH = np.array([nu.vth for nu in neurons], dtype=np.float32)
    W   = np.zeros((n, n), dtype=np.float32)
    for s in synapses:
        if 0 <= s.pre < n and 0 <= s.post < n:
            W[s.post, s.pre] += s.weight
    I_ext = np.zeros(n, dtype=np.float32)
    for nid, curr in input_currents.items():
        if 0 <= nid < n:
            I_ext[nid] = float(curr)
    has_post   = (W != 0)
    V          = np.zeros(n, dtype=np.float32)
    refractory = np.zeros(n, dtype=np.int32)
    epsc       = np.zeros((n, n), dtype=np.int32)
    spikes     = {i: [] for i in range(n)}
    vmem       = {i: np.zeros(N_STEPS, dtype=np.float32) for i in track_ids}
    for step in range(N_STEPS):
        for tid in track_ids:
            vmem[tid][step] = V[tid]
        I_syn  = (W * (epsc > 0)).sum(axis=1)
        active = refractory == 0
        dV     = (DT / TAU) * (-(V - V_REST) + R_M * (I_syn + I_ext))
        V      = np.where(active, V + dV, V_RESET)
        refractory = np.maximum(refractory - 1, 0)
        fired  = active & (V >= VTH)
        for i in np.where(fired)[0]:
            spikes[i].append(step * DT)
            if i in vmem:
                vmem[i][step] = float(VTH[i]) + 0.1
            V[i]          = V_RESET
            refractory[i] = REFRAC_STEPS
            epsc[has_post[:, i], i] = EPSC_STEPS
        epsc = np.maximum(epsc - 1, 0)
    return spikes, vmem


def build_truth_table(genome, target, arch):
    """Return the truth-table report for `target` as a string."""
    _, ns, ss = interpret_for(genome, target, arch)
    in_ids = []
    for pos in target.inputs:
        nrn = next((n for n in ns if (n.x, n.y) == pos and n.is_input), None)
        in_ids.append(nrn.id if nrn else None)
    out_ids = []
    for term in target.outputs:
        nrn = next((n for n in ns if n.is_output and n.out_role == term.role), None)
        out_ids.append(nrn.id if nrn else None)

    lines = ['Target: ' + target.name,
             'Circuit: ' + circuit_summary(ns, ss)]
    for term, oid in zip(target.outputs, out_ids):
        if oid is not None:
            o = ns[oid]
            enc = []
            if term.complement_inputs: enc.append('complement-in')
            if term.invert_spike:      enc.append('invert')
            tag = ('  [%s]' % ', '.join(enc)) if enc else ''
            lines.append("  out '%s': pos=(%d,%d) vth=%.1f excit=%s%s" %
                         (term.role, o.x, o.y, o.vth, o.excit, tag))
        else:
            lines.append("  out '%s': (not found)" % term.role)

    if any(i is None for i in in_ids) or any(o is None for o in out_ids):
        lines.append('')
        lines.append('(circuit incomplete — some input/output neurons missing)')
        return '\n'.join(lines)

    in_hdr  = ' '.join('i%d' % i for i in range(len(target.inputs)))
    out_hdr = ' '.join('%s:e/a' % t.role for t in target.outputs)
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
            fired = len(sp.get(out_ids[i], [])) >= 1
            act   = (0 if fired else 1) if term.invert_spike else (1 if fired else 0)
            ok    = act == out_bits[i]
            row_ok = row_ok and ok
            total += 1
            correct += 1 if ok else 0
            cells.append('%d/%d' % (out_bits[i], act))
        in_str  = ' '.join(str(b) for b in in_bits).ljust(len(in_hdr))
        out_str = ' '.join(c.ljust(len('%s:e/a' % t.role))
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
            img[row, col] = [0.9, 0.1, 0.1, 1.0]            # red — seed
        elif (x, y) in output_pos:
            img[row, col] = [0.1, 0.8, 0.2, 1.0]            # green — output
        else:
            excit = not bool((state >> 3) & 0x1)
            alpha = 0.35 + 0.55 * (state & 0x7) / 7.0
            if excit:
                img[row, col] = [0.15, 0.35, 0.85, alpha]   # blue — excitatory
            else:
                img[row, col] = [0.90, 0.50, 0.10, alpha]   # orange — inhibitory
    return img


def grid_to_rgba_nervous(grid, grid_size, seed_set, output_pos):
    """RGBA image of a nervous net: nodes coloured by routing type."""
    img = np.ones((grid_size, grid_size, 4), dtype=float)
    for (x, y), state in grid.items():
        row, col = grid_size - 1 - y, x
        if (x, y) in seed_set:
            img[row, col] = [0.9, 0.1, 0.1, 1.0]             # red — input seed
        elif (x, y) in output_pos:
            img[row, col] = [0.1, 0.8, 0.2, 1.0]             # green — output
        else:
            e1, e2, i1 = ROUTING[state & 0xF]
            if i1 is not None:
                img[row, col] = [0.90, 0.50, 0.10, 0.95]     # orange — inhibited / veto
            elif e1 == e2:
                img[row, col] = [0.55, 0.75, 0.95, 0.9]      # light blue — buffer
            else:
                img[row, col] = [0.15, 0.35, 0.85, 0.95]     # blue — coincidence (AND)
    return img


def build_genome_text(genome, fitness=None):
    """Render the genome as a readable, aligned chromosome/gene table."""
    chroms  = list(getattr(genome, 'chromosomes', []) or [])
    n_total = sum(len(c.genes) for c in chroms)
    L = []
    head = 'Genome  -  %d chromosome%s,  %d genes' % (
        len(chroms), '' if len(chroms) == 1 else 's', n_total)
    if fitness is not None:
        head += '   (fitness = %.4f)' % fitness
    L.append(head)
    L.append('')
    g0 = chroms[0].genes[0] if (chroms and chroms[0].genes) else None
    hexmode = g0 is not None and hasattr(g0, 'ctx_l')
    lutmode = g0 is not None and hasattr(g0, 'ctx_n')
    sides = ('hex sides L/R/D' if hexmode
             else 'the 4 sides N/E/S/W (each a 16-bit LUT)' if lutmode
             else 'the 4 sides N/E/S/W')
    L.append('Each gene maps an expected neighbourhood (%s + the cell' % sides)
    L.append("itself) to an output state. During growth a cell adopts the output of the")
    L.append('gene whose pattern is closest (min Hamming distance) to its real neighbours.')
    if hexmode:
        L.append("out = 0 means the cell is off (dead); L/R are read in the node's")
        L.append("own orientation (context rotation).")
    elif lutmode:
        L.append('Each 16-bit LUT is really a BOOLEAN FUNCTION of the four bits the cell')
        L.append('receives from its neighbours — shown here minimised as out = f(N,S,E,W)')
        L.append("(¬ = not, · = and, + = or).  'out' is the LUT this gene installs; a")
        L.append("growth gene is one whose own context is empty (self = dead), the only")
        L.append('kind that can bring a dead cell to life.  out = 0 means that direction')
        L.append('stays dead.  Raw hex is kept in brackets for reference.')
    else:
        L.append("'until' = the gene only applies while the growth iteration <= that limit.")
    L.append('split = the crossover point used when two genomes mate.')
    L.append('')
    if not chroms:
        L.append('(empty genome)')
        return '\n'.join(L)

    if hexmode:
        hdr = '    #  |   L    R    D  | self |  out'
        sep = '  -----+----------------+------+------'
    elif lutmode:
        hdr = sep = None                          # block layout instead of a row
    else:
        hdr = '    #  |   N    E    S    W  | self |  out | until'
        sep = '  -----+---------------------+------+------+-------'
    for ci, c in enumerate(chroms):
        tel = getattr(c, 'telomere', None)
        L.append('Chromosome %s     tag=%-4d  split=%-2d%s   (%d genes)'
                 % (chr(ord('a') + ci), c.tag, c.split,
                    ('  telomere=%d' % tel) if tel is not None else '',
                    len(c.genes)))
        if hdr:
            L.append(hdr)
            L.append(sep)
        for gi, g in enumerate(c.genes):
            if 0 < c.split == gi:
                L.append('       - - - - - - split - - - - - -')
            if hexmode:
                L.append('  %4d | %4d %4d %4d | %4d | %4d'
                         % (gi, g.ctx_l, g.ctx_r, g.ctx_d, g.self_in, g.self_out))
            elif lutmode:
                L.extend(_lut_gene_lines(gi, g))
            else:
                L.append('  %4d | %4d %4d %4d %4d | %4d | %4d | %5d'
                         % (gi, g.state_n, g.state_e, g.state_s, g.state_w,
                            g.self_in, g.self_out, g.limit))
        L.append('')
    return '\n'.join(L)


def _lut_gene_lines(gi, g):
    """Readable two-line view of one LUT gene: the boolean function the gene
    installs (out, decoded from its 16-bit table to logic over the neighbour
    inputs N/S/E/W) plus the raw hex of every table for reference. The tables
    themselves are drawn as truth grids on the Growth tab. See lut_evo/boolfn.py."""
    from lut_evo.boolfn import lut_sop, popcount
    kind = 'growth' if g.self_in == 0 else 'maint '
    return [
        '  %3d %s  out = %s' % (gi, kind, lut_sop(g.self_out)),
        '            hex  out=%04X (%d/16 on)   ctx N/E/S/W=%04X %04X %04X %04X   self=%04X'
        % (g.self_out, popcount(g.self_out),
           g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in),
    ]


# ── GA background worker ──────────────────────────────────────────────────────

def _ga_worker(gens, pop, n_chroms, tries, target, arch, q, stop_event,
               base_seed=None, backend='snn'):
    """
    Run the GA with random restarts against `target` / `arch` / `backend`. Posts to queue q:
        ('gen',  try, gen, best_fit, mean_fit)
        ('best', genome, fitness)
        ('done', genome, fitness)

    base_seed=None  -> reseed from system entropy each restart (different runs).
    base_seed=int   -> reproducible: restart i uses seed base_seed + i.
    """
    # each backend brings its own genome, operators and evaluator (snn_evo/ga.py
    # vs nv_evo/ga.py — the nervous GA is tuned for loops/memory); this loop
    # only orchestrates restarts and reports progress.
    # `step(pop, fits, cases)` advances one generation. Convergence is NOT forced:
    # every backend runs a steady exploratory GA (elitism + tournament/lexicase +
    # immigrants), so the population contracts onto a genome only as far as
    # selection naturally drives it. The SNN GA takes no per-case vectors.
    diversify_fn = None      # set for nervous/lut: build a generation of solvers
    # Two dynamic mutation-rate controls compose each generation:
    #   base_mm * alpha**gen           — simulated-annealing cooldown (MUT_DECAY)
    #                     * stress_fn() — SOS reheat while a run is stalled
    # alpha defaults to 1.0 (no anneal); the SNN backend holds both at their
    # neutral values (stress_fn ≡ 1, no rate threading).
    stress_fn, base_mm, alpha = (lambda s: 1.0), 1.0, 1.0
    _pool = None             # persistent worker pool (nervous/lut) reused across gens
    if backend == 'nervous':
        from nv_evo import random_hex_genome
        from nv_evo.ga import (eval_batch_cases, next_population as next_pop_nv,
                               diversify as _div, stress_multiplier,
                               MEAN_MUTATIONS, MUT_DECAY, N_WORKERS)
        cache       = {}     # fitness cache shared across restarts (same target)
        _pool       = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse (no per-gen respawn)
        make_genome = lambda: random_hex_genome(n_chroms)
        eval_batch  = lambda genomes: eval_batch_cases(genomes, target, cache, _pool)
        step        = lambda p, fits, cv, mm: next_pop_nv(p, fits, make_genome, cv, mm)
        stress_fn, base_mm, alpha = stress_multiplier, MEAN_MUTATIONS, MUT_DECAY
        diversify_fn = lambda seeds, valid: _div(seeds, target, pop, valid=valid, cache=cache)
    elif backend == 'lut':
        from lut_evo.ga import (eval_batch_cases, next_population as next_pop_lut,
                                diversify as _div, make_seed_genome,
                                MEAN_MUTATIONS, MUT_DECAY, N_WORKERS)
        cache       = {}
        _pool       = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse (no per-gen respawn)
        make_genome = lambda: make_seed_genome(n_chroms)   # honours SEED_MODE toggle
        eval_batch  = lambda genomes: eval_batch_cases(genomes, target, cache, _pool)
        step        = lambda p, fits, cv, mm: next_pop_lut(p, fits, make_genome, cv, mm)
        base_mm, alpha = MEAN_MUTATIONS, MUT_DECAY
        diversify_fn = lambda seeds, valid: _div(seeds, target, pop, valid=valid, cache=cache)
    else:
        make_genome = lambda: random_genome(n_chroms)
        eval_batch  = lambda genomes: (_eval_batch(genomes, target, arch), None)
        step        = lambda p, fits, cv, mm: next_population(p, fits)

    def size(genome):
        return sum(len(c.genes) for c in genome.chromosomes)

    best_fit = 0.0
    best_g   = None
    try:                                     # ensure the worker pool is released
        for try_i in range(tries):
            if stop_event.is_set():
                break
            if base_seed is None:
                random.seed()                # entropy — different every run
            else:
                random.seed(base_seed + try_i)
            population = [make_genome() for _ in range(pop)]
            fitnesses, cases = eval_batch(population)
            bi         = max(range(pop),
                             key=lambda i: (fitnesses[i], -size(population[i])))
            g, f       = copy.deepcopy(population[bi]), fitnesses[bi]
            if f > best_fit:
                best_fit, best_g = f, copy.deepcopy(g)
                q.put(('best', copy.deepcopy(best_g), best_fit))
            # generation-0 point so the chart shows something even when the very
            # first population already contains a solution
            q.put(('gen', try_i + 1, 0, f, sum(fitnesses) / len(fitnesses)))
            # no early stop at fitness 1.0: the run continues so the parsimony
            # tie-break keeps shrinking a solved genome. Convergence is not forced —
            # the population contracts only as far as selection naturally drives it.
            stagnation = 0
            mut_rate   = base_mm                          # annealing schedule
            for gen in range(gens):
                if stop_event.is_set():
                    break
                mut_rate *= alpha                         # anneal: cool down
                mm = mut_rate * stress_fn(stagnation)     # SOS reheats plateaus
                population = step(population, fitnesses, cases, mm)
                fitnesses, cases = eval_batch(population)
                gi         = max(range(pop),
                                 key=lambda i: (fitnesses[i], -size(population[i])))
                # SOS tracks FITNESS plateaus only — a parsimony-only win updates the
                # champion but must not reset the stress counter (see nv_evo.ga).
                if fitnesses[gi] > f + 1e-12:
                    stagnation = 0
                else:
                    stagnation += 1
                if (fitnesses[gi] > f
                        or (fitnesses[gi] == f and size(population[gi]) < size(g))):
                    f = fitnesses[gi]
                    g = copy.deepcopy(population[gi])
                    if (f > best_fit
                            or (f == best_fit and best_g is not None
                                and size(g) < size(best_g))):
                        best_fit, best_g = f, copy.deepcopy(g)
                        q.put(('best', copy.deepcopy(best_g), best_fit))
                mean_f = sum(fitnesses) / len(fitnesses)
                q.put(('gen', try_i + 1, gen + 1, f, mean_f))
            if best_fit >= 1.0:
                break                        # solved after a full run: no restart
    finally:
        if _pool is not None:
            _pool.shutdown()                 # released even if evaluation errors

    # Diversification: rather than a monoculture of the champion, spread across
    # the neutral network to build a whole generation of DISTINCT valid solvers
    # (each genotypically unique, each still solving). The count is honest — it
    # returns as many unique solvers as the target's neutral network admits.
    n_unique = 0
    if diversify_fn is not None and best_g is not None and best_fit > 0 and not stop_event.is_set():
        valid = 1.0 if best_fit >= 0.999 else max(0.0, best_fit - 1e-6)
        try:
            seeds = [gg for gg, ff in zip(population, fitnesses) if ff >= valid] or [best_g]
            pool  = diversify_fn(seeds, valid)
            n_unique = len(pool)
            if pool:
                with open(os.path.join(RESULTS_DIR, 'solver_generation.pkl'), 'wb') as fp:
                    pickle.dump({'genomes': pool, 'target': target,
                                 'backend': backend, 'valid': valid}, fp)
                q.put(('diverse', n_unique, valid))
        except Exception:
            pass

    q.put(('done', copy.deepcopy(best_g) if best_g else None, best_fit))


# ── custom truth-table dialog ─────────────────────────────────────────────────

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

    def _build(self):
        for w in self._table_frame.winfo_children():
            w.destroy()
        self._rows = []
        try:
            n_in  = max(1, min(6, int(self._nin.get())))
            n_out = max(1, min(6, int(self._nout.get())))
        except (tk.TclError, ValueError):
            return
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


# ── main application ──────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        root.title('Evolvable Hardware — Edwards Indirect Encoding')
        root.minsize(1040, 720)
        self.q            = queue.Queue()
        self._worker      = None
        self._stop_event  = threading.Event()
        self.best_genome  = None
        self.best_fitness = 0.0
        self._gen_history = []
        self._abs_gen     = 0
        self._custom      = {}     # name -> Target for user-built targets
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
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 1040)
        h = max(root.winfo_reqheight(), 720)
        root.geometry('%dx%d' % (w, h))
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=(6, 4))
        ctrl.pack(fill='x', side='top')

        ttk.Label(ctrl, text='Model:').pack(side='left', padx=(2, 2))
        self._backend_var = tk.StringVar(value='SNN')
        self._backend_cb  = ttk.Combobox(ctrl, textvariable=self._backend_var,
                                         values=['SNN', 'Nervous', 'LUT'],
                                         width=8, state='readonly')
        self._backend_cb.pack(side='left')
        self._backend_cb.bind('<<ComboboxSelected>>', self._on_backend_change)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)

        ttk.Label(ctrl, text='Target:').pack(side='left', padx=(2, 2))
        self._target_var = tk.StringVar(value=DEFAULT_TARGET)
        self._target_cb  = ttk.Combobox(ctrl, textvariable=self._target_var,
                                        values=list(self._all_targets()), width=16, state='readonly')
        self._target_cb.pack(side='left')
        self._target_cb.bind('<<ComboboxSelected>>', self._on_target_change)
        ttk.Button(ctrl, text='Custom…', command=self._open_custom).pack(side='left', padx=3)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)

        def lentry(label, default, width=6):
            ttk.Label(ctrl, text=label).pack(side='left', padx=(6, 2))
            v = tk.StringVar(value=str(default))
            ttk.Entry(ctrl, textvariable=v, width=width).pack(side='left')
            return v

        self._pop_var   = lentry('Pop:',   50)
        # long single run instead of many short restarts, so slow steady progress
        # is visible (user-preferred; was Gens=30, Tries=20 which restarted every
        # 30 generations before any trend showed).
        self._gens_var  = lentry('Gens:',  500)
        self._tries_var = lentry('Tries:', 1)
        self._seed_var  = lentry('Seed:',  'random', width=7)

        self._graded_var = tk.BooleanVar(value=False)
        self._graded_chk = ttk.Checkbutton(ctrl, text='Graded', variable=self._graded_var)
        self._graded_chk.pack(side='left', padx=(8, 0))

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)

        self._run_btn  = ttk.Button(ctrl, text='Run',        command=self._start_ga)
        self._stop_btn = ttk.Button(ctrl, text='Stop',       command=self._stop_ga, state='disabled')
        self._load_btn = ttk.Button(ctrl, text='Load Saved', command=self._load_saved)
        self._save_btn = ttk.Button(ctrl, text='Save PNGs',  command=self._save_pngs, state='disabled')
        for b in (self._run_btn, self._stop_btn, self._load_btn, self._save_btn):
            b.pack(side='left', padx=3)

        # ── second row: model-specific parameters ──
        ctrl2 = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ctrl2.pack(fill='x', side='top')

        def aentry(parent, label, default, width=5):
            ttk.Label(parent, text=label).pack(side='left', padx=(6, 2))
            v = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=v, width=width).pack(side='left')
            return v

        # substrate group (SNN only — hidden for nervous nets)
        self._arch_frame = ttk.Frame(ctrl2)
        self._arch_frame.pack(side='left')
        ttk.Label(self._arch_frame, text='Substrate:').pack(side='left', padx=(2, 0))
        self._syn_var  = aentry(self._arch_frame, 'Syn weight:', DEFAULT_ARCH.syn_weight)
        self._vmin_var = aentry(self._arch_frame, 'Vth min:',    DEFAULT_ARCH.vth_levels[0])
        self._vmax_var = aentry(self._arch_frame, 'Vth max:',    DEFAULT_ARCH.vth_levels[-1])
        self._cur_var  = aentry(self._arch_frame, 'Input I:',    self.target.high)
        self._arch_sep = ttk.Separator(ctrl2, orient='vertical')
        self._arch_sep.pack(side='left', fill='y', padx=8)

        # genome group. Grid / Iters were removed: growth is now self-limiting —
        # the nervous telomere is a Hayflick bound, so neither a grid clip nor an
        # iteration cap governs it (grid_size survives only as each target's I/O
        # layout scale, taken from the target itself). Only the genome's
        # chromosome count remains tunable here.
        self._layout_frame = ttk.Frame(ctrl2)
        self._layout_frame.pack(side='left')
        ttk.Label(self._layout_frame, text='Genome:').pack(side='left', padx=(2, 0))
        self._chroms_var = aentry(self._layout_frame, 'Chroms:', 2, width=4)
        # LUT only: seed the population with dense sim6-style ontogeny genomes
        # (rich morphology) and drop parsimony so they aren't pruned back to
        # sparse. Much slower, and measured to solve WORSE than random — an
        # exploratory shape mode, off by default. Shown only for the LUT backend.
        self._ontogeny_var = tk.BooleanVar(value=False)
        self._ontogeny_chk = ttk.Checkbutton(self._layout_frame, text='Ontogeny seed',
                                              variable=self._ontogeny_var)
        self._layout_reset_btn = ttk.Button(self._layout_frame, text='Reset', width=6,
                                             command=self._reset_arch)
        self._layout_reset_btn.pack(side='left', padx=8)

        self._model_note = ttk.Label(ctrl2, text='', foreground='#888888')
        self._model_note.pack(side='left', padx=6)

        # ── third row: GA + substrate-physics tuning (applied on Run) ──
        from nv_evo.ga import (MEAN_MUTATIONS as _MM, IMMIGRANT_FRAC as _IM,
                               TOURNAMENT_K as _TK, MUT_DECAY as _AL)
        from nv_evo.pulse import DELAY as _D, WIDTH as _W, COINC as _C
        self._tune_defaults = dict(mut=_MM, imm=_IM, tk=_TK, alpha=_AL,
                                   delay=_D, width=_W, coinc=_C)
        ctrl3 = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        ctrl3.pack(fill='x', side='top')

        ga_frame = ttk.Frame(ctrl3); ga_frame.pack(side='left')
        ttk.Label(ga_frame, text='GA:').pack(side='left', padx=(2, 0))
        self._mut_var  = aentry(ga_frame, 'Mutations/child:', _MM, width=4)
        self._imm_var  = aentry(ga_frame, 'Immigrants:',      _IM, width=4)
        self._tourn_var = aentry(ga_frame, 'Tournament:',     _TK, width=3)
        # simulated-annealing decay: mutation rate *= α each generation (1 = off)
        self._alpha_var = aentry(ga_frame, 'Anneal α:',       _AL, width=6)
        ttk.Separator(ctrl3, orient='vertical').pack(side='left', fill='y', padx=8)

        # pulse-physics group (nervous net only)
        self._pulse_frame = ttk.Frame(ctrl3)
        self._pulse_frame.pack(side='left')
        ttk.Label(self._pulse_frame, text='Pulse:').pack(side='left', padx=(2, 0))
        self._delay_var = aentry(self._pulse_frame, 'Delay:', _D, width=4)
        self._width_var = aentry(self._pulse_frame, 'Width:', _W, width=4)
        self._coinc_var = aentry(self._pulse_frame, 'Coinc:', _C, width=4)
        self._pulse_sep = ttk.Separator(ctrl3, orient='vertical')
        self._pulse_sep.pack(side='left', fill='y', padx=8)
        self._tune_reset_btn = ttk.Button(ctrl3, text='Reset tuning', width=12,
                                           command=self._reset_tuning)
        self._tune_reset_btn.pack(side='left', padx=4)

        # Fix a natural size on the notebook and stop it from resizing to fit
        # whichever tab is shown. On X11 a matplotlib canvas in a freshly-mapped
        # tab requests its figsize*dpi in pixels, and without this the whole
        # window would jump size every time you switched tabs. width/height give
        # the notebook a fixed request; the tab frames fill it via expand.
        self._nb = nb = ttk.Notebook(self.root, width=1120, height=680)
        nb.pack(fill='both', expand=True, padx=4, pady=(0, 2))
        self._build_evolve_tab(nb)
        self._build_growth_tab(nb)
        self._build_voltage_tab(nb)
        self._build_genome_tab(nb)
        self._interactive = InteractiveTab(self._add_tab(nb, 'Interactive'),
                                           self.current_circuit)

        self._status = tk.StringVar(
            value='Ready — pick a model and target, set parameters, click Run (or Load Saved).')
        ttk.Label(self.root, textvariable=self._status, anchor='w',
                  relief='sunken', padding=(6, 2)).pack(
            fill='x', side='bottom', padx=4, pady=(0, 4))

        self._reconfigure_for_backend()      # set initial model-specific layout

    def _reconfigure_for_backend(self):
        """Show only the controls / tab labels relevant to the selected model."""
        backend = self._backend()
        if backend != 'snn':
            self._arch_frame.pack_forget()
            self._arch_sep.pack_forget()
            self._graded_chk.state(['disabled'])
            if backend == 'nervous':
                note = ('Nervous net — HEX array (each cell wired to 3 neighbours L/R/D); '
                        'coincidence (AND) + inhibition, loops circulate injected pulses '
                        '(memory). Substrate (Vth/Syn/Input) and Graded do not apply.')
            else:
                note = ('LUT array — SQUARE array (each cell wired to 4 neighbours N/S/E/W), '
                        '4 directional 16-bit lookup tables per cell, latched & synchronous '
                        '(paper Architecture 2 / sim6). Recurrent & dynamical — TEMPORAL '
                        'targets only (it cannot settle to combinational logic). '
                        'Substrate/Graded do not apply.')
            self._model_note.config(text=note)
            self._set_tab_label(self._volt_tab, 'Activity')
        else:
            self._arch_frame.pack(side='left', before=self._layout_frame)
            self._arch_sep.pack(side='left', fill='y', padx=8, before=self._layout_frame)
            self._graded_chk.state(['!disabled'])
            self._model_note.config(text='SNN — leaky integrate-and-fire neurons.')
            self._set_tab_label(self._volt_tab, 'Voltage Traces')
        # pulse-physics knobs apply only to the nervous net's pulse engine.
        # Re-pack relative to the always-present Reset button (the separator is
        # itself hidden for other models, so it can't be a pack anchor).
        if hasattr(self, '_pulse_frame'):
            if backend == 'nervous':
                self._pulse_frame.pack(side='left', before=self._tune_reset_btn)
                self._pulse_sep.pack(side='left', fill='y', padx=8,
                                     before=self._tune_reset_btn)
            else:
                self._pulse_frame.pack_forget()
                self._pulse_sep.pack_forget()
        # ontogeny-seed toggle applies only to the LUT backend
        if hasattr(self, '_ontogeny_chk'):
            if backend == 'lut':
                self._ontogeny_chk.pack(side='left', padx=(4, 0),
                                        before=self._layout_reset_btn)
            else:
                self._ontogeny_chk.pack_forget()
        # dropdown offers only backend-valid targets (LUT hides combinational)
        if hasattr(self, '_target_cb'):
            self._refresh_target_list()

    def _set_tab_label(self, frame, text):
        try:
            self._nb.tab(frame, text=text)
        except Exception:
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
        self._best_line, = self._fit_ax.plot([], [], 'b-',  lw=1.5, label='Best')
        self._mean_line, = self._fit_ax.plot([], [], 'r--', lw=1.0, label='Mean', alpha=0.7)
        self._fit_ax.legend(fontsize=8)
        self._fit_fig.tight_layout()
        self._fit_canvas = FigureCanvasTkAgg(self._fit_fig, master=left)
        self._fit_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        right = ttk.LabelFrame(frame, text='Truth Table', padding=6)
        right.pack(side='right', fill='both', padx=(0, 6), pady=6)
        self._tt_text = scrolledtext.ScrolledText(
            right, width=58, height=22,
            font=(self._mono, 9), state='disabled', wrap='none')
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

    # ── target selection ──────────────────────────────────────────────────────

    def _all_targets(self):
        d = dict(TARGETS)
        d.update(TEMPORAL_TARGETS)
        d.update(self._custom)
        return d

    def _targets_for_backend(self, backend):
        """Targets offered in the dropdown for a backend. The LUT substrate is a
        chaotic recurrent CA that cannot settle to input-dependent combinational
        logic — it only works for TEMPORAL targets — so combinational targets are
        hidden when LUT is selected (see lut_evo settling note)."""
        if backend == 'lut':
            d = dict(TEMPORAL_TARGETS)
            d.update({k: v for k, v in self._custom.items()
                      if getattr(v, 'temporal', False)})
            return d
        return self._all_targets()

    def _refresh_target_list(self):
        """Repopulate the dropdown for the current backend; if the current
        selection is no longer valid there, switch to the first available."""
        names = list(self._targets_for_backend(self._backend()))
        self._target_cb['values'] = names
        if names and self._target_var.get() not in names:
            self._target_var.set(names[0])
            self._on_target_change()

    def _on_target_change(self, _evt=None):
        name = self._target_var.get()
        self.target = self._all_targets().get(name, get_target(DEFAULT_TARGET))
        self._cur_var.set(str(self.target.high))
        if getattr(self.target, 'temporal', False):
            if self._backend() == 'snn':          # temporal needs nervous or LUT
                self._backend_var.set('Nervous')
            self._reconfigure_for_backend()
            # show what this target IS right away in the Evolution tab's panel
            try:
                self._set_tt(temporal_report(self.target))
            except Exception:
                pass
            self._status.set('Target: %s — temporal nervous net (loops / memory); '
                             '%d in, %d out, %d ticks. See the Evolution tab for '
                             'what it must do; test it in the Interactive tab.'
                             % (self.target.name, self.target.n_inputs,
                                self.target.n_outputs, self.target.T))
            return
        n_cases = len(self.target.cases)
        self._status.set('Target: %s — %d inputs, %d outputs, %d cases%s' % (
            self.target.name, self.target.n_inputs, self.target.n_outputs, n_cases,
            '   (large — evolution will be slow)' if n_cases > 32 else ''))

    def _effective_target(self, high, graded):
        """Apply the GUI's high/graded knobs to the selected target. Growth is no
        longer grid/iters-bounded (the nervous telomere is self-limiting), so the
        target keeps its own I/O layout — grid_size/iters are left untouched."""
        if getattr(self.target, 'temporal', False):
            # temporal targets have fixed close I/O and no high/graded fields
            return dataclasses.replace(self.target)
        return dataclasses.replace(self.target, high=high, graded=graded)

    def _reset_tuning(self):
        d = self._tune_defaults
        self._mut_var.set(str(d['mut']));   self._imm_var.set(str(d['imm']))
        self._tourn_var.set(str(d['tk']));  self._alpha_var.set(str(d['alpha']))
        self._delay_var.set(str(d['delay']))
        self._width_var.set(str(d['width'])); self._coinc_var.set(str(d['coinc']))

    def _apply_tuning(self):
        """Push the GUI's GA / pulse-physics knobs into the backend modules and
        os.environ (so the fresh ProcessPool workers that evaluate genomes pick
        the physics up at import). Returns True on success."""
        try:
            mut = float(self._mut_var.get()); imm = float(self._imm_var.get())
            tk_ = int(self._tourn_var.get()); alpha = float(self._alpha_var.get())
            delay = float(self._delay_var.get()); width = float(self._width_var.get())
            coinc = float(self._coinc_var.get())
            if (mut < 0 or not (0 <= imm < 1) or tk_ < 1 or not (0 < alpha <= 1)
                    or delay <= 0 or width <= 0 or coinc < 0):
                raise ValueError
        except ValueError:
            return False
        import nv_evo.ga as nga, lut_evo.ga as lga, nv_evo.pulse as npulse
        for mod in (nga, lga):
            mod.MEAN_MUTATIONS = mut
            mod.IMMIGRANT_FRAC = imm
            mod.TOURNAMENT_K   = tk_
            mod.MUT_DECAY      = alpha
        # LUT ontogeny-seed toggle: dense biomorph seeds + no parsimony pruning
        onto = bool(self._ontogeny_var.get())
        lga.SEED_MODE = 'ontogeny' if onto else 'random'
        lga.PARSIMONY = not onto
        npulse.DELAY, npulse.WIDTH, npulse.COINC = delay, width, coinc
        os.environ['NV_DELAY'] = str(delay)
        os.environ['NV_WIDTH'] = str(width)
        os.environ['NV_COINC'] = str(coinc)
        return True

    def _backend(self):
        v = self._backend_var.get().lower()
        if v.startswith('nerv'):
            return 'nervous'
        if v.startswith('lut'):
            return 'lut'
        return 'snn'

    def _on_backend_change(self, _evt=None):
        self._reconfigure_for_backend()
        backend = self._backend()
        if backend == 'nervous':
            self._status.set('Model: Nervous net — coincidence + inhibition + loops; '
                             'best with small grids and close I/O.')
        elif backend == 'lut':
            self._status.set('Model: LUT array — square grid, 4 neighbours, four 16-bit '
                             'lookup tables per cell, latched & synchronous (sim6 / Arch 2).')
        else:
            self._status.set('Model: SNN — leaky integrate-and-fire neurons.')

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
        return Arch(syn_weight=sw, vth_levels=levels,
                    tau_levels=DEFAULT_ARCH.tau_levels), high, True

    def _reset_arch(self):
        self._syn_var.set(str(DEFAULT_ARCH.syn_weight))
        self._vmin_var.set(str(DEFAULT_ARCH.vth_levels[0]))
        self._vmax_var.set(str(DEFAULT_ARCH.vth_levels[-1]))
        self._cur_var.set(str(self.target.high))
        self._chroms_var.set('2')
        self._ontogeny_var.set(False)

    def _open_custom(self):
        CustomTargetDialog(self.root, self._on_custom_built)

    def _on_custom_built(self, target):
        self._custom[target.name] = target
        self._target_cb['values'] = list(self._all_targets())
        self._target_var.set(target.name)
        self._on_target_change()

    # ── GA control ────────────────────────────────────────────────────────────

    def _start_ga(self):
        try:
            pop   = int(self._pop_var.get())
            gens  = int(self._gens_var.get())
            tries = int(self._tries_var.get())
            if pop < 2 or gens < 1 or tries < 1:
                raise ValueError
        except ValueError:
            self._status.set('Invalid parameters — Pop>=2, Gens>=1, Tries>=1 (integers).')
            return

        seed_txt = self._seed_var.get().strip().lower()
        if seed_txt in ('', 'random', 'rand', 'none'):
            # pick an explicit seed so the run is recorded and replayable
            base_seed = random.randrange(1, 2_000_000_000)
        else:
            try:
                base_seed = int(seed_txt)
            except ValueError:
                self._status.set("Invalid seed — use an integer or 'random'.")
                return
        self._active_seed = base_seed
        # show the actual seed in the field so it's visible and replayable
        # (type 'random' again to draw a fresh one)
        self._seed_var.set(str(base_seed))

        arch, high, ok = self._read_arch()
        if not ok:
            self._status.set('Invalid substrate — Syn weight>0, Input I>0, 0<=Vth min<=Vth max.')
            return

        try:
            n_chroms = int(self._chroms_var.get())
            if n_chroms < 1:
                raise ValueError
        except ValueError:
            self._status.set('Invalid genome — Chroms>=1 (integer).')
            return

        if not self._apply_tuning():
            self._status.set('Invalid tuning — Mutations>=0, 0<=Immigrants<1, '
                             'Tournament>=1, 0<α<=1, Delay/Width>0, Coinc>=0.')
            return

        backend = self._backend()
        eff_target = self._effective_target(high, self._graded_var.get())
        self._active_target  = eff_target
        self._active_arch    = arch
        self._active_chroms  = n_chroms
        self._active_backend = backend
        self._disp_target    = eff_target
        self._disp_arch      = arch
        self._disp_backend   = backend

        self._gen_history.clear()
        self._abs_gen = 0
        self._n_unique_solvers = 0
        self._best_line.set_data([], [])
        self._mean_line.set_data([], [])
        self._fit_ax.set_xlim(0, max(gens * tries, 10) + 1)
        self._fit_ax.set_title('Fitness vs Generation — %s' % self.target.name, fontsize=10)
        self._fit_canvas.draw_idle()
        self._set_tt('Evolving %s …\n' % self.target.name)

        self._run_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self._load_btn.config(state='disabled')
        self._save_btn.config(state='disabled')
        self._target_cb.config(state='disabled')
        self._backend_cb.config(state='disabled')

        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=_ga_worker,
            args=(gens, pop, n_chroms, tries, eff_target, arch, self.q, self._stop_event,
                  base_seed, backend),
            daemon=True)
        self._worker.start()
        self._status.set('Evolving %s [%s] …  pop=%d  gens=%d  tries=%d  seed=%d%s' %
                         (self.target.name, backend, pop, gens, tries, base_seed,
                          '  [graded]' if self._graded_var.get() else ''))

    def _stop_ga(self):
        self._stop_event.set()
        self._stop_btn.config(state='disabled')
        self._status.set('Stopping after current generation…')

    def _load_saved(self):
        if not os.path.exists(CKPT):
            self._status.set('No saved genome at %s — run the GA first.' % CKPT)
            return
        with open(CKPT, 'rb') as f:
            state = pickle.load(f)
        self.best_genome  = state['best_genome']
        self.best_fitness = state['best_fitness']
        saved_target = state.get('target')
        if saved_target is None:
            saved_target = get_target(state.get('target_name', DEFAULT_TARGET))
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
        saved_backend = state.get('backend', 'snn')
        self._disp_backend   = saved_backend
        self._active_backend = saved_backend
        self._backend_var.set({'nervous': 'Nervous', 'lut': 'LUT'}.get(saved_backend, 'SNN'))
        self._reconfigure_for_backend()
        if saved_target.name not in self._all_targets():
            self._custom[saved_target.name] = saved_target
            self._target_cb['values'] = list(self._all_targets())
        self._target_var.set(saved_target.name)
        self._status.set('Loaded %s  fitness=%.4f  syn_w=%.2f' %
                         (saved_target.name, self.best_fitness, saved_arch.syn_weight))
        self._update_all(self.best_genome, self.best_fitness)
        self._save_btn.config(state='normal')

    # ── queue polling ─────────────────────────────────────────────────────────

    def _redraw_fit_chart(self, max_pts=2000):
        """Push the best/mean history to the chart. The full history is kept (it's
        only three floats per generation) but the PLOTTED line is decimated to at
        most `max_pts` points, so the redraw cost stays flat whether the run is
        500 generations or 100,000 — the chart never becomes the bottleneck."""
        hist = self._gen_history
        if not hist:
            return
        if len(hist) > max_pts:
            step = len(hist) // max_pts + 1
            pts  = hist[::step]
            if pts[-1][0] != hist[-1][0]:      # always keep the latest point
                pts = pts + [hist[-1]]
        else:
            pts = hist
        self._best_line.set_data([d[0] for d in pts], [d[1] for d in pts])
        self._mean_line.set_data([d[0] for d in pts], [d[2] for d in pts])
        self._fit_ax.set_xlim(0, hist[-1][0] + 1)
        self._fit_canvas.draw_idle()

    def _poll(self):
        last_gen = None                        # newest 'gen' this poll — redraw ONCE
        try:
            while True:
                msg  = self.q.get_nowait()
                kind = msg[0]
                if kind == 'gen':
                    _, try_n, gen, best_f, mean_f = msg
                    self._abs_gen += 1
                    self._gen_history.append((self._abs_gen, best_f, mean_f))
                    last_gen = (try_n, gen, best_f, mean_f)
                elif kind == 'diverse':
                    _, n_unique, valid = msg
                    self._n_unique_solvers = n_unique
                    self._status.set('Diversifying — built %d genotypically UNIQUE '
                                     'solvers (each ≥ %.3f) → results/solver_generation.pkl'
                                     % (n_unique, valid))
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
                    _, genome, fit = msg
                    if genome is not None:
                        self.best_genome  = genome
                        self.best_fitness = fit
                        save_target = getattr(self, '_active_target', self.target)
                        save_arch   = getattr(self, '_active_arch', DEFAULT_ARCH)
                        with open(CKPT, 'wb') as f:
                            pickle.dump({'best_genome': genome, 'best_fitness': fit,
                                         'target': save_target,
                                         'target_name': save_target.name,
                                         'arch': save_arch,
                                         'seed': getattr(self, '_active_seed', None),
                                         'backend': getattr(self, '_active_backend', 'snn')}, f)
                        self._pending_best = None      # final draw supersedes it
                        self._update_all(genome, fit)
                    self._run_btn.config(state='normal')
                    self._stop_btn.config(state='disabled')
                    self._load_btn.config(state='normal')
                    self._target_cb.config(state='readonly')
                    self._backend_cb.config(state='readonly')
                    self._save_btn.config(state='normal' if genome else 'disabled')
                    saved = ('  saved → %s' % CKPT) if genome else ''
                    nu = getattr(self, '_n_unique_solvers', 0)
                    div = ('  |  %d unique valid solvers' % nu) if nu else ''
                    self._status.set('Done — %s fitness=%.4f  seed=%d (type it in Seed to replay)%s%s' %
                                     (self.target.name, fit,
                                      getattr(self, '_active_seed', 0), saved, div))
        except queue.Empty:
            pass
        if last_gen is not None:               # redraw the fitness chart once per poll
            self._redraw_fit_chart()
            try_n, gen, best_f, mean_f = last_gen
            self._status.set('%s  seed=%d  try=%d  gen=%d  best=%.4f  mean=%.4f' %
                             (self.target.name, getattr(self, '_active_seed', 0),
                              try_n, gen, best_f, mean_f))
        # throttled repaint of the display panels for the latest best (≤ ~1.6/s)
        pend = getattr(self, '_pending_best', None)
        if pend is not None:
            now = time.time()
            if now - getattr(self, '_last_full_draw', 0.0) > 0.6:
                self._pending_best = None
                self._last_full_draw = now
                self._update_all(*pend)
        self.root.after(60, self._poll)

    # ── display updates ───────────────────────────────────────────────────────

    def _seed_tag(self):
        s = getattr(self, '_active_seed', None)
        return ('   seed=%d' % s) if s is not None else ''

    def _update_all(self, genome, fitness):
        self._update_truth_table(genome)
        self._draw_growth(genome, fitness)
        self._draw_voltages(genome)
        self._draw_genome(genome, fitness)

    def _set_tt(self, text):
        self._tt_text.config(state='normal')
        self._tt_text.delete('1.0', 'end')
        self._tt_text.insert('end', text)
        self._tt_text.config(state='disabled')

    def _update_truth_table(self, genome):
        try:
            if getattr(self._disp_target, 'temporal', False):
                if self._disp_backend == 'lut':
                    from lut_evo import lut_report
                    text = lut_report(self._disp_target, genome)
                else:
                    text = temporal_report(self._disp_target, genome)
            elif self._disp_backend == 'nervous':
                text = nervous_truth_table(genome, self._disp_target)
            elif self._disp_backend == 'lut':
                from lut_evo import lut_truth_table
                text = lut_truth_table(genome, self._disp_target)
            else:
                text = build_truth_table(genome, self._disp_target, self._disp_arch)
        except Exception as exc:
            text = 'Error building truth table:\n' + str(exc)
        self._set_tt(text)

    def _draw_growth(self, genome, fitness):
        target = self._disp_target
        if self._disp_backend == 'nervous':
            self._draw_growth_hex(genome, target, fitness)
            return
        if self._disp_backend == 'lut':
            self._draw_growth_lut(genome, target, fitness)
            return
        try:
            grid, ns_list, _ = interpret_for(genome, target, self._disp_arch)
            output_pos = {(n.x, n.y): n.out_role for n in ns_list if n.is_output}
            snapshots  = grow_snn_snapshots(genome, seeds=tuple(target.inputs),
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
            '%s — Circuit Growth   (fitness=%.4f)%s' % (target.name, fitness, self._seed_tag()),
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
        if self._disp_backend == 'nervous':
            patches = [
                mpatches.Patch(color=(0.9, 0.1, 0.1),    label='Input seed'),
                mpatches.Patch(color=(0.1, 0.8, 0.2),    label='Output'),
                mpatches.Patch(color=(0.55, 0.75, 0.95), label='Buffer (relay)'),
                mpatches.Patch(color=(0.15, 0.35, 0.85), label='Coincidence (AND)'),
                mpatches.Patch(color=(0.90, 0.50, 0.10), label='Inhibited (veto)'),
                mpatches.Patch(color=(1.0, 1.0, 1.0),    label='Empty', edgecolor='#aaa'),
            ]
        else:
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

    def _draw_growth_hex(self, genome, target, fitness):
        """Growth tab for the hex nervous net: snapshots as honeycomb panels,
        the final one wired (excitatory green / inhibitory red)."""
        try:
            snaps = grow_nervous_snapshots(genome, seeds=tuple(target.inputs),
                                           grid_size=target.grid_size, iters=target.iters)
            routing, in_pos, out_pos = interpret_nervous(snaps[-1], target)
            if getattr(target, 'temporal', False):
                from nv_evo import place_outputs_by_trace
                out_pos, _ = place_outputs_by_trace(snaps[-1], routing, in_pos, target)
        except Exception:
            return
        fig = self._growth_fig
        fig.clf()
        fig.suptitle('%s — hex nervous-net growth   (fitness=%.4f)%s'
                     % (target.name, fitness, self._seed_tag()),
                     fontsize=10, fontweight='bold', y=0.995)
        gs    = target.grid_size
        n     = len(snaps)
        ncols = min(5, max(3, math.ceil(math.sqrt(n))))
        nrows = math.ceil(n / ncols)
        axes  = fig.subplots(nrows, ncols, squeeze=False)
        flat  = [a for row in axes for a in row]
        for idx, snap in enumerate(snaps):
            rt   = {p: ROUTING[s & 0xF] for p, s in snap.items()}
            last = (idx == n - 1)
            draw_hex_net(flat[idx], snap, gs, routing=rt, in_pos=in_pos,
                         out_pos=(out_pos if last else {}), show_edges=last,
                         title=('Iter %d (%d)' % (idx, len(snap))) if idx else 'Seed (%d)' % len(snap))
        for idx in range(n, len(flat)):
            flat[idx].set_visible(False)
        self._growth_canvas.draw_idle()

    def _draw_growth_lut(self, genome, target, fitness):
        """Growth tab for the LUT array: honeycomb-equivalent square panels drawn
        with the first-class renderer (per-side capability nibs + I/O markers),
        the final panel showing the mature organism, plus a legend."""
        try:
            from lut_evo import grow_lut_snapshots, place_outputs_by_trace
            from lut_evo.ga import _place_outputs_combinational
            snaps = grow_lut_snapshots(genome, seeds=tuple(target.inputs),
                                       grid_size=target.grid_size, iters=target.iters)
        except Exception:
            return
        final  = snaps[-1] if snaps else {}
        in_pos = [p for p in target.inputs if p in final]
        try:
            if getattr(target, 'temporal', False):
                out_pos, _ = place_outputs_by_trace(final, list(target.inputs), target)
            else:
                out_pos = _place_outputs_combinational(final, target)
        except Exception:
            out_pos = {}
        fig = self._growth_fig
        fig.clf()
        fig.suptitle('%s — LUT array growth   (fitness=%.4f)%s'
                     % (target.name, fitness, self._seed_tag()),
                     fontsize=10, fontweight='bold', y=0.995)
        n = len(snaps)
        # The distinct lookup tables actually present in the mature organism,
        # most common first — shown below as REAL truth tables (green=1/white=0)
        # so the raw table content is visible, not just the wedge-colour hash.
        from collections import Counter
        from lut_evo import lut_sop
        from lut_evo.viz import _lut_color
        counts = Counter(v for st in final.values() for v in st if v)
        luts   = [v for v, _ in counts.most_common(8)]
        total  = n + 1 + len(luts)                  # snapshots + legend + LUT gallery
        ncols  = min(6, max(3, math.ceil(math.sqrt(total))))
        nrows  = math.ceil(total / ncols)
        axes   = fig.subplots(nrows, ncols, squeeze=False)
        flat   = [a for row in axes for a in row]
        for idx, snap in enumerate(snaps):
            last = (idx == n - 1)
            draw_lut_net(flat[idx], snap, in_pos=in_pos,
                         out_pos=(out_pos if last else {}), show_edges=True,
                         title=('Iter %d (%d)' % (idx, len(snap))) if idx
                               else 'Seed (%d)' % len(snap))
        leg = flat[n]
        leg.set_visible(True); leg.axis('off')
        patches = [
            mpatches.Patch(facecolor='white', edgecolor='#b02020', lw=2, label='Input seed'),
            mpatches.Patch(facecolor='white', edgecolor='#17902f', lw=2, label='Output cell'),
            mpatches.Patch(facecolor='#7ec8e3', edgecolor='#c3ccd6',
                           label='Cell = 4 directional LUTs'),
            mpatches.Patch(facecolor='#e0b0ff', edgecolor='#c3ccd6',
                           label='wedge colour = that LUT’s state'),
            mpatches.Patch(facecolor=(0.95, 0.96, 0.97), edgecolor='#c3ccd6',
                           label='Dead direction (LUT = 0)'),
            mpatches.Patch(facecolor='#1ea64a', edgecolor='#b9c2cc',
                           label='tables below: green=1, white=0'),
        ]
        leg.legend(handles=patches, loc='center', fontsize=7.5, frameon=False,
                   title='LUT array  (colour = LUT state, per Fig. 13;\nactual lookup tables shown below)',
                   title_fontsize=8)
        for k, v in enumerate(luts):
            sop = lut_sop(v)
            if len(sop) > 16:
                sop = sop[:15] + '…'
            draw_lut_table(flat[n + 1 + k], v, swatch=_lut_color(v),
                           title='LUT %04X  ×%d\n%s' % (v, counts[v], sop))
        for idx in range(total, len(flat)):
            flat[idx].set_visible(False)
        self._growth_canvas.draw_idle()

    def _draw_lut_dynamics(self, genome):
        """Activity tab for a temporal LUT target: the paper's Fig. 14 'motion
        picture' as a filmstrip — the array's OUTPUT BITS (green=1 / red=0 per
        direction) at a spread of ticks through the first trial, so the running
        dynamics that ARE the LUT's computation are visible over time."""
        target = self._disp_target
        try:
            from lut_evo import grow_lut, LutSim, place_outputs_by_trace
            grid = grow_lut(genome, seeds=tuple(target.inputs),
                            grid_size=target.grid_size, iters=target.iters)
            if len(grid) <= target.n_inputs:
                raise ValueError
            out_pos, _ = place_outputs_by_trace(grid, list(target.inputs), target)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: circuit incomplete — grew too little)')
            return
        trial   = target.trials[0]
        in_pos  = [p for p in target.inputs if p in grid]
        sim     = LutSim(grid)
        frames  = []                                   # (tick, nibble-map)
        for t in range(target.T):
            sim.step({target.inputs[i]: trial.streams[t][i]
                      for i in range(target.n_inputs)})
            frames.append((t, dict(sim.out)))
        # show up to 8 evenly-spaced ticks
        k    = min(8, len(frames))
        idxs = [round(i * (len(frames) - 1) / max(1, k - 1)) for i in range(k)]
        self._volt_fig.clf()
        self._volt_fig.suptitle('%s — LUT output dynamics over ticks  '
                                '(green=1 / red=0, Fig. 14)%s'
                                % (target.name, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        ncols = min(4, k)
        nrows = math.ceil(k / ncols)
        axes  = self._volt_fig.subplots(nrows, ncols, squeeze=False)
        flat  = [a for row in axes for a in row]
        for j, fi in enumerate(idxs):
            tick, nib = frames[fi]
            draw_lut_net(flat[j], grid, activity=nib, in_pos=in_pos,
                         out_pos=out_pos, show_edges=True, title='tick %d' % tick)
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
            from lut_evo import lut_case_outputs
            grid, out_pos, cases = lut_case_outputs(genome, target)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: cannot evaluate this circuit)')
            return
        if not cases:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(LUT: circuit incomplete — inputs/outputs missing)')
            return
        in_pos = [p for p in target.inputs if p in grid]
        cases  = cases[:MAX_VOLT_CASES]
        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s — LUT array response per case%s%s'
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
                         title='in=%s\n%s  %s%s' % (''.join(map(str, case['in_bits'])),
                                                  res, 'OK' if all_ok else 'FAIL', settled))
            axes[ci].set_title(axes[ci].get_title(),
                               color='green' if all_ok else 'red', fontsize=7)
        self._volt_canvas.draw_idle()

    def _draw_voltages(self, genome):
        if self._disp_backend == 'nervous':
            self._draw_activity(genome)
            return
        if self._disp_backend == 'lut':
            self._draw_activity_lut(genome)
            return
        target = self._disp_target
        try:
            _, ns, ss = interpret_for(genome, target, self._disp_arch)
            in_ids = []
            for pos in target.inputs:
                nrn = next((n for n in ns if (n.x, n.y) == pos and n.is_input), None)
                if nrn is None:
                    raise ValueError
                in_ids.append(nrn.id)
            out_neurons = []
            for term in target.outputs:
                nrn = next((n for n in ns if n.is_output and n.out_role == term.role), None)
                if nrn is None:
                    raise ValueError
                out_neurons.append(nrn)
        except Exception:
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   '(circuit incomplete — no I/O neurons to trace)')
            return

        cases   = target.cases[:MAX_VOLT_CASES]
        n_rows  = len(cases)
        n_cols  = len(target.outputs)
        track   = [n.id for n in out_neurons]
        t       = np.arange(N_STEPS) * DT
        palette = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d cases)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s — Membrane Voltages%s%s' % (target.name, extra, self._seed_tag()),
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
                neuron     = out_neurons[col]
                spk, vmem  = sims[term.complement_inputs]
                fired      = len(spk.get(neuron.id, [])) >= 1
                act        = (0 if fired else 1) if term.invert_spike else (1 if fired else 0)
                exp        = out_bits[col]
                color      = palette[col % len(palette)]
                ax = self._volt_fig.add_subplot(gspec[row, col])
                ax.plot(t, vmem[neuron.id], color=color, lw=1.1)
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
                    ax.set_title("%s  pos=%s%s" % (term.role, (neuron.x, neuron.y), note),
                                 fontsize=8.5)
                ax.text(0.02, 0.88, 'in=%s' % ''.join(map(str, in_bits)),
                        transform=ax.transAxes, fontsize=7.5, va='top')
                # Make the spike->logic mapping explicit so an inverted (carry)
                # output doesn't look like the result contradicts the trace.
                fy  = 'Y' if fired else 'N'
                res = 'fired=%s -> %d  exp=%d  %s' % (fy, act, exp,
                                                          'OK' if act == exp else 'FAIL')
                ax.text(0.98, 0.88, res, transform=ax.transAxes,
                        fontsize=7.5, va='top', ha='right',
                        color='green' if act == exp else 'red', fontweight='bold')
                if row == n_rows - 1:
                    ax.set_xlabel('Time (ms)', fontsize=8)
        self._volt_canvas.draw_idle()

    def _draw_activity(self, genome):
        """Nervous-net version of the Voltage tab: node activity per input case."""
        target = self._disp_target
        if getattr(target, 'temporal', False):
            self._draw_placeholder(self._volt_fig, self._volt_canvas,
                                   'Temporal target — drive it over time in the Interactive tab '
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
                                   '(nervous: circuit incomplete — inputs/outputs missing)')
            return

        gs      = target.grid_size
        cases   = cases[:MAX_VOLT_CASES]
        routing = interpret_nervous(grid, target)[0]
        self._volt_fig.clf()
        extra = '' if len(target.cases) <= MAX_VOLT_CASES else \
                '  (first %d of %d)' % (MAX_VOLT_CASES, len(target.cases))
        self._volt_fig.suptitle('%s — hex nervous-net activity per case%s%s'
                                % (target.name, extra, self._seed_tag()),
                                fontsize=10, fontweight='bold', y=0.99)
        axes = self._volt_fig.subplots(1, len(cases), squeeze=False)[0]
        for ci, case in enumerate(cases):
            all_ok = all(case['acts'][t.role] == case['out_bits'][i]
                         for i, t in enumerate(target.outputs))
            res = '  '.join('%s %d/%d' % (t.role, case['out_bits'][i], case['acts'][t.role])
                            for i, t in enumerate(target.outputs))
            draw_hex_net(axes[ci], grid, gs, routing=routing, in_pos=in_pos, out_pos=out_pos,
                         activity=case['node_outputs'], show_edges=True,
                         title='in=%s\n%s  %s' % (''.join(map(str, case['in_bits'])), res,
                                                  'OK' if all_ok else 'FAIL'))
            axes[ci].set_title(axes[ci].get_title(),
                               color='green' if all_ok else 'red', fontsize=7)
        self._volt_canvas.draw_idle()

    # ── genome viewer ─────────────────────────────────────────────────────────

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
        # growth-limit badge (top-right of card) — SNN genes only; the hex gene
        # has no per-gene iteration limit (paper-faithful associative memory)
        if hasattr(gene, 'limit'):
            ax.text(ox + 5.85, oy - 0.18, 'it<=%d' % gene.limit, ha='right', va='top',
                    fontsize=6.5, color='#888', zorder=3)

    def _draw_genome(self, genome, fitness):
        fig = self._genome_fig
        fig.clf()
        chroms = list(getattr(genome, 'chromosomes', []) or [])
        if not chroms:
            self._draw_placeholder(fig, self._genome_canvas, '(empty genome)')
            return
        n_total = sum(len(c.genes) for c in chroms)
        title = 'Genome  —  %d chromosome%s,  %d genes' % (
            len(chroms), '' if len(chroms) == 1 else 's', n_total)
        if fitness is not None:
            title += '   (fitness = %.4f)' % fitness
        title += self._seed_tag()

        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.axis('off')

        CW, CH  = 7.0, 4.4        # card footprint (incl. spacing)
        per_row = 4
        cap     = 48              # keep cards readable; note the rest
        y       = 0.0
        drawn   = 0
        truncated = False

        for ci, chrom in enumerate(chroms):
            tel = getattr(chrom, 'telomere', None)
            ax.text(0.0, y, "Chromosome %s   ·   tag %d   ·   split %d%s   ·   %d genes"
                    % (chr(ord('a') + ci), chrom.tag, chrom.split,
                       ('   ·   telomere %d' % tel) if tel is not None else '',
                       len(chrom.genes)),
                    fontsize=9.5, fontweight='bold', color='#334', va='bottom')
            y += 1.0
            ng = len(chrom.genes)
            for gi, gene in enumerate(chrom.genes):
                if drawn >= cap:
                    truncated = True
                    break
                self._draw_gene(ax, (gi % per_row) * CW, y + (gi // per_row) * CH, gene)
                drawn += 1
            y += (math.ceil(ng / per_row) if ng else 1) * CH + 0.7
            if truncated:
                ax.text(0.0, y, '…  %d more genes not shown' % (n_total - cap),
                        fontsize=9, color='#a00', va='bottom')
                y += 1.0
                break

        _hex = bool(chroms and chroms[0].genes and hasattr(chroms[0].genes[0], 'ctx_l'))
        _sides = 'hex L/R/D sides' if _hex else 'N/E/S/W sides'
        _tail = ('.' if _hex else
                 ', active while iter <= limit.')
        ax.text(0.0, y + 0.2,
                'card = one gene: the cluster is the expected neighbourhood '
                '(%s + centre = self), the chip after -> is the output state; '
                'growth picks the gene closest (min Hamming distance) to a cell%s'
                % (_sides, _tail),
                fontsize=7.5, color='#666', va='bottom', wrap=True)

        ax.set_xlim(-0.4, per_row * CW)
        ax.set_ylim(0, y + 1.4)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        self._genome_canvas.draw_idle()

    # ── save PNGs ─────────────────────────────────────────────────────────────

    def _save_pngs(self):
        if self.best_genome is None:
            self._status.set('No genome loaded — nothing to save.')
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
            f.write(build_genome_text(self.best_genome, self.best_fitness))
        self._status.set('Saved growth / voltage / genome (png + txt) (%s) → results/' % ts)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
