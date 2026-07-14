"""
designer.py — manual circuit designer for the nervous-net and LUT backends.

Lets a human design a circuit BY HAND at either level of the indirect encoding:

  * the GENOME (the DNA): a structured editor over chromosomes/genes with
    range-validated fields. Pressing Grow runs the real growth engine
    (grow_nervous / grow_lut) and REPLACES the working grid — this direction
    is exact and deterministic.
  * the CIRCUIT (the organism): the working grid itself is a first-class,
    directly editable artifact — place/delete cells, set each nv cell's
    three independent 4-bit Figure-3 routing configurations (one L/R/D output circuit;
    neighbours resolved through hex_dirs so parity is honest) or a LUT cell's four directional
    tables (hex / clickable 4x4 truth grid / decoded SOP), mark inputs and
    output roles. Both PulseSim and LutSim run on a bare grid, so a
    hand-built circuit needs NO genome to be simulated or scored. The nervous
    net is driven ASYNCHRONOUSLY — input pulses placed on a clickable timeline,
    played in continuous time via the paper-faithful PulseSim used by Nervous
    evolution — while the clocked LUT keeps its tick-numbered bit streams.

Genome -> grid is exact via Grow. Grid -> genome is a separate best-effort
synthesis pass: because growth is many-to-one it verifies the phenotype it can
regrow and reports misses/extras rather than pretending to recover original DNA.
A visible sync indicator tracks in-sync / grid-edited / genome-edited / no-genome.

Simulation mirrors interactive.py (input schedules + Step/Run/Reset + a trace
strip). Scoring uses the target's declared event, cadence, or persistence
metric, exactly as the evolver does.

Imports evolved solutions (results/best_genome.json, solver_generation.json,
or the running app's current_circuit() hand-off) so a user can start from an
evolved genome and refine it by hand. Saves designs in the app's pickle
schema plus {grid, in_pos, out_pos, hand_built} so hand-built phenotypes
round-trip even without a genome.

Run standalone:  py designer.py
"""
from __future__ import annotations
import os, sys, math, json, pickle, dataclasses

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D

from nv_evo.hexgrid import (hex_dirs, hex_pixel, PAPER_ROUTING,
                            DIRECTIONS, GENOME_ENCODING, TILE_ENCODING,
                            CIRCUIT_STATE_COUNT,
                            pack_tile_state,
                            unpack_tile_state, promote_legacy_state,
                            decode_tile_routing, channel_tile, tile_channel,
                            normalize_output_channel, facing_channel)
from nv_evo.genome import (HexGene, Chromosome as NvChromosome, Genome as NvGenome,
                           random_hex_gene, random_hex_genome,
                           MAX_TELOMERE as NV_MAX_TEL)
from nv_evo.nervous import (grow_nervous, interpret_nervous, evaluate_nervous,
                            circuit_summary_nervous, SEED_STATE as NV_SEED_STATE)
from nv_evo.reverse import grid_to_genome_nervous, repair_genome_nervous
from nv_evo.playback import NervousPlayer, PulseLaneEditor, pulses_from_trial
from nv_evo.temporal import (run_nervous_events,
                             place_outputs_by_trace, TemporalTraces,
                             windowed_score, exact_tick_accuracy,
                             _pr_counts, _f1, _best_shift, _obs_len,
                             _best_event_shift, _event_case_score,
                             _expected_events, _role_events,
                             cadence_score, sampled_events,
                             trial_input_summary, expected_window_summary,
                             event_list_summary, period_stepper_score)
from nv_evo.targets import TEMPORAL_TARGETS
from nv_evo.viz import draw_hex_net, channel_pixel

from lut_evo.genome import (LutGene, Chromosome as LutChromosome, Genome as LutGenome,
                            random_lut_gene, MAX_TELOMERE as LUT_MAX_TEL)
from lut_evo.lut import grow_lut, LutSim, SEED_STATE as LUT_SEED_STATE
from lut_evo.reverse import grid_to_genome_lut, repair_genome_lut
from lut_evo.ga import (place_outputs_by_trace as lut_place_by_trace,
                        make_seed_genome as make_lut_seed_genome,
                        compact_genome as lut_compact_genome)
from lut_evo.viz import draw_lut_net, draw_lut_table
from lut_evo.boolfn import lut_sop

from snn_evo.targets import TARGETS as COMB_TARGETS
from target_ui import TargetPicker

import ui_compat

RESULTS_DIR = os.path.join(ROOT, 'results')

_NV_GENE_FIELDS  = ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out')
_LUT_GENE_FIELDS = ('ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in', 'self_out')
_LUT_DIRS        = ('N', 'S', 'E', 'W')        # grid tuple order (Ln, Ls, Le, Lw)
_GENE_ROW_CAP    = 40      # widget rows per chromosome (evolved LUT genomes can be huge)


def _valid_split(genes, value):
    count = len(genes)
    return 0 if count < 2 else max(1, min(int(value), count - 1))


def _nv_routing(grid):
    return {pos: decode_tile_routing(state) for pos, state in grid.items()}


def _output_is_live(grid, pos, backend):
    if not pos:
        return False
    return (channel_tile(pos) in grid if backend == 'nervous'
            else pos in grid)


class _Tip:
    """Minimal hover tooltip (Tk ships none)."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _show(self, _e=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry('+%d+%d' % (x, y))
        tk.Label(tw, text=self.text, justify='left', background='#fffbe8',
                 relief='solid', borderwidth=1, font=('TkDefaultFont', 8),
                 wraplength=340).pack(ipadx=4, ipady=2)

    def _hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


DESIGN_FORMAT = 'evohw-design-v1'


# ── text (JSON) design serialisation — human-editable, replaces the pickle ───────
# A design is genome + working grid + I/O + target name, all as plain
# numbers/strings so the file can be opened and hand-edited in any text editor.
# The genome is one row per gene ([ctx…, self_in, self_out]); grid keys are
# "x,y". Targets are stored by NAME and re-resolved from the registry on load.

def _genome_to_dict(genome, backend):
    fields = _NV_GENE_FIELDS if backend == 'nervous' else _LUT_GENE_FIELDS

    def gene_row(gene):
        values = [int(getattr(gene, field)) for field in fields]
        if backend == 'nervous' and any(
                not 0 <= value < CIRCUIT_STATE_COUNT for value in values):
            raise ValueError(
                'nervous genome fields must be 4-bit circuit states (0..15)')
        return values

    document = {'tag': genome.tag,
                'gene_fields': list(fields),
                'chromosomes': [{'tag': c.tag,
                                 'split': _valid_split(c.genes, c.split),
                                 'telomere': getattr(c, 'telomere', None),
                                 'genes': [gene_row(g) for g in c.genes]}
                                for c in genome.chromosomes]}
    if backend == 'nervous':
        document['state_encoding'] = GENOME_ENCODING
    return document


def _genome_from_dict(d, backend):
    if backend == 'nervous':
        Gene, Chrom, Gen, fields = HexGene, NvChromosome, NvGenome, _NV_GENE_FIELDS
    else:
        Gene, Chrom, Gen, fields = LutGene, LutChromosome, LutGenome, _LUT_GENE_FIELDS
    fields = tuple(d.get('gene_fields') or fields)
    encoding = d.get('state_encoding') if backend == 'nervous' else None
    if backend == 'nervous':
        from evo_runtime.checkpoint import _nervous_gene_rows
    chroms = []
    for c in d['chromosomes']:
        genes = []
        for row in c['genes']:
            values = [int(v) for v in row]
            rows = (_nervous_gene_rows(values, encoding)
                    if backend == 'nervous' else [values])
            genes.extend(Gene(**{f: v for f, v in zip(fields, migrated)})
                         for migrated in rows)
        raw_split = int(c.get('split', 0))
        if backend == 'nervous' and encoding == TILE_ENCODING:
            raw_split *= 3
        kw = {'genes': genes,
              'split': _valid_split(genes, raw_split),
              'tag': int(c.get('tag', 0))}
        if c.get('telomere') is not None:
            kw['telomere'] = int(c['telomere'])
        chroms.append(Chrom(**kw))
    return Gen(chromosomes=chroms, tag=int(d.get('tag', 0)))


def _grid_to_dict(grid, backend):
    if backend == 'nervous':
        for state in grid.values():
            unpack_tile_state(state)  # validate the packed phenotype word
        return {'%d,%d' % p: int(s) for p, s in grid.items()}
    return {'%d,%d' % p: [int(v) for v in s] for p, s in grid.items()}


def _grid_from_dict(d, backend, state_encoding=None):
    out = {}
    for k, v in d.items():
        x, y = (int(n) for n in k.split(','))
        if backend == 'nervous':
            state = int(v)
            if state_encoding != TILE_ENCODING:
                state = promote_legacy_state(state)
            else:
                unpack_tile_state(state)  # validate current packed encoding
            out[(x, y)] = state
        else:
            out[(x, y)] = tuple(int(n) for n in v)
    return out


def read_saved_file(path):
    """Return an app-style state dict from EITHER a text JSON design or a pickle
    (evolver output / legacy design). Shared by the main app's Load Saved so the
    editable JSON format works everywhere. Reconstructs the genome from JSON and
    resolves the target by name; raises on unreadable/omit-both files."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
    except (UnicodeDecodeError, ValueError, OSError):
        doc = None
    if isinstance(doc, dict) and str(doc.get('format', '')).startswith('evohw-checkpoint'):
        from evo_runtime.checkpoint import load_checkpoint
        return load_checkpoint(path)
    if isinstance(doc, dict) and str(doc.get('format', '')).startswith('evohw-design'):
        backend = doc.get('backend', 'nervous')
        genome = _genome_from_dict(doc['genome'], backend) if doc.get('genome') else None
        name = doc.get('target') or ''
        target = TEMPORAL_TARGETS.get(name)
        if target is None and backend == 'nervous':
            target = COMB_TARGETS.get(name)
        return {'best_genome': genome, 'best_fitness': doc.get('fitness', 0.0),
                'target': target, 'target_name': name, 'arch': None, 'seed': None,
                'backend': backend, 'hand_built': True,
                'grid': _grid_from_dict(
                    doc.get('grid', {}), backend,
                    doc.get('state_encoding') or
                    (doc.get('genome') or {}).get('state_encoding')),
                'in_pos': [tuple(p) for p in doc.get('inputs', [])],
                'out_pos': {r: tuple(p) for r, p in (doc.get('outputs') or {}).items()},
                'grid_edited': doc.get('grid_edited', True)}
    with open(path, 'rb') as f:
        return pickle.load(f)


def _route_text(idx, x, y):
    """One paper Figure-3 configuration with resolved E1/E2/I1 labels."""
    entry = PAPER_ROUTING[idx]
    e1, e2, i1 = entry[0], entry[1], entry[2]
    op = entry[3] if len(entry) > 3 else 'and'
    nb = hex_dirs(x, y)

    def cell(d):
        return '%s%s' % (d, nb[d])
    if e1 is None:
        return '%X  off (no routed input)' % idx
    i1s = i1 if i1 is not None else '-'
    if e1 == e2:
        # This is one physical nerve fanned into the two terminals required by
        # the coincidence circuit, not two excitatory wires.
        return ('%X  input=%s -> E1+E2; I1=%s   buffer: relays %s'
                % (idx, e1, i1s, cell(e1)))
    else:
        desc = 'coincidence: %s AND %s' % (cell(e1), cell(e2))
    if i1 is not None:
        desc += ', veto %s' % cell(i1)
    return '%X  E1=%s E2=%s I1=%s   %s' % (idx, e1, e2, i1s, desc)


_GUIDE_NERVOUS = """\
ARCHITECTURE 1 — nervous net (paper Figs. 2-4)

Honeycomb array: every node touches exactly THREE
neighbours — Left, Right, Down IN ITS OWN ORIENTATION
("the concepts right, left and down are rotated as
necessary to match the network topology", Fig. 4). Up-
and down-oriented nodes are mirror images, so the Cell
tab always shows which real cell each label reaches.

A physical tile has THREE independent core circuits
(L/R/D outputs).
Each circuit has two excitatory inputs and one inhibitory:

        out = (E1 AND E2) AND NOT I1

"Neither input alone can trigger a response" for
coincidence. A BUFFER routes ONE nerve to both E1 and E2
(the required internal fan-out, drawn as one green wire);
I1, if active, is one separate veto input.
Each circuit independently selects one of the paper's
16 Fig. 3 configurations (0-F), so a tile stores three
4-bit selectors. The packed 12-bit integer is storage,
not a menu of 4096 primitive circuit states.

The GENOME is deliberately smaller: every context,
self-input and output field is ONE 4-bit circuit selector
(decimal 0-15), never a packed tile word. During growth
the same gene is evaluated independently for the L, R and
D core circuits, with the neighbourhood rotated to face
the circuit being updated. Only the three developed
phenotype results are packed into a 12-bit tile.
An external target input drives all three channels of
its perimeter tile; outputs read one named channel.

MEMORY: a pulse injected by an input circulates a loop
of buffers "until stopped by application of an
inhibitory input". A node is refractory after firing,
so a stored 1 reads as RINGING (1010...) — the Score
metric knows this (±1-tick tolerance on expected 1s).

GENES (Genome tab): the associative memory — a context
(L, R, D, self) mapped to a new state by minimum
Hamming distance. Rows with self=0 are GROWTH rules,
the only kind that can bring an empty cell to life.
A chromosome's telomere is a Hayflick division limit:
growth halts by itself at that radius from the seeds.

WORKFLOW
 1. Mark the inputs first (Input mode, or Adopt target
    I/O) — inputs are the growth SEEDS.
 2. Grow (ontogeny) develops the genome into a circuit.
    One-way: editing the circuit never alters the DNA.
 3. Or skip DNA entirely: Place/Delete cells, then set
    each tile's three Fig. 3 states in the Cell tab.
 4. Step/Run injects pulses at the inputs; Score uses
    the target's event, cadence, or persistence metric."""

_GUIDE_LUT = """\
ARCHITECTURE 2 — LUT array (paper §5-7, sim6)

Square array, four neighbours, LATCHED and SYNCHRONOUS.
Every cell is the SAME hardware: FOUR 16-bit lookup
tables, one driving each output direction (N, S, E, W).
Each LUT is a truth table over the four bits the cell
receives from its neighbours — what differs between
cells is only the CONTENTS of the tables. The hardware
never changes: a "dead" cell is simply one whose four
tables are all zero (drawn as a dashed square around
the organism; the sparse grid stores only live cells).
In Select mode you can click a dead position and write
a table into it — that IS how a cell comes to life.
Zeroing all four tables of a live cell kills it.

Colours follow the paper: in structure view each wedge
is coloured by its LUT state ("unique colours are
assigned to unique LUT states", Fig. 13); during
Step/Run the wedges show the instantaneous digital
state in red (0) and green (1) (Fig. 14).

GENES (Genome tab): Fig. 10 verbatim — a context of
FIVE 16-bit LUT states (the four neighbours' facing
LUTs + the cell's own LUT) mapped to a new 16-bit LUT
by minimum Hamming distance over all 80 bits. During
growth the context is ROTATED so the output direction
is "front": one gene expresses all four directions.
Rows with self=0 are GROWTH rules (the only kind that
can bring a dead direction to life); a chromosome's
telomere expires its growth rules, so expansion
provably stops and maintenance settles the attractor.

Click ▦ next to any 16-bit field to open it in the
table editor below: click squares to toggle bits; the
decoded boolean function (sum of products over
N,S,E,W) updates live.

WORKFLOW
 1. Mark inputs (the growth seeds — pinned to the
    always-relay table FFFE during Grow).
 2. Grow (ontogeny), or hand-build: Place cells and
    edit their four tables in the Cell tab.
 3. Step/Run drives the latched array one tick at a
    time (Fig. 14 view); Score uses the target's
    declared metric, exactly as the evolver does."""


def _routed_channels(idx, x, y):
    """Exact source circuits wired into one Figure-3 circuit at (x,y)."""
    e1, e2, i1 = PAPER_ROUTING[idx][:3]
    pos = (x, y)
    exc = {facing_channel(pos, d) for d in (e1, e2) if d is not None}
    veto = facing_channel(pos, i1) if i1 is not None else None
    return exc, veto


def _traces_report(ttarget, traces, out_pos, label):
    """Semantic score report for a manually edited working grid."""
    lines = ['Target: %s   [%s — designer working grid]' % (ttarget.name, label)]
    mode = getattr(ttarget, 'score_mode', 'trace')
    if mode == 'period_stepper':
        total, cases = period_stepper_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos.get(term.role)))
        for ti, (trial, score) in enumerate(zip(ttarget.trials, cases), 1):
            lines.append('Test %d: %s  cadence-step score %.3f %s' % (
                ti, trial_input_summary(trial, ttarget.n_inputs), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence-stepper score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return total, '\n'.join(lines)
    if mode == 'cadence':
        total, cases = cadence_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos.get(term.role)))
        for ti, score in enumerate(cases):
            role = ttarget.outputs[0].role
            lines.append('Test %d: %s  output edges@%s  cadence %.3f %s' % (
                ti + 1, trial_input_summary(ttarget.trials[ti], ttarget.n_inputs),
                event_list_summary(_role_events(traces, role, ti)), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return total, '\n'.join(lines)
    if mode == 'events':
        best_s, total = _best_event_shift(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos.get(term.role)))
        lines.append('measured output latency offset: %+.3f' % best_s)
        for ti, trial in enumerate(ttarget.trials):
            for role in trial.expected:
                score = _event_case_score(traces, ttarget, ti, role, best_s)
                lines += ['', 'Test %d: %s' % (
                    ti + 1, trial_input_summary(trial, ttarget.n_inputs)),
                    '  expect %s edges@%s' % (
                        role, event_list_summary(_expected_events(trial, role))),
                    '  actual edges@%s  F1 %.3f %s' % (
                        event_list_summary(_role_events(traces, role, ti)), score,
                        'PASS' if score >= 0.999 else 'FAIL')]
        lines += ['', '=> event score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return total, '\n'.join(lines)
    best_s = _best_shift(traces, ttarget)[0]    # latency-invariant: one global shift
    for term in ttarget.outputs:
        lines.append("out '%s' read at %s" % (term.role, out_pos.get(term.role)))
    lines.append('measured output latency offset: %+d tick(s)' % best_s)
    for ti, trial in enumerate(ttarget.trials):
        lines += ['', 'Test %d: %s' % (
            ti + 1, trial_input_summary(trial, ttarget.n_inputs))]
        for role, exp in trial.expected.items():
            lines.append('  expect %s%s' % (
                expected_window_summary(exp),
                ' (%s)' % role if len(trial.expected) > 1 else ''))
            tr = traces.get(role, [])
            tr_i = tr[ti] if ti < len(tr) else []
            tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp, best_s)
            s = _f1(tp_rec, n_exp, tp_prec, n_act)
            lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                         % (''.join(str(v) for v in tr_i), s,
                            'PASS' if s >= 0.999 else 'FAIL',
                            tp_rec, n_exp, tp_prec, n_act))
    total = windowed_score(traces, ttarget, shift=best_s)
    lines += ['', '=> behavioural score %.4f%s   (exact per-tick %.4f)'
              % (total, '   SOLVED' if total >= 0.999 else '',
                 exact_tick_accuracy(traces, ttarget))]
    return total, '\n'.join(lines)


class DesignerTab:
    """Manual designer: genome editor + circuit editor + simulation + scoring.
    `get_circuit` is the app's current_circuit() callback (None standalone)."""

    def __init__(self, parent, get_circuit=None):
        self.parent      = parent
        self.get_circuit = get_circuit
        self.backend     = 'nervous'
        self.genome      = None
        self.grid        = {}            # nv {pos:int} / lut {pos:(Ln,Ls,Le,Lw)}
        self.in_pos      = []            # ordered input cells (A, B, ...)
        self.out_pos     = {}            # {role: pos}
        self._grid_edited   = False      # grid touched since last Grow
        self._genome_edited = False      # genome touched since last Grow
        self._selected   = None
        self._last_score = None
        self._sim        = None
        self._tick       = 0
        self._state      = {}
        self._nibbles    = {}
        self._traces     = {}
        self._running    = False
        self._stream_vars = []           # per-input tick-numbered bit-stream entries
        self._trial_idx  = tk.IntVar(value=1)
        self._lut_bind   = None          # (get, set, describe) for the table editor
        self._mono       = ui_compat.mono_family(parent)
        self._build_ui()
        self._refresh_all()

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        top = ttk.Frame(self.parent, padding=(6, 4))
        top.pack(fill='x')
        ttk.Label(top, text='Architecture:').pack(side='left')
        self._backend_var = tk.StringVar(value='Nervous net (Arch 1)')
        cb = ttk.Combobox(top, textvariable=self._backend_var, width=18,
                          values=['Nervous net (Arch 1)', 'LUT array (Arch 2)'],
                          state='readonly')
        cb.pack(side='left', padx=(2, 8))
        cb.bind('<<ComboboxSelected>>', self._on_backend_change)
        _Tip(cb, "The paper's two substrates: Architecture 1 = honeycomb nervous "
                 "net (coincidence + inhibition, pulse dynamics); Architecture 2 = "
                 "square array of four 16-bit LUTs per cell, latched & synchronous.")

        ttk.Label(top, text='Target:').pack(side='left')
        self._target_var = tk.StringVar(value='(none)')
        self._target_picker = TargetPicker(
            top, {}, variable=self._target_var, command=self._on_target_selected,
            include_none=True, target_width=20)
        self._target_picker.pack(side='left', padx=(2, 4))
        self._target_cb = self._target_picker.target_cb
        b = ttk.Button(top, text='Adopt target I/O', command=self._adopt_target_io)
        b.pack(side='left', padx=2)
        _Tip(b, "Copy the target's input terminals onto the grid as the growth "
                "seeds, and free its output roles for assignment (or auto-"
                "placement at scoring).")

        files = ttk.Frame(self.parent, padding=(6, 0))
        files.pack(fill='x')
        ttk.Label(files, text='Files:').pack(side='left')
        b = ttk.Button(files, text='Load evolved…', command=self._load_evolved)
        b.pack(side='left', padx=2)
        _Tip(b, 'Import results/best_genome.json or solver_generation.json: the '
                'evolved genome opens in the Genome tab, is grown, and its '
                'target is adopted — then refine it by hand.')
        if self.get_circuit is not None:
            b = ttk.Button(files, text='From app', command=self._from_app)
            b.pack(side='left', padx=2)
            _Tip(b, "Pull the main window's current best solution into the designer.")
        ttk.Button(files, text='Load design…', command=self._load_design).pack(side='left', padx=2)
        b = ttk.Button(files, text='Save design…', command=self._save_design)
        b.pack(side='left', padx=2)
        _Tip(b, 'Saves the genome (when one exists) AND always the working grid '
                '+ inputs/outputs + target, so a hand-built phenotype round-trips '
                'even without DNA.')

        row2 = ttk.Frame(self.parent, padding=(6, 0))
        row2.pack(fill='x')
        ttk.Label(row2, text='Click mode:').pack(side='left')
        self._mode = tk.StringVar(value='select')
        self._mode.trace_add('write', self._on_mode_change)
        for label, val, tip in (
                ('Select', 'select',
                 'Click a live cell to inspect/edit it in the Cell tab — its '
                 'three directional Fig. 3 states (nervous) or its four LUTs.'),
                ('Place/Delete', 'cell',
                 'Click an empty lattice position to add a cell; click a live '
                 'cell to remove it. On the LUT array this only changes '
                 'CONTENTS — the hardware field is uniform: place loads the '
                 'relay seed table (FFFE x4), delete zeroes all four tables.'),
                ('Input', 'input',
                 'Toggle cells as inputs (A, B, …). Inputs are the growth SEEDS '
                 '(pinned during Grow) and receive the scheduled pulses/clocked '
                 'streams during simulation.'),
                ('Output', 'output',
                 'Assign the next unfilled output role to a live cell. Nervous '
                 'outputs cycle through that tile\'s exact L/R/D channels.')):
            r = ttk.Radiobutton(row2, text=label, value=val, variable=self._mode)
            r.pack(side='left', padx=2)
            _Tip(r, tip)

        actions = ttk.Frame(self.parent, padding=(6, 2, 6, 0))
        actions.pack(fill='x')
        ttk.Label(actions, text='Circuit:').pack(side='left')
        b = ttk.Button(actions, text='Grow (ontogeny)', command=self._grow)
        b.pack(side='left', padx=2)
        _Tip(b, 'Develop the genome into a circuit with the associative-memory '
                'ontogeny (min-Hamming gene lookup), seeds = the designated '
                'inputs. ONE-WAY: replaces the working grid; growth is not '
                'invertible, so hand edits never flow back into the DNA.')
        b = ttk.Button(actions, text='⤺ To genome', command=self._reverse_to_genome)
        b.pack(side='left', padx=2)
        _Tip(b, 'Best-effort INVERSE of Grow: reconstruct a genome that develops '
                'back into the current working grid, so a hand-built circuit gets '
                'DNA it can be saved/evolved from. If retained DNA exists, repair '
                'only its hand-edited delta and preserve the working bulk. '
                'Growth is many-to-one and not '
                'truly invertible, so it replays a monotone development with the '
                'grid as an oracle — reproducing every cell it can, with harmless '
                'extra cells allowed. Seeds = the designated inputs. Press Grow '
                'afterwards to realise it, Score to confirm the outputs still match.')
        ttk.Button(actions, text='Random genome', command=self._random_genome).pack(side='left', padx=2)
        self._compact_btn = ttk.Button(actions, text='Compact genome', command=self._compact)
        self._compact_btn.pack(side='left', padx=2)
        _Tip(self._compact_btn,
             'LUT only: drop never-expressed genes (win no growth lookup). Provably '
             'behaviour-preserving — the grown organism is bit-identical — so it '
             'reveals the genome’s functional core for inspection/editing. '
             'Biologically, pseudogene loss.')
        ttk.Button(actions, text='Clear grid', command=self._clear_grid).pack(side='left', padx=2)
        ttk.Separator(actions, orient='vertical').pack(side='left', fill='y', padx=6)
        b = ttk.Button(actions, text='Score', command=self._score)
        b.pack(side='left', padx=2)
        _Tip(b, 'Judge the working grid with the target’s actual semantic metric: '
                'raw events, cadence, persistence windows, or settled logic. No '
                'genome is required; circulating pulses are valid stored state.')
        self._sync_lbl = ttk.Label(actions, text='', font=(self._mono, 9))
        self._sync_lbl.pack(side='left', padx=10, fill='x', expand=True)

        # A weighted splitter lets the inspector grow on wide/fullscreen windows
        # and remains user-adjustable.  The old fixed 360 px column clipped the
        # nervous-cell routing descriptions even when ample space was available.
        main = ttk.Panedwindow(self.parent, orient='horizontal')
        main.pack(fill='both', expand=True)

        # left: genome / inspector / report notebook + shared LUT table editor
        left = ttk.Frame(main, padding=(4, 4, 2, 4))
        self._nb = ttk.Notebook(left)
        self._nb.pack(fill='both', expand=True)
        guide_tab = ttk.Frame(self._nb)
        self._nb.add(guide_tab, text='Guide')
        self._guide = scrolledtext.ScrolledText(guide_tab, width=60, height=20,
                                                font=(self._mono, 8), wrap='word')
        self._guide.pack(fill='both', expand=True)
        self._genome_tab = ttk.Frame(self._nb)
        self._nb.add(self._genome_tab, text='Genome')
        self._insp_tab = ttk.Frame(self._nb)
        self._nb.add(self._insp_tab, text='Cell')
        self._report_tab = ttk.Frame(self._nb)
        self._nb.add(self._report_tab, text='Report')
        self._report = scrolledtext.ScrolledText(self._report_tab, width=60, height=20,
                                                 font=(self._mono, 8), wrap='word')
        self._report.pack(fill='both', expand=True)

        # scrollable genome frame — BOTH axes. A gene row (esp. LUT hex fields, or
        # a chromosome header ending in +gene) is wider than the left panel,
        # so a horizontal scrollbar keeps the rightmost fields reachable instead of
        # clipping them off the edge.
        gt = self._genome_tab
        canvas = tk.Canvas(gt, highlightthickness=0)
        vscroll = ttk.Scrollbar(gt, orient='vertical',   command=canvas.yview)
        hscroll = ttk.Scrollbar(gt, orient='horizontal', command=canvas.xview)
        self._genome_frame = ttk.Frame(canvas)
        self._genome_frame.bind('<Configure>',
                                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self._genome_frame, anchor='nw')
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')
        hscroll.grid(row=1, column=0, sticky='ew')
        gt.rowconfigure(0, weight=1)
        gt.columnconfigure(0, weight=1)

        self._insp_body = ttk.Frame(self._insp_tab, padding=4)
        self._insp_body.pack(fill='both', expand=True)

        # shared 4x4 LUT table editor (one canvas, rebound to whichever 16-bit
        # field has focus — cheap, and honours "click a cell to toggle a bit")
        lut_ed = ttk.LabelFrame(left, text='LUT table editor (click to toggle)', padding=2)
        lut_ed.pack(fill='x', pady=(4, 0))
        self._lut_fig = plt.Figure(figsize=(2.4, 2.2))
        self._lut_ax = self._lut_fig.add_subplot(111)
        self._lut_canvas = FigureCanvasTkAgg(self._lut_fig, master=lut_ed)
        self._lut_canvas.get_tk_widget().pack(fill='x')
        self._lut_canvas.mpl_connect('button_press_event', self._on_lut_click)
        self._lut_ax.axis('off')
        self._lut_fig.tight_layout()

        # right: circuit canvas + sim controls + trace strip
        right = ttk.Frame(main, padding=(2, 4, 4, 4))
        main.add(left, weight=3)
        main.add(right, weight=5)
        self._main_pane = main
        sim = ttk.Frame(right)
        sim.pack(fill='x')
        ttk.Button(sim, text='Step', command=self._step).pack(side='left', padx=2)
        self._run_btn = ttk.Button(sim, text='Run', command=self._toggle_run)
        self._run_btn.pack(side='left', padx=2)
        ttk.Button(sim, text='Reset', command=self._reset_sim_ui).pack(side='left', padx=2)
        self._tick_lbl = ttk.Label(sim, text='t = 0.0', font=(self._mono, 9))
        self._tick_lbl.pack(side='left', padx=8)
        # input streams get their own row (they can be wide)
        srow = ttk.Frame(right)
        srow.pack(fill='x', pady=(2, 0))
        self._input_heading = tk.StringVar(value='Input pulses (continuous time):')
        input_header = ttk.Frame(srow)
        input_header.pack(fill='x')
        ttk.Label(input_header, textvariable=self._input_heading).pack(side='left')
        self._input_actions = ttk.Frame(input_header)
        self._input_actions.pack(side='right')
        self._inbar = ttk.Frame(srow)
        self._inbar.pack(fill='both', expand=True)

        self.fig = plt.Figure(figsize=(6.4, 4.8))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.22)
        self._ax_grid = self.fig.add_subplot(gs[0, 0])
        self._ax_trace = self.fig.add_subplot(gs[0, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        self._status = tk.StringVar(
            value='Design by hand or Load evolved…  (Grow replaces the grid from the genome)')
        self._status_label = ttk.Label(
            right, textvariable=self._status, anchor='w', relief='sunken',
            padding=(6, 2), wraplength=650, justify='left')
        self._status_label.pack(fill='x', pady=(2, 0))
        right.bind('<Configure>', lambda event: self._status_label.configure(
            wraplength=max(360, event.width - 20)), add='+')

        self._refresh_target_list()

    # ── backend / target plumbing ────────────────────────────────────────────────

    _MODE_HINTS = {
        'select': 'Select mode — click a cell to open it in the Cell tab (on the '
                  'LUT array dead positions open too: write a table to bring '
                  'one to life).',
        'cell':   'Place/Delete mode — click an empty lattice position to add a '
                  'cell, a live cell to remove it.',
        'input':  'Input mode — click cells to toggle them as inputs (A, B, …); '
                  'inputs are the growth seeds and the driven cells in Step/Run.',
        'output': 'Output mode — click a live cell to assign the next output '
                  'role; click an assigned cell to clear it.',
    }

    def _on_mode_change(self, *_a):
        self._status.set(self._MODE_HINTS.get(self._mode.get(), ''))

    def _set_guide(self):
        text = _GUIDE_NERVOUS if self.backend == 'nervous' else _GUIDE_LUT
        self._guide.config(state='normal')
        self._guide.delete('1.0', 'end')
        self._guide.insert('end', text)
        self._guide.config(state='disabled')

    def _on_backend_change(self, _evt=None):
        backend = 'lut' if self._backend_var.get().startswith('LUT') else 'nervous'
        if backend == self.backend:
            return
        self.backend = backend
        self.genome = None
        self.grid, self.in_pos, self.out_pos = {}, [], {}
        self._selected, self._lut_bind = None, None
        self._loaded_target = None
        self._grid_edited = self._genome_edited = False
        self._reset_sim()
        self._refresh_target_list()
        self._refresh_all()

    def _refresh_target_list(self):
        targets = dict(TEMPORAL_TARGETS)
        if self.backend == 'nervous':
            targets.update(COMB_TARGETS)
        loaded = getattr(self, '_loaded_target', None)
        if loaded is not None and loaded.name not in targets:
            targets[loaded.name] = loaded
        self._target_picker.set_targets(targets)

    def _on_target_selected(self):
        target = self._current_target()
        if target is None:
            self._status.set('No scoring target selected.')
        else:
            self._status.set('Selected %s — click Adopt target I/O to place its inputs.'
                             % target.name)
        self._rebuild_input_bar()

    def _current_target(self):
        name = self._target_var.get()
        if name in TEMPORAL_TARGETS:
            return TEMPORAL_TARGETS[name]
        if self.backend == 'nervous' and name in COMB_TARGETS:
            return COMB_TARGETS[name]
        # an imported target whose .name is not a registry KEY (e.g. a custom
        # target, or 'Oscillator (period 2)' vs the key 'Oscillator')
        lt = getattr(self, '_loaded_target', None)
        if lt is not None and name == lt.name:
            return lt
        return None

    def _adopt_target_io(self):
        t = self._current_target()
        if t is None:
            self._status.set('Pick a target first.')
            return
        self.in_pos = list(t.inputs)
        for p in self.in_pos:                     # inputs must exist as cells
            if p not in self.grid:
                self.grid[p] = self._default_cell()
        self.out_pos = {}
        self._grid_edited = True
        self._reset_sim()
        self._refresh_all()
        self._status.set('Adopted %d input cell(s) from %s; outputs auto-place at scoring '
                         '(or assign them in Output mode).' % (len(self.in_pos), t.name))

    def _default_cell(self):
        return NV_SEED_STATE if self.backend == 'nervous' else LUT_SEED_STATE

    # ── genome operations ───────────────────────────────────────────────────────

    def _random_genome(self):
        if self.backend == 'nervous':
            self.genome = random_hex_genome(2)
            note = 'random nervous genome'
        else:
            # LUT seeds are dense sim6-style ontogeny biomorphs (rich, irregular
            # morphology) — the same factory the evolver uses; sparse random
            # genomes just grow uninteresting uniform diamonds
            self._status.set('Growing a sim6-style ontogeny biomorph seed…')
            self.parent.update_idletasks()
            self.genome = make_lut_seed_genome(2)
            note = ('ontogeny biomorph genome (%d genes)'
                    % sum(len(c.genes) for c in self.genome.chromosomes))
        self._genome_edited = True
        self._rebuild_genome_panel()
        self._refresh_sync()
        self._status.set('Loaded a %s — press Grow (ontogeny) to develop it.' % note)

    def _grow(self):
        if self.genome is None:
            self._status.set('No genome — load or randomize one (or design the grid by hand).')
            return
        if not self.in_pos:
            self._status.set('Designate at least one input cell first (Input mode, or '
                             'Adopt target I/O) — inputs are the growth seeds.')
            return
        t = self._current_target()
        if self.backend == 'nervous':
            self.grid = grow_nervous(self.genome, seeds=tuple(self.in_pos))
        else:
            gs = getattr(t, 'grid_size', 7) if t else 7
            it = getattr(t, 'iters', 30) if t else 30
            self.grid = grow_lut(self.genome, seeds=tuple(self.in_pos),
                                 grid_size=gs, iters=it)
        self.out_pos = {r: p for r, p in self.out_pos.items()
                        if _output_is_live(self.grid, p, self.backend)}
        self._grid_edited = self._genome_edited = False
        self._selected = None
        self._reset_sim()
        self._refresh_all()
        self._status.set('Grew %d cells from the genome (seeds pinned at the inputs).'
                         % len(self.grid))

    def _reverse_to_genome(self):
        """Best-effort INVERSE of Grow: reconstruct a genome that develops back
        into the current working grid (see nv_evo/reverse.py, lut_evo/reverse.py).
        Growth is many-to-one, so the reconstruction replays a monotone
        development with the grid as an oracle — it reproduces every cell it can,
        allowing harmless EXTRA cells (the user's contract). The genome is loaded
        into the Genome tab (not yet grown): the working grid is left untouched so
        the hand-built phenotype is preserved; press Grow to realise the genome,
        Score to confirm the outputs still match."""
        if not self.grid:
            self._status.set('Empty grid — build or grow a circuit before reversing '
                             'it to a genome.')
            return
        if not self.in_pos:
            self._status.set('Designate at least one input first (Input mode, or '
                             'Adopt target I/O) — inputs are the growth seeds the '
                             'reconstruction develops from.')
            return
        self._status.set('Reconstructing a genome from the working grid…')
        self.parent.update_idletasks()
        retained = self.genome
        delta = retained is not None and self._grid_edited
        t = self._current_target_obj()
        if self.backend == 'nervous':
            if delta:
                genome, rep = repair_genome_nervous(
                    retained, self.grid, seeds=tuple(self.in_pos))
            else:
                genome, rep = grid_to_genome_nervous(
                    self.grid, seeds=tuple(self.in_pos))
        elif delta:
            genome, rep = repair_genome_lut(
                retained, self.grid, seeds=tuple(self.in_pos),
                grid_size=getattr(t, 'grid_size', 7),
                iters=getattr(t, 'iters', 30))
        else:
            genome, rep = grid_to_genome_lut(
                self.grid, seeds=tuple(self.in_pos),
                grid_size=getattr(t, 'grid_size', 7), iters=getattr(t, 'iters', 30))
        improved = genome is not retained
        self.genome = genome
        self._genome_edited = bool(improved or not delta)
        if t is not None:
            rep['working_behavior'] = self._behavioral_score_for_grid(self.grid, t)
            rep['grown_behavior'] = self._behavioral_score_for_grid(rep['grown'], t)
        self._selected, self._lut_bind = None, None
        self._rebuild_genome_panel()
        self._refresh_sync()
        self._show_report(self._reverse_report(rep))
        n_genes = sum(len(c.genes) for c in genome.chromosomes)
        verb = 'Reconciled' if delta else 'Reconstructed'
        suffix = (', %d/%d edits absorbed' %
                  (rep.get('edits_reproduced', 0), rep.get('edits', 0))
                  if delta else '')
        if delta and not improved and rep.get('edits_reproduced', 0) < rep.get('edits', 0):
            self._status.set(
                'No safe genome delta found — retained the original DNA; %d/%d '
                'edits reproduce without damaging unchanged cells. The hand-edited '
                'grid is still active and can be saved directly.'
                % (rep.get('edits_reproduced', 0), rep.get('edits', 0)))
        else:
            self._status.set(
                '%s a %d-gene genome — reproduces %d/%d cells%s%s%s. '
                'Press Grow to realise it (extra cells may appear); Score to confirm '
                'the outputs. Your hand-built grid is kept until you Grow.'
                % (verb, n_genes, rep['matched'], rep['target'],
                   ', %d extra' % rep['extra'] if rep['extra'] else '',
                   ', %d unreproducible' % (rep['target'] - rep['matched'])
                   if rep['matched'] < rep['target'] else '', suffix))

    def _reverse_report(self, rep):
        """Human-readable account of a network→genome reconstruction."""
        exact = rep['matched'] == rep['target'] and not rep['extra']
        full  = rep['matched'] == rep['target']
        nervous = rep['backend'] == 'nervous'

        def shown_state(value):
            # Reverse synthesis describes gene alleles here. Nervous alleles
            # are individual 4-bit core-circuit selectors, not packed tiles.
            return str(int(value)) if nervous else str(value)

        def shown_context(context):
            return ('(' + ', '.join(shown_state(value) for value in context) + ')'
                    if nervous else str(context))

        L = ['Network → genome (verified reconciliation / synthesis)',
             'Backend: %s' % ('nervous net' if nervous
                               else 'LUT array'), '']
        if rep.get('note'):
            L += [rep['note'], '']
        L += ['Target cells reproduced : %d / %d' % (rep['matched'], rep['target']),
              'Extra cells (harmless)  : %d' % rep['extra'],
              'Unreproducible cells    : %d' % (rep['target'] - rep['matched'])]
        if rep.get('strategy') == 'delta-repair':
            L.append('Retained telomere      : %d' % rep['telomere'])
        else:
            L.append('Growth radius / telomere: %d / %d' %
                     (rep['radius'], rep['telomere']))
        if rep.get('strategy') == 'delta-repair':
            L += ['Baseline target matches : %d / %d' %
                  (rep.get('baseline_matched', 0), rep['target']),
                  'Unchanged cells kept    : %d / %d' %
                  (rep.get('unchanged_preserved', 0), rep.get('unchanged', 0)),
                  'User edits reproduced   : %d / %d' %
                  (rep.get('edits_reproduced', 0), rep.get('edits', 0)),
                  'Repair genes added      : %d' % rep.get('added_genes', 0)]
        if rep.get('strategy'):
            L.append('Synthesis trajectory    : %s' % rep['strategy'])
        if rep.get('intermediate_states'):
            L.append(('Temporary gene values (0-15): %s' if nervous
                      else 'Temporary LUT states    : %s') %
                     ', '.join(shown_state(value)
                               for value in rep['intermediate_states']))
        if rep.get('multistep_collisions'):
            L.append('Collision groups tested : %d' % rep['multistep_collisions'])
        if rep.get('working_behavior') is not None:
            L.append('Edited-grid behavior    : %.4f' % rep['working_behavior'])
        if rep.get('grown_behavior') is not None:
            L.append('Regrown behavior        : %.4f' % rep['grown_behavior'])
        if exact:
            L += ['', '=> EXACT: the genome regrows this circuit cell-for-cell.']
        elif full:
            L += ['', '=> FULL COVER: every target cell is reproduced; the grown '
                  'organism also has %d extra cell(s). Per the design contract '
                  'these are fine as long as they do not change the scored '
                  'outputs — press Grow then Score to confirm.' % rep['extra']]
        else:
            L += ['', '=> PARTIAL: %d cell(s) could not be reproduced by growing '
                  'from the seeds (see below). This is an unresolved reachability '
                  'or context collision in the direct-to-final replay. A multi-step '
                  'developmental program may still do better. Grow + Score to see '
                  'the verified phenotype.' % (rep['target'] - rep['matched'])]
        if rep['conflicts']:
            L += ['', 'Monotone replay collisions (%d): two births share an '
                  'identical neighbourhood but request different final states. '
                  'This direct-to-final reconstruction keeps the first value; '
                  'it is not proof that a multi-step genome is impossible:'
                  % len(rep['conflicts'])]
            for ctx, states in rep['conflicts'][:8]:
                L.append('   %s -> %s' %
                         (shown_context(ctx),
                          ', '.join(shown_state(value) for value in states)))
            if len(rep['conflicts']) > 8:
                L.append('   … %d more' % (len(rep['conflicts']) - 8))
        if rep['seed_mismatch']:
            L += ['', 'Seed states pinned (%d): a seed/input cell is always held at '
                  'the seed state during growth, so a different state there cannot '
                  'be reproduced: %s' % (len(rep['seed_mismatch']),
                                         ', '.join(map(str, rep['seed_mismatch'][:12])))]
        if rep['unreachable']:
            L += ['', 'Unreachable from seeds (%d): no growth path connects these '
                  'to an input, so they cannot develop: %s'
                  % (len(rep['unreachable']),
                     ', '.join(map(str, rep['unreachable'][:12])))]
        return '\n'.join(L)

    def _behavioral_score_for_grid(self, grid, target):
        """Score a candidate phenotype without changing the working Designer grid."""
        if not grid or len(self.in_pos) != len(target.inputs):
            return None
        roles = [output.role for output in target.outputs]
        manual = {role: self.out_pos.get(role) for role in roles}
        use_manual = all(_output_is_live(grid, manual[role], self.backend)
                         for role in roles)
        if getattr(target, 'temporal', False):
            obs = _obs_len(target)
            if self.backend == 'nervous':
                routing = _nv_routing(grid)
                if use_manual:
                    out_pos = manual
                    traces = TemporalTraces({role: [] for role in roles},
                                            events={role: [] for role in roles})
                    for trial in target.trials:
                        _, sampled, raw, overflow = run_nervous_events(
                            grid, routing, list(self.in_pos), out_pos,
                            trial.streams, obs,
                            max_events=getattr(target, 'max_events', 2048))
                        traces.overflow = traces.overflow or overflow
                        for role in roles:
                            traces[role].append(sampled[role])
                            traces.events[role].append(
                                list(raw.get(out_pos[role], ())))
                else:
                    out_pos, traces = place_outputs_by_trace(
                        grid, routing, list(self.in_pos), target)
            else:
                if use_manual:
                    out_pos = manual
                    traces = TemporalTraces({role: [] for role in roles},
                                            events={role: [] for role in roles})
                    for trial in target.trials:
                        sim = LutSim(grid)
                        per = {role: [] for role in roles}
                        for tick in range(obs):
                            row = (trial.streams[tick]
                                   if tick < len(trial.streams) else None)
                            state = sim.step({
                                self.in_pos[i]: (row[i] if row else 0)
                                for i in range(len(self.in_pos))})
                            for role in roles:
                                per[role].append(state.get(out_pos[role], 0))
                        for role in roles:
                            traces[role].append(per[role])
                            traces.events[role].append(sampled_events(per[role]))
                else:
                    out_pos, traces = lut_place_by_trace(
                        grid, list(self.in_pos), target)
            if any(out_pos.get(role) is None for role in roles):
                return None
            return _traces_report(target, traces, out_pos, self.backend)[0]

        if self.backend != 'nervous':
            return None
        routing = _nv_routing(grid)
        out_pos = manual
        if not use_manual:
            _, _, out_pos = interpret_nervous(grid, target)
        if any(out_pos.get(role) is None for role in roles):
            return None
        correct = total = 0
        for in_bits, expected in target.cases:
            invals = {self.in_pos[i]: in_bits[i]
                      for i in range(len(self.in_pos))}
            actual = evaluate_nervous(grid, routing, invals, target.grid_size)
            for i, role in enumerate(roles):
                total += 1
                correct += actual.get(out_pos[role], 0) == expected[i]
        return correct / total if total else None

    def _compact(self):
        """Drop never-expressed genes from the LUT genome (neutral: the grown
        organism is unchanged). Uses the designated inputs as growth seeds so the
        expression tally matches how this genome actually develops here."""
        if self.backend != 'lut':
            self._status.set('Compaction applies to the LUT array (dense genomes).')
            return
        if self.genome is None:
            self._status.set('No genome to compact — load or randomize one first.')
            return
        if not self.in_pos:
            self._status.set('Designate the input cell(s) first — they are the '
                             'growth seeds compaction develops from.')
            return
        t = self._current_target_obj()
        n0 = sum(len(c.genes) for c in self.genome.chromosomes)
        self.genome = lut_compact_genome(
            self.genome, seeds=tuple(self.in_pos),
            grid_size=getattr(t, 'grid_size', 7), iters=getattr(t, 'iters', 30))
        n1 = sum(len(c.genes) for c in self.genome.chromosomes)
        self._genome_edited = True             # genome changed (grid regrows identical)
        self._rebuild_genome_panel()
        self._refresh_sync()
        self._status.set('Compacted %d -> %d genes (%d unexpressed removed; the grown '
                         'circuit is identical).' % (n0, n1, n0 - n1))

    def _clear_grid(self):
        self.grid, self.out_pos = {}, {}
        self.in_pos = []
        self._grid_edited = True
        self._selected = None
        self._reset_sim()
        self._refresh_all()

    # ── genome editor panel ─────────────────────────────────────────────────────

    def _rebuild_genome_panel(self):
        for w in self._genome_frame.winfo_children():
            w.destroy()
        g = self.genome
        if g is None:
            ttk.Label(self._genome_frame,
                      text='(no genome)\n\nLoad evolved…, Random genome, or design\n'
                           'the circuit directly — a hand-built grid\n'
                           'needs no genome to simulate or score.',
                      foreground='#777').pack(anchor='w', padx=6, pady=6)
            return
        nv = self.backend == 'nervous'
        fields = _NV_GENE_FIELDS if nv else _LUT_GENE_FIELDS
        intro = ('Associative memory for ONE core circuit:\n'
                 'context (L, R, D, self) -> output, matched by min Hamming.\n'
                 'Every field is one decimal Figure-3 selector, 0-15. The same\n'
                 'gene is applied independently to tile L/R/D; self=0 grows.'
                 if nv else
                 'Fig. 10 genes: context of five 16-bit LUTs -> new LUT,\n'
                 'min Hamming over all 80 bits; rotated so one gene serves\n'
                 'all four directions. self=0 rows are GROWTH rules.\n'
                 'Click a field\'s ▦ to edit it as a truth table below.')
        ttk.Label(self._genome_frame, text=intro, font=(self._mono, 8),
                  foreground='#556').pack(anchor='w', padx=2, pady=(4, 0))
        for ci, chrom in enumerate(g.chromosomes):
            chrom.split = _valid_split(chrom.genes, chrom.split)
            head = ttk.Frame(self._genome_frame)
            head.pack(fill='x', pady=(6, 0))
            ttk.Label(head, text='Chromosome %s' % chr(97 + ci),
                      font=(self._mono, 9, 'bold')).pack(side='left')
            self._int_spin(head, 'tag', chrom, 'tag', 0, 9999, width=5)
            if len(chrom.genes) < 2:
                ttk.Label(head, text=' split=gene fields',
                          font=(self._mono, 8), foreground='#667').pack(side='left')
            else:
                self._int_spin(head, 'split', chrom, 'split', 1,
                               len(chrom.genes) - 1, width=4)
            self._int_spin(head, 'telomere', chrom, 'telomere', 1,
                           NV_MAX_TEL if nv else LUT_MAX_TEL, width=4)
            ttk.Button(head, text='+gene', width=6,
                       command=lambda c=chrom: self._add_gene(c)).pack(side='left', padx=2)
            ttk.Button(head, text='x', width=2,
                       command=lambda c=chrom: self._del_chrom(c)).pack(side='left')
            hdr = ('   #    Lctx Rctx Dctx self -> out\n'
                   '        [one 4-bit value per field: 0-15]' if nv else
                   '   #    N      E      S      W      self    ->  out')
            ttk.Label(self._genome_frame, text=hdr, font=(self._mono, 8),
                      foreground='#666').pack(anchor='w')
            for gi, gene in enumerate(chrom.genes[:_GENE_ROW_CAP]):
                self._gene_row(chrom, gi, gene, fields, nv)
            if len(chrom.genes) > _GENE_ROW_CAP:
                ttk.Label(self._genome_frame,
                          text='   … %d more genes not shown (evolved dense genome)'
                               % (len(chrom.genes) - _GENE_ROW_CAP),
                          foreground='#a00').pack(anchor='w')
        ttk.Button(self._genome_frame, text='+ chromosome',
                   command=self._add_chrom).pack(anchor='w', pady=4)

    def _gene_row(self, chrom, gi, gene, fields, nv):
        row = ttk.Frame(self._genome_frame)
        row.pack(fill='x')
        ttk.Label(row, text='%4d' % gi, font=(self._mono, 8)).pack(side='left')
        for f in fields:
            if f == 'self_out':                     # the gene reads context -> out
                ttk.Label(row, text='->', font=(self._mono, 8),
                          foreground='#666').pack(side='left', padx=(3, 1))
            if nv:
                self._nv_circuit_entry(row, gene, f)
            else:
                self._hex_entry(row, gene, f)
        ttk.Button(row, text='rnd', width=4,
                   command=lambda c=chrom, i=gi: self._randomize_gene(c, i)).pack(side='left')
        ttk.Button(row, text='dup', width=4,
                   command=lambda c=chrom, i=gi: self._dup_gene(c, i)).pack(side='left')
        ttk.Button(row, text='x', width=2,
                   command=lambda c=chrom, i=gi: self._del_gene(c, i)).pack(side='left')

    def _nv_circuit_entry(self, parent, obj, field):
        """Edit one nervous-gene allele; packed tile words never enter DNA."""
        current = int(getattr(obj, field))
        if not 0 <= current < CIRCUIT_STATE_COUNT:
            # This should only be reachable through a stale in-memory object;
            # encoded files are migrated before HexGene is constructed.
            current = max(0, min(CIRCUIT_STATE_COUNT - 1, current))
            setattr(obj, field, current)
        var = tk.StringVar(value=str(current))

        def valid(candidate):
            return (candidate == '' or
                    (candidate.isdecimal() and
                     0 <= int(candidate) < CIRCUIT_STATE_COUNT))

        def commit(*_a):
            if not var.get():
                var.set(str(getattr(obj, field)))
                return
            value = int(var.get())
            if getattr(obj, field) != value:
                setattr(obj, field, value)
                self._genome_edited = True
                self._refresh_sync()
            var.set(str(value))

        validate = (parent.register(valid), '%P')
        entry = ttk.Spinbox(
            parent, from_=0, to=CIRCUIT_STATE_COUNT - 1,
            textvariable=var, width=3, font=(self._mono, 8),
            validate='key', validatecommand=validate, command=commit)
        entry.bind('<FocusOut>', commit)
        entry.bind('<Return>', commit)
        entry.pack(side='left', padx=1)
        _Tip(entry, 'One 4-bit Figure-3 core-circuit selector (decimal 0-15). '
                    'Growth applies this scalar rule separately to L, R and D; '
                    'the developed tile, not the genome, packs three selectors.')
        return entry

    def _int_spin(self, parent, label, obj, field, lo, hi, width=4):
        if label:
            ttk.Label(parent, text=' %s' % label, font=(self._mono, 8)).pack(side='left')
        var = tk.StringVar(value=str(getattr(obj, field)))

        def commit(*_a):
            try:
                v = max(lo, min(hi, int(var.get())))
            except ValueError:
                return
            if getattr(obj, field) != v:
                setattr(obj, field, v)
                self._genome_edited = True
                self._refresh_sync()
        sp = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=width,
                         command=commit)
        sp.bind('<FocusOut>', commit)
        sp.bind('<Return>', commit)
        sp.pack(side='left', padx=1)
        return sp

    def _hex_entry(self, parent, obj, field):
        """16-bit field as %04X hex; the ▦ button binds it into the shared 4x4
        table editor (click-to-toggle); SOP shown there via lut_sop."""
        var = tk.StringVar(value='%04X' % getattr(obj, field))

        def commit(*_a):
            try:
                v = int(var.get(), 16) & 0xFFFF
            except ValueError:
                var.set('%04X' % getattr(obj, field))
                return
            if getattr(obj, field) != v:
                setattr(obj, field, v)
                self._genome_edited = True
                self._refresh_sync()
                if self._lut_bind and self._lut_bind[0]() == v:
                    self._draw_lut_editor()
        e = ttk.Entry(parent, textvariable=var, width=5, font=(self._mono, 8))
        e.bind('<FocusOut>', commit)
        e.bind('<Return>', commit)
        e.pack(side='left', padx=1)

        def bind_editor(o=obj, f=field, v=var):
            def setter(new):
                setattr(o, f, new)
                v.set('%04X' % new)
                self._genome_edited = True
                self._refresh_sync()
            self._lut_bind = (lambda: getattr(o, f), setter,
                              'gene.%s' % f)
            self._draw_lut_editor()
        ttk.Button(parent, text='▦', width=2, command=bind_editor).pack(side='left')

    def _add_gene(self, chrom):
        chrom.genes.append(random_hex_gene() if self.backend == 'nervous'
                           else random_lut_gene())
        chrom.split = _valid_split(chrom.genes, chrom.split)
        self._genome_edited = True
        self._rebuild_genome_panel()
        self._refresh_sync()

    def _randomize_gene(self, chrom, gi):
        chrom.genes[gi] = (random_hex_gene() if self.backend == 'nervous'
                           else random_lut_gene())
        self._genome_edited = True
        self._rebuild_genome_panel()
        self._refresh_sync()

    def _dup_gene(self, chrom, gi):
        chrom.genes.insert(gi + 1, dataclasses.replace(chrom.genes[gi]))
        chrom.split = _valid_split(chrom.genes, chrom.split)
        self._genome_edited = True
        self._rebuild_genome_panel()
        self._refresh_sync()

    def _del_gene(self, chrom, gi):
        if len(chrom.genes) > 1:
            chrom.genes.pop(gi)
            chrom.split = _valid_split(chrom.genes, chrom.split)
            self._genome_edited = True
            self._rebuild_genome_panel()
            self._refresh_sync()

    def _add_chrom(self):
        if self.genome is None:
            return
        c = (NvChromosome(genes=[random_hex_gene()], split=0, tag=0)
             if self.backend == 'nervous'
             else LutChromosome(genes=[random_lut_gene()], split=0, tag=0))
        self.genome.chromosomes.append(c)
        self._genome_edited = True
        self._rebuild_genome_panel()
        self._refresh_sync()

    def _del_chrom(self, chrom):
        if self.genome and len(self.genome.chromosomes) > 1:
            self.genome.chromosomes.remove(chrom)
            self._genome_edited = True
            self._rebuild_genome_panel()
            self._refresh_sync()

    # ── shared LUT table editor ─────────────────────────────────────────────────

    def _draw_lut_editor(self):
        self._lut_ax.clear()
        if self._lut_bind is None:
            self._lut_ax.axis('off')
            self._lut_ax.text(0.5, 0.5, '(no LUT bound)', ha='center', va='center',
                              fontsize=8, color='#888')
        else:
            get, _set, label = self._lut_bind
            v = get()
            draw_lut_table(self._lut_ax, v,
                           title='%s = %04X\n%s' % (label, v, lut_sop(v)))
        self._lut_canvas.draw_idle()

    def _on_lut_click(self, event):
        if self._lut_bind is None or event.inaxes is not self._lut_ax:
            return
        if event.xdata is None:
            return
        c, rs = int(math.floor(event.xdata)), int(math.floor(event.ydata))
        if not (0 <= c < 4 and 0 <= rs < 4):
            return
        r = 3 - rs                                   # draw_lut_table row layout
        idx = (c & 1) | ((c >> 1) & 1) << 1 | (r & 1) << 2 | ((r >> 1) & 1) << 3
        get, setter, _ = self._lut_bind
        setter((get() ^ (1 << idx)) & 0xFFFF)
        self._draw_lut_editor()

    # ── circuit editor (canvas clicks + cell inspector) ─────────────────────────

    def _cell_at(self, xd, yd):
        if xd is None:
            return None
        if self.backend == 'lut':
            p = (int(round(xd)), int(round(yd)))
            return p if abs(xd - p[0]) < 0.45 and abs(yd - p[1]) < 0.45 else None
        # nervous: search the lattice neighbourhood around the click
        cand = set(self.grid) | set(self.in_pos)
        xs = [x for x, _ in cand] or [0]
        ys = [y for _, y in cand] or [0]
        best, best_d = None, 0.45
        for x in range(min(xs) - 3, max(xs) + 4):
            for y in range(min(ys) - 3, max(ys) + 4):
                px, py = hex_pixel(x, y)
                d = math.hypot(px - xd, py - yd)
                if d < best_d:
                    best_d, best = d, (x, y)
        return best

    def _on_canvas_click(self, event):
        if event.inaxes is not self._ax_grid:
            return
        pos = self._cell_at(event.xdata, event.ydata)
        if pos is None:
            return
        mode = self._mode.get()
        if mode == 'select':
            # the LUT field is UNIFORM hardware: every position holds the same
            # four LUTs, so an empty position is inspectable too (its tables are
            # all zero — writing a non-zero table brings it to life)
            if self.backend == 'lut':
                self._selected = pos
            else:
                self._selected = pos if pos in self.grid else None
            self._rebuild_inspector()
        elif mode == 'cell':
            if pos in self.grid:
                del self.grid[pos]
                if pos in self.in_pos:
                    self.in_pos.remove(pos)
                self.out_pos = {
                    r: p for r, p in self.out_pos.items()
                    if (channel_tile(p) if self.backend == 'nervous' else p) != pos}
                if self._selected == pos:
                    self._selected = None
                    self._rebuild_inspector()
            else:
                self.grid[pos] = self._default_cell()
            self._mark_grid_edit()
        elif mode == 'input':
            if pos in self.in_pos:
                self.in_pos.remove(pos)
            else:
                if pos not in self.grid:
                    self.grid[pos] = self._default_cell()
                self.in_pos.append(pos)
            self._mark_grid_edit()
        elif mode == 'output':
            if pos not in self.grid:
                return
            held = [r for r, p in self.out_pos.items()
                    if (channel_tile(p) if self.backend == 'nervous' else p) == pos]
            if held:
                if self.backend == 'nervous':
                    role = held[0]
                    current = normalize_output_channel(self.grid, self.out_pos[role])
                    index = DIRECTIONS.index(current[2])
                    if index + 1 < len(DIRECTIONS):
                        self.out_pos[role] = (pos[0], pos[1], DIRECTIONS[index + 1])
                        self._status.set("Output '%s' now reads %s channel; click "
                                         "again to cycle." %
                                         (role, DIRECTIONS[index + 1]))
                    else:
                        del self.out_pos[role]
                else:
                    for r in held:
                        del self.out_pos[r]
            else:
                t = self._current_target()
                roles = [o.role for o in t.outputs] if t else ['Q']
                free = [r for r in roles if self.out_pos.get(r) is None]
                if not free:
                    self._status.set('All output roles assigned — click an assigned '
                                     'cell to clear its role.')
                    return
                self.out_pos[free[0]] = (tile_channel(pos, 'L')
                                         if self.backend == 'nervous' else pos)
                if self.backend == 'nervous':
                    self._status.set("Output '%s' reads %s channel; click the tile "
                                     "to cycle L/R/D or clear." %
                                     (free[0], self.out_pos[free[0]][2]))
            self._mark_grid_edit()
        self._refresh_canvas()
        self._rebuild_input_bar()
        self._refresh_sync()

    def _mark_grid_edit(self):
        self._grid_edited = True
        self._reset_sim()

    def _set_lut_dir(self, pos, k, v):
        """Write one directional table of the LUT cell at `pos`. The paper's
        field is UNIFORM hardware — every position always contains the same four
        LUTs and only their CONTENTS change — so this is the single mutation
        point: writing a non-zero table to a dead position brings that cell to
        life, and zeroing the last non-zero table kills it (the sparse grid
        stores only live cells; absent == all four tables 0). Returns True if
        the cell's live/dead status flipped."""
        st = list(self.grid.get(pos, (0, 0, 0, 0)))
        was_live = pos in self.grid
        st[k] = v & 0xFFFF
        if any(st):
            self.grid[pos] = tuple(st)
        elif was_live:
            del self.grid[pos]
            if pos in self.in_pos:
                self.in_pos.remove(pos)
                self._rebuild_input_bar()
            self.out_pos = {r: p for r, p in self.out_pos.items() if p != pos}
            self._status.set('All four tables are 0 — the cell at %s is dead '
                             '(same hardware, zero contents).' % (pos,))
        self._mark_grid_edit()
        self._refresh_canvas()
        self._refresh_sync()
        return was_live != (pos in self.grid)

    def _rebuild_inspector(self):
        for w in self._insp_body.winfo_children():
            w.destroy()
        pos = self._selected
        # LUT positions are ALWAYS inspectable (uniform hardware, contents may
        # be zero); nervous cells only when live.
        ok = pos is not None and (pos in self.grid or self.backend == 'lut')
        if not ok:
            ttk.Label(self._insp_body, text='(click a cell in Select mode)',
                      foreground='#777').pack(anchor='w')
            return
        live = pos in self.grid
        ttk.Label(self._insp_body, text='Cell %s%s' % (pos, '' if live
                                                       else '   [dead: tables all 0]'),
                  font=(self._mono, 10, 'bold')).pack(anchor='w')
        if self.backend == 'nervous':
            self._nv_inspector(pos)
        else:
            self._lut_inspector(pos)
        self._nb.select(self._insp_tab)

    def _nv_inspector(self, pos):
        x, y = pos
        state = list(unpack_tile_state(self.grid[pos]))
        nb = hex_dirs(x, y)
        orient = 'up-oriented' if (x + y) % 2 == 0 else 'down-oriented'
        ttk.Label(self._insp_body,
                  text='%s tile — its L/R/D resolve to (Fig. 4 rotation):\n'
                       '  L->%s   R->%s   D->%s'
                       % (orient, nb['L'], nb['R'], nb['D']),
                  font=(self._mono, 8), justify='left').pack(anchor='w', pady=(2, 4))
        ttk.Label(self._insp_body,
                   text='Three independent L/R/D output circuits; each computes\n'
                        '(E1 AND E2) AND NOT I1 using one Figure-3 configuration '\
                        '(0-F).\nE1=E2 means one nerve fanned into both required terminals:',
                  font=(self._mono, 8), justify='left').pack(anchor='w')
        picker = ttk.Frame(self._insp_body)
        picker.pack(fill='x', pady=(3, 1))
        ttk.Label(picker, text='Edit output circuit:',
                  font=(self._mono, 8)).pack(side='left')
        direction_var = tk.StringVar(value='L')
        direction_box = ttk.Combobox(
            picker, textvariable=direction_var, values=DIRECTIONS,
            state='readonly', width=3, font=(self._mono, 8))
        direction_box.pack(side='left', padx=4)
        routes = ttk.Frame(self._insp_body)
        routes.pack(fill='both', expand=True)
        lb = tk.Listbox(routes, height=16, font=(self._mono, 8),
                        exportselection=False)
        vscroll = ttk.Scrollbar(routes, orient='vertical', command=lb.yview)
        hscroll = ttk.Scrollbar(routes, orient='horizontal', command=lb.xview)
        lb.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        for i in range(len(PAPER_ROUTING)):
            lb.insert('end', _route_text(i, x, y))
        lb.selection_set(state[0])
        lb.see(state[0])
        lb.grid(row=0, column=0, sticky='nsew')
        vscroll.grid(row=0, column=1, sticky='ns')
        hscroll.grid(row=1, column=0, sticky='ew')
        routes.rowconfigure(0, weight=1)
        routes.columnconfigure(0, weight=1)

        def on_pick(_evt=None):
            sel = lb.curselection()
            if not sel:
                return
            selectors = list(unpack_tile_state(self.grid[pos]))
            selectors[DIRECTIONS.index(direction_var.get())] = sel[0]
            self.grid[pos] = pack_tile_state(*selectors)
            self._mark_grid_edit()
            self._refresh_canvas()
            self._refresh_sync()

        def on_direction(_evt=None):
            selectors = unpack_tile_state(self.grid[pos])
            selected = selectors[DIRECTIONS.index(direction_var.get())]
            lb.selection_clear(0, 'end')
            lb.selection_set(selected)
            lb.see(selected)

        def on_hover(evt):
            i = lb.nearest(evt.y)
            if 0 <= i < len(PAPER_ROUTING):
                self._refresh_canvas(preview=(pos, direction_var.get(), i))
        direction_box.bind('<<ComboboxSelected>>', on_direction)
        lb.bind('<<ListboxSelect>>', on_pick)
        lb.bind('<Motion>', on_hover)
        lb.bind('<Leave>', lambda e: self._refresh_canvas())

    def _lut_inspector(self, pos):
        state = list(self.grid.get(pos, (0, 0, 0, 0)))
        ttk.Label(self._insp_body,
                  text='Every cell is the SAME hardware: four 16-bit LUTs,\n'
                       'one driving each output direction; each is a truth\n'
                       'table over the 4 bits received from the neighbours\n'
                       '(N%s S%s E%s W%s).\n'
                       'Only the CONTENTS differ between cells: all four\n'
                       'tables 0 = dead; write a non-zero table to bring\n'
                       'this cell to life. Edit as hex, or ▦ opens the\n'
                       'table below (click squares to toggle bits).'
                       % ((pos[0], pos[1] + 1), (pos[0], pos[1] - 1),
                          (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])),
                  font=(self._mono, 8), justify='left').pack(anchor='w', pady=(2, 4))
        for k, d in enumerate(_LUT_DIRS):
            row = ttk.Frame(self._insp_body)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text='%s out' % d, width=6,
                      font=(self._mono, 9)).pack(side='left')
            var = tk.StringVar(value='%04X' % state[k])
            e = ttk.Entry(row, textvariable=var, width=6, font=(self._mono, 9))
            e.pack(side='left', padx=2)
            sop = ttk.Label(row, text=lut_sop(state[k]), font=(self._mono, 8),
                            foreground='#555')
            sop.pack(side='left', padx=4)

            def make(k=k, var=var, sop=sop):
                def setter(v):
                    flipped = self._set_lut_dir(pos, k, v)     # uniform-hardware
                    cur = self.grid.get(pos, (0, 0, 0, 0))[k]  # semantics live here
                    var.set('%04X' % cur)
                    sop.config(text=lut_sop(cur))
                    if flipped:                    # live<->dead: retitle inspector
                        self.parent.after_idle(self._rebuild_inspector)

                def on_commit(_e=None):
                    try:
                        setter(int(var.get(), 16))
                    except ValueError:
                        var.set('%04X' % self.grid.get(pos, (0, 0, 0, 0))[k])

                def bind_editor():
                    self._lut_bind = (lambda: self.grid.get(pos, (0, 0, 0, 0))[k],
                                      setter, 'cell%s %s-out' % (pos, _LUT_DIRS[k]))
                    self._draw_lut_editor()
                return on_commit, bind_editor
            on_commit, bind_editor = make()
            e.bind('<FocusOut>', on_commit)
            e.bind('<Return>', on_commit)
            ttk.Button(row, text='▦', width=2, command=bind_editor).pack(side='left')

    # ── canvas rendering ────────────────────────────────────────────────────────

    def _refresh_canvas(self, preview=None):
        ax = self._ax_grid
        activity = None
        if self._tick > 0:
            activity = self._nibbles if self.backend == 'lut' else self._state
        title = ('%s  —  %s' % (self.backend,
                 circuit_summary_nervous(self.grid) if self.backend == 'nervous'
                 else '%d cells' % len(self.grid)))
        if self.backend == 'nervous':
            routing = _nv_routing(self.grid)
            if preview:
                entries = list(routing[preview[0]])
                entries[DIRECTIONS.index(preview[1])] = PAPER_ROUTING[preview[2]]
                routing[preview[0]] = tuple(entries)
            draw_hex_net(ax, self.grid, 7, routing=routing, in_pos=self.in_pos,
                         out_pos=self.out_pos, activity=activity,
                         show_edges=True, title=title)
            if self._selected and self._selected in self.grid:
                px, py = hex_pixel(*self._selected)
                ax.add_patch(Circle((px, py), radius=0.36, fill=False,
                                    edgecolor='#e6a817', lw=2.2, zorder=6))
            if preview:
                exc, veto = _routed_channels(preview[2], *preview[0])
                destination = tile_channel(preview[0], preview[1])
                px, py = channel_pixel(destination)
                ax.add_patch(Circle((px, py), radius=0.14, fill=False,
                                    edgecolor='#e6a817', lw=2.0, zorder=7))
                for source in exc:
                    if channel_tile(source) in self.grid:
                        px, py = channel_pixel(source)
                        ax.add_patch(Circle((px, py), radius=0.14, fill=False,
                                             edgecolor='#2e8b57', lw=2.0, zorder=6))
                if veto and channel_tile(veto) in self.grid:
                    px, py = channel_pixel(veto)
                    ax.add_patch(Circle((px, py), radius=0.14, fill=False,
                                         edgecolor='#d0332e', lw=2.0, zorder=6))
        else:
            draw_lut_net(ax, self.grid, activity=activity, in_pos=self.in_pos,
                         out_pos=self.out_pos, show_edges=True, title=title)
            if activity is None and self.grid:
                # the field is UNIFORM hardware (paper): every position holds
                # the same four LUTs, dead ones just contain zeros — show the
                # dead ring around the organism as dashed cells so "editing"
                # reads as changing contents, not placing hardware
                for (x, y) in self.grid:
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        p = (x + dx, y + dy)
                        if p not in self.grid:
                            ax.add_patch(Rectangle((p[0] - 0.4, p[1] - 0.4),
                                                   0.8, 0.8, fill=False,
                                                   linestyle=':', lw=0.7,
                                                   edgecolor='#c9cfd6', zorder=1))
            if self._selected:
                x, y = self._selected
                ax.add_patch(Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, fill=False,
                                       edgecolor='#e6a817', lw=2.2, zorder=6))
        if self.grid:
            ax.legend(handles=self._legend_handles(activity is not None),
                      loc='upper left', fontsize=6, framealpha=0.85,
                      borderpad=0.4, handlelength=1.2)
        else:
            ax.clear()
            ax.axis('off')
            ax.text(0.5, 0.5, 'empty grid — Place/Delete mode to add cells,\n'
                              'or Grow (ontogeny) a genome', ha='center', va='center',
                    fontsize=10, color='#888')
        self._draw_trace_strip()
        self.canvas.draw_idle()

    def _legend_handles(self, showing_activity):
        """Canvas legend in the paper's own colour conventions (nervous circuit
        kinds + wire colours; LUT Fig. 13 state colours / Fig. 14 red-green)."""
        if self.backend == 'nervous':
            if showing_activity:
                nodes = [Patch(color='#18b34a', label='pulsing this tick'),
                         Patch(color='#e6e9ee', label='quiet')]
            else:
                nodes = [Patch(color='#8fb3e0', label='buffer (one nerve -> E1+E2)'),
                         Patch(color='#2f6fc0', label='coincidence (E1·E2)'),
                         Patch(color='#e0902e', label='inhibited (has I1)'),
                         Patch(color='#e8e8e8', label='off circuit')]
            return nodes + [
                Line2D([0], [0], color='#2e8b57', lw=1.7, label='excitatory wire'),
                Line2D([0], [0], color='#d0332e', lw=1.7,
                       label='inhibitory veto (T-bar)'),
                Patch(facecolor='white', edgecolor='#b02020', label='input (seed)')]
        if showing_activity:
            return [Patch(color='#1ea64a', label='output bit 1 (Fig. 14)'),
                    Patch(color='#d64a3c', label='output bit 0 (Fig. 14)')]
        return [Patch(color='#7ec8e3', label='wedge colour = LUT state (Fig. 13)'),
                Patch(color=(0.95, 0.96, 0.97), label='dead direction (LUT = 0)'),
                Patch(facecolor='white', edgecolor='#c9cfd6', linestyle=':',
                      label='same hardware, tables all 0 (dead)'),
                Patch(facecolor='white', edgecolor='#b02020', label='input (seed)'),
                Patch(facecolor='white', edgecolor='#17902f', label='output cell')]

    def _draw_trace_strip(self):
        axt = self._ax_trace
        axt.clear()
        roles = list(self.out_pos)
        # nervous playback: output pulse EDGES on a real-time axis
        if (self.backend == 'nervous' and getattr(self, '_player', None) is not None
                and getattr(self, '_nv_playing', False)):
            axt.set_title('output pulse edges (real time)', fontsize=9)
            for k, role in enumerate(roles):
                cell = self.out_pos.get(role)
                evs = self._player.events_upto(cell) if cell else []
                if evs:
                    axt.vlines(evs, k - 0.4, k + 0.4, color='#1a6fd0', lw=1.6)
            axt.axvline(self._player.cursor, color='#e8a33d', lw=1.4, alpha=0.9)
            axt.set_xlim(0, self._player.horizon)
            axt.set_ylim(-0.6, max(1, len(roles)) - 0.4)
            axt.set_yticks(range(len(roles)))
            axt.set_yticklabels([r[:6] for r in roles], fontsize=8)
            axt.set_xlabel('continuous time (tick units)', fontsize=8)
            return
        axt.set_title('output traces', fontsize=9)
        for k, role in enumerate(roles):
            tr = self._traces.get(role, [])
            if tr:
                axt.step(range(len(tr)), [v + k * 1.4 for v in tr],
                         where='post', lw=1.4, label=role)
        axt.set_ylim(-0.3, 1.4 * max(1, len(roles)))
        axt.set_xlabel('tick', fontsize=8)
        axt.set_yticks([])
        if any(self._traces.get(r) for r in roles):
            axt.legend(fontsize=7, loc='upper right')

    def _refresh_sync(self):
        if self.genome is None:
            txt, col = 'no genome (hand-built grid)', '#777777'
        elif self._grid_edited:
            txt, col = 'grid EDITED since grow — genome not updated', '#b06010'
        elif self._genome_edited:
            txt, col = 'genome edited — press Grow to regrow', '#b06010'
        else:
            txt, col = 'genome -> grid: in sync', '#177a2f'
        self._sync_lbl.config(text=txt, foreground=col)

    def _refresh_all(self):
        if hasattr(self, '_compact_btn'):      # compaction is LUT-only
            self._compact_btn.configure(state='normal' if self.backend == 'lut'
                                        else 'disabled')
        self._set_guide()
        self._rebuild_genome_panel()
        self._rebuild_inspector()
        self._rebuild_input_bar()
        self._draw_lut_editor()
        self._refresh_canvas()
        self._refresh_sync()

    # ── simulation (mirrors interactive.py) ─────────────────────────────────────

    def _rebuild_input_bar(self):
        # The nervous net is asynchronous: drive it with a clickable pulse
        # TIMELINE (real, possibly sub-tick times). The LUT is clocked, so it
        # keeps the tick-numbered bit STREAM editor.
        old = [v.get() for v in getattr(self, '_stream_vars', [])]
        if getattr(self, '_editor', None) is not None:
            self._editor.disconnect()
            self._editor = None
        for w in self._inbar.winfo_children():
            w.destroy()
        for w in self._input_actions.winfo_children():
            w.destroy()
        self._stream_vars = []
        if self.backend == 'nervous':
            self._input_heading.set('Input pulses (continuous time):')
            self._build_pulse_timeline()
            return
        self._input_heading.set('Input streams (clock ticks):')
        for i, _p in enumerate(self.in_pos):
            lbl = chr(65 + i) if i < 26 else 'i%d' % i
            ttk.Label(self._inbar, text=lbl, font=(self._mono, 9)).pack(side='left', padx=(4, 1))
            v = tk.StringVar(value=old[i] if i < len(old) else '')
            ttk.Entry(self._inbar, textvariable=v, width=16,
                      font=(self._mono, 9)).pack(side='left')
            self._stream_vars.append(v)
        if self.in_pos:
            ttk.Label(self._inbar, text='test', font=(self._mono, 8),
                      foreground='#888').pack(side='left', padx=(6, 1))
            ttk.Spinbox(self._inbar, from_=1, to=99, width=3,
                        textvariable=self._trial_idx).pack(side='left')
            ttk.Button(self._inbar, text='From test', width=9,
                       command=self._streams_from_trial).pack(side='left', padx=(2, 1))
            ttk.Button(self._inbar, text='Load', width=5,
                       command=self._load_streams).pack(side='left')
            ttk.Button(self._inbar, text='Save', width=5,
                       command=self._save_streams).pack(side='left')

    # ── nervous net: asynchronous pulse timeline + continuous-time playback ───────

    def _sim_horizon(self):
        t = self._current_target()
        return float(max(24, getattr(t, 'T', 24) or 24))

    def _build_pulse_timeline(self):
        labels = [chr(65 + i) if i < 26 else 'i%d' % i for i in range(len(self.in_pos))]
        # Keep the timeline legible when the surrounding Designer pane expands.
        # Width follows the pane; height reserves enough room for lane labels,
        # title, ticks, and click targets instead of collapsing to an 80 px strip.
        height = max(1.45, 0.85 + 0.34 * max(1, len(labels)))
        self._tl_fig = plt.Figure(figsize=(8.0, height))
        self._tl_ax = self._tl_fig.add_subplot(111)
        self._tl_canvas = FigureCanvasTkAgg(self._tl_fig, master=self._inbar)
        self._tl_canvas.get_tk_widget().pack(fill='both', expand=True)
        self._editor = PulseLaneEditor(self._tl_ax, self._tl_canvas,
                                       labels or ['(no inputs)'],
                                       horizon=self._sim_horizon(), snap=0.5,
                                       on_change=self._nv_schedule_changed)
        t = self._current_target()
        if t is not None and self.in_pos:
            self._editor.set_pulses(pulses_from_trial(t, len(self.in_pos)))
        else:
            self._editor.draw()
        ttk.Button(self._input_actions, text='Clear pulses',
                   command=lambda: self._editor.clear()).pack(side='right', padx=3)
        try:
            self._tl_fig.subplots_adjust(left=0.055, right=0.99,
                                         top=0.78, bottom=0.30)
        except Exception:
            pass

    def _build_nv_player(self):
        routing = _nv_routing(self.grid)
        self._player = NervousPlayer(
            self.grid, routing, horizon=self._sim_horizon(),
            max_events=getattr(self.target, 'max_events', 2048),
            config=getattr(self.target, 'pulse_config', None))
        sched = (self._editor.schedule(self.in_pos)
                 if getattr(self, '_editor', None) is not None else {})
        self._player.set_schedule(sched)
        self._traces = {r: [] for r in self.out_pos}

    def _nv_schedule_changed(self):
        # a pulse edit invalidates the running playback; rebuild from t=0
        self._running = False
        if hasattr(self, '_run_btn'):
            self._run_btn.config(text='Run')
        self._player = None
        self._nv_playing = False
        self._tick = 0
        if hasattr(self, '_tick_lbl'):
            self._tick_lbl.config(text='t = 0.0')
        self._state = {}
        self._refresh_canvas()

    def _stream_bit(self, i, tick):
        if i >= len(self._stream_vars):
            return 0
        s = self._stream_vars[i].get()
        j = 0
        for ch in s:                       # ignore spaces/separators; count 0/1 only
            if ch in '01':
                if j == tick:
                    return int(ch)
                j += 1
        return 0

    def _streams_from_trial(self):
        t = self._current_target()
        if t is None or not getattr(t, 'trials', None):
            self._status.set('Pick a temporal target first — it supplies the input tests.')
            return
        if len(self.in_pos) != t.n_inputs:
            self._adopt_target_io()          # align inputs to the target (rebuilds bar)
        ti = max(1, min(self._trial_idx.get(), len(t.trials))) - 1
        trial = t.trials[ti]
        for i in range(min(len(self._stream_vars), t.n_inputs)):
            self._stream_vars[i].set(''.join(str(trial.streams[k][i]) for k in range(t.T)))
        self._reset_sim_ui()
        self._status.set('Loaded test %d/%d input streams (%d ticks) — press Run.'
                         % (ti + 1, len(t.trials), t.T))

    def _load_streams(self):
        path = filedialog.askopenfilename(
            title='Load input streams (one 0/1 line per input)',
            filetypes=[('text', '*.txt *.csv'), ('all files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                lines = [ln.strip() for ln in f
                         if ln.strip() and not ln.strip().startswith('#')]
        except Exception as exc:
            messagebox.showerror('Load streams', str(exc))
            return
        def clean(ln):                       # accept "A: 0010" or bare "0010"
            body = ln.split(':', 1)[1] if ':' in ln else ln
            return ''.join(ch for ch in body if ch in '01')
        streams = [clean(ln) for ln in lines]
        for i, v in enumerate(self._stream_vars):
            v.set(streams[i] if i < len(streams) else '')
        self._reset_sim_ui()
        self._status.set('Loaded %d input stream(s) from %s.'
                         % (len(streams), os.path.basename(path)))

    def _save_streams(self):
        if not self._stream_vars:
            self._status.set('No inputs to save streams for.')
            return
        path = filedialog.asksaveasfilename(
            title='Save input streams', defaultextension='.txt',
            initialfile='streams.txt', filetypes=[('text', '*.txt'), ('all files', '*.*')])
        if not path:
            return
        lines = ['# one 0/1 input stream per line, tick 0 first (label is a comment)']
        for i, v in enumerate(self._stream_vars):
            lbl = chr(65 + i) if i < 26 else 'i%d' % i
            lines.append('%s: %s' % (lbl, ''.join(ch for ch in v.get() if ch in '01')))
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as exc:
            messagebox.showerror('Save streams', str(exc))
            return
        self._status.set('Saved %d input stream(s) → %s'
                         % (len(self._stream_vars), path))

    def _reset_sim(self):
        self._running = False
        self._sim = None
        self._player = None
        self._nv_playing = False
        self._tick = 0
        self._state, self._nibbles = {}, {}
        self._traces = {r: [] for r in self.out_pos}
        if self.backend == 'nervous' and getattr(self, '_editor', None) is not None:
            self._editor.set_cursor(0.0)
        if hasattr(self, '_run_btn'):
            self._run_btn.config(text='Run')
        if hasattr(self, '_tick_lbl'):
            self._tick_lbl.config(
                text='t = 0.0' if self.backend == 'nervous' else 'tick 0')

    def _reset_sim_ui(self):
        self._reset_sim()
        self._refresh_canvas()

    def _step(self):
        if not self.grid:
            self._status.set('Empty grid — nothing to simulate.')
            return
        if self.backend == 'nervous':
            self._step_nervous_async()
            return
        if self._sim is None:                    # LUT — synchronous, clocked
            self._sim = LutSim(self.grid)
            self._traces = {r: [] for r in self.out_pos}
        invals = {p: self._stream_bit(i, self._tick)
                  for i, p in enumerate(self.in_pos)}
        self._state = self._sim.step(invals)
        self._nibbles = dict(self._sim.out)
        self._tick += 1
        for role, p in self.out_pos.items():
            self._traces.setdefault(role, []).append(self._state.get(p, 0))
            if len(self._traces[role]) > 60:
                self._traces[role] = self._traces[role][-60:]
        self._tick_lbl.config(text='tick %d' % self._tick)
        self._refresh_canvas()

    def _step_nervous_async(self):
        if getattr(self, '_player', None) is None:
            self._build_nv_player()
        if self._running and self._player.at_end():
            self._player.reset()                 # loop while Playing
        else:
            self._player.step()
        self._nv_playing = True
        self._tick = 1                            # marks "simulating" for _refresh_canvas
        self._state = self._player.activity()
        if getattr(self, '_editor', None) is not None:
            self._editor.set_cursor(self._player.cursor)
        self._tick_lbl.config(text='t = %.1f%s' % (
            self._player.cursor, '  (event cap)' if self._player.overflow else ''))
        self._refresh_canvas()

    def _toggle_run(self):
        if self._running:
            self._running = False
            self._run_btn.config(text='Run')
        else:
            self._running = True
            self._run_btn.config(text='Pause')
            self._tick_loop()

    def _tick_loop(self):
        if not self._running:
            return
        self._step()
        self.parent.after(70 if self.backend == 'nervous' else 180, self._tick_loop)

    def close(self):
        """Release playback callbacks and matplotlib event bindings."""
        self._running = False
        if getattr(self, '_editor', None) is not None:
            self._editor.disconnect()
            self._editor = None

    # ── scoring ─────────────────────────────────────────────────────────────────

    def _score(self):
        t = self._current_target()
        if t is None:
            self._status.set('Pick a target to score against.')
            return
        if not self.grid:
            self._status.set('Empty grid — nothing to score.')
            return
        if getattr(t, 'temporal', False):
            self._score_temporal(t)
        elif self.backend == 'nervous':
            self._score_combinational(t)
        else:
            self._status.set('The LUT backend scores temporal targets only '
                             '(it cannot settle to combinational logic).')

    def _score_temporal(self, t):
        if len(self.in_pos) != t.n_inputs:
            self._status.set('Target needs %d input(s); the grid has %d designated. '
                             'Use Input mode or Adopt target I/O.'
                             % (t.n_inputs, len(self.in_pos)))
            return
        roles = [o.role for o in t.outputs]
        manual = {r: self.out_pos.get(r) for r in roles}
        use_manual = all(_output_is_live(self.grid, manual[r], self.backend)
                         for r in roles)
        obs = _obs_len(t)                    # observe past T so a delayed Q is seen
        if self.backend == 'nervous':
            routing = _nv_routing(self.grid)
            if use_manual:
                out_pos = manual
                traces = TemporalTraces({r: [] for r in roles},
                                        events={r: [] for r in roles})
                for trial in t.trials:
                    _, trs, raw, overflow = run_nervous_events(
                        self.grid, routing, list(self.in_pos), out_pos,
                        trial.streams, obs,
                        max_events=getattr(t, 'max_events', 2048))
                    traces.overflow = traces.overflow or overflow
                    for r in roles:
                        traces[r].append(trs[r])
                        traces.events[r].append(list(raw.get(out_pos[r], ())))
            else:
                out_pos, traces = place_outputs_by_trace(self.grid, routing,
                                                         list(self.in_pos), t)
        else:
            if use_manual:
                out_pos = manual
                traces = TemporalTraces({r: [] for r in roles},
                                        events={r: [] for r in roles})
                for trial in t.trials:
                    sim = LutSim(self.grid)
                    per = {r: [] for r in roles}
                    ns = len(trial.streams)
                    for tick in range(obs):
                        row = trial.streams[tick] if tick < ns else None
                        st = sim.step({self.in_pos[i]: (row[i] if row else 0)
                                       for i in range(len(self.in_pos))})
                        for r in roles:
                            per[r].append(st.get(out_pos[r], 0))
                    for r in roles:
                        traces[r].append(per[r])
                        traces.events[r].append(sampled_events(per[r]))
            else:
                out_pos, traces = lut_place_by_trace(self.grid, list(self.in_pos), t)
        if any(out_pos.get(r) is None for r in roles):
            self._status.set('Could not place all outputs — grid too small?')
            return
        score, report = _traces_report(t, traces, out_pos, self.backend)
        self._last_score = score
        self._show_report(report)
        self._status.set('Behavioural score %.4f against %s  (%s outputs%s)'
                         % (score, t.name,
                            'manual' if use_manual else 'trace-matched auto',
                            '' if use_manual else ' — assign roles in Output mode to pin them'))

    def _score_combinational(self, t):
        if len(self.in_pos) != len(t.inputs):
            self._status.set('Target needs %d input(s); the grid has %d designated.'
                             % (len(t.inputs), len(self.in_pos)))
            return
        routing = _nv_routing(self.grid)
        roles = [o.role for o in t.outputs]
        out_pos = {r: self.out_pos.get(r) for r in roles}
        if (not all(_output_is_live(self.grid, out_pos[r], 'nervous')
                    for r in roles)):
            _, _, out_pos = interpret_nervous(self.grid, t)
        if any(out_pos.get(r) is None for r in roles):
            self._status.set('Could not place all outputs.')
            return
        lines = ['Target: %s   [nervous — designer working grid]' % t.name,
                 'Circuit: ' + circuit_summary_nervous(self.grid)]
        for r in roles:
            lines.append("  out '%s': pos=%s" % (r, out_pos[r]))
        in_hdr = ' '.join('i%d' % i for i in range(len(t.inputs)))
        out_hdr = ' '.join('%s:e/a' % r for r in roles)
        lines += ['', '  %s | %s | result' % (in_hdr, out_hdr),
                  '  ' + '-' * (len(in_hdr) + len(out_hdr) + 14)]
        correct = total = 0
        for in_bits, out_bits in t.cases:
            invals = {self.in_pos[i]: in_bits[i] for i in range(len(self.in_pos))}
            outs = evaluate_nervous(self.grid, routing, invals, t.grid_size)
            cells, row_ok = [], True
            for i, r in enumerate(roles):
                act = outs.get(out_pos[r], 0)
                ok = act == out_bits[i]
                row_ok = row_ok and ok
                total += 1
                correct += 1 if ok else 0
                cells.append('%d/%d' % (out_bits[i], act))
            in_str = ' '.join(str(b) for b in in_bits).ljust(len(in_hdr))
            out_str = ' '.join(c.ljust(len('%s:e/a' % r))
                               for c, r in zip(cells, roles))
            lines.append('  %s | %s | %s' % (in_str, out_str,
                                             'PASS' if row_ok else 'FAIL'))
        fit = correct / total if total else 0.0
        lines += ['', '  => %d/%d checks  (fitness = %.4f)%s'
                  % (correct, total, fit, '   ALL PASS' if correct == total else '')]
        self._last_score = fit
        self._show_report('\n'.join(lines))
        self._status.set('Fitness %.4f against %s' % (fit, t.name))

    def _show_report(self, text):
        self._report.delete('1.0', 'end')
        self._report.insert('end', text)
        self._nb.select(self._report_tab)

    # ── import / export ─────────────────────────────────────────────────────────

    def _load_evolved(self):
        path = filedialog.askopenfilename(
            initialdir=RESULTS_DIR if os.path.isdir(RESULTS_DIR) else ROOT,
            title='Load design or evolved solution',
            filetypes=[('design or solution', '*.json *.pkl'),
                       ('text design/checkpoint', '*.json'), ('legacy pickle', '*.pkl'),
                       ('all files', '*.*')])
        if not path:
            return
        # text (JSON) design? try to parse before falling back to pickle
        try:
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except (UnicodeDecodeError, ValueError, OSError):
            doc = None
        if isinstance(doc, dict) and str(doc.get('format', '')).startswith('evohw-design'):
            self._load_json_design(doc, path)
            return
        try:
            state = read_saved_file(path)
        except Exception as exc:
            messagebox.showerror('Load', 'Could not read %s:\n%s' % (path, exc))
            return
        if not isinstance(state, dict):
            messagebox.showerror('Load', 'Not a recognised solution/design file:\n%s\n\n'
                                 'Expected a saved genome or design document '
                                 'from the evolver or this designer.' % path)
            return
        if 'genomes' in state:                                    # solver_generation.json
            self._pick_solver(state)
            return
        if state.get('best_genome') is None and not state.get('grid'):
            messagebox.showerror('Load', 'No genome or circuit found in:\n%s' % path)
            return
        backend = state.get('backend', 'snn')
        if backend not in ('nervous', 'lut'):
            messagebox.showinfo('Load', 'This file is a %s-backend solution; the '
                                'designer handles only the nervous and LUT '
                                'architectures.\n\n(Open SNN runs in the main app.)'
                                % backend.upper())
            return
        self._adopt_state(genome=state.get('best_genome'),
                          target=state.get('target'),
                          backend=backend,
                          grid=state.get('grid'),
                          in_pos=state.get('in_pos'),
                          out_pos=state.get('out_pos'),
                          grid_edited=state.get('grid_edited', True),
                          note='loaded %s (fitness %.4f)'
                               % (os.path.basename(path),
                                  state.get('best_fitness', 0.0) or 0.0))

    def _pick_solver(self, state):
        genomes = state.get('genomes') or []
        backend = state.get('backend', 'nervous')
        if backend not in ('nervous', 'lut'):
            messagebox.showinfo('Load', 'This solver set is for the %s backend, '
                                'which the designer does not handle.' % backend.upper())
            return
        if not genomes:
            messagebox.showinfo('Load', 'That solver file contains no genomes.')
            return
        top = tk.Toplevel(self.parent)
        top.title('Pick a solver (%d unique valid genomes)' % len(genomes))
        top.transient(self.parent)
        lb = tk.Listbox(top, width=44, height=min(20, max(1, len(genomes))),
                        font=(self._mono, 9))
        for i, g in enumerate(genomes):
            n = sum(len(c.genes) for c in g.chromosomes)
            lb.insert('end', 'genome %2d — %d chromosomes, %d genes'
                      % (i, len(g.chromosomes), n))
        lb.selection_set(0)
        lb.pack(padx=6, pady=6)

        def ok():
            sel = lb.curselection()
            top.destroy()
            if sel:
                self._adopt_state(genome=genomes[sel[0]], target=state.get('target'),
                                  backend=backend,
                                  note='solver %d of %d (all ≥ %.3f)'
                                       % (sel[0], len(genomes),
                                          state.get('valid', 0.0)))
        ttk.Button(top, text='Load', command=ok).pack(pady=(0, 6))

    def _from_app(self):
        c = self.get_circuit() if self.get_circuit else None
        if not c or c.get('genome') is None:
            self._status.set('No current solution in the app — Run or Load Saved first.')
            return
        backend = c.get('backend', 'snn')
        if backend not in ('nervous', 'lut'):
            self._status.set('The app\'s current solution is a %s circuit; the '
                             'designer handles only the nervous and LUT '
                             'architectures.' % backend.upper())
            return
        self._adopt_state(genome=c['genome'], target=c.get('target'),
                          backend=backend, note='current app solution')

    def _adopt_state(self, genome, target, backend, grid=None, in_pos=None,
                     out_pos=None, grid_edited=True, note=''):
        if backend not in ('nervous', 'lut'):
            self._status.set('Designer supports the nervous and LUT backends '
                             '(loaded file is %r).' % backend)
            return
        self.backend = backend
        self._backend_var.set('LUT array (Arch 2)' if backend == 'lut'
                              else 'Nervous net (Arch 1)')
        self._refresh_target_list()
        self.genome = genome
        if target is not None:
            self._loaded_target = target
            # select by registry KEY when one holds this target (keys and .name
            # differ, e.g. key 'Oscillator' vs name 'Oscillator (period 2)');
            # otherwise offer the imported target under its own name
            regs = dict(TEMPORAL_TARGETS)
            if backend == 'nervous':
                regs.update(COMB_TARGETS)
            key = next((k for k, v in regs.items() if v.name == target.name), None)
            if key is None:
                self._refresh_target_list()
            self._target_picker.select(key or target.name)
        self._genome_edited = False
        if grid:
            # a saved design: restore the GRID as-is — a hand-edited phenotype
            # must never be silently replaced by a regrow of the genome
            self.grid = dict(grid)
            self.in_pos = list(in_pos or [])
            self.out_pos = dict(out_pos or {})
            if backend == 'nervous':
                self.out_pos = {
                    role: normalize_output_channel(self.grid, pos)
                    for role, pos in self.out_pos.items() if pos}
            self._grid_edited = bool(grid_edited)
        else:
            self.in_pos = list(target.inputs) if target is not None else []
            self.out_pos = {}
            if genome is not None and self.in_pos:
                if backend == 'nervous':
                    self.grid = grow_nervous(genome, seeds=tuple(self.in_pos))
                else:
                    self.grid = grow_lut(genome, seeds=tuple(self.in_pos),
                                         grid_size=getattr(target, 'grid_size', 7),
                                         iters=getattr(target, 'iters', 30))
                self._grid_edited = False
            else:
                self.grid = {}
        self._selected, self._lut_bind = None, None
        self._reset_sim()
        self._refresh_all()
        self._status.set('Imported %s — %d cells.' % (note, len(self.grid)))

    def _current_target_obj(self):
        t = self._current_target()
        if t is None and getattr(self, '_loaded_target', None) is not None:
            return self._loaded_target
        return t

    def _save_design(self):
        if not self.grid and self.genome is None:
            self._status.set('Nothing to save — design a circuit or load a genome first.')
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=RESULTS_DIR, title='Save design (text/JSON)',
            defaultextension='.json', initialfile='design.json',
            filetypes=[('design (text)', '*.json'), ('all files', '*.*')])
        if not path:
            return
        t = self._current_target_obj()
        grid_edited = bool(self._grid_edited or self.genome is None)
        try:
            genome_doc = (_genome_to_dict(self.genome, self.backend)
                          if self.genome else None)
        except (TypeError, ValueError) as exc:
            messagebox.showerror(
                'Save design', 'Genome is outside the supported encoding:\n%s' % exc)
            return
        doc = {
            'format':      DESIGN_FORMAT,
            'backend':     self.backend,
            'state_encoding': (TILE_ENCODING
                               if self.backend == 'nervous' else None),
            'target':      t.name if t is not None else '',
            'fitness':     self._last_score if self._last_score is not None else 0.0,
            'hand_built':  True,
            'grid_edited': grid_edited,
            'genome':      genome_doc,
            'grid':        _grid_to_dict(self.grid, self.backend),
            'inputs':      [list(p) for p in self.in_pos],
            'outputs':     {r: list(p) for r, p in self.out_pos.items() if p},
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=1)
        except Exception as exc:
            messagebox.showerror('Save design', 'Could not write %s:\n%s' % (path, exc))
            return
        kind = ('genome + hand-edited grid' if grid_edited and self.genome
                else 'grid only (no genome)' if self.genome is None
                else 'genome + grown grid')
        self._status.set('Saved editable design (%s) → %s' % (kind, path))

    def _load_json_design(self, doc, path):
        backend = doc.get('backend', 'nervous')
        if backend not in ('nervous', 'lut'):
            messagebox.showinfo('Load', 'Design is for the %s backend, not supported.'
                                % str(backend).upper())
            return
        try:
            genome = _genome_from_dict(doc['genome'], backend) if doc.get('genome') else None
            grid = _grid_from_dict(
                doc.get('grid', {}), backend,
                doc.get('state_encoding') or
                (doc.get('genome') or {}).get('state_encoding'))
            in_pos = [tuple(p) for p in doc.get('inputs', [])]
            out_pos = {r: tuple(p) for r, p in (doc.get('outputs') or {}).items()}
        except (KeyError, ValueError, TypeError) as exc:
            messagebox.showerror('Load', 'Malformed design file %s:\n%s'
                                 % (os.path.basename(path), exc))
            return
        name = doc.get('target') or ''
        target = TEMPORAL_TARGETS.get(name)
        if target is None and backend == 'nervous':
            target = COMB_TARGETS.get(name)
        self._adopt_state(genome=genome, target=target, backend=backend,
                          grid=grid or None, in_pos=in_pos, out_pos=out_pos,
                          grid_edited=doc.get('grid_edited', True),
                          note='text design %s' % os.path.basename(path))

    def _load_design(self):
        self._load_evolved()          # one reader auto-detects JSON design vs pickle


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    root.title('Evolvable Hardware — Manual Designer')
    ui_compat.apply_theme(root)
    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)
    designer = DesignerTab(frame)

    def close():
        designer.close()
        ui_compat.cancel_after_callbacks(root)
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    ui_compat.fit_window(root, min_width=960, min_height=640)
    root.mainloop()


if __name__ == '__main__':
    main()
