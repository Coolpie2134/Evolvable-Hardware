"""
lut_evo/ga.py — scoring and GA for the LUT array.

Problem definitions (TemporalTarget / snn Target) and the trace-scoring maths
(windowed, level-balanced, phase-tolerant) are shared with the rest of the
project; only the substrate differs. Selection uses the same recipe as the
nervous GA — elites + random immigrants + tournament, NO early stop at 1.0 —
but WITHOUT the parsimony tie-break: seeds are dense sim6-style ontogeny
biomorphs (rich morphology), and parsimony pruned them back to sparse diamonds.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from evo_runtime.cache import LRUCache
from evo_runtime.mutation import (STRESS_MAX_MULT, STRESS_PATIENCE,
                                  adaptive_mutation_rate)
from evo_runtime.parallel import map_ordered

from nv_evo.temporal import (_pr_counts, _f1,
                             _best_shift, _obs_len, _output_candidates,
                             windowed_score, exact_tick_accuracy,
                             period_stepper_score, TemporalTraces,
                             event_score, _best_event_shift,
                             _event_case_score, _expected_events, _role_events,
                             cadence_score, score_temporal_bundle,
                             _score_output_candidate, trial_input_summary,
                             expected_window_summary, event_list_summary)
from .genome import (LUT_STATES, MAX_CHROMS, MAX_TELOMERE,
                     Genome, Chromosome,
                     random_lut_gene, random_lut_chromosome)
from .lut import grow_lut, grow_lut_tracked
from .pulse import AsyncLutSim


def clone_genome(genome):
    """Fast structural copy (shared, never-in-place-mutated gene objects; fresh
    structure) — an identical-behaviour, ~10x cheaper replacement for
    copy.deepcopy on the reproduction hot path. See nv_evo.ga.clone_genome."""
    return Genome(
        chromosomes=[Chromosome(genes=c.genes[:], split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE))
                     for c in genome.chromosomes],
        tag=genome.tag)

POPSIZE        = 120
ELITE_FRAC     = 0.10        # elites = this fraction of pop, UNLESS ELITE_COUNT set
ELITE_COUNT    = None        # exact elite count (GUI override); None = use ELITE_FRAC
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
EXPLORATION_PARENT_FRAC = 0.30
MEAN_MUTATIONS = 4.0         # hot-start rate for annealing (see nv_evo.ga)
MUT_DECAY      = 0.997       # slow cooldown: 4.0 -> ~0.89 by gen 500
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))  # see nv_evo.ga
FITNESS_CACHE_MAX = 200_000  # cap the fitness cache on very long runs

# LUT temporal search has the same flat recurrent landscapes as the nervous
# net; reheat after a plateau instead of cooling indefinitely.
# ── running trials / placing outputs (trace-matched, as in nv) ──────────────────
# Output candidates are the OUT_RADIUS cells nearest each terminal, via the
# shared nv_evo.temporal._output_candidates (same radius, same tie-breaks).


def _run_lut_trials(grid, in_pos, ttarget, watch_cells):
    """Run every trial once on the asynchronous engine (lut_evo.pulse).

    Returns (trial_B, trial_events, overflow): the [obs, ncells] mid-tick
    sample matrix per trial (empty when the score mode consumes edges, not
    samples), the continuous leading-edge trains of the ``watch_cells`` per
    trial, and whether any run blew its event budget. A trial that carries a
    physical ``input_events`` schedule is injected at its real (possibly
    sub-tick) times — the LUT backend no longer quantises such targets."""
    obs = _obs_len(ttarget)                  # observe past T to catch delayed events
    mode = getattr(ttarget, 'score_mode', 'trace')
    need_samples = mode not in ('events', 'cadence')
    sim = AsyncLutSim(grid, config=getattr(ttarget, 'lut_config', None))
    trial_B, trial_events, overflow = [], [], False
    first = True
    for tr in ttarget.trials:
        if not first:
            sim.reset()
        first = False
        events = getattr(tr, 'input_events', None)
        if events is not None:
            B = sim.run_input_events(events, in_pos, obs, sample=need_samples)
        else:
            B = sim.run_bits(tr.streams, in_pos, obs)
        trial_B.append(B)
        trial_events.append(sim.rise_trains(watch_cells))
        overflow = overflow or sim.overflow
    return sim, trial_B, trial_events, overflow


def place_outputs_by_trace(grid, in_pos, ttarget):
    """{role: cell}, traces — each role read at the live non-input cell nearest
    its terminal whose trace best matches the expectation across all trials
    (terminal-distance tie-break), exactly like the nervous backend.

    The dynamics are the asynchronous engine (lut_evo.pulse.AsyncLutSim):
    trace/persistence modes score the mid-tick sample matrix (a column slice
    per candidate cell — no per-tick dicts), while event/cadence modes read
    the wires' raw continuous leading-edge timestamps, exactly as the nervous
    backend does."""
    in_set = set(in_pos)
    out_pos = {t.role: None for t in ttarget.outputs}
    traces  = TemporalTraces()
    # LUT holds a level (a latch), it does not ring like the nervous net — so a
    # commanded HOLD must be genuinely high on every tick, not satisfied by a
    # sparse 2-3 tick burst under the nervous net's ±1 ring tolerance. Mark these
    # traces strict (hold_tol=0) so scoring + placement demand the full hold.
    traces.hold_tol = 0
    if len(grid) <= len(in_set):
        return out_pos, traces
    mode = getattr(ttarget, 'score_mode', 'trace')
    cands_by_role = {term.role: _output_candidates(grid, in_set, term)
                     for term in ttarget.outputs}
    watch = sorted(set().union(*cands_by_role.values()))
    sim, trial_B, trial_events, overflow = _run_lut_trials(
        grid, in_pos, ttarget, watch)
    traces.overflow = overflow
    cidx = sim._cidx
    need_samples = mode not in ('events', 'cadence')
    used = set()
    for term in ttarget.outputs:
        best, best_key, best_aux = None, None, None
        score_cache = {}
        for c in cands_by_role[term.role]:
            if c in used:
                continue
            col = cidx[c]
            ctr, cevents, cexp = [], [], []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                ctr.append(trial_B[ti][:, col].tolist() if need_samples else [])
                cevents.append(list(trial_events[ti].get(c, ())))
                cexp.append(exp)
            if not ctr:
                s, aux = 0.0, None
            else:
                source = cevents if mode in ('events', 'cadence') else ctr
                signature = tuple(tuple(seq) for seq in source)
                cached = score_cache.get(signature)
                s, aux = cached if cached is not None else (None, None)
            if ctr and s is None:
                s, aux = _score_output_candidate(
                    ctr, cevents, cexp, term.role, ttarget, traces.overflow,
                    tol=0)                              # LUT: strict hold
            if ctr:
                score_cache[signature] = (s, aux)
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best, best_aux = key, c, aux
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        col = cidx[best]
        traces[term.role]  = [trial_B[ti][:, col].tolist() if need_samples else []
                              for ti in range(len(ttarget.trials))]
        traces.events[term.role] = [list(trial_events[ti].get(best, ()))
                                    for ti in range(len(ttarget.trials))]
        if len(ttarget.outputs) == 1 and mode == 'events':
            traces._event_result = best_aux
        elif len(ttarget.outputs) == 1 and mode == 'cadence':
            traces._cadence_result = best_aux
    return out_pos, traces


def trace_fixed_outputs(grid, in_pos, out_pos, ttarget):
    """Run LUT trials at pre-selected output cells without validation search."""
    if any(pos not in grid for pos in out_pos.values()):
        return None
    mode = getattr(ttarget, 'score_mode', 'trace')
    need_samples = mode not in ('events', 'cadence')
    watch = sorted(set(out_pos.values()))
    sim, trial_B, trial_events, overflow = _run_lut_trials(
        grid, in_pos, ttarget, watch)
    traces = TemporalTraces(overflow=overflow)
    traces.hold_tol = 0                        # LUT holds strictly (see place_outputs_by_trace)
    for role, pos in out_pos.items():
        col = sim._cidx[pos]
        traces[role] = [B[:, col].tolist() if need_samples else []
                        for B in trial_B]
        traces.events[role] = [list(ev.get(pos, ())) for ev in trial_events]
    return traces


def prepare_lut(genome, ttarget):
    grid = grow_lut(genome, seeds=tuple(ttarget.inputs),
                    grid_size=ttarget.grid_size, iters=ttarget.iters)
    if len(grid) <= ttarget.n_inputs:
        return None
    if any(p not in grid for p in ttarget.inputs):
        return None
    out_pos, traces = place_outputs_by_trace(grid, list(ttarget.inputs), ttarget)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, out_pos, traces


def score_lut_temporal(genome, ttarget):
    prep = prepare_lut(genome, ttarget)
    if prep is None:
        return 0.0
    return score_temporal_bundle(prep[2], ttarget)[0]


def _place_outputs_combinational(grid, target):
    """{role: cell|None} — each output read at the nearest live non-input cell to
    its terminal (distinct cells; None if the grid runs out)."""
    in_set    = set(target.inputs)
    non_input = [p for p in sorted(grid) if p not in in_set]
    out_pos, used = {}, set()
    for term in target.outputs:
        cands = [p for p in non_input if p not in used]
        if not cands:
            out_pos[term.role] = None
            continue
        best = min(cands, key=lambda p: abs(p[0] - term.pos[0]) + abs(p[1] - term.pos[1]))
        used.add(best)
        out_pos[term.role] = best
    return out_pos


# The LUT array is a SYNCHRONOUS RECURRENT cellular automaton: every cell both
# reads and drives all four neighbours, so with inputs held it generally does NOT
# come to rest — it oscillates or goes chaotic ("most circuits produce immediate
# sustained activity ... chaotic behavior", paper §6). Reading a combinational
# answer at one fixed tick therefore samples a single phase of that oscillation
# and rewards phase-luck, not computation. A real combinational result requires
# the output to reach a FIXED POINT: hold the same value over a settling window.
# If it never settles the case is simply wrong (the array isn't computing a stable
# function of its inputs). This is the difference between a genuine logic circuit
# and a chaotic oscillator that happens to line up at one sampled tick.
SETTLE_WINDOW = 5


def _settled_outputs(grid, target, out_pos, in_bits, window=SETTLE_WINDOW,
                     horizon=None):
    """Hold `in_bits`, run, and return ({role: settled_bit|None}, final_sim).
    A role's value is the output cell's bit ONLY if it held constant over the
    last `window` ticks (a fixed point); otherwise None (never settled)."""
    steps  = horizon or (4 * target.grid_size + 12)
    invals = {p: in_bits[i] for i, p in enumerate(target.inputs)}
    sim    = AsyncLutSim(grid)
    tails  = {r: [] for r in out_pos if out_pos[r] is not None}
    for _ in range(steps):
        bits = sim.step(invals)
        for r, tail in tails.items():
            tail.append(bits.get(out_pos[r], 0))
            if len(tail) > window:
                tail.pop(0)
    settled = {r: (tail[-1] if len(set(tail)) == 1 else None)
               for r, tail in tails.items()}
    return settled, sim


def score_lut_combinational(genome, target):
    """Hold input levels, run until the output reaches a FIXED POINT, and read
    the settled bit at each output cell. An output that never settles (still
    oscillating) counts as wrong — see the note above on why single-tick reads
    on this recurrent substrate are phase-luck rather than computation."""
    grid = grow_lut(genome, seeds=tuple(target.inputs),
                    grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0
    out_pos = _place_outputs_combinational(grid, target)
    if any(out_pos[t.role] is None for t in target.outputs):
        return 0.0
    n_checks = len(target.cases) * len(target.outputs)
    if n_checks == 0:
        return 0.0
    correct = 0
    for in_bits, out_bits in target.cases:
        settled, _ = _settled_outputs(grid, target, out_pos, in_bits)
        for i, term in enumerate(target.outputs):
            if settled.get(term.role) == out_bits[i]:   # None != 0/1 -> unstable fails
                correct += 1
    return correct / n_checks


def lut_case_outputs(genome, target):
    """Grow, place outputs, and run each combinational case to the horizon.
    Returns (grid, out_pos, cases); each case carries the settled node activity
    (`node_outputs` 0/1 and `node_nibbles` 4-bit N/S/E/W) so the GUI can draw the
    array's response. The LUT analogue of nv_evo.nervous_case_outputs."""
    grid = grow_lut(genome, seeds=tuple(target.inputs),
                    grid_size=target.grid_size, iters=target.iters)
    out_pos = _place_outputs_combinational(grid, target)
    cases = []
    if (len(grid) <= target.n_inputs
            or any(out_pos[t.role] is None for t in target.outputs)):
        return grid, out_pos, cases
    for in_bits, out_bits in target.cases:
        settled, sim = _settled_outputs(grid, target, out_pos, in_bits)
        acts = {t.role: settled.get(t.role) for t in target.outputs}    # bit | None
        node_out = {c: (1 if sim.out[c] else 0) for c in grid}
        cases.append({'in_bits': in_bits, 'out_bits': out_bits,
                      'node_outputs': node_out, 'node_nibbles': dict(sim.out),
                      'acts': acts,
                      'stable': all(v is not None for v in acts.values())})
    return grid, out_pos, cases


def lut_truth_table(genome, target):
    """Per-case truth-table report for a combinational target on the LUT array
    (mirrors nv_evo.nervous_truth_table)."""
    grid, out_pos, cases = lut_case_outputs(genome, target)
    lines = ['Target: %s   [LUT array]' % target.name,
             'Circuit: %d live cells  (square array, four 16-bit LUTs/cell,'
             % len(grid),
             '          asynchronous level logic — paper Architecture 2)']
    for term in target.outputs:
        p = out_pos.get(term.role)
        lines.append("  out '%s': %s" % (term.role,
                     ('read at %s' % (p,)) if p else '(not found)'))
    if not cases:
        lines += ['', '(circuit incomplete — grew too little or outputs missing)']
        return '\n'.join(lines)
    lines += ['', 'Inputs held; output read only if it SETTLES to a fixed point',
              "('?' = never settled — the recurrent array is still oscillating).", '']
    in_hdr  = ' '.join('i%d' % i for i in range(len(target.inputs)))
    out_hdr = ' '.join('%s:e/a' % t.role for t in target.outputs)
    lines += ['  %s | %s | result' % (in_hdr, out_hdr),
              '  ' + '-' * (len(in_hdr) + len(out_hdr) + 14)]
    correct = total = 0
    for case in cases:
        cells, row_ok = [], True
        for i, term in enumerate(target.outputs):
            act = case['acts'][term.role]; exp = case['out_bits'][i]
            ok = act == exp; row_ok = row_ok and ok
            total += 1; correct += 1 if ok else 0
            cells.append('%d/%s' % (exp, '?' if act is None else act))
        in_str  = ' '.join(str(b) for b in case['in_bits']).ljust(len(in_hdr))
        out_str = ' '.join(c.ljust(len('%s:e/a' % t.role))
                           for c, t in zip(cells, target.outputs))
        lines.append('  %s | %s | %s' % (in_str, out_str, 'PASS' if row_ok else 'FAIL'))
    fit = correct / total if total else 0.0
    lines += ['', '  => %d/%d checks  (fitness = %.4f)%s'
              % (correct, total, fit, '   ALL PASS' if correct == total else '')]
    return '\n'.join(lines)


def evaluate_lut_full(genome, target):
    """(scalar fitness, per-case vector) — cases are per (trial, role) traces,
    the units ε-lexicase streams over (None for combinational targets)."""
    if not getattr(target, 'temporal', False):
        return score_lut_combinational(genome, target), None
    n_cases = sum(len(tr.expected) for tr in target.trials)
    prep = prepare_lut(genome, target)
    if prep is None:
        return 0.0, (0.0,) * n_cases
    _, _, traces = prep
    score, cases, _ = score_temporal_bundle(traces, target)
    return score, cases


def evaluate_lut(genome, target):
    return evaluate_lut_full(genome, target)[0]


def genome_signature(genome):
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0),
         tuple((g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in, g.self_out)
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_cases(genomes, target, cache=None, executor=None,
                     should_stop=None, on_progress=None):
    """(fitnesses, case_vectors) in parallel; cache holds (fit, cases). A
    persistent `executor` is reused instead of spawning a fresh pool per call
    (large speed-up on Windows); omitting it keeps the one-shot-pool behaviour.
    `should_stop`/`on_progress` are threaded to map_ordered so a run stays
    cancellable without a chunk barrier."""
    out  = [None] * len(genomes)
    todo = list(range(len(genomes)))
    if cache is not None and len(cache) > FITNESS_CACHE_MAX:
        cache.clear()                      # bound memory on very long runs
    if cache is not None:
        sigs = [genome_signature(g) for g in genomes]
        todo = [i for i in todo if sigs[i] not in cache]
        for i in range(len(genomes)):
            if sigs[i] in cache:
                out[i] = cache[sigs[i]]
    if todo:
        fn = partial(evaluate_lut_full, target=target)
        if cache is not None:
            unique = {}
            for i in todo:
                unique.setdefault(sigs[i], []).append(i)
            representatives = [(sig, indices[0]) for sig, indices in unique.items()]
        else:
            representatives = [(i, i) for i in todo]
            unique = {i: [i] for i in todo}
        subset = [genomes[i] for _, i in representatives]
        if executor is not None:
            results = map_ordered(executor, fn, subset, should_stop, on_progress)
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                results = map_ordered(ex, fn, subset, should_stop, on_progress)
        for (sig, _), r in zip(representatives, results):
            for i in unique[sig]:
                out[i] = r
            if cache is not None:
                cache[sig] = r
    return [r[0] for r in out], [r[1] for r in out]


def eval_batch_lut(genomes, target, cache=None):
    return eval_batch_cases(genomes, target, cache)[0]


# ── genetic operators ────────────────────────────────────────────────────────────

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_n", "ctx_e", "ctx_s", "ctx_w", "self_in", "self_out"]


def _normalize_split(chromosome):
    """Keep split on a real between-gene boundary (or zero for one gene)."""
    count = len(chromosome.genes)
    chromosome.split = (0 if count < 2 else
                        max(1, min(int(chromosome.split), count - 1)))


def _recombine_gene_fields(gene_a, gene_b):
    """Uniformly recombine a single LUT rule's active alleles."""
    differing = [field for field in _GENE_FIELDS
                  if getattr(gene_a, field) != getattr(gene_b, field)]
    if len(differing) < 2:
        return gene_a, gene_b
    exchanged = set(random.sample(
        differing, random.randint(1, len(differing) - 1)))
    child_a, child_b = copy.copy(gene_a), copy.copy(gene_b)
    for field in exchanged:
        value_a, value_b = getattr(gene_a, field), getattr(gene_b, field)
        setattr(child_a, field, value_b)
        setattr(child_b, field, value_a)
    return child_a, child_b


def _recombination_signature(genome):
    """Alleles crossover can actually exchange (not slot/object identity)."""
    return tuple(
        tuple(tuple(getattr(gene, field) for field in _GENE_FIELDS)
              for gene in chromosome.genes)
        for chromosome in genome.chromosomes)


def _other_lut(value):
    """Choose a genuinely different 16-bit LUT value."""
    return (value + random.randrange(1, LUT_STATES)) % LUT_STATES


_SOFT_LUT_MASKS = tuple(
    sum(1 << bit for bit in bits)
    for width in (1, 2, 3)
    for bits in combinations(range(16), width))
_SOFT_LUT_MASK_INDEX = {mask: index for index, mask in enumerate(_SOFT_LUT_MASKS)}


def _soft_lut_excluding(value, parent_value):
    """Flip 1-3 bits without returning either current or parental LUT."""
    forbidden = (None if parent_value is None else value ^ parent_value)
    forbidden_index = _SOFT_LUT_MASK_INDEX.get(forbidden)
    if forbidden_index is None:
        return value ^ random.choice(_SOFT_LUT_MASKS)
    pick = random.randrange(len(_SOFT_LUT_MASKS) - 1)
    if pick >= forbidden_index:
        pick += 1
    return value ^ _SOFT_LUT_MASKS[pick]


def _force_nonparent_tweak(genome, parent):
    """End a multi-edit transaction at an allele distinct from its parent."""
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(random_lut_chromosome())
        else:
            random.choice(genome.chromosomes).genes.append(random_lut_gene())
        with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                      if chromosome.genes]
    ci = random.choice(with_genes)
    gi = random.randrange(len(genome.chromosomes[ci].genes))
    field = random.choice(_GENE_FIELDS)
    gene = copy.copy(genome.chromosomes[ci].genes[gi])
    parent_value = None
    if (ci < len(parent.chromosomes)
            and gi < len(parent.chromosomes[ci].genes)):
        parent_value = getattr(parent.chromosomes[ci].genes[gi], field)
    setattr(gene, field, _soft_lut_excluding(
        getattr(gene, field), parent_value))
    genome.chromosomes[ci].genes[gi] = gene


def _tweak_gene(gene):
    """Soft mutation: usually flip 1-3 bits of one 16-bit field (a nearby
    boolean function), sometimes replace the field entirely. self_in has an
    extra chance of snapping to 0 — growth rules must stay reachable."""
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and g.self_in != 0 and random.random() < 0.2:
        g.self_in = 0
    elif random.random() < 0.5:
        v = getattr(g, fld)
        for bit in random.sample(range(16), random.randint(1, 3)):
            v ^= 1 << bit
        setattr(g, fld, v)
    else:
        setattr(g, fld, _other_lut(getattr(g, fld)))
    return g


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]


def _mutate_once_lut(genome, max_telomere=MAX_TELOMERE,
                     chromosome_count=None):
    """Apply one feasible, state-changing mutation to a LUT genome."""
    if not genome.chromosomes:
        genome.chromosomes.append(random_lut_chromosome())
        return

    chroms = genome.chromosomes
    with_genes = [chromosome for chromosome in chroms if chromosome.genes]
    options = []
    if with_genes:
        options.append('tweak')
    if any(chromosome.genes and len(chromosome.genes) < _gene_cap()
           for chromosome in chroms):
        options.append('duplicate')
    if any(len(chromosome.genes) < _gene_cap() for chromosome in chroms):
        options.append('add_gene')
    if any(len(chromosome.genes) > 1 for chromosome in chroms):
        options.append('del_gene')
    if chromosome_count is None and len(chroms) < MAX_CHROMS:
        options.append('add_chrom')
    if chromosome_count is None and len(chroms) > 1:
        options.append('del_chrom')
    split_options = []
    for chromosome in chroms:
        values = [value for value in range(1, len(chromosome.genes))
                  if value != chromosome.split]
        if values:
            split_options.append((chromosome, values))
    if split_options:
        options.append('split')
    telomere_options = []
    for chromosome in chroms:
        base = getattr(chromosome, 'telomere', 10)
        values = [base + delta for delta in (-3, -2, -1, 1, 2, 3)
                  if 1 <= base + delta <= max_telomere]
        if values:
            telomere_options.append((chromosome, values))
    if telomere_options:
        options.append('telomere')

    weights = [_MUT_WEIGHTS[_MUT_OPS.index(op)] for op in options]
    op = random.choices(options, weights=weights)[0]
    if op == 'tweak':
        chromosome = random.choice(with_genes)
        index = random.randrange(len(chromosome.genes))
        chromosome.genes[index] = _tweak_gene(chromosome.genes[index])
    elif op == 'duplicate':
        chromosome = random.choice([
            item for item in chroms
            if item.genes and len(item.genes) < _gene_cap()
        ])
        chromosome.genes.insert(
            random.randrange(len(chromosome.genes) + 1),
            _tweak_gene(random.choice(chromosome.genes)))
    elif op == 'add_gene':
        chromosome = random.choice([
            item for item in chroms if len(item.genes) < _gene_cap()
        ])
        chromosome.genes.append(random_lut_gene())
    elif op == 'del_gene':
        chromosome = random.choice([
            item for item in chroms if len(item.genes) > 1
        ])
        chromosome.genes.pop(random.randrange(len(chromosome.genes)))
    elif op == 'add_chrom':
        chroms.append(random_lut_chromosome())
    elif op == 'del_chrom':
        chroms.remove(min(chroms, key=lambda item: len(item.genes)))
    elif op == 'split':
        chromosome, values = random.choice(split_options)
        chromosome.split = random.choice(values)
    else:  # telomere
        chromosome, values = random.choice(telomere_options)
        chromosome.telomere = random.choice(values)


def mutate_lut(genome, mean_mutations=None, max_telomere=MAX_TELOMERE,
               chromosome_count=None):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    g = clone_genome(genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    events = max(1, _poisson(lam))
    for _ in range(events - 1):
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count)
    if events == 1:
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count)
    else:
        _force_nonparent_tweak(g, genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    return g


def crossover_lut(pa, pb):
    """Tag-matched hierarchical crossover, including one-rule chromosomes."""
    ca, cb = clone_genome(pa), clone_genome(pb)
    used_b = set()
    for i, chrom_a in enumerate(ca.chromosomes):
        best_j, best_dist = None, float("inf")
        for j, chrom_b in enumerate(cb.chromosomes):
            if j in used_b:
                continue
            d = abs(chrom_a.tag - chrom_b.tag)
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is None:
            continue
        used_b.add(best_j)
        chrom_b = cb.chromosomes[best_j]
        # Snapshot both inputs: chrom_a aliases child A, so assigning A before
        # reading its reciprocal suffix used to make child B a parent-B clone.
        genes_a, genes_b = chrom_a.genes[:], chrom_b.genes[:]
        common = min(len(genes_a), len(genes_b))
        if common >= 2:
            sp = max(1, min(int(chrom_a.split), common - 1))
            ca.chromosomes[i].genes = genes_a[:sp] + genes_b[sp:]
            cb.chromosomes[best_j].genes = genes_b[:sp] + genes_a[sp:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = sp
        elif common == 1:
            gene_a, gene_b = _recombine_gene_fields(genes_a[0], genes_b[0])
            ca.chromosomes[i].genes = [gene_a] + genes_b[1:]
            cb.chromosomes[best_j].genes = [gene_b] + genes_a[1:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = 0
        else:
            ca.chromosomes[i].genes = genes_b
            cb.chromosomes[best_j].genes = genes_a
        _normalize_split(ca.chromosomes[i])
        _normalize_split(cb.chromosomes[best_j])
    for chromosome in ca.chromosomes + cb.chromosomes:
        _normalize_split(chromosome)
    return ca, cb


def n_genes(genome):
    return sum(len(c.genes) for c in genome.chromosomes)


# ── neutral compaction (gene expression) ─────────────────────────────────────────
# A dense ontogeny genome carries genes that win NO growth lookup — they are
# unexpressed, and (proven) removing a never-winning gene cannot change any argmin,
# so the grown organism is bit-identical and the fitness is unchanged. Compaction
# drops exactly those: the threshold is zero expression (a neutrality boundary, not
# a tuned cutoff), leaving the genome's functional core with behaviour untouched.
# Biologically: pseudogene loss. NB measured only a modest shrink on fresh ontogeny
# seeds (~7-14% unexpressed — the density is mostly FUNCTIONAL morphology, not junk),
# so this is a human-facing cleanup for inspecting/refining an evolved genome
# (the Designer), NOT an evolution driver — it did not improve solving in tests.
def compact_genome(genome, seeds, grid_size=7, iters=30):
    """Return a copy of `genome` with all never-expressed genes removed (neutral:
    same grown organism, same fitness). `seeds` must be the SAME seeds the genome
    is grown with, or the expression tally is for a different organism."""
    _, counts = grow_lut_tracked(genome, seeds=tuple(seeds),
                                 grid_size=grid_size, iters=iters)
    new_chroms, k = [], 0
    for c in genome.chromosomes:
        kept = [g for gi, g in enumerate(c.genes) if counts[k + gi] > 0]
        k += len(c.genes)
        if kept:
            split = (0 if len(kept) < 2 else
                     max(1, min(int(c.split), len(kept) - 1)))
            new_chroms.append(Chromosome(
                genes=kept, split=split,
                tag=c.tag, telomere=getattr(c, 'telomere', MAX_TELOMERE)))
    if not new_chroms:                     # everything unexpressed -> keep 1 gene
        c0 = genome.chromosomes[0]
        new_chroms = [Chromosome(genes=c0.genes[:1], split=0, tag=c0.tag,
                                 telomere=getattr(c0, 'telomere', MAX_TELOMERE))]
    return Genome(chromosomes=new_chroms, tag=genome.tag)


# ── ontogeny seeding is THE seed path (see lut_evo/ontogeny.py) ──────────────────
# Sparse random genomes grow near-uniform DIAMONDS — the "uninteresting" LUTs.
# Richness tracks gene density (sim6's table_create invents a gene per unseen
# context), so the population/immigrants are seeded with sim6-style biomorph
# genomes, and the smaller-genome parsimony tie-break is dropped: it pruned the
# dense genomes straight back to sparse (re-diamonding them). This is a user
# decision (2026-07-07): rich morphology over solve speed, no random/ontogeny
# switch. Size is bounded by ONTOGENY_CAP + telomeres instead of parsimony.
ONTOGENY_CAP  = 350               # max genes kept from an ontogeny biomorph


# Growing one sim6 ontogeny seed is expensive (measured ~0.3 s: up to 60
# biomorph growths per accepted genome), and make_seed_genome is called for
# EVERY immigrant EVERY generation — that alone was a large share of ontogeny
# mode's ~50x slowdown. So the factory grows only the first _ONTO_POOL_SIZE
# genomes per chromosome count and caches them; later calls return a mutated
# variant of a random pool member (still fresh genetic material, ~free).
_ONTO_POOL      = {}     # n_chroms -> [cached seed genomes]
_ONTO_POOL_SIZE = 24


def make_seed_genome(n_chroms=2):
    """The population/immigrant factory: a dense sim6-style biomorph genome
    (the rich-morphology seeds — see the ONTOGENY_CAP note above)."""
    from .ontogeny import random_ontogeny_genome
    pool = _ONTO_POOL.setdefault(n_chroms, [])
    if len(pool) < _ONTO_POOL_SIZE:
        g = random_ontogeny_genome(n_chroms, cap_genes=ONTOGENY_CAP)
        pool.append(g)
        return clone_genome(g)     # the pool master stays pristine
    return mutate_lut(random.choice(pool), chromosome_count=n_chroms)


def _tiebreak(genome):
    """Tie-break term for equal fitness: NEUTRAL — the smaller-genome parsimony
    pressure was what pruned dense ontogeny genomes back to sparse diamonds
    (re-diamonding). Size is bounded by ONTOGENY_CAP + telomeres instead."""
    return 0


def _gene_cap():
    """Per-chromosome gene cap for add/duplicate mutations — sized for dense
    ontogeny genomes (MAX_GENES was the sparse random path's tight cap)."""
    return ONTOGENY_CAP


def tournament_lut(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: (fitnesses[i], _tiebreak(population[i])))]


def select_parent(population, fitnesses, case_vecs=None):
    import nv_evo.ga as _nga
    if (_nga.SELECTION == 'lexicase' and case_vecs is not None
            and case_vecs[0] is not None):
        return _nga._lexicase_parent(population, case_vecs)
    return tournament_lut(population, fitnesses)


def consolidate_population(parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
    """Keep the best evaluated parent/offspring entries after a terminal solve."""
    pop = len(parents)
    if (len(parent_fitnesses) != pop or len(offspring) != pop
            or len(offspring_fitnesses) != pop):
        raise ValueError('parent/offspring population sizes must match')
    case_vectors = parent_cases is not None or offspring_cases is not None
    if case_vectors and (parent_cases is None or offspring_cases is None):
        raise ValueError('parent and offspring case vectors must both be present')
    if case_vectors and (len(parent_cases) != pop or len(offspring_cases) != pop):
        raise ValueError('case-vector count must match population size')

    genomes = list(parents) + list(offspring)
    fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
    cases = ((list(parent_cases) + list(offspring_cases))
             if case_vectors else None)
    order = list(range(len(genomes)))
    random.shuffle(order)
    order.sort(key=lambda i: (fitnesses[i], _tiebreak(genomes[i])), reverse=True)
    keep = order[:pop]
    return ([genomes[i] for i in keep],
            [fitnesses[i] for i in keep],
            ([cases[i] for i in keep] if cases is not None else None))


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None, ga_config=None,
                    chromosome_count=None, recombination=True):
    """One generation of a steady, exploratory GA — elitism + immigrants +
    recombination/mutation, parents via ε-lexicase when per-case vectors are
    available (temporal targets), else tournament. This operator stays full
    replacement; terminal convergence is handled after evaluation by
    ``consolidate_population``. `mean_mutations` overrides the mutation rate for
    this generation (used by the annealing schedule)."""
    pop = len(population)
    elite_count = (ELITE_COUNT if ga_config is None else ga_config.elite_count)
    immigrant_fraction = (IMMIGRANT_FRAC if ga_config is None
                          else ga_config.immigrant_fraction)
    tournament_size = (TOURNAMENT_K if ga_config is None
                       else ga_config.tournament_size)
    if ga_config is not None and ga_config.chromosome_count is not None:
        chromosome_count = ga_config.chromosome_count
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    import nv_evo.ga as _nga
    selection = (_nga.SELECTION if ga_config is None else ga_config.selection)
    if make_genome is None:
        make_genome = lambda: make_seed_genome(chromosome_count or 2)
    n_elite = elite_count if elite_count is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = min(int(round(pop * immigrant_fraction)), pop)
    order   = sorted(range(pop),
                     key=lambda i: (fitnesses[i], _tiebreak(population[i])),
                     reverse=True)
    # Elites choose parents here; terminal survivors are chosen later from the
    # evaluated parent+offspring union rather than copied inside reproduction.
    new_pop = [make_genome() for _ in range(n_imm)]                  # immigrants
    use_lexicase = (selection == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    residual = order[n_elite:] if n_elite < pop else order
    elite_pool = order[:n_elite] if n_elite else order
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]

    def pick_index(candidates):
        if use_lexicase:
            local_population = [population[index] for index in candidates]
            local_cases = [case_vecs[index] for index in candidates]
            chosen = _nga._lexicase_parent(local_population, local_cases)
            return candidates[next(
                index for index, genome in enumerate(local_population)
                if genome is chosen)]
        k = min(tournament_size, len(candidates))
        return max(
            random.sample(candidates, k),
            key=lambda index: (
                fitnesses[index], _tiebreak(population[index])))

    def parent_pair():
        if pop == 1:
            return population[0], population[0]

        def choose(exclude=None):
            if use_lexicase:
                candidates = [index for index in range(pop) if index != exclude]
                return pick_index(candidates)
            candidates = [index for index in elite_pool if index != exclude]
            exploratory = [index for index in residual if index != exclude]
            if exploratory and random.random() < EXPLORATION_PARENT_FRAC:
                candidates = exploratory
            if not candidates:
                candidates = [index for index in range(pop) if index != exclude]
            return pick_index(candidates)

        first = choose()
        second = choose(first)
        if recombination_signatures[second] == recombination_signatures[first]:
            distinct = [
                index for index in range(pop)
                if index != first
                and recombination_signatures[index]
                != recombination_signatures[first]
            ]
            if distinct:
                second = pick_index(distinct)
        return population[first], population[second]

    while len(new_pop) < pop:
        pa, pb = parent_pair()
        ca, cb = (crossover_lut(pa, pb) if recombination else
                  (clone_genome(pa), clone_genome(pb)))
        max_telomere = (MAX_TELOMERE if ga_config is None
                        else ga_config.max_telomere)
        new_pop.append(mutate_lut(
            ca, mean_mutations, max_telomere,
            chromosome_count=chromosome_count))
        if len(new_pop) < pop:
            new_pop.append(mutate_lut(
                cb, mean_mutations, max_telomere,
                chromosome_count=chromosome_count))
    if (chromosome_count is not None
            and any(len(genome.chromosomes) != chromosome_count
                    for genome in new_pop)):
        raise ValueError('genome factory violated configured chromosome count')
    return new_pop[:pop]


def evolve_lut(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True):
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    make_genome = lambda: make_seed_genome(n_chroms)     # ontogeny biomorph seeds
    cache       = LRUCache(FITNESS_CACHE_MAX)
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:
        population  = [make_genome() for _ in range(pop)]
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        bi           = max(range(pop),
                           key=lambda i: (fitnesses[i], _tiebreak(population[i])))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        stagnation   = 0
        mut_rate     = MEAN_MUTATIONS        # annealing schedule (see MUT_DECAY)
        for gen in range(generations):
            mut_rate *= MUT_DECAY
            mm = adaptive_mutation_rate(mut_rate, stagnation,
                                        solved=best_fitness >= 1.0)
            parents, parent_fitnesses, parent_cases = population, fitnesses, cases
            offspring = next_population(
                parents, parent_fitnesses, make_genome, parent_cases, mm,
                chromosome_count=n_chroms)
            offspring_fitnesses, offspring_cases = eval_batch_cases(
                offspring, target, cache, ex)
            if max(best_fitness, max(offspring_fitnesses)) >= 1.0:
                population, fitnesses, cases = consolidate_population(
                    parents, parent_fitnesses, parent_cases,
                    offspring, offspring_fitnesses, offspring_cases)
            else:
                population, fitnesses, cases = (
                    offspring, offspring_fitnesses, offspring_cases)
            gi = max(range(pop),
                     key=lambda i: (fitnesses[i], _tiebreak(population[i])))
            if fitnesses[gi] > best_fitness + 1e-12:
                stagnation = 0
            else:
                stagnation += 1
            if (fitnesses[gi] > best_fitness
                    or (fitnesses[gi] == best_fitness
                        and _tiebreak(population[gi]) > _tiebreak(best_genome))):
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
            if verbose and gen % 10 == 0:
                print("%5d  %6.4f  %6.4f" % (gen, best_fitness,
                                             sum(fitnesses) / pop))
        return best_genome, best_fitness
    finally:
        ex.shutdown()


def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None, executor=None, should_stop=None, on_progress=None,
              max_telomere=MAX_TELOMERE, chromosome_count=None):
    """Build evaluated, rule-distinct valid offspring; see NV's implementation."""
    if cache is None:
        cache = LRUCache(FITNESS_CACHE_MAX)
    if batch is None:
        batch = max(48, pop_size)
    should_stop = should_stop or (lambda: False)
    seeds = list(seeds)
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count for genome in seeds):
            raise ValueError('seed violates configured chromosome count')
    _eval = lambda gs: eval_batch_cases(gs, target, cache, executor)[0]   # reuse pool
    pool, pool_signatures, seen = [], [], set()
    for g, f in zip(seeds, _eval(seeds)):
        s = _recombination_signature(g)
        if f >= valid and s not in seen:
            pool.append(g); pool_signatures.append(s); seen.add(s)
    if not pool:
        return pool
    for round_i in range(rounds):
        if len(pool) >= pop_size or should_stop():
            break
        if on_progress is not None:
            on_progress(round_i + 1, rounds, len(pool))
        cands, candidate_signatures = [], []
        for _ in range(batch):
            parent_index = random.randrange(len(pool))
            parent_a = pool[parent_index]
            signature_a = pool_signatures[parent_index]
            mates = [index for index, signature in enumerate(pool_signatures)
                     if signature != signature_a]
            if mates:
                base = random.choice(crossover_lut(
                    parent_a, pool[random.choice(mates)]))
            else:
                base = parent_a
            c = mutate_lut(base, max_telomere=max_telomere,
                           chromosome_count=chromosome_count)
            s = _recombination_signature(c)
            if s not in seen:
                seen.add(s)
                cands.append(c); candidate_signatures.append(s)
        if should_stop():
            break
        for c, s, f in zip(cands, candidate_signatures, _eval(cands)):
            if f >= valid and len(pool) < pop_size:
                pool.append(c); pool_signatures.append(s)
    return pool[:pop_size]


# ── GUI report ───────────────────────────────────────────────────────────────────

def lut_report(ttarget, genome=None):
    """Temporal-target report for the LUT model (mirrors nv temporal_report)."""
    lines = ['Target: %s   [LUT array]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    if getattr(ttarget, 'score_mode', 'trace') == 'period_stepper':
        lines += ['', 'Backend detail: LUT cadence transitions may occur at sub-second times.']
        if genome is None:
            return '\n'.join(lines + ['', '(run the GA or Load Saved to inspect a circuit)'])
        prep = prepare_lut(genome, ttarget)
        if prep is None:
            return '\n'.join(lines + ['', '(circuit incomplete - grew too little or inputs dead)'])
        _, out_pos, traces = prep
        total, cases = period_stepper_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos[term.role]))
        for ti, (trial, score) in enumerate(zip(ttarget.trials, cases), 1):
            lines.append('Test %d: %s  cadence score %.3f %s' % (
                ti, trial_input_summary(trial, ttarget.n_inputs), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines.append('')
        lines.append('=> cadence-stepper score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else ''))
        return '\n'.join(lines)
    if getattr(ttarget, 'score_mode', 'trace') == 'cadence':
        lines += ['', 'Backend detail: cadence uses raw LUT wire edge timestamps.']
        if genome is None:
            return '\n'.join(lines + ['', '(run the GA or Load Saved to inspect a circuit)'])
        prep = prepare_lut(genome, ttarget)
        if prep is None:
            return '\n'.join(lines + ['', '(circuit incomplete - grew too little or inputs dead)'])
        _, out_pos, traces = prep
        total, cases = cadence_score(traces, ttarget)
        for term in ttarget.outputs:
            lines.append("out '%s' read at %s" % (term.role, out_pos[term.role]))
        for ti, score in enumerate(cases):
            events = _role_events(traces, ttarget.outputs[0].role, ti)
            lines.append('Test %d: %s  output edges@%s  cadence score %.3f %s' % (
                ti + 1, trial_input_summary(ttarget.trials[ti], ttarget.n_inputs),
                event_list_summary(events), score,
                'PASS' if score >= 0.999 else 'FAIL'))
        lines += ['', '=> cadence score %.4f%s' % (
            total, '   SOLVED' if total >= 0.999 else '')]
        return '\n'.join(lines)
    if getattr(ttarget, 'score_mode', 'trace') == 'events':
        lines += ['', 'Backend detail: raw LUT wire timestamps retain sub-second edges.']
        prep = None if genome is None else prepare_lut(genome, ttarget)
        traces = prep[2] if prep is not None else None
        best_s = _best_event_shift(traces, ttarget)[0] if traces is not None else 0.0
        if prep is not None:
            for term in ttarget.outputs:
                lines.append("out '%s' read at %s" % (term.role, prep[1][term.role]))
            if getattr(ttarget, 'fit_latency', True):
                lines.append('measured output latency offset: %+.3f seconds' % best_s)
            else:
                lines.append('fixed timing: required delay %g seconds; no offset fitted'
                             % getattr(ttarget, 'latency', 0))
        for ti, trial in enumerate(ttarget.trials):
            for role in trial.expected:
                lines.append('Test %d: %s' % (
                    ti + 1, trial_input_summary(trial, ttarget.n_inputs)))
                lines.append('  expect %s edges@%s' % (
                    role, event_list_summary(_expected_events(trial, role))))
                if traces is not None:
                    score = _event_case_score(traces, ttarget, ti, role, best_s)
                    lines.append('        actual edges@%s  F1 %.3f %s' % (
                        event_list_summary(_role_events(traces, role, ti)), score,
                        'PASS' if score >= 0.999 else 'FAIL'))
        if traces is not None:
            total = event_score(traces, ttarget, shift=best_s)
            lines += ['', '=> event score %.4f%s' % (
                total, '   SOLVED' if total >= 0.999 else '')]
        return '\n'.join(lines)
    lines += ['',
              'Scored as persistent behaviour in active and quiet windows.',
              'A one-second phase tolerance allows recurrent outputs to ring.',
              'One shared input-to-output latency is used across every test/output.',
              'The exact per-second value is diagnostic only and ignores tolerance.']
    traces = None
    best_s = 0
    if genome is not None:
        prep = prepare_lut(genome, ttarget)
        if prep is None:
            lines += ['', '(circuit incomplete — grew too little or inputs dead)']
        else:
            _, out_pos, traces = prep
            best_s = _best_shift(traces, ttarget)[0]
            for t in ttarget.outputs:
                lines.append("out '%s' read at %s" % (t.role, out_pos[t.role]))
            lines.append('measured output latency offset: %+d second(s)' % best_s)
    for ti, trial in enumerate(ttarget.trials):
        lines += ['', 'Test %d: %s' % (
            ti + 1, trial_input_summary(trial, ttarget.n_inputs))]
        for role, exp in trial.expected.items():
            lines.append('  expect %s%s' % (
                expected_window_summary(exp),
                ' (%s)' % role if len(trial.expected) > 1 else ''))
            if traces is not None:
                tr = traces.get(role, [])
                tr_i = tr[ti] if ti < len(tr) else []
                # per-trial F1 at the genome's global latency shift (matches the
                # latency-invariant score; mirrors nv temporal_report)
                tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp, best_s)
                s = _f1(tp_rec, n_exp, tp_prec, n_act)
                lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                             % (''.join(str(v) for v in tr_i), s,
                                'PASS' if s >= 0.999 else 'FAIL',
                                tp_rec, n_exp, tp_prec, n_act))
    if traces is not None:
        s = windowed_score(traces, ttarget, shift=best_s)
        lines += ['', '=> behavioural score %.4f%s   (exact per-second %.4f)'
                  % (s, '   SOLVED' if s >= 0.999 else '',
                     exact_tick_accuracy(traces, ttarget))]
    return '\n'.join(lines)
