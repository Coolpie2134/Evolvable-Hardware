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

from nv_evo.temporal import (_role_trace_score, _pr_counts, _f1,
                             windowed_score, exact_tick_accuracy)
from .genome import (LUT_STATES, MAX_GENES, MAX_CHROMS, MAX_TELOMERE,
                     Genome, Chromosome,
                     random_lut_gene, random_lut_chromosome, random_lut_genome)
from .lut import grow_lut, LutSim


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
MEAN_MUTATIONS = 4.0         # hot-start rate for annealing (see nv_evo.ga)
MUT_DECAY      = 0.99        # simulated-annealing α; 4.0 -> ~0.03 by gen 500
N_WORKERS      = min(os.cpu_count() or 2, 8)
FITNESS_CACHE_MAX = 200_000  # cap the fitness cache on very long runs


# ── running trials / placing outputs (trace-matched, as in nv) ──────────────────

OUT_RADIUS = 12   # read the output near its terminal (see nv temporal.py)


def place_outputs_by_trace(grid, in_pos, ttarget):
    """{role: cell}, traces — each role read at the live non-input cell nearest
    its terminal whose trace best matches the expectation across all trials
    (terminal-distance tie-break), exactly like the nervous backend.

    The per-trial dynamics run in numpy (LutSim.run_bits): one [T, ncells]
    emit-bit matrix per trial, from which any candidate cell's trace is a column
    slice — no per-tick dicts (that dict churn dominated LUT scoring)."""
    in_set = set(in_pos)
    out_pos = {t.role: None for t in ttarget.outputs}
    traces  = {}
    if len(grid) <= len(in_set):
        return out_pos, traces
    sim = LutSim(grid)                       # one sim, reset between trials
    cidx = sim._cidx
    trial_B = []                             # [T, ncells] emit-bit matrix per trial
    for tr in ttarget.trials:
        sim.reset()
        trial_B.append(sim.run_bits(tr.streams, in_pos, ttarget.T))
    used = set()
    for term in ttarget.outputs:
        tx, ty = term.pos
        cands = sorted((c for c in grid if c not in in_set),
                       key=lambda c: (abs(c[0] - tx) + abs(c[1] - ty), c))[:OUT_RADIUS]
        best, best_key = None, None
        for c in cands:
            if c in used:
                continue
            col = cidx[c]
            scores = []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                tr = trial_B[ti][:, col].tolist()
                scores.append(_role_trace_score(tr, exp))
            s   = sum(scores) / len(scores) if scores else 0.0
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best = key, c
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        col = cidx[best]
        traces[term.role]  = [trial_B[ti][:, col].tolist()
                              for ti in range(len(ttarget.trials))]
    return out_pos, traces


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
    return windowed_score(prep[2], ttarget)


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
    sim    = LutSim(grid)
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
             '          latched & synchronous — paper Architecture 2)']
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
    from nv_evo.temporal import _trace_metric
    cases = []
    for ti, trial in enumerate(target.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            cases.append(_trace_metric(tr[ti], exp) if ti < len(tr) else 0.0)
    return windowed_score(traces, target), tuple(cases)


def evaluate_lut(genome, target):
    return evaluate_lut_full(genome, target)[0]


def genome_signature(genome):
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0),
         tuple((g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in, g.self_out)
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_cases(genomes, target, cache=None, executor=None):
    """(fitnesses, case_vectors) in parallel; cache holds (fit, cases). A
    persistent `executor` is reused instead of spawning a fresh pool per call
    (large speed-up on Windows); omitting it keeps the one-shot-pool behaviour."""
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
        subset = [genomes[i] for i in todo]
        if executor is not None:
            results = list(executor.map(fn, subset))
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                results = list(ex.map(fn, subset))
        for i, r in zip(todo, results):
            out[i] = r
            if cache is not None:
                cache[sigs[i]] = r
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


def _tweak_gene(gene):
    """Soft mutation: usually flip 1-3 bits of one 16-bit field (a nearby
    boolean function), sometimes replace the field entirely. self_in has an
    extra chance of snapping to 0 — growth rules must stay reachable."""
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and random.random() < 0.2:
        g.self_in = 0
    elif random.random() < 0.5:
        v = getattr(g, fld)
        for _ in range(random.randint(1, 3)):
            v ^= 1 << random.randrange(16)
        setattr(g, fld, v)
    else:
        setattr(g, fld, random.randrange(LUT_STATES))
    return g


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]


def mutate_lut(genome, mean_mutations=None):
    g = clone_genome(genome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    for _ in range(_poisson(lam)):
        if not g.chromosomes:
            g.chromosomes.append(random_lut_chromosome())
            continue
        op    = random.choices(_MUT_OPS, weights=_MUT_WEIGHTS)[0]
        chrom = random.choice(g.chromosomes)
        if op == "telomere":
            base = getattr(chrom, 'telomere', 10)
            chrom.telomere = max(1, min(MAX_TELOMERE,
                                        base + random.randint(-3, 3)))
        elif op == "tweak" and chrom.genes:
            idx = random.randrange(len(chrom.genes))
            chrom.genes[idx] = _tweak_gene(chrom.genes[idx])
        elif op == "duplicate" and chrom.genes and len(chrom.genes) < _gene_cap():
            src = random.choice(chrom.genes)
            chrom.genes.insert(random.randrange(len(chrom.genes) + 1),
                               _tweak_gene(src))
        elif op == "add_gene" and len(chrom.genes) < _gene_cap():
            chrom.genes.append(random_lut_gene())
        elif op == "del_gene" and len(chrom.genes) > 1:
            chrom.genes.pop(random.randrange(len(chrom.genes)))
        elif op == "add_chrom" and len(g.chromosomes) < MAX_CHROMS:
            g.chromosomes.append(random_lut_chromosome())
        elif op == "del_chrom" and len(g.chromosomes) > 1:
            g.chromosomes.remove(random.choice(g.chromosomes))
        elif op == "split" and len(chrom.genes) > 1:
            chrom.split = random.randint(1, len(chrom.genes) - 1)
    return g


def crossover_lut(pa, pb):
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
        sp      = max(0, min(chrom_a.split, len(chrom_a.genes), len(chrom_b.genes)))
        ca.chromosomes[i].genes      = chrom_a.genes[:sp] + chrom_b.genes[sp:]
        cb.chromosomes[best_j].genes = chrom_b.genes[:sp] + chrom_a.genes[sp:]
    return ca, cb


def n_genes(genome):
    return sum(len(c.genes) for c in genome.chromosomes)


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
    return mutate_lut(random.choice(pool))


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


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None):
    """One generation of a steady, exploratory GA — elitism + immigrants +
    recombination/mutation, parents via ε-lexicase when per-case vectors are
    available (temporal targets), else tournament. Convergence is NOT forced:
    the population contracts onto a genome only insofar as selection favours it
    (see nv_evo.ga.next_population). `mean_mutations` overrides the mutation rate
    for this generation (used by the annealing schedule)."""
    pop = len(population)
    if make_genome is None:
        make_genome = make_seed_genome
    n_elite = ELITE_COUNT if ELITE_COUNT is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = int(round(pop * IMMIGRANT_FRAC))
    order   = sorted(range(pop),
                     key=lambda i: (fitnesses[i], _tiebreak(population[i])),
                     reverse=True)
    # Elites = recombination parent pool, NOT verbatim survivors (see nv_evo.ga).
    new_pop = [make_genome() for _ in range(min(n_imm, pop))]      # random immigrants
    import nv_evo.ga as _nga           # lut shares the nervous SELECTION switch
    use_lexicase = (_nga.SELECTION == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    if n_elite > 0 and not use_lexicase:
        elite = order[:n_elite]
        k = min(TOURNAMENT_K, len(elite))
        parent = lambda: population[max(random.sample(elite, k),
                                        key=lambda i: (fitnesses[i], _tiebreak(population[i])))]
    else:
        parent = lambda: select_parent(population, fitnesses, case_vecs)
    while len(new_pop) < pop:
        ca, cb = crossover_lut(parent(), parent())
        new_pop.append(mutate_lut(ca, mean_mutations))
        if len(new_pop) < pop:
            new_pop.append(mutate_lut(cb, mean_mutations))
    return new_pop[:pop]


def evolve_lut(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True):
    make_genome = lambda: make_seed_genome(n_chroms)     # ontogeny biomorph seeds
    cache       = {}
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:
        population  = [make_genome() for _ in range(pop)]
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        bi           = max(range(pop),
                           key=lambda i: (fitnesses[i], _tiebreak(population[i])))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        mut_rate     = MEAN_MUTATIONS        # annealing schedule (see MUT_DECAY)
        for gen in range(generations):
            mut_rate *= MUT_DECAY
            population = next_population(population, fitnesses, make_genome, cases, mut_rate)
            fitnesses, cases = eval_batch_cases(population, target, cache, ex)
            gi = max(range(pop),
                     key=lambda i: (fitnesses[i], _tiebreak(population[i])))
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
              cache=None, executor=None):
    """Fill a population with genomes that are BOTH valid (>= `valid`) and
    genotypically unique, by walking the neutral network via neutral mutation
    with rejection. See nv_evo.ga.diversify — same idea on the LUT substrate."""
    if cache is None:
        cache = {}
    if batch is None:
        batch = max(48, pop_size)
    _eval = lambda gs: eval_batch_cases(gs, target, cache, executor)[0]   # reuse pool
    pool, seen = [], set()
    for g, f in zip(seeds, _eval(list(seeds))):
        s = genome_signature(g)
        if f >= valid and s not in seen:
            pool.append(g); seen.add(s)
    if not pool:
        return pool
    for _ in range(rounds):
        if len(pool) >= pop_size:
            break
        cands = []
        for _ in range(batch):
            c = mutate_lut(random.choice(pool))
            s = genome_signature(c)
            if s not in seen:
                seen.add(s)
                cands.append(c)
        for c, f in zip(cands, _eval(cands)):
            if f >= valid and len(pool) < pop_size:
                pool.append(c)
    return pool[:pop_size]


# ── GUI report ───────────────────────────────────────────────────────────────────

def lut_report(ttarget, genome=None):
    """Temporal-target report for the LUT model (mirrors nv temporal_report)."""
    lines = ['Target: %s   [LUT array]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    lines += ['',
              'Square grid; each cell is a 16-bit lookup table over its 4',
              'neighbours\' bits, latched and synchronous (paper Architecture 2).',
              "'.' = unscored tick. Scored by the shared spike-event F1 (the",
              'metric the GA optimises): an expected 1 is hit if the cell fires',
              'on it or an adjacent tick (±1 ring tolerance — holds may ring),',
              'and every pulse on an expected-0 tick costs precision exactly.']
    traces = None
    if genome is not None:
        prep = prepare_lut(genome, ttarget)
        if prep is None:
            lines += ['', '(circuit incomplete — grew too little or inputs dead)']
        else:
            _, out_pos, traces = prep
            for t in ttarget.outputs:
                lines.append("out '%s' read at %s" % (t.role, out_pos[t.role]))
    names = [chr(65 + i) for i in range(ttarget.n_inputs)]
    for ti, trial in enumerate(ttarget.trials):
        pulses = {n: [t for t in range(len(trial.streams)) if trial.streams[t][i]]
                  for i, n in enumerate(names)}
        lines += ['', 'Trial %d:  %s' % (ti + 1, '   '.join(
            '%s pulses@%s' % (n, p if p else '(none)') for n, p in pulses.items()))]
        for role, exp in trial.expected.items():
            lines.append('  expect %s' % ''.join('.' if e is None else str(e)
                                                 for e in exp))
            if traces is not None:
                tr = traces.get(role, [])
                tr_i = tr[ti] if ti < len(tr) else []
                # per-trial F1 (the selection metric) with components, matching
                # nv temporal_report — not the old 'balanced' diagnostic
                tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp)
                s = _f1(tp_rec, n_exp, tp_prec, n_act)
                lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                             % (''.join(str(v) for v in tr_i), s,
                                'PASS' if s >= 0.999 else 'FAIL',
                                tp_rec, n_exp, tp_prec, n_act))
    if traces is not None:
        s = windowed_score(traces, ttarget)
        lines += ['', '=> behavioural score %.4f%s   (exact per-tick %.4f)'
                  % (s, '   SOLVED' if s >= 0.999 else '',
                     exact_tick_accuracy(traces, ttarget))]
    return '\n'.join(lines)
