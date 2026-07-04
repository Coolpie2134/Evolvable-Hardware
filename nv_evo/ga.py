"""
nv_evo/ga.py — genetic algorithm native to the nervous net, tuned for
evolving loops and memory.

Temporal fitness landscapes are deceptive: an SR latch scores nothing until a
feedback loop *and* its input/output wiring all appear together, so a plain GA
stalls on flat plateaus and converges on loop-free delay chains. This GA
differs from snn_evo's in seven ways:

  1. Trace-matched outputs + spike-event scoring (temporal targets):
     each output role is read at the live cell whose trace best matches the
     expected trace across ALL trials (terminal distance only breaks ties), so
     evolution only has to build the mechanism, not also route its answer to
     one prescribed cell. The trace is scored by SPIKE-EVENT precision/recall
     (F1, see nv_evo/temporal.METRIC): reward each correctly-timed output
     spike, penalise expected spikes that never occur, penalise extra spikes
     that weren't expected. A do-nothing output recalls nothing and so scores
     0 — silence is not rewarded (measured: under plain per-tick accuracy the
     GA converged onto the constant-0 baseline on every preset). 1.0 iff every
     expected spike is produced and nothing fires where it shouldn't.

  2. Loop-aware shaping: a small bonus, scaled by (1 - score) so a perfect
     score is still exactly 1.0, rewards nets whose signal graph contains
     directed cycles — especially "relevant" cycles that inputs can write and
     outputs can read (loop_profile). Among equally scoring nets, the ones
     structurally *capable* of memory win the tie.

  3. Gene duplication as a mutation operator: loops are built from repeated
     local routing motifs (two cells buffering each other), and duplicating a
     working gene then tweaking one field reaches those far more often than
     fresh random genes.

  4. Random immigrants: a few fresh genomes replace the worst each generation,
     keeping exploration alive on plateaus instead of inbreeding to a stall.

  5. Fitness caching: temporal evaluation (trials x T ticks) is expensive and
     converged populations re-submit the same genomes; a signature cache skips
     re-evaluating elites and duplicates.

  6. Senescence / metabolic cost (rank_key): among equally-fit genomes the one
     that grows a cheaper, smaller body wins — fewer genes, then a shorter
     telomere (a smaller organism, since telomeres are now a Hayflick growth
     bound). This never distorts the fitness value, so a solved run still reads
     exactly 1.0 while both genome and body shrink when the task allows.

  7. Stress-induced mutagenesis (stress_multiplier): the bacterial SOS response —
     hold the mutation rate at baseline until the run stalls, then ramp it up the
     longer it stays stuck and relax the instant progress resumes. Aimed squarely
     at the deceptive temporal plateaus where only a burst of variation reaches
     the next rung.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .genome import (MAX_STATE, MAX_GENES, MAX_CHROMS, MAX_TELOMERE,
                     random_hex_gene, random_hex_chromosome, random_hex_genome)
from .nervous import score_nervous
from .temporal import prepare_net, windowed_score, loop_profile

POPSIZE        = 120
ELITE_FRAC     = 0.10
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
MEAN_MUTATIONS = 1.2
LOOP_WEIGHT    = 0.05          # max shaping bonus, as a fraction of (1 - score)
N_WORKERS      = min(os.cpu_count() or 2, 8)


# ── evaluation ─────────────────────────────────────────────────────────────────

def _loop_bonus(grid, routing, in_pos, out_pos):
    """[0,1] structural credit for memory capability. Any cycle earns a little;
    cycles that inputs can write and outputs can read (a latch's skeleton)
    earn the rest, saturating at 4 relevant cycle nodes."""
    prof = loop_profile(grid, routing, in_pos, out_pos)
    if not prof['n_cycle']:
        return 0.0
    return 0.3 + 0.7 * min(1.0, prof['n_relevant'] / 4.0)


def evaluate_nv_full(genome, target):
    """(scalar fitness, per-case score vector). Cases are the individual
    (trial, role) traces — the units ε-lexicase selection streams over. For
    combinational targets there is no trial structure: cases is None and
    selection falls back to tournament."""
    if not getattr(target, 'temporal', False):
        return score_nervous(genome, target), None
    n_cases = sum(len(tr.expected) for tr in target.trials)
    prep = prepare_net(genome, target)
    if prep is None:
        return 0.0, (0.0,) * n_cases
    grid, routing, in_pos, out_pos, traces = prep
    from .temporal import _trace_metric
    cases = []
    for ti, trial in enumerate(target.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            cases.append(_trace_metric(tr[ti], exp) if ti < len(tr) else 0.0)
    s = windowed_score(traces, target)
    if s < 1.0:
        s = s + (1.0 - s) * LOOP_WEIGHT * _loop_bonus(grid, routing, in_pos, out_pos)
    return s, tuple(cases)


def evaluate_nv(genome, target):
    """Scalar fitness (back-compat wrapper around evaluate_nv_full)."""
    return evaluate_nv_full(genome, target)[0]


def genome_signature(genome):
    """Hashable identity of a genome's evolvable content (for the fitness cache)."""
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0),
         tuple((g.ctx_l, g.ctx_r, g.ctx_d, g.self_in, g.self_out)
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_cases(genomes, target, cache=None):
    """Evaluate a population in parallel -> (fitnesses, case_vectors). `cache`
    ({signature: (fit, cases)}, owned by the caller) skips seen genomes."""
    out  = [None] * len(genomes)
    todo = list(range(len(genomes)))
    if cache is not None:
        sigs = [genome_signature(g) for g in genomes]
        todo = [i for i in todo if sigs[i] not in cache]
        for i in range(len(genomes)):
            if sigs[i] in cache:
                out[i] = cache[sigs[i]]
    if todo:
        fn = partial(evaluate_nv_full, target=target)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            results = list(ex.map(fn, [genomes[i] for i in todo]))
        for i, r in zip(todo, results):
            out[i] = r
            if cache is not None:
                cache[sigs[i]] = r
    return [r[0] for r in out], [r[1] for r in out]


def eval_batch_nv(genomes, target, cache=None):
    """Back-compat scalar batch (shares the (fit, cases) cache format)."""
    return eval_batch_cases(genomes, target, cache)[0]


# ── genetic operators ────────────────────────────────────────────────────────────

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"]


def _tweak_gene(gene):
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and random.random() < 0.2:
        g.self_in = 0                     # keep growth rules reachable
    else:
        setattr(g, fld, random.randrange(MAX_STATE))
    return g


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]


def mutate_nv(genome, mean_mutations=None):
    g = copy.deepcopy(genome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    for _ in range(_poisson(lam)):
        if not g.chromosomes:
            g.chromosomes.append(random_hex_chromosome())
            continue
        op    = random.choices(_MUT_OPS, weights=_MUT_WEIGHTS)[0]
        chrom = random.choice(g.chromosomes)
        if op == "telomere":
            # lengthen / shorten the chromosome's growth phase
            base = getattr(chrom, 'telomere', 10)
            chrom.telomere = max(1, min(MAX_TELOMERE,
                                        base + random.randint(-3, 3)))
        elif op == "tweak" and chrom.genes:
            idx = random.randrange(len(chrom.genes))
            chrom.genes[idx] = _tweak_gene(chrom.genes[idx])
        elif op == "duplicate" and chrom.genes and len(chrom.genes) < MAX_GENES:
            # copy a working rule and vary it — how repeated loop motifs arise
            src = random.choice(chrom.genes)
            chrom.genes.insert(random.randrange(len(chrom.genes) + 1),
                               _tweak_gene(src))
        elif op == "add_gene" and len(chrom.genes) < MAX_GENES:
            chrom.genes.append(random_hex_gene())
        elif op == "del_gene" and len(chrom.genes) > 1:
            chrom.genes.pop(random.randrange(len(chrom.genes)))
        elif op == "add_chrom" and len(g.chromosomes) < MAX_CHROMS:
            g.chromosomes.append(random_hex_chromosome())
        elif op == "del_chrom" and len(g.chromosomes) > 1:
            g.chromosomes.remove(random.choice(g.chromosomes))
        elif op == "split" and len(chrom.genes) > 1:
            chrom.split = random.randint(1, len(chrom.genes) - 1)
    return g


# back-compat name (the old mutation operator lived in nv_evo/genome.py)
mutate_hex = mutate_nv


def crossover_nv(pa, pb):
    """Tag-matched single-point crossover (same scheme as the SNN GA)."""
    ca, cb = copy.deepcopy(pa), copy.deepcopy(pb)
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


def total_telomere(genome):
    """Sum of the chromosome telomeres — a proxy for how large a body the genome
    is licensed to grow (radius scales with the germline telomere)."""
    return sum(getattr(c, 'telomere', 0) for c in genome.chromosomes)


def rank_key(genome, fitness):
    """Selection / ranking key (maximise). Fitness dominates absolutely; ties
    break toward a cheaper organism — SENESCENCE / METABOLIC COST: fewer genes
    first (the paper's parsimony pressure), then a shorter total telomere, so a
    genome that solves the task with a SMALLER, self-limiting body out-competes an
    equally-fit bloated one. Both tie-breaks leave the fitness value itself
    untouched, so a solved run still reads exactly 1.0 while complexity shrinks."""
    return (fitness, -n_genes(genome), -total_telomere(genome))


def tournament_nv(population, fitnesses):
    """Tournament selection with a metabolic-cost tie-break: at EQUAL fitness the
    cheaper organism wins (fewer genes, then a shorter telomere — see rank_key),
    applied without distorting the fitness value itself, so a solved run still
    reads exactly 1.0 while complexity keeps shrinking when it isn't needed."""
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: rank_key(population[i], fitnesses[i]))]


def _lexicase_parent(population, case_vecs):
    """ε-lexicase selection (La Cava et al.): stream the cases in random order;
    at each case keep only candidates within ε (median absolute deviation) of
    that case's best. Averages hide a single failing trial (~1/12 of a mean);
    lexicase makes every case a hard filter some of the time, so specialists on
    the currently-failing cases are selected and recombined — the mechanism that
    drives populations to ALL-cases-perfect rather than high-average."""
    n_cases = len(case_vecs[0])
    cand = list(range(len(population)))
    for c in random.sample(range(n_cases), n_cases):
        vals = [case_vecs[i][c] for i in cand]
        best = max(vals)
        srt  = sorted(vals)
        med  = srt[len(srt) // 2]
        eps  = sorted(abs(v - med) for v in vals)[len(vals) // 2]
        cand = [i for i, v in zip(cand, vals) if v >= best - eps]
        if len(cand) == 1:
            break
    return population[random.choice(cand)]


# Parent-selection scheme. Measured head-to-head at equal budget (see memory):
# lexicase helped only the pair detector (+0.11), regressed coincidence (-0.13)
# and pattern (-0.13), and left every plateau (latch/toggle/stepper) at the
# IDENTICAL value tournament reaches — the plateaus are representational, not a
# selection-pressure problem. Tournament is the better default; lexicase stays
# available for experiments.
SELECTION = 'tournament'          # 'tournament' | 'lexicase'


def select_parent(population, fitnesses, case_vecs=None):
    if (SELECTION == 'lexicase' and case_vecs is not None
            and case_vecs[0] is not None):
        return _lexicase_parent(population, case_vecs)
    return tournament_nv(population, fitnesses)


# Stress-induced mutagenesis (the bacterial SOS response): baseline mutation
# until the population has stalled for STRESS_PATIENCE generations, then the rate
# ramps toward STRESS_MAX_MULT the longer it stays stuck — raising variation to
# climb off a plateau, then relaxing the instant progress resumes. The temporal
# plateaus (latch/toggle/stepper) are exactly the deceptive landscapes this is
# meant for: flat regions where only a burst of variation reaches the next rung.
STRESS_PATIENCE = 12
STRESS_MAX_MULT = 4.0


def stress_multiplier(stagnation):
    """Mutation-rate multiplier as a function of generations-without-improvement."""
    if stagnation < STRESS_PATIENCE:
        return 1.0
    return min(STRESS_MAX_MULT, 1.0 + (stagnation - STRESS_PATIENCE) / STRESS_PATIENCE)


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None):
    """One generation of a steady, exploratory GA — elitism preserves the best,
    the rest are tournament/ε-lexicase parents recombined and mutated, plus a
    few random immigrants that keep exploration alive.

    Convergence is NOT forced: there is no stagnation ramp that fades immigrants
    and clones the champion. The population contracts onto a genome only insofar
    as SELECTION favours it — so it converges naturally when (and only when) a
    genome genuinely out-competes the field, and keeps exploring otherwise. The
    trade is deliberate: the population MEAN won't be herded up to the best
    (immigrant + crossover noise is always present), because that herding was the
    artificial convergence being removed."""
    pop = len(population)
    if make_genome is None:
        make_genome = random_hex_genome
    n_elite = max(1, int(pop * ELITE_FRAC))
    n_imm   = int(round(pop * IMMIGRANT_FRAC))
    order   = sorted(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]),
                     reverse=True)
    new_pop = [copy.deepcopy(population[i]) for i in order[:n_elite]]
    new_pop += [make_genome() for _ in range(min(n_imm, pop - len(new_pop)))]
    parent = lambda: select_parent(population, fitnesses, case_vecs)
    while len(new_pop) < pop:
        ca, cb = crossover_nv(parent(), parent())
        new_pop.append(mutate_nv(ca, mean_mutations))
        if len(new_pop) < pop:
            new_pop.append(mutate_nv(cb, mean_mutations))
    return new_pop[:pop]


# ── main loop (headless; the GUI runs its own equivalent in app.py) ──────────────

def evolve_nervous(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True):
    make_genome = lambda: random_hex_genome(n_chroms)
    cache       = {}
    population  = [make_genome() for _ in range(pop)]
    fitnesses, cases = eval_batch_cases(population, target, cache)
    bi           = max(range(pop),
                       key=lambda i: rank_key(population[i], fitnesses[i]))
    best_genome  = copy.deepcopy(population[bi])
    best_fitness = fitnesses[bi]
    best_rank    = rank_key(best_genome, best_fitness)

    solved_at, stagnation = None, 0
    if verbose:
        print("%5s  %6s  %6s  %5s" % ("Gen", "Best", "Mean", "Mut"))
        print("-" * 30)
    # NO early stop at best == 1.0: the run continues so the parsimony tie-break
    # keeps shrinking a solved genome. Convergence is not forced — the population
    # contracts only as far as selection naturally drives it (see next_population).
    for gen in range(generations):
        mm = MEAN_MUTATIONS * stress_multiplier(stagnation)   # SOS response
        population = next_population(population, fitnesses, make_genome, cases, mm)
        fitnesses, cases = eval_batch_cases(population, target, cache)
        gi = max(range(pop),
                 key=lambda i: rank_key(population[i], fitnesses[i]))
        gen_rank = rank_key(population[gi], fitnesses[gi])
        # SOS tracks FITNESS plateaus only: a parsimony-only win (same fitness,
        # leaner body) still updates the champion but must NOT reset the stress
        # counter, or the ever-shrinking tie-break would keep it pinned at 1.0.
        if fitnesses[gi] > best_fitness + 1e-12:
            stagnation = 0
        else:
            stagnation += 1
        if gen_rank > best_rank:
            best_rank    = gen_rank
            best_fitness = fitnesses[gi]
            best_genome  = copy.deepcopy(population[gi])
        if solved_at is None and best_fitness >= 1.0:
            solved_at = gen
            if verbose:
                print("Solved at generation %d." % gen)
        if verbose and gen % 10 == 0:
            print("%5d  %6.4f  %6.4f  %5.2f" % (gen, best_fitness,
                                                sum(fitnesses) / pop,
                                                stress_multiplier(stagnation)))
    return best_genome, best_fitness


# ── diversification: a whole generation of DISTINCT valid solutions ──────────────

def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None):
    """Fill a population with genomes that are BOTH valid (fitness >= `valid`)
    AND genotypically unique (no two share a genome_signature). Starting from the
    valid `seeds`, it walks the NEUTRAL NETWORK — neutral mutation produces a
    genotypically different genome; keep it only if it still solves and hasn't
    been seen. This is not convergence (which clones one champion): it spreads
    across the set of distinct genomes that all compute the goal.

    Returns the list of unique valid genomes found (up to pop_size). Where the
    target has a broad neutral network it fills the population; where solutions
    are isolated spikes it returns however few exist — an honest ceiling, not a
    monoculture faked with copies."""
    if cache is None:
        cache = {}
    if batch is None:
        batch = max(48, pop_size)
    pool, seen = [], set()
    for g, f in zip(seeds, eval_batch_nv(list(seeds), target, cache)):
        s = genome_signature(g)
        if f >= valid and s not in seen:
            pool.append(g); seen.add(s)
    if not pool:
        return pool
    for _ in range(rounds):
        if len(pool) >= pop_size:
            break
        cands, sigs = [], []
        for _ in range(batch):
            c = mutate_nv(random.choice(pool))
            s = genome_signature(c)
            if s not in seen:                 # reject clones/known genotypes
                seen.add(s)                   # (invalid ones stay flagged: no re-eval)
                cands.append(c); sigs.append(s)
        for c, f in zip(cands, eval_batch_nv(cands, target, cache)):
            if f >= valid and len(pool) < pop_size:
                pool.append(c)
    return pool[:pop_size]
