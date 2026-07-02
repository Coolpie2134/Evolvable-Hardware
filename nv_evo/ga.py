"""
nv_evo/ga.py — genetic algorithm native to the nervous net, tuned for
evolving loops and memory.

Temporal fitness landscapes are deceptive: an SR latch scores nothing until a
feedback loop *and* its input/output wiring all appear together, so a plain GA
stalls on flat plateaus and converges on loop-free delay chains. This GA
differs from snn_evo's in five ways:

  1. Trace-matched outputs + windowed balanced scoring (temporal targets):
     each output role is read at the live cell whose trace best matches the
     expected trace across ALL trials (terminal distance only breaks ties), so
     evolution only has to build the mechanism, not also route its answer to
     one prescribed cell. The trace is scored per WINDOW (each hold phase
     weighs the same, so a long hold can't drown a missed reset), balanced
     across levels (a constant output caps at 0.5 — measured: under plain
     accuracy the GA converges onto the constant-0 baseline on every preset).
     Still 1.0 iff every scored tick is right; score_temporal stays the plain
     per-tick metric for reporting.

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
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .genome import (MAX_STATE, MAX_GENES, MAX_CHROMS,
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


def evaluate_nv(genome, target):
    """Fitness for the nervous backend. Combinational targets score plainly;
    temporal targets use windowed level-balanced scoring at trace-matched
    output cells, plus the loop-aware shaping (1.0 iff the trace is perfect)."""
    if not getattr(target, 'temporal', False):
        return score_nervous(genome, target)
    prep = prepare_net(genome, target)
    if prep is None:
        return 0.0
    grid, routing, in_pos, out_pos, traces = prep
    s = windowed_score(traces, target)
    if s >= 1.0:
        return 1.0
    return s + (1.0 - s) * LOOP_WEIGHT * _loop_bonus(grid, routing, in_pos, out_pos)


def genome_signature(genome):
    """Hashable identity of a genome's evolvable content (for the fitness cache)."""
    return tuple(
        (c.tag, c.split,
         tuple((g.ctx_l, g.ctx_r, g.ctx_d, g.self_in, g.self_out)
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_nv(genomes, target, cache=None):
    """Evaluate a population in parallel. `cache` ({signature: fitness}, owned
    by the caller so it dies with the run) skips already-seen genomes."""
    fits = [None] * len(genomes)
    todo = list(range(len(genomes)))
    if cache is not None:
        sigs = [genome_signature(g) for g in genomes]
        todo = [i for i in todo if sigs[i] not in cache]
        for i in range(len(genomes)):
            if sigs[i] in cache:
                fits[i] = cache[sigs[i]]
    if todo:
        fn = partial(evaluate_nv, target=target)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            results = list(ex.map(fn, [genomes[i] for i in todo]))
        for i, f in zip(todo, results):
            fits[i] = f
            if cache is not None:
                cache[sigs[i]] = f
    return fits


# ── genetic operators ────────────────────────────────────────────────────────────

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"]


def _tweak_gene(gene):
    g = copy.copy(gene)
    setattr(g, random.choice(_GENE_FIELDS), random.randrange(MAX_STATE))
    return g


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split"]
_MUT_WEIGHTS = [0.35, 0.15, 0.15, 0.12, 0.05, 0.05, 0.13]


def mutate_nv(genome):
    g = copy.deepcopy(genome)
    for _ in range(_poisson(MEAN_MUTATIONS)):
        if not g.chromosomes:
            g.chromosomes.append(random_hex_chromosome())
            continue
        op    = random.choices(_MUT_OPS, weights=_MUT_WEIGHTS)[0]
        chrom = random.choice(g.chromosomes)
        if op == "tweak" and chrom.genes:
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


def tournament_nv(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: fitnesses[i])]


def next_population(population, fitnesses, make_genome=None):
    """One generation: elites survive, a few random immigrants come in, and the
    rest are tournament-selected crossover + mutation offspring."""
    pop = len(population)
    if make_genome is None:
        make_genome = random_hex_genome
    n_elite = max(1, int(pop * ELITE_FRAC))
    n_imm   = max(1, int(pop * IMMIGRANT_FRAC))
    order   = sorted(range(pop), key=lambda i: fitnesses[i], reverse=True)
    new_pop = [copy.deepcopy(population[i]) for i in order[:n_elite]]
    new_pop += [make_genome() for _ in range(min(n_imm, pop - len(new_pop)))]
    while len(new_pop) < pop:
        ca, cb = crossover_nv(tournament_nv(population, fitnesses),
                              tournament_nv(population, fitnesses))
        new_pop.append(mutate_nv(ca))
        if len(new_pop) < pop:
            new_pop.append(mutate_nv(cb))
    return new_pop[:pop]


# ── main loop (headless; the GUI runs its own equivalent in app.py) ──────────────

def evolve_nervous(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True):
    make_genome = lambda: random_hex_genome(n_chroms)
    cache       = {}
    population  = [make_genome() for _ in range(pop)]
    fitnesses   = eval_batch_nv(population, target, cache)
    bi           = max(range(pop), key=lambda i: fitnesses[i])
    best_genome  = copy.deepcopy(population[bi])
    best_fitness = fitnesses[bi]

    if verbose:
        print("%5s  %6s  %6s" % ("Gen", "Best", "Mean"))
        print("-" * 24)
    for gen in range(generations):
        population = next_population(population, fitnesses, make_genome)
        fitnesses  = eval_batch_nv(population, target, cache)
        gi = max(range(pop), key=lambda i: fitnesses[i])
        if fitnesses[gi] > best_fitness:
            best_fitness = fitnesses[gi]
            best_genome  = copy.deepcopy(population[gi])
        if verbose and (gen % 10 == 0 or best_fitness >= 1.0):
            print("%5d  %6.4f  %6.4f" % (gen, best_fitness,
                                         sum(fitnesses) / pop))
        if best_fitness >= 1.0:
            if verbose:
                print("Solved at generation %d!" % gen)
            break
    return best_genome, best_fitness
