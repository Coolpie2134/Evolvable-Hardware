from __future__ import annotations
import copy, math, random, os
from concurrent.futures import ProcessPoolExecutor

from functools import partial

from .genome import (Genome, Chromosome, random_gene, random_chromosome,
                     random_genome, germline_telomere,
                     MAX_STATE, MAX_GENES, MAX_CHROMS, MAX_TELOMERE)
from .growth import grow_snn
from .snn import interpret_grid, circuit_summary
from .fitness import evaluate, score, N_OUTPUTS
from .targets import get_target, DEFAULT_TARGET

POPSIZE        = 120
ELITE_FRAC     = 0.10
TOURNAMENT_K   = 4
MEAN_MUTATIONS = 1.2
N_WORKERS      = min(os.cpu_count() or 2, 8)
# Below this fitness, rank on fitness alone — don't let body/gene parsimony bias
# the search before the behaviour is solved (a counter needs its structure).
PARSIMONY_START_FITNESS = 0.999


def clone_genome(g):
    """Shallow structural copy: new chromosome/gene lists, shared (never mutated
    in place) gene objects. Mirrors nv_evo.clone_genome — far cheaper than
    deepcopy, which dominated reproduction."""
    return Genome(
        chromosomes=[Chromosome(genes=list(c.genes), split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE))
                     for c in g.chromosomes],
        tag=g.tag)


def n_genes(g):
    return sum(len(c.genes) for c in g.chromosomes)


def rank_key(genome, fitness):
    """Selection key: fitness first, then (once solved) a senescence/parsimony
    tie-break — fewer genes, then a shorter telomere (a smaller, cheaper body).
    Never distorts the fitness value, so a solved run still reads exactly 1.0."""
    if fitness < PARSIMONY_START_FITNESS:
        return (fitness, 0, 0)
    return (fitness, -n_genes(genome), -germline_telomere(genome))

# ── evaluation ────────────────────────────────────────────────────
# (the nervous / temporal backends have their own GA: see nv_evo/ga.py)

def evaluate_genome(genome, target=None, arch=None):
    if target is None:
        target = get_target(DEFAULT_TARGET)
    grid = grow_snn(genome, seeds=tuple(target.inputs),
                    grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0
    neurons, synapses = interpret_grid(grid, target=target, arch=arch)
    return score(neurons, synapses, target)

def _eval_batch(genomes, target=None, arch=None, executor=None):
    """Evaluate a population -> list of fitnesses. A persistent `executor`
    (ProcessPoolExecutor) is reused instead of spawning a fresh pool every call —
    on Windows the per-generation spawn+re-import dominated runtime, so reuse is a
    large speed-up (matches nv_evo/lut_evo). Omitting it keeps the one-shot pool."""
    fn = partial(evaluate_genome, target=target, arch=arch)
    if executor is not None:
        return list(executor.map(fn, genomes))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        return list(ex.map(fn, genomes))

# ── genetic operators ──────────────────────────────────────────────

def _other_value(value, low, high):
    """Choose a value from [low, high) that is genuinely different."""
    span = high - low
    if span < 2:
        raise ValueError('mutation domain must contain at least two values')
    return low + ((value - low + random.randrange(1, span)) % span)


def _mutate_gene(gene):
    g     = copy.copy(gene)
    field = random.choice(["state_n","state_s","state_e","state_w",
                           "self_in","self_out"])
    if field == "self_out":
        g.self_out = _other_value(g.self_out, 1, MAX_STATE)
    else:
        setattr(g, field, _other_value(getattr(g, field), 0, MAX_STATE))
    return g

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L: k += 1; p *= random.random()
    return k - 1


def _mutate_once(genome):
    """Apply one feasible, state-changing mutation to ``genome``.

    Invalid operations (such as deleting the last gene) are excluded before
    drawing.  That makes an offspring mutation a real genetic event rather
    than a random chance to do nothing.
    """
    if not genome.chromosomes:
        genome.chromosomes.append(random_chromosome())
        return

    chroms = genome.chromosomes
    with_genes = [c for c in chroms if c.genes]
    choices = []
    if with_genes:
        choices.append('tweak')
    if any(len(c.genes) < MAX_GENES for c in chroms):
        choices.append('add_gene')
    if any(len(c.genes) > 1 for c in chroms):
        choices.append('del_gene')
    if len(chroms) < MAX_CHROMS:
        choices.append('add_chrom')
    if len(chroms) > 1:
        choices.append('del_chrom')
    if any(len(c.genes) > 2 for c in chroms):
        choices.append('split')
    if any(1 < getattr(c, 'telomere', MAX_TELOMERE) < MAX_TELOMERE
           or getattr(c, 'telomere', MAX_TELOMERE) in (1, MAX_TELOMERE)
           for c in chroms):
        choices.append('telomere')

    op = random.choice(choices)
    if op == 'tweak':
        chrom = random.choice(with_genes)
        idx = random.randrange(len(chrom.genes))
        chrom.genes[idx] = _mutate_gene(chrom.genes[idx])
    elif op == 'add_gene':
        random.choice([c for c in chroms if len(c.genes) < MAX_GENES]).genes.append(random_gene())
    elif op == 'del_gene':
        chrom = random.choice([c for c in chroms if len(c.genes) > 1])
        chrom.genes.pop(random.randrange(len(chrom.genes)))
    elif op == 'add_chrom':
        chroms.append(random_chromosome())
    elif op == 'del_chrom':
        chroms.pop(random.randrange(len(chroms)))
    elif op == 'split':
        chrom = random.choice([c for c in chroms if len(c.genes) > 2])
        options = [s for s in range(1, len(chrom.genes)) if s != chrom.split]
        chrom.split = random.choice(options)
    else:  # telomere
        chrom = random.choice(chroms)
        base = getattr(chrom, 'telomere', MAX_TELOMERE)
        options = [base + d for d in (-1, 1) if 1 <= base + d <= MAX_TELOMERE]
        chrom.telomere = random.choice(options)


def mutate(genome):
    g = clone_genome(genome)
    # This is an offspring-only GA: recombination is always followed by at
    # least one real mutation.  The Poisson draw supplies any additional edits.
    for _ in range(max(1, _poisson(MEAN_MUTATIONS))):
        _mutate_once(g)
    return g

def crossover(pa, pb):
    ca, cb  = clone_genome(pa), clone_genome(pb)
    used_b  = set()
    for i, chrom_a in enumerate(ca.chromosomes):
        best_j, best_dist = None, float("inf")
        for j, chrom_b in enumerate(cb.chromosomes):
            if j in used_b: continue
            d = abs(chrom_a.tag - chrom_b.tag)
            if d < best_dist: best_dist, best_j = d, j
        if best_j is None: continue
        used_b.add(best_j)
        chrom_b = cb.chromosomes[best_j]
        sp      = max(0, min(chrom_a.split, len(chrom_a.genes), len(chrom_b.genes)))
        ca.chromosomes[i].genes          = chrom_a.genes[:sp] + chrom_b.genes[sp:]
        cb.chromosomes[best_j].genes     = chrom_b.genes[:sp] + chrom_a.genes[sp:]
    return ca, cb

def tournament(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: rank_key(population[i], fitnesses[i]))]

def next_population(population, fitnesses):
    """One generation of offspring only.

    Elites are a recombination parent pool, not verbatim survivors.  The
    separately tracked champion remains safe outside the population, while the
    current-generation best stays an honest value for the chart.
    """
    pop     = len(population)
    n_elite = max(1, int(pop * ELITE_FRAC))
    order   = sorted(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]), reverse=True)
    elite   = order[:n_elite]

    def pick_index(candidates):
        k = min(TOURNAMENT_K, len(candidates))
        return max(random.sample(candidates, k),
                   key=lambda i: rank_key(population[i], fitnesses[i]))

    def parent_pair():
        # Sampling each parent independently let a one-member elite pool mate
        # the champion with itself, turning crossover into a clone operation.
        # Draw the second parent from a different individual; if there is just
        # one elite, cross it with a non-elite rather than selfing it.
        ia = pick_index(elite)
        second_pool = [i for i in elite if i != ia]
        if not second_pool:
            second_pool = [i for i in range(pop) if i != ia]
        ib = pick_index(second_pool)
        return population[ia], population[ib]

    new_pop = []
    while len(new_pop) < pop:
        ca, cb = crossover(*parent_pair())
        new_pop.append(mutate(ca))
        if len(new_pop) < pop:
            new_pop.append(mutate(cb))
    return new_pop[:pop]

# ── main loop ─────────────────────────────────────────────────────

def evolve(generations=100, verbose=True, n_chroms=2, pop=None, target=None, arch=None):
    if target is None:
        target = get_target(DEFAULT_TARGET)
    popsize    = pop or POPSIZE
    population = [random_genome(n_chroms) for _ in range(popsize)]
    # Reuse ONE worker pool across generations (matches nv_evo/lut_evo). Spawning
    # a fresh pool every generation dominated runtime on Windows.
    ex = ProcessPoolExecutor(max_workers=N_WORKERS)
    try:
        fitnesses  = _eval_batch(population, target, arch, ex)
        best_idx   = max(range(popsize), key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[best_idx])
        best_fitness = fitnesses[best_idx]

        if verbose:
            print("%5s  %6s  %6s  Summary" % ("Gen", "Best", "Mean"))
            print("-" * 72)

        for gen in range(generations):
            population = next_population(population, fitnesses)
            fitnesses  = _eval_batch(population, target, arch, ex)
            gi = max(range(popsize), key=lambda i: rank_key(population[i], fitnesses[i]))
            if rank_key(population[gi], fitnesses[gi]) > rank_key(best_genome, best_fitness):
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
            if verbose and (gen % 10 == 0 or fitnesses[gi] >= 1.0):
                mean_f = sum(fitnesses) / popsize
                grid   = grow_snn(best_genome, seeds=tuple(target.inputs),
                                  grid_size=target.grid_size)
                ns, ss = interpret_grid(grid, target=target, arch=arch)
                print("%5d  %6.4f  %6.4f  %s" % (gen, best_fitness, mean_f,
                                                   circuit_summary(ns, ss)))
            if best_fitness >= 1.0:
                if verbose: print("Solved at generation %d!" % gen)
                break
    finally:
        ex.shutdown()

    return best_genome, best_fitness
