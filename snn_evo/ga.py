from __future__ import annotations
import copy, math, random, os
from concurrent.futures import ProcessPoolExecutor

from functools import partial
from evo_runtime.cache import LRUCache
from evo_runtime.parallel import map_ordered

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
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))  # see nv_evo.ga
FITNESS_CACHE_MAX = 200_000
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

def genome_signature(genome):
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0),
         tuple((g.state_n, g.state_s, g.state_e, g.state_w,
                g.self_in, g.self_out) for g in c.genes))
        for c in genome.chromosomes)


def _eval_batch(genomes, target=None, arch=None, executor=None, cache=None,
                should_stop=None, on_progress=None):
    """Evaluate a population -> list of fitnesses. A persistent `executor`
    (ProcessPoolExecutor) is reused instead of spawning a fresh pool every call —
    on Windows the per-generation spawn+re-import dominated runtime, so reuse is a
    large speed-up (matches nv_evo/lut_evo). Omitting it keeps the one-shot pool.
    `should_stop`/`on_progress` are threaded to map_ordered (saturated, no chunk
    barrier, cancellable)."""
    fn = partial(evaluate_genome, target=target, arch=arch)
    if cache is None:
        if executor is not None:
            return map_ordered(executor, fn, genomes, should_stop, on_progress)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            return map_ordered(ex, fn, genomes, should_stop, on_progress)
    if len(cache) > FITNESS_CACHE_MAX:
        cache.clear()
    sigs = [genome_signature(g) for g in genomes]
    out = [cache.get(sig) for sig in sigs]
    unique = {}
    for i, (sig, result) in enumerate(zip(sigs, out)):
        if result is None:
            unique.setdefault(sig, []).append(i)
    representatives = [(sig, indices[0]) for sig, indices in unique.items()]
    subset = [genomes[i] for _, i in representatives]
    if not subset:
        return out
    if executor is not None:
        results = map_ordered(executor, fn, subset, should_stop, on_progress)
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            results = map_ordered(ex, fn, subset, should_stop, on_progress)
    for (sig, _), result in zip(representatives, results):
        cache[sig] = result
        for i in unique[sig]:
            out[i] = result
    return out

# ── genetic operators ──────────────────────────────────────────────

_GENE_FIELDS = ("state_n", "state_s", "state_e", "state_w",
                "self_in", "self_out")


def _normalize_split(chromosome):
    """Keep split on a real between-gene boundary (or zero for one gene)."""
    count = len(chromosome.genes)
    chromosome.split = (0 if count < 2 else
                        max(1, min(int(chromosome.split), count - 1)))


def _recombine_gene_fields(gene_a, gene_b):
    """Uniformly recombine a single rule's active alleles.

    ``limit`` is a legacy pickle field and is no longer used by growth, so it is
    deliberately not presented as fake breeding variation.
    """
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


def _other_value(value, low, high):
    """Choose a value from [low, high) that is genuinely different."""
    span = high - low
    if span < 2:
        raise ValueError('mutation domain must contain at least two values')
    return low + ((value - low + random.randrange(1, span)) % span)


def _value_excluding(low, high, *values):
    """Choose uniformly from [low, high), excluding the supplied values."""
    excluded = sorted({int(value) for value in values
                       if value is not None and low <= int(value) < high})
    pick = random.randrange((high - low) - len(excluded)) + low
    for value in excluded:
        if pick >= value:
            pick += 1
    return pick


def _force_nonparent_tweak(genome, parent):
    """End a multi-edit transaction at an allele distinct from its parent."""
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(random_chromosome())
        else:
            random.choice(genome.chromosomes).genes.append(random_gene())
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
    low = 1 if field == 'self_out' else 0
    setattr(gene, field, _value_excluding(
        low, MAX_STATE, getattr(gene, field), parent_value))
    genome.chromosomes[ci].genes[gi] = gene


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


def _mutate_once(genome, chromosome_count=None):
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
    if chromosome_count is None and len(chroms) < MAX_CHROMS:
        choices.append('add_chrom')
    if chromosome_count is None and len(chroms) > 1:
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


def mutate(genome, chromosome_count=None):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    g = clone_genome(genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    # This is an offspring-only GA: recombination is always followed by at
    # least one real mutation.  For a multi-edit transaction, reserve the final
    # allele edit so earlier inverse operations cannot cancel back to a clone.
    events = max(1, _poisson(MEAN_MUTATIONS))
    for _ in range(events - 1):
        _mutate_once(g, chromosome_count=chromosome_count)
    if events == 1:
        _mutate_once(g, chromosome_count=chromosome_count)
    else:
        _force_nonparent_tweak(g, genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    return g

def crossover(pa, pb):
    """Tag-matched hierarchical crossover, including one-rule chromosomes."""
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
        # Snapshot both inputs before assigning either reciprocal child.
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

def tournament(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: rank_key(population[i], fitnesses[i]))]

def next_population(population, fitnesses, chromosome_count=None,
                    recombination=True):
    """One generation of offspring only.

    Elites are a recombination parent pool, not verbatim survivors.  The
    separately tracked champion remains safe outside the population, while the
    current-generation best stays an honest value for the chart.
    """
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    pop     = len(population)
    n_elite = max(1, int(pop * ELITE_FRAC))
    order   = sorted(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]), reverse=True)
    elite   = order[:n_elite]
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]

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
        if not second_pool:
            return population[ia], population[ia]
        distinct = [
            i for i in second_pool
            if recombination_signatures[i] != recombination_signatures[ia]
        ]
        if not distinct:
            distinct = [
                i for i in range(pop)
                if i != ia
                and recombination_signatures[i] != recombination_signatures[ia]
            ]
        if distinct:
            second_pool = distinct
        ib = pick_index(second_pool)
        return population[ia], population[ib]

    new_pop = []
    while len(new_pop) < pop:
        pa, pb = parent_pair()
        ca, cb = (crossover(pa, pb) if recombination else
                  (clone_genome(pa), clone_genome(pb)))
        new_pop.append(mutate(ca, chromosome_count=chromosome_count))
        if len(new_pop) < pop:
            new_pop.append(mutate(cb, chromosome_count=chromosome_count))
    return new_pop[:pop]

# ── main loop ─────────────────────────────────────────────────────

def evolve(generations=100, verbose=True, n_chroms=2, pop=None, target=None,
           arch=None, seed=None):
    if seed is not None:
        random.seed(seed)
    if target is None:
        target = get_target(DEFAULT_TARGET)
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    popsize    = pop or POPSIZE
    population = [random_genome(n_chroms) for _ in range(popsize)]
    cache = LRUCache(FITNESS_CACHE_MAX)
    # Reuse ONE worker pool across generations (matches nv_evo/lut_evo). Spawning
    # a fresh pool every generation dominated runtime on Windows.
    ex = ProcessPoolExecutor(max_workers=N_WORKERS)
    try:
        fitnesses  = _eval_batch(population, target, arch, ex, cache)
        best_idx   = max(range(popsize), key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[best_idx])
        best_fitness = fitnesses[best_idx]

        if verbose:
            print("%5s  %6s  %6s  Summary" % ("Gen", "Best", "Mean"))
            print("-" * 72)

        for gen in range(generations):
            population = next_population(
                population, fitnesses, chromosome_count=n_chroms)
            fitnesses  = _eval_batch(population, target, arch, ex, cache)
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
            # Keep running after a solve so a requested generation budget is a
            # real, comparable budget across all backends.
    finally:
        ex.shutdown()

    return best_genome, best_fitness
