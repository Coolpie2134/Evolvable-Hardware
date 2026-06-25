from __future__ import annotations
import copy, math, random, os
from concurrent.futures import ProcessPoolExecutor

from .genome import (Genome, random_gene, random_chromosome,
                     random_genome, MAX_STATE, MAX_GENES, MAX_CHROMS)
from .growth import grow_snn
from .snn import interpret_grid, circuit_summary
from .fitness import evaluate, N_OUTPUTS

POPSIZE        = 120
ELITE_FRAC     = 0.10
TOURNAMENT_K   = 4
MEAN_MUTATIONS = 1.2
N_WORKERS      = min(os.cpu_count() or 2, 8)

# ── evaluation ────────────────────────────────────────────────────

def evaluate_genome(genome):
    grid = grow_snn(genome)
    if len(grid) <= 2:
        return 0.0
    neurons, synapses = interpret_grid(grid, n_outputs=N_OUTPUTS)
    return evaluate(neurons, synapses)

def _eval_batch(genomes):
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        return [f.result() for f in [ex.submit(evaluate_genome, g) for g in genomes]]

# ── genetic operators ──────────────────────────────────────────────

def _mutate_gene(gene):
    g     = copy.copy(gene)
    field = random.choice(["state_n","state_s","state_e","state_w",
                           "self_in","self_out","limit"])
    if field == "limit":
        from .genome import MAX_ITER
        g.limit = MAX_ITER - random.randint(0, MAX_ITER // 3)
    elif field == "self_out":
        g.self_out = random.randint(1, MAX_STATE - 1)
    else:
        setattr(g, field, random.randint(0, MAX_STATE - 1))
    return g

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L: k += 1; p *= random.random()
    return k - 1

def mutate(genome):
    g = copy.deepcopy(genome)
    for _ in range(_poisson(MEAN_MUTATIONS)):
        if not g.chromosomes:
            g.chromosomes.append(random_chromosome()); continue
        op    = random.randint(0, 5)
        chrom = random.choice(g.chromosomes)
        if   op == 0 and chrom.genes:
            idx = random.randrange(len(chrom.genes))
            chrom.genes[idx] = _mutate_gene(chrom.genes[idx])
        elif op == 1 and len(chrom.genes) < MAX_GENES:
            chrom.genes.append(random_gene())
        elif op == 2 and len(chrom.genes) > 1:
            chrom.genes.pop(random.randrange(len(chrom.genes)))
        elif op == 3 and len(g.chromosomes) < MAX_CHROMS:
            g.chromosomes.append(random_chromosome())
        elif op == 4 and len(g.chromosomes) > 1:
            g.chromosomes.pop(random.randrange(len(g.chromosomes)))
        elif op == 5 and len(chrom.genes) > 1:
            chrom.split = random.randint(1, len(chrom.genes) - 1)
    return g

def crossover(pa, pb):
    ca, cb  = copy.deepcopy(pa), copy.deepcopy(pb)
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
    return population[max(idx, key=lambda i: fitnesses[i])]

# ── main loop ─────────────────────────────────────────────────────

def evolve(generations=100, verbose=True, n_chroms=2, pop=None):
    popsize    = pop or POPSIZE
    population = [random_genome(n_chroms) for _ in range(popsize)]
    fitnesses  = _eval_batch(population)
    best_idx   = max(range(popsize), key=lambda i: fitnesses[i])
    best_genome  = copy.deepcopy(population[best_idx])
    best_fitness = fitnesses[best_idx]

    if verbose:
        print("%5s  %6s  %6s  Summary" % ("Gen", "Best", "Mean"))
        print("-" * 72)

    for gen in range(generations):
        n_elite   = max(1, int(popsize * ELITE_FRAC))
        elite_idx = sorted(range(popsize), key=lambda i: fitnesses[i], reverse=True)
        new_pop   = [copy.deepcopy(population[i]) for i in elite_idx[:n_elite]]
        while len(new_pop) < popsize:
            ca, cb = crossover(tournament(population, fitnesses),
                               tournament(population, fitnesses))
            new_pop.append(mutate(ca))
            if len(new_pop) < popsize:
                new_pop.append(mutate(cb))
        population = new_pop[:popsize]
        fitnesses  = _eval_batch(population)
        gi = max(range(popsize), key=lambda i: fitnesses[i])
        if fitnesses[gi] > best_fitness:
            best_fitness = fitnesses[gi]
            best_genome  = copy.deepcopy(population[gi])
        if verbose and (gen % 10 == 0 or fitnesses[gi] >= 1.0):
            mean_f = sum(fitnesses) / popsize
            grid   = grow_snn(best_genome)
            ns, ss = interpret_grid(grid, n_outputs=N_OUTPUTS)
            print("%5d  %6.4f  %6.4f  %s" % (gen, best_fitness, mean_f,
                                               circuit_summary(ns, ss)))
        if best_fitness >= 1.0:
            if verbose: print("Solved at generation %d!" % gen)
            break

    return best_genome, best_fitness
