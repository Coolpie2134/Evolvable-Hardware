"""Genetic operators and population evaluation for the Functional NV Net."""
from __future__ import annotations

import copy
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from runtime.mutation import adaptive_mutation_rate
from runtime.parallel import map_ordered
from substrates.nervous.hexgrid import hex_frontier_cells

from .catalogue import (
    BY_ID, COMPONENTS, DEFAULT_FAMILIES, enabled_component_ids,
    local_component_ids, normalise_families,
)
from .evaluation import evaluate_functional_full
from .genome import (
    MAX_CHROMS, MAX_GENES, MAX_TELOMERE, Chromosome, FunctionalGene, Genome,
    input_layout_domain, input_layout_radius, random_component_id,
    random_functional_chromosome, random_functional_gene,
    random_functional_genome,
)

N_WORKERS = max(1, min((os.cpu_count() or 2) - 2, 16))
FITNESS_CACHE_MAX = 200_000
GENE_FIELDS = ("ctx_l", "ctx_r", "ctx_d", "self_in", "self_out")


def clone_genome(genome):
    """Copy mutable FNV structure without recursively walking scalar fields.

    FNV mutations edit gene objects in place, so unlike NV/LUT their genes
    cannot be shared between offspring.  They contain only scalar alleles,
    however, making one shallow copy per gene equivalent to ``deepcopy`` while
    avoiding its memo/dispatch overhead throughout reproduction.
    """
    clone = copy.copy(genome)
    clone.chromosomes = []
    for chromosome in genome.chromosomes:
        copied = copy.copy(chromosome)
        copied.genes = [copy.copy(gene) for gene in chromosome.genes]
        clone.chromosomes.append(copied)
    layout = getattr(genome, "input_layout", None)
    if layout is not None:
        clone.input_layout = tuple(tuple(cell) for cell in layout)
    return clone


def genome_signature(genome):
    layout = (
        None if getattr(genome, "input_layout", None) is None
        else tuple(tuple(cell) for cell in genome.input_layout)
    )
    return (layout, tuple(
            (
                chromosome.tag,
                chromosome.split,
                chromosome.telomere,
                tuple(tuple(getattr(gene, field) for field in GENE_FIELDS)
                      for gene in chromosome.genes),
            )
            for chromosome in genome.chromosomes
        ))


def _evaluate_record(genome, target):
    fitness, cases, topology = evaluate_functional_full(
        genome, target, include_topology=True)
    total = target.n_inputs + len(target.outputs)
    return fitness, cases, (total, total), 0.0, None, topology.score


def eval_batch_cases(genomes, target, cache=None, executor=None,
                     should_stop=None, on_progress=None):
    records = [None] * len(genomes)
    signatures = [genome_signature(genome) for genome in genomes]
    missing = []
    groups = {}
    for index, signature in enumerate(signatures):
        cached = cache.get(signature) if cache is not None else None
        if cached is not None:
            records[index] = cached
        else:
            groups.setdefault(signature, []).append(index)
    representatives = [(signature, indices[0])
                       for signature, indices in groups.items()]
    if representatives:
        fn = partial(_evaluate_record, target=target)
        subset = [genomes[index] for _, index in representatives]
        if executor is None:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as local_executor:
                missing = map_ordered(
                    local_executor, fn, subset, should_stop, on_progress)
        else:
            missing = map_ordered(
                executor, fn, subset, should_stop, on_progress)
        for (signature, _), record in zip(representatives, missing):
            for index in groups[signature]:
                records[index] = record
            if cache is not None:
                cache[signature] = record
    for genome, record in zip(genomes, records):
        genome._juvenile_score = 0.0
        genome._robust_cases = None
        genome._robustness = 0.0
        genome._topology_score = (
            float(record[5]) if len(record) > 5 else 0.0)
    return [record[0] for record in records], [record[1] for record in records]


def _poisson(mean):
    if mean <= 0:
        return 0
    limit, count, product = math.exp(-mean), 0, 1.0
    while product > limit:
        count += 1
        product *= random.random()
    return count - 1


def _normalize_split(chromosome):
    count = len(chromosome.genes)
    chromosome.split = (
        0 if count < 2 else max(1, min(int(chromosome.split), count - 1)))


def _different_state(current, families, *, local_probability=0.85,
                     empty_probability=0.08):
    if current != 0 and random.random() < local_probability:
        if random.random() < 0.75:
            nearby = local_component_ids(current, families)
            if nearby:
                return random.choice(nearby)
        same_family = [
            entry.id for entry in COMPONENTS
            if entry.id != current
            and entry.family == BY_ID[current].family
            and entry.family in families
        ]
        if same_family:
            return random.choice(same_family)
    candidate = random_component_id(
        families, empty_probability=empty_probability)
    if candidate == current:
        alternatives = [
            state for state in enabled_component_ids(
                families, include_empty=True)
            if state != current
        ]
        if alternatives:
            candidate = random.choice(alternatives)
    return candidate


def _mutate_allele(genome, families, preferred_loci=(), focused_loci=()):
    locus = None
    if focused_loci and random.random() < 0.65:
        candidates = [
            (chromosome_index, gene_index)
            for chromosome_index, gene_index in focused_loci
            if chromosome_index < len(genome.chromosomes)
            and gene_index < len(genome.chromosomes[
                chromosome_index].genes)
        ]
        if candidates:
            locus = random.choice(candidates)
    if locus is None and preferred_loci and random.random() < 0.80:
        candidates = [
            (chromosome_index, gene_index)
            for chromosome_index, gene_index in preferred_loci
            if chromosome_index < len(genome.chromosomes)
            and gene_index < len(genome.chromosomes[
                chromosome_index].genes)
        ]
        if candidates:
            locus = random.choice(candidates)
    if locus is None:
        chromosome = random.choice(genome.chromosomes)
        gene = random.choice(chromosome.genes)
    else:
        chromosome = genome.chromosomes[locus[0]]
        gene = chromosome.genes[locus[1]]
    # On a rule known to control development, changing its expressed output is
    # much more likely to create a coherent phenotype neighbor than perturbing
    # one context coordinate and leaving the winning output unchanged.
    field = (
        "self_out"
        if locus is not None and random.random() < 0.60
        else random.choice(GENE_FIELDS)
    )
    current = getattr(gene, field)
    # EMPTY remains an independently reachable absence allele.  Within a
    # family, route/output-count/timing neighbors dominate.
    setattr(gene, field, _different_state(
        current, families,
        empty_probability=(0.12 if field == "self_in" else 0.08)))


def mutate_input_layout(genome, max_telomere=MAX_TELOMERE):
    """Move one non-anchor input pad by one physical honeycomb edge.

    The first input stays at the origin as a coordinate gauge. Every actual
    relative placement remains reachable by moving the other pads.
    """
    layout = getattr(genome, "input_layout", None)
    if layout is None or len(layout) < 2:
        return False
    sites = [tuple(map(int, cell)) for cell in layout]
    domain = set(input_layout_domain(
        input_layout_radius(max_telomere, len(sites))))
    occupied = set(sites)
    indices = list(range(1, len(sites)))
    random.shuffle(indices)
    for index in indices:
        options = [
            neighbor for neighbor in hex_frontier_cells(*sites[index])
            if neighbor in domain and neighbor not in occupied
        ]
        if not options:
            continue
        new = random.choice(options)
        sites[index] = new
        genome.input_layout = tuple(sites)
        return True
    return False


def _mutate_structure(genome, families, max_telomere, chromosome_count,
                      preferred_loci=(), focused_loci=()):
    choices = ["telomere", "add_gene", "delete_gene", "duplicate_gene",
               "split", "tag"]
    if chromosome_count is None:
        choices.extend(["add_chromosome", "delete_chromosome"])
    action = random.choice(choices)
    chromosome = random.choice(genome.chromosomes)
    if action == "telomere":
        step = random.choice((-1, 1))
        chromosome.telomere = max(
            1, min(int(max_telomere), chromosome.telomere + step))
    elif action == "add_gene" and len(chromosome.genes) < MAX_GENES:
        index = random.randrange(len(chromosome.genes) + 1)
        chromosome.genes.insert(
            index, random_functional_gene(families))
    elif action == "delete_gene" and len(chromosome.genes) > 1:
        del chromosome.genes[random.randrange(len(chromosome.genes))]
    elif action == "duplicate_gene" and len(chromosome.genes) < MAX_GENES:
        index = random.randrange(len(chromosome.genes))
        chromosome.genes.insert(index + 1, copy.copy(
            chromosome.genes[index]))
    elif action == "split":
        if len(chromosome.genes) > 1:
            chromosome.split = random.randint(1, len(chromosome.genes) - 1)
    elif action == "tag":
        chromosome.tag = random.randint(0, 999)
    elif action == "add_chromosome" and len(genome.chromosomes) < MAX_CHROMS:
        genome.chromosomes.append(random_functional_chromosome(
            max_telomere=max_telomere, families=families))
    elif action == "delete_chromosome" and len(genome.chromosomes) > 1:
        del genome.chromosomes[random.randrange(len(genome.chromosomes))]
    else:
        _mutate_allele(
            genome, families, preferred_loci, focused_loci)
    for item in genome.chromosomes:
        _normalize_split(item)


def mutate_functional(genome, mean_mutations=None, *,
                      max_telomere=MAX_TELOMERE, chromosome_count=None,
                      families=DEFAULT_FAMILIES, growth_seeds=None,
                      focus_families=()):
    enabled = normalise_families(families)
    preferred_loci = ()
    effective_seeds = (
        tuple(getattr(genome, "input_layout"))
        if getattr(genome, "input_layout", None) is not None
        else tuple(growth_seeds or ()))
    if effective_seeds:
        from .growth import active_gene_loci
        preferred_loci = active_gene_loci(genome, effective_seeds)
    focused_loci = tuple(
        (chromosome_index, gene_index)
        for chromosome_index, gene_index in preferred_loci
        if (
            getattr(genome.chromosomes[chromosome_index].genes[gene_index],
                    "self_out", 0) != 0
            and BY_ID[
                genome.chromosomes[chromosome_index].genes[
                    gene_index].self_out
            ].family in focus_families
        )
    )
    mean = 4.0 if mean_mutations is None else max(0.0, float(mean_mutations))
    count = max(1, _poisson(mean))
    for _ in range(count):
        if (getattr(genome, "input_layout", None) is not None
                and len(genome.input_layout) > 1
                and random.random() < 0.12
                and mutate_input_layout(genome, max_telomere)):
            continue
        if random.random() < 0.78:
            _mutate_allele(
                genome, enabled, preferred_loci, focused_loci)
        else:
            _mutate_structure(
                genome, enabled, max_telomere, chromosome_count,
                preferred_loci, focused_loci)
    if chromosome_count is not None:
        while len(genome.chromosomes) < chromosome_count:
            genome.chromosomes.append(random_functional_chromosome(
                max_telomere=max_telomere, families=enabled))
        if len(genome.chromosomes) > chromosome_count:
            del genome.chromosomes[chromosome_count:]
    return genome


def crossover_functional(parent_a, parent_b):
    """Chromosome/gene recombination followed by mutation in the breeder."""
    child = clone_genome(parent_a)
    layout_b = getattr(parent_b, "input_layout", None)
    if (layout_b is not None
            and (getattr(child, "input_layout", None) is None
                 or len(child.input_layout) == len(layout_b))
            and random.random() < 0.5):
        # Input geometry is one co-adapted physical module. Per-pad crossover
        # would manufacture collisions and tear apart useful relative layouts.
        child.input_layout = tuple(tuple(cell) for cell in layout_b)
    shared = min(len(child.chromosomes), len(parent_b.chromosomes))
    for index in range(shared):
        if random.random() >= 0.5:
            continue
        a = child.chromosomes[index]
        b = parent_b.chromosomes[index]
        if len(a.genes) > 1 and len(b.genes) > 1:
            cut_a = max(1, min(a.split, len(a.genes) - 1))
            cut_b = max(1, min(b.split, len(b.genes) - 1))
            genes = (
                [copy.copy(gene) for gene in a.genes[:cut_a]]
                + [copy.copy(gene) for gene in b.genes[cut_b:]]
            )[:MAX_GENES]
            if genes:
                a.genes = genes
        elif a.genes and b.genes:
            gene_a, gene_b = a.genes[0], random.choice(b.genes)
            fields = [field for field in GENE_FIELDS
                      if getattr(gene_a, field) != getattr(gene_b, field)]
            if fields:
                for field in random.sample(
                        fields, random.randint(1, len(fields))):
                    setattr(gene_a, field, getattr(gene_b, field))
        if random.random() < 0.5:
            a.telomere = b.telomere
        _normalize_split(a)
    return child


def rank_key(genome, fitness):
    """Correctness first; FNV topology, never genome size, breaks final ties."""
    return (
        float(fitness),
        float(getattr(genome, "_robustness", 0.0)),
        float(getattr(genome, "_juvenile_score", 0.0)),
        float(getattr(genome, "_topology_score", 0.0)),
    )


def _tournament(population, fitnesses, size=4):
    indices = random.sample(
        range(len(population)), min(int(size), len(population)))
    return population[max(
        indices, key=lambda index: rank_key(
            population[index], fitnesses[index]))]


def _lexicase(population, case_vectors):
    if not case_vectors or not case_vectors[0]:
        return None
    candidates = list(range(len(population)))
    for case in random.sample(
            range(len(case_vectors[0])), len(case_vectors[0])):
        values = [case_vectors[index][case] for index in candidates]
        best = max(values)
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        epsilon = sorted(
            abs(value - median) for value in values)[len(values) // 2]
        candidates = [
            index for index, value in zip(candidates, values)
            if value >= best - epsilon
        ]
        if len(candidates) == 1:
            break
    # Standard lexicase has already filtered on every behavioral case. FNV's
    # target-agnostic topology potential chooses among the remaining acceptable
    # candidates; exact topology ties remain random.
    best_topology = max(
        float(getattr(population[index], "_topology_score", 0.0))
        for index in candidates)
    candidates = [
        index for index in candidates
        if float(getattr(
            population[index], "_topology_score", 0.0)) == best_topology
    ]
    return population[random.choice(candidates)]


def next_population(population, fitnesses, make_genome=None,
                    case_vecs=None, mean_mutations=None, selection=None,
                    ga_config=None, chromosome_count=None,
                    recombination=True, archive_parent=None,
                    stagnation=0, rescue_candidates=None,
                    families=DEFAULT_FAMILIES, growth_seeds=None,
                    focus_families=(), **_ignored):
    del stagnation
    count = len(population)
    if not count:
        return []
    enabled = normalise_families(families)
    max_telomere = (
        getattr(ga_config, "max_telomere", MAX_TELOMERE)
        if ga_config is not None else MAX_TELOMERE)
    if chromosome_count is None and ga_config is not None:
        chromosome_count = ga_config.chromosome_count
    if make_genome is None:
        reference_layout = getattr(population[0], "input_layout", None)
        make_genome = lambda: random_functional_genome(
            chromosome_count or 2, max_telomere=max_telomere,
            families=enabled,
            n_inputs=(
                len(reference_layout)
                if reference_layout is not None else None))
    immigrant_fraction = (
        ga_config.immigrant_fraction if ga_config is not None else 0.08)
    tournament_size = (
        ga_config.tournament_size if ga_config is not None else 4)
    recombination = (
        recombination and
        (ga_config.recombination_enabled if ga_config is not None else True))
    escape = getattr(ga_config, "escape", None)
    mutation_limit = getattr(ga_config, "mutation_limit", 8.0)
    mean = 4.0 if mean_mutations is None else mean_mutations

    children = [clone_genome(genome)
                for genome in list(rescue_candidates or ())[:count]]
    if archive_parent is not None and len(children) < count:
        children.append(mutate_functional(
            clone_genome(archive_parent), mean,
            max_telomere=max_telomere, chromosome_count=chromosome_count,
            families=enabled, growth_seeds=growth_seeds,
            focus_families=focus_families))
    immigrant_count = min(
        count - len(children), int(round(count * immigrant_fraction)))
    for _ in range(immigrant_count):
        children.append(make_genome())

    def parent():
        if selection == "lexicase":
            selected = _lexicase(population, case_vecs)
            if selected is not None:
                return selected
        return _tournament(population, fitnesses, tournament_size)

    while len(children) < count:
        left = parent()
        right = parent()
        child = (
            crossover_functional(left, right)
            if recombination else clone_genome(left))
        individual_mean = mean
        if escape is not None and escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate, mutation_rate_of
            inherit_mutation_rate(
                child, left, right, escape, mean, mutation_limit)
            individual_mean = mutation_rate_of(child, mean)
        mutate_functional(
            child, individual_mean, max_telomere=max_telomere,
            chromosome_count=chromosome_count, families=enabled,
            growth_seeds=growth_seeds,
            focus_families=focus_families)
        children.append(child)
    return children


def consolidate_population(parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
    count = len(parents)
    genomes = list(parents) + list(offspring)
    fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
    cases = (
        list(parent_cases) + list(offspring_cases)
        if parent_cases is not None and offspring_cases is not None else None)
    order = list(range(len(genomes)))
    random.shuffle(order)
    order.sort(
        key=lambda index: rank_key(genomes[index], fitnesses[index]),
        reverse=True)
    keep = order[:count]
    return (
        [genomes[index] for index in keep],
        [fitnesses[index] for index in keep],
        ([cases[index] for index in keep] if cases is not None else None),
    )


def diversify(seeds, target, pop_size, valid=0.999, rounds=25, cache=None,
              executor=None, should_stop=None, max_telomere=MAX_TELOMERE,
              chromosome_count=None, families=DEFAULT_FAMILIES,
              on_progress=None, **_ignored):
    solutions = [clone_genome(genome) for genome in seeds[:pop_size]]
    frontier = list(solutions)
    for round_index in range(int(rounds)):
        if len(solutions) >= pop_size or (
                should_stop is not None and should_stop()):
            break
        proposals = []
        while len(proposals) < pop_size:
            parent = random.choice(frontier or solutions)
            proposals.append(mutate_functional(
                clone_genome(parent), 2.0, max_telomere=max_telomere,
                chromosome_count=chromosome_count, families=families,
                growth_seeds=target.inputs))
        fitnesses, _ = eval_batch_cases(
            proposals, target, cache, executor, should_stop)
        frontier = [
            genome for genome, fitness in zip(proposals, fitnesses)
            if fitness >= valid
        ]
        known = {genome_signature(genome) for genome in solutions}
        for genome in frontier:
            signature = genome_signature(genome)
            if signature not in known:
                solutions.append(genome)
                known.add(signature)
                if len(solutions) >= pop_size:
                    break
        if on_progress is not None:
            on_progress(round_index + 1, rounds, len(solutions))
    return solutions[:pop_size]
