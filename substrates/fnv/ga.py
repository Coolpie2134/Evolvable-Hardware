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

from .catalogue import DEFAULT_FAMILIES, normalise_families
from .evaluation import evaluate_functional_full
from .genome import (
    MAX_CHROMS, MAX_GENES, MAX_TELOMERE,
    input_layout_domain, input_layout_radius, random_functional_genome,
)

N_WORKERS = max(1, min((os.cpu_count() or 2) - 2, 16))
FITNESS_CACHE_MAX = 200_000
# A recombined child must be evaluated before mutation can tell selection
# whether its inherited output modules are actually compatible.  This stays
# small so the normal mutation/search stream remains the majority.
RECOMBINATION_EVALUATION_FRACTION = 0.10
# A specialist may contribute to the next generation, but never by being
# copied into it unchanged.  This quota creates crossed/mutated descendants
# of distinct hard-case experts and replaces FNV's former environmental
# parent survival.
SPECIALIST_BREEDING_FRACTION = 0.10
DEVELOPMENTAL_SEED_CANDIDATES = 1


def clone_genome(genome):
    """Copy mutable FNV structure without recursively walking scalar fields.

    FNV mutations edit gene objects in place, so unlike NV/LUT their genes
    cannot be shared between offspring.
    """
    from .construction_ga import clone_constructive
    return clone_constructive(genome)


def genome_signature(genome):
    from .construction_ga import branched_signature
    return branched_signature(genome)


def genome_morphology_signature(genome):
    from .construction_ga import branched_morphology_signature
    return branched_morphology_signature(genome)


def _recombination_mate_pool(population, first, candidates, signatures,
                             morphology_signatures):
    """Return mates whose inherited arms share ``first``'s pad environment.

    An FNV arm is a developmental program anchored to physical input pads.  A
    morphology match alone is not enough: two otherwise identical branch trees
    grown against different pad positions do not have interchangeable arms.
    Prefer the same-layout morphology match, then any same-layout mate.  Only
    fall back to another environment when the population has no compatible
    genome at all; ``crossover_branched`` turns that fallback into a clone
    rather than grafting an arm into the wrong environment.
    """
    distinct = [
        index for index in candidates
        if signatures[index] != signatures[first]]
    pool = distinct or list(candidates)
    first_layout = tuple(getattr(population[first], "input_layout", ()) or ())
    compatible = [
        index for index in pool
        if tuple(getattr(population[index], "input_layout", ()) or ())
        == first_layout]
    same_morphology = [
        index for index in compatible
        if morphology_signatures[index] == morphology_signatures[first]]
    return same_morphology or compatible or pool


def _topology_tuple(topology):
    return (
        topology.min_function_capacity,
        topology.total_function_capacity,
        topology.min_output_input_convergence,
        topology.total_output_input_connections,
        topology.min_output_branch_input_convergence,
        topology.total_output_branch_input_convergence,
        topology.min_output_input_edges,
        topology.total_output_input_edges,
        topology.min_output_branch_input_edges,
        topology.total_output_branch_input_edges,
        topology.min_output_input_proximity,
        topology.total_output_input_proximity,
        topology.live_output_roots,
        topology.output_integrating_nodes,
        topology.distinct_convergence_cones,
        topology.fully_integrating_nodes,
        topology.max_input_convergence,
        topology.distinct_convergence_cones,
        topology.fully_integrating_nodes,
        topology.integrating_nodes,
        topology.loop_rank,
        topology.loop_regions,
        topology.cyclic_nodes,
        topology.output_integrating_nodes,
        topology.integrating_nodes,
    )


def _evaluate_record(genome, target):
    fitness, cases, topology = evaluate_functional_full(
        genome, target, include_topology=True)
    selection_cases = _selection_case_vector(cases, target)
    output_scores = _output_balanced_scores(cases, target)
    total = target.n_inputs + len(target.outputs)
    # Correctness ties are resolved only by target-blind physical structure.
    # Truth-signature repertoire remains useful diagnostic telemetry, but it
    # must not steer selection toward Boolean-rich bodies under another name.
    topology_rank = _topology_tuple(topology)
    behavior_diagnostic = (
        topology.distinct_behaviors,
        topology.multi_input_behaviors,
        topology.fully_input_dependent_behaviors,
        topology.max_behavioral_inputs,
    )
    return (fitness, selection_cases, (total, total), 0.0, None, topology.score,
            topology_rank, behavior_diagnostic, output_scores,
            topology.function_capacities)


def select_developmental_seed(make_genome, attempts=DEVELOPMENTAL_SEED_CANDIDATES,
                              prefer_logic_capacity=False):
    """Choose the richest of a few random ontogenic starts, target-blindly."""
    from .construction import grow_functional
    from .evaluation import functional_topology, logic_morphology_capacity
    count = max(1, int(attempts))
    # The default is one candidate.  Scoring it would only re-grow the very
    # same organism before returning it, so skip that target-blind duplicate
    # work without changing the selected genome or random-number stream.
    if count == 1:
        return make_genome()
    candidates = [make_genome() for _ in range(count)]

    def key(genome):
        grid = grow_functional(genome, genome.input_layout)
        outputs = dict(getattr(genome, "output_layout", ()) or ())
        topology = functional_topology(
            grid, genome.input_layout, output_positions=outputs)
        capacity = (
            logic_morphology_capacity(
                grid, genome.input_layout, outputs)
            if prefer_logic_capacity else (0, 0))
        return capacity + _topology_tuple(topology)

    return max(candidates, key=key)


def _output_balanced_scores(cases, target):
    base = tuple(float(value) for value in cases)
    if getattr(target, "temporal", False) or not getattr(target, "cases", ()):
        return ()
    n_outputs = len(getattr(target, "outputs", ()))
    n_rows = len(target.cases)
    if not n_outputs or len(base) != n_rows * n_outputs:
        return ()
    output_scores = []
    for output_index in range(n_outputs):
        by_level = {0: [], 1: []}
        for row_index, (_inputs, expected) in enumerate(target.cases):
            level = 1 if expected[output_index] else 0
            by_level[level].append(
                base[row_index * n_outputs + output_index])
        level_means = [sum(values) / len(values)
                       for values in by_level.values() if values]
        output_scores.append(
            sum(level_means) / len(level_means) if level_means else 0.0)
    return tuple(output_scores)


def _selection_case_vector(cases, target):
    """Add coherent static-contract views for FNV selection only.

    A flat truth table exposes each (row, output) bit independently. On a
    multi-output target that lets a circuit selected for one easy bit erase a
    useful Sum or Carry specialist before recombination can join them. Retain
    the exact cells, then add generic contract-derived views: balanced accuracy
    for each output, joint correctness for each row, and weakest-output
    accuracy. Reported fitness and certification continue to use only the
    executable target contract; these extra views affect selection resolution,
    not what counts as a solution.
    """
    base = tuple(float(value) for value in cases)
    if getattr(target, "temporal", False) or not getattr(target, "cases", ()):
        return base
    n_outputs = len(getattr(target, "outputs", ()))
    n_rows = len(target.cases)
    expected_size = n_rows * n_outputs
    if not n_outputs or len(base) != expected_size:
        return base

    output_scores = _output_balanced_scores(cases, target)

    joint_rows = tuple(
        min(base[row_index * n_outputs:(row_index + 1) * n_outputs])
        for row_index in range(n_rows))
    return base + tuple(output_scores) + joint_rows + (min(output_scores),)


def _specialist_parent_indices(case_vecs, fitnesses, limit):
    """Choose distinct hard-case experts without reading target answers."""
    if not case_vecs or limit < 1 or any(vector is None for vector in case_vecs):
        return []
    vectors = [tuple(float(value) for value in vector)
               for vector in case_vecs]
    width = len(vectors[0]) if vectors else 0
    if width < 1 or any(len(vector) != width for vector in vectors):
        return []
    best_by_case = [max(vector[case] for vector in vectors)
                    for case in range(width)]
    # Cases with the lowest population ceiling are the missing pieces most at
    # risk of disappearing. Random tie order prevents a permanent row bias.
    case_order = list(range(width))
    random.shuffle(case_order)
    case_order.sort(key=lambda case: best_by_case[case])
    chosen = []
    behaviors = set()
    for case in case_order:
        candidates = [
            index for index, vector in enumerate(vectors)
            if vector not in behaviors]
        if not candidates:
            break
        winner = max(candidates, key=lambda index: (
            vectors[index][case], tuple(sorted(vectors[index])),
            sum(vectors[index]) / width, float(fitnesses[index])))
        chosen.append(winner)
        behaviors.add(vectors[winner])
        if len(chosen) >= limit:
            break
    return chosen


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
        genome._topology_rank = (
            tuple(record[6]) if len(record) > 6 else
            (0,) * 25)
        genome._behavior_diagnostic = (
            tuple(record[7]) if len(record) > 7 else (0, 0, 0, 0))
        genome._output_scores = (
            tuple(record[8]) if len(record) > 8 else ())
        genome._function_capacities = (
            tuple(record[9]) if len(record) > 9 else ())
    return [record[0] for record in records], [record[1] for record in records]


def _poisson(mean):
    if mean <= 0:
        return 0
    limit, count, product = math.exp(-mean), 0, 1.0
    while product > limit:
        count += 1
        product *= random.random()
    return count - 1


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


def mutate_functional(genome, mean_mutations=None, *,
                      max_telomere=MAX_TELOMERE, chromosome_count=None,
                      families=DEFAULT_FAMILIES, growth_seeds=None):
    from .construction_ga import mutate_branched, new_branched_chromosome
    enabled = normalise_families(families)
    # Pad placement is the input chromosome's job now (the "inputs" operator),
    # so there is no separate layout edit here to fall out of step with it.
    mutate_branched(
        genome, mean_mutations, enabled,
        len(getattr(genome, "input_layout", None) or growth_seeds or (1,)),
        max_telomere)
    if chromosome_count is not None:
        output_count = len(getattr(
            getattr(genome, "output_chromosome", None), "genes", ()))
        if 2 * int(chromosome_count) < output_count:
            raise ValueError("FNV needs at least one chromosome arm per output")
        while len(genome.chromosomes) < chromosome_count:
            genome.chromosomes.append(new_branched_chromosome(max_telomere))
        if len(genome.chromosomes) > chromosome_count:
            # Preserve rules by moving removed containers into the last
            # retained chromosome when capacity permits.
            retained = genome.chromosomes[:chromosome_count]
            overflow = [gene for chromosome in genome.chromosomes[
                chromosome_count:] for gene in chromosome.genes]
            for gene in overflow:
                destinations = [chromosome for chromosome in retained
                                if len(chromosome.genes) < MAX_GENES]
                if destinations:
                    random.choice(destinations).genes.append(gene)
            genome.chromosomes = retained
    return genome


def crossover_functional(parent_a, parent_b, families=DEFAULT_FAMILIES):
    """Each arm from either parent, then mutation in the breeder."""
    from .construction_ga import crossover_branched
    return crossover_branched(parent_a, parent_b, families)


def _topology_rank(genome):
    return tuple(getattr(
        genome, "_topology_rank",
        (0,) * 25))


def rank_key(genome, fitness):
    """Correctness first; FNV topology, never genome size, breaks final ties."""
    return (
        float(fitness),
        float(getattr(genome, "_robustness", 0.0)),
        float(getattr(genome, "_juvenile_score", 0.0)),
        _topology_rank(genome),
    )


def _lexicase(population, case_vectors):
    if not case_vectors or not case_vectors[0]:
        return None
    candidates = list(range(len(population)))
    for case in random.sample(
            range(len(case_vectors[0])), len(case_vectors[0])):
        values = [case_vectors[index][case] for index in candidates]
        best = max(values)
        if all(abs(value - round(value)) <= 1e-12 for value in values):
            epsilon = 0.0
        else:
            ordered = sorted(values)
            median = ordered[len(ordered) // 2]
            epsilon = sorted(
                abs(value - median)
                for value in values)[len(values) // 2]
        candidates = [
            index for index, value in zip(candidates, values)
            if value >= best - epsilon
        ]
        if len(candidates) == 1:
            break
    # Standard lexicase has already filtered on every behavioral case. FNV's
    # target-agnostic topology potential chooses among the remaining acceptable
    # candidates; exact topology ties remain random.
    best_topology = max(_topology_rank(population[index])
                        for index in candidates)
    candidates = [
        index for index in candidates
        if _topology_rank(population[index]) == best_topology
    ]
    return population[random.choice(candidates)]


def next_population(population, fitnesses, make_genome=None,
                    case_vecs=None, mean_mutations=None, selection=None,
                    ga_config=None, chromosome_count=None,
                    recombination=True, archive_parent=None,
                    stagnation=0,
                    families=DEFAULT_FAMILIES, growth_seeds=None, **_ignored):
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
        output_roles = tuple(
            str(gene.role) for gene in getattr(
                getattr(population[0], "output_chromosome", None),
                "genes", ()))
        make_genome = lambda: random_functional_genome(
            chromosome_count or 2, max_telomere=max_telomere,
            families=enabled,
            n_inputs=(
                len(reference_layout)
                if reference_layout is not None else None),
            output_roles=output_roles)
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

    children = []
    if archive_parent is not None and len(children) < count:
        children.append(mutate_functional(
            clone_genome(archive_parent), mean,
            max_telomere=max_telomere, chromosome_count=chromosome_count,
            families=enabled, growth_seeds=growth_seeds))
    immigrant_count = min(
        count - len(children), int(round(count * immigrant_fraction)))
    for _ in range(immigrant_count):
        children.append(make_genome())

    def parent_index():
        if selection == "lexicase":
            selected = _lexicase(population, case_vecs)
            if selected is not None:
                return next(
                    index for index, genome in enumerate(population)
                    if genome is selected)
        sample = random.sample(
            range(count), min(int(tournament_size), count))
        return max(sample, key=lambda index: rank_key(
            population[index], fitnesses[index]))

    signatures = [genome_signature(genome) for genome in population]
    morphology_signatures = [
        genome_morphology_signature(genome) for genome in population]

    def parent_pair():
        first = parent_index()
        if count == 1:
            return population[first], population[first]
        candidates = [index for index in range(count) if index != first]
        parent_pool = _recombination_mate_pool(
            population, first, candidates, signatures, morphology_signatures)
        if selection == "lexicase" and case_vecs:
            from runtime.escape import complementary_parent_index
            second = complementary_parent_index(
                first, parent_pool, case_vecs, fitnesses)
        else:
            second = parent_index()
            if (second == first or second not in parent_pool
                    or signatures[second] == signatures[first]):
                second = random.choice(parent_pool)
        return population[first], population[second]

    # Specialists receive offspring, not immortality. Each selected parent is
    # crossed with a behaviorally complementary compatible mate when crossing
    # is enabled, and every resulting child is mutated before evaluation.
    specialist_slots = min(
        max(0, count - len(children)),
        int(round(count * SPECIALIST_BREEDING_FRACTION)))
    for first in _specialist_parent_indices(
            case_vecs, fitnesses, specialist_slots):
        candidates = [index for index in range(count) if index != first]
        if candidates:
            parent_pool = _recombination_mate_pool(
                population, first, candidates, signatures,
                morphology_signatures)
            if selection == 'lexicase' and case_vecs:
                from runtime.escape import complementary_parent_index
                second = complementary_parent_index(
                    first, parent_pool, case_vecs, fitnesses)
            else:
                second = random.choice(parent_pool)
        else:
            second = first
        left, right = population[first], population[second]
        child = (crossover_functional(left, right, enabled)
                 if recombination and count > 1
                 else clone_genome(left))
        individual_mean = mean
        if escape is not None and escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate, mutation_rate_of
            inherit_mutation_rate(
                child, left, right, escape, mean, mutation_limit)
            individual_mean = mutation_rate_of(child, mean)
        mutate_functional(
            child, individual_mean, max_telomere=max_telomere,
            chromosome_count=chromosome_count, families=enabled,
            growth_seeds=growth_seeds)
        children.append(child)

    # The ordinary path below immediately mutates every crossover.  Keep a
    # bounded cohort intact long enough to measure the recombination itself;
    # these are not privileged survivors and are filtered by the same next
    # generation selection as every other child.
    crossover_slots = min(
        max(0, count - len(children)),
        max(1, int(round(count * RECOMBINATION_EVALUATION_FRACTION))))
    while recombination and count > 1 and crossover_slots > 0:
        left, right = parent_pair()
        child = crossover_functional(left, right, enabled)
        if escape is not None and escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate
            inherit_mutation_rate(
                child, left, right, escape, mean, mutation_limit)
        children.append(child)
        crossover_slots -= 1

    while len(children) < count:
        left, right = parent_pair()
        child = (
            crossover_functional(left, right, enabled)
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
            growth_seeds=growth_seeds)
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
