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
from .evaluation import (
    evaluate_functional_full, logic_morphology_capacities)
from .genome import (
    MAX_CHROMS, MAX_GENES, MAX_TELOMERE,
    input_layout_domain, input_layout_radius, random_functional_genome,
)

N_WORKERS = max(1, min((os.cpu_count() or 2) - 2, 16))
FITNESS_CACHE_MAX = 200_000
MORPHOLOGY_PARENT_RATE = 0.40
MORPHOLOGY_ELITE_FRACTION = 0.10
MODULE_ASSEMBLY_FRACTION = 0.04
FUNCTION_EXPLORER_FRACTION = 0.35
FUNCTION_EXPLORER_WARM_FRACTION = 0.08
FUNCTION_EXPLORER_PATIENCE = 4
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
    candidates = [make_genome() for _ in range(max(1, int(attempts)))]

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


def _contract_input_dependencies(target, output_index):
    """Inputs that can change one static output in the supplied contract.

    This extracts no gate or circuit recipe. It is only the generic Boolean
    influence relation used by stalled output-arm regrowth to avoid wiring a
    role to pads that provably cannot affect it. A partial table falls back to
    every input when it does not contain enough paired rows to decide safely.
    """
    rows = {
        tuple(int(bit) & 1 for bit in inputs): int(expected[output_index]) & 1
        for inputs, expected in getattr(target, "cases", ())
        if len(expected) > int(output_index)}
    n_inputs = int(getattr(target, "n_inputs", 0))
    dependencies = set()
    paired = False
    for inputs, value in rows.items():
        if len(inputs) != n_inputs:
            continue
        for index in range(n_inputs):
            other = list(inputs)
            other[index] ^= 1
            other = tuple(other)
            if other not in rows:
                continue
            paired = True
            if rows[other] != value:
                dependencies.add(index)
    if not paired:
        return tuple(range(n_inputs))
    return tuple(sorted(dependencies))


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


def plateau_rescue_candidates(
        genome, limit=48, max_telomere=MAX_TELOMERE,
        families=DEFAULT_FAMILIES, growth_seeds=(), focus_families=(),
        target=None):
    """Bounded local neighbours after a long stall.

    A branched rule is not tied to any one live tip - it fires wherever its
    neighbourhood occurs - so rescue is simply extra mutated descendants, the
    same thing the other substrates' archive rescue does. Static FNV contracts
    additionally regrow one role arm and choose the closest physically
    attainable fixed-gate signature; temporal rescue remains target-blind.
    """
    from .construction_ga import (
        clone_constructive, mutate_branched, randomize_branch_behavior,
        regrow_branch)
    limit = max(0, int(limit))
    if not limit:
        return []
    enabled = normalise_families(families)
    n_inputs = len(
        getattr(genome, "input_layout", None) or growth_seeds or (1,))
    proposals, seen = [], {genome_signature(genome)}
    output_genes = tuple(getattr(
        getattr(genome, "output_chromosome", None), "genes", ()))
    regrowth_families = (
        normalise_families(focus_families)
        if focus_families else enabled)
    for attempt_index in range(limit * 3):
        if len(proposals) >= limit:
            break
        candidate = clone_constructive(genome)
        static_contract = (
            target is not None
            and not getattr(target, "temporal", False)
            and bool(getattr(target, "cases", ()))
            and bool(output_genes))
        if static_contract:
            # Alternate attempted roles even when one candidate is a duplicate.
            # Keying this to len(proposals) repeatedly retried the same role
            # after a rejected duplicate and could starve its siblings.
            output_index = attempt_index % len(output_genes)
            branch_id = int(output_genes[output_index].branch_id)
            if (attempt_index // len(output_genes)) % 2 == 0:
                regrow_branch(
                    candidate, branch_id, regrowth_families, n_inputs,
                    max_telomere=max_telomere,
                    required_inputs=_contract_input_dependencies(
                        target, output_index))
            preferred = sum(
                (int(expected[output_index]) & 1) << row
                for row, (_inputs, expected) in enumerate(target.cases))
            randomize_branch_behavior(
                candidate, branch_id, n_inputs, limit=20_000,
                preferred_signature=preferred,
                input_patterns=tuple(inputs for inputs, _expected
                                     in target.cases))
        else:
            mutate_branched(
                candidate, None, enabled, n_inputs, max_telomere,
                focus_families=focus_families)
        signature = genome_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        proposals.append(candidate)
    return proposals


def mutate_functional(genome, mean_mutations=None, *,
                      max_telomere=MAX_TELOMERE, chromosome_count=None,
                      families=DEFAULT_FAMILIES, growth_seeds=None,
                      focus_families=()):
    from .construction_ga import mutate_branched, new_branched_chromosome
    enabled = normalise_families(families)
    # Pad placement is the input chromosome's job now (the "inputs" operator),
    # so there is no separate layout edit here to fall out of step with it.
    mutate_branched(
        genome, mean_mutations, enabled,
        len(getattr(genome, "input_layout", None) or growth_seeds or (1,)),
        max_telomere, focus_families=focus_families)
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


def _role_capacity(genome, role_index, output_genes=()):
    """Best physically consistent capacity sample available for one role."""
    if role_index < len(output_genes):
        label = int(output_genes[role_index].branch_id)
        sampled = getattr(genome, "_sampled_branch_capacities", {})
        if label in sampled:
            return int(sampled[label])
    capacities = getattr(genome, "_function_capacities", ())
    return int(capacities[role_index]) if role_index < len(capacities) else 0


def rank_key(genome, fitness):
    """Correctness first; FNV topology, never genome size, breaks final ties."""
    return (
        float(fitness),
        float(getattr(genome, "_robustness", 0.0)),
        float(getattr(genome, "_juvenile_score", 0.0)),
        _topology_rank(genome),
    )


def initialization_families(families, target):
    """Choose a contract-class seed palette without narrowing evolution.

    Exhaustive Boolean targets need a dense gate scaffold before routing,
    holds and oscillators become useful. Drawing initial genes uniformly
    family-first from the entire enabled bank diluted that scaffold and
    repeatedly stranded Half Adder at one missing output. Seed those runs from
    LOGIC; focused mutation may then introduce DELAY routing/fan-out while all
    mutations still receive the complete user-selected bank. Temporal runs and
    intentionally logic-free banks are unchanged.
    """
    enabled = normalise_families(families)
    logic_contract = (
        bool(getattr(target, "combinational_cases", ()))
        or (not getattr(target, "temporal", False)
            and bool(getattr(target, "cases", ()))))
    logic_families = tuple(
        family for family in ("LOGIC", "DELAY") if family in enabled)
    if logic_contract and "LOGIC" in enabled:
        return frozenset(logic_families)
    return enabled


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
                    stagnation=0, rescue_candidates=None,
                    families=DEFAULT_FAMILIES, growth_seeds=None,
                    focus_families=(), target=None, **_ignored):
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

    children = [clone_genome(genome)
                for genome in list(rescue_candidates or ())[:count]]
    if archive_parent is not None and len(children) < count:
        children.append(mutate_functional(
            clone_genome(archive_parent), mean,
            max_telomere=max_telomere, chromosome_count=chromosome_count,
            families=enabled, growth_seeds=growth_seeds,
            focus_families=focus_families))
    output_genes = tuple(getattr(
        getattr(population[0], "output_chromosome", None), "genes", ()))
    # Output-rooted heredity only pays off if independently useful role modules
    # get a chance to coexist. Ordinary crossover made that join and then
    # immediately applied ~4 mutations, often destroying Sum or Carry before
    # the assembled child was ever evaluated. Build a tiny unmutated cohort
    # from the best role specialists that share one input environment. This is
    # generic multi-output recombination: it uses measured per-role quality,
    # never a target name, gate recipe, route, or expected truth-table value.
    if len(output_genes) > 1 and len(children) < count:
        from .construction_ga import assemble_role_modules
        groups = {}
        for index, genome in enumerate(population):
            groups.setdefault(
                tuple(getattr(genome, "input_layout", ()) or ()), []).append(index)
        assemblies = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            donor_indices = []
            donor_scores = []
            valid = True
            for role_index, _output in enumerate(output_genes):
                eligible = [
                    index for index in indices
                    if len(getattr(population[index], "_output_scores", ()))
                    > role_index]
                if not eligible:
                    valid = False
                    break
                donor_index = max(
                    eligible,
                    key=lambda index: (
                        population[index]._output_scores[role_index],
                        rank_key(population[index], fitnesses[index])))
                donor_indices.append(donor_index)
                donor_scores.append(
                    population[donor_index]._output_scores[role_index])
            if not valid or len(set(donor_indices)) < 2:
                continue
            base_index = max(
                indices,
                key=lambda index: rank_key(population[index], fitnesses[index]))
            donors = {
                int(output_genes[role_index].branch_id): population[donor_index]
                for role_index, donor_index in enumerate(donor_indices)}
            candidate = assemble_role_modules(
                population[base_index], donors, enabled)
            assemblies.append((
                (min(donor_scores), sum(donor_scores),
                 rank_key(population[base_index], fitnesses[base_index])),
                candidate))
        assembly_count = min(
            count - len(children),
            max(1, int(round(count * MODULE_ASSEMBLY_FRACTION))))
        for _quality, candidate in sorted(
                assemblies, key=lambda row: row[0], reverse=True)[:assembly_count]:
            children.append(candidate)
    # Preserve the best complete behavior of each inherited output module.
    # Row-level environmental memory can retain a genome that happens to pass
    # one Sum row while deleting the only arm that computes Sum coherently over
    # all rows.  With output-rooted heredity that is equivalent to throwing away
    # a useful organ before crossover can use it.  One unmutated specialist per
    # role is small, target-generic, and ordered early enough to survive the
    # generational merge; it uses the same scored role cells as selection and
    # does not prescribe a function or morphology.
    role_elite_signatures = set()
    for role_index, _output in enumerate(output_genes):
        eligible = [
            index for index, genome in enumerate(population)
            if len(getattr(genome, "_output_scores", ())) > role_index]
        if not eligible or len(children) >= count:
            continue
        index = max(
            eligible,
            key=lambda candidate: (
                population[candidate]._output_scores[role_index],
                rank_key(population[candidate], fitnesses[candidate])))
        signature = genome_signature(population[index])
        if signature in role_elite_signatures:
            continue
        children.append(clone_genome(population[index]))
        role_elite_signatures.add(signature)
    # A parent selected for computational morphology still vanished under
    # generational replacement unless it was also a current contract expert.
    # Preserve a small target-blind repertoire reserve so converged source
    # cones and internally useful multi-input functions can be extended across
    # generations. This is deliberately independent of genome size and target
    # answers; the ordinary contract reserve retains most of the population.
    morphology_count = min(
        count - len(children),
        max(1, int(round(count * MORPHOLOGY_ELITE_FRACTION))))
    morphology_order = sorted(
        range(count),
        key=lambda index: _topology_rank(population[index]),
        reverse=True)
    seen_morphologies = set()
    if morphology_count:
        role_count = len(output_genes)
        # A body may devote its available territory to one exceptionally rich
        # output cone. Preserve the best cone for EACH role before asking for a
        # balanced organism; crossover/regrowth can then combine specialists.
        for role_index in range(role_count):
            index = max(
                range(count),
                key=lambda candidate: (
                    _role_capacity(
                        population[candidate], role_index, output_genes),
                    _topology_rank(population[candidate])))
            signature = genome_morphology_signature(population[index])
            if signature in seen_morphologies:
                continue
            children.append(clone_genome(population[index]))
            seen_morphologies.add(signature)
            if len(seen_morphologies) >= morphology_count:
                break
        for index in morphology_order:
            signature = genome_morphology_signature(population[index])
            if signature in seen_morphologies:
                continue
            children.append(clone_genome(population[index]))
            seen_morphologies.add(signature)
            if len(seen_morphologies) >= morphology_count:
                break
    # A connected output tree can be separated from a better gate assignment
    # by several individually worse substitutions (Majority-3's familiar 7/8
    # basin is the small example). Dedicate a bounded cohort to resampling the
    # fixed components on ONE intact output arm. Static contracts may request
    # their closest attainable truth signature; physical routes, contexts,
    # pads and all other role modules stay unchanged.
    from .construction_ga import (
        mutate_branch_function, randomize_branch_behavior,
        randomize_branch_functions, regrow_branch)
    deep_function_search = int(stagnation) >= FUNCTION_EXPLORER_PATIENCE
    regrowth_families = (
        normalise_families(focus_families)
        if focus_families else enabled)
    function_fraction = (
        FUNCTION_EXPLORER_FRACTION if deep_function_search
        else FUNCTION_EXPLORER_WARM_FRACTION)
    function_count = min(
        count - len(children),
        max(1, int(round(count * function_fraction))))
    behavior_order = sorted(
        range(count),
        key=lambda index: rank_key(population[index], fitnesses[index]),
        reverse=True)
    for slot in range(function_count):
        preserve_index = slot % max(1, len(output_genes))
        specialists = [
            index for index, genome in enumerate(population)
            if len(getattr(genome, "_output_scores", ())) > preserve_index]
        if specialists and len(output_genes) > 1:
            preserved_label = int(output_genes[preserve_index].branch_id)
            mutable = [
                (index, int(gene.branch_id))
                for index, gene in enumerate(output_genes)
                if int(gene.branch_id) != preserved_label]
            mutable_index, branch_id = random.choice(mutable)
            preferred_signature = (
                sum((int(expected[mutable_index]) & 1) << row
                    for row, (_inputs, expected) in enumerate(target.cases))
                if (target is not None
                    and not getattr(target, "temporal", False)
                    and getattr(target, "cases", ())
                    and all(len(expected) > mutable_index
                            for _inputs, expected in target.cases))
                else None)
            input_patterns = (
                tuple(inputs for inputs, _expected in target.cases)
                if preferred_signature is not None else None)
            required_inputs = (
                _contract_input_dependencies(target, mutable_index)
                if preferred_signature is not None else None)
            source_index = max(
                specialists,
                key=lambda index: (
                    population[index]._output_scores[preserve_index],
                    _role_capacity(
                        population[index], mutable_index, output_genes),
                    _topology_rank(population[index]),
                    rank_key(population[index], fitnesses[index])))
        else:
            order = morphology_order if slot % 2 == 0 else behavior_order
            source_index = order[slot % len(order)]
            branch_id = None
            preferred_signature = None
            input_patterns = None
            required_inputs = None
        candidate = clone_genome(population[source_index])
        if not deep_function_search:
            mutate_branch_function(candidate, branch_id=branch_id)
        elif branch_id is not None and slot % 4 == 0:
            from .construction import grow_functional
            n_inputs = len(getattr(candidate, "input_layout", ()) or (1,))
            variants = []
            attempt_count = 2
            for _attempt in range(attempt_count):
                variant = clone_genome(candidate)
                regrow_branch(
                    variant, branch_id, regrowth_families, n_inputs,
                    max_telomere=max_telomere,
                    required_inputs=required_inputs)
                randomize_branch_behavior(
                    variant, branch_id, n_inputs, limit=1000,
                    preferred_signature=preferred_signature,
                    input_patterns=input_patterns)
                grid = grow_functional(variant, variant.input_layout)
                capacities = logic_morphology_capacities(
                    grid, variant.input_layout,
                    dict(variant.output_layout))
                variants.append((
                    (evaluate_functional_full(variant, target)[0]
                     if target is not None else 0.0),
                    _role_capacity(
                        variant, mutable_index, output_genes),
                    sum(capacities), variant))
            candidate = max(variants, key=lambda row: row[:3])[3]
        elif slot % 7 == 6:
            randomize_branch_functions(candidate, branch_id=branch_id)
        else:
            if (branch_id is None or not randomize_branch_behavior(
                    candidate, branch_id,
                    len(getattr(candidate, "input_layout", ()) or (1,)),
                    preferred_signature=preferred_signature,
                    input_patterns=input_patterns)):
                mutate_branch_function(candidate, branch_id=branch_id)
        children.append(candidate)
    immigrant_count = min(
        count - len(children), int(round(count * immigrant_fraction)))
    for _ in range(immigrant_count):
        children.append(make_genome())

    def parent_index():
        # A small morphology niche keeps source cones that have actually begun
        # to meet. Without it, Full Adder populations grew hundreds of nodes
        # yet repeatedly had zero multi-input convergence, making their best
        # possible behavior a one-input approximation. This is target-blind:
        # the same reachable convergence/feedback key is used for every FNV
        # contract and never changes reported fitness.
        if random.random() < MORPHOLOGY_PARENT_RATE:
            sample = random.sample(
                range(count), min(int(tournament_size), count))
            return max(sample, key=lambda index: _topology_rank(
                population[index]))
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
        distinct = [index for index in candidates
                    if signatures[index] != signatures[first]]
        same_morphology = [
            index for index in distinct
            if morphology_signatures[index] == morphology_signatures[first]]
        # Inputs are the shared environment of every output module. Prefer a
        # matching pad geometry when available; each output root itself travels
        # atomically with its arm during crossover.
        first_layout = getattr(population[first], "input_layout", None)
        compatible = [
            index for index in (distinct or candidates)
            if getattr(population[index], "input_layout", None) == first_layout
        ]
        # Function variants of one body plan are the highest-value mate: their
        # intramodule crossover can combine gate alleles without rebuilding a
        # single wire. Structurally different role modules remain the fallback.
        parent_pool = same_morphology or compatible or distinct or candidates
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
