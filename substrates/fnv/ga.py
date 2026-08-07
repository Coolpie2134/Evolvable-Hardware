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
    BY_ID, COMPONENTS, DEFAULT_FAMILIES, behavior_component_ids,
    enabled_component_ids,
    local_component_ids, normalise_families,
)
from .evaluation import evaluate_functional_full
from .genome import (
    MAX_CHROMS, MAX_GENES, MAX_TELOMERE, Chromosome, FunctionalGene, Genome,
    SEED_STATES, input_layout_domain, input_layout_radius, is_constructive,
    is_typed,
    random_component_id,
    random_functional_chromosome, random_functional_gene,
    random_functional_genome,
)

N_WORKERS = max(1, min((os.cpu_count() or 2) - 2, 16))
FITNESS_CACHE_MAX = 200_000
GENE_FIELDS = ("ctx_l", "ctx_r", "ctx_d", "self_in", "self_out")
MORPHOLOGY_PARENT_RATE = 0.15
MORPHOLOGY_ELITE_FRACTION = 0.10


def clone_genome(genome):
    """Copy mutable FNV structure without recursively walking scalar fields.

    FNV mutations edit gene objects in place, so unlike NV/LUT their genes
    cannot be shared between offspring.  They contain only scalar alleles,
    however, making one shallow copy per gene equivalent to ``deepcopy`` while
    avoiding its memo/dispatch overhead throughout reproduction.
    """
    if is_constructive(genome):
        from .construction_ga import clone_constructive
        return clone_constructive(genome)
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
    if is_constructive(genome):
        from .construction_ga import constructive_signature
        return (getattr(genome, "encoding", None),
                constructive_signature(genome))
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
    selection_cases = _selection_case_vector(cases, target)
    total = target.n_inputs + len(target.outputs)
    # Correctness ties are resolved only by target-blind physical structure.
    # Truth-signature repertoire remains useful diagnostic telemetry, but it
    # must not steer selection toward Boolean-rich bodies under another name.
    topology_rank = (
        topology.max_input_convergence,
        topology.distinct_convergence_cones,
        topology.fully_integrating_nodes,
        topology.integrating_nodes,
        topology.loop_rank,
        topology.loop_regions,
        topology.cyclic_nodes,
        topology.reachable_edges,
        topology.reachable_nodes,
    )
    behavior_diagnostic = (
        topology.distinct_behaviors,
        topology.multi_input_behaviors,
        topology.fully_input_dependent_behaviors,
        topology.max_behavioral_inputs,
    )
    return (fitness, selection_cases, (total, total), 0.0, None, topology.score,
            topology_rank, behavior_diagnostic)


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

    output_scores = []
    for output_index in range(n_outputs):
        by_level = {0: [], 1: []}
        for row_index, (_inputs, expected) in enumerate(target.cases):
            level = 1 if expected[output_index] else 0
            by_level[level].append(
                base[row_index * n_outputs + output_index])
        level_means = [
            sum(values) / len(values)
            for values in by_level.values() if values]
        output_scores.append(
            sum(level_means) / len(level_means) if level_means else 0.0)

    joint_rows = tuple(
        min(base[row_index * n_outputs:(row_index + 1) * n_outputs])
        for row_index in range(n_rows))
    return base + tuple(output_scores) + joint_rows + (min(output_scores),)


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
            (0, 0, 0, 0, 0, 0, 0, 0, 0))
        genome._behavior_diagnostic = (
            tuple(record[7]) if len(record) > 7 else (0, 0, 0, 0))
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


def _mutate_allele(genome, families, preferred_loci=(), focused_loci=(),
                   focused_families=()):
    locus = None
    if focused_loci and random.random() < 0.95:
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
    # A target-class focus is a search bias, not a hidden family restriction.
    # Developmentally active rules stay inside the useful family. Inactive
    # rules and structural/new-gene mutations keep the complete user-enabled
    # bank, so every allowed physical component remains reachable without
    # repeatedly destroying the scaffold that made a rule active.
    focused = bool(locus is not None and locus in focused_loci)
    mutation_families = (
        focused_families
        if focused and focused_families
        else families)
    # EMPTY remains an independently reachable absence allele.  Within a
    # family, route/output-count/timing neighbors dominate.
    behavior_choices = (
        behavior_component_ids(current)
        if (field == "self_out" and current != 0
            and BY_ID[current].family == "LOGIC"
            and BY_ID[current].family in mutation_families)
        else ())
    if behavior_choices and random.random() < 0.35:
        replacement = random.choice(behavior_choices)
    else:
        replacement = _different_state(
            current, mutation_families,
            empty_probability=(0.12 if field == "self_in" else 0.08))
    setattr(gene, field, replacement)


def _diverged_duplicate(gene, families, focused_families=()):
    """Copy a rule and immediately make the later copy selectable.

    An exact duplicate after its source can never win the developmental
    first-match tie. Diverging one context coordinate creates a neighboring
    domain, while diverging the output gives that domain a new physical role.
    """
    duplicate = copy.copy(gene)
    context = random.choice(("ctx_l", "ctx_r", "ctx_d", "self_in"))
    setattr(duplicate, context, _different_state(
        getattr(duplicate, context), families))
    output_families = focused_families or families
    choices = behavior_component_ids(duplicate.self_out)
    duplicate.self_out = (
        random.choice(choices)
        if (choices and duplicate.self_out != 0
            and BY_ID[duplicate.self_out].family in output_families)
        else _different_state(duplicate.self_out, output_families))
    return duplicate


def _mutate_observed_context(genome, contexts, families,
                             focused_families=()):
    """Insert a random rule for a context development actually encounters.

    The physical component is not chosen from target behavior. Exact context
    capture merely avoids the overwhelmingly common no-op where a new random
    associative rule never wins any lookup. Inserting first makes a captured
    exact match authoritative while retaining indirect repeated expression.
    """
    if not contexts:
        return False
    candidates = [chromosome for chromosome in genome.chromosomes
                  if len(chromosome.genes) < MAX_GENES]
    if not candidates:
        return False
    context_pool = tuple(contexts)
    # Prefer genuine live frontiers. The previous occupancy-only draw commonly
    # inserted a gate next to a component whose facing port was not an output,
    # creating impressive-looking but electrically disconnected tissue.
    live_frontier = tuple(
        context for context in context_pool
        if int(context[3]) == 0
        and _driven_context_directions(*context[:3]))
    ctx_l, ctx_r, ctx_d, self_in = random.choice(
        live_frontier if live_frontier and random.random() < 0.75
        else context_pool)
    output_families = focused_families or families
    gene = FunctionalGene(
        ctx_l=int(ctx_l), ctx_r=int(ctx_r), ctx_d=int(ctx_d),
        self_in=int(self_in),
        self_out=_contextual_component_id(
            ctx_l, ctx_r, ctx_d, self_in, output_families))
    # Lookup order is chromosome-major.  An exact rule inserted into a later
    # chromosome can still lose a zero-distance tie to an older exact rule in
    # chromosome 0, contradicting this operator's locality guarantee.  Use the
    # earliest chromosome with capacity so the captured context really wins.
    chromosome = candidates[0]
    chromosome.genes.insert(0, gene)
    _normalize_split(chromosome)
    return True


def _specialize_observed_context(genome, context, families,
                                 focused_families=(), replacement=None):
    """Split one encountered context from a broad developmental rule.

    The existing winner supplies the component route.  A different basic gate
    behavior on those same physical pins is then installed as an exact,
    first-priority rule.  The broad source rule remains intact for the rest of
    its context domain, making the developmental representation substantially
    more local without translating a desired phenotype back into a genome.
    """
    candidates = [chromosome for chromosome in genome.chromosomes
                  if len(chromosome.genes) < MAX_GENES]
    if not candidates:
        return False
    from .growth import lookup
    ctx_l, ctx_r, ctx_d, self_in = tuple(map(int, context))
    current = int(lookup(genome, ctx_l, ctx_r, ctx_d, self_in))
    output_families = normalise_families(focused_families or families)
    alternatives = tuple(
        component_id for component_id in behavior_component_ids(current)
        if BY_ID[component_id].family in output_families)
    if replacement is None:
        if not alternatives:
            return False
        replacement = random.choice(alternatives)
    elif int(replacement) not in alternatives:
        return False
    candidates[0].genes.insert(0, FunctionalGene(
        ctx_l=ctx_l, ctx_r=ctx_r, ctx_d=ctx_d, self_in=self_in,
        self_out=int(replacement)))
    _normalize_split(candidates[0])
    return True


def _contextual_component_id(ctx_l, ctx_r, ctx_d, self_in, families):
    """Choose hardware whose pins can consume an observed local context.

    This is target-blind physical locality, not inverse development: no desired
    output or target answer is inspected. A new frontier rule should read the
    neighbours that caused it to be expressed instead of usually pointing its
    input pins into empty space. With two occupied sides, prefer a two-input
    gate; with one, prefer a unary transport/fan-out component. Mature-cell
    rules retain the same principle while remaining free to change behavior.
    """
    enabled = normalise_families(families)
    states = {"L": int(ctx_l), "R": int(ctx_r), "D": int(ctx_d)}
    driven = _driven_context_directions(ctx_l, ctx_r, ctx_d)
    candidates = [
        entry for entry in COMPONENTS
        if entry.id != 0 and entry.family in enabled and entry.outputs
    ]
    if driven:
        fully_driven = [
            entry for entry in candidates
            if entry.inputs and set(entry.inputs).issubset(driven)]
        if fully_driven:
            max_inputs = max(len(entry.inputs) for entry in fully_driven)
            candidates = [entry for entry in fully_driven
                          if len(entry.inputs) == max_inputs]
        else:
            partly_driven = [
                entry for entry in candidates
                if driven.intersection(entry.inputs)]
            if partly_driven:
                candidates = partly_driven
    if self_in:
        # Prefer a local route/behavior neighbor when that is also physically
        # compatible with the encountered context.
        local = set(local_component_ids(int(self_in), enabled))
        compatible_local = [entry for entry in candidates if entry.id in local]
        if compatible_local:
            candidates = compatible_local
    if not candidates:
        return random_component_id(enabled, empty_probability=0.05)
    return random.choice(candidates).id


def _driven_context_directions(ctx_l, ctx_r, ctx_d):
    """Receiver-local sides whose neighboring component drives this edge."""
    states = {"L": int(ctx_l), "R": int(ctx_r), "D": int(ctx_d)}
    return frozenset(
        direction for direction, state in states.items()
        if state != 0 and (
            state in SEED_STATES or direction in BY_ID[state].outputs))


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
        families=DEFAULT_FAMILIES, growth_seeds=(), focus_families=()):
    """Enumerate honest local program neighbours after a long FNV stall.

    Random mutation repeatedly spends events on inactive associative rules.
    During a plateau, expose a bounded set of ordinary genomes that change one
    expressed gate behavior/route, capture one observed context, move one input
    pad, or adjust one telomere. The proposals never inspect expected outputs
    and are evaluated by the unchanged contract, so this improves locality
    without inverse-growing a target-shaped circuit.
    """
    limit = max(0, int(limit))
    if not limit:
        return []
    enabled = normalise_families(families)
    focused = tuple(family for family in focus_families if family in enabled)
    if is_typed(genome):
        # The constructive rescue enumerates placements at live frontier tips.
        # A typed gene has no single tip - it is a rule that fires wherever its
        # pattern occurs - so rescue is simply extra mutated descendants, which
        # is what the other substrates' archive rescue does.
        from .construction_ga import clone_constructive, mutate_typed
        n_inputs = len(
            getattr(genome, "input_layout", None) or growth_seeds or (1,))
        proposals, seen = [], {genome_signature(genome)}
        for _ in range(limit * 3):
            if len(proposals) >= limit:
                break
            candidate = clone_constructive(genome)
            mutate_typed(candidate, None, enabled, n_inputs, max_telomere)
            signature = genome_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            proposals.append(candidate)
        return proposals
    if is_constructive(genome):
        from .construction_ga import plateau_candidates_constructive
        return plateau_candidates_constructive(
            genome, limit, enabled, growth_seeds, focused,
            layout_mutator=lambda candidate: mutate_input_layout(
                candidate, max_telomere))
    seeds = (
        tuple(genome.input_layout)
        if getattr(genome, "input_layout", None) is not None
        else tuple(growth_seeds or ()))
    from .growth import active_gene_loci_and_contexts
    active, contexts = active_gene_loci_and_contexts(genome, seeds)
    proposals = []
    seen = {genome_signature(genome)}

    def add(candidate):
        signature = genome_signature(candidate)
        if signature in seen:
            return
        seen.add(signature)
        proposals.append(candidate)

    loci = list(active)
    random.shuffle(loci)
    for chromosome_index, gene_index in loci:
        gene = genome.chromosomes[chromosome_index].genes[gene_index]
        current = int(gene.self_out)
        replacements = list(dict.fromkeys(
            list(behavior_component_ids(current))
            + list(local_component_ids(current, focused or enabled))))
        random.shuffle(replacements)
        for replacement in replacements:
            if BY_ID[replacement].family not in enabled:
                continue
            candidate = clone_genome(genome)
            candidate.chromosomes[chromosome_index].genes[
                gene_index].self_out = replacement
            add(candidate)

    from .growth import lookup
    context_order = list(contexts)
    random.shuffle(context_order)
    for context in context_order:
        current = int(lookup(genome, *context))
        replacements = [
            component_id for component_id in behavior_component_ids(current)
            if BY_ID[component_id].family in (focused or enabled)]
        random.shuffle(replacements)
        # Enumerate local gate substitutions before route-changing growth.
        # This is the missing edit for repeated broad rules: one exact context
        # can change AND/OR/XOR/VETO behavior without changing its pins or the
        # other cells expressed by the old rule.
        for replacement in replacements:
            candidate = clone_genome(genome)
            if _specialize_observed_context(
                    candidate, context, enabled, focused, replacement):
                add(candidate)
        for _ in range(2):
            candidate = clone_genome(genome)
            if _mutate_observed_context(
                    candidate, (context,), enabled, focused):
                add(candidate)

    for _ in range(min(6, max(0, len(getattr(
            genome, "input_layout", ()) or ()) - 1) * 2)):
        candidate = clone_genome(genome)
        if mutate_input_layout(candidate, max_telomere):
            add(candidate)

    for chromosome_index, chromosome in enumerate(genome.chromosomes):
        for step in (-1, 1):
            telomere = max(
                1, min(int(max_telomere), int(chromosome.telomere) + step))
            if telomere == chromosome.telomere:
                continue
            candidate = clone_genome(genome)
            candidate.chromosomes[chromosome_index].telomere = telomere
            add(candidate)

    # Bias the bounded batch toward direct expressed-rule repairs while still
    # varying which part of a large neighbourhood is exposed on later stalls.
    head = proposals[:max(1, limit // 2)]
    tail = proposals[len(head):]
    random.shuffle(tail)
    return (head + tail)[:limit]


def _mutate_structure(genome, families, max_telomere, chromosome_count,
                      preferred_loci=(), focused_loci=(),
                      focused_families=()):
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
        chromosome.genes.insert(index + 1, _diverged_duplicate(
            chromosome.genes[index], families, focused_families))
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
            genome, families, preferred_loci, focused_loci,
            focused_families)
    for item in genome.chromosomes:
        _normalize_split(item)


def mutate_functional(genome, mean_mutations=None, *,
                      max_telomere=MAX_TELOMERE, chromosome_count=None,
                      families=DEFAULT_FAMILIES, growth_seeds=None,
                      focus_families=()):
    enabled = normalise_families(families)
    if is_constructive(genome):
        from .construction_ga import (
            mutate_constructive, mutate_typed, new_typed_chromosome)
        # Input pads are named roots, so a local pad move translates only the
        # branches that depend on that source. It is an independent physical
        # mutation, not a rewrite of placement coordinates.
        if (getattr(genome, "input_layout", None) is not None
                and len(genome.input_layout) > 1
                and random.random() < 0.04):
            mutate_input_layout(genome, max_telomere)
        if is_typed(genome):
            # A typed gene is a reusable rule, so the frontier-walking operators
            # do not apply: there is no single live tip to extend. Its menu is
            # the one the other substrates use - retarget, duplicate, add,
            # delete, and adjust the chromosome's growth lifespan.
            mutate_typed(
                genome, mean_mutations, enabled,
                len(getattr(genome, "input_layout", None) or growth_seeds
                    or (1,)),
                max_telomere)
        else:
            # The germline telomere is the constructive organism's growth
            # radius, so a genome inherited from a looser run has to be brought
            # under this run's ceiling. Downward only, and never onto a legacy
            # genome: a telomere of 1 means unbounded (see growth_radius), and
            # raising it to 2 would collapse a resumed checkpoint's body.
            ceiling = max(2, int(max_telomere))
            for chromosome in genome.chromosomes:
                if int(chromosome.telomere) > 1:
                    chromosome.telomere = min(
                        int(chromosome.telomere), ceiling)
            mutate_constructive(
                genome, mean_mutations, enabled, growth_seeds, focus_families)
        if chromosome_count is not None:
            while len(genome.chromosomes) < chromosome_count:
                # Both encodings read the telomere now - as a lifespan in
                # developmental rounds for typed, as a growth radius for
                # constructive - so a new container must be born at the run's
                # ceiling rather than at a hardcoded 1.
                genome.chromosomes.append(
                    new_typed_chromosome(max_telomere) if is_typed(genome)
                    else Chromosome(
                        genes=[], split=0, tag=random.randint(0, 999),
                        telomere=max(2, int(max_telomere))))
            if len(genome.chromosomes) > chromosome_count:
                # Preserve placements by moving removed containers into the
                # last retained chromosome when capacity permits.
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
    effective_seeds = (
        tuple(getattr(genome, "input_layout"))
        if getattr(genome, "input_layout", None) is not None
        else tuple(growth_seeds or ()))
    focused_families = tuple(
        family for family in focus_families if family in enabled)

    def development_hints():
        preferred = ()
        contexts = ()
        if effective_seeds:
            from .growth import active_gene_loci_and_contexts
            preferred, contexts = active_gene_loci_and_contexts(
                genome, effective_seeds)
        focused = tuple(
            (chromosome_index, gene_index)
            for chromosome_index, gene_index in preferred
            if (
                chromosome_index < len(genome.chromosomes)
                and gene_index < len(
                    genome.chromosomes[chromosome_index].genes)
                and getattr(
                    genome.chromosomes[chromosome_index].genes[gene_index],
                    "self_out", 0) != 0
                and BY_ID[
                    genome.chromosomes[chromosome_index].genes[
                        gene_index].self_out
                ].family in focused_families
            )
        )
        return preferred, contexts, focused

    preferred_loci, observed_contexts, focused_loci = development_hints()
    mean = 4.0 if mean_mutations is None else max(0.0, float(mean_mutations))
    count = max(1, _poisson(mean))
    for mutation_index in range(count):
        refresh = False
        if (getattr(genome, "input_layout", None) is not None
                and len(genome.input_layout) > 1
                and random.random() < 0.12
                and mutate_input_layout(genome, max_telomere)):
            refresh = True
        elif (observed_contexts and random.random() < 0.25
              and (
                  (random.random() < 0.70 and _specialize_observed_context(
                      genome, random.choice(observed_contexts), enabled,
                      focused_families))
                  or _mutate_observed_context(
                      genome, observed_contexts, enabled,
                      focused_families))):
            refresh = True
        elif random.random() < 0.78:
            _mutate_allele(
                genome, enabled, preferred_loci, focused_loci,
                focused_families)
        else:
            _mutate_structure(
                genome, enabled, max_telomere, chromosome_count,
                preferred_loci, focused_loci, focused_families)
            refresh = True
        # Insertions/deletions shift loci, and context/layout edits change the
        # phenotype on which the next edit should build.  Continuing with the
        # pre-edit trace mutated unrelated genes and prevented a multi-event
        # child from assembling a causal chain of basic components.
        if refresh and mutation_index + 1 < count:
            effective_seeds = (
                tuple(getattr(genome, "input_layout"))
                if getattr(genome, "input_layout", None) is not None
                else tuple(growth_seeds or ()))
            preferred_loci, observed_contexts, focused_loci = (
                development_hints())
    if chromosome_count is not None:
        while len(genome.chromosomes) < chromosome_count:
            genome.chromosomes.append(random_functional_chromosome(
                max_telomere=max_telomere, families=enabled))
        if len(genome.chromosomes) > chromosome_count:
            del genome.chromosomes[chromosome_count:]
    return genome


def crossover_functional(parent_a, parent_b, families=DEFAULT_FAMILIES):
    """Chromosome/gene recombination followed by mutation in the breeder."""
    if is_constructive(parent_a) or is_constructive(parent_b):
        if not (is_constructive(parent_a) and is_constructive(parent_b)):
            return clone_genome(parent_a)
        if is_typed(parent_a) and is_typed(parent_b):
            from .construction_ga import crossover_typed
            return crossover_typed(parent_a, parent_b, families)
        if is_typed(parent_a) or is_typed(parent_b):
            return clone_genome(parent_a)
        from .construction_ga import crossover_constructive
        return crossover_constructive(parent_a, parent_b, families)
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
    crossover_draw = random.random()
    if shared >= 2 and crossover_draw < 0.55:
        # A chromosome is a co-adapted developmental rule module. Replacing a
        # proper subset of whole chromosomes lets complementary specialists
        # coexist without first tearing apart the contexts that make each
        # module express. Fine-grained suffix/field crossover remains below as
        # the alternate path for discovering new rules inside a module.
        inherited = set(random.sample(
            range(shared), random.randint(1, shared - 1)))
        for index in inherited:
            source = parent_b.chromosomes[index]
            copied = copy.copy(source)
            copied.genes = [copy.copy(gene) for gene in source.genes]
            child.chromosomes[index] = copied
        return child
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


def _topology_rank(genome):
    return tuple(getattr(
        genome, "_topology_rank",
        (0, 0, 0, 0, 0, 0, 0, 0, 0)))


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
        for index in morphology_order:
            topology = _topology_rank(population[index])
            if topology in seen_morphologies:
                continue
            children.append(clone_genome(population[index]))
            seen_morphologies.add(topology)
            if len(seen_morphologies) >= morphology_count:
                break
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

    def parent_pair():
        first = parent_index()
        if count == 1:
            return population[first], population[first]
        candidates = [index for index in range(count) if index != first]
        distinct = [index for index in candidates
                    if signatures[index] != signatures[first]]
        # Exact module transplantation is possible when constructive parents
        # share their co-adapted input geometry. Prefer that compatibility when
        # available; complementary behavior still chooses within the pool.
        if is_constructive(population[first]):
            first_layout = getattr(population[first], "input_layout", None)
            compatible = [
                index for index in (distinct or candidates)
                if getattr(population[index], "input_layout", None)
                == first_layout
            ]
        else:
            compatible = []
        parent_pool = compatible or distinct or candidates
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
