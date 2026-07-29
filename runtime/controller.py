from __future__ import annotations

import copy
import dataclasses
import os
import random
import statistics
import traceback
from concurrent.futures import ProcessPoolExecutor

from .cache import LRUCache
from .checkpoint import save_population
from .config import (RunConfig, MAX_CHROMOSOME_COUNT, nv_run_config,
                     validate_new_nv_profile)
from .escape import build_escape_state, population_mutation_rate
from .mutation import STRESS_PATIENCE
from .mutation import adaptive_mutation_rate as snn_adaptive_mutation_rate
from .parallel import EvolutionCancelled   # re-exported for back-compat


SOLVER_VALID = 0.999
LATEST_POPULATION_NAME = 'latest_population.json'
SOLVER_POPULATION_NAME = 'solver_generation.json'


def save_evaluated_generation(results_dir, population, fitnesses, target,
                              backend, run_config, *, status, source_try=None,
                              source_generation=None, certification=None):
    """Persist the latest *complete* generation, including failed genomes.

    The solver snapshot intentionally contains only genomes above
    ``SOLVER_VALID``.  It therefore becomes empty after an unsuccessful run and
    cannot be the Diversity tab's only data source.  This companion snapshot is
    the honest live population at the last safe generation boundary.
    """
    population = list(population)
    fitnesses = list(fitnesses)
    if len(population) != len(fitnesses):
        raise ValueError('population snapshot population/fitness count mismatch')
    path = os.path.join(results_dir, LATEST_POPULATION_NAME)
    save_population(
        path, population, target, backend, SOLVER_VALID, run_config,
        certification=certification, fitnesses=fitnesses,
        metadata={
            'status': status,
            'source': 'latest-fully-evaluated-generation',
            'try': source_try,
            'generation': source_generation,
        })
    return path, len(population)


def save_solver_generation(results_dir, population, fitnesses, target, backend,
                           run_config, *, status, source_try=None,
                           source_generation=None, certification=None):
    """Replace the fixed solver snapshot with valid members of one generation.

    ``population`` and ``fitnesses`` must describe a fully evaluated generation.
    Keeping this operation here makes Stop use the same atomic population writer
    as normal completion, without creating timestamped history files.
    """
    population = list(population)
    fitnesses = list(fitnesses)
    if len(population) != len(fitnesses):
        raise ValueError('solver snapshot population/fitness count mismatch')
    valid_pairs = [
        (genome, fitness)
        for genome, fitness in zip(population, fitnesses)
        if fitness >= SOLVER_VALID
    ]
    genomes = [genome for genome, _fitness in valid_pairs]
    solver_fitnesses = [fitness for _genome, fitness in valid_pairs]
    path = os.path.join(results_dir, SOLVER_POPULATION_NAME)
    save_population(
        path, genomes, target, backend, SOLVER_VALID, run_config,
        certification=certification, fitnesses=solver_fitnesses,
        metadata={
            'status': status,
            'source': 'latest-fully-evaluated-generation',
            'try': source_try,
            'generation': source_generation,
        })
    return path, len(genomes)


def wait_for_resume(pause_event, stop_event, messages):
    """Block at a safe boundary until Resume, remaining Stop-responsive."""
    announced = False
    while pause_event is not None and pause_event.is_set():
        if not announced:
            messages.put(('paused', True))
            announced = True
        if stop_event.wait(0.05):
            raise EvolutionCancelled
    if announced:
        messages.put(('paused', False))


def run_evolution(gens, pop, n_chroms, tries, target, arch, messages,
                  stop_event, base_seed=None, backend='snn', run_config=None,
                  results_dir='results', pause_event=None,
                  recombination_event=None):
    """Backend-neutral evolution worker used by the desktop application."""
    from substrates.snn.genome import random_genome
    from substrates.snn.ga import (_eval_batch as eval_snn,
                            next_population as next_snn,
                            rank_key as snn_rank_key)

    # A caller that supplies no configuration gets the ONE live nervous
    # profile rather than GAConfig's field defaults, which still describe the
    # retired single-tile engine and would be rejected by the gate below.
    config = run_config or (nv_run_config() if backend == 'nervous'
                            else RunConfig())
    if backend == 'snn' and config.ga.io_placement == 'terminal_nodes':
        raise ValueError(
            "io_placement='terminal_nodes' is available for nervous and LUT "
            'substrates; the SNN substrate has no directional terminal-cell '
            'physics')
    if backend == 'nervous':
        validate_new_nv_profile(config.ga)
    if backend == 'fnv' and config.ga.io_placement != 'fixed':
        raise ValueError(
            "Functional NV Net currently uses fixed physical terminals; "
            "set io_placement='fixed'")
    if (backend == 'fnv' and (
            config.ga.escape.lifespan_scoring
            or config.ga.escape.robustness
            or config.ga.escape.lexicase_downsample < 1.0)):
        raise ValueError(
            'FNV does not yet implement lifespan/physics-jitter objectives or '
            'lexicase downsampling; leave those escape options off')
    chromosome_count = (config.ga.chromosome_count
                        if config.ga.chromosome_count is not None else n_chroms)
    if not 1 <= chromosome_count <= MAX_CHROMOSOME_COUNT:
        raise ValueError('chromosome count must be between 1 and %d' %
                         MAX_CHROMOSOME_COUNT)
    if (config.ga.chromosome_count is not None
            and n_chroms != config.ga.chromosome_count):
        raise ValueError('n_chroms disagrees with run_config chromosome_count')
    if config.ga.chromosome_count is None:
        config = dataclasses.replace(
            config, ga=dataclasses.replace(
                config.ga, chromosome_count=chromosome_count))
    if (config.ga.io_placement in (
            'wiring_chromosome', 'spatial_chromosome')
            and chromosome_count < 3):
        raise ValueError(
            'chromosome-based I/O requires at least 3 chromosomes; '
            'chromosome 3 is reserved as the evolvable port map')
    setattr(target, 'pulse_config', config.pulse)
    # The escape configuration rides on the TARGET so it reaches evaluation
    # worker processes (same idiom as pulse_config / _lifetime_samples). It is
    # what turns lifespan scoring and the robustness objective on inside the
    # worker; the population-level mechanisms are driven from this loop.
    setattr(target, '_escape', config.ga.escape)
    n_ports = target.n_inputs + len(target.outputs)
    pool = None
    diversify_fn = None
    consolidate_fn = None
    plateau_rescue_fn = None
    rank_fn = snn_rank_key
    rate_fn = lambda rate, stagnation, solved=False, beta=0.0, limit=8.0: rate
    base_rate, decay = config.ga.mean_mutations, config.ga.mutation_decay

    if backend == 'nervous':
        from substrates.nervous import random_hex_genome
        from substrates.nervous.ga import (eval_batch_cases, next_population, diversify,
                               consolidate_population,
                               adaptive_mutation_rate, rank_key, N_WORKERS)
        # Evolvable I/O binding: the strategy lives on the target. Method A
        # evolves body-gene expression priorities. Chromosome strategies evolve
        # either type/selector mappings or normalised x/y anchors.
        io_strategy = getattr(config.ga, 'io_placement', 'fixed')
        setattr(target, 'io_placement', io_strategy)
        evolve_io = io_strategy in (
            'terminal_nodes', 'tag_rank', 'wiring_chromosome',
            'spatial_chromosome')
        port_chromosome = io_strategy in (
            'wiring_chromosome', 'spatial_chromosome')
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome():
            genome = random_hex_genome(
                chromosome_count, max_telomere=config.ga.max_telomere,
                arch=getattr(config.ga, 'tile_arch', 'single'),
                wiring_chromosome=(io_strategy == 'wiring_chromosome'),
                spatial_chromosome=(io_strategy == 'spatial_chromosome'),
                n_ports=n_ports, tag_rank=(io_strategy == 'tag_rank'),
                terminal_nodes=(io_strategy == 'terminal_nodes'),
                n_inputs=target.n_inputs, n_outputs=len(target.outputs),
                input_layout=True)
            if io_strategy == 'spatial_chromosome':
                from substrates.nervous.io_placement import seed_spatial
                seed_spatial(genome, None, target)
            elif port_chromosome:
                from substrates.nervous.io_placement import (
                    growth_seeds, seed_wiring_from_phenotype)
                from substrates.nervous.nervous import grow_nervous
                grid = grow_nervous(
                    genome, seeds=growth_seeds(
                        target, io_strategy, genome),
                    grid_size=target.grid_size, iters=target.iters)
                seed_wiring_from_phenotype(genome, grid, target)
            return genome
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        selection_mode = (
            'lexicase'
            if getattr(target, 'combinational_cases', ())
            else config.ga.selection)
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga,
            selection=selection_mode,
            chromosome_count=chromosome_count, recombination=recombine,
            evolve_io=evolve_io, io_placement=io_strategy,
            archive_parent=archive, stagnation=stagnation,
            rescue_candidates=rescue)
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        if io_strategy == 'spatial_chromosome':
            from substrates.nervous.io_placement import (
                growth_seeds, spatial_input_sites,
                spatial_output_variants, spatial_routing_variants)
            from substrates.nervous.nervous import grow_nervous
            rescue_queues = {}

            def nervous_spatial_rescue(champion, limit):
                grid = grow_nervous(
                    champion,
                    seeds=growth_seeds(
                        target, 'spatial_chromosome', champion),
                    grid_size=target.grid_size, iters=target.iters)
                body_key = (
                    tuple(sorted(grid.items())),
                    tuple(spatial_input_sites(champion, target)))
                queues = rescue_queues.get(body_key)
                if queues is None:
                    # At most ports*field-size proposals (small on the GUI
                    # domains). Serve the deterministic readout scan in chunks;
                    # once exhausted, this body spends no more generations
                    # repeating the same output placements or routing flips.
                    # Only the current champion body matters; discard scans for
                    # obsolete bodies instead of retaining a run-long history.
                    rescue_queues.clear()
                    queues = {
                        'outputs': spatial_output_variants(
                            champion, target, limit=10_000),
                        'routing': spatial_routing_variants(
                            champion, target, limit=10_000),
                    }
                    rescue_queues[body_key] = queues
                output_queue = queues['outputs']
                output_count = min(
                    len(output_queue), max(1, limit // 3))
                output_candidates = output_queue[:output_count]
                del output_queue[:output_count]
                routing_queue = queues['routing']
                routing_count = min(
                    len(routing_queue),
                    max(0, limit - len(output_candidates)))
                local_candidates = routing_queue[:routing_count]
                del routing_queue[:routing_count]
                return output_candidates + local_candidates

            plateau_rescue_fn = nervous_spatial_rescue
        # Resolve the delay-mutation toggle once so diversification mutates
        # under exactly the same operator set as the main GA loop.
        evolve_delay = config.ga.timing_mutations()
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count,
            evolve_delay=evolve_delay, evolve_io=evolve_io,
            io_placement=io_strategy,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    elif backend == 'fnv':
        from substrates.fnv.ga import (
            N_WORKERS, adaptive_mutation_rate, consolidate_population,
            diversify, eval_batch_cases, next_population, rank_key)
        from substrates.fnv.genome import random_functional_genome
        families = config.fnv.families
        setattr(target, '_fnv_families', families)
        setattr(target, 'io_placement', 'fixed')
        evolve_io = False
        cache = LRUCache(config.ga.cache_size)
        pool = ProcessPoolExecutor(max_workers=N_WORKERS)

        def make_genome():
            # FNV search is forward-only: initialization and immigrants sample
            # developmental rules directly, then the ordinary growth engine
            # determines the body. No target-shaped phenotype is constructed
            # and no body is translated back into a genome.
            return random_functional_genome(
                chromosome_count, max_telomere=config.ga.max_telomere,
                families=families, n_inputs=target.n_inputs)

        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(
                genomes, target, cache, pool, should_stop, on_progress)
        selection_mode = (
            'lexicase' if config.ga.selection == 'lexicase' else 'tournament')
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: \
            next_population(
                p, f, make_genome, c, mm, selection=selection_mode,
                ga_config=config.ga, chromosome_count=chromosome_count,
                recombination=recombine, archive_parent=archive,
                stagnation=stagnation, rescue_candidates=rescue,
                families=families, growth_seeds=target.inputs,
                focus_families=(
                    ("LOGIC", "C_ELEMENT")
                    if getattr(target, "combinational_cases", ())
                    else ()))
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count, families=families,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    elif backend == 'lut':
        from substrates.lut.ga import (
            eval_batch_cases, next_population, diversify,
            consolidate_population, make_seed_genome,
            adaptive_mutation_rate, rank_key, N_WORKERS,
            plateau_rescue_candidates)
        from substrates.nervous.io_placement import seed_io_metadata
        # Evolvable I/O binding (see the nervous branch): body priorities or a
        # dedicated chromosome of type/selector or spatial anchor alleles.
        io_strategy = getattr(config.ga, 'io_placement', 'fixed')
        setattr(target, 'io_placement', io_strategy)
        evolve_io = io_strategy in (
            'terminal_nodes', 'tag_rank', 'wiring_chromosome',
            'spatial_chromosome')
        port_chromosome = io_strategy in (
            'wiring_chromosome', 'spatial_chromosome')
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome():
            genome = make_seed_genome(chromosome_count)
            for chromosome in genome.chromosomes:
                chromosome.telomere = min(
                    chromosome.telomere, config.ga.max_telomere)
            if evolve_io:
                seed_io_metadata(
                    genome,
                    wiring_chromosome=(io_strategy == 'wiring_chromosome'),
                    spatial_chromosome=(
                        io_strategy == 'spatial_chromosome'),
                    n_ports=n_ports, tag_rank=(io_strategy == 'tag_rank'))
            if io_strategy == 'terminal_nodes':
                from substrates.nervous.io_placement import seed_terminal_kinds
                seed_terminal_kinds(
                    genome, target.n_inputs, len(target.outputs))
            if io_strategy == 'spatial_chromosome':
                from substrates.nervous.io_placement import seed_spatial
                seed_spatial(genome, None, target)
            elif port_chromosome:
                from substrates.lut.lut import grow_lut, cell_io_tags
                from substrates.nervous.io_placement import (
                    growth_seeds, seed_wiring_from_phenotype)
                grid = grow_lut(
                    genome, seeds=growth_seeds(
                        target, io_strategy, genome),
                    grid_size=target.grid_size, iters=target.iters)
                tags = cell_io_tags(genome, grid)
                seed_wiring_from_phenotype(
                    genome, grid, target, tags=tags)
            return genome
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga,
            chromosome_count=chromosome_count, recombination=recombine,
            evolve_io=evolve_io, io_placement=io_strategy,
            archive_parent=archive, stagnation=stagnation,
            rescue_candidates=rescue)
        plateau_rescue_fn = lambda champion, limit: \
            plateau_rescue_candidates(
                champion, target, limit=limit,
                max_telomere=config.ga.max_telomere)
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count,
            evolve_io=evolve_io, io_placement=io_strategy,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    else:
        from substrates.snn.ga import N_WORKERS
        # Evolvable I/O binding for the SNN combinational scorer.
        io_strategy = getattr(config.ga, 'io_placement', 'fixed')
        setattr(target, 'io_placement', io_strategy)
        evolve_io = io_strategy in (
            'tag_rank', 'wiring_chromosome', 'spatial_chromosome')
        port_chromosome = io_strategy in (
            'wiring_chromosome', 'spatial_chromosome')
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome():
            genome = random_genome(
                chromosome_count,
                wiring_chromosome=(io_strategy == 'wiring_chromosome'),
                spatial_chromosome=(io_strategy == 'spatial_chromosome'),
                n_ports=n_ports, tag_rank=(io_strategy == 'tag_rank'))
            if io_strategy == 'spatial_chromosome':
                from substrates.nervous.io_placement import seed_spatial
                seed_spatial(genome, None, target)
            elif port_chromosome:
                from substrates.snn.growth import grow_snn, cell_io_tags
                from substrates.nervous.io_placement import (
                    growth_seeds, seed_wiring_from_phenotype)
                grid = grow_snn(
                    genome, seeds=growth_seeds(
                        target, io_strategy, genome),
                    grid_size=target.grid_size, iters=target.iters)
                tags = cell_io_tags(genome, grid)
                seed_wiring_from_phenotype(
                    genome, grid, target, tags=tags)
            return genome
        raw_eval = lambda genomes, should_stop=None, on_progress=None: (
            eval_snn(genomes, target, arch, pool, cache, should_stop, on_progress),
            None)
        # Same plateau machinery the other two backends get: the annealed and
        # reheated mutation rate, fresh immigrants, a stressed archive parent,
        # and spatial output-rescue proposals. substrates.snn.ga.next_population and
        # substrates.snn.ga.evolve already support all of it; only this lambda was
        # still dropping mm/archive/stagnation/rescue on the floor, which left
        # the SNN backend running at a fixed mutation rate with no plateau
        # response at all.
        # The SNN breeder takes no ga_config, so the escape configuration and
        # the mutation cap are handed to it explicitly — otherwise self-adaptive
        # mutation would silently do nothing on this backend.
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: next_snn(
            p, f, chromosome_count=chromosome_count,
            recombination=recombine, evolve_io=evolve_io,
            io_placement=io_strategy,
            mean_mutations=mm, make_genome=make_genome,
            archive_parent=archive, stagnation=stagnation,
            rescue_candidates=rescue, escape=config.ga.escape,
            mutation_limit=config.ga.mutation_limit)
        rate_fn = snn_adaptive_mutation_rate
        if io_strategy == 'spatial_chromosome':
            from substrates.nervous.io_placement import spatial_output_variants
            plateau_rescue_fn = lambda champion, limit: \
                spatial_output_variants(champion, target, limit=limit)

    # One construction point for the escape machinery, shared with the headless
    # driver (substrates.nervous.ga.evolve_nervous) so the two drive paths
    # cannot breed, crowd or rebirth under different rules.
    escape_state = build_escape_state(
        backend, config.ga, chromosome_count=chromosome_count,
        io_placement=getattr(config.ga, 'io_placement', 'fixed'),
        evolve_io=evolve_io,
        evolve_delay=(config.ga.timing_mutations()
                      if backend == 'nervous' else None),
        fnv_families=(config.fnv.families if backend == 'fnv' else None))
    escape_active = config.ga.escape.any_enabled

    def recombination_enabled():
        if recombination_event is not None:
            return recombination_event.is_set()
        return config.ga.recombination_enabled

    def evaluate(genomes, phase, try_i, generation):
        # One saturated pool pass over the whole population (no chunk barrier):
        # map_ordered keeps every worker busy and polls the stop signal as each
        # genome finishes, so cancellation stays responsive without idling
        # workers on a slow genome. Progress is reported per completion.
        if stop_event.is_set():
            raise EvolutionCancelled
        messages.put(('phase', phase, try_i, generation, len(genomes)))

        def progress(done, total):
            # Coarse progress: ~20 updates/generation, not one per genome — the
            # display only shows a count, so keep queue traffic near the old
            # per-chunk cadence.
            if done == total or done % max(1, total // 20) == 0:
                messages.put(('phase', phase, try_i, generation, done))

        return raw_eval(genomes, should_stop=stop_event.is_set,
                        on_progress=progress)

    def validate_population(genomes):
        if any(len(genome.chromosomes) != chromosome_count
               for genome in genomes):
            raise ValueError(
                'population violates configured chromosome count %d' %
                chromosome_count)

    best_fit, best_genome, best_rank = 0.0, None, None
    population, fitnesses = [], []
    latest_population, latest_fitnesses = [], []
    latest_try, latest_generation = None, None
    certification = None

    def save_latest_solver_generation(status):
        """Non-fatal persistence: a save problem must not lose best_genome."""
        try:
            path, count = save_solver_generation(
                results_dir, latest_population, latest_fitnesses,
                target, backend, config, status=status,
                source_try=latest_try, source_generation=latest_generation,
                certification=certification)
            messages.put(('solver_saved', count, SOLVER_VALID, status, path))
        except Exception:
            messages.put(('solver_save_error', traceback.format_exc(limit=3)))

    def save_latest_evaluated_generation(status):
        """Non-fatal full-population persistence for post-run analysis."""
        try:
            path, count = save_evaluated_generation(
                results_dir, latest_population, latest_fitnesses,
                target, backend, config, status=status,
                source_try=latest_try, source_generation=latest_generation,
                certification=certification)
            messages.put(('population_saved', count, status, path))
        except Exception:
            messages.put(('population_save_error',
                          traceback.format_exc(limit=3)))

    try:
        for try_i in range(1, tries + 1):
            if stop_event.is_set():
                break
            wait_for_resume(pause_event, stop_event, messages)
            random.seed(None if base_seed is None else base_seed + try_i - 1)
            # Build one genome at a time with a stop check: constructing the
            # initial population of dense LUT ontogeny seeds takes seconds, and a
            # plain comprehension ignored Stop until the whole population was
            # built (the "stop still grows a generation for LUTs" lag).
            population = []
            for _ in range(pop):
                if stop_event.is_set():
                    raise EvolutionCancelled
                population.append(make_genome())
            validate_population(population)
            fitnesses, cases = evaluate(
                population, 'Evaluating initial population', try_i, 0)
            latest_population, latest_fitnesses = population, fitnesses
            latest_try, latest_generation = try_i, 0
            # Robustness scalars must exist before anything is ranked.
            escape_state.apply_robustness_blend(population, max(fitnesses))
            bi = max(range(pop),
                     key=lambda index: rank_fn(population[index], fitnesses[index]))
            champion, run_fit = copy.deepcopy(population[bi]), fitnesses[bi]
            run_rank = rank_fn(champion, run_fit)
            if best_rank is None or run_rank > best_rank:
                best_fit, best_genome, best_rank = (
                    run_fit, copy.deepcopy(champion), run_rank)
            messages.put(('gen', try_i, 0, best_fit,
                          sum(fitnesses) / len(fitnesses), run_fit, base_rate,
                          statistics.pstdev(fitnesses)))
            stagnation, mutation_rate = 0, base_rate
            for generation in range(1, gens + 1):
                if stop_event.is_set():
                    raise EvolutionCancelled
                wait_for_resume(pause_event, stop_event, messages)
                mutation_rate *= decay
                actual_rate = rate_fn(
                    mutation_rate, stagnation, run_fit >= 1.0,
                    config.ga.stagnation_beta, config.ga.mutation_limit)
                parents, parent_fitnesses, parent_cases = population, fitnesses, cases
                rescue = ()
                if (plateau_rescue_fn is not None and run_fit < 1.0
                        and stagnation >= STRESS_PATIENCE):
                    rescue = plateau_rescue_fn(
                        champion, min(48, max(1, pop // 2)))
                # One pool, or separate demes at their own mutation rates when
                # islands are on. Shared with the headless driver.
                offspring = escape_state.breed(
                    generation, parents, parent_fitnesses, parent_cases,
                    actual_rate,
                    lambda deme, deme_fitnesses, deme_cases, deme_rate: step(
                        deme, deme_fitnesses, deme_cases, deme_rate,
                        recombination_enabled(),
                        champion if run_fit < 1.0 else None,
                        stagnation, rescue))
                offspring_fitnesses, offspring_cases = evaluate(
                    offspring, 'Evaluating population', try_i, generation)
                offspring_best = max(offspring_fitnesses)
                # Collapse each genome's robust case vector under the current
                # anneal BEFORE anything is ranked; rank_fn reads the scalar.
                escape_state.apply_robustness_blend(
                    list(parents) + list(offspring), max(run_fit, offspring_best))
                # Survivor selection: terminal consolidation once solved,
                # otherwise crowding when it is on, otherwise strict
                # generational replacement. Shared with the headless driver.
                population, fitnesses, cases = escape_state.merge_generation(
                    parents, parent_fitnesses, parent_cases,
                    offspring, offspring_fitnesses, offspring_cases,
                    consolidate=consolidate_fn,
                    solved=(consolidate_fn is not None
                            and max(run_fit, offspring_best) >= 1.0))
                validate_population(population)
                gi = max(
                    range(pop),
                    key=lambda index: rank_fn(
                        population[index], fitnesses[index]))
                generation_rank = rank_fn(population[gi], fitnesses[gi])
                stagnation = 0 if fitnesses[gi] > run_fit + 1e-12 else stagnation + 1
                if escape_state.accepts(generation_rank, run_rank):
                    run_fit, champion, run_rank = (
                        fitnesses[gi], copy.deepcopy(population[gi]), generation_rank)
                    if escape_state.accepts(run_rank, best_rank):
                        best_fit, best_genome, best_rank = (
                            run_fit, copy.deepcopy(champion), run_rank)
                escape_state.record_champion(generation, champion, run_fit)
                population, fitnesses, cases, rebirth_info = \
                    escape_state.maybe_rebirth(
                        generation, population, fitnesses, cases, actual_rate,
                        stagnation, run_fit,
                        lambda genomes: evaluate(
                            genomes, 'Re-evaluating reborn cohort', try_i,
                            generation))
                if rebirth_info is not None:
                    # A rebirth IS progress against the stall it answered; not
                    # clearing the counter would re-fire it every generation.
                    stagnation = 0
                    validate_population(population)
                    # The reborn cohort was evaluated AFTER the champion was
                    # chosen above, so re-check it here. Skipping this let a
                    # reborn genome that beat the champion sit in the reported
                    # population for a generation while the reported best still
                    # showed the old value — the one way this loop could print a
                    # best below its own mean.
                    ri = max(range(pop),
                             key=lambda index: rank_fn(
                                 population[index], fitnesses[index]))
                    reborn_rank = rank_fn(population[ri], fitnesses[ri])
                    if escape_state.accepts(reborn_rank, run_rank):
                        run_fit, champion, run_rank = (
                            fitnesses[ri], copy.deepcopy(population[ri]),
                            reborn_rank)
                        if escape_state.accepts(run_rank, best_rank):
                            best_fit, best_genome, best_rank = (
                                run_fit, copy.deepcopy(champion), run_rank)
                    messages.put(('rebirth', rebirth_info))
                escape_state.tick()
                latest_population, latest_fitnesses = population, fitnesses
                latest_try, latest_generation = try_i, generation
                messages.put(('gen', try_i, generation, best_fit,
                              sum(fitnesses) / len(fitnesses), offspring_best,
                              actual_rate, statistics.pstdev(fitnesses)))
                if escape_active:
                    stats = escape_state.stats()
                    stats['mean_rate'] = population_mutation_rate(
                        population, actual_rate)
                    messages.put(('escape', stats))
            if best_fit >= 1.0:
                break

        # Credibility gate: a high training fitness is not a claim. For temporal
        # backends whose target has a reference oracle, re-score the winner on
        # FRESH held-out schedules (readout/alignment frozen) and emit a verdict
        # so a memorised-timing / leaky solution is flagged rather than trusted.
        # Advisory only — a failure here must never sink the run.
        if (backend in ('nervous', 'lut', 'fnv') and best_genome is not None
                and not stop_event.is_set()):
            try:
                from substrates.nervous.certification import certify
                certification = certify(best_genome, target, train=best_fit,
                                        backend=backend)
                messages.put(('certified', certification))
            except Exception:
                certification = None

        if backend in ('nervous', 'lut', 'fnv'):
            save_latest_evaluated_generation(
                'stopped' if stop_event.is_set() else 'complete')

        if stop_event.is_set() and backend in ('nervous', 'lut', 'fnv'):
            save_latest_solver_generation('stopped')
        elif (diversify_fn is not None and best_genome is not None
                and best_fit >= SOLVER_VALID):
            valid = SOLVER_VALID
            messages.put(('phase', 'Preparing solved-circuit diversity', 0, 25, 0))
            seeds = [g for g, f in zip(population, fitnesses) if f >= valid] or [best_genome]
            diverse = diversify_fn(seeds, valid)
            path = os.path.join(results_dir, SOLVER_POPULATION_NAME)
            save_population(
                path, diverse, target, backend, valid, config,
                certification=certification,
                metadata={
                    'status': ('stopped' if stop_event.is_set() else 'complete'),
                    'source': 'diversified-solvers',
                    'try': latest_try,
                    'generation': latest_generation,
                })
            if stop_event.is_set():
                messages.put(('solver_saved', len(diverse), valid, 'stopped', path))
            else:
                messages.put(('diverse', len(diverse), valid))
        elif backend in ('nervous', 'lut', 'fnv'):
            # Replace an older solved run with an honest empty snapshot.  An
            # unsolved current run must never leave stale solvers looking current.
            save_latest_solver_generation('complete')
    except EvolutionCancelled:
        if backend in ('nervous', 'lut', 'fnv'):
            save_latest_evaluated_generation('stopped')
            save_latest_solver_generation('stopped')
    except Exception:
        messages.put(('error', traceback.format_exc(limit=5)))
    finally:
        if pool is not None:
            # Do not announce completion while evaluation processes from the
            # stopped run are still alive.  With wait=False the UI enabled Run
            # immediately, allowing old and new pools to overlap; repeated
            # mid-generation stops could exhaust Windows process/IPC resources
            # and leave the app unresponsive. Pending work is cancelled first,
            # then only the already-running evaluations are allowed to drain.
            # This happens on the worker thread, so Tk remains responsive.
            pool.shutdown(wait=True, cancel_futures=True)
        messages.put(('done', copy.deepcopy(best_genome), best_fit))


def worker_entry(*args, **kwargs):
    messages = args[6]
    try:
        run_evolution(*args, **kwargs)
    except BaseException:
        messages.put(('error', traceback.format_exc(limit=5)))
        messages.put(('done', None, 0.0))
