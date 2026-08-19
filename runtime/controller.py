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
from .mutation import adaptive_mutation_rate as snn_adaptive_mutation_rate
from .parallel import EvolutionCancelled   # re-exported for back-compat


SOLVER_VALID = 0.999
#: Rescue candidates per stalled generation. Deliberately still pop//2.
#:
#: A 90-second-budget ablation said 8 was better: over 24 paired runs across 8
#: targets it beat pop//2 (12-3, sign p=0.035) and beat no rescue (13-3,
#: p=0.021), while pop//2 looked indistinguishable from no rescue at all
#: (7-4-13, p=0.55) despite building 125,970 genomes. Solve counts were equal
#: at 6/24 under every setting, so 8 looked like a free ~80% saving.
#:
#: IT DID NOT REPLICATE AT FULL BUDGET. Full adder needs 561-869s to solve and
#: never solved inside the 90s ablation under ANY variant, so that experiment
#: could not see solve behaviour on the expensive targets at all. Re-tested on
#: one seed at the real 200-generation budget: pop//2 certified 2/4, limit 8
#: certified 0/4 - all four parked on the 0.9062 best-wrong ceiling - for a
#: ~26% wall saving. Trading solves for speed is the wrong trade here.
#:
#: The knob (GAConfig.plateau_rescue_limit) stays so the experiment can be
#: finished properly at full budget across the stall-prone targets. Until then
#: the measured-at-90s value is NOT the default.
#: See results/fnv_rescue_ablation.md.
FNV_PLATEAU_RESCUE_LIMIT = None
LATEST_POPULATION_NAME = 'latest_population.json'
SOLVER_POPULATION_NAME = 'solver_generation.json'


def _certification_permits_diversity(target, certification):
    """Whether a training solver is credible enough for the solver bank.

    Autonomous targets retain the historical behavior. Oracle-backed temporal
    targets and exhaustive combinational contracts require a certified winner
    before another 25 rounds of solver-bank search; overfit, below-threshold,
    or failed certification does not qualify.
    """
    from substrates.nervous.certification import oracle_spec_for
    static_logic = (
        not getattr(target, 'temporal', False)
        and bool(getattr(target, 'cases', ())))
    if (oracle_spec_for(target) is None
            and not getattr(target, 'combinational_cases', ())
            and not getattr(target, 'temporal_logic_cases', ())
            and not static_logic):
        return True
    verdict = str((certification or {}).get('verdict') or '')
    return verdict.startswith('CERTIFIED')


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
                  recombination_event=None, budget=None):
    """Backend-neutral evolution worker used by the desktop application.

    ``budget`` is an optional ``callable(best_fit) -> reason or None`` consulted
    once per generation. Returning a reason ends the run early, in the same
    orderly way as running out of generations: the champion is kept and the
    certification gate still runs. It is deliberately NOT ``stop_event``, which
    means "the user aborted" and suppresses certification.
    """
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
            "io_placement='terminal_nodes' is retained for programmatic LUT "
            'runs; SNN has no directional terminal-cell physics, and Nervous '
            'uses evolved source pads plus fitted output probes')
    if backend == 'nervous':
        validate_new_nv_profile(config.ga)
        if config.ga.io_placement != 'fixed':
            raise ValueError(
                'retired Nervous I/O placement: %r. The nervous substrate now '
                'uses one native mechanism - an evolved input layout of source '
                'pads plus whole-organism fitted output probes. The tag_rank, '
                'wiring_chromosome, spatial_chromosome and terminal_nodes '
                'compatibility strategies remain only where supported by SNN '
                'or programmatic LUT runs.' % (config.ga.io_placement,))
    if backend == 'fnv' and config.ga.io_placement != 'fixed':
        raise ValueError(
            "Functional NV Net uses native evolved source pads and its own "
            "fitted/genetic output selector; set the compatibility field "
            "io_placement='fixed'")
    if (backend == 'lut'
            and getattr(config.ga, 'lut_io_mode',
                        'source_pads') == 'exterior_edges'
            and config.ga.io_placement != 'fixed'):
        raise ValueError(
            "LUT exterior-edge I/O replaces cell-binding strategies; "
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
    if backend == 'fnv' and 2 * chromosome_count < len(target.outputs):
        raise ValueError(
            'Functional NV Net needs at least one chromosome arm per output; '
            'increase Chroms to at least %d' %
            ((len(target.outputs) + 1) // 2))
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
    if backend == 'lut':
        setattr(
            target, 'lut_io_mode',
            getattr(config.ga, 'lut_io_mode', 'source_pads'))
        setattr(
            target, '_lut_function_families',
            getattr(config.ga, 'lut_function_families',
                    ('UNRESTRICTED',)))
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
    # More workers than genomes only adds process-start and context-switching
    # overhead. The configured cap is shared by GUI and benchmark runs.
    workers = max(1, min(config.ga.evaluation_workers, pop))

    if backend == 'nervous':
        from substrates.nervous.branched_ga import (
            plateau_rescue_candidates, random_branched_hex_genome,
            select_developmental_seed)
        from substrates.nervous.ga import (eval_batch_cases, next_population, diversify,
                               consolidate_population,
                               adaptive_mutation_rate, rank_key)
        # Nervous I/O is ONE native mechanism: an evolved input layout of
        # discrete source pads, plus whole-organism fitted output probes. The
        # tag / wiring / spatial / terminal-node placement strategies are
        # retired here (they remain available to LUT and SNN).
        io_strategy = 'fixed'
        setattr(target, 'io_placement', io_strategy)
        evolve_io = False
        cache = LRUCache(config.ga.cache_size)
        from substrates.nervous.ga import init_eval_worker
        setattr(target, '_worker_target_installed', True)
        pool = ProcessPoolExecutor(max_workers=workers,
                                   initializer=init_eval_worker,
                                   initargs=(target,))
        # Target-specific developmental selection is an injection mechanism,
        # not part of an unbiased evolutionary run. Keep initialization purely
        # random; reusable generic mutation operators remain available.
        pure_evolution = True
        def make_genome(input_genes=None):
            # Branched, output-rooted development is the nervous encoding now
            # (substrates/nervous/branched.py). Arms start at genetic output
            # roots and grow backward to the evolved input pads, so the genome
            # names parts rather than describing neighbourhoods - the ceiling
            # the min-Hamming lookup could not get past.
            # Richest of a few random starts, judged only on whether the
            # inputs can reach the outputs - most random starts have an output
            # nothing can drive, and those are all the same score to selection.
            factory = lambda: random_branched_hex_genome(
                chromosome_count, max_telomere=config.ga.max_telomere,
                n_inputs=target.n_inputs,
                output_roles=tuple(terminal.role
                                   for terminal in target.outputs),
                input_genes=input_genes)
            return factory() if pure_evolution else select_developmental_seed(
                factory, attempts=make_genome.developmental_seed_candidates)
        make_genome.developmental_seed_candidates = 6
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        selection_mode = (
            'lexicase'
            if (not getattr(target, 'temporal', False)
                or getattr(target, 'combinational_cases', ())
                or getattr(target, 'temporal_logic_cases', ()))
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
        plateau_rescue_fn = lambda champion, limit: \
            plateau_rescue_candidates(
                champion, target, limit=limit,
                max_telomere=config.ga.max_telomere)
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
            adaptive_mutation_rate, consolidate_population, diversify,
            eval_batch_cases, initialization_families, next_population,
            plateau_rescue_candidates, rank_key, select_developmental_seed)
        from substrates.fnv.genome import random_functional_genome
        families = config.fnv.families
        seed_families = initialization_families(families, target)
        setattr(target, '_fnv_families', families)
        setattr(target, 'io_placement', 'fixed')
        setattr(target, '_fnv_readout_mode', config.fnv.readout_mode)
        evolve_io = False
        cache = LRUCache(config.ga.cache_size)
        pool = ProcessPoolExecutor(max_workers=workers)
        fnv_logic_contract = (
            bool(getattr(target, "combinational_cases", ()))
            or (not getattr(target, "temporal", False)
                and bool(getattr(target, "cases", ()))))

        def make_genome():
            # Role names seed genetic output niches, not desired behavior or
            # coordinates. Local rules grow backward from those roots toward
            # the evolved source pads.
            return select_developmental_seed(lambda: random_functional_genome(
                chromosome_count, max_telomere=config.ga.max_telomere,
                families=seed_families, n_inputs=target.n_inputs,
                output_roles=tuple(terminal.role
                                   for terminal in target.outputs)),
                prefer_logic_capacity=fnv_logic_contract)

        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(
                genomes, target, cache, pool, should_stop, on_progress)
        selection_mode = (
            'lexicase'
            if (not getattr(target, 'temporal', False)
                or getattr(target, 'combinational_cases', ())
                or getattr(target, 'temporal_logic_cases', ()))
            else config.ga.selection)
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: \
            next_population(
                p, f, make_genome, c, mm, selection=selection_mode,
                ga_config=config.ga, chromosome_count=chromosome_count,
                recombination=recombine, archive_parent=archive,
                stagnation=stagnation, rescue_candidates=rescue,
                families=families, growth_seeds=target.inputs,
                target=target,
                focus_families=(
                    tuple(family for family in ("LOGIC", "DELAY")
                          if family in families)
                    if fnv_logic_contract
                    else ()))
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        plateau_rescue_fn = lambda champion, limit: \
            plateau_rescue_candidates(
                champion, limit=limit,
                max_telomere=config.ga.max_telomere,
                families=families, growth_seeds=target.inputs,
                focus_families=(
                    tuple(family for family in ("LOGIC", "DELAY")
                          if family in families)
                    if fnv_logic_contract else ()),
                target=target)
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
            consolidate_population, constrain_genome_functions,
            make_seed_genome,
            adaptive_mutation_rate, rank_key,
            plateau_rescue_candidates)
        from substrates.lut.branched_ga import (
            random_branched_lut_genome,
            select_developmental_seed as select_lut_seed)
        from substrates.lut.genome import random_input_layout
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
        function_families = config.ga.lut_function_families
        selection_ga = (
            dataclasses.replace(config.ga, selection='lexicase')
            if (not getattr(target, 'temporal', False)
                or getattr(target, 'combinational_cases', ())
                or getattr(target, 'temporal_logic_cases', ()))
            else config.ga)
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome(input_genes=None):
            # Branched, output-rooted development is the LUT encoding now
            # (substrates/lut/branched.py): arms start at genetic output roots
            # and grow backward to the evolved input pads, installing named
            # gates from the enabled banks rather than nearest-match tables.
            if io_strategy == 'fixed' and getattr(
                    config.ga, 'lut_io_mode',
                    'source_pads') != 'exterior_edges':
                return select_lut_seed(
                    lambda: random_branched_lut_genome(
                        chromosome_count, max_telomere=config.ga.max_telomere,
                        n_inputs=target.n_inputs,
                        output_roles=tuple(terminal.role
                                           for terminal in target.outputs),
                        families=function_families,
                        input_genes=input_genes),
                    attempts=make_genome.developmental_seed_candidates)
            genome = constrain_genome_functions(
                make_seed_genome(chromosome_count), function_families)
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
        make_genome.developmental_seed_candidates = 6
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        step = lambda p, f, c, mm, recombine, archive, stagnation, rescue: next_population(
            p, f, make_genome, c, mm, ga_config=selection_ga,
            chromosome_count=chromosome_count, recombination=recombine,
            evolve_io=evolve_io, io_placement=io_strategy,
            archive_parent=archive, stagnation=stagnation,
            rescue_candidates=rescue)
        plateau_rescue_fn = lambda champion, limit: \
            plateau_rescue_candidates(
                champion, target, limit=limit,
                max_telomere=config.ga.max_telomere,
                function_families=function_families)
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count,
            evolve_io=evolve_io, io_placement=io_strategy,
            function_families=function_families,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    else:
        # Evolvable I/O binding for the SNN combinational scorer.
        io_strategy = getattr(config.ga, 'io_placement', 'fixed')
        setattr(target, 'io_placement', io_strategy)
        evolve_io = io_strategy in (
            'tag_rank', 'wiring_chromosome', 'spatial_chromosome')
        port_chromosome = io_strategy in (
            'wiring_chromosome', 'spatial_chromosome')
        cache = LRUCache(config.ga.cache_size)
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
        # the mutation cap are handed to it explicitly - otherwise self-adaptive
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
    # Target-specific witness/rescue candidates are diagnostic tools only and
    # must never enter production evolution or benchmark populations.
    plateau_rescue_fn = None

    def new_escape_state():
        # Restarts are independent searches. Reusing this mutable object leaked
        # rebirth archives, cooldowns, pending island migrations, and now walker
        # lineage state from one restart into the next, making a two-restart run
        # neither two trials nor one continuous population.
        return build_escape_state(
            backend, config.ga, chromosome_count=chromosome_count,
            io_placement=getattr(config.ga, 'io_placement', 'fixed'),
            evolve_io=evolve_io,
            evolve_delay=(config.ga.timing_mutations()
                          if backend == 'nervous' else None),
            fnv_families=(config.fnv.families if backend == 'fnv' else None),
            lut_function_families=(
                config.ga.lut_function_families
                if backend == 'lut' else None))
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
            # Coarse progress: ~20 updates/generation, not one per genome - the
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
    budget_reason = None
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
            escape_state = new_escape_state()
            # Build one genome at a time with a stop check: constructing the
            # initial population of dense LUT ontogeny seeds takes seconds, and a
            # plain comprehension ignored Stop until the whole population was
            # built (the "stop still grows a generation for LUTs" lag).
            population = []
            if hasattr(make_genome, 'developmental_seed_candidates'):
                make_genome.developmental_seed_candidates = 6
            # Arms are only safe to recombine when they grew against the same
            # concrete source pads. Independent random starts almost never
            # meet that condition, so guarded crossover had become a clone
            # followed by mutation. A few cohorts retain layout exploration,
            # but give every arm several compatible potential mates.
            cohort_inputs = []
            cohort_count = (
                min(4, max(1, pop // 8))
                if backend in ('nervous', 'lut') else 0)
            for index in range(pop):
                if stop_event.is_set():
                    raise EvolutionCancelled
                if not cohort_count:
                    genome = make_genome()
                elif index < cohort_count:
                    genome = make_genome()
                    cohort_inputs.append(list(getattr(genome, 'inputs', ())))
                else:
                    genome = make_genome(
                        cohort_inputs[(index - cohort_count) % cohort_count])
                population.append(genome)
            # Initial cohorts pay for the strongest target-blind connectivity
            # filter. Immigrants exist to inject diversity every generation;
            # rebuilding up to six complete organisms for each one dominated
            # LUT wall time and selected that diversity back toward the same
            # morphology. Two candidates still reject most inert starts while
            # cutting the recurring construction bill by roughly two thirds.
            if hasattr(make_genome, 'developmental_seed_candidates'):
                make_genome.developmental_seed_candidates = 2
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
            # Rebirth needs an actual branch point before the stall trigger.
            # Waiting for the first archive interval threw away the initial
            # population and made short runs fire with a one-entry archive.
            escape_state.record_champion(0, champion, run_fit)
            escape_state.note_contract_progress(cases, fitnesses)
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
                if run_fit >= 1.0 and consolidate_fn is not None:
                    # Terminal consolidation exists so perfect circuits can
                    # ACCUMULATE and the mean can converge to 1. It could not:
                    # every offspring is mutated, mutation almost always breaks
                    # a solution, so the population sat at exactly ONE perfect
                    # member forever while the mean pinned to whatever the rest
                    # scored. Elites are deliberately never copied, which stops
                    # premature convergence BEFORE a solve; afterwards there is
                    # nothing left to converge away from.
                    #
                    # Re-enter the solved genomes as unmutated children through
                    # the same channel plateau rescue uses (those are cloned,
                    # not mutated). Capped at a quarter of the population so the
                    # remaining offspring keep exploring.
                    rescue = [
                        genome
                        for genome, fitness in zip(parents, parent_fitnesses)
                        if fitness >= SOLVER_VALID][:max(1, pop // 4)]
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
                # Survivor selection: terminal consolidation once solved;
                # otherwise optional crowding and the baseline rotating
                # contract-elite reserve. Shared with the headless driver.
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
                scalar_progress = fitnesses[gi] > run_fit + 1e-12
                case_progress = escape_state.note_contract_progress(
                    cases, fitnesses)
                stagnation = (
                    0 if scalar_progress or case_progress else stagnation + 1)
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
                    escape_state.note_contract_progress(cases, fitnesses)
                    validate_population(population)
                    # The reborn cohort was evaluated AFTER the champion was
                    # chosen above, so re-check it here. Skipping this let a
                    # reborn genome that beat the champion sit in the reported
                    # population for a generation while the reported best still
                    # showed the old value - the one way this loop could print a
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
                # A spent budget ends the run the same way exhausting the
                # generations does - NOT the way Stop does. The distinction
                # matters: stop_event suppresses the certification gate below,
                # because a user abort leaves a half-finished run that must not
                # be credited. A run that ends because it hit its time cap or
                # reached fitness 1.0 is a complete result and still has to be
                # certified, or every early solve would report as uncertified.
                if budget is not None:
                    spent = budget(best_fit)
                    if spent:
                        budget_reason = spent
                        messages.put(('budget', spent, try_i, generation))
                        break
            if best_fit >= 1.0 or budget_reason:
                break

        # Credibility gate: a high training fitness is not a claim. For temporal
        # backends whose target has a reference oracle, re-score the winner on
        # FRESH held-out schedules (readout/alignment frozen) and emit a verdict
        # so a memorised-timing / leaky solution is flagged rather than trusted.
        # Advisory only - a failure here must never sink the run.
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
        elif (config.ga.diversify_solvers
                and diversify_fn is not None and best_genome is not None
                and best_fit >= SOLVER_VALID
                and _certification_permits_diversity(
                    target, certification)):
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
