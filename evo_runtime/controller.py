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
from .config import RunConfig, MAX_CHROMOSOME_COUNT, validate_new_nv_profile
from .parallel import EvolutionCancelled   # re-exported for back-compat


SOLVER_VALID = 0.999


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
    path = os.path.join(results_dir, 'solver_generation.json')
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
    from snn_evo.genome import random_genome
    from snn_evo.ga import (_eval_batch as eval_snn,
                            next_population as next_snn,
                            rank_key as snn_rank_key)

    config = run_config or RunConfig()
    if backend == 'nervous':
        validate_new_nv_profile(config.ga)
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
    setattr(target, 'pulse_config', config.pulse)
    pool = None
    diversify_fn = None
    consolidate_fn = None
    rank_fn = snn_rank_key
    rate_fn = lambda rate, stagnation, solved=False, beta=0.0, limit=8.0: rate
    base_rate, decay = config.ga.mean_mutations, config.ga.mutation_decay

    if backend == 'nervous':
        from nv_evo import random_hex_genome
        from nv_evo.ga import (eval_batch_cases, next_population, diversify,
                               consolidate_population,
                               adaptive_mutation_rate, rank_key, N_WORKERS)
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        make_genome = lambda: random_hex_genome(
            chromosome_count, max_telomere=config.ga.max_telomere,
            arch=getattr(config.ga, 'tile_arch', 'single'))
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        step = lambda p, f, c, mm, recombine: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga,
            chromosome_count=chromosome_count, recombination=recombine)
        rate_fn = adaptive_mutation_rate
        rank_fn = rank_key
        consolidate_fn = consolidate_population
        # Resolve the delay-mutation toggle once so diversification mutates
        # under exactly the same operator set as the main GA loop.
        evolve_delay = config.ga.timing_mutations()
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count,
            evolve_delay=evolve_delay,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    elif backend == 'lut':
        from lut_evo.ga import (eval_batch_cases, next_population, diversify,
                                consolidate_population, make_seed_genome,
                                adaptive_mutation_rate, _tiebreak, N_WORKERS)
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome():
            genome = make_seed_genome(chromosome_count)
            for chromosome in genome.chromosomes:
                chromosome.telomere = min(
                    chromosome.telomere, config.ga.max_telomere)
            return genome
        raw_eval = lambda genomes, should_stop=None, on_progress=None: \
            eval_batch_cases(genomes, target, cache, pool, should_stop, on_progress)
        step = lambda p, f, c, mm, recombine: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga,
            chromosome_count=chromosome_count, recombination=recombine)
        rate_fn = adaptive_mutation_rate
        rank_fn = lambda genome, fitness: (fitness, _tiebreak(genome))
        consolidate_fn = consolidate_population
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            chromosome_count=chromosome_count,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    else:
        from snn_evo.ga import N_WORKERS
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        make_genome = lambda: random_genome(chromosome_count)
        raw_eval = lambda genomes, should_stop=None, on_progress=None: (
            eval_snn(genomes, target, arch, pool, cache, should_stop, on_progress),
            None)
        step = lambda p, f, c, mm, recombine: next_snn(
            p, f, chromosome_count=chromosome_count,
            recombination=recombine)

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
                offspring = step(
                    parents, parent_fitnesses, parent_cases, actual_rate,
                    recombination_enabled())
                offspring_fitnesses, offspring_cases = evaluate(
                    offspring, 'Evaluating population', try_i, generation)
                # Report reproduction separately from survivor selection.  Once
                # a terminal LUT solution exists, environmental selection keeps
                # that solved parent alive; reporting the selected population's
                # best would therefore duplicate the all-time-best line even
                # when every newly generated offspring is worse.
                offspring_best = max(offspring_fitnesses)
                if (consolidate_fn is not None
                        and max(run_fit, offspring_best) >= 1.0):
                    population, fitnesses, cases = consolidate_fn(
                        parents, parent_fitnesses, parent_cases,
                        offspring, offspring_fitnesses, offspring_cases)
                else:
                    population, fitnesses, cases = (
                        offspring, offspring_fitnesses, offspring_cases)
                validate_population(population)
                latest_population, latest_fitnesses = population, fitnesses
                latest_try, latest_generation = try_i, generation
                gi = max(
                    range(pop),
                    key=lambda index: rank_fn(
                        population[index], fitnesses[index]))
                generation_rank = rank_fn(population[gi], fitnesses[gi])
                stagnation = 0 if fitnesses[gi] > run_fit + 1e-12 else stagnation + 1
                if generation_rank > run_rank:
                    run_fit, champion, run_rank = (
                        fitnesses[gi], copy.deepcopy(population[gi]), generation_rank)
                    if best_rank is None or run_rank > best_rank:
                        best_fit, best_genome, best_rank = (
                            run_fit, copy.deepcopy(champion), run_rank)
                messages.put(('gen', try_i, generation, best_fit,
                              sum(fitnesses) / len(fitnesses), offspring_best,
                              actual_rate, statistics.pstdev(fitnesses)))
            if best_fit >= 1.0:
                break

        # Credibility gate: a high training fitness is not a claim. For temporal
        # backends whose target has a reference oracle, re-score the winner on
        # FRESH held-out schedules (readout/alignment frozen) and emit a verdict
        # so a memorised-timing / leaky solution is flagged rather than trusted.
        # Advisory only — a failure here must never sink the run.
        if (backend in ('nervous', 'lut') and best_genome is not None
                and not stop_event.is_set()):
            try:
                from nv_evo.certification import certify
                certification = certify(best_genome, target, train=best_fit,
                                        backend=backend)
                messages.put(('certified', certification))
            except Exception:
                certification = None

        if stop_event.is_set() and backend in ('nervous', 'lut'):
            save_latest_solver_generation('stopped')
        elif (diversify_fn is not None and best_genome is not None
                and best_fit >= SOLVER_VALID):
            valid = SOLVER_VALID
            messages.put(('phase', 'Preparing solved-circuit diversity', 0, 25, 0))
            seeds = [g for g, f in zip(population, fitnesses) if f >= valid] or [best_genome]
            diverse = diversify_fn(seeds, valid)
            path = os.path.join(results_dir, 'solver_generation.json')
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
        elif backend in ('nervous', 'lut'):
            # Replace an older solved run with an honest empty snapshot.  An
            # unsolved current run must never leave stale solvers looking current.
            save_latest_solver_generation('complete')
    except EvolutionCancelled:
        if backend in ('nervous', 'lut'):
            save_latest_solver_generation('stopped')
    except Exception:
        messages.put(('error', traceback.format_exc(limit=5)))
    finally:
        if pool is not None:
            # wait=False so a Stop returns immediately: pending genomes are
            # cancelled and the few in-flight dense-LUT evals finish (and are
            # discarded) in the background instead of blocking the UI for the
            # ~2s it took to drain + join the worker processes. Results for a
            # normal completion were already collected before this point.
            pool.shutdown(wait=False, cancel_futures=True)
        messages.put(('done', copy.deepcopy(best_genome), best_fit))


def worker_entry(*args, **kwargs):
    messages = args[6]
    try:
        run_evolution(*args, **kwargs)
    except BaseException:
        messages.put(('error', traceback.format_exc(limit=5)))
        messages.put(('done', None, 0.0))
