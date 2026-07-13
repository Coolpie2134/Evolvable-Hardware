from __future__ import annotations

import copy
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor

from .cache import LRUCache
from .checkpoint import save_population
from .config import RunConfig


class EvolutionCancelled(Exception):
    pass


def run_evolution(gens, pop, n_chroms, tries, target, arch, messages,
                  stop_event, base_seed=None, backend='snn', run_config=None,
                  results_dir='results'):
    """Backend-neutral evolution worker used by the desktop application."""
    from snn_evo.genome import random_genome
    from snn_evo.ga import _eval_batch as eval_snn, next_population as next_snn

    config = run_config or RunConfig()
    setattr(target, 'pulse_config', config.pulse)
    pool = None
    diversify_fn = None
    rate_fn = lambda rate, stagnation: rate
    base_rate, decay = config.ga.mean_mutations, config.ga.mutation_decay

    if backend == 'nervous':
        from nv_evo import random_hex_genome
        from nv_evo.ga import (eval_batch_cases, next_population, diversify,
                               adaptive_mutation_rate, N_WORKERS)
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        make_genome = lambda: random_hex_genome(
            n_chroms, max_telomere=config.ga.max_telomere)
        raw_eval = lambda genomes: eval_batch_cases(genomes, target, cache, pool)
        step = lambda p, f, c, mm: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga)
        rate_fn = adaptive_mutation_rate
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    elif backend == 'lut':
        from lut_evo.ga import (eval_batch_cases, next_population, diversify,
                                make_seed_genome, adaptive_mutation_rate, N_WORKERS)
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        def make_genome():
            genome = make_seed_genome(n_chroms)
            for chromosome in genome.chromosomes:
                chromosome.telomere = min(
                    chromosome.telomere, config.ga.max_telomere)
            return genome
        raw_eval = lambda genomes: eval_batch_cases(genomes, target, cache, pool)
        step = lambda p, f, c, mm: next_population(
            p, f, make_genome, c, mm, ga_config=config.ga)
        rate_fn = adaptive_mutation_rate
        diversify_fn = lambda seeds, valid: diversify(
            seeds, target, pop, valid=valid, cache=cache, executor=pool,
            should_stop=stop_event.is_set,
            max_telomere=config.ga.max_telomere,
            on_progress=lambda r, total, found: messages.put(
                ('phase', 'Diversifying solved circuits', r, total, found)))
    else:
        from snn_evo.ga import N_WORKERS
        cache = LRUCache(config.ga.cache_size)
        workers = N_WORKERS
        pool = ProcessPoolExecutor(max_workers=workers)
        make_genome = lambda: random_genome(n_chroms)
        raw_eval = lambda genomes: (eval_snn(genomes, target, arch, pool, cache), None)
        step = lambda p, f, c, mm: next_snn(p, f)

    chunk_size = max(workers, workers * config.ga.evaluation_chunk_multiplier)

    def evaluate(genomes, phase, try_i, generation):
        fits, cases = [], []
        case_vectors = True
        for start in range(0, len(genomes), chunk_size):
            if stop_event.is_set():
                raise EvolutionCancelled
            end = min(len(genomes), start + chunk_size)
            messages.put(('phase', phase, try_i, generation, end))
            batch_fits, batch_cases = raw_eval(genomes[start:end])
            fits.extend(batch_fits)
            if batch_cases is None:
                case_vectors = False
            elif case_vectors:
                cases.extend(batch_cases)
        return fits, (cases if case_vectors else None)

    size = lambda genome: sum(len(c.genes) for c in genome.chromosomes)
    best_fit, best_genome = 0.0, None
    population, fitnesses = [], []
    try:
        for try_i in range(1, tries + 1):
            if stop_event.is_set():
                break
            random.seed(None if base_seed is None else base_seed + try_i - 1)
            population = [make_genome() for _ in range(pop)]
            fitnesses, cases = evaluate(
                population, 'Evaluating initial population', try_i, 0)
            bi = max(range(pop), key=lambda i: (fitnesses[i], -size(population[i])))
            champion, run_fit = copy.deepcopy(population[bi]), fitnesses[bi]
            if run_fit > best_fit:
                best_fit, best_genome = run_fit, copy.deepcopy(champion)
            messages.put(('gen', try_i, 0, best_fit,
                          sum(fitnesses) / len(fitnesses), run_fit))
            stagnation, mutation_rate = 0, base_rate
            for generation in range(1, gens + 1):
                if stop_event.is_set():
                    raise EvolutionCancelled
                mutation_rate *= decay
                actual_rate = rate_fn(mutation_rate, stagnation)
                population = step(population, fitnesses, cases, actual_rate)
                fitnesses, cases = evaluate(
                    population, 'Evaluating population', try_i, generation)
                gi = max(range(pop), key=lambda i: (fitnesses[i], -size(population[i])))
                stagnation = 0 if fitnesses[gi] > run_fit + 1e-12 else stagnation + 1
                if (fitnesses[gi] > run_fit or
                        (fitnesses[gi] == run_fit and size(population[gi]) < size(champion))):
                    run_fit, champion = fitnesses[gi], copy.deepcopy(population[gi])
                    if (run_fit > best_fit or
                            (run_fit == best_fit and best_genome is not None
                             and size(champion) < size(best_genome))):
                        best_fit, best_genome = run_fit, copy.deepcopy(champion)
                messages.put(('gen', try_i, generation, best_fit,
                              sum(fitnesses) / len(fitnesses), fitnesses[gi]))
            if best_fit >= 1.0:
                break

        if (diversify_fn is not None and best_genome is not None
                and best_fit >= 0.999 and not stop_event.is_set()):
            valid = 0.999
            messages.put(('phase', 'Preparing solved-circuit diversity', 0, 25, 0))
            seeds = [g for g, f in zip(population, fitnesses) if f >= valid] or [best_genome]
            diverse = diversify_fn(seeds, valid)
            if diverse:
                path = os.path.join(results_dir, 'solver_generation.json')
                save_population(path, diverse, target, backend, valid, config)
                messages.put(('diverse', len(diverse), valid))
    except EvolutionCancelled:
        pass
    except Exception:
        messages.put(('error', traceback.format_exc(limit=5)))
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        messages.put(('done', copy.deepcopy(best_genome), best_fit))


def worker_entry(*args, **kwargs):
    messages = args[6]
    try:
        run_evolution(*args, **kwargs)
    except BaseException:
        messages.put(('error', traceback.format_exc(limit=5)))
        messages.put(('done', None, 0.0))
