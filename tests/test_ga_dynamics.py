"""Regression checks for GA convergence and configured genome structure."""
from __future__ import annotations

import dataclasses
import os
import random
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo_runtime.config import (GAConfig, MAX_CHROMOSOME_COUNT, RunConfig,
                                default_max_telomere)
from evo_runtime.checkpoint import load_checkpoint, save_checkpoint
import lut_evo.ga as lut_ga
import nv_evo.ga as nv_ga
import snn_evo.ga as snn_ga
from lut_evo.ga import (crossover_lut, diversify as diversify_lut, mutate_lut)
from lut_evo.genome import (MAX_CHROMS as LUT_MAX_CHROMS,
                            Chromosome as LutChromosome, Genome as LutGenome,
                            LutGene, random_lut_gene, random_lut_genome)
from lut_evo.ontogeny import _pack
from nv_evo.ga import (adaptive_mutation_rate, consolidate_population,
                       crossover_nv, diversify as diversify_nv, mutate_nv,
                       next_population)
from nv_evo.genome import (MAX_CHROMS as NV_MAX_CHROMS,
                           Chromosome as NvChromosome, Genome as NvGenome,
                           HexGene, random_hex_genome)
from nv_evo.targets import TEMPORAL_TARGETS
from snn_evo.ga import crossover as crossover_snn, mutate as mutate_snn
from snn_evo.genome import (MAX_CHROMS as SNN_MAX_CHROMS,
                            Chromosome as SnnChromosome, Gene as SnnGene,
                            Genome as SnnGenome, random_genome)


def test_lut_uses_a_smaller_fresh_run_telomere_default():
    assert default_max_telomere('lut') == 8
    assert default_max_telomere('nervous') == 20
    assert default_max_telomere('snn') == 20
    assert GAConfig().max_telomere == 20


_CROSSOVER_CASES = (
    (crossover_nv, NvGenome, NvChromosome, HexGene,
     ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out')),
    (crossover_lut, LutGenome, LutChromosome, LutGene,
     ('ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in', 'self_out')),
    (crossover_snn, SnnGenome, SnnChromosome, SnnGene,
     ('state_n', 'state_s', 'state_e', 'state_w', 'self_in', 'self_out')),
)


def _gene(gene_type, fields, value, **extra):
    return gene_type(**dict({field: value for field in fields}, **extra))


def _alleles(gene, fields):
    return tuple(getattr(gene, field) for field in fields)


def test_solved_mutation_gate_suppresses_only_the_sos_reheat():
    assert adaptive_mutation_rate(2.0, 100, solved=True) == 2.0
    assert adaptive_mutation_rate(2.0, 100, solved=False) == 8.0


def test_reciprocal_crossover_uses_both_original_parent_suffixes():
    """Child B used to read child A's overwritten list and remain parent B."""
    for crossover, genome_type, chromosome_type, gene_type, fields in _CROSSOVER_CASES:
        genes_a = [_gene(gene_type, fields, value) for value in (1, 2, 3)]
        genes_b = [_gene(gene_type, fields, value) for value in (11, 12, 13)]
        parent_a = genome_type([
            chromosome_type(genes=genes_a, split=1, tag=10, telomere=3)
        ], tag=1)
        parent_b = genome_type([
            chromosome_type(genes=genes_b, split=1, tag=10, telomere=3)
        ], tag=2)

        child_a, child_b = crossover(parent_a, parent_b)

        assert [_alleles(gene, fields) for gene in child_a.chromosomes[0].genes] == [
            _alleles(genes_a[0], fields),
            _alleles(genes_b[1], fields),
            _alleles(genes_b[2], fields),
        ]
        assert [_alleles(gene, fields) for gene in child_b.chromosomes[0].genes] == [
            _alleles(genes_b[0], fields),
            _alleles(genes_a[1], fields),
            _alleles(genes_a[2], fields),
        ]
        assert parent_a.chromosomes[0].genes == genes_a
        assert parent_b.chromosomes[0].genes == genes_b


def test_one_gene_chromosomes_recombine_rule_fields_in_every_backend():
    random.seed(7001)
    for crossover, genome_type, chromosome_type, gene_type, fields in _CROSSOVER_CASES:
        gene_a = _gene(gene_type, fields, 1)
        gene_b = _gene(gene_type, fields, 9)
        if hasattr(gene_a, 'limit'):
            gene_a.limit, gene_b.limit = 3, 8
        parent_a = genome_type([
            chromosome_type(genes=[gene_a], split=7, tag=10, telomere=3)
        ], tag=1)
        parent_b = genome_type([
            chromosome_type(genes=[gene_b], split=9, tag=10, telomere=3)
        ], tag=2)

        child_a, child_b = crossover(parent_a, parent_b)
        alleles_a = _alleles(child_a.chromosomes[0].genes[0], fields)
        alleles_b = _alleles(child_b.chromosomes[0].genes[0], fields)
        original_a, original_b = _alleles(gene_a, fields), _alleles(gene_b, fields)

        assert alleles_a not in (original_a, original_b)
        assert alleles_b not in (original_a, original_b)
        assert all(value in (1, 9) for value in alleles_a + alleles_b)
        assert 1 in alleles_a and 9 in alleles_a
        assert 1 in alleles_b and 9 in alleles_b
        assert child_a.chromosomes[0].split == 0
        assert child_b.chromosomes[0].split == 0
        if hasattr(gene_a, 'limit'):
            assert child_a.chromosomes[0].genes[0].limit == 3
            assert child_b.chromosomes[0].genes[0].limit == 8


def test_multi_event_mutation_transactions_cannot_cancel_back_to_parent():
    random.seed(7002)
    nv_parent = NvGenome([NvChromosome(
        genes=[_gene(HexGene, _CROSSOVER_CASES[0][4], 3)], split=0, tag=1)])
    lut_parent = LutGenome([LutChromosome(
        genes=[_gene(LutGene, _CROSSOVER_CASES[1][4], 300)], split=0, tag=1)])
    snn_parent = SnnGenome([SnnChromosome(
        genes=[_gene(SnnGene, _CROSSOVER_CASES[2][4], 3)], split=0, tag=1)])
    nv_signature = nv_ga.genome_signature(nv_parent)
    lut_signature = lut_ga.genome_signature(lut_parent)
    snn_signature = snn_ga.genome_signature(snn_parent)

    for _ in range(3000):
        assert nv_ga.genome_signature(
            mutate_nv(nv_parent, mean_mutations=4.0, chromosome_count=1)
        ) != nv_signature
        assert lut_ga.genome_signature(
            mutate_lut(lut_parent, mean_mutations=4.0, chromosome_count=1)
        ) != lut_signature
        assert snn_ga.genome_signature(
            mutate_snn(snn_parent, chromosome_count=1)
        ) != snn_signature


def test_lut_protected_novelty_keeps_the_soft_one_to_three_bit_step():
    random.seed(7004)
    for current, parent in ((0x1234, 0x1234), (0x1234, 0x1235),
                            (0x1234, 0xFFFF)):
        for _ in range(1000):
            child = lut_ga._soft_lut_excluding(current, parent)
            assert child != current
            assert child != parent
            assert 1 <= (child ^ current).bit_count() <= 3


def test_stale_terminal_splits_are_made_interior_before_crossover():
    for crossover, genome_type, chromosome_type, gene_type, fields in _CROSSOVER_CASES:
        genes_a = [_gene(gene_type, fields, value) for value in (1, 2, 3)]
        genes_b = [_gene(gene_type, fields, value) for value in (11, 12, 13)]
        parent_a = genome_type([
            chromosome_type(genes=genes_a, split=99, tag=10, telomere=3)
        ])
        parent_b = genome_type([
            chromosome_type(genes=genes_b, split=99, tag=10, telomere=3)
        ])

        child_a, child_b = crossover(parent_a, parent_b)
        assert child_a.chromosomes[0].split == 2
        assert child_b.chromosomes[0].split == 2
        assert _alleles(child_a.chromosomes[0].genes[2], fields) == (
            _alleles(genes_b[2], fields))
        assert _alleles(child_b.chromosomes[0].genes[2], fields) == (
            _alleles(genes_a[2], fields))

        # An unmatched chromosome in the legacy variable-count path must not
        # carry a stale terminal split through direct crossover either.
        parent_a.chromosomes.append(chromosome_type(
            genes=list(genes_a), split=99, tag=999, telomere=3))
        unequal_a, _ = crossover(parent_a, parent_b)
        assert unequal_a.chromosomes[1].split == 2


def test_empty_homologs_exchange_gene_lists_reciprocally():
    for crossover, genome_type, chromosome_type, gene_type, fields in _CROSSOVER_CASES:
        genes = [_gene(gene_type, fields, value) for value in (2, 3)]
        empty_parent = genome_type([
            chromosome_type(genes=[], split=9, tag=10, telomere=3)
        ])
        full_parent = genome_type([
            chromosome_type(genes=genes, split=1, tag=10, telomere=3)
        ])

        child_a, child_b = crossover(empty_parent, full_parent)
        assert child_a.chromosomes[0].genes == genes
        assert child_b.chromosomes[0].genes == []
        assert child_a.chromosomes[0].split == 1
        assert child_b.chromosomes[0].split == 0


def test_chromosome_count_round_trips_and_rejects_out_of_range_values():
    assert MAX_CHROMOSOME_COUNT == 32
    assert {LUT_MAX_CHROMS, NV_MAX_CHROMS, SNN_MAX_CHROMS} == {
        MAX_CHROMOSOME_COUNT}

    original = RunConfig(ga=GAConfig(chromosome_count=MAX_CHROMOSOME_COUNT))
    rebuilt = RunConfig.from_dict(dataclasses.asdict(original))
    assert rebuilt.ga.chromosome_count == MAX_CHROMOSOME_COUNT

    legacy = dataclasses.asdict(original)
    legacy['ga'].pop('chromosome_count')
    assert RunConfig.from_dict(legacy).ga.chromosome_count is None

    try:
        GAConfig(chromosome_count=MAX_CHROMOSOME_COUNT + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('chromosome_count above the backend limit was accepted')


def test_checkpoint_persists_count_and_rejects_genome_config_mismatch():
    config = RunConfig(ga=GAConfig(chromosome_count=2))
    genome = random_hex_genome(2)
    genome.chromosomes[0].genes = genome.chromosomes[0].genes[:1]
    genome.chromosomes[0].split = 99
    target = TEMPORAL_TARGETS['Pair detection gap (2x pulse width)']
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'checkpoint.json')
        save_checkpoint(
            path, genome, 0.5, target, None, 12, 'nervous', config)
        restored = load_checkpoint(path)
        assert restored['run_config'].ga.chromosome_count == 2
        assert len(restored['best_genome'].chromosomes) == 2
        assert restored['best_genome'].chromosomes[0].split == 0
        assert restored['target'].trials[0].input_events == target.trials[0].input_events

        try:
            save_checkpoint(
                path, random_hex_genome(1), 0.5, target, None, 12,
                'nervous', config)
        except ValueError:
            pass
        else:
            raise AssertionError('checkpoint accepted a count/config mismatch')


def test_fixed_chromosome_count_survives_each_backend_mutation_operator():
    random.seed(1201)
    nervous = random_hex_genome(3)
    lut = random_lut_genome(3)
    snn = random_genome(3)
    for _ in range(100):
        nervous = mutate_nv(
            nervous, mean_mutations=8.0, chromosome_count=3)
        lut = mutate_lut(lut, mean_mutations=8.0, chromosome_count=3)
        snn = mutate_snn(snn, chromosome_count=3)
        assert len(nervous.chromosomes) == 3
        assert len(lut.chromosomes) == 3
        assert len(snn.chromosomes) == 3


def test_lut_children_use_distinct_parents_and_a_real_mutation():
    random.seed(1202)
    genome = random_lut_genome(2)
    before = lut_ga.genome_signature(genome)
    mutated = mutate_lut(genome, mean_mutations=0.0, chromosome_count=2)
    assert lut_ga.genome_signature(mutated) != before

    elite = random_lut_genome(2)
    population = [elite, lut_ga.clone_genome(elite)] + [
        random_lut_genome(2) for _ in range(4)]
    fitnesses = [1.0, 0.9, 0.4, 0.3, 0.2, 0.1]
    pairs = []
    real_crossover = lut_ga.crossover_lut

    def record_pair(first, second):
        pairs.append((first, second))
        return real_crossover(first, second)

    config = GAConfig(
        chromosome_count=2, immigrant_fraction=0.0,
        elite_count=2, mean_mutations=0.0)
    with mock.patch.object(lut_ga, 'crossover_lut', side_effect=record_pair):
        lut_ga.next_population(
            population, fitnesses, mean_mutations=0.0, ga_config=config)
    assert pairs
    assert all(first is not second for first, second in pairs)
    assert all(lut_ga._recombination_signature(first)
               != lut_ga._recombination_signature(second)
               for first, second in pairs)


def test_nv_reproduction_preserves_configured_count_for_children_and_immigrants():
    random.seed(91)
    population = [random_hex_genome(3) for _ in range(10)]
    fitnesses = [i / 10.0 for i in range(10)]
    config = GAConfig(
        chromosome_count=3, mean_mutations=8.0,
        immigrant_fraction=0.5, elite_count=3)
    children = next_population(
        population, fitnesses,
        make_genome=lambda: random_hex_genome(3),
        mean_mutations=8.0, ga_config=config)
    assert len(children) == len(population)
    assert {len(genome.chromosomes) for genome in children} == {3}


def test_lut_cached_immigrant_factory_preserves_requested_chromosome_count():
    random.seed(404)
    previous = lut_ga._ONTO_POOL.get(3)
    try:
        lut_ga._ONTO_POOL[3] = [
            random_lut_genome(3) for _ in range(lut_ga._ONTO_POOL_SIZE)
        ]
        generated = [lut_ga.make_seed_genome(3) for _ in range(40)]
        assert {len(genome.chromosomes) for genome in generated} == {3}
    finally:
        if previous is None:
            lut_ga._ONTO_POOL.pop(3, None)
        else:
            lut_ga._ONTO_POOL[3] = previous


def test_lut_ontogeny_pack_produces_exact_requested_count():
    random.seed(405)
    for chromosome_count in (1, 2, 6, 16, MAX_CHROMOSOME_COUNT):
        for gene_count in (0, 1, 17, 151, 350):
            genes = [random_lut_gene() for _ in range(gene_count)]
            genome = _pack(genes, chromosome_count)
            assert len(genome.chromosomes) == chromosome_count
            flattened = [
                gene for chromosome in genome.chromosomes
                for gene in chromosome.genes
            ]
            assert flattened == genes


def test_configured_diversification_rejects_mismatched_seed_counts():
    for diversify, seed in (
            (diversify_nv, random_hex_genome(2)),
            (diversify_lut, random_lut_genome(2))):
        try:
            diversify([seed], None, 1, chromosome_count=3)
        except ValueError:
            pass
        else:
            raise AssertionError('mismatched diversification seed was accepted')


def test_solver_generation_breeds_and_counts_distinct_rule_programs():
    random.seed(7003)
    fields = _CROSSOVER_CASES[0][4]
    seed = NvGenome([NvChromosome(
        genes=[_gene(HexGene, fields, 4)], split=0, tag=1, telomere=3)])

    def perfect(genomes, *_args, **_kwargs):
        return [1.0] * len(genomes), [None] * len(genomes)

    real_crossover = nv_ga.crossover_nv
    with mock.patch.object(nv_ga, 'eval_batch_cases', side_effect=perfect), \
            mock.patch.object(nv_ga, 'crossover_nv', wraps=real_crossover) as crossed:
        generation = diversify_nv(
            [seed], None, 5, rounds=10, batch=1, chromosome_count=1)

    signatures = [nv_ga._recombination_signature(genome)
                  for genome in generation]
    assert len(generation) == 5
    assert len(set(signatures)) == 5
    assert crossed.call_count >= 1


def test_terminal_survivor_selection_retains_solver_and_case_alignment():
    random.seed(7)
    parents = [random_hex_genome(2) for _ in range(4)]
    offspring = [random_hex_genome(2) for _ in range(4)]
    for marker, genome in enumerate(parents + offspring):
        genome.tag = marker

    parent_fitnesses = [0.4, 0.3, 0.2, 0.1]
    offspring_fitnesses = [1.0, 0.9, 0.8, 0.7]
    parent_cases = [[genome.tag] for genome in parents]
    offspring_cases = [[genome.tag] for genome in offspring]
    expected_fitness = {
        genome.tag: fitness
        for genome, fitness in zip(
            parents + offspring, parent_fitnesses + offspring_fitnesses)
    }

    selected, fitnesses, cases = consolidate_population(
        parents, parent_fitnesses, parent_cases,
        offspring, offspring_fitnesses, offspring_cases)

    assert max(fitnesses) == 1.0
    assert sum(fitnesses) / len(fitnesses) >= (
        sum(parent_fitnesses) / len(parent_fitnesses))
    for genome, fitness, case_vector in zip(selected, fitnesses, cases):
        assert case_vector == [genome.tag]
        assert fitness == expected_fitness[genome.tag]


def test_terminal_survivor_selection_can_fill_the_live_population_with_solvers():
    random.seed(11)
    parents = [random_hex_genome(2) for _ in range(5)]
    offspring = [random_hex_genome(2) for _ in range(5)]
    parent_fitnesses = [1.0, 1.0, 0.4, 0.3, 0.2]
    offspring_fitnesses = [1.0, 1.0, 1.0, 0.9, 0.8]

    _, selected_fitnesses, selected_cases = consolidate_population(
        parents, parent_fitnesses, None,
        offspring, offspring_fitnesses, None)

    assert selected_fitnesses == [1.0] * 5
    assert selected_cases is None
