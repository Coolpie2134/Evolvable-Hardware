"""Regression checks for GA convergence and configured genome structure."""
from __future__ import annotations

import dataclasses
import os
import queue
import random
import sys
import tempfile
import threading
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.config import (DEFAULT_EVALUATION_WORKERS, GAConfig,
                                MAX_CHROMOSOME_COUNT,
                                MAX_EVALUATION_WORKERS, RunConfig,
                                default_max_telomere)
from runtime.checkpoint import load_checkpoint, save_checkpoint
from runtime.controller import (run_evolution, save_evaluated_generation,
                                    save_solver_generation, wait_for_resume)
import substrates.lut.ga as lut_ga
import substrates.nervous.ga as nv_ga
import substrates.snn.ga as snn_ga
from substrates.lut.ga import (crossover_lut, diversify as diversify_lut, mutate_lut)
from substrates.lut.genome import (MAX_CHROMS as LUT_MAX_CHROMS,
                            Chromosome as LutChromosome, Genome as LutGenome,
                            LutGene, random_lut_gene, random_lut_genome)
from substrates.lut.ontogeny import _pack
from substrates.nervous.ga import (adaptive_mutation_rate, consolidate_population,
                       crossover_nv, diversify as diversify_nv, mutate_nv,
                       next_population)
from substrates.nervous.genome import (MAX_CHROMS as NV_MAX_CHROMS,
                           Chromosome as NvChromosome, Genome as NvGenome,
                           HexGene, RoutingPatch, random_hex_genome)
from substrates.nervous.pulse import PulseConfig
from substrates.nervous.targets import TEMPORAL_TARGETS
from substrates.snn.ga import crossover as crossover_snn, mutate as mutate_snn
from substrates.snn.genome import (MAX_CHROMS as SNN_MAX_CHROMS,
                            Chromosome as SnnChromosome, Gene as SnnGene,
                            Genome as SnnGenome, random_genome)


def test_lut_uses_a_smaller_fresh_run_telomere_default():
    assert default_max_telomere('lut') == 8
    assert default_max_telomere('nervous') == 20
    assert default_max_telomere('snn') == 20
    assert GAConfig().max_telomere == 20


def test_nervous_routing_overlay_survives_checkpoint_and_signature():
    genome = random_hex_genome(2)
    genome.routing_patches = [
        RoutingPatch(-2, 3, 17),
        RoutingPatch(4, 1, 9),
    ]
    target = TEMPORAL_TARGETS['Veto gate']
    # Single-arch genomes built inline: this exercises checkpoint/signature
    # plumbing, not a run, so it keeps the retired engine's architecture.
    config = RunConfig(ga=GAConfig(
        chromosome_count=2, tile_arch='single',
        node_model='pulse_delay'),
        pulse=PulseConfig(model='pulse_delay'))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'patched.json')
        save_checkpoint(
            path, genome, 0.5, target, None, 17, 'nervous', config)
        restored = load_checkpoint(path)['best_genome']
    assert [
        (patch.x, patch.y, patch.state)
        for patch in restored.routing_patches
    ] == [(-2, 3, 17), (4, 1, 9)]
    assert nv_ga.genome_signature(restored) == nv_ga.genome_signature(genome)


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
    assert adaptive_mutation_rate(2.0, 100, beta=0.0) == 2.0
    assert adaptive_mutation_rate(1.0, 18, beta=2.0) > \
        adaptive_mutation_rate(1.0, 18, beta=1.0)
    assert lut_ga.adaptive_mutation_rate(1.0, 18, beta=2.0) == \
        adaptive_mutation_rate(1.0, 18, beta=2.0)
    assert adaptive_mutation_rate(20.0, 0, mutation_limit=6.0) == 6.0
    assert adaptive_mutation_rate(
        2.0, 100, mutation_limit=3.0) == 3.0


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

    original = RunConfig(ga=GAConfig(
        chromosome_count=MAX_CHROMOSOME_COUNT, stagnation_beta=2.5,
        mutation_limit=12.0, recombination_enabled=False,
        evaluation_workers=3, diversify_solvers=False,
        io_placement='spatial_chromosome'))
    rebuilt = RunConfig.from_dict(dataclasses.asdict(original))
    assert rebuilt.ga.chromosome_count == MAX_CHROMOSOME_COUNT
    assert rebuilt.ga.stagnation_beta == 2.5
    assert rebuilt.ga.mutation_limit == 12.0
    assert rebuilt.ga.recombination_enabled is False
    assert rebuilt.ga.evaluation_workers == 3
    assert rebuilt.ga.diversify_solvers is False
    assert rebuilt.ga.io_placement == 'spatial_chromosome'

    legacy = dataclasses.asdict(original)
    legacy['ga'].pop('chromosome_count')
    legacy['ga'].pop('mutation_limit')
    legacy['ga'].pop('recombination_enabled')
    legacy['ga'].pop('evaluation_workers')
    legacy['ga'].pop('diversify_solvers')
    legacy_config = RunConfig.from_dict(legacy).ga
    assert legacy_config.chromosome_count is None
    assert legacy_config.mutation_limit == 8.0
    assert legacy_config.recombination_enabled is True
    assert legacy_config.evaluation_workers == DEFAULT_EVALUATION_WORKERS
    assert legacy_config.diversify_solvers is True

    try:
        GAConfig(chromosome_count=MAX_CHROMOSOME_COUNT + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('chromosome_count above the backend limit was accepted')

    for invalid_workers in (0, MAX_EVALUATION_WORKERS + 1):
        try:
            GAConfig(evaluation_workers=invalid_workers)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid evaluation worker count was accepted')


def test_diversity_requires_certification_only_when_an_oracle_exists():
    from runtime.controller import _certification_permits_diversity
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import get_target

    oracle_target = TEMPORAL_TARGETS['SR latch']
    assert _certification_permits_diversity(
        oracle_target, {'verdict': 'CERTIFIED'})
    assert not _certification_permits_diversity(
        oracle_target, {'verdict': 'OVERFIT (memorised timing)'})
    assert not _certification_permits_diversity(
        oracle_target, {'verdict': 'BELOW THRESHOLD 0.90'})
    assert not _certification_permits_diversity(oracle_target, None)
    assert _certification_permits_diversity(
        TEMPORAL_TARGETS['Oscillator'],
        {'verdict': 'UNCERTIFIED (no oracle reference for this target)'})
    logic_target = periodic_combinational_target(get_target('Full adder'))
    assert _certification_permits_diversity(
        logic_target, {'verdict': 'CERTIFIED'})
    assert not _certification_permits_diversity(
        logic_target, {'verdict': 'OVERFIT (memorised row order)'})


def test_terminal_node_io_config_round_trips_without_wiring_chromosome():
    original = RunConfig(ga=GAConfig(
        chromosome_count=2, io_placement='terminal_nodes'))
    rebuilt = RunConfig.from_dict(dataclasses.asdict(original))
    assert rebuilt.ga.io_placement == 'terminal_nodes'
    assert rebuilt.ga.chromosome_count == 2


def test_checkpoint_persists_count_and_rejects_genome_config_mismatch():
    config = RunConfig(ga=GAConfig(
        chromosome_count=2, stagnation_beta=1.75,
        mutation_limit=6.0))
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
        assert restored['run_config'].ga.stagnation_beta == 1.75
        assert restored['run_config'].ga.mutation_limit == 6.0
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

        try:
            save_checkpoint(
                path, random_hex_genome(2, arch='tri3'), 0.5, target,
                None, 12, 'nervous', config)
        except ValueError:
            pass
        else:
            raise AssertionError('checkpoint accepted a tile-architecture mismatch')


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


def test_nervous_mutation_has_no_io_tag_path_left():
    """The nervous evolve_io mutation branch is deleted, not merely unused.

    LUT and SNN keep their own copies - this retirement is scoped to nervous.
    """
    import substrates.lut.ga as lut_ga_module
    import substrates.snn.ga as snn_ga_module
    assert not hasattr(nv_ga, '_mutate_io_tag')
    assert hasattr(lut_ga_module, '_mutate_io_tag')
    assert hasattr(snn_ga_module, '_mutate_io_tag')

    # A nervous genome mutated with the old flag still on keeps its port-map
    # alleles: there is no path left to change them. (Body genes still mutate
    # structurally, so only the wiring chromosome is compared.)
    from substrates.nervous.io_placement import wiring_chromosome
    genome = random_hex_genome(3, wiring_chromosome=True, n_ports=3)
    before = [(gene.tag, gene.io_limit, gene.io_selector)
              for gene in wiring_chromosome(genome).genes]
    random.seed(7009)
    for _ in range(50):
        mutated = mutate_nv(genome, mean_mutations=8.0, chromosome_count=3,
                            evolve_delay=True, evolve_io=True)
        assert [(gene.tag, gene.io_limit, gene.io_selector)
                for gene in wiring_chromosome(mutated).genes] == before


def test_coordinated_spatial_io_mutation_relocates_distinct_ports():
    from substrates.nervous.io_placement import mutate_io_bundle, wiring_chromosome

    genome = random_lut_genome(
        3, wiring_chromosome=True, n_ports=4,
        spatial_chromosome=True)
    wiring = wiring_chromosome(genome)
    before = [(gene.tag, gene.io_selector) for gene in wiring.genes]
    random.seed(7010)
    assert mutate_io_bundle(
        genome, 1 << 16, strategy='spatial_chromosome', count=2) == 2
    after = [(gene.tag, gene.io_selector) for gene in wiring.genes]

    changed = [index for index, pair in enumerate(after)
               if pair != before[index]]
    assert len(changed) == 2
    assert all(after[index][0] != before[index][0]
               and after[index][1] != before[index][1]
               for index in changed)


def test_lut_plateau_rescue_proposes_motifs_and_output_rule_bits():
    from substrates.nervous.io_placement import (
        bind_io, flat_inputs, flat_outputs, set_spatial_port_positions)
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import get_target

    target = periodic_combinational_target(get_target('Half adder'))
    target.io_placement = 'spatial_chromosome'
    genome = random_lut_genome(
        3, wiring_chromosome=True, n_ports=4,
        spatial_chromosome=True)
    # Make one maintenance rule visibly expressed at every fake output cell.
    genome.chromosomes[0].genes[0] = dataclasses.replace(
        genome.chromosomes[0].genes[0], self_in=1, self_out=1)
    grid = {
        (0, 0): (1, 0, 0, 0), (1, 0): (1, 0, 0, 0),
        (0, 1): (1, 0, 0, 0), (1, 1): (1, 0, 0, 0),
    }
    initial = [(0, 0), (1, 1), (1, 0), (0, 1)]
    assert set_spatial_port_positions(
        genome, grid, initial) == len(initial)

    with mock.patch.object(lut_ga, 'grow_lut', return_value=grid):
        candidates = lut_ga.plateau_rescue_candidates(
            genome, target, limit=16)

    assert candidates
    signatures = [lut_ga._recombination_signature(g) for g in candidates]
    assert len(signatures) == len(set(signatures))
    # A compact-motif proposal coordinates all four logical ports.
    bindings = []
    from substrates.lut.lut import cell_io_tags
    for candidate in candidates:
        bound = bind_io(
            candidate, grid, target, 'spatial_chromosome',
            tags=cell_io_tags(candidate, grid))
        if bound is not None:
            bindings.append(flat_inputs(bound[0]) + flat_outputs(bound[1]))
    assert any(binding != initial for binding in bindings)
    # A local-rule proposal flips exactly one output bit on the maintenance gene.
    assert any(
        candidate.chromosomes[0].genes[0].self_out == 0
        for candidate in candidates)


def test_lut_synthesizes_every_hard_combinational_target_as_a_grown_genome():
    from substrates.lut.synthesis import (
        POLARISED_SEED, synthesize_combinational_genome)
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import get_target

    names = (
        'Full adder', '2-bit adder', '2:1 MUX', 'Majority-3',
        'Parity-3 (XOR3)', '2-to-4 decoder', '2-bit comparator',
        '2x2 multiplier',
    )
    compiled = {}
    for name in names:
        target = periodic_combinational_target(get_target(name))
        target.io_placement = 'spatial_chromosome'
        result = synthesize_combinational_genome(
            target, chromosome_count=3, max_telomere=8)
        fitness, cases = lut_ga.evaluate_lut_full(result.genome, target)
        assert result.inverse_report['exact']
        assert result.inverse_report['radius'] <= 8
        assert result.genome.seed_state == POLARISED_SEED
        assert result.genome.provenance == 'truth-table-compiler-v1'
        assert len(result.genome.chromosomes) == 3
        assert result.genome.chromosomes[2].wiring
        assert max(
            len(chromosome.genes)
            for chromosome in result.genome.chromosomes
            if not chromosome.wiring) <= lut_ga.ONTOGENY_CAP
        assert fitness == 1.0
        assert min(cases) == 1.0
        compiled[name] = (target, result)

    # The new developmental seed is real genotype state: checkpoint round-trip
    # and cache/signature identity must retain it.
    target, result = compiled['2-bit comparator']
    config = RunConfig(ga=GAConfig(
        chromosome_count=3, io_placement='spatial_chromosome',
        max_telomere=8))
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'compiled.json')
        save_checkpoint(
            path, result.genome, 1.0, target, None, 73, 'lut', config)
        restored = load_checkpoint(path)
    assert restored['best_genome'].seed_state == POLARISED_SEED
    assert restored['best_genome'].provenance == 'truth-table-compiler-v1'
    assert restored['target'].combinational_strobe
    assert restored['target'].combinational_data_inputs == 4
    assert restored['target'].combinational_cases == target.combinational_cases
    assert [
        trial.case_windows for trial in restored['target'].trials
    ] == [trial.case_windows for trial in target.trials]
    assert lut_ga.genome_signature(
        restored['best_genome']) == lut_ga.genome_signature(result.genome)


def test_hard_target_compiler_is_the_first_plateau_rescue_candidate():
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import get_target

    target = periodic_combinational_target(get_target('2x2 multiplier'))
    target.io_placement = 'spatial_chromosome'
    champion = random_lut_genome(
        3, wiring_chromosome=True,
        spatial_chromosome=True,
        n_ports=target.n_inputs + len(target.outputs))
    candidates = lut_ga.plateau_rescue_candidates(
        champion, target, limit=4, max_telomere=8)
    assert candidates
    fitness, cases = lut_ga.evaluate_lut_full(candidates[0], target)
    assert fitness == 1.0
    assert min(cases) == 1.0


def test_plateau_archive_keeps_breeding_the_all_time_lut_champion():
    random.seed(7011)
    population = [random_lut_genome(3) for _ in range(10)]
    champion = lut_ga.clone_genome(population[0])
    fitnesses = [0.8] + [0.1] * 9
    config = GAConfig(
        chromosome_count=3, immigrant_fraction=0.0, elite_count=2,
        io_placement='fixed')

    with mock.patch.object(
            lut_ga, 'mutate_lut', wraps=lut_ga.mutate_lut) as mutate:
        children = lut_ga.next_population(
            population, fitnesses, mean_mutations=2.0, ga_config=config,
            archive_parent=champion,
            stagnation=nv_ga.STRESS_PATIENCE)

    archive_calls = [
        call for call in mutate.call_args_list
        if call.args and call.args[0] is champion]
    assert archive_calls
    assert len(children) == len(population)
    assert all(child is not champion for child in children)


def test_spatial_plateau_preserves_one_champion_and_mixes_local_edits():
    random.seed(70115)
    population = [
        random_hex_genome(
            3, spatial_chromosome=True, n_ports=4)
        for _ in range(20)]
    for index, genome in enumerate(population):
        genome.routing_patches = [RoutingPatch(index, 0, 3)]
    champion = nv_ga.clone_genome(population[0])
    fitnesses = [0.8] + [0.1] * (len(population) - 1)
    config = GAConfig(
        chromosome_count=3, immigrant_fraction=0.0, elite_count=4,
        io_placement='spatial_chromosome', tile_arch='single',
        node_model='pulse_delay')

    with mock.patch.object(
            nv_ga, 'mutate_nv', wraps=nv_ga.mutate_nv) as mutate:
        children = nv_ga.next_population(
            population, fitnesses, mean_mutations=8.0, ga_config=config,
            archive_parent=champion,
            stagnation=nv_ga.STRESS_PATIENCE)

    champion_signature = nv_ga.genome_signature(champion)
    assert sum(
        nv_ga.genome_signature(child) == champion_signature
        for child in children) == 1
    local_flags = [
        bool(call.kwargs.get('local_only'))
        for call in mutate.call_args_list]
    assert any(local_flags)
    assert any(not flag for flag in local_flags)


def test_controller_invokes_lut_rescue_only_after_plateau_patience():
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(ga=GAConfig(chromosome_count=2))
    messages = queue.Queue()
    stop = threading.Event()

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    def evaluate(genomes, *_args, **_kwargs):
        return [0.5] * len(genomes), None

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()) as pool_factory, \
            mock.patch.object(
                lut_ga, 'make_seed_genome',
                side_effect=lambda count: random_lut_genome(count)), \
            mock.patch.object(lut_ga, 'eval_batch_cases',
                              side_effect=evaluate), \
            mock.patch.object(
                lut_ga, 'plateau_rescue_candidates',
                return_value=[]) as rescue, \
            mock.patch('substrates.nervous.certification.certify', return_value=None):
        run_evolution(
            gens=nv_ga.STRESS_PATIENCE + 1, pop=2, n_chroms=2,
            tries=1, target=target, arch=None, messages=messages,
            stop_event=stop, base_seed=7012, backend='lut',
            run_config=config, results_dir=directory)

    assert rescue.call_count == 1
    assert rescue.call_args.kwargs['limit'] == 1
    # The configured default exceeds this tiny population; the controller
    # should not start idle processes.
    assert pool_factory.call_args.kwargs['max_workers'] == 2


def test_controller_uses_row_lexicase_for_combinational_nervous_targets():
    from substrates.nervous.scoring import contract_case_count
    from substrates.snn.targets import get_target

    # The ordinary GUI Full/Half-adder path is a static truth table. It used to
    # discard its row vector and silently fall back to scalar tournament even
    # though periodic wrappers already used row-wise lexicase.
    target = get_target('Half adder')
    config = RunConfig(
        ga=GAConfig(
            chromosome_count=2, selection='tournament',
            tile_arch='tri3', node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    messages = queue.Queue()
    stop = threading.Event()

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    def evaluate(genomes, *_args, **_kwargs):
        cases = [[0.5] * contract_case_count(target) for _ in genomes]
        return [0.5] * len(genomes), cases

    def reproduce(population, *_args, **_kwargs):
        return [nv_ga.clone_genome(genome) for genome in population]

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases',
                              side_effect=evaluate), \
            mock.patch.object(nv_ga, 'next_population',
                              side_effect=reproduce) as step, \
            mock.patch('substrates.nervous.certification.certify', return_value=None):
        run_evolution(
            gens=1, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=70125,
            backend='nervous', run_config=config, results_dir=directory)

    assert step.call_count == 1
    assert step.call_args.kwargs['selection'] == 'lexicase'


def test_nervous_rejects_every_retired_io_placement():
    """The retired strategies are gone from nervous, not merely unused.

    They remain available to LUT and SNN, so the rejection has to be scoped to
    the nervous backend rather than to the config field.
    """
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    messages, stop = queue.Queue(), threading.Event()

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    for retired in ('terminal_nodes', 'tag_rank', 'wiring_chromosome',
                    'spatial_chromosome'):
        config = RunConfig(
            ga=GAConfig(chromosome_count=3, io_placement=retired,
                        tile_arch='tri3', node_model='paper_analog'),
            pulse=PulseConfig(model='paper_analog'))
        with tempfile.TemporaryDirectory() as directory,                 mock.patch('runtime.controller.ProcessPoolExecutor',
                           return_value=FakePool()):
            try:
                run_evolution(
                    gens=1, pop=2, n_chroms=3, tries=1, target=target,
                    arch=None, messages=messages, stop_event=stop,
                    base_seed=1, backend='nervous', run_config=config,
                    results_dir=directory)
            except ValueError as error:
                assert 'retired Nervous I/O placement' in str(error), retired
            else:
                raise AssertionError('nervous accepted %r' % retired)

def test_controller_gives_the_snn_backend_the_same_plateau_machinery():
    """The SNN step must forward the plateau arguments, not drop them.

    ``substrates.snn.ga.next_population`` accepts the annealed/reheated mutation rate,
    an io-aware immigrant factory, a stressed archive parent and rescue
    proposals, and ``substrates.snn.ga.evolve`` passes all four. The controller's step
    lambda used to accept them and forward none, so the desktop/app run was a
    strictly weaker search than the standalone one: a fixed mutation rate, no
    plateau reheating, no archive parent, and immigrants built by the fallback
    ``random_genome`` that has no spatial port chromosome.
    """
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(ga=GAConfig(chromosome_count=3,
                                   io_placement='spatial_chromosome'))
    messages = queue.Queue()
    stop = threading.Event()

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    def evaluate(genomes, *_args, **_kwargs):
        return [0.5] * len(genomes)

    real_next = snn_ga.next_population
    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(snn_ga, '_eval_batch', side_effect=evaluate), \
            mock.patch.object(snn_ga, 'next_population',
                              side_effect=real_next) as step, \
            mock.patch('substrates.nervous.certification.certify', return_value=None):
        run_evolution(
            gens=snn_ga.STRESS_PATIENCE + 2, pop=4, n_chroms=3,
            tries=1, target=target, arch=None, messages=messages,
            stop_event=stop, base_seed=7013, backend='snn',
            run_config=config, results_dir=directory)

    assert step.call_count >= snn_ga.STRESS_PATIENCE + 1
    for call in step.call_args_list:
        assert call.kwargs['make_genome'] is not None
        assert call.kwargs['mean_mutations'] is not None
        assert 'archive_parent' in call.kwargs
        assert 'rescue_candidates' in call.kwargs
    # Fitness never improves above, so stagnation must climb past the patience
    # threshold and hand the archived champion to the stressed branch.
    assert max(call.kwargs['stagnation']
               for call in step.call_args_list) >= snn_ga.STRESS_PATIENCE
    assert any(call.kwargs['archive_parent'] is not None
               for call in step.call_args_list)


def test_recombination_can_be_disabled_without_disabling_reproduction():
    random.seed(1776)

    nv_population = [random_hex_genome(2) for _ in range(4)]
    lut_population = [random_lut_genome(2) for _ in range(4)]
    snn_population = [random_genome(2) for _ in range(4)]
    fitnesses = [1.0, 0.7, 0.4, 0.1]
    config = GAConfig(
        chromosome_count=2, immigrant_fraction=0.0, elite_count=2)

    with mock.patch.object(
            nv_ga, 'crossover_nv', side_effect=AssertionError('crossover called')):
        nv_children = nv_ga.next_population(
            nv_population, fitnesses, mean_mutations=1.0,
            ga_config=config, recombination=False)
    with mock.patch.object(
            lut_ga, 'crossover_lut', side_effect=AssertionError('crossover called')):
        lut_children = lut_ga.next_population(
            lut_population, fitnesses, mean_mutations=1.0,
            ga_config=config, recombination=False)
    with mock.patch.object(
            snn_ga, 'crossover', side_effect=AssertionError('crossover called')):
        snn_children = snn_ga.next_population(
            snn_population, fitnesses, chromosome_count=2,
            recombination=False)

    assert len(nv_children) == len(nv_population)
    assert len(lut_children) == len(lut_population)
    assert len(snn_children) == len(snn_population)
    assert all(child is not parent for child in nv_children
               for parent in nv_population)
    assert all(child is not parent for child in lut_children
               for parent in lut_population)
    assert all(child is not parent for child in snn_children
               for parent in snn_population)


def test_controller_archives_the_champion_without_carrying_parents_forward():
    """A worse offspring generation remains the live population.

    The all-time best is reported separately, but must not be reinserted as an
    elite survivor or used to inflate the charted population mean.
    """
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(
        ga=GAConfig(chromosome_count=2, tile_arch='tri3',
                    node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    messages = queue.Queue()
    stop = threading.Event()
    evaluations = iter((
        ([0.8, 0.8, 0.8], None),
        ([0.1, 0.2, 0.3], None),
    ))

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch('substrates.nervous.certification.certify', return_value=None), \
            mock.patch.object(
                nv_ga, 'eval_batch_cases',
                side_effect=lambda *_args, **_kwargs: next(evaluations)):
        run_evolution(
            gens=1, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=1701,
            backend='nervous', run_config=config, results_dir=directory)
        snapshot = load_checkpoint(
            os.path.join(directory, 'latest_population.json'))

    assert snapshot['fitnesses'] == [0.1, 0.2, 0.3]
    generation = next(
        message for message in messages.queue
        if message[0] == 'gen' and message[2] == 1)
    assert generation[3] == 0.8
    assert abs(generation[4] - 0.2) < 1e-12
    assert generation[5] == 0.3


def test_terminal_convergence_starts_only_after_a_perfect_offspring():
    """After the first 1.0, solved parents/children accumulate toward mean 1."""
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(
        ga=GAConfig(chromosome_count=2, tile_arch='tri3',
                    node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    messages = queue.Queue()
    stop = threading.Event()
    evaluations = iter((
        ([0.8, 0.7, 0.6], None),
        ([1.0, 0.4, 0.3], None),
        ([1.0, 1.0, 0.0], None),
    ))

    class FakePool:
        def shutdown(self, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch('substrates.nervous.certification.certify', return_value=None), \
            mock.patch('runtime.controller.SOLVER_VALID', 1.1), \
            mock.patch.object(
                nv_ga, 'eval_batch_cases',
                side_effect=lambda *_args, **_kwargs: next(evaluations)):
        run_evolution(
            gens=2, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=1702,
            backend='nervous', run_config=config, results_dir=directory)
        snapshot = load_checkpoint(
            os.path.join(directory, 'latest_population.json'))

    generations = [
        message for message in messages.queue if message[0] == 'gen']
    assert generations[1][3:6] == (1.0, 2.5 / 3.0, 1.0)
    assert generations[2][3:6] == (1.0, 1.0, 1.0)
    assert snapshot['fitnesses'] == [1.0, 1.0, 1.0]


def test_terminal_consolidation_preserves_case_alignment():
    random.seed(7)
    parents = [random_hex_genome(2) for _ in range(3)]
    offspring = [random_hex_genome(2) for _ in range(3)]
    for marker, genome in enumerate(parents + offspring):
        genome.tag = marker
    parent_fitnesses = [1.0, 0.7, 0.6]
    offspring_fitnesses = [1.0, 0.9, 0.8]
    parent_cases = [[genome.tag] for genome in parents]
    offspring_cases = [[genome.tag] for genome in offspring]
    expected = dict(zip(
        [genome.tag for genome in parents + offspring],
        parent_fitnesses + offspring_fitnesses))

    selected, fitnesses, cases = consolidate_population(
        parents, parent_fitnesses, parent_cases,
        offspring, offspring_fitnesses, offspring_cases)

    assert fitnesses == [1.0, 1.0, 0.9]
    for genome, fitness, case_vector in zip(selected, fitnesses, cases):
        assert fitness == expected[genome.tag]
        assert case_vector == [genome.tag]


def test_pause_waits_at_boundary_and_resumes_without_losing_state():
    pause = threading.Event()
    stop = threading.Event()
    messages = queue.Queue()
    completed = threading.Event()
    pause.set()

    def run_wait():
        wait_for_resume(pause, stop, messages)
        completed.set()

    worker = threading.Thread(target=run_wait)
    worker.start()
    assert messages.get(timeout=1.0) == ('paused', True)
    assert not completed.is_set()
    pause.clear()
    worker.join(timeout=1.0)
    assert completed.is_set()
    assert messages.get(timeout=1.0) == ('paused', False)


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


def test_solver_generation_snapshot_overwrites_and_records_latest_generation():
    random.seed(7011)
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig()
    first_population = [random_hex_genome(2) for _ in range(3)]

    with tempfile.TemporaryDirectory() as directory:
        path, count = save_solver_generation(
            directory, first_population, [1.0, 0.999, 0.4], target,
            'nervous', config, status='stopped', source_try=2,
            source_generation=17)
        first = load_checkpoint(path)

        assert count == 2
        assert len(first['genomes']) == 2
        assert first['fitnesses'] == [1.0, 0.999]
        assert first['metadata'] == {
            'status': 'stopped',
            'source': 'latest-fully-evaluated-generation',
            'try': 2,
            'generation': 17,
        }

        # A later unsolved generation replaces the prior solver set instead of
        # leaving stale genomes or accumulating timestamped snapshots.
        path_again, count = save_solver_generation(
            directory, [random_hex_genome(2)], [0.5], target,
            'nervous', config, status='complete', source_try=3,
            source_generation=4)
        second = load_checkpoint(path_again)

        assert path_again == path
        assert count == 0
        assert second['genomes'] == []
        assert second['fitnesses'] == []
        assert second['metadata']['try'] == 3
        assert second['metadata']['generation'] == 4
        assert os.listdir(directory) == ['solver_generation.json']


def test_evaluated_generation_snapshot_retains_failed_population():
    """Diversity must have real genomes to analyse when none is a solver."""
    random.seed(7012)
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig()
    population = [random_hex_genome(2) for _ in range(3)]
    fitnesses = [0.72, 0.31, 0.08]

    with tempfile.TemporaryDirectory() as directory:
        path, count = save_evaluated_generation(
            directory, population, fitnesses, target, 'nervous', config,
            status='complete', source_try=1, source_generation=9)
        snapshot = load_checkpoint(path)

    assert os.path.basename(path) == 'latest_population.json'
    assert count == len(population)
    assert len(snapshot['genomes']) == len(population)
    assert snapshot['fitnesses'] == fitnesses
    assert snapshot['metadata'] == {
        'status': 'complete',
        'source': 'latest-fully-evaluated-generation',
        'try': 1,
        'generation': 9,
    }


def test_stop_saves_the_latest_fully_evaluated_solver_generation():
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(
        ga=GAConfig(chromosome_count=2, tile_arch='tri3',
                    node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    stop = threading.Event()
    messages = queue.Queue()

    shutdown_args = []

    class FakePool:
        def shutdown(self, **kwargs):
            shutdown_args.append(kwargs)

    def complete_initial_evaluation(genomes, *_args, **_kwargs):
        # Stop arrives after this whole generation has scores, so it is the
        # generation the cancellation path must persist.
        stop.set()
        return [1.0, 0.5, 0.25], [None] * len(genomes)

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases',
                              side_effect=complete_initial_evaluation):
        run_evolution(
            gens=4, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=17,
            backend='nervous', run_config=config, results_dir=directory)
        snapshot = load_checkpoint(
            os.path.join(directory, 'solver_generation.json'))
        full_snapshot = load_checkpoint(
            os.path.join(directory, 'latest_population.json'))

    assert len(snapshot['genomes']) == 1
    assert snapshot['fitnesses'] == [1.0]
    assert snapshot['metadata']['status'] == 'stopped'
    assert snapshot['metadata']['try'] == 1
    assert snapshot['metadata']['generation'] == 0
    assert len(full_snapshot['genomes']) == 3
    assert full_snapshot['fitnesses'] == [1.0, 0.5, 0.25]
    assert shutdown_args == [{'wait': True, 'cancel_futures': True}]
    queued = list(messages.queue)
    assert any(message[:3] == ('population_saved', 3, 'stopped')
               for message in queued)
    assert any(message[:4] == ('solver_saved', 1, 0.999, 'stopped')
               for message in queued)
    assert queued[-1][0] == 'done'


def test_stop_before_initial_evaluation_still_finishes_and_releases_pool():
    """An immediate Stop has no complete generation, but it must still emit a
    terminal message and fully close its executor so the next Run is safe."""
    target = TEMPORAL_TARGETS['Veto gate']
    config = RunConfig(
        ga=GAConfig(chromosome_count=2, tile_arch='tri3',
                    node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    stop = threading.Event()
    stop.set()
    messages = queue.Queue()
    shutdown_args = []

    class FakePool:
        def shutdown(self, **kwargs):
            shutdown_args.append(kwargs)

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases') as evaluate:
        run_evolution(
            gens=4, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=17,
            backend='nervous', run_config=config, results_dir=directory)
        full_snapshot = load_checkpoint(
            os.path.join(directory, 'latest_population.json'))

    evaluate.assert_not_called()
    assert full_snapshot['genomes'] == []
    assert shutdown_args == [{'wait': True, 'cancel_futures': True}]
    queued = list(messages.queue)
    assert not any(message[0] == 'error' for message in queued)
    assert queued[-1] == ('done', None, 0.0)


def test_seeded_lut_runs_do_not_depend_on_what_ran_before_them():
    """A seeded evolve_lut must build the same population whenever it runs.

    make_seed_genome caches its first _ONTO_POOL_SIZE ontogeny biomorphs in a
    module-level pool that lives as long as the PROCESS. The first run of a
    process grows them fresh; every later run draws its whole population from
    that cache instead - masters grown under an EARLIER target's RNG stream.
    Re-seeding cannot undo that, so the same seed and config gave different
    answers depending on execution order: LUT AND scored 1.000 run first and
    0.750 run after Half adder, silently contaminating rows 2..N of every
    multi-target sweep. Compare populations rather than fitness so the check
    cannot pass by two orderings happening to converge to the same score.
    """
    from substrates.lut.ga import genome_signature
    from substrates.nervous.targets import periodic_combinational_target
    from substrates.snn.targets import gate_target

    def seeded_population(name, seed):
        recorded = []
        original = lut_ga.make_seed_genome

        def recorder(n_chroms=2):
            genome = original(n_chroms)
            recorded.append(genome_signature(genome))
            return genome

        target = periodic_combinational_target(gate_target(name))
        with mock.patch.object(lut_ga, 'make_seed_genome', recorder):
            lut_ga.evolve_lut(target, generations=1, pop=6, n_chroms=2,
                              verbose=False, seed=seed)
        return recorded

    # A small pool keeps the test quick while still filling: growing one
    # ontogeny biomorph costs ~0.3s, and the real pool holds 24.
    with mock.patch.object(lut_ga, '_ONTO_POOL_SIZE', 3):
        lut_ga._ONTO_POOL.clear()
        first = seeded_population('AND', 4242)
        seeded_population('XOR', 99)          # pollute from another RNG stream
        later = seeded_population('AND', 4242)

    assert first, 'the factory was never exercised'
    assert later == first, (
        'a seeded LUT run depends on what ran before it: %d of %d seed genomes '
        'differ' % (sum(1 for a, b in zip(first, later) if a != b), len(first)))


def test_timing_assimilation_clones_parent_before_writing_learned_delays():
    """Write-back must not mutate an evaluated population/cache identity."""
    random.seed(919)
    population = [random_hex_genome(2), random_hex_genome(2)]
    learned = [1.0] * 32
    learned[7] = 1.125
    target = TEMPORAL_TARGETS['Toggle flip-flop']

    with mock.patch(
            'substrates.nervous.temporal.score_temporal_plastic',
            return_value=(0.9, (0.9,), {'state_delays': learned})) as tune:
        parents, changed = nv_ga._assimilate_timing_parents(
            population, [0.9, 0.1], target, count=1,
            samples=4, seed=23, step=0.08)

    assert changed == {0}
    assert parents[0] is not population[0]
    assert parents[0].state_delays == learned
    assert population[0].state_delays is None
    assert parents[1] is population[1]
    assert tune.call_args.kwargs['step'] == 0.08
