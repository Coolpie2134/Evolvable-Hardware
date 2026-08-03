"""Executable truth-table inventory checks for the LUT substrate."""
from __future__ import annotations

import dataclasses
import os
import random
import tempfile

from runtime.config import GAConfig, LUT_FUNCTION_FAMILIES, RunConfig
from runtime.checkpoint import load_checkpoint, save_checkpoint
from substrates.lut.functions import (
    AND, MUX, OR, ROUTING, THRESHOLD, UNRESTRICTED, VETO, XOR,
    FAMILY_TABLES, INPUT_TABLES, allowed_function_table,
    enabled_named_tables,
)
from substrates.lut.ga import (
    constrain_genome_functions, mutate_lut, next_population,
)
from substrates.lut.genome import (
    Chromosome, Genome, LutGene, random_lut_genome,
)
from substrates.lut.lut import grow_lut
from substrates.nervous.targets import TEMPORAL_TARGETS


NAMED = (ROUTING, AND, OR, XOR, VETO, THRESHOLD, MUX)


def _executable_tables(genome):
    for chromosome in genome.chromosomes:
        if getattr(chromosome, 'wiring', False):
            continue
        for gene in chromosome.genes:
            yield gene.self_out
    if genome.seed_state is not None:
        yield from genome.seed_state


def _assert_uses_only(genome, families):
    assert all(
        allowed_function_table(table, families)
        for table in _executable_tables(genome))


def _assert_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_named_lut_banks_are_permanent_quiescent_physical_tables():
    assert LUT_FUNCTION_FAMILIES == NAMED + (UNRESTRICTED,)
    assert {family: len(FAMILY_TABLES[family]) for family in NAMED} == {
        ROUTING: 4,
        AND: 11,
        OR: 11,
        XOR: 11,
        VETO: 12,
        THRESHOLD: 6,
        MUX: 24,
    }
    assert INPUT_TABLES == {
        'N': 0xAAAA,
        'S': 0xCCCC,
        'E': 0xF0F0,
        'W': 0xFF00,
    }
    assert all(
        table & 1 == 0
        for family in NAMED
        for table in FAMILY_TABLES[family])
    assert 0 not in enabled_named_tables(NAMED)
    assert allowed_function_table(0, NAMED)


def test_lut_function_selection_round_trips_and_old_runs_stay_unrestricted():
    config = RunConfig(ga=GAConfig(
        chromosome_count=2,
        lut_function_families=(XOR, ROUTING, AND)))
    assert config.ga.lut_function_families == (ROUTING, AND, XOR)
    restored = RunConfig.from_dict(dataclasses.asdict(config))
    assert restored == config

    legacy = dataclasses.asdict(config)
    legacy['ga'].pop('lut_function_families')
    assert RunConfig.from_dict(
        legacy).ga.lut_function_families == (UNRESTRICTED,)

    _assert_value_error(lambda: GAConfig(lut_function_families=()))
    _assert_value_error(
        lambda: GAConfig(lut_function_families=(AND, AND)))
    _assert_value_error(
        lambda: GAConfig(lut_function_families=('NAND',)))


def test_lut_function_selection_round_trips_through_a_checkpoint():
    families = (ROUTING, XOR)
    config = RunConfig(ga=GAConfig(
        chromosome_count=2,
        lut_function_families=families))
    genome = random_lut_genome(2, function_families=families)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'lut-function-banks.json')
        save_checkpoint(
            path, genome, 0.75, TEMPORAL_TARGETS['SR latch'],
            None, 91, 'lut', config)
        restored = load_checkpoint(path)

    assert restored['run_config'].ga.lut_function_families == families
    assert restored['target']._lut_function_families == families


def test_explicit_unrestricted_mode_is_bit_identical_to_the_legacy_default():
    random.seed(13017)
    legacy = random_lut_genome(2)
    random.seed(13017)
    explicit = random_lut_genome(
        2, function_families=(UNRESTRICTED,))
    assert explicit == legacy

    random.seed(31071)
    legacy_child = mutate_lut(
        legacy, mean_mutations=5.0, chromosome_count=2)
    random.seed(31071)
    explicit_child = mutate_lut(
        explicit, mean_mutations=5.0, chromosome_count=2,
        function_families=(UNRESTRICTED,))
    assert explicit_child == legacy_child


def test_restricted_initialisation_mutation_and_growth_never_escape_the_bank():
    families = (ROUTING, AND, XOR, VETO)
    random.seed(8021)
    genome = random_lut_genome(
        2, function_families=families)
    _assert_uses_only(genome, families)

    for _ in range(40):
        genome = mutate_lut(
            genome, mean_mutations=5.0, chromosome_count=2,
            function_families=families)
        _assert_uses_only(genome, families)

    grid = grow_lut(genome, seeds=((0, 0),), grid_size=9, iters=3)
    assert all(
        allowed_function_table(table, families)
        for state in grid.values()
        for table in state)


def test_constraint_changes_only_executable_tables_not_cam_contexts():
    contexts = (0x1234, 0x2345, 0x3456, 0x4567, 0x5678)
    gene = LutGene(*contexts, self_out=0x6789)
    genome = Genome(
        chromosomes=[Chromosome(genes=[gene])],
        seed_state=None)

    random.seed(71)
    constrained = constrain_genome_functions(genome, (ROUTING,))
    kept = constrained.chromosomes[0].genes[0]
    assert (
        kept.ctx_n, kept.ctx_e, kept.ctx_s, kept.ctx_w, kept.self_in
    ) == contexts
    _assert_uses_only(constrained, (ROUTING,))


def test_restricting_a_legacy_genome_projects_inherited_executable_tables():
    genome = random_lut_genome(2)
    genome.seed_state = None
    random.seed(712)
    child = mutate_lut(
        genome, mean_mutations=1.0, chromosome_count=2,
        function_families=(XOR,))
    _assert_uses_only(child, (XOR,))


def test_population_factory_and_breeding_obey_the_selected_inventory():
    families = (AND, OR, MUX)
    random.seed(411)
    population = [
        random_lut_genome(2, function_families=families)
        for _ in range(8)
    ]
    config = GAConfig(
        chromosome_count=2,
        immigrant_fraction=0.25,
        lut_function_families=families)

    children = next_population(
        population,
        [index / 10 for index in range(len(population))],
        make_genome=lambda: random_lut_genome(
            2, function_families=families),
        mean_mutations=4.0,
        ga_config=config)

    assert len(children) == len(population)
    assert all(len(child.chromosomes) == 2 for child in children)
    for child in children:
        _assert_uses_only(child, families)
