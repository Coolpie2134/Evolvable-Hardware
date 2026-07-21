"""
tests/test_diversity.py — the evaluated-population diversity contract
(nv_evo/diversity.py).

Fitness spread is identically zero once everyone solves, so these levels are
what remains. Each test pins one claim about what a level must and must not
distinguish; the cheap ones (signatures, cluster maths) run without growth,
and only the phenotype/behaviour checks develop organisms.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import diversity as dv                            # noqa: E402
from nv_evo.ga import clone_genome, mutate_nv                 # noqa: E402
from nv_evo.genome import random_hex_genome                   # noqa: E402
from nv_evo.pulse import PulseConfig                          # noqa: E402
from nv_evo.targets import TEMPORAL_TARGETS                   # noqa: E402
from evo_runtime.config import GAConfig, RunConfig            # noqa: E402


def _target(name='Coincidence (2-in)', model='pulse_delay'):
    target = TEMPORAL_TARGETS[name]
    config = RunConfig(ga=GAConfig(node_model=model),
                       pulse=PulseConfig(model=model))
    setattr(target, 'pulse_config', config.pulse)
    return target, config


# ── cluster maths ────────────────────────────────────────────────────────────

def test_cluster_stats_separate_count_from_shape():
    """Distinct count alone cannot tell an even spread from a monoculture with
    a long tail — that is what largest_share and effective diversity are for."""
    even = dv.cluster_stats('exact', ['a', 'b', 'c', 'd'])
    assert even.distinct == 4 and even.total == 4
    assert even.largest == 1 and abs(even.effective - 4.0) < 1e-9

    skewed = dv.cluster_stats('exact', ['a', 'a', 'a', 'b'])
    assert skewed.distinct == 2
    assert skewed.largest == 3 and abs(skewed.largest_share - 0.75) < 1e-9
    # same distinct count as an even 2-way split, but far lower effective
    assert skewed.effective < dv.cluster_stats('exact', ['a', 'a', 'b', 'b']).effective


def test_cluster_stats_report_unmeasured_instead_of_dropping():
    stats = dv.cluster_stats('phenotype', ['a', None, 'a', None, 'b'])
    assert stats.total == 3 and stats.distinct == 2
    assert stats.unmeasured == 2


# ── what each genotype level must / must not distinguish ─────────────────────

def test_functional_signature_ignores_tags_and_splits():
    """Tags and split points steer future crossover but do not touch THIS
    organism, so the functional level must not count them as variation."""
    _, config = _target()
    genome = random_hex_genome(3)
    other = clone_genome(genome)
    for index, chromosome in enumerate(other.chromosomes):
        chromosome.tag = chromosome.tag + 7 + index
        chromosome.split = (chromosome.split + 1) % max(1, len(chromosome.genes))
    assert (dv.functional_signature(genome, 'nervous', config)
            == dv.functional_signature(other, 'nervous', config))
    # ...while the exact level, which is full inherited identity, does see them
    assert (dv.exact_signature(genome, 'nervous')
            != dv.exact_signature(other, 'nervous'))


def test_functional_signature_counts_germline_telomere():
    """The telomere sets growth radius L and the settle budget, so it is not
    neutral metadata and must split the functional level."""
    _, config = _target()
    genome = random_hex_genome(2)
    longer = clone_genome(genome)
    top = max(range(len(longer.chromosomes)),
              key=lambda i: getattr(longer.chromosomes[i], 'telomere', 1))
    longer.chromosomes[top].telomere += 3
    assert (dv.functional_signature(genome, 'nervous', config)
            != dv.functional_signature(longer, 'nervous', config))


def test_functional_signature_reads_only_the_live_timing_vector():
    """A dormant vector carried by an old checkpoint must not inflate diversity
    on a run whose node model never reads it."""
    genome = random_hex_genome(2)
    genome.state_delays = [1.0] * 32
    other = clone_genome(genome)
    other.state_delays = [1.0] * 32
    other.state_delays[3] = 2.5

    delay_cfg = RunConfig(ga=GAConfig(node_model='pulse_delay'),
                          pulse=PulseConfig(model='pulse_delay'))
    uniform_cfg = RunConfig(ga=GAConfig(node_model='uniform'),
                            pulse=PulseConfig(model='uniform'))
    # the model that transports evolved delays sees the difference...
    assert (dv.functional_signature(genome, 'nervous', delay_cfg)
            != dv.functional_signature(other, 'nervous', delay_cfg))
    # ...the one that ignores the vector does not
    assert (dv.functional_signature(genome, 'nervous', uniform_cfg)
            == dv.functional_signature(other, 'nervous', uniform_cfg))


def test_tri3_functional_signature_ignores_timing_vectors():
    """tri3 never consults the width/delay vectors, so they cannot be
    functional variation there."""
    config = RunConfig(ga=GAConfig(tile_arch='tri3', node_model='paper_analog'),
                       pulse=PulseConfig(model='paper_analog'))
    genome = random_hex_genome(2, arch='tri3')
    other = clone_genome(genome)
    other.state_delays = [2.0] * 32
    assert (dv.functional_signature(genome, 'nervous', config)
            == dv.functional_signature(other, 'nervous', config))


# ── phenotype ────────────────────────────────────────────────────────────────

def test_phenotype_signature_is_stable_and_grid_sensitive():
    target, config = _target()
    random.seed(31)
    genomes = [random_hex_genome(2) for _ in range(6)]
    signatures = [dv.phenotype_signature(g, 'nervous', target, config.pulse)
                  for g in genomes]
    # deterministic: same genome, same signature
    for genome, signature in zip(genomes, signatures):
        assert dv.phenotype_signature(
            genome, 'nervous', target, config.pulse) == signature
    # a clone is the same circuit
    assert dv.phenotype_signature(
        clone_genome(genomes[0]), 'nervous', target, config.pulse) == signatures[0]
    # and at least some of a random sample are genuinely different bodies
    assert len({s for s in signatures if s is not None}) > 1


def test_phenotype_separates_identical_grids_with_different_delays():
    """Under the legacy profile two genomes can grow the same state grid while
    carrying different per-node delays — physically different circuits."""
    target, config = _target()
    random.seed(5)
    genome = random_hex_genome(2)
    genome.state_delays = [1.0] * 32
    base = dv.phenotype_signature(genome, 'nervous', target, config.pulse)
    if base is None:
        return                                  # dead organism; nothing to test
    slower = clone_genome(genome)
    slower.state_delays = [2.0] * 32
    other = dv.phenotype_signature(slower, 'nervous', target, config.pulse)
    assert other is not None
    assert other[1] == base[1]                  # identical grown grid
    assert other != base                        # but not the same circuit


# ── probe bank + behaviour ───────────────────────────────────────────────────

def test_probe_bank_is_frozen_and_off_spec():
    target, _ = _target()
    first = dv.make_probe_bank(target)
    second = dv.make_probe_bank(target)
    schedules = [tuple(tuple(lane) for lane in trial.input_events)
                 for trial in first.trials]
    assert schedules == [tuple(tuple(lane) for lane in trial.input_events)
                         for trial in second.trials]
    # a different seed is a different measurement
    assert schedules != [tuple(tuple(lane) for lane in trial.input_events)
                         for trial in dv.make_probe_bank(target, seed=1).trials]
    # off-spec: these schedules are not the target's own bank
    own = [getattr(trial, 'input_events', None) for trial in target.trials]
    assert schedules[0] not in [tuple(tuple(l) for l in ev)
                                for ev in own if ev]
    assert first.n_inputs == target.n_inputs
    assert [t.role for t in first.outputs] == [t.role for t in target.outputs]


def test_behavior_signature_is_deterministic_and_quantised():
    target, config = _target()
    random.seed(77)
    probe = dv.make_probe_bank(target)
    genome = random_hex_genome(2)
    first = dv.behavior_signature(genome, 'nervous', target, probe, config.pulse)
    second = dv.behavior_signature(genome, 'nervous', target, probe, config.pulse)
    assert first == second
    if first is not None:
        quantum, roles = first
        assert quantum > 0
        for _role, trains in roles:
            for train in trains:
                assert all(isinstance(t, int) for t in train)


# ── funnel + robustness plumbing ─────────────────────────────────────────────

def test_funnel_is_monotone_and_reports_every_level():
    """Each step down the ladder can only merge clusters, never split them."""
    target, config = _target()
    random.seed(101)
    seed_genome = random_hex_genome(2)
    population = [seed_genome] + [
        mutate_nv(clone_genome(seed_genome), 1.0, chromosome_count=2)
        for _ in range(7)]
    report = dv.diversity_funnel(population, 'nervous', target, config)
    assert [s.level for s in report.levels] == list(dv.LEVELS)
    exact = report.by_level('exact')
    assert exact.distinct <= len(population)
    for coarser, finer in zip(report.levels, report.levels[1:]):
        # a coarser level can only ever have fewer-or-equal distinct classes
        # among the genomes it could measure
        assert finer.distinct <= coarser.distinct + finer.unmeasured
    text = dv.format_funnel(report)
    for level in dv.LEVELS:
        assert dv.LEVEL_LABEL[level] in text
    # the probe bank's provenance must stay visible: a different version or
    # seed is a different measurement and the numbers stop being comparable
    assert 'probe bank v%d' % dv.PROBE_VERSION in text
    assert 'seed %d' % dv.PROBE_SEED in text

    # the report defines its terms and reports measurements — and does NOT
    # editorialise about them
    full = dv.format_report(report, population=8, target_name='X')
    assert 'DEFINITIONS' in full and 'GROUP SIZE DISTRIBUTION' in full
    for term, _meaning in dv.METRIC_MEANING:
        assert term in full
    for level in dv.LEVELS:
        assert dv.LEVEL_MEANING[level].split(':')[0][:24] in full
    for editorial in ('WHAT THIS MEANS', 'monoculture', 'SUCCESS',
                      'Is one behaviour bad'):
        assert editorial not in full, editorial
    assert all(ord(char) < 128 for char in full)   # console-safe (cp1252)
    assert max(len(line) for line in full.splitlines()) <= 80


def test_funnel_accepts_unsolved_nv_analog_tri_population():
    """The analyser measures structure, so it must not require successful
    fitness or digital-node physics."""
    target = TEMPORAL_TARGETS['Coincidence (2-in)']
    config = RunConfig(
        ga=GAConfig(chromosome_count=2, tile_arch='tri3',
                    node_model='paper_analog'),
        pulse=PulseConfig(model='paper_analog'))
    setattr(target, 'pulse_config', config.pulse)
    random.seed(910)
    population = [random_hex_genome(2, arch='tri3') for _ in range(4)]

    report = dv.diversity_funnel(population, 'nervous', target, config)

    assert report.backend == 'nervous'
    for stats in report.levels:
        assert stats.total + stats.unmeasured == len(population)


def test_robustness_reports_kernel_and_bounded_rates():
    target, config = _target()
    random.seed(202)
    population = [random_hex_genome(2) for _ in range(3)]
    report = dv.robustness(population, 'nervous', target, config, samples=2,
                           valid=0.0, seed=9)
    # valid=0.0 accepts every mutant, so local robustness is pinned at 1.0
    assert report.local == 1.0
    assert 0.0 <= report.novel_valid <= 1.0
    assert len(report.per_genome_local) == len(population)
    assert report.kernel == dv.ROBUSTNESS_KERNEL
    assert sum(report.histogram(10)) == len(population)
    text = dv.format_robustness(report, bins=4)
    # measurements and their definitions, no interpretation
    assert 'Novel-valid rate' in text and 'Local robustness' in text
    assert 'Kernel' in text
    # the kernel description is prose and must WRAP, not overrun the panel:
    # it is the longest value in the block and was the one line that didn't
    assert dv.ROBUSTNESS_KERNEL.split()[0] in text
    assert max(len(line) for line in text.splitlines()) <= 80
    assert all(ord(char) < 128 for char in text)


def test_silent_mutants_are_separated_from_survivors():
    """The kernel never returns a clone, but a mutation can land on something
    the phenotype does not express (tag, split point, unexpressed rule,
    non-maximal telomere) and grow the identical circuit. Those score
    identically by construction, so they must not pad the survival rate."""
    target, config = _target()
    random.seed(404)
    population = [random_hex_genome(2) for _ in range(3)]
    report = dv.robustness(population, 'nervous', target, config, samples=6,
                           valid=0.0, seed=5)
    # valid=0.0 accepts everything, so BOTH rates are pinned at 1.0 and the
    # only free quantity is how many samples were silent
    assert report.local == 1.0
    assert report.effective_local in (0.0, 1.0)   # 1.0 unless all were silent
    assert 0.0 <= report.silent <= 1.0
    assert len(report.per_genome_silent) == len(population)
    assert len(report.per_genome_effective) == len(population)
    for value in report.per_genome_effective:
        assert value is None or 0.0 <= value <= 1.0

    text = dv.format_robustness(report, bins=4)
    for term in ('Silent rate', 'Local robustness',
                 'Effective local robustness', 'Novel-valid rate'):
        assert term in text, term
    assert max(len(line) for line in text.splitlines()) <= 80


def test_a_phenotype_neutral_mutant_counts_as_silent_not_as_survival():
    """A hand-made 'mutation' that only moves a split point grows the same
    circuit, so it is silent and contributes nothing to effective robustness."""
    target, config = _target()
    random.seed(77)
    parent = random_hex_genome(3)
    twin = clone_genome(parent)
    for chromosome in twin.chromosomes:
        if len(chromosome.genes) > 2:
            chromosome.split = (chromosome.split + 1) % len(chromosome.genes)
    adapter = dv._adapter('nervous')
    before = dv.phenotype_signature(parent, 'nervous', target, config.pulse,
                                    adapter=adapter)
    after = dv.phenotype_signature(twin, 'nervous', target, config.pulse,
                                   adapter=adapter)
    assert before == after                     # same grown circuit...
    assert (dv.exact_signature(parent, 'nervous')
            != dv.exact_signature(twin, 'nervous'))   # ...different genotype


def test_robustness_is_repeatable_for_a_fixed_seed():
    target, config = _target()
    random.seed(303)
    population = [random_hex_genome(2) for _ in range(2)]
    first = dv.robustness(population, 'nervous', target, config, samples=3,
                          valid=0.5, seed=17)
    second = dv.robustness(population, 'nervous', target, config, samples=3,
                           valid=0.5, seed=17)
    assert first.per_genome_local == second.per_genome_local
    assert first.per_genome_novel == second.per_genome_novel
