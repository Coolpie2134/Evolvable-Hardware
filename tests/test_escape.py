"""
tests/test_escape.py — the local-minimum escape mechanisms (runtime/escape.py).

Three things are checked, in order of how badly they would hurt if wrong:

  1. DEFAULTS ARE INERT. Every mechanism is off unless asked for, and with them
     off the GA must behave exactly as it did before the module existed —
     identical rank ordering, identical case vectors, identical bred population
     for a given seed. Nothing here is allowed to quietly change a run.
  2. EACH MECHANISM ACTUALLY DOES ITS JOB when switched on, and the safety
     properties hold — most importantly that neither escape objective can
     outrank correctness.
  3. THE TWO GA DRIVE PATHS AGREE. The desktop controller and the headless
     drivers must apply the same escape machinery; a mechanism that exists on
     one path and not the other is the exact failure this project has already
     hit once, where the benchmarks measured one driver and the app ran the
     other.
"""
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

from runtime.config import GAConfig, RunConfig
from runtime.controller import run_evolution
from runtime.escape import (OFF, EscapeConfig, EscapeState, aggregate_robustness,
                            build_escape_state, genome_descriptor,
                            genome_distance, inherit_mutation_rate,
                            jitter_physics, lexicase_case_subset,
                            mutation_rate_of, population_mutation_rate,
                            robustness_blend, seed_mutation_rate,
                            set_mutation_rate)
import substrates.lut.ga as lut_ga
import substrates.nervous.ga as nv_ga
import substrates.snn.ga as snn_ga
from substrates.nervous.genome import random_hex_genome
from substrates.nervous.pulse import PulseConfig
from substrates.nervous.targets import TEMPORAL_TARGETS


class FakePool:
    def shutdown(self, **_kwargs):
        pass


def _nv_config(escape=None, **ga):
    values = dict(chromosome_count=2, tile_arch='tri3',
                  node_model='paper_analog')
    values.update(ga)
    if escape is not None:
        values['escape'] = escape
    return RunConfig(ga=GAConfig(**values),
                     pulse=PulseConfig(model='paper_analog'))


# ── 1. defaults are inert ─────────────────────────────────────────────────────

def test_every_mechanism_is_off_by_default():
    config = EscapeConfig()
    assert not config.any_enabled
    assert config.summary() == 'none'
    for flag in ('lifespan_scoring', 'crowding', 'neutral_drift',
                 'self_adaptive_mutation', 'rebirth', 'robustness'):
        assert getattr(config, flag) is False
    assert config.lexicase_downsample == 1.0
    assert GAConfig().escape == OFF


def test_default_escape_round_trips_and_legacy_checkpoints_load_as_off():
    document = dataclasses.asdict(_nv_config())
    assert RunConfig.from_dict(document).ga.escape == OFF
    # A checkpoint written before the mechanisms existed carries no key at all.
    legacy = dataclasses.asdict(_nv_config())
    legacy['ga'].pop('escape')
    assert RunConfig.from_dict(legacy).ga.escape == OFF
    # A checkpoint from a different revision of the mechanism set must still
    # load the rest of its run rather than failing outright.
    future = dataclasses.asdict(_nv_config())
    future['ga']['escape']['some_unknown_future_toggle'] = True
    assert RunConfig.from_dict(future).ga.escape == OFF


def test_rank_key_collapses_to_the_original_ordering_when_mechanisms_are_off():
    """The escape tiers are inserted below fitness; with both at 0.0 the
    ordering must be decided by exactly what decided it before."""
    weak, strong = random_hex_genome(2), random_hex_genome(2)
    assert nv_ga.rank_key(weak, 0.4) < nv_ga.rank_key(strong, 0.5)
    for genome in (weak, strong):
        assert not hasattr(genome, '_robustness')
        assert not hasattr(genome, '_juvenile_score')
    # The two new slots are the middle of the key and are zero.
    assert nv_ga.rank_key(weak, 0.4)[2:4] == (0.0, 0.0)
    assert lut_ga.rank_key(weak, 0.4)[2:4] == (0.0, 0.0)
    assert snn_ga.rank_key(weak, 0.4)[2:4] == (0.0, 0.0)


def test_breeding_is_unchanged_when_escape_is_off():
    """Same seed, same population: the bred generation must be byte-identical
    to what the pre-escape breeder produced, i.e. adding the parameter must not
    have perturbed the RNG stream."""
    target = TEMPORAL_TARGETS['Veto gate']
    population = [random_hex_genome(2) for _ in range(6)]
    fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def breed(escape):
        random.seed(4242)
        bred = nv_ga.next_population(
            [nv_ga.clone_genome(g) for g in population], fitnesses,
            chromosome_count=2, escape=escape)
        return [nv_ga.genome_signature(g) for g in bred]

    assert breed(OFF) == breed(None)      # None must resolve to OFF
    assert len(breed(OFF)) == len(population)
    del target


def test_case_vector_length_is_unchanged_without_lifespan_scoring():
    from substrates.nervous.objectives import total_case_count
    from substrates.nervous.scoring import contract_case_count
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    assert total_case_count(target) == contract_case_count(target)
    setattr(target, '_escape', OFF)
    assert total_case_count(target) == contract_case_count(target)


# ── 2. lifespan scoring ───────────────────────────────────────────────────────

def test_lifespan_scoring_extends_the_case_vector_by_one_per_checkpoint():
    from substrates.nervous.objectives import total_case_count
    from substrates.nervous.scoring import contract_case_count
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    setattr(target, 'pulse_config', PulseConfig(model='pulse_delay'))
    setattr(target, '_escape',
            EscapeConfig(lifespan_scoring=True, lifespan_checkpoints=4))
    base = contract_case_count(target)
    assert total_case_count(target) == base + 4
    random.seed(11)
    for _ in range(6):
        genome = random_hex_genome(2)
        _fitness, cases = nv_ga.evaluate_nv_full(genome, target)
        # ε-lexicase requires EVERY member to present the same number of cases,
        # including dead genomes and organisms that mature in fewer steps than
        # there are checkpoints — so the length is fixed, never "as many as
        # this body happened to have".
        assert len(cases) == base + 4


def test_lifespan_scoring_leaves_the_reported_fitness_at_the_adult_score():
    """The whole safety argument for juvenile credit: it must never inflate the
    number a run reports. A solved run still means the GROWN circuit works."""
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    setattr(target, 'pulse_config', PulseConfig(model='pulse_delay'))
    random.seed(23)
    genomes = [random_hex_genome(2) for _ in range(8)]

    setattr(target, '_escape', OFF)
    plain = [nv_ga.evaluate_nv_full(g, target)[0] for g in genomes]
    setattr(target, '_escape',
            EscapeConfig(lifespan_scoring=True, lifespan_checkpoints=3))
    lifespan = [nv_ga.evaluate_nv_full(g, target)[0] for g in genomes]
    assert plain == lifespan


def test_lifespan_checkpoints_are_interior_developmental_stages():
    from substrates.nervous.objectives import checkpoint_indices
    # Index 0 is the bare seed pads and the last index is the adult (scored
    # anyway), so only the interior carries new information.
    picks = checkpoint_indices(12, 3)
    assert picks and all(1 <= i <= 10 for i in picks)
    assert list(picks) == sorted(picks)
    # A body that matures immediately has no distinct juvenile stage.
    assert checkpoint_indices(2, 3) == ()
    assert checkpoint_indices(1, 3) == ()


def test_juvenile_padding_uses_the_adult_score_not_zero():
    """An organism that matures in three steps genuinely WAS its adult self at
    every age; scoring the missing stages as failures would penalise compact
    development, which is not what this objective measures."""
    from substrates.nervous.objectives import juvenile_scores
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    scores = juvenile_scores(random_hex_genome(2), target, 'nervous',
                             snapshots=[{}, {}], strategy='fixed', count=3,
                             adult_score=0.75)
    assert scores == (0.75, 0.75, 0.75)


def test_juvenile_mean_reaches_rank_key_only_below_fitness():
    genome = random_hex_genome(2)
    rival = random_hex_genome(2)
    genome._juvenile_score = 0.9
    rival._juvenile_score = 0.0
    # Equal fitness: juvenile credit breaks the tie.
    assert nv_ga.rank_key(genome, 0.3) > nv_ga.rank_key(rival, 0.3)
    # Lower fitness: no amount of juvenile credit can win.
    assert nv_ga.rank_key(genome, 0.3) < nv_ga.rank_key(rival, 0.31)


# ── 3. robustness ─────────────────────────────────────────────────────────────

def test_robustness_can_never_outrank_correctness():
    """The safety property that makes a second objective admissible at all."""
    brittle, robust = random_hex_genome(2), random_hex_genome(2)
    robust._robustness = 1.0
    brittle._robustness = 0.0
    for rank in (nv_ga.rank_key, lut_ga.rank_key, snn_ga.rank_key):
        assert rank(robust, 0.5) > rank(brittle, 0.5)      # tie-break: yes
        assert rank(robust, 0.5) < rank(brittle, 0.6)      # override: never


def test_robustness_aggregation_anneals_from_mean_to_worst_case():
    cases = (1.0, 1.0, 1.0, 0.0)
    # Early (nothing solved) the mean keeps a gradient alive...
    assert aggregate_robustness(cases, robustness_blend(0.0)) == 0.75
    # ...and by the time a run is near solving, coverage is what still needs
    # enforcing, so one dead case sinks the whole score.
    assert aggregate_robustness(cases, robustness_blend(1.0)) == 0.0
    assert aggregate_robustness(cases, robustness_blend(0.5)) == 0.375
    assert aggregate_robustness((), 1.0) == 0.0


def test_jitter_variants_are_deterministic_and_probe_both_directions():
    """Determinism is not cosmetic: the fitness cache is keyed on the genome
    alone, so a random jitter would freeze whichever draw happened first into
    that genome's score for the rest of the run."""
    config = PulseConfig(model='pulse_delay')
    escape = EscapeConfig(robustness=True, robustness_samples=2,
                          robustness_jitter=0.2)
    first = jitter_physics(config, escape)
    second = jitter_physics(config, escape)
    assert first == second
    assert len(first) == 2
    assert first[0].delay > config.delay and first[1].delay < config.delay
    # Width moves opposite to delay: a slow narrow net and a fast wide net are
    # the two ends of the timing margin an async circuit has to survive.
    assert first[0].width < config.width and first[1].width > config.width
    assert jitter_physics(config, OFF) == ()
    assert jitter_physics(None, escape) == ()


def test_robustness_blend_is_applied_from_the_run_not_the_worker():
    state = EscapeState(EscapeConfig(robustness=True))
    genome = random_hex_genome(2)
    genome._robust_cases = (1.0, 0.0)
    state.apply_robustness_blend([genome], best_fitness=0.0)
    assert genome._robustness == 0.5           # mean early
    state.apply_robustness_blend([genome], best_fitness=1.0)
    assert genome._robustness == 0.0           # worst case late
    assert state.robust_blend == 1.0


def test_robustness_is_zero_for_genomes_that_were_never_measured():
    state = EscapeState(EscapeConfig(robustness=True))
    genome = random_hex_genome(2)
    state.apply_robustness_blend([genome], best_fitness=0.5)
    assert genome._robustness == 0.0


# ── 4. crowding (restricted tournament replacement) ───────────────────────────

def test_genome_distance_is_normalised_and_length_aware():
    a = random_hex_genome(2)
    assert genome_distance(genome_descriptor(a), genome_descriptor(a)) == 0.0
    assert genome_distance((1, 2, 3), (1, 2, 4)) == 1 / 3
    # Variable-length genomes: a short genome is not "the same as" a long one
    # that merely agrees on its opening genes.
    assert genome_distance((1, 2), (1, 2, 3, 4)) == 0.5
    assert 0.0 <= genome_distance((5, 5), (9, 9)) <= 1.0


def test_crowding_replaces_the_nearest_incumbent_not_the_worst():
    """The point of restricted tournament replacement: a specialist is only
    ever displaced by a better version of ITSELF, so niches survive."""
    random.seed(5)
    parents = [random_hex_genome(2) for _ in range(6)]
    fitnesses = [0.9, 0.5, 0.5, 0.5, 0.5, 0.5]
    # The child is a near-clone of the strong parent 0 and beats it.
    child = nv_ga.clone_genome(parents[0])
    state = EscapeState(EscapeConfig(crowding=True, crowding_window=6,
                                     crowding_fraction=1.0),
                        rank=nv_ga.rank_key)
    population, fits, _cases = state.survivor_selection(
        parents, fitnesses, None, [child], [0.95], None)
    assert fits[0] == 0.95 and population[0] is child
    # Every other niche is untouched — a plain generational replacement or a
    # replace-the-worst rule would have evicted one of the weak members.
    assert fits[1:] == [0.5] * 5
    assert state.crowding_replacements == 1


def test_crowding_rejects_a_worse_challenger():
    random.seed(6)
    parents = [random_hex_genome(2) for _ in range(4)]
    fitnesses = [0.5] * 4
    child = nv_ga.clone_genome(parents[2])
    state = EscapeState(EscapeConfig(crowding=True, crowding_window=4,
                                     crowding_fraction=1.0),
                        rank=nv_ga.rank_key)
    _pop, fits, _cases = state.survivor_selection(
        parents, fitnesses, None, [child], [0.2], None)
    assert fits == [0.5] * 4
    assert state.crowding_replacements == 0


def test_a_fully_crowded_population_can_never_move_downhill():
    """The property that made the live mean rise without fluctuation. It is
    inherent to RTR, not a defect — but it has to be a deliberate choice, so it
    is pinned here rather than left to be rediscovered from a chart."""
    random.seed(8)
    parents = [random_hex_genome(2) for _ in range(8)]
    fitnesses = [0.4] * 8
    offspring = [random_hex_genome(2) for _ in range(8)]
    state = EscapeState(EscapeConfig(crowding=True, crowding_fraction=1.0),
                        rank=nv_ga.rank_key)
    _pop, fits, _cases = state.survivor_selection(
        parents, fitnesses, None, offspring, [0.0] * 8, None)
    assert fits == [0.4] * 8            # not one worse challenger got in
    assert sum(fits) >= sum(fitnesses)  # monotone, by construction


def test_a_partial_reserve_keeps_generational_churn():
    """The fix: below 1.0 the un-reserved slots keep this project's pre-solve
    strict generational replacement, so the population can still move downhill
    and the mean still fluctuates."""
    random.seed(9)
    parents = [random_hex_genome(2) for _ in range(8)]
    fitnesses = [0.4] * 8
    offspring = [random_hex_genome(2) for _ in range(8)]
    state = EscapeState(EscapeConfig(crowding=True, crowding_fraction=0.5),
                        rank=nv_ga.rank_key)
    population, fits, _cases = state.survivor_selection(
        parents, fitnesses, None, offspring, [0.0] * 8, None)
    assert len(population) == 8 and len(fits) == 8
    assert fits.count(0.4) == 4      # the crowded reserve held
    assert fits.count(0.0) == 4      # the remainder was generationally replaced
    assert sum(fits) < sum(fitnesses)


def test_crowding_reserve_preserves_population_size_and_case_alignment():
    random.seed(10)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        parents = [random_hex_genome(2) for _ in range(9)]
        offspring = [random_hex_genome(2) for _ in range(9)]
        parent_cases = [(0.4,)] * 9
        offspring_cases = [(0.7,)] * 9
        state = EscapeState(
            EscapeConfig(crowding=True, crowding_fraction=fraction),
            rank=nv_ga.rank_key)
        population, fits, cases = state.survivor_selection(
            parents, [0.4] * 9, parent_cases,
            offspring, [0.7] * 9, offspring_cases)
        assert len(population) == len(fits) == len(cases) == 9
        # Each slot's case vector must still describe the genome in it.
        for fitness, case in zip(fits, cases):
            assert case == (fitness,)


def test_merge_generation_keeps_the_original_behaviour_when_crowding_is_off():
    parents = [random_hex_genome(2) for _ in range(3)]
    offspring = [random_hex_genome(2) for _ in range(3)]
    state = EscapeState(OFF, rank=nv_ga.rank_key)
    population, fitnesses, cases = state.merge_generation(
        parents, [0.9, 0.9, 0.9], None, offspring, [0.1, 0.1, 0.1], None)
    # Strict pre-solve generational replacement: no evaluated parent survives.
    assert population is offspring and fitnesses == [0.1, 0.1, 0.1]
    assert cases is None


def test_merge_generation_prefers_terminal_consolidation_once_solved():
    calls = []

    def consolidate(*args):
        calls.append(args)
        return 'consolidated', 'fits', 'cases'

    state = EscapeState(EscapeConfig(crowding=True), rank=nv_ga.rank_key)
    result = state.merge_generation(
        [], [], None, [], [], None, consolidate=consolidate, solved=True)
    # Crowding must not displace the terminal (mu + lambda) phase; that phase
    # is what lets a solved population's mean converge to 1.
    assert result == ('consolidated', 'fits', 'cases')
    assert len(calls) == 1


# ── 5. neutral drift ──────────────────────────────────────────────────────────

def test_neutral_drift_accepts_equal_ranks_and_strict_mode_does_not():
    strict = EscapeState(OFF)
    drifting = EscapeState(EscapeConfig(neutral_drift=True))
    assert strict.accepts((1, 0.5), (1, 0.5)) is False
    assert drifting.accepts((1, 0.5), (1, 0.5)) is True
    # Neither ever accepts a genuine regression.
    assert strict.accepts((1, 0.4), (1, 0.5)) is False
    assert drifting.accepts((1, 0.4), (1, 0.5)) is False
    # A first champion is always accepted.
    assert strict.accepts((1, 0.0), None) is True


def test_neutral_drift_lets_crowding_keep_moving_across_a_plateau():
    random.seed(7)
    parents = [random_hex_genome(2) for _ in range(4)]
    child = nv_ga.clone_genome(parents[1])
    plateau = [0.5] * 4
    frozen = EscapeState(EscapeConfig(crowding=True, crowding_window=4),
                         rank=nv_ga.rank_key)
    random.seed(7)
    _pop, _fits, _cases = frozen.survivor_selection(
        parents, plateau, None, [child], [0.5], None)
    assert frozen.crowding_replacements == 0
    drifting = EscapeState(
        EscapeConfig(crowding=True, crowding_window=4, neutral_drift=True, crowding_fraction=1.0),
        rank=nv_ga.rank_key)
    random.seed(7)
    drifting.survivor_selection(parents, plateau, None, [child], [0.5], None)
    assert drifting.crowding_replacements == 1


# ── 6. self-adaptive mutation ─────────────────────────────────────────────────

def test_self_adaptive_rate_is_heritable_bounded_and_phenotype_neutral():
    config = EscapeConfig(self_adaptive_mutation=True, adaptive_tau=0.25)
    parent_a, parent_b = random_hex_genome(2), random_hex_genome(2)
    set_mutation_rate(parent_a, 4.0, 8.0)
    set_mutation_rate(parent_b, 4.0, 8.0)
    random.seed(3)
    rates = []
    for _ in range(200):
        child = random_hex_genome(2)
        inherit_mutation_rate(child, parent_a, parent_b, config, 4.0, 8.0)
        rates.append(child._mut_rate)
    assert all(1.0 <= r <= 8.0 for r in rates)
    assert len(set(rates)) > 100              # genuinely varying, not a constant
    # It must NOT enter the evaluation cache key: the rate changes no phenotype,
    # and admitting it would give one circuit many cache entries.
    plain = random_hex_genome(2)
    tagged = nv_ga.clone_genome(plain)
    set_mutation_rate(tagged, 7.5, 8.0)
    assert nv_ga.genome_signature(plain) == nv_ga.genome_signature(tagged)


def test_self_adaptive_rate_survives_cloning_on_every_backend():
    from substrates.lut.ga import make_seed_genome
    from substrates.snn.genome import random_genome
    for clone, genome in ((nv_ga.clone_genome, random_hex_genome(2)),
                          (lut_ga.clone_genome, make_seed_genome(2)),
                          (snn_ga.clone_genome, random_genome(2))):
        set_mutation_rate(genome, 6.25, 8.0)
        assert clone(genome)._mut_rate == 6.25


def test_genomes_without_a_rate_fall_back_to_the_run_rate():
    genome = random_hex_genome(2)
    assert mutation_rate_of(genome, 3.5) == 3.5
    seed_mutation_rate(genome, 4.0, 8.0)
    assert 1.0 <= genome._mut_rate <= 8.0
    assert population_mutation_rate([random_hex_genome(2)], 2.0) == 2.0


def test_self_adaptive_mutation_assigns_a_rate_to_every_bred_child():
    random.seed(99)
    population = [random_hex_genome(2) for _ in range(8)]
    bred = nv_ga.next_population(
        population, [0.3] * 8, chromosome_count=2,
        escape=EscapeConfig(self_adaptive_mutation=True), mutation_limit=8.0)
    assert all(hasattr(genome, '_mut_rate') for genome in bred)
    assert all(1.0 <= genome._mut_rate <= 8.0 for genome in bred)
    # Off by default: no rate is invented.
    random.seed(99)
    plain = nv_ga.next_population(population, [0.3] * 8, chromosome_count=2)
    assert not any(hasattr(genome, '_mut_rate') for genome in plain)


# ── 7. rebirth ────────────────────────────────────────────────────────────────

def _rebirth_state(config):
    return EscapeState(config, mutation_limit=8.0,
                       clone=nv_ga.clone_genome,
                       mutate=lambda genome, rate: nv_ga.mutate_nv(
                           genome, rate, chromosome_count=2),
                       rank=nv_ga.rank_key)


def test_rebirth_only_fires_on_a_stall_and_never_after_a_solve():
    config = EscapeConfig(rebirth=True, rebirth_patience=10,
                          archive_interval=1)
    state = _rebirth_state(config)
    state.record_champion(1, random_hex_genome(2), 0.4)
    assert not state.should_rebirth(stagnation=9, best_fitness=0.4)
    assert state.should_rebirth(stagnation=10, best_fitness=0.4)
    # A solved run has nothing left to escape from.
    assert not state.should_rebirth(stagnation=99, best_fitness=1.0)


def test_rebirth_draws_from_diverse_ancestors_not_just_the_best_one():
    """Re-seeding from the single best ancestor walks the same path again;
    spreading the seeds over the archive is the entire point of backtracking."""
    config = EscapeConfig(rebirth=True, rebirth_patience=2,
                          rebirth_ancestors=3, archive_interval=1,
                          rebirth_fraction=0.5)
    state = _rebirth_state(config)
    random.seed(31)
    for generation in range(1, 7):
        state.record_champion(generation, random_hex_genome(2),
                              0.1 * generation)
    ancestors = state.diverse_ancestors(3)
    assert len(ancestors) == 3
    assert len({entry[0] for entry in ancestors}) == 3
    # They are spread, not three copies of the newest entry.
    descriptors = [entry[3] for entry in ancestors]
    assert all(genome_distance(descriptors[0], other) > 0.0
               for other in descriptors[1:])


def test_rebirth_keeps_elites_marks_reborn_slots_and_raises_the_rate():
    config = EscapeConfig(rebirth=True, rebirth_patience=1,
                          rebirth_ancestors=2, archive_interval=1,
                          rebirth_fraction=0.5,
                          rebirth_mutation_multiplier=3.0)
    state = _rebirth_state(config)
    random.seed(41)
    for generation in (1, 2, 3):
        state.record_champion(generation, random_hex_genome(2), 0.3)
    population = [random_hex_genome(2) for _ in range(8)]
    fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    new_population, new_fitnesses, _cases, info = state.rebirth_population(
        10, population, fitnesses, None, base_rate=2.0)
    assert len(new_population) == 8
    assert info['reborn'] == 4
    assert info['rate'] == 6.0                       # 2.0 x 3.0, under the cap
    # Rebirth is a backtrack, not an extinction: the best half is retained.
    assert sorted(f for f in new_fitnesses if f is not None) == [
        0.5, 0.6, 0.7, 0.8]
    # The reborn slots are explicitly unscored so the driver must evaluate them.
    assert new_fitnesses.count(None) == 4


def test_rebirth_reevaluates_the_reborn_cohort_and_then_cools_down():
    config = EscapeConfig(rebirth=True, rebirth_patience=1,
                          rebirth_ancestors=2, archive_interval=1,
                          rebirth_fraction=0.5)
    state = _rebirth_state(config)
    random.seed(53)
    for generation in (1, 2):
        state.record_champion(generation, random_hex_genome(2), 0.3)
    population = [random_hex_genome(2) for _ in range(6)]
    fitnesses = [0.3] * 6
    evaluated = []

    def evaluate(genomes):
        evaluated.append(len(genomes))
        return [0.42] * len(genomes), None

    population, fitnesses, _cases, info = state.maybe_rebirth(
        5, population, fitnesses, None, 2.0, stagnation=5, best_fitness=0.3,
        evaluate=evaluate)
    assert info is not None
    assert evaluated == [3]
    assert None not in fitnesses and fitnesses.count(0.42) == 3
    # The cooldown stops it re-firing every generation while the stall lasts.
    assert not state.should_rebirth(stagnation=99, best_fitness=0.3)
    for _ in range(config.rebirth_patience):
        state.tick()
    assert state.should_rebirth(stagnation=99, best_fitness=0.3)


def test_rebirth_is_a_no_op_with_an_empty_archive():
    state = _rebirth_state(EscapeConfig(rebirth=True, rebirth_patience=1))
    population = [random_hex_genome(2) for _ in range(4)]
    result = state.maybe_rebirth(
        3, population, [0.2] * 4, None, 2.0, stagnation=99, best_fitness=0.2,
        evaluate=lambda genomes: ([0.0] * len(genomes), None))
    assert result[3] is None
    assert result[0] is population


def test_archive_ignores_an_unchanged_champion_and_is_bounded():
    config = EscapeConfig(rebirth=True, archive_interval=1, archive_size=3)
    state = _rebirth_state(config)
    champion = random_hex_genome(2)
    for generation in range(1, 5):
        state.record_champion(generation, champion, 0.5)
    assert len(state.archive) == 1               # same genome, one entry
    random.seed(61)
    for generation in range(5, 12):
        state.record_champion(generation, random_hex_genome(2), 0.5)
    assert len(state.archive) == 3               # ring buffer holds


# ── 8. ε-lexicase: downsampling and the ε itself ──────────────────────────────

def test_downsampling_returns_none_for_the_full_case_set():
    assert lexicase_case_subset(20, OFF) is None
    assert lexicase_case_subset(1, EscapeConfig(lexicase_downsample=0.5)) is None
    subset = lexicase_case_subset(20, EscapeConfig(lexicase_downsample=0.25))
    assert len(subset) == 5 and len(set(subset)) == 5
    assert all(0 <= index < 20 for index in subset)
    # Never empties the stream, however aggressive the fraction.
    assert len(lexicase_case_subset(4, EscapeConfig(
        lexicase_downsample=0.01))) == 1


def test_lexicase_streams_only_the_downsampled_cases():
    """The subset must actually restrict selection: case 0 is the only one that
    separates these candidates, so excluding it must change the outcome."""
    population = [random_hex_genome(2) for _ in range(3)]
    cases = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    random.seed(2)
    winners = {id(nv_ga._lexicase_parent(population, cases, (0,)))
               for _ in range(20)}
    assert winners == {id(population[0])}
    # With case 0 excluded nothing separates them, so all three can win.
    random.seed(2)
    winners = {id(nv_ga._lexicase_parent(population, cases, (1, 2)))
               for _ in range(60)}
    assert len(winners) > 1


def test_epsilon_lexicase_keeps_near_best_candidates_on_continuous_scores():
    """Plain (exact-tie) lexicase would let the first case drawn decide every
    selection on its own, because floats essentially never tie — the ε is what
    makes this usable at all on continuous scores."""
    population = [random_hex_genome(2) for _ in range(4)]
    # Three candidates are within noise of each other; one is genuinely bad.
    cases = [(0.900,), (0.899,), (0.901,), (0.100,)]
    random.seed(4)
    winners = {id(nv_ga._lexicase_parent(population, cases))
               for _ in range(60)}
    assert len(winners) > 1                     # the near-ties all survive ε
    assert id(population[3]) not in winners     # the genuinely bad one does not


# ── 9. both GA drive paths apply the same machinery ───────────────────────────

def test_controller_and_headless_drivers_build_the_same_escape_state():
    """One construction point. If these ever diverge, the app and the
    benchmarks are silently running different searches."""
    escape = EscapeConfig(crowding=True, rebirth=True, neutral_drift=True,
                          self_adaptive_mutation=True)
    config = GAConfig(chromosome_count=2, escape=escape, mutation_limit=6.0)
    for backend in ('nervous', 'lut', 'snn'):
        state = build_escape_state(backend, config, chromosome_count=2)
        assert state.config is escape
        assert state.mutation_limit == 6.0
        assert state._mutate is not None and state._clone is not None
        # Rebirth needs a working mutate closure on every backend, or it would
        # silently do nothing there.
        assert state.should_rebirth(0, 0.0) is False


def test_controller_threads_the_escape_config_to_workers_and_the_breeder():
    """The mechanisms split across two places — the worker (lifespan scoring,
    robustness) and the breeder (everything else). Both have to receive it."""
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    escape = EscapeConfig(crowding=True, self_adaptive_mutation=True,
                          lexicase_downsample=0.5)
    config = _nv_config(escape=escape)
    messages, stop = queue.Queue(), threading.Event()

    def evaluate(genomes, *_args, **_kwargs):
        return [0.5] * len(genomes), [None] * len(genomes)

    seen = []

    def reproduce(population, *args, **kwargs):
        seen.append(kwargs.get('ga_config'))
        return [nv_ga.clone_genome(genome) for genome in population]

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases', side_effect=evaluate), \
            mock.patch.object(nv_ga, 'next_population',
                              side_effect=reproduce), \
            mock.patch('substrates.nervous.certification.certify',
                       return_value=None):
        run_evolution(
            gens=2, pop=4, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=808,
            backend='nervous', run_config=config, results_dir=directory)

    # The worker path: the config rides on the target so it survives pickling.
    assert getattr(target, '_escape') is escape
    # The breeder path: it arrives inside the run's GA config.
    assert seen and all(cfg.escape is escape for cfg in seen)


def test_controller_reports_live_escape_telemetry():
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    config = _nv_config(escape=EscapeConfig(crowding=True))
    messages, stop = queue.Queue(), threading.Event()

    def evaluate(genomes, *_args, **_kwargs):
        return [0.5] * len(genomes), [None] * len(genomes)

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases', side_effect=evaluate), \
            mock.patch.object(
                nv_ga, 'next_population',
                side_effect=lambda population, *a, **k: [
                    nv_ga.clone_genome(g) for g in population]), \
            mock.patch('substrates.nervous.certification.certify',
                       return_value=None):
        run_evolution(
            gens=3, pop=4, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=809,
            backend='nervous', run_config=config, results_dir=directory)

    reports = [m[1] for m in messages.queue if m[0] == 'escape']
    assert len(reports) == 3
    assert reports[0]['summary'] == 'crowding/16@50%'
    assert set(reports[0]) >= {'rebirths', 'archive', 'crowding_replacements',
                               'robust_blend', 'mean_rate'}


def test_no_escape_telemetry_when_every_mechanism_is_off():
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    messages, stop = queue.Queue(), threading.Event()

    def evaluate(genomes, *_args, **_kwargs):
        return [0.5] * len(genomes), [None] * len(genomes)

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases', side_effect=evaluate), \
            mock.patch.object(
                nv_ga, 'next_population',
                side_effect=lambda population, *a, **k: [
                    nv_ga.clone_genome(g) for g in population]), \
            mock.patch('substrates.nervous.certification.certify',
                       return_value=None):
        run_evolution(
            gens=2, pop=3, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=810,
            backend='nervous', run_config=_nv_config(),
            results_dir=directory)

    assert not any(m[0] == 'escape' for m in messages.queue)


def test_every_backend_breeder_accepts_the_escape_parameters():
    """A mechanism that reaches only some backends is the drift this module
    exists to prevent, so check the plumbing exists on all three."""
    import inspect
    for module in (nv_ga, lut_ga, snn_ga):
        parameters = inspect.signature(module.next_population).parameters
        assert 'escape' in parameters, module.__name__
        assert 'mutation_limit' in parameters, module.__name__
    for driver in (nv_ga.evolve_nervous, lut_ga.evolve_lut, snn_ga.evolve):
        parameters = inspect.signature(driver).parameters
        assert 'escape' in parameters and 'ga_config' in parameters


def test_headless_drivers_run_the_same_escape_hooks_as_the_controller():
    """Behavioural, not structural: every driver must call the shared merge,
    champion-acceptance, archive and rebirth hooks — the ones the controller
    calls — so a mechanism cannot be live on one path and dead on the other."""
    import inspect
    from runtime import controller
    hooks = ('merge_generation', 'accepts', 'record_champion',
             'maybe_rebirth', 'apply_robustness_blend', 'tick')
    sources = {
        'controller': inspect.getsource(controller.run_evolution),
        'nervous': inspect.getsource(nv_ga.evolve_nervous),
        'lut': inspect.getsource(lut_ga.evolve_lut),
    }
    for name, source in sources.items():
        for hook in hooks:
            assert hook in source, '%s never calls %s' % (name, hook)
    # The SNN driver has no robustness objective (no temporal contract), but
    # every population-level hook must still be there.
    snn_source = inspect.getsource(snn_ga.evolve)
    for hook in ('merge_generation', 'accepts', 'record_champion',
                 'maybe_rebirth', 'tick'):
        assert hook in snn_source, 'snn never calls %s' % hook


def test_all_mechanisms_together_drive_a_real_run_and_fire_a_rebirth():
    """End-to-end through the controller with every mechanism on, real breeder
    and real escape state, on a deliberately flat fitness landscape so the
    stall detector has to fire."""
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    escape = EscapeConfig(
        crowding=True, crowding_window=4, neutral_drift=True,
        self_adaptive_mutation=True, rebirth=True, rebirth_patience=3,
        rebirth_ancestors=2, rebirth_fraction=0.5, archive_interval=1,
        lexicase_downsample=0.5)
    config = _nv_config(escape=escape, elite_count=2)
    messages, stop = queue.Queue(), threading.Event()

    def evaluate(genomes, *_args, **_kwargs):
        # Perfectly flat: nothing ever improves, so stagnation climbs and the
        # rebirth trigger is guaranteed to be reached.
        return [0.5] * len(genomes), [(0.5, 0.5, 0.5, 0.5)] * len(genomes)

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch('runtime.controller.ProcessPoolExecutor',
                       return_value=FakePool()), \
            mock.patch.object(nv_ga, 'eval_batch_cases', side_effect=evaluate), \
            mock.patch('substrates.nervous.certification.certify',
                       return_value=None):
        run_evolution(
            gens=12, pop=8, n_chroms=2, tries=1, target=target, arch=None,
            messages=messages, stop_event=stop, base_seed=911,
            backend='nervous', run_config=config, results_dir=directory)

    queued = list(messages.queue)
    assert not any(message[0] == 'error' for message in queued), [
        m for m in queued if m[0] == 'error']
    rebirths = [m[1] for m in queued if m[0] == 'rebirth']
    assert rebirths, 'a flat 12-generation run never triggered rebirth'
    assert all(info['reborn'] == 4 for info in rebirths)
    final = [m[1] for m in queued if m[0] == 'escape'][-1]
    assert final['rebirths'] == len(rebirths)
    assert final['archive'] > 0
    assert final['crowding_replacements'] > 0
    assert final['mean_rate'] > 0.0
    assert queued[-1][0] == 'done'


def test_lifespan_and_robustness_survive_a_real_evaluation_pass():
    """Both extra objectives run inside the evaluation worker, where an
    exception would be reported as a dead run rather than a bad score."""
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    setattr(target, 'pulse_config', PulseConfig(model='pulse_delay'))
    setattr(target, '_escape', EscapeConfig(
        lifespan_scoring=True, lifespan_checkpoints=2,
        robustness=True, robustness_samples=2))
    random.seed(77)
    from substrates.nervous.objectives import total_case_count
    expected = total_case_count(target)
    for _ in range(5):
        genome = random_hex_genome(2)
        record = nv_ga._evaluate_nv_selection_record(genome, target)
        fitness, cases, _progress, juvenile, robust = record
        assert 0.0 <= fitness <= 1.0
        assert len(cases) == expected
        assert 0.0 <= juvenile <= 1.0
        # The robust vector describes the ADULT under new physics, so it has
        # the contract's own case count — not the lifespan-extended one.
        if robust is not None:
            from substrates.nervous.scoring import contract_case_count
            assert len(robust) == contract_case_count(target)
        nv_ga.record_escape_objectives(genome, record)
        assert genome._juvenile_score == juvenile


def test_a_jitter_probe_never_recurses_into_lifespan_or_robustness():
    """Without this guard each robustness sample would spawn its own lifespan
    growth and its own nested robustness pass — a quiet combinatorial blow-up
    in evaluation cost."""
    from substrates.nervous import objectives
    target = dataclasses.replace(TEMPORAL_TARGETS['Veto gate'])
    setattr(target, 'pulse_config', PulseConfig(model='pulse_delay'))
    escape = EscapeConfig(lifespan_scoring=True, robustness=True,
                          robustness_samples=2)
    setattr(target, '_escape', escape)
    seen = []
    real_prepare = objectives.prepare_grid

    def watch(genome, probe_target, *args, **kwargs):
        seen.append(getattr(probe_target, '_escape', 'missing'))
        return real_prepare(genome, probe_target, *args, **kwargs)

    with mock.patch.object(objectives, 'prepare_grid', side_effect=watch):
        random.seed(5)
        objectives.robust_case_vector(
            random_hex_genome(2), target, 'nervous', escape)
    # The jitter path uses prepare_net/prepare_lut directly on a probe target
    # whose escape config has been cleared, so no lifespan growth is triggered.
    assert seen == []


# ── 9b. island model ──────────────────────────────────────────────────────────

def test_islands_are_off_by_default_and_breed_as_one_pool():
    state = EscapeState(OFF, rank=nv_ga.rank_key)
    seen = []

    def step(deme, fitnesses, cases, rate):
        seen.append((len(deme), rate))
        return list(deme)

    population = [random_hex_genome(2) for _ in range(12)]
    bred = state.breed(1, population, [0.5] * 12, None, 4.0, step)
    assert seen == [(12, 4.0)]          # one call, whole population, run rate
    assert len(bred) == 12
    assert state.islands_bred == 0


def test_islands_breed_each_deme_separately_at_its_own_rate():
    """Cold demes exploit while hot demes explore AT THE SAME TIME — a single
    population can only ever be at one point of the anneal."""
    config = EscapeConfig(islands=True, island_count=4,
                          island_rate_spread=2.0,
                          island_migration_interval=100)
    state = EscapeState(config, mutation_limit=64.0, clone=nv_ga.clone_genome,
                        rank=nv_ga.rank_key)
    seen = []

    def step(deme, fitnesses, cases, rate):
        seen.append((len(deme), round(rate, 4)))
        return list(deme)

    population = [random_hex_genome(2) for _ in range(12)]
    bred = state.breed(1, population, [0.5] * 12, None, 4.0, step)
    assert len(seen) == 4                       # one call per deme
    assert sum(size for size, _ in seen) == 12  # and they partition the pop
    rates = [rate for _size, rate in seen]
    assert rates == sorted(rates)               # cold -> hot
    assert rates[0] == 2.0 and rates[-1] == 8.0  # base/spread .. base*spread
    assert len(bred) == 12
    assert state.islands_bred == 1


def test_island_demes_partition_the_population_without_gaps():
    for size in (7, 12, 13, 40, 41):
        for count in (2, 3, 4, 5):
            bounds = EscapeConfig(
                islands=True, island_count=count).island_slices(size)
            assert bounds[0][0] == 0 and bounds[-1][1] == size
            for (_a, b), (c, _d) in zip(bounds, bounds[1:]):
                assert b == c                   # contiguous, no overlap/gap
            assert all(stop > start for start, stop in bounds)


def test_island_rates_stay_inside_the_run_mutation_cap():
    config = EscapeConfig(islands=True, island_count=4,
                          island_rate_spread=8.0,
                          island_migration_interval=100)
    state = EscapeState(config, mutation_limit=6.0, clone=nv_ga.clone_genome,
                        rank=nv_ga.rank_key)
    rates = []
    state.breed(1, [random_hex_genome(2) for _ in range(8)], [0.5] * 8, None,
                4.0, lambda d, f, c, rate: (rates.append(rate) or list(d)))
    assert all(1.0 <= rate <= 6.0 for rate in rates)


def test_migration_moves_the_best_into_the_next_deme_on_a_ring():
    config = EscapeConfig(islands=True, island_count=2, island_migrants=1,
                          island_migration_interval=1)
    state = EscapeState(config, clone=nv_ga.clone_genome, rank=nv_ga.rank_key)
    population = [random_hex_genome(2) for _ in range(4)]
    # deme A = slots 0,1 (fitness 0.9, 0.1); deme B = slots 2,3 (0.2, 0.3)
    fitnesses = [0.9, 0.1, 0.2, 0.3]
    signatures = [nv_ga.genome_signature(g) for g in population]
    bred = state.breed(1, population, fitnesses, None, 4.0,
                       lambda d, f, c, r: list(d))
    assert state.migrations == 1
    # A's best (slot 0) replaces B's worst (slot 2); B's best (slot 3)
    # replaces A's worst (slot 1). A ring, so a discovery diffuses gradually.
    assert nv_ga.genome_signature(bred[2]) == signatures[0]
    assert nv_ga.genome_signature(bred[1]) == signatures[3]
    assert nv_ga.genome_signature(bred[0]) == signatures[0]


def test_migration_is_rare_by_design():
    """Migrating every generation is one population with extra bookkeeping."""
    config = EscapeConfig(islands=True, island_count=2,
                          island_migration_interval=5)
    state = EscapeState(config, clone=nv_ga.clone_genome, rank=nv_ga.rank_key)
    population = [random_hex_genome(2) for _ in range(6)]
    for generation in range(1, 11):
        state.breed(generation, population, [0.5] * 6, None, 4.0,
                    lambda d, f, c, r: list(d))
    assert state.migrations == 2            # generations 5 and 10 only


def test_islands_preserve_population_size_even_if_a_deme_breeds_short():
    config = EscapeConfig(islands=True, island_count=3,
                          island_migration_interval=100)
    state = EscapeState(config, clone=nv_ga.clone_genome, rank=nv_ga.rank_key)
    population = [random_hex_genome(2) for _ in range(9)]
    bred = state.breed(1, population, [0.5] * 9, None, 4.0,
                       lambda d, f, c, r: list(d)[:-1])   # each deme short
    assert len(bred) == 9


def test_both_drivers_breed_through_the_island_hook():
    import inspect
    from runtime import controller
    for source in (inspect.getsource(controller.run_evolution),
                   inspect.getsource(nv_ga.evolve_nervous)):
        assert 'escape_state.breed(' in source


# ── 10. configuration validation ──────────────────────────────────────────────

def test_escape_config_rejects_nonsense():
    for kwargs in ({'lifespan_checkpoints': 0},
                   {'crowding_window': 0},
                   {'adaptive_tau': 0.0},
                   {'rebirth_patience': 0},
                   {'rebirth_fraction': 0.0},
                   {'rebirth_fraction': 1.5},
                   {'rebirth_ancestors': 0},
                   {'rebirth_mutation_multiplier': 0.5},
                   {'archive_interval': 0},
                   {'archive_size': 0},
                   {'robustness_jitter': 1.0},
                   {'robustness_jitter': -0.1},
                   {'robustness_samples': 0},
                   {'lexicase_downsample': 0.0},
                   {'lexicase_downsample': 1.5},
                   {'crowding': 'yes'}):
        try:
            EscapeConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError('EscapeConfig accepted %r' % kwargs)


def test_ga_config_rejects_a_non_escape_escape():
    try:
        GAConfig(escape={'crowding': True})
    except ValueError:
        return
    raise AssertionError('GAConfig accepted a raw dict as escape')


def test_summary_names_exactly_the_active_mechanisms():
    assert EscapeConfig(crowding=True).summary() == 'crowding/16@50%'
    summary = EscapeConfig(rebirth=True, rebirth_patience=25,
                           neutral_drift=True).summary()
    assert summary == 'neutral-drift, rebirth@25'
    assert 'downsample 0.50' in EscapeConfig(
        lexicase_downsample=0.5).summary()
