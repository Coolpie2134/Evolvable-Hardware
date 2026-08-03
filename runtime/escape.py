"""
runtime/escape.py — mechanisms for getting a stalled run out of a local minimum.

Everything here is backend-neutral and lives in ONE module on purpose. The
project has two GA drive paths (``substrates/*/ga.py::evolve_*`` for headless
runs and ``runtime/controller.py`` for the desktop app), and they have drifted
before: a mechanism landed in one, benchmarks measured the other. Both drivers
now own an :class:`EscapeState` and call the same hooks, so a mechanism cannot
exist on one path and not the other.

Eight optional mechanisms are OFF by default. Case-aware contract elites and
complementary mating are baseline selection behavior rather than escape
switches: they preserve evidence already earned on declared cases without
changing the fitness contract or the 1.0 boundary.

  LIFESPAN SCORING
      Score the organism at several points along its DEVELOPMENT rather than
      only as a finished adult. A genome whose stage-6 body half-works but
      whose stage-12 body is broken currently scores zero and dies with no
      gradient to climb; its juvenile scores become extra ε-lexicase cases and
      a selection-only tie-break, so it now has somewhere to climb from. The
      REPORTED fitness is still the adult score — a run that reads 1.0 still
      means the grown circuit works.

  CROWDING (restricted tournament replacement)
      An offspring competes against the most genetically SIMILAR member of a
      random window of the population, not against a random one. Niches survive
      without needing an explicit species concept.

  NEUTRAL DRIFT
      Accept equal-ranked challengers instead of demanding strict improvement.
      Drifting across a plateau's neutral network is how this class of
      substrate finds new phenotypes; strict ``>`` freezes a lineage in place.

  SELF-ADAPTIVE MUTATION
      Each individual carries its own mutation rate, inherited with a
      log-normal nudge. A stuck lineage heats up on its own while a lineage
      that is still improving stays cool — per-lineage, unlike the global SOS
      reheat in runtime/mutation.py, which is a population-wide sledgehammer.

  REBIRTH
      On a stall, re-seed part of the population from a DIVERSE SET of archived
      ancestors (not the single best one — that walks the same path again) at
      an elevated mutation rate, deliberately backtracking to an earlier branch
      point to leave it in a different direction.

  LINEAGE WALK
      Reserve a small cohort for mutation-only, fitness-blind random walks.
      Each walker descends from its own previous state, so a temporarily worse
      intermediate survives long enough to mutate again. Useful discoveries
      are copied back into the ordinary breeding pool. This is the mechanism
      here that can cross a genuine fitness valley rather than only move along
      a plateau or jump across it in one mutation transaction.

  ISLANDS
      Breed independent cold-to-hot demes and occasionally copy each evaluated
      deme's best into its neighbour. Separate demes keep alternate basins from
      immediately collapsing back into the global champion's lineage.

  ROBUSTNESS
      A second objective scored under jittered physics, aggregated by WORST
      case rather than mean, and ranked strictly BELOW nominal fitness so it
      can never trade against correctness. Brittle solutions sit on narrow
      spikes; preferring the broader basin among equally-correct circuits both
      smooths the landscape and generalises better.

Plus one fix to selection that is not a mechanism of its own: ε-lexicase can
now stream a random SUBSET of cases per generation (downsampling), which is
also what "rotate the stimulus set" amounts to once the cases are resampled
every generation.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass


# Defaults for every tunable. Named so the GUI, the config validator and the
# tests all quote the same number.
DEFAULT_LIFESPAN_CHECKPOINTS = 3
DEFAULT_CROWDING_WINDOW = 16
# Half the population crowded, half generationally replaced. Measured: a full
# (1.0) crowded population produced ZERO mean-fitness decreases in 9 of 9 runs
# across three targets and three seeds — monotone by construction, and the
# exploratory churn the rest of the plateau machinery depends on was gone.
DEFAULT_CROWDING_FRACTION = 0.5
DEFAULT_ADAPTIVE_TAU = 0.25
# A 40-generation trigger gave a 50-generation run one late attempt with no
# time for the rebuilt cohort to reproduce. Rebirth is useful only when it can
# fire, develop, and (if necessary) fire again inside a normal short run.
DEFAULT_REBIRTH_PATIENCE = 15
DEFAULT_REBIRTH_FRACTION = 0.5
DEFAULT_REBIRTH_ANCESTORS = 4
DEFAULT_REBIRTH_MULTIPLIER = 3.0
DEFAULT_ARCHIVE_INTERVAL = 5
DEFAULT_ARCHIVE_SIZE = 24
# A lean walker reserve leaves most capacity under behavioral selection. The
# fraction is only a tuning default when the mechanism is explicitly enabled;
# it is not a recommended preset. The initial raw-score screen did not measure
# solved/certified outcomes and therefore cannot establish an escape winner.
DEFAULT_LINEAGE_WALK_FRACTION = 0.10
DEFAULT_ROBUSTNESS_JITTER = 0.15
DEFAULT_ROBUSTNESS_SAMPLES = 2
DEFAULT_ISLAND_COUNT = 4
DEFAULT_ISLAND_MIGRATION_INTERVAL = 20
DEFAULT_ISLAND_MIGRANTS = 1
DEFAULT_ISLAND_RATE_SPREAD = 2.0
# Environmental case memory, independent of the optional escape switches.
# Most of the population must remain generational so exploration still churns;
# two fifths can retain one expert for every row/output of a 16-case contract
# in the standard 40-member diagnostic population while leaving most slots to
# generational exploration.
CONTRACT_ELITE_FRACTION = 0.40

# Self-adaptive rates are clamped to this band. The floor is 1 because the
# mutation operators always perform at least one event (same reasoning as
# runtime/mutation.adaptive_mutation_rate); the ceiling comes from the run's
# own mutation cap, passed in at call time.
MIN_ADAPTIVE_RATE = 1.0


def contract_progress_key(case_vectors, fitnesses=None):
    """Best integrated all-case progress in an evaluated population.

    The headline fitness is intentionally a useful aggregate, but a flat
    aggregate does not imply that search is stalled: an individual may have
    improved its weakest declared case while trading a little score on an easy
    one.  Sort each behavior vector from weakest to strongest and compare it
    lexicographically (leximin), then use mean and reported fitness only as
    ties.  This is target-agnostic and rewards progress toward *one organism*
    satisfying the whole contract, not a population whose specialists merely
    cover it collectively.

    ``None`` means that no valid case evidence was supplied.
    """
    if not case_vectors:
        return None
    vectors = []
    width = None
    for index, vector in enumerate(case_vectors):
        if vector is None:
            return None
        values = tuple(float(value) for value in vector)
        if not values:
            return None
        if width is None:
            width = len(values)
        elif len(values) != width:
            return None
        scalar = (
            float(fitnesses[index])
            if fitnesses is not None and index < len(fitnesses) else 0.0)
        vectors.append((
            tuple(sorted(values)), sum(values) / len(values), scalar))
    return max(vectors) if vectors else None


def contract_elite_survivors(
        parents, parent_fitnesses, parent_cases,
        offspring, offspring_fitnesses, offspring_cases,
        *, fraction=CONTRACT_ELITE_FRACTION, case_offset=0):
    """Keep a small rotating reserve of distinct best-on-case behaviors.

    A scalar champion is not a substitute for a population on a multi-case
    contract. Before a perfect solution exists, strict generational
    replacement can delete the only genome that passes a missing row/trial;
    next generation's lexicase selection cannot choose a specialist that no
    longer exists. This merge preserves a bounded number of evaluated
    specialists from ``parents + offspring`` and fills every other slot from
    the new offspring in their original order.

    Cases are ordered hardest-first by the best score currently available.
    Equal-hardness cases rotate via ``case_offset`` so a large contract does
    not permanently privilege its first rows. Candidates tie on the selected
    case by leximin quality across the whole vector, then mean, scalar fitness,
    and finally recency (offspring). Identical behavior vectors consume only
    one reserve slot. No target name, topology, or desired mechanism enters.
    """
    pop = len(offspring)
    if (pop < 2 or parent_cases is None or offspring_cases is None
            or len(parents) != len(parent_fitnesses)
            or len(offspring) != len(offspring_fitnesses)
            or len(parent_cases) != len(parents)
            or len(offspring_cases) != len(offspring)):
        return offspring, offspring_fitnesses, offspring_cases
    all_cases = list(parent_cases) + list(offspring_cases)
    if not all_cases or any(vector is None for vector in all_cases):
        return offspring, offspring_fitnesses, offspring_cases
    n_cases = len(all_cases[0])
    if n_cases < 1 or any(len(vector) != n_cases for vector in all_cases):
        return offspring, offspring_fitnesses, offspring_cases

    parents = list(parents)
    offspring = list(offspring)
    genomes = parents + offspring
    fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
    vectors = [tuple(float(value) for value in vector)
               for vector in all_cases]
    reserve = min(
        n_cases, pop - 1,
        max(1, int(round(pop * float(fraction)))))
    offset = int(case_offset) % n_cases
    best_by_case = [max(vector[case] for vector in vectors)
                    for case in range(n_cases)]
    case_order = sorted(
        range(n_cases),
        key=lambda case: (
            best_by_case[case], (case - offset) % n_cases))

    selected = []
    selected_behaviors = set()
    for case in case_order:
        eligible = [
            index for index, vector in enumerate(vectors)
            if vector not in selected_behaviors]
        if not eligible:
            break

        def candidate_key(index):
            vector = vectors[index]
            ordered = tuple(sorted(vector))
            mean = sum(vector) / n_cases
            return (
                vector[case], ordered, mean, float(fitnesses[index]),
                index >= len(parents))

        winner = max(eligible, key=candidate_key)
        selected.append(winner)
        selected_behaviors.add(vectors[winner])
        if len(selected) >= reserve:
            break

    # Preserve the chosen specialists, then use the genuinely new generation
    # for all remaining capacity. An offspring already selected is not copied
    # twice; parent specialists survive only when no equivalent offspring does.
    kept_genomes = [genomes[index] for index in selected]
    kept_fitnesses = [fitnesses[index] for index in selected]
    kept_cases = [all_cases[index] for index in selected]
    selected_offspring = {
        index - len(parents) for index in selected
        if index >= len(parents)}
    for index, genome in enumerate(offspring):
        if len(kept_genomes) >= pop:
            break
        if index in selected_offspring:
            continue
        kept_genomes.append(genome)
        kept_fitnesses.append(offspring_fitnesses[index])
        kept_cases.append(offspring_cases[index])
    return kept_genomes, kept_fitnesses, kept_cases


def complementary_parent_index(
        first, candidates, case_vectors, fitnesses=None, case_subset=None):
    """Choose a mate whose contract behavior best fills the first parent's gaps.

    Independent lexicase draws often return two specialists for the same easy
    cases. Crossover can only assemble a complete solution when its parents
    carry complementary partial behaviors. Rank each possible pair by the
    leximin profile of their per-case envelope ``max(left, right)``; this first
    improves the pair's weakest jointly covered case, then its next weakest,
    and so on. The mate's own leximin profile and scalar fitness break exact
    envelope ties. No assumption is made about what a case means.
    """
    candidates = list(candidates)
    if not candidates:
        return first
    if case_vectors is None or first >= len(case_vectors):
        return random.choice(candidates)
    left = case_vectors[first]
    if left is None:
        return random.choice(candidates)
    indices = (list(case_subset) if case_subset is not None
               else list(range(len(left))))
    if not indices:
        return random.choice(candidates)
    viable = [
        index for index in candidates
        if index < len(case_vectors)
        and case_vectors[index] is not None
        and len(case_vectors[index]) == len(left)]
    if not viable:
        return random.choice(candidates)

    def key(index):
        right = case_vectors[index]
        envelope = tuple(sorted(
            max(float(left[case]), float(right[case]))
            for case in indices))
        own = tuple(sorted(float(right[case]) for case in indices))
        scalar = (
            float(fitnesses[index])
            if fitnesses is not None and index < len(fitnesses) else 0.0)
        return envelope, own, scalar

    best_key = max(key(index) for index in viable)
    best = [index for index in viable if key(index) == best_key]
    return random.choice(best)


@dataclass(frozen=True)
class EscapeConfig:
    """Immutable, process-safe escape configuration carried with one run.

    Defaults disable every mechanism. ``dataclasses.asdict`` serialises this
    inside :class:`runtime.config.GAConfig`, and unknown/absent keys fall back
    to the defaults, so a checkpoint written before this module existed loads
    as "all mechanisms off" — which is exactly what it was run under.
    """

    # ── lifespan (ontogeny checkpoint) scoring ──
    lifespan_scoring: bool = False
    #: developmental snapshots scored in ADDITION to the adult body.
    lifespan_checkpoints: int = DEFAULT_LIFESPAN_CHECKPOINTS

    # ── restricted tournament replacement ──
    crowding: bool = False
    #: population members sampled to find the offspring's nearest incumbent.
    crowding_window: int = DEFAULT_CROWDING_WINDOW
    #: share of the next population held as a crowded RESERVE. RTR is monotone
    #: by construction — an incumbent is only ever replaced by something that
    #: ranks at least as well — so at 1.0 the whole population can never move
    #: downhill and the mean rises without fluctuation. That is niche
    #: PRESERVATION, and it is the opposite of what crossing a valley needs.
    #: Below 1.0 the remaining slots keep this project's pre-solve generational
    #: churn, subject to the bounded contract-elite reserve.
    crowding_fraction: float = DEFAULT_CROWDING_FRACTION

    # ── neutral drift ──
    neutral_drift: bool = False

    # ── self-adaptive per-individual mutation rate ──
    self_adaptive_mutation: bool = False
    #: log-normal learning rate for the inherited mutation rate.
    adaptive_tau: float = DEFAULT_ADAPTIVE_TAU

    # ── rebirth ──
    rebirth: bool = False
    #: flat generations before a rebirth fires.
    rebirth_patience: int = DEFAULT_REBIRTH_PATIENCE
    #: share of the population rebuilt from ancestors.
    rebirth_fraction: float = DEFAULT_REBIRTH_FRACTION
    #: distinct ancestors drawn from the archive (never just the best one).
    rebirth_ancestors: int = DEFAULT_REBIRTH_ANCESTORS
    #: mutation-rate multiplier for the reborn cohort.
    rebirth_mutation_multiplier: float = DEFAULT_REBIRTH_MULTIPLIER
    #: generations between champion archive snapshots.
    archive_interval: int = DEFAULT_ARCHIVE_INTERVAL
    #: archive ring-buffer length.
    archive_size: int = DEFAULT_ARCHIVE_SIZE

    # ── fitness-blind stepping-stone lineages ──
    lineage_walk: bool = False
    #: share of population slots that take one mutation-only step per
    #: generation without behavioral selection. Their score is still measured
    #: normally and a useful walker is admitted to the ordinary breeding pool.
    lineage_walk_fraction: float = DEFAULT_LINEAGE_WALK_FRACTION

    # ── robustness second objective ──
    robustness: bool = False
    #: fractional physics perturbation (0.15 = ±15% on delay/width).
    robustness_jitter: float = DEFAULT_ROBUSTNESS_JITTER
    #: jittered physics variants scored per genome.
    robustness_samples: int = DEFAULT_ROBUSTNESS_SAMPLES

    # ── island model ──
    #: split the population into demes that breed separately. They share ONE
    #: objective and differ only in SEARCH DYNAMICS (mutation rate, selection
    #: pressure). Islands that differ by objective — one per test case — were
    #: tried in this project and failed: they specialise into mutually
    #: incompatible optima and every migrant is a hybrid that fails on both
    #: sides. Same landscape, different walkers, is the version that works.
    islands: bool = False
    #: number of demes.
    island_count: int = DEFAULT_ISLAND_COUNT
    #: generations between migrations. Rare migration is the point — frequent
    #: migration is just one population with extra bookkeeping.
    island_migration_interval: int = DEFAULT_ISLAND_MIGRATION_INTERVAL
    #: emigrants sent per island per migration.
    island_migrants: int = DEFAULT_ISLAND_MIGRANTS
    #: widest mutation-rate multiplier across the demes. Island i runs at a
    #: rate geometrically spaced in [1/spread, spread] around the run rate, so
    #: cold islands exploit while hot ones explore, at the same time.
    island_rate_spread: float = DEFAULT_ISLAND_RATE_SPREAD

    # ── ε-lexicase downsampling ──
    #: fraction of cases streamed per generation. 1.0 = every case (default).
    lexicase_downsample: float = 1.0

    def __post_init__(self):
        if self.lifespan_checkpoints < 1:
            raise ValueError('lifespan_checkpoints must be at least 1')
        if self.crowding_window < 1:
            raise ValueError('crowding_window must be at least 1')
        if not 0 < self.crowding_fraction <= 1:
            raise ValueError('crowding_fraction must be in (0, 1]')
        if not 0 < self.adaptive_tau <= 2:
            raise ValueError('adaptive_tau must be in (0, 2]')
        if self.rebirth_patience < 1:
            raise ValueError('rebirth_patience must be at least 1')
        if not 0 < self.rebirth_fraction <= 1:
            raise ValueError('rebirth_fraction must be in (0, 1]')
        if self.rebirth_ancestors < 1:
            raise ValueError('rebirth_ancestors must be at least 1')
        if self.rebirth_mutation_multiplier < 1:
            raise ValueError('rebirth_mutation_multiplier must be at least 1')
        if self.archive_interval < 1:
            raise ValueError('archive_interval must be at least 1')
        if self.archive_size < 1:
            raise ValueError('archive_size must be at least 1')
        if not 0 < self.lineage_walk_fraction < 1:
            raise ValueError('lineage_walk_fraction must be in (0, 1)')
        if not 0 <= self.robustness_jitter < 1:
            raise ValueError('robustness_jitter must be in [0, 1)')
        if self.robustness_samples < 1:
            raise ValueError('robustness_samples must be at least 1')
        if not 0 < self.lexicase_downsample <= 1:
            raise ValueError('lexicase_downsample must be in (0, 1]')
        if self.island_count < 2:
            raise ValueError('island_count must be at least 2')
        if self.island_migration_interval < 1:
            raise ValueError('island_migration_interval must be at least 1')
        if self.island_migrants < 1:
            raise ValueError('island_migrants must be at least 1')
        if self.island_rate_spread < 1:
            raise ValueError('island_rate_spread must be at least 1')
        for name in ('lifespan_scoring', 'crowding', 'neutral_drift',
                     'self_adaptive_mutation', 'rebirth', 'lineage_walk',
                     'robustness', 'islands'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError('%s must be boolean' % name)

    def island_slices(self, population_size):
        """Contiguous ``(start, stop)`` deme bounds over the population.

        Contiguous rather than interleaved so a deme is a stable set of slots:
        every other mechanism here indexes the population directly, and a deme
        that moved every generation would scramble them.
        """
        count = max(2, min(int(self.island_count), max(2, population_size)))
        edges = [round(i * population_size / count) for i in range(count + 1)]
        return [(edges[i], edges[i + 1]) for i in range(count)
                if edges[i + 1] > edges[i]]

    def island_rate(self, index, count, base_rate):
        """Mutation rate for deme ``index``, geometrically spaced.

        Island 0 is the coldest (base / spread) and the last the hottest
        (base * spread). Running exploit and explore SIDE BY SIDE is the whole
        point: a single population has to anneal between the two over time and
        can only ever be at one setting at once.
        """
        if count < 2:
            return float(base_rate)
        spread = float(self.island_rate_spread)
        step = (index / float(count - 1)) * 2.0 - 1.0      # -1 .. +1
        return float(base_rate) * (spread ** step)

    @property
    def any_enabled(self):
        """True when this run departs from the pre-escape-module behaviour."""
        return bool(
            self.lifespan_scoring or self.crowding or self.neutral_drift
            or self.self_adaptive_mutation or self.rebirth
            or self.lineage_walk or self.robustness or self.islands
            or self.lexicase_downsample < 1.0)

    def summary(self):
        """One short line naming the active mechanisms (for the GUI/status)."""
        active = []
        if self.lifespan_scoring:
            active.append('lifespan×%d' % self.lifespan_checkpoints)
        if self.crowding:
            active.append('crowding/%d@%.0f%%'
                          % (self.crowding_window,
                             self.crowding_fraction * 100))
        if self.neutral_drift:
            active.append('neutral-drift')
        if self.self_adaptive_mutation:
            active.append('self-adaptive-mut')
        if self.rebirth:
            active.append('rebirth@%d' % self.rebirth_patience)
        if self.lineage_walk:
            active.append('lineage-walk@%.0f%%'
                          % (self.lineage_walk_fraction * 100))
        if self.robustness:
            active.append('robustness±%d%%'
                          % round(self.robustness_jitter * 100))
        if self.islands:
            active.append('islands×%d/%dgen'
                          % (self.island_count,
                             self.island_migration_interval))
        if self.lexicase_downsample < 1.0:
            active.append('downsample %.2f' % self.lexicase_downsample)
        return ', '.join(active) if active else 'none'

    @classmethod
    def from_dict(cls, values):
        """Tolerant load: unknown keys are dropped, missing keys default."""
        values = dict(values or {})
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


#: The all-off configuration. Identity-comparable, so callers can cheaply skip
#: every hook when a run does not use any of this.
OFF = EscapeConfig()


# ── genome distance (backend-neutral) ─────────────────────────────────────────

def _gene_values(gene):
    """Every integer allele of one gene, in a stable order.

    Walks the dataclass fields so this works for HexGene (nervous), the LUT
    gene and the SNN gene without this module importing any substrate.
    """
    try:
        fields = dataclasses.fields(gene)
    except TypeError:
        return ()
    out = []
    for f in fields:
        value = getattr(gene, f.name, 0)
        if isinstance(value, bool):
            out.append(int(value))
        elif isinstance(value, (int, float)):
            out.append(float(value))
    return tuple(out)


def genome_descriptor(genome):
    """A flat numeric fingerprint of a genome's heritable content.

    Only used for CROWDING DISTANCE and for picking diverse rebirth ancestors —
    never for fitness, never for the evaluation cache (which has its own exact
    per-backend signature). Approximate is fine here; cheap and backend-neutral
    is what matters.
    """
    out = []
    for chromosome in getattr(genome, 'chromosomes', ()) or ():
        out.append(float(getattr(chromosome, 'tag', 0)))
        out.append(float(getattr(chromosome, 'split', 0)))
        out.append(float(getattr(chromosome, 'telomere', 0)))
        for gene in chromosome.genes:
            out.extend(_gene_values(gene))
    delays = getattr(genome, 'state_delays', None)
    if delays:
        out.extend(float(d) for d in delays)
    for patch in (getattr(genome, 'routing_patches', None) or ()):
        out.extend((float(patch.x), float(patch.y), float(patch.state)))
    layout = getattr(genome, 'input_layout', None)
    if layout is not None:
        for x, y in layout:
            out.extend((float(x), float(y)))
    return tuple(out)


def genome_distance(a, b):
    """Normalised distance in [0, 1] between two genome descriptors.

    Positional mismatches over the shared prefix plus the length difference,
    divided by the longer length. Genomes here are variable-length, so length
    difference genuinely is distance — a 6-gene genome is not "the same as" a
    24-gene one that happens to agree on its first six.
    """
    if a is b:
        return 0.0
    n, m = len(a), len(b)
    if not n and not m:
        return 0.0
    shared = min(n, m)
    mismatches = sum(1 for i in range(shared) if a[i] != b[i])
    return (mismatches + abs(n - m)) / float(max(n, m))


# ── self-adaptive mutation rate ───────────────────────────────────────────────

def mutation_rate_of(genome, default):
    """This individual's own mutation rate, or the run's rate if it has none."""
    rate = getattr(genome, '_mut_rate', None)
    return float(default) if rate is None else float(rate)


def set_mutation_rate(genome, rate, limit):
    try:
        genome._mut_rate = min(float(limit), max(MIN_ADAPTIVE_RATE, float(rate)))
    except AttributeError:      # pragma: no cover - genomes are plain objects
        pass
    return genome


def inherit_mutation_rate(child, parent_a, parent_b, config, base, limit):
    """Give ``child`` a mutation rate descended from its parents.

    The classic log-normal self-adaptation rule: take the parental rate and
    multiply by exp(tau * N(0,1)). Selection then acts on the rate indirectly —
    lineages whose rate is producing useful offspring keep it, and a stuck
    lineage random-walks upward because only its higher-rate descendants ever
    find anything.
    """
    rate_a = mutation_rate_of(parent_a, base)
    rate_b = mutation_rate_of(parent_b, base)
    parental = math.sqrt(max(1e-9, rate_a * rate_b))     # geometric mean
    nudged = parental * math.exp(config.adaptive_tau * random.gauss(0.0, 1.0))
    return set_mutation_rate(child, nudged, limit)


def seed_mutation_rate(genome, base, limit):
    """Give a fresh immigrant a randomised starting rate around ``base``."""
    return set_mutation_rate(
        genome, float(base) * math.exp(random.gauss(0.0, 0.5)), limit)


# ── ε-lexicase downsampling ───────────────────────────────────────────────────

def lexicase_case_subset(n_cases, config):
    """Case indices ε-lexicase should stream THIS generation, or None for all.

    Downsampling is not a shortcut: at equal evaluation budget it buys several
    times more generations for the same selection quality, and resampling the
    subset every generation is what "rotate the stimulus set" reduces to once
    the cases already exist.
    """
    if config.lexicase_downsample >= 1.0 or n_cases <= 1:
        return None
    keep = max(1, int(round(n_cases * config.lexicase_downsample)))
    if keep >= n_cases:
        return None
    return tuple(sorted(random.sample(range(n_cases), keep)))


# ── robustness ────────────────────────────────────────────────────────────────

def jitter_physics(config, escape):
    """Deterministic jittered variants of a run's physics config.

    Determinism matters more than it looks: the fitness cache is keyed on the
    GENOME alone, so a genome must score the same every time it is evaluated.
    Random jitter would poison the cache with whichever draw happened first.
    The variants below are therefore a fixed, alternating ± ladder.
    """
    if config is None or not escape.robustness:
        return ()
    fields = {f.name for f in dataclasses.fields(type(config))}
    variants = []
    for i in range(escape.robustness_samples):
        # Alternate the sign and taper the magnitude so successive samples
        # probe distinct corners rather than re-testing one perturbation.
        sign = 1.0 if i % 2 == 0 else -1.0
        scale = escape.robustness_jitter * (1.0 - 0.35 * (i // 2))
        factor = 1.0 + sign * scale
        changes = {}
        # Delay is the one parameter every substrate's physics has.
        if 'delay' in fields:
            changes['delay'] = max(1e-6, config.delay * factor)
        # Widths move the OPPOSITE way from delay: a slow, narrow net and a
        # fast, wide net are the two ends of the timing margin that an
        # asynchronous circuit actually has to survive.
        if 'width' in fields:
            changes['width'] = max(1e-6, config.width / factor)
        if 'analog_tau_leak' in fields:
            changes['analog_tau_leak'] = max(
                1e-6, config.analog_tau_leak * factor)
        if not changes:
            continue
        try:
            variants.append(dataclasses.replace(config, **changes))
        except (ValueError, TypeError):
            # A jitter that violates the config's own coupling constraints
            # (e.g. the analog step/threshold band) is skipped rather than
            # crashing the run — a smaller margin is still a valid probe.
            continue
    return tuple(variants)


def robustness_blend(best_fitness):
    """How far the case aggregator has annealed from mean toward worst-case.

    Worst-case (min) aggregation is what stops "perfect on three cases, dead on
    the fourth" from ranking well — but it is brutally flat early, when every
    genome fails something and every min is zero. So start at the mean, and
    slide to the min as the run approaches a solution, where coverage is
    exactly the thing that still needs enforcing.
    """
    return max(0.0, min(1.0, float(best_fitness)))


def aggregate_robustness(case_scores, blend):
    """Collapse a robust case vector to a scalar under the annealed blend."""
    values = [float(v) for v in (case_scores or ()) if v is not None]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (1.0 - blend) * mean + blend * min(values)


# ── the per-run state both drivers own ────────────────────────────────────────

class EscapeState:
    """Mutable per-run state for the escape mechanisms.

    One instance per run, owned by whichever loop is driving. Both
    ``substrates.nervous.ga.evolve_nervous`` and
    ``runtime.controller.run_evolution`` construct one and call the same hooks,
    which is what keeps the two drive paths from drifting apart again.
    """

    def __init__(self, config=None, mutation_limit=8.0, clone=None,
                 mutate=None, rank=None):
        self.config = config or OFF
        self.mutation_limit = float(mutation_limit)
        self._clone = clone or (lambda genome: genome)
        self._mutate = mutate
        self._rank = rank or (lambda genome, fitness: fitness)
        #: ring buffer of (generation, genome, fitness, descriptor)
        self.archive = []
        self.rebirths = 0
        self.last_rebirth_gen = None
        self.crowding_replacements = 0
        self.robust_blend = 0.0
        self.migrations = 0
        self.islands_bred = 0
        self.lineage_walk_steps = 0
        self._cooldown = 0
        self._lineage_started = False
        self._lineage_start = None
        self._island_bounds = ()
        self._pending_migration = False
        self._contract_case_offset = 0
        self._contract_progress = None
        self.contract_elite_carries = 0
        self.contract_progress_events = 0

    def _clone_evaluated(self, genome):
        """Clone a survivor while retaining its already-measured rank data.

        Backend clone functions deliberately copy only heritable structure:
        normal clones are offspring and will immediately be evaluated. A
        post-evaluation migrant or main-pool admission is different; dropping
        these transient values would pair the right behavioral fitness with
        zeroed robustness/juvenile/topology tiers until its next evaluation.
        """
        clone = self._clone(genome)
        for name in ('_io_binding_progress', '_robust_cases', '_robustness',
                     '_juvenile_score', '_topology_score', '_mut_rate'):
            if hasattr(genome, name):
                setattr(clone, name, getattr(genome, name))
        return clone

    # ── champion / archive ──

    def accepts(self, new_rank, old_rank):
        """Should ``new_rank`` replace the incumbent?

        Strict improvement normally; equal-ranked challengers are also accepted
        under neutral drift, so the archived champion keeps moving along the
        plateau's neutral network instead of pinning to the first genome that
        reached the plateau. That matters because the champion is the seed for
        plateau rescue and for rebirth.
        """
        if old_rank is None:
            return True
        if self.config.neutral_drift:
            return new_rank >= old_rank
        return new_rank > old_rank

    def record_champion(self, generation, genome, fitness):
        """Snapshot the champion into the ancestor archive, periodically."""
        if not self.config.rebirth or genome is None:
            return
        if generation % self.config.archive_interval:
            return
        descriptor = genome_descriptor(genome)
        if self.archive and self.archive[-1][3] == descriptor:
            return                      # unchanged champion: nothing new to keep
        self.archive.append(
            (int(generation), self._clone(genome), float(fitness), descriptor))
        if len(self.archive) > self.config.archive_size:
            del self.archive[0]

    def diverse_ancestors(self, count):
        """Pick ``count`` maximally DIFFERENT archived ancestors.

        Greedy farthest-point selection seeded from the oldest entry. Taking
        the single best ancestor instead is what makes a naive restart walk the
        same path again; spreading the seeds over the archive is the whole
        point of backtracking to a branch point.
        """
        if not self.archive:
            return []
        pool = list(self.archive)
        chosen = [pool.pop(0)]
        while pool and len(chosen) < count:
            far = max(pool, key=lambda entry: min(
                genome_distance(entry[3], picked[3]) for picked in chosen))
            pool.remove(far)
            chosen.append(far)
        return chosen

    # ── rebirth ──

    def should_rebirth(self, stagnation, best_fitness):
        if not self.config.rebirth or self._mutate is None:
            return False
        if best_fitness >= 1.0 or self._cooldown > 0:
            return False
        return stagnation >= self.config.rebirth_patience and bool(self.archive)

    def rebirth_population(self, generation, population, fitnesses, cases,
                           base_rate):
        """Rebuild part of the population from diverse ancestors.

        Returns ``(population, fitnesses, cases, info)`` where the rebuilt
        slots carry ``None`` fitness — the caller must re-evaluate them. The
        current elites are kept: rebirth is a backtrack, not an extinction.
        """
        cfg = self.config
        pop = len(population)
        ancestors = self.diverse_ancestors(cfg.rebirth_ancestors)
        if not ancestors:
            return population, fitnesses, cases, None
        n_reborn = max(1, min(pop - 1, int(round(pop * cfg.rebirth_fraction))))
        order = sorted(range(pop),
                       key=lambda i: self._rank(population[i], fitnesses[i]),
                       reverse=True)
        keep = order[:pop - n_reborn]
        rate = min(self.mutation_limit,
                   base_rate * cfg.rebirth_mutation_multiplier)
        new_population = [population[i] for i in keep]
        new_fitnesses = [fitnesses[i] for i in keep]
        new_cases = ([cases[i] for i in keep]
                     if cases is not None else None)
        for index in range(n_reborn):
            _gen, ancestor, _fit, _desc = ancestors[index % len(ancestors)]
            child = self._mutate(self._clone(ancestor), rate)
            # The reborn cohort explores at an elevated rate; without this it
            # re-converges on the same attractor within a few dozen
            # generations and the rebirth was wasted.
            if cfg.self_adaptive_mutation:
                set_mutation_rate(child, rate, self.mutation_limit)
            new_population.append(child)
            new_fitnesses.append(None)
            if new_cases is not None:
                new_cases.append(None)
        self.rebirths += 1
        self.last_rebirth_gen = int(generation)
        self._cooldown = cfg.rebirth_patience
        info = {
            'generation': int(generation),
            'reborn': n_reborn,
            'ancestors': [int(entry[0]) for entry in ancestors],
            'rate': rate,
        }
        return new_population, new_fitnesses, new_cases, info

    def maybe_rebirth(self, generation, population, fitnesses, cases,
                      base_rate, stagnation, best_fitness, evaluate):
        """Rebirth if the run has stalled, re-evaluating the reborn slots.

        ``evaluate(genomes) -> (fitnesses, cases)`` is the driver's own batch
        evaluator. Sequencing the re-evaluation HERE rather than in each loop
        is deliberate: it is the step most likely to be implemented slightly
        differently on the two drive paths, and a difference there would show
        up as an unexplained divergence between the app and the benchmarks.

        Returns ``(population, fitnesses, cases, info | None)``.
        """
        if not self.should_rebirth(stagnation, best_fitness):
            return population, fitnesses, cases, None
        population, fitnesses, cases, info = self.rebirth_population(
            generation, population, fitnesses, cases, base_rate)
        if info is None:
            return population, fitnesses, cases, None
        reborn = [i for i, fitness in enumerate(fitnesses) if fitness is None]
        if reborn:
            fresh_fitnesses, fresh_cases = evaluate(
                [population[i] for i in reborn])
            for slot, i in enumerate(reborn):
                fitnesses[i] = fresh_fitnesses[slot]
                if cases is not None and fresh_cases is not None:
                    cases[i] = fresh_cases[slot]
        return population, fitnesses, cases, info

    def merge_generation(self, parents, parent_fitnesses, parent_cases,
                         offspring, offspring_fitnesses, offspring_cases,
                         consolidate=None, solved=False):
        """Choose the next live population. Both drive paths call exactly this.

        Precedence, highest first:

          solved      terminal (mu + lambda) consolidation — unchanged: once a
                      perfect circuit exists, evaluated parents and offspring
                      compete so the population mean can converge to 1.
          crowding    restricted tournament replacement.
          otherwise   generational offspring plus a bounded, rotating reserve
                      of distinct best-on-case behaviors.
        """
        if solved and consolidate is not None:
            result = consolidate(
                parents, parent_fitnesses, parent_cases,
                offspring, offspring_fitnesses, offspring_cases)
            # Once solved, terminal consolidation deliberately mixes the whole
            # population. There is no remaining local minimum to escape and an
            # island migration would only duplicate already-selected entries.
            self._pending_migration = False
            return result

        if self.config.crowding and self._island_bounds:
            # Crowding used to merge all demes back into one pool immediately,
            # nullifying the island model whenever both boxes were checked.
            # Select survivors inside each deme; a lineage-walk tail (which is
            # outside the bounds) remains generational and fitness-blind.
            pop, fits = [], []
            cases = [] if (parent_cases is not None) else None
            for start, stop in self._island_bounds:
                selected = self.survivor_selection(
                    parents[start:stop], parent_fitnesses[start:stop],
                    (parent_cases[start:stop]
                     if parent_cases is not None else None),
                    offspring[start:stop], offspring_fitnesses[start:stop],
                    (offspring_cases[start:stop]
                     if offspring_cases is not None else None))
                deme, deme_fits, deme_cases = selected
                pop.extend(deme)
                fits.extend(deme_fits)
                if cases is not None:
                    cases.extend(deme_cases)
            end = self._island_bounds[-1][1]
            pop.extend(offspring[end:])
            fits.extend(offspring_fitnesses[end:])
            if cases is not None:
                cases.extend(offspring_cases[end:])
            result = pop, fits, cases
        elif self.config.crowding and self._lineage_start is not None:
            # Without islands there is still one selected main pool followed
            # by the protected walker cohort. Crowding only the main pool keeps
            # RTR from randomly deleting or reordering those persistent walks.
            end = self._lineage_start
            pop, fits, cases = self.survivor_selection(
                parents[:end], parent_fitnesses[:end],
                (parent_cases[:end] if parent_cases is not None else None),
                offspring[:end], offspring_fitnesses[:end],
                (offspring_cases[:end]
                 if offspring_cases is not None else None))
            pop.extend(offspring[end:])
            fits.extend(offspring_fitnesses[end:])
            if cases is not None:
                cases.extend(offspring_cases[end:])
            result = pop, fits, cases
        else:
            crowded = self.survivor_selection(
                parents, parent_fitnesses, parent_cases,
                offspring, offspring_fitnesses, offspring_cases)
            result = (crowded if crowded is not None else
                      (offspring, offspring_fitnesses, offspring_cases))

        # Keep the best known behaviors on a rotating subset of the hardest
        # declared cases. This is environmental memory, not an escape toggle:
        # parent selection cannot recombine a missing-case specialist after
        # strict generational replacement has deleted it. Island demes and the
        # lineage tail depend on stable slot boundaries, so their own survival
        # rules remain authoritative when explicitly enabled.
        if (parent_cases is not None and result[2] is not None
                and not self._island_bounds
                and self._lineage_start is None):
            before = {id(genome) for genome in result[0]}
            result = contract_elite_survivors(
                parents, parent_fitnesses, parent_cases,
                result[0], result[1], result[2],
                case_offset=self._contract_case_offset)
            self.contract_elite_carries += sum(
                1 for genome in result[0]
                if id(genome) not in before)
            if result[2] and result[2][0]:
                self._contract_case_offset = (
                    self._contract_case_offset
                    + max(1, int(round(
                        len(result[0]) * CONTRACT_ELITE_FRACTION)))) \
                    % len(result[2][0])

        if self._pending_migration and self._island_bounds:
            result = self._migrate(*result, self._island_bounds)
        self._pending_migration = False
        return result

    def note_contract_progress(self, case_vectors, fitnesses=None):
        """Record case-level high-water progress; return True on improvement.

        Drivers use this alongside scalar progress to decide whether the
        stress clock should advance.  The first observation establishes the
        baseline and is not itself an improvement event.
        """
        key = contract_progress_key(case_vectors, fitnesses)
        if key is None:
            return False
        if self._contract_progress is None:
            self._contract_progress = key
            return False
        if key > self._contract_progress:
            self._contract_progress = key
            self.contract_progress_events += 1
            return True
        return False

    # ── islands ──

    def breed(self, generation, population, fitnesses, cases, rate, step):
        """Breed one generation, as one pool or as separate demes.

        ``step(parents, parent_fitnesses, parent_cases, rate) -> offspring`` is
        the driver's ordinary breeding call. With islands off this is exactly
        that call and nothing changes.

        With islands on, the population is cut into contiguous demes that breed
        SEPARATELY, each at its own mutation rate. They share one objective and
        differ only in search dynamics — per-case islands were tried here and
        failed, because demes with different objectives specialise into
        incompatible optima and every migrant is a hybrid that fails on both
        sides. Same landscape, different walkers.

        Migration is rare by design: frequent migration is one population with
        extra bookkeeping. Every ``island_migration_interval`` generations each
        deme sends copies of its best ``island_migrants`` to the next deme,
        replacing that deme's worst — a ring topology, so a discovery diffuses
        gradually instead of sweeping every deme at once.
        """
        self._pending_migration = False
        self._lineage_start = None
        if (self.config.lineage_walk and self._mutate is not None
                and len(population) > 1):
            return self._breed_with_lineage_walk(
                generation, population, fitnesses, cases, rate, step)
        return self._breed_selected_pool(
            generation, population, fitnesses, cases, rate, step)

    def _breed_selected_pool(self, generation, population, fitnesses, cases,
                             rate, step):
        """Breed the behaviorally selected pool, optionally as islands."""
        if not self.config.islands:
            self._island_bounds = ()
            return step(population, fitnesses, cases, rate)
        bounds = self.config.island_slices(len(population))
        if len(bounds) < 2:
            self._island_bounds = ()
            return step(population, fitnesses, cases, rate)
        self._island_bounds = tuple(bounds)
        offspring = []
        for index, (start, stop) in enumerate(bounds):
            deme_rate = min(self.mutation_limit, max(
                MIN_ADAPTIVE_RATE,
                self.config.island_rate(index, len(bounds), rate)))
            bred = step(
                population[start:stop], fitnesses[start:stop],
                (cases[start:stop] if cases is not None else None),
                deme_rate)
            offspring.extend(bred[:stop - start])
        # A deme whose breeder returned short would silently shrink the run.
        while len(offspring) < len(population):
            offspring.append(self._clone(offspring[-1]))
        self.islands_bred += 1
        self._pending_migration = (
            generation % self.config.island_migration_interval == 0)
        return offspring[:len(population)]

    def _lineage_seed_indices(self, population, fitnesses, count):
        """Choose a promising but structurally spread set of first walkers."""
        pool = list(range(len(population)))
        first = max(pool, key=lambda i: self._rank(
            population[i], fitnesses[i]))
        chosen = [first]
        pool.remove(first)
        descriptors = [genome_descriptor(genome) for genome in population]
        while pool and len(chosen) < count:
            pick = max(pool, key=lambda i: min(
                genome_distance(descriptors[i], descriptors[j])
                for j in chosen))
            pool.remove(pick)
            chosen.append(pick)
        return chosen

    def _breed_with_lineage_walk(self, generation, population, fitnesses,
                                 cases, rate, step):
        """Breed an ordinary pool plus persistent fitness-blind random walks.

        A local minimum is separated from a better basin by genomes that score
        worse. Neutral drift cannot retain them and a high-rate restart must
        clear the entire valley in one lucky transaction. Walkers instead take
        exactly one mutation event from their own prior state every generation,
        regardless of score. They therefore retain partial, temporarily bad
        construction long enough for the next edit to build on it.
        """
        size = len(population)
        count = max(1, min(size - 1, int(round(
            size * self.config.lineage_walk_fraction))))
        main_count = size - count
        self._lineage_start = main_count
        main_pop = list(population[:main_count])
        main_fits = list(fitnesses[:main_count])
        main_cases = (list(cases[:main_count]) if cases is not None else None)

        if self._lineage_started:
            walker_indices = list(range(main_count, size))
        else:
            walker_indices = self._lineage_seed_indices(
                population, fitnesses, count)
            self._lineage_started = True

        # Feed an improvement found by a walker back into ordinary selection,
        # but never replace a main-pool parent with a worse walker. The walker's
        # own lineage continues independently either way.
        if main_pop and walker_indices:
            best_walker = max(walker_indices, key=lambda i: self._rank(
                population[i], fitnesses[i]))
            worst_main = min(range(main_count), key=lambda i: self._rank(
                main_pop[i], main_fits[i]))
            if (self._rank(population[best_walker], fitnesses[best_walker])
                    > self._rank(main_pop[worst_main],
                                 main_fits[worst_main])):
                main_pop[worst_main] = self._clone_evaluated(
                    population[best_walker])
                main_fits[worst_main] = fitnesses[best_walker]
                if main_cases is not None:
                    main_cases[worst_main] = cases[best_walker]

        offspring = self._breed_selected_pool(
            generation, main_pop, main_fits, main_cases, rate, step)
        # One event, not the globally reheated rate: the point is to accumulate
        # small edits over generations instead of making another destructive
        # jump from the same champion.
        for index in walker_indices:
            offspring.append(self._mutate(
                self._clone(population[index]), 0.0))
            self.lineage_walk_steps += 1
        return offspring[:size]

    def _migrate(self, population, fitnesses, cases, bounds):
        """Ring migration using the evaluated genomes' own fitnesses.

        Migration previously ran in :meth:`breed`, before offspring had been
        evaluated, and paired each new genome with the old parent's fitness at
        the same list slot. That made "best migrant" effectively arbitrary.
        It now runs from :meth:`merge_generation` after evaluation and moves a
        matching genome/fitness/case-vector snapshot as one unit.
        """
        count = min(int(self.config.island_migrants),
                    min(stop - start for start, stop in bounds))
        if count < 1:
            return population, fitnesses, cases
        senders = []
        for start, stop in bounds:
            order = sorted(range(start, stop),
                           key=lambda i: self._rank(population[i],
                                                    fitnesses[i]),
                           reverse=True)
            senders.append([
                (self._clone_evaluated(population[i]), fitnesses[i],
                 (cases[i] if cases is not None else None))
                for i in order[:count]])
        for index, (start, stop) in enumerate(bounds):
            source = senders[index - 1]                  # -1 wraps: a ring
            order = sorted(range(start, stop),
                           key=lambda i: self._rank(population[i],
                                                    fitnesses[i]))
            for slot, (donor, donor_fit, donor_cases) in zip(
                    order[:count], source):
                population[slot] = donor
                fitnesses[slot] = donor_fit
                if cases is not None:
                    cases[slot] = donor_cases
        self.migrations += 1
        return population, fitnesses, cases

    def tick(self):
        """Advance one generation of rebirth cooldown."""
        if self._cooldown > 0:
            self._cooldown -= 1

    # ── robustness ──

    def apply_robustness_blend(self, population, best_fitness):
        """Turn each genome's robust case vector into its ranked scalar.

        Done here rather than in the evaluation worker because the aggregator
        anneals with the run's best fitness, which a worker cannot see. The
        genome's robust CASE VECTOR is what evaluation measured and it never
        changes; only how those cases are collapsed does.
        """
        if not self.config.robustness:
            return
        self.robust_blend = robustness_blend(best_fitness)
        for genome in population:
            cases = getattr(genome, '_robust_cases', None)
            genome._robustness = (
                aggregate_robustness(cases, self.robust_blend)
                if cases is not None else 0.0)

    # ── survivor selection ──

    def survivor_selection(self, parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
        """Restricted tournament replacement, or None when crowding is off.

        Each offspring is compared against the most genetically SIMILAR member
        of a random window of the crowded RESERVE and replaces it only if it
        ranks at least as well. Comparing against a similar incumbent rather
        than a random one is what preserves niches: a specialist is only ever
        displaced by a better version of itself, never by an unrelated champion
        that happens to score higher overall.

        RTR IS MONOTONE BY CONSTRUCTION. Nothing in the reserve can ever be
        replaced by something worse, so the reserve's fitness multiset only
        rises. Applied to the WHOLE population that is measurable and stark:
        zero mean-fitness decreases in 9 of 9 measured runs, against 24-31 in
        60 generations for the ordinary loop. That is what niche preservation
        costs, and on its own it is the wrong shape for escaping a basin — a
        population that can never move downhill cannot cross a valley.

        So only ``crowding_fraction`` of the next population is crowded. The
        rest is filled from the offspring, subject to the bounded contract-
        elite reserve, which is where the exploratory churn lives. At
        ``crowding_fraction == 1.0`` this reduces exactly to textbook RTR over
        the whole population.
        """
        if not self.config.crowding:
            return None
        pop_size = len(parents)
        reserve_size = max(1, min(pop_size, int(round(
            pop_size * self.config.crowding_fraction))))
        # The reserve holds the niche representatives worth protecting: the
        # best-ranked parents. Anything below them is exactly what generational
        # replacement is supposed to sweep away each generation.
        order = sorted(range(pop_size),
                       key=lambda i: self._rank(parents[i],
                                                parent_fitnesses[i]),
                       reverse=True)
        keep = order[:reserve_size]
        pop = [parents[i] for i in keep]
        fits = [parent_fitnesses[i] for i in keep]
        cases = ([parent_cases[i] for i in keep]
                 if parent_cases is not None else None)
        descriptors = [genome_descriptor(genome) for genome in pop]
        window = min(self.config.crowding_window, len(pop))
        # Every offspring challenges the reserve, whether or not it also
        # survives into the generationally-replaced remainder.
        for index, child in enumerate(offspring):
            child_descriptor = genome_descriptor(child)
            sample = random.sample(range(len(pop)), window)
            nearest = min(sample, key=lambda i: genome_distance(
                descriptors[i], child_descriptor))
            child_rank = self._rank(child, offspring_fitnesses[index])
            incumbent_rank = self._rank(pop[nearest], fits[nearest])
            if self.accepts(child_rank, incumbent_rank):
                pop[nearest] = child
                fits[nearest] = offspring_fitnesses[index]
                descriptors[nearest] = child_descriptor
                if cases is not None and offspring_cases is not None:
                    cases[nearest] = offspring_cases[index]
                self.crowding_replacements += 1
        # The GA loop always breeds exactly one offspring per population slot,
        # but this is a public entry point and a caller may pass fewer, so the
        # remainder is clamped and any shortfall is topped up from the parents
        # that missed the reserve. Population size is an invariant every
        # caller relies on.
        remainder = min(pop_size - reserve_size, len(offspring))
        if remainder > 0:
            # A random sample, not a prefix: the offspring list is ordered
            # rescue proposals, then immigrants, then archive descendants, then
            # bred children, so taking a prefix would systematically stock the
            # remainder with immigrants.
            picks = random.sample(range(len(offspring)), remainder)
            pop += [offspring[i] for i in picks]
            fits += [offspring_fitnesses[i] for i in picks]
            if cases is not None:
                cases += ([offspring_cases[i] for i in picks]
                          if offspring_cases is not None
                          else [None] * remainder)
        for i in order[reserve_size:]:
            if len(pop) >= pop_size:
                break
            pop.append(parents[i])
            fits.append(parent_fitnesses[i])
            if cases is not None:
                cases.append(parent_cases[i]
                             if parent_cases is not None else None)
        return pop, fits, cases

    # ── reporting ──

    def stats(self):
        """Live escape telemetry for the GUI."""
        return {
            'summary': self.config.summary(),
            'rebirths': self.rebirths,
            'last_rebirth_gen': self.last_rebirth_gen,
            'archive': len(self.archive),
            'crowding_replacements': self.crowding_replacements,
            'robust_blend': self.robust_blend,
            'migrations': self.migrations,
            'lineage_walk_steps': self.lineage_walk_steps,
            'contract_elite_carries': self.contract_elite_carries,
            'contract_progress_events': self.contract_progress_events,
        }


def build_escape_state(backend, ga_config, chromosome_count=None,
                       io_placement='fixed', evolve_io=False,
                       evolve_delay=None, fnv_families=None,
                       lut_function_families=None):
    """Construct the run's :class:`EscapeState` for one backend.

    THE single construction point. Both drive paths call this rather than
    assembling their own clone/mutate/rank closures, because those closures are
    exactly where the two paths would otherwise diverge — a rebirth that
    mutates under different operators on the app than in the benchmarks would
    be almost impossible to spot from the outside.
    """
    escape = getattr(ga_config, 'escape', None) or OFF
    limit = float(getattr(ga_config, 'mutation_limit', 8.0))
    max_telomere = getattr(ga_config, 'max_telomere', 20)
    if backend == 'nervous':
        from substrates.nervous.ga import clone_genome, mutate_nv, rank_key
        mutate = lambda genome, rate: mutate_nv(
            genome, rate, max_telomere=max_telomere,
            chromosome_count=chromosome_count, evolve_delay=evolve_delay,
            evolve_io=evolve_io, io_placement=io_placement)
    elif backend == 'lut':
        from substrates.lut.ga import clone_genome, mutate_lut, rank_key
        mutate = lambda genome, rate: mutate_lut(
            genome, rate, max_telomere, chromosome_count=chromosome_count,
            evolve_io=evolve_io, io_placement=io_placement,
            function_families=lut_function_families)
    elif backend == 'fnv':
        from substrates.fnv.catalogue import DEFAULT_FAMILIES
        from substrates.fnv.ga import (
            clone_genome, mutate_functional, rank_key)
        families = (
            DEFAULT_FAMILIES if fnv_families is None else fnv_families)
        mutate = lambda genome, rate: mutate_functional(
            genome, rate, max_telomere=max_telomere,
            chromosome_count=chromosome_count, families=families)
    else:
        from substrates.snn.ga import clone_genome, mutate, rank_key as _rank
        rank_key = _rank
        _mutate = mutate
        mutate = lambda genome, rate: _mutate(
            genome, chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=io_placement, mean_mutations=rate)
    return EscapeState(escape, mutation_limit=limit, clone=clone_genome,
                       mutate=mutate, rank=rank_key)


def population_mutation_rate(population, base):
    """Mean self-adaptive rate across a population (for the GUI readout)."""
    rates = [getattr(genome, '_mut_rate', None) for genome in population]
    present = [float(r) for r in rates if r is not None]
    if not present:
        return float(base)
    return sum(present) / len(present)
