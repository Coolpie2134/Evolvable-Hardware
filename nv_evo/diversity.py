"""
nv_evo/diversity.py — how much genuine variety is in an evaluated population?

Once every genome scores 1.0, every fitness-derived spread measure (sigma,
best-minus-mean, ...) is identically zero: they go blind exactly where the
question gets interesting. Diversity then has to be read off STRUCTURE, and
"structure" is not one question but four, which give very different numbers.
This module reports all four and the collapse between them, because the gap is
the finding: 120 genotypes collapsing to 3 behaviours is a monoculture wearing
costumes, while 120 collapsing to 40 is a broad neutral network.

The four levels, from most to least generous:

  exact       — the full inherited genome identity (``genome_signature``):
                rule alleles, gene order, chromosome tags, split points,
                telomeres, timing vectors, architecture. Tags and splits do not
                touch THIS organism's development, but they steer future
                crossover, so they are real heritable variation.
  functional  — variation that can affect a phenotype: architecture, ordered
                rule alleles, germline telomere (it sets growth radius L AND
                the settle budget, so it is emphatically not neutral), and the
                timing vectors the run's node model actually reads. Tags and
                split points are excluded — they are the parts crossover
                bookkeeps rather than expresses.
  phenotype   — the realised circuit: architecture, the grown state grid, and
                the timing values referenced by states PRESENT in that grid.
                Two different rule sets that grow the same body with the same
                per-node delays are the same circuit.
  behavior    — quantised output edges on a frozen OFF-SPEC probe bank, read at
                the genome's own fitted output cell. Off-spec matters: if the
                target pins one exact behaviour then every perfect solver
                SHOULD look identical on the target's own stimuli — that is
                success, not monoculture — so differences can only show up
                where selection never constrained them.

What this does NOT measure is mechanism. Two structurally different circuits
can be behaviourally identical on every probe, and identical mechanisms can
differ in edge timing. ``behavior`` is functional diversity under a declared
probe suite, nothing more; a lesion/perturbation fingerprint would be needed to
speak about mechanism.

Counts alone mislead ("120 unique" hides whether they differ by one bit or are
scattered), so every level reports the cluster-size distribution: distinct
count, largest cluster share, and effective diversity exp(Shannon entropy).
"""
from __future__ import annotations

import math
import random
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .targets import TemporalTarget, Trial

# Probe edges are quantised before hashing so float noise does not manufacture
# diversity. The unit is a fraction of the run's propagation delay, and it is
# recorded in the report: a different value is a different measurement.
PROBE_QUANTUM_FRAC = 0.25
# Default off-spec probe bank size and the RNG seed that fixes it. The bank must
# be frozen and versioned like the scoring goldens, or numbers from two runs are
# not comparable.
PROBE_TRIALS = 6
PROBE_SEED = 606
PROBE_VERSION = 1

LEVELS = ('exact', 'functional', 'phenotype', 'behavior')
#: Level titles used in every report and plot. LEVEL_MEANING defines each one;
#: reports print the definitions so the terms are never assumed.
LEVEL_LABEL = {
    'exact': 'Genotype (exact)',
    'functional': 'Functional genotype',
    'phenotype': 'Phenotype (grown circuit)',
    'behavior': 'Behaviour (off-spec probes)',
}
LEVEL_MEANING = {
    'exact': 'Every inherited difference, including ones with no effect: '
             'chromosome tags, split points, all telomeres, timing vectors, '
             'architecture.',
    'functional': 'Only differences that can change a circuit: rule alleles, '
                  'the germline telomere, the timing vector the run reads, '
                  'architecture. Excludes tags and split points.',
    'phenotype': 'The grown circuit: architecture, the grown state grid, and '
                 'the timing values of states present in it.',
    'behavior': 'Quantised output edges on a fixed probe bank of stimuli the '
                'target does not specify, read at the fitted output cell.',
}
#: Column/metric definitions, printed with every report.
METRIC_MEANING = (
    ('Group', 'A set of genomes identical at that level.'),
    ('Largest', 'Number of genomes in the biggest group.'),
    ('Largest %', 'That group as a percentage of the population.'),
    ('Effective', 'exp(Shannon entropy) of the group sizes: the number of '
                  'equal-sized groups with the same entropy. Equals the group '
                  'count when all groups are the same size.'),
    ('Unmeasured', 'Genomes with no signature at that level (dead organism or '
                   'no usable readout).'),
)
ROBUSTNESS_MEANING = (
    ('Kernel', 'The mutation applied to draw each sample.'),
    ('Silent rate', 'Fraction of sampled mutants that grow the parent\'s '
                    'circuit unchanged. The kernel never returns a clone, but '
                    'a mutation can land on a chromosome tag, a split point, '
                    'an unexpressed rule or a non-maximal telomere. Such a '
                    'mutant scores identically by construction.'),
    ('Local robustness', 'Fraction of ALL sampled mutants that still score at '
                         'or above the validity threshold. Includes silent '
                         'mutants, which always pass.'),
    ('Effective local robustness',
     'Fraction of PHENOTYPE-CHANGING mutants that still score valid. Silent '
     'mutants are excluded, so this measures the circuit rather than the '
     'mutation kernel.'),
    ('Novel-valid rate', 'Fraction of sampled mutants that still score valid '
                         'AND fall in a phenotype group not present in the '
                         'population.'),
)


# ── cluster statistics ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterStats:
    """Distribution of a population over one level's equivalence classes."""

    level: str
    total: int
    distinct: int
    largest: int
    sizes: Tuple[int, ...]          # descending
    unmeasured: int = 0             # genomes whose signature could not be built

    @property
    def largest_share(self):
        return (self.largest / self.total) if self.total else 0.0

    @property
    def effective(self):
        """exp(Shannon entropy) — the number of EVENLY-sized clusters that
        would give the same entropy. Equals `distinct` only when every cluster
        is the same size; collapses toward 1 when one cluster dominates."""
        if not self.total:
            return 0.0
        entropy = 0.0
        for size in self.sizes:
            p = size / self.total
            if p > 0:
                entropy -= p * math.log(p)
        return math.exp(entropy)


def cluster_stats(level, signatures):
    """Bucket signatures into equivalence classes. ``None`` marks a genome that
    could not be measured (dead organism, unusable readout) — counted and
    reported rather than silently dropped."""
    counts = {}
    unmeasured = 0
    for signature in signatures:
        if signature is None:
            unmeasured += 1
            continue
        counts[signature] = counts.get(signature, 0) + 1
    sizes = tuple(sorted(counts.values(), reverse=True))
    return ClusterStats(level=level, total=sum(sizes), distinct=len(sizes),
                        largest=(sizes[0] if sizes else 0), sizes=sizes,
                        unmeasured=unmeasured)


# ── backend adapters ─────────────────────────────────────────────────────────

def _hashable(value):
    """Coerce a grown-grid cell value into something hashable and comparable."""
    if isinstance(value, (int, float, str, bytes, bool)) or value is None:
        return value
    try:
        return tuple(_hashable(item) for item in value)
    except TypeError:
        return repr(value)


def _grid_signature(grid):
    return tuple(sorted((pos, _hashable(state)) for pos, state in grid.items()))


def _nv_adapter():
    from .ga import genome_signature
    from .nervous import (grow_nervous, interpret_nervous, node_delays)
    from .temporal import prepare_net, trace_fixed_outputs
    from .scoring import score_temporal_bundle          # noqa: F401 (contract)
    from .temporal import score_temporal

    def grow(genome, target, config):
        arch = getattr(genome, 'arch', 'single')
        grid = grow_nervous(genome, seeds=tuple(target.inputs),
                            grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs:
            return None
        delays = None if arch == 'tri3' else node_delays(genome, grid, config)
        return grid, delays

    def probe(genome, target, probe_target, config):
        prep = prepare_net(genome, target)
        if prep is None:
            return None
        grid, routing, in_pos, out_pos, _ = prep
        if any(out_pos.get(term.role) is None for term in target.outputs):
            return None
        arch = getattr(genome, 'arch', 'single')
        delays = None if arch == 'tri3' else node_delays(genome, grid, config)
        return trace_fixed_outputs(grid, routing, in_pos, out_pos, probe_target,
                                   delays=delays, arch=arch)

    return {'signature': genome_signature, 'grow': grow, 'probe': probe,
            'score': score_temporal, 'mutate': _nv_mutate}


def _nv_mutate(genome, config, chromosome_count):
    from .ga import mutate_nv, clone_genome
    ga = getattr(config, 'ga', None)
    return mutate_nv(clone_genome(genome), 1.0,
                     chromosome_count=chromosome_count,
                     evolve_delay=bool(getattr(ga, 'evolve_delay', None)))


def _lut_adapter():
    from lut_evo.ga import (genome_signature, prepare_lut, score_lut_temporal,
                            trace_fixed_outputs, mutate_lut, clone_genome)
    from lut_evo.lut import grow_lut

    def grow(genome, target, config):
        grid = grow_lut(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs:
            return None
        return grid, None

    def probe(genome, target, probe_target, config):
        prep = prepare_lut(genome, target)
        if prep is None:
            return None
        grid, out_pos, _ = prep
        if any(out_pos.get(term.role) is None for term in target.outputs):
            return None
        return trace_fixed_outputs(grid, list(target.inputs), out_pos,
                                   probe_target)

    def mutate(genome, config, chromosome_count):
        return mutate_lut(clone_genome(genome), 1.0,
                          chromosome_count=chromosome_count)

    return {'signature': genome_signature, 'grow': grow, 'probe': probe,
            'score': score_lut_temporal, 'mutate': mutate}


def _adapter(backend):
    if backend == 'nervous':
        return _nv_adapter()
    if backend == 'lut':
        return _lut_adapter()
    raise ValueError('diversity is defined for the asynchronous backends '
                     "('nervous', 'lut'), not %r" % (backend,))


# ── the four signatures ──────────────────────────────────────────────────────

def exact_signature(genome, backend):
    """Complete inherited identity, including tags/splits/telomeres/timing."""
    return _adapter(backend)['signature'](genome)


def _active_timing(genome, config):
    """The delay vector the run's node model actually READS.

    Gated on the model exactly as ``nervous.node_delays`` gates, so a dormant
    vector carried by an old checkpoint cannot inflate the count on a run that
    ignores it. (Width was an evolvable vector here too until width evolution
    was retired; only delay remains heritable.)"""
    model = getattr(getattr(config, 'pulse', config), 'model', None)
    delays = getattr(genome, 'state_delays', None)
    if getattr(genome, 'arch', 'single') == 'tri3':
        return None                  # tri3 never consults the vector
    return tuple(delays) if (model == 'pulse_delay' and delays) else None


def functional_signature(genome, backend, config=None):
    """Variation that could change a phenotype: architecture, ordered rule
    alleles, germline telomere, and the timing vectors the model reads.

    Chromosome tags and split points are deliberately excluded: they are
    crossover bookkeeping, not expression. The germline telomere is deliberately
    INCLUDED — it sets the growth radius and the settle budget."""
    if backend == 'nervous':
        from .genome import germline_telomere
        fields = ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out')
        alleles = tuple(
            tuple(tuple(getattr(gene, f) for f in fields)
                  for gene in chromosome.genes)
            for chromosome in genome.chromosomes)
        delays = _active_timing(genome, config)
        return (getattr(genome, 'arch', 'single'), alleles,
                germline_telomere(genome), delays)
    fields = ('ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in', 'self_out')
    alleles = tuple(
        tuple(tuple(getattr(gene, f) for f in fields)
              for gene in chromosome.genes)
        for chromosome in genome.chromosomes)
    telomeres = tuple(getattr(c, 'telomere', 0) for c in genome.chromosomes)
    return ('lut', alleles, telomeres)


def phenotype_signature(genome, backend, target, config=None, adapter=None):
    """The realised circuit: architecture + grown grid + the timing values
    referenced by states actually present in the organism.

    Timing has to be in here: under the legacy profile two genomes can grow an
    identical state grid while carrying different per-node delays, which are
    physically different circuits."""
    adapter = adapter or _adapter(backend)
    grown = adapter['grow'](genome, target, config)
    if grown is None:
        return None
    grid, delays = grown
    live_delays = (tuple(sorted((pos, round(float(v), 9))
                                for pos, v in delays.items()))
                   if delays else None)
    return (getattr(genome, 'arch', 'single') if backend == 'nervous' else 'lut',
            _grid_signature(grid), live_delays)


# ── off-spec behavioural probes ──────────────────────────────────────────────

def make_probe_bank(target, seed=PROBE_SEED, n_trials=PROBE_TRIALS):
    """A frozen OFF-SPEC stimulus bank: same I/O geometry as ``target`` but
    randomly placed float pulses that the target's own banks never contained.

    Expectations are dummies — nothing here is scored. The point is to observe
    the circuit where selection did NOT constrain it, which is the only place
    two equally-perfect solvers are free to differ."""
    rng = random.Random(seed)
    horizon = int(getattr(target, 'T', 24))
    n_inputs = target.n_inputs
    roles = [term.role for term in target.outputs]
    trials = []
    for _ in range(n_trials):
        events = []
        for _ in range(n_inputs):
            lane, t = [], rng.uniform(1.0, 3.0)
            while t < horizon - 1.0:
                lane.append((round(t, 3), round(rng.choice(
                    (0.5, 0.75, 1.0, 1.5, 2.0)), 3)))
                t += rng.uniform(2.5, 6.0)
            events.append(lane)
        streams = [tuple(0 for _ in range(n_inputs)) for _ in range(horizon)]
        trials.append(Trial(streams, {role: [0] * horizon for role in roles},
                            input_events=events))
    probe = TemporalTarget(
        '%s [probe v%d]' % (target.name, PROBE_VERSION), list(target.inputs),
        list(target.outputs), horizon, trials,
        grid_size=target.grid_size, iters=target.iters, score_mode='events',
        max_events=getattr(target, 'max_events', 2048))
    config = getattr(target, 'pulse_config', None)
    if config is not None:
        setattr(probe, 'pulse_config', config)
    lut_config = getattr(target, 'lut_config', None)
    if lut_config is not None:
        setattr(probe, 'lut_config', lut_config)
    return probe


def behavior_signature(genome, backend, target, probe_target=None, config=None,
                       adapter=None, quantum=None):
    """Quantised output edges on the off-spec bank, at the genome's own fitted
    readout cell. Returns None when the circuit has no usable readout."""
    adapter = adapter or _adapter(backend)
    probe_target = probe_target or make_probe_bank(target)
    traces = adapter['probe'](genome, target, probe_target, config)
    if traces is None or getattr(traces, 'overflow', False):
        return None
    if quantum is None:
        delay = getattr(getattr(target, 'pulse_config', None), 'delay', 1.0)
        quantum = PROBE_QUANTUM_FRAC * float(delay or 1.0)
    events = getattr(traces, 'events', {})
    out = []
    for role in sorted(events):
        per_trial = []
        for train in events[role]:
            per_trial.append(tuple(int(round(float(t) / quantum))
                                   for t in sorted(train)))
        out.append((role, tuple(per_trial)))
    return (round(quantum, 9), tuple(out))


# ── the funnel ───────────────────────────────────────────────────────────────

@dataclass
class DiversityReport:
    levels: Tuple[ClusterStats, ...]
    backend: str
    probe_seed: int = PROBE_SEED
    probe_trials: int = PROBE_TRIALS
    probe_version: int = PROBE_VERSION
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def by_level(self, level):
        for stats in self.levels:
            if stats.level == level:
                return stats
        return None


def diversity_funnel(genomes, backend, target, config=None, probe_target=None,
                     levels=LEVELS, on_progress=None):
    """Cluster an evaluated population at each level and return the collapse.

    ``config`` is the run's RunConfig (or anything exposing ``.pulse``/``.ga``);
    it decides which timing vectors are live."""
    genomes = list(genomes)
    adapter = _adapter(backend)
    pulse_config = getattr(config, 'pulse', None) or getattr(
        target, 'pulse_config', None)
    probe_target = probe_target or (make_probe_bank(target)
                                    if 'behavior' in levels else None)
    stats = []
    for level in levels:
        signatures = []
        for index, genome in enumerate(genomes):
            if on_progress is not None:
                on_progress(level, index, len(genomes))
            if level == 'exact':
                signatures.append(exact_signature(genome, backend))
            elif level == 'functional':
                signatures.append(functional_signature(genome, backend, config))
            elif level == 'phenotype':
                signatures.append(phenotype_signature(
                    genome, backend, target, pulse_config, adapter=adapter))
            elif level == 'behavior':
                signatures.append(behavior_signature(
                    genome, backend, target, probe_target, pulse_config,
                    adapter=adapter))
            else:
                raise ValueError('unknown diversity level: %r' % (level,))
        stats.append(cluster_stats(level, signatures))
    return DiversityReport(levels=tuple(stats), backend=backend)


def _wrapped(label, text, label_width=27, total=78):
    """`label   text`, continuation lines hanging under the text column."""
    body = textwrap.wrap(text, width=max(20, total - label_width - 3))
    out = ['  %-*s %s' % (label_width, label, body[0] if body else '')]
    for extra in body[1:]:
        out.append('  %-*s %s' % (label_width, '', extra))
    return out


def format_funnel(report):
    """Per-level group counts, with the probe bank's provenance."""
    header = ('%-27s %7s %8s %10s %10s'
              % ('Level', 'Groups', 'Largest', 'Largest %', 'Effective'))
    lines = ['GROUPS BY LEVEL   (%s, probe bank v%d, seed %d, %d trials)'
             % (report.backend, report.probe_version, report.probe_seed,
                report.probe_trials),
             '  ' + header,
             '  ' + '-' * len(header)]
    for stats in report.levels:
        row = ('  %-27s %7d %8d %9.1f%% %10.2f'
               % (LEVEL_LABEL.get(stats.level, stats.level),
                  stats.distinct, stats.largest,
                  100.0 * stats.largest_share, stats.effective))
        if stats.unmeasured:
            row += '   unmeasured %d' % stats.unmeasured
        lines.append(row)
    for note in report.notes:
        lines.append('  note: %s' % note)
    return '\n'.join(lines)


def format_legend():
    """Definitions of the level names and the metric columns."""
    lines = ['DEFINITIONS']
    for level in LEVELS:
        lines += _wrapped(LEVEL_LABEL.get(level, level),
                          LEVEL_MEANING.get(level, ''))
    lines.append('')
    for term, meaning in METRIC_MEANING:
        lines += _wrapped(term, meaning)
    return '\n'.join(lines)


def format_levels_table(report):
    """Alias for the per-level group table (see format_funnel)."""
    return format_funnel(report)


def format_cluster_breakdown(report):
    """Full cluster-size distribution per level.

    Sizes are grouped (``12 clusters of 1``) rather than listed one per line:
    a 120-genome population of singletons is 120 identical rows otherwise, and
    the grouping loses nothing.
    """
    lines = ['GROUP SIZE DISTRIBUTION']
    for stats in report.levels:
        # ASCII only: this report is printed to the Windows console, where a
        # UTF-8 em dash comes out as mojibake under the cp1252 code page
        lines.append('  %-27s groups: %d   genomes: %d'
                     % (LEVEL_LABEL.get(stats.level, stats.level),
                        stats.distinct, stats.total))
        if not stats.sizes:
            lines.append('      (none measurable)')
            lines.append('')
            continue
        lines.append('      %7s  %7s  %9s' % ('groups', 'size', 'share'))
        grouped = {}
        for size in stats.sizes:
            grouped[size] = grouped.get(size, 0) + 1
        for size in sorted(grouped, reverse=True):
            count = grouped[size]
            share = 100.0 * size / stats.total if stats.total else 0.0
            lines.append('      %7d  %7d  %8.1f%%' % (count, size, share))
        if stats.unmeasured:
            lines.append('      unmeasured: %d' % stats.unmeasured)
        lines.append('')
    return '\n'.join(lines).rstrip()


def format_report(report, population=None, target_name=None, valid=None):
    """The full text report: header, definitions, per-level table, and the
    group-size distribution. Data and term definitions only."""
    lines = []
    if target_name is not None:
        # two short lines: one long header overruns the 80-column report panel
        lines.append('Target:   %s' % target_name)
        second = 'Backend:  %s' % report.backend
        if population is not None:
            second += '   Population: %d' % population
        if valid is not None:
            second += '   Validity threshold: %.3f' % valid
        lines += [second, '']
    lines.append(format_funnel(report))
    lines += ['', format_cluster_breakdown(report)]
    lines += ['', format_legend()]
    return '\n'.join(lines)


# ── mutational robustness ────────────────────────────────────────────────────
# Diversity and robustness are independent axes: a population can be one cloned
# genotype with a huge neutral neighbourhood, or many genotypes each on an
# isolated spike. Robustness answers "how much neutral room surrounds each
# genome", NOT "how many distinct genomes are there".

#: The mutation kernel is part of the measurement and is reported with it.
#: One call to the backend's own weighted mutation operator at mean 1.0 — which
#: draws Poisson(1) events with a floor of one and guarantees a non-clone. This
#: is the kernel evolution actually uses, so the number means something for
#: evolvability rather than describing an idealised single bit flip.
ROBUSTNESS_KERNEL = 'one weighted GA mutation event (Poisson mean 1, min 1, non-clone)'


@dataclass(frozen=True)
class RobustnessReport:
    kernel: str
    samples: int
    valid_threshold: float
    per_genome_local: Tuple[float, ...]
    per_genome_novel: Tuple[float, ...]
    known_phenotypes: int
    #: fraction of samples whose PHENOTYPE matched the parent's. The kernel
    #: never returns a clone, but a mutation can land on a chromosome tag, a
    #: split point, an unexpressed rule, or a non-maximal telomere and grow the
    #: identical circuit. Such a sample is guaranteed to score identically, so
    #: counting it as "survived" measures the kernel, not the circuit.
    per_genome_silent: Tuple[float, ...] = ()
    #: fraction of PHENOTYPE-CHANGING samples that stayed valid; None for a
    #: genome whose every sample was silent.
    per_genome_effective: Tuple[Optional[float], ...] = ()

    @property
    def local(self):
        n = len(self.per_genome_local)
        return sum(self.per_genome_local) / n if n else 0.0

    @property
    def novel_valid(self):
        n = len(self.per_genome_novel)
        return sum(self.per_genome_novel) / n if n else 0.0

    @property
    def silent(self):
        n = len(self.per_genome_silent)
        return sum(self.per_genome_silent) / n if n else 0.0

    @property
    def effective_local(self):
        """Local robustness over samples that actually changed the circuit."""
        values = [v for v in self.per_genome_effective if v is not None]
        return sum(values) / len(values) if values else 0.0

    def histogram(self, bins=10):
        counts = [0] * bins
        for value in self.per_genome_local:
            index = min(bins - 1, int(value * bins))
            counts[index] += 1
        return tuple(counts)


def robustness(genomes, backend, target, config=None, samples=8, valid=0.999,
               seed=4242, on_progress=None):
    """Local valid volume around each genome, plus the rate at which a single
    mutation reaches a DIFFERENT working circuit.

    ``local`` is the fraction of sampled mutants that still solve;
    ``effective_local`` is that fraction over the mutants that actually changed
    the circuit, and ``novel_valid`` is the fraction that still solve AND land
    on a phenotype not already present in the population.

    The split matters: the kernel never returns a clone, but ~1 in 5 one-event
    mutations lands on something the phenotype does not express (a chromosome
    tag, a split point, an unexpressed rule, a non-maximal telomere) and grows
    the identical circuit. Those score identically by construction, so pooling
    them into ``local`` measures the mutation operator as much as the circuit.
    ``silent`` reports their share and ``effective_local`` excludes them.

    Validity uses the RAW behavioural score (no loop-bonus shaping), so a
    structurally-rewarded genome cannot be counted as a solver.
    """
    genomes = list(genomes)
    adapter = _adapter(backend)
    pulse_config = getattr(config, 'pulse', None) or getattr(
        target, 'pulse_config', None)
    chromosome_count = getattr(getattr(config, 'ga', None),
                               'chromosome_count', None)
    known = set()
    for genome in genomes:
        signature = phenotype_signature(genome, backend, target, pulse_config,
                                        adapter=adapter)
        if signature is not None:
            known.add(signature)
    # mutate_* draw from the process-global RNG, so pin it for repeatability
    random.seed(seed)
    local, novel, silent, effective = [], [], [], []
    for index, genome in enumerate(genomes):
        parent = phenotype_signature(genome, backend, target, pulse_config,
                                     adapter=adapter)
        hits = new_hits = quiet = changed = changed_hits = 0
        for sample in range(samples):
            if on_progress is not None:
                on_progress(index, len(genomes), sample, samples)
            mutant = adapter['mutate'](genome, config, chromosome_count)
            signature = phenotype_signature(mutant, backend, target,
                                            pulse_config, adapter=adapter)
            # A mutant that grows the parent's circuit is the SAME hardware, so
            # its score is identical by construction. Track it separately
            # instead of letting it pad the survival rate.
            is_silent = (parent is not None and signature is not None
                         and signature == parent)
            valid_now = adapter['score'](mutant, target) >= valid
            if is_silent:
                quiet += 1
            else:
                changed += 1
            if not valid_now:
                continue
            hits += 1
            if not is_silent:
                changed_hits += 1
            if signature is not None and signature not in known:
                new_hits += 1
        local.append(hits / samples if samples else 0.0)
        novel.append(new_hits / samples if samples else 0.0)
        silent.append(quiet / samples if samples else 0.0)
        effective.append((changed_hits / changed) if changed else None)
    return RobustnessReport(kernel=ROBUSTNESS_KERNEL, samples=samples,
                            valid_threshold=valid,
                            per_genome_local=tuple(local),
                            per_genome_novel=tuple(novel),
                            per_genome_silent=tuple(silent),
                            per_genome_effective=tuple(effective),
                            known_phenotypes=len(known))


def format_robustness(report, bins=10):
    """Robustness measurements and their definitions. Data only."""
    # the kernel description is a sentence, not a number: wrap it like every
    # other prose line instead of running past the panel width
    lines = ['MUTATIONAL ROBUSTNESS']
    lines += _wrapped('Kernel', report.kernel)
    # same 27-wide label column as _wrapped, so values and prose align
    lines += ['  %-27s %d' % ('Samples per genome', report.samples),
              '  %-27s %.3f' % ('Validity threshold',
                                report.valid_threshold),
              '  %-27s %d' % ('Population phenotypes',
                              report.known_phenotypes),
              '  %-27s %.3f' % ('Silent rate', report.silent),
              '  %-27s %.3f' % ('Local robustness', report.local),
              '  %-27s %.3f' % ('Effective local robustness',
                                report.effective_local),
              '  %-27s %.3f' % ('Novel-valid rate', report.novel_valid),
              '',
              '  Local robustness distribution (genomes per band)',
              '      %11s  %6s' % ('band', 'count')]
    counts = report.histogram(bins)
    width = max(counts) or 1
    for index, count in enumerate(counts):
        lo, hi = index / bins, (index + 1) / bins
        bar = '#' * int(round(18.0 * count / width))
        lines.append('      %4.1f - %4.1f  %6d  %s' % (lo, hi, count, bar))
    lines.append('')
    for term, meaning in ROBUSTNESS_MEANING:
        lines += _wrapped(term, meaning)
    return '\n'.join(lines)
