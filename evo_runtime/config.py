from __future__ import annotations

from dataclasses import dataclass

from nv_evo.pulse import PulseConfig
from .limits import MAX_CHROMOSOME_COUNT
from .mutation import DEFAULT_MUTATION_LIMIT, DEFAULT_STAGNATION_BETA

DEFAULT_MAX_TELOMERE = 20
DEFAULT_LUT_MAX_TELOMERE = 8


def default_max_telomere(backend):
    """Backend-specific growth ceiling used by fresh GUI runs."""
    return (DEFAULT_LUT_MAX_TELOMERE
            if str(backend).lower() == 'lut' else DEFAULT_MAX_TELOMERE)


@dataclass(frozen=True)
class GAConfig:
    mean_mutations: float = 4.0
    mutation_limit: float = DEFAULT_MUTATION_LIMIT
    immigrant_fraction: float = 0.08
    tournament_size: int = 4
    elite_count: int = 1
    mutation_decay: float = 0.997
    stagnation_beta: float = DEFAULT_STAGNATION_BETA
    selection: str = 'tournament'
    # When false, selected parents are cloned separately and then mutated;
    # crossover is skipped without turning mutation or immigration off.
    recombination_enabled: bool = True
    # Nervous-net node-timing model (mirrors PulseConfig.model; must agree with
    # it). 'evolved_width' turns on pulse-width mutation; 'pulse_delay' turns on
    # per-node delay mutation while preserving transported width.
    node_model: str = 'uniform'
    # Timing-mutation toggles, decoupled from the model name for ablations.
    # ``None`` keeps the model's pairing (evolved_width <-> width mutation,
    # pulse_delay <-> delay mutation). An explicit False disables the paired
    # mutation — e.g. node_model='pulse_delay' with evolve_delay=False is
    # width-preserving transport at the FIXED base delay, isolating width
    # preservation from delay evolvability. True is only valid under the
    # matching model: the engine ignores the vectors everywhere else, so
    # enabling the mutation there would silently evolve dead genes.
    evolve_width: bool | None = None
    evolve_delay: bool | None = None
    max_telomere: int = DEFAULT_MAX_TELOMERE
    # ``None`` keeps the legacy/direct-API behaviour where chromosome count may
    # evolve. GUI runs set this explicitly: the "Chroms" option is a structural
    # constraint, not merely an initial-population hint.
    chromosome_count: int | None = None
    cache_size: int = 200_000
    # Reserved / no-op: evaluation now runs one saturated, cancellation-aware
    # pool pass per generation (evo_runtime.parallel.map_ordered) instead of
    # chunked barriers, so this multiplier is no longer consumed. Kept as a
    # validated field so existing v2 checkpoints still round-trip.
    evaluation_chunk_multiplier: int = 2

    def __post_init__(self):
        if self.mean_mutations < 0:
            raise ValueError('mean_mutations must be non-negative')
        if self.mutation_limit < 1:
            raise ValueError('mutation_limit must be at least 1')
        if not 0 <= self.immigrant_fraction <= 1:
            raise ValueError('immigrant_fraction must be between 0 and 1')
        if self.tournament_size < 1 or self.elite_count < 0:
            raise ValueError('tournament_size must be positive and elite_count non-negative')
        if not 0 < self.mutation_decay <= 1:
            raise ValueError('mutation_decay must be in (0, 1]')
        if not 0 <= self.stagnation_beta <= 10:
            raise ValueError('stagnation_beta must be between 0 and 10')
        if self.selection not in ('tournament', 'lexicase'):
            raise ValueError('selection must be tournament or lexicase')
        if self.node_model not in ('uniform', 'evolved_width', 'pulse_delay'):
            raise ValueError('node_model must be uniform, evolved_width or pulse_delay')
        for name, model in (('evolve_width', 'evolved_width'),
                            ('evolve_delay', 'pulse_delay')):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError('%s must be None (model pairing) or boolean' % name)
            if value is True and self.node_model != model:
                raise ValueError(
                    "%s=True requires node_model='%s' (the mutated vector is "
                    'ignored under every other model)' % (name, model))
        if not isinstance(self.recombination_enabled, bool):
            raise ValueError('recombination_enabled must be boolean')
        if self.max_telomere < 1 or self.cache_size < 1:
            raise ValueError('max_telomere and cache_size must be positive')
        if (self.chromosome_count is not None
                and not 1 <= self.chromosome_count <= MAX_CHROMOSOME_COUNT):
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMOSOME_COUNT)
        if self.evaluation_chunk_multiplier < 1:
            raise ValueError('evaluation_chunk_multiplier must be positive')

    def timing_mutations(self):
        """Resolved ``(evolve_width, evolve_delay)`` booleans for the GA.

        ``None`` toggles fall back to the node model's pairing, so existing
        configs and checkpoints keep today's behaviour unchanged."""
        width = (self.node_model == 'evolved_width'
                 if self.evolve_width is None else self.evolve_width)
        delay = (self.node_model == 'pulse_delay'
                 if self.evolve_delay is None else self.evolve_delay)
        return width, delay

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))


@dataclass(frozen=True)
class RunConfig:
    ga: GAConfig = GAConfig()
    pulse: PulseConfig = PulseConfig()

    def __post_init__(self):
        if self.ga.node_model != self.pulse.model:
            raise ValueError('ga.node_model must match pulse.model')

    @classmethod
    def from_dict(cls, values):
        values = values or {}
        pulse_values = dict(values.get('pulse', {}))
        # Development-era pulse-delay checkpoints carried a width-to-delay Gain.
        # Delay is now an evolved genome vector, so discard that obsolete global
        # coupling while retaining the base delay and other run physics.
        pulse_values.pop('delay_gain', None)
        ga_values = dict(values.get('ga') or {})
        # Older checkpoints stored the timing model only with pulse physics.
        # Promote that value so loading a width/delay-evolving run cannot quietly
        # disable its mutation operator.
        ga_values['node_model'] = pulse_values.get(
            'model', ga_values.get('node_model', 'uniform'))
        return cls(ga=GAConfig.from_dict(ga_values),
                   pulse=PulseConfig(**pulse_values))
