from __future__ import annotations

from dataclasses import dataclass

from nv_evo.pulse import PulseConfig

MAX_CHROMOSOME_COUNT = 6
DEFAULT_MAX_TELOMERE = 20
DEFAULT_LUT_MAX_TELOMERE = 8


def default_max_telomere(backend):
    """Backend-specific growth ceiling used by fresh GUI runs."""
    return (DEFAULT_LUT_MAX_TELOMERE
            if str(backend).lower() == 'lut' else DEFAULT_MAX_TELOMERE)


@dataclass(frozen=True)
class GAConfig:
    mean_mutations: float = 4.0
    immigrant_fraction: float = 0.08
    tournament_size: int = 4
    elite_count: int = 1
    mutation_decay: float = 0.997
    selection: str = 'tournament'
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
        if not 0 <= self.immigrant_fraction <= 1:
            raise ValueError('immigrant_fraction must be between 0 and 1')
        if self.tournament_size < 1 or self.elite_count < 0:
            raise ValueError('tournament_size must be positive and elite_count non-negative')
        if not 0 < self.mutation_decay <= 1:
            raise ValueError('mutation_decay must be in (0, 1]')
        if self.selection not in ('tournament', 'lexicase'):
            raise ValueError('selection must be tournament or lexicase')
        if self.max_telomere < 1 or self.cache_size < 1:
            raise ValueError('max_telomere and cache_size must be positive')
        if (self.chromosome_count is not None
                and not 1 <= self.chromosome_count <= MAX_CHROMOSOME_COUNT):
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMOSOME_COUNT)
        if self.evaluation_chunk_multiplier < 1:
            raise ValueError('evaluation_chunk_multiplier must be positive')

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))


@dataclass(frozen=True)
class RunConfig:
    ga: GAConfig = GAConfig()
    pulse: PulseConfig = PulseConfig()

    @classmethod
    def from_dict(cls, values):
        values = values or {}
        return cls(ga=GAConfig.from_dict(values.get('ga')),
                   pulse=PulseConfig(**values.get('pulse', {})))
