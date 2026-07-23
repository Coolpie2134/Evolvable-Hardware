from __future__ import annotations

from dataclasses import dataclass

from nv_evo.pulse import PulseConfig
from .limits import MAX_CHROMOSOME_COUNT
from .mutation import DEFAULT_MUTATION_LIMIT, DEFAULT_STAGNATION_BETA

DEFAULT_MAX_TELOMERE = 20
DEFAULT_LUT_MAX_TELOMERE = 8

# The only NV Net substrates exposed for NEW runs. Older node models remain in
# the engine solely so existing checkpoints and controlled comparisons load.
#: (tile_arch, node_model, evolve_delay) per profile.
NV_NEW_RUN_PROFILES = {
    'legacy': ('single', 'pulse_delay', None),
    # digital_tri isolates the tile-topology variable: the paper's three-circuit
    # tile under the frozen digital node abstraction. With analog_tri and legacy
    # it completes the topology-vs-physics ablation pair.
    'digital_tri': ('tri3', 'uniform', None),
    'analog_tri': ('tri3', 'paper_analog', None),
}


def is_current_nv_profile(ga_config):
    profile = (ga_config.tile_arch, ga_config.node_model,
               ga_config.evolve_delay)
    return profile in NV_NEW_RUN_PROFILES.values()


def validate_new_nv_profile(ga_config):
    """Reject unsupported architecture/physics pairings for fresh NV runs.

    GAConfig itself remains permissive because retired configurations must
    still deserialize for checkpoint playback and controlled comparisons.
    """
    if not is_current_nv_profile(ga_config):
        raise ValueError(
            'new nervous-net runs require one of the current profiles: legacy '
            '(single-tile width-preserving/evolved-delay), digital tri-circuit '
            'or analog tri-circuit')


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
    # Nervous-net node-timing model (mirrors PulseConfig.model). The dataclass
    # default remains 'uniform' for checkpoint/API compatibility; fresh GUI runs
    # use one of NV_NEW_RUN_PROFILES instead.
    node_model: str = 'uniform'
    # Nervous-net TILE architecture (nv_evo/genome.py TILE_ARCHS):
    # 'single' — one Fig. 3 circuit per tile. 'tri3' — the paper's
    #            three-circuit tile (three independent L/R/D outputs per tile).
    # Tri3 supports the fixed digital abstraction and paper_analog physics; the
    # per-node-type width/delay vectors are single-tile features.
    tile_arch: str = 'single'
    # I/O binding strategy (nv_evo/io_placement.py IO_STRATEGIES), honoured by
    # ALL backends (nervous, LUT, SNN):
    #   'fixed'          — inputs are the seed pads in declared order; outputs are
    #                      trace-/duty-fitted near their terminals (the legacy
    #                      default).
    #   'tag_rank'       — body-gene expression tags rank mature cells; ports
    #                      claim the highest still-free cells in order (Method A).
    #   'wiring_chromosome' — chromosome 3 is a non-developmental but evolvable
    #                      port map. Each gene selects one cell by node type and
    #                      offset in an unbiased stable site order (Method B).
    #   'spatial_chromosome' — chromosome 3 evolves one normalised (x,y)
    #                      anchor per port and attaches it to the nearest free
    #                      living cell.
    # Kept 'fixed' by default so every existing run and checkpoint is
    # byte-identical; the literal is duplicated from IO_STRATEGIES to keep
    # evo_runtime backend-neutral (no nv_evo import at module level).
    io_placement: str = 'fixed'
    # Delay-mutation toggle, decoupled from the model name for ablations.
    # ``None`` keeps the model's pairing (pulse_delay <-> delay mutation). An
    # explicit False disables it — node_model='pulse_delay' with
    # evolve_delay=False is width-preserving transport at the FIXED base delay,
    # isolating width preservation from delay evolvability. True is only valid
    # under the matching model: the engine ignores the vector everywhere else,
    # so enabling the mutation there would silently evolve dead genes.
    # (A companion evolve_width toggle existed until width evolution was
    # retired from the substrate; old checkpoints carrying it are migrated.)
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
        if self.node_model not in ('uniform', 'pulse_delay', 'paper_analog'):
            raise ValueError('node_model must be uniform, pulse_delay or '
                             'paper_analog')
        if self.tile_arch not in ('single', 'tri3'):
            raise ValueError("tile_arch must be 'single' or 'tri3'")
        if self.io_placement not in (
                'fixed', 'tag_rank', 'wiring_chromosome',
                'spatial_chromosome'):
            raise ValueError(
                "io_placement must be 'fixed', 'tag_rank', "
                "'wiring_chromosome' or 'spatial_chromosome'")
        # tri3 evolves routing only, so it pairs with the routing-only engines:
        # 'uniform' (digital) or 'paper_analog' (analog). The width/delay vectors
        # are single-tile node-type features and have no tri3 meaning.
        if self.tile_arch == 'tri3' and self.node_model not in ('uniform',
                                                                'paper_analog'):
            raise ValueError("tri3 tile_arch supports node_model 'uniform' or "
                             "'paper_analog' (width/delay vectors are "
                             'single-tile features)')
        for name, model in (('evolve_delay', 'pulse_delay'),):
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
        """Resolved ``evolve_delay`` boolean for the GA.

        ``None`` falls back to the node model's pairing, so existing configs
        and checkpoints keep today's behaviour unchanged."""
        return (self.node_model == 'pulse_delay'
                if self.evolve_delay is None else self.evolve_delay)

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        # Width evolution has been retired from the substrate. Old checkpoints
        # still carry its toggle; drop it rather than fail to load.
        values.pop('evolve_width', None)
        # The wiring-chromosome strategy briefly shipped under another name;
        # migrate on load so those checkpoints keep working.
        if values.get('io_placement') == 'sex_chromosome':
            values['io_placement'] = 'wiring_chromosome'
        return cls(**values)


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
        # Promote that value so loading a delay-evolving run cannot quietly
        # disable its mutation operator.
        model = pulse_values.get('model', ga_values.get('node_model', 'uniform'))
        # Width evolution is retired: a checkpoint saved under it is loaded on
        # the paper's fixed-width node instead. The genome's dormant width
        # vector is dropped on load (see checkpoint.genome_from_dict), and the
        # GUI flags the run as retired physics because 'uniform'+'single' is
        # not one of NV_NEW_RUN_PROFILES.
        if model == 'evolved_width':
            model = 'uniform'
            pulse_values['model'] = 'uniform'
        ga_values['node_model'] = model
        return cls(ga=GAConfig.from_dict(ga_values),
                   pulse=PulseConfig(**pulse_values))
