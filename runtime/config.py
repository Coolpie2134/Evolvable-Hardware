from __future__ import annotations

from dataclasses import dataclass

from substrates.nervous.pulse import PulseConfig
from .escape import EscapeConfig
from .limits import MAX_CHROMOSOME_COUNT
from .mutation import DEFAULT_MUTATION_LIMIT, DEFAULT_STAGNATION_BETA

DEFAULT_MAX_TELOMERE = 20
DEFAULT_LUT_MAX_TELOMERE = 8
FNV_FAMILIES = (
    'LOGIC', 'DELAY', 'NORMALIZER', 'HOLD', 'C_ELEMENT', 'TOGGLE',
    'GATED_OSCILLATOR',
)

# The only NV Net substrates exposed for NEW runs. Older node models remain in
# the engine solely so existing checkpoints and controlled comparisons load.
#: (tile_arch, node_model, evolve_delay) per profile.
NV_NEW_RUN_PROFILES = {
    # ONE profile. The paper's three-circuit tile on the paper's Fig. 1 analog
    # node — the most physically faithful configuration, and also the best
    # measured, which is the rare case where fidelity and results agree.
    #
    # The topology-vs-physics ablation that the retired 'digital_tri' profile
    # existed to run has been settled (8 targets x 2 seeds, 40 gens, pop 30):
    #
    #     legacy       (single tile, digital)  mean 0.9097   solved 4/8
    #     digital_tri  (paper tile,  digital)  mean 0.8823   solved 3/8
    #     analog_tri   (paper tile,  analog)   mean 0.9554   solved 5/8
    #
    # The paper's tile ALONE does not help — digital_tri scored below the
    # single-tile engine. The analog node physics is what does the work.
    #
    # Held-out certification decided it. On Toggle, legacy trained to 1.000 on
    # 5/5 seeds but only 2/5 CERTIFIED — three memorised timing. analog_tri was
    # 5/5 CERTIFIED. The digital engine's fixed pulse width and rectangular
    # coincidence window are exploitable timing invariants; the analog node's
    # window, output width and refractory all EMERGE from charge/leak/comparator
    # constants, so there is no fixed rectangle to memorise and a circuit has to
    # work by real dynamics.
    #
    # The retired engines remain in the codebase as reference implementations
    # (tests/test_pulse_models.py, tests/test_node_contracts.py audit them);
    # they are simply no longer offered for new runs.
    'analog_tri': ('tri3', 'paper_analog', None),
}


def is_current_nv_profile(ga_config):
    profile = (ga_config.tile_arch, ga_config.node_model,
               ga_config.evolve_delay)
    return profile in NV_NEW_RUN_PROFILES.values()


def validate_new_nv_profile(ga_config):
    """Reject unsupported architecture/physics pairings for fresh NV runs.

    GAConfig itself remains permissive so the retired engines can still be
    constructed directly by the audits that test them as reference
    implementations. They are simply not offered for new runs.
    """
    if not is_current_nv_profile(ga_config):
        raise ValueError(
            'new nervous-net runs use the analog tri-circuit profile '
            "(tile_arch='tri3', node_model='paper_analog'); the single-tile "
            'and digital tri-circuit engines are retired')


def nv_run_config(**ga):
    """A runnable configuration for a NEW nervous-net run.

    The one live profile, assembled in a single place. GAConfig's own field
    defaults still describe the retired single-tile engine because that class
    doubles as the checkpoint deserialisation target and is constructed
    partially throughout the tests; shifting those defaults made every
    partially-built config an invalid tri3/pulse_delay pairing. So the live
    profile is supplied here instead, and ``validate_new_nv_profile`` remains
    the single gate that says which pairings a fresh run may use.
    """
    arch, model, evolve_delay = NV_NEW_RUN_PROFILES['analog_tri']
    ga.setdefault('tile_arch', arch)
    ga.setdefault('node_model', model)
    ga.setdefault('evolve_delay', evolve_delay)
    return RunConfig(ga=GAConfig(**ga), pulse=PulseConfig(model=model))


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
    # default stays 'uniform': this class is also the DESERIALISATION target
    # and is built partially all over the tests, so the live profile is
    # supplied by nv_run_config() rather than by shifting the field default.
    node_model: str = 'uniform'
    # Nervous-net TILE architecture (substrates/nervous/genome.py TILE_ARCHS):
    # 'single' — one Fig. 3 circuit per tile. 'tri3' — the paper's
    #            three-circuit tile (three independent L/R/D outputs per tile).
    # Tri3 supports the fixed digital abstraction and paper_analog physics; the
    # per-node-type width/delay vectors are single-tile features.
    tile_arch: str = 'single'
    # Compatibility I/O binding strategy
    # (substrates/nervous/io_placement.py IO_STRATEGIES).
    # Fresh Nervous/FNV runs require 'fixed' because their native genomes carry
    # evolved source layouts and use fitted probes. SNN supports every legacy
    # strategy except directional terminal nodes; LUT keeps the legacy
    # strategies programmatically in addition to its two native lut_io_mode
    # choices.
    #   'fixed'          — compatibility selector. SNN uses declared pads;
    #                      Nervous/FNV genomes resolve native layouts; LUT uses
    #                      its selected native lut_io_mode.
    #   'terminal_nodes' - body genes evolve ordinary/input/output identity;
    #                      matching mature cells become one-way terminals.
    #   'tag_rank'       — body-gene expression tags rank mature cells; ports
    #                      claim the highest still-free cells in order (Method A).
    #   'wiring_chromosome' — chromosome 3 is a non-developmental but evolvable
    #                      port map. Each gene selects one cell by node type and
    #                      offset in an unbiased stable site order (Method B).
    #   'spatial_chromosome' — chromosome 3 evolves one normalised (x,y)
    #                      anchor per port. Input anchors are developmental
    #                      germlines; outputs attach to nearest free live cells.
    # Kept 'fixed' by default so native-pad backends and old fixed checkpoints
    # are unambiguous; the literal is duplicated from IO_STRATEGIES to keep
    # runtime backend-neutral (no substrates.nervous import at module level).
    io_placement: str = 'fixed'
    # LUT-only physical input architecture. Source pads are ordinary developed
    # cells made source-only at runtime; exterior edges are fixed alternating
    # logical-input buses whose taps face inward through one directional input
    # of every perimeter LUT face.
    lut_io_mode: str = 'source_pads'
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
    # pool pass per generation (runtime.parallel.map_ordered) instead of
    # chunked barriers, so this multiplier is no longer consumed. Kept as a
    # validated field so existing v2 checkpoints still round-trip.
    evaluation_chunk_multiplier: int = 2
    # Local-minimum escape mechanisms (runtime/escape.py). Every one is off by
    # default, so an unconfigured run — and any checkpoint written before the
    # module existed — behaves exactly as it did before.
    escape: EscapeConfig = EscapeConfig()

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
                'fixed', 'terminal_nodes', 'tag_rank', 'wiring_chromosome',
                'spatial_chromosome'):
            raise ValueError(
                "io_placement must be 'fixed', 'terminal_nodes', 'tag_rank', "
                "'wiring_chromosome' or 'spatial_chromosome'")
        if self.lut_io_mode not in ('source_pads', 'exterior_edges'):
            raise ValueError(
                "lut_io_mode must be 'source_pads' or 'exterior_edges'")
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
        if not isinstance(self.escape, EscapeConfig):
            raise ValueError('escape must be an EscapeConfig')

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
        # dataclasses.asdict flattens the nested escape config to a plain dict;
        # rebuild it (tolerantly, so a checkpoint from a different revision of
        # the mechanism set still loads with the rest of its run intact).
        escape = values.get('escape')
        if escape is not None and not isinstance(escape, EscapeConfig):
            values['escape'] = EscapeConfig.from_dict(escape)
        elif escape is None:
            values.pop('escape', None)
        return cls(**values)


@dataclass(frozen=True)
class FNVConfig:
    """Family-level component-bank selection for Functional NV Net runs."""

    families: tuple[str, ...] = FNV_FAMILIES

    def __post_init__(self):
        families = tuple(str(family).upper() for family in self.families)
        unknown = set(families).difference(FNV_FAMILIES)
        if unknown:
            raise ValueError(
                'unknown FNV component families: %s' %
                ', '.join(sorted(unknown)))
        if not families:
            raise ValueError('at least one FNV component family must be enabled')
        if len(set(families)) != len(families):
            raise ValueError('FNV component families may not be repeated')
        object.__setattr__(
            self, 'families',
            tuple(family for family in FNV_FAMILIES if family in families))

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        if 'families' in values:
            values['families'] = tuple(values['families'])
        return cls(**values)


@dataclass(frozen=True)
class RunConfig:
    ga: GAConfig = GAConfig()
    pulse: PulseConfig = PulseConfig()
    fnv: FNVConfig = FNVConfig()

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
                   pulse=PulseConfig(**pulse_values),
                   fnv=FNVConfig.from_dict(values.get('fnv')))
