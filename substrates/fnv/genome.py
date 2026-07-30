"""Native indirect genome for the Functional NV Net."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import random

from runtime.limits import MAX_CHROMOSOME_COUNT
from substrates.nervous.hexgrid import hex_frontier_cells

from .catalogue import (
    BY_NAME, DEFAULT_FAMILIES, FAMILIES, _IDS_BY_FAMILY,
    enabled_component_ids, normalise_families,
)

# Fixed-function phenotypes need more distinct developmental contexts than the
# compact original NV routing alphabet. This is a ceiling, not an initial size:
# forward-random chromosomes start with 3..12 genes and may subsequently grow
# by ordinary structural mutation.
MAX_GENES = 64
MAX_CHROMS = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20
DEVELOPMENT_VERSION = 2
SEED_STATE = BY_NAME["DELAY1_D_TO_LR"].id
# Fixed source-interface identities break an otherwise artificial symmetry
# between logical input roles during development. FunctionalSim treats every
# declared input as an external driver regardless of this local component, so
# the palette changes morphogenesis only—not the injected signal or wire rules.
SEED_STATES = (
    SEED_STATE,
    BY_NAME["DELAY1_L_TO_RD"].id,
    BY_NAME["DELAY1_R_TO_LD"].id,
    BY_NAME["DELAY2_D_TO_LR"].id,
)


def input_seed_state(index: int) -> int:
    return SEED_STATES[int(index) % len(SEED_STATES)]


def input_seed_grid(seeds) -> dict:
    return {
        tuple(seed): input_seed_state(index)
        for index, seed in enumerate(seeds)
    }


@lru_cache(maxsize=None)
def input_layout_domain(radius: int) -> tuple[tuple[int, int], ...]:
    """Honeycomb sites within ``radius`` graph steps of the canonical origin."""
    reached = {(0, 0)}
    frontier = {(0, 0)}
    for _ in range(max(0, int(radius))):
        frontier = {
            neighbor
            for cell in frontier
            for neighbor in hex_frontier_cells(*cell)
            if neighbor not in reached
        }
        reached.update(frontier)
    return tuple(sorted(reached))


@lru_cache(maxsize=None)
def input_layout_radius(max_telomere: int, n_inputs: int) -> int:
    """Compact placement radius with enough distinct sites for every pad."""
    radius = max(1, min(8, int(max_telomere)))
    while len(input_layout_domain(radius)) < max(1, int(n_inputs)):
        radius += 1
    return radius


def random_input_layout(
        n_inputs: int, max_telomere: int = MAX_TELOMERE,
) -> tuple[tuple[int, int], ...]:
    """Draw compact distinct pads; later local mutations can spread them out."""
    count = max(0, int(n_inputs))
    if not count:
        return ()
    mutation_radius = input_layout_radius(max_telomere, count)
    initial_radius = min(2, mutation_radius)
    while len(input_layout_domain(initial_radius)) < count:
        initial_radius += 1
    domain = list(input_layout_domain(initial_radius))
    domain.remove((0, 0))
    return ((0, 0),) + tuple(random.sample(domain, count - 1))


@dataclass
class FunctionalGene:
    ctx_l: int = 0
    ctx_r: int = 0
    ctx_d: int = 0
    self_in: int = 0
    self_out: int = 0


@dataclass
class Chromosome:
    genes: list[FunctionalGene] = field(default_factory=list)
    split: int = 0
    tag: int = 0
    telomere: int = 3


@dataclass
class Genome:
    chromosomes: list[Chromosome] = field(default_factory=list)
    tag: int = 0
    # None marks a legacy fixed-input genome. New FNV runs carry one discrete
    # pad position per logical input. Input zero is pinned only to remove
    # behaviorally meaningless whole-organism translation.
    input_layout: tuple[tuple[int, int], ...] | None = None


def functional_input_positions(genome: Genome, fallback) -> tuple:
    """Return evolved pads, or target-declared pads for a legacy genome.

    Invalid evolved layouts return no pads instead of being silently repaired
    during evaluation.
    """
    fallback = tuple(tuple(cell) for cell in fallback)
    layout = getattr(genome, "input_layout", None)
    if layout is None:
        return fallback
    try:
        if any(len(cell) != 2 for cell in layout):
            return ()
        sites = tuple((int(cell[0]), int(cell[1])) for cell in layout)
    except (TypeError, ValueError, IndexError):
        return ()
    if (len(sites) != len(fallback) or len(set(sites)) != len(sites)
            or (sites and sites[0] != (0, 0))):
        return ()
    return sites


def germline_telomere(genome: Genome) -> int:
    return max((int(chromosome.telomere)
                for chromosome in genome.chromosomes), default=1)


def random_component_id(families=None, *, empty_probability=0.0) -> int:
    """Family-first draw so large families do not dominate initialization."""
    enabled = normalise_families(families)
    if random.random() < empty_probability:
        return 0
    family = random.choice(tuple(family for family in FAMILIES
                                 if family in enabled))
    return random.choice(_IDS_BY_FAMILY[family])


def random_functional_gene(families=None, *,
                           context_empty_probability=0.15,
                           output_empty_probability=0.10) -> FunctionalGene:
    enabled = normalise_families(families)
    draw_context = lambda: random_component_id(
        enabled, empty_probability=context_empty_probability)
    # A healthy fraction of explicit growth rules is necessary because empty
    # cells may only be born from a gene whose self context is EMPTY.
    self_in = (0 if random.random() < 0.30 else draw_context())
    return FunctionalGene(
        ctx_l=draw_context(),
        ctx_r=draw_context(),
        ctx_d=draw_context(),
        self_in=self_in,
        self_out=random_component_id(
            enabled, empty_probability=output_empty_probability),
    )


def random_functional_chromosome(
        n_genes=None, max_telomere=MAX_TELOMERE, families=None,
        context_empty_probability=0.15,
    output_empty_probability=0.10) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, min(12, MAX_GENES))
    genes = [
        random_functional_gene(
            families,
            context_empty_probability=context_empty_probability,
            output_empty_probability=output_empty_probability)
        for _ in range(max(1, int(n_genes)))
    ]
    return Chromosome(
        genes=genes,
        split=(0 if len(genes) < 2 else random.randint(1, len(genes) - 1)),
        tag=random.randint(0, 999),
        telomere=random.randint(2, min(5, max(2, int(max_telomere)))),
    )


def random_functional_genome(
        n_chroms=2, max_telomere=MAX_TELOMERE, families=DEFAULT_FAMILIES,
        context_empty_probability=0.15,
        output_empty_probability=0.10, n_inputs=None) -> Genome:
    if not 1 <= int(n_chroms) <= MAX_CHROMS:
        raise ValueError("FNV chromosome count is out of range")
    enabled = normalise_families(families)
    if not enabled:
        raise ValueError("an FNV run must enable at least one component family")
    return Genome(
        chromosomes=[
            random_functional_chromosome(
                max_telomere=max_telomere, families=enabled,
                context_empty_probability=context_empty_probability,
                output_empty_probability=output_empty_probability)
            for _ in range(int(n_chroms))
        ],
        tag=random.randint(0, 9999),
        input_layout=(
            random_input_layout(n_inputs, max_telomere)
            if n_inputs is not None else None),
    )


def validate_genome(genome: Genome, families=None) -> None:
    enabled_ids = set(enabled_component_ids(
        DEFAULT_FAMILIES if families is None else families,
        include_empty=True))
    if not genome.chromosomes:
        raise ValueError("FNV genome has no chromosomes")
    layout = getattr(genome, "input_layout", None)
    if layout is not None:
        try:
            if any(len(cell) != 2 for cell in layout):
                raise ValueError
            sites = tuple((int(cell[0]), int(cell[1])) for cell in layout)
        except (TypeError, ValueError, IndexError):
            raise ValueError("FNV input layout must contain integer pairs")
        if sites and sites[0] != (0, 0):
            raise ValueError("FNV input zero must be at the canonical origin")
        if len(set(sites)) != len(sites):
            raise ValueError("FNV input pads must occupy distinct sites")
    for chromosome in genome.chromosomes:
        if not chromosome.genes:
            raise ValueError("FNV chromosome has no genes")
        if chromosome.telomere < 1:
            raise ValueError("FNV chromosome telomere must be positive")
        for gene in chromosome.genes:
            for name in ("ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"):
                if int(getattr(gene, name)) not in enabled_ids:
                    raise ValueError(
                        f"FNV gene {name} uses a disabled or unknown component")
