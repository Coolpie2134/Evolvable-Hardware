"""Constructive FNV genome plus exact associative-v2 compatibility types."""
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
MAX_PLACEMENTS = 128
MAX_CHROMS = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20
LEGACY_DEVELOPMENT_VERSION = 2
DEVELOPMENT_VERSION = 3
TYPED_DEVELOPMENT_VERSION = 5
ASSOCIATIVE_ENCODING = "associative_v2"
CONSTRUCTIVE_ENCODING = "constructive_v3"
#: Generative variant of the constructive genome. A gene names the COMPONENT
#: TYPE its inputs must come from instead of one specific earlier gene, so a
#: single gene builds wherever its pattern occurs rather than exactly once.
#: That reuse is what makes growth open-ended, so growth needs a bound -
#: without one, "attach a delay to every AND" would place a delay on each AND
#: it just created, forever. The bound is the CHROMOSOME's telomere, exactly as
#: in the associative encoding: a chromosome is one growth program and its
#: telomere is how many developmental rounds that program stays alive for.
#: Rules do not each carry their own budget; a chromosome is what lives and
#: dies, so a block of co-operating rules shares one lifespan.
#:
#: Input pads stay addressed individually (negative ids). Collapsing them into
#: one "pad" type would make logical input A indistinguishable from B, and a
#: circuit that cannot tell its inputs apart cannot compute an asymmetric
#: function like a full adder.
TYPED_ENCODING = "typed_v4"
SEED_STATE = BY_NAME["DELAY1_D_TO_LR"].id
# Fixed source-interface identities break an otherwise artificial symmetry
# between logical input roles during development. FunctionalSim treats every
# declared input as an external driver regardless of this local component, so
# the palette changes morphogenesis only-not the injected signal or wire rules.
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


@dataclass(frozen=True, order=True)
class BranchRef:
    """Stable reference to one physical output port.

    Negative ``node_id`` values name logical input pads (input zero is -1).
    Positive values name a :class:`PlacementGene`. ``direction`` is the
    source component's local L/R/D output port, not an absolute coordinate.
    """

    node_id: int
    direction: str


@dataclass
class PlacementGene:
    """One fixed component placement, or one rule that makes them.

    Under constructive_v3 every input explicitly names the output port that must
    face this component, and the gene places exactly once.

    Under the branched encoding a gene means one of two things, decided purely
    by WHERE it sits in its arm:

    * the rule at the centromere end is its branch's SPAWN. Its single input
      names an input pad and the direction out of that pad, which is where the
      branch starts, and it places its component in that cell.
    * every other rule is a CONTEXT rule. Each input names the grown node TYPE
      that must drive that pin, so the rule fires in every empty cell whose
      neighbourhood matches - wherever that is, and whichever branch built it.

    ``branch_id`` records which arm a gene currently sits in. It is derived from
    position, not hereditary.
    """

    gene_id: int
    component_id: int
    inputs: tuple[BranchRef, ...] = ()
    branch_id: int = 0


@dataclass
class Chromosome:
    genes: list[FunctionalGene] = field(default_factory=list)
    split: int = 0
    tag: int = 0
    telomere: int = 3
    #: Branched encoding only: one lifespan per arm, counted in PLACEMENTS its
    #: rules make. A chromosome has two arms and therefore two branches, each
    #: living and dying on its own. ``telomere`` above stays what it always was
    #: - the constructive growth radius - and neither field reads the other.
    telomere_top: int = 0
    telomere_bottom: int = 0


@dataclass
class Genome:
    chromosomes: list[Chromosome] = field(default_factory=list)
    tag: int = 0
    # None marks a legacy fixed-input genome. New FNV runs carry one discrete
    # pad position per logical input. Input zero is pinned only to remove
    # behaviorally meaningless whole-organism translation.
    input_layout: tuple[tuple[int, int], ...] | None = None
    # Explicit because old FNV checkpoints remain loadable under their exact
    # associative physics. Fresh runs use dependency-addressed construction.
    encoding: str = ASSOCIATIVE_ENCODING
    next_gene_id: int = 1


def is_typed(genome: Genome) -> bool:
    """True for the generative variant whose refs name component types."""
    return getattr(genome, "encoding", ASSOCIATIVE_ENCODING) == TYPED_ENCODING


def is_constructive(genome: Genome) -> bool:
    """True for both placement-gene encodings.

    The typed variant shares the gene structure, mutation set and crossover of
    constructive_v3 and differs only in how a reference RESOLVES, so every
    caller that asks "is this a placement genome?" must accept it too.
    """
    return getattr(genome, "encoding", ASSOCIATIVE_ENCODING) in (
        CONSTRUCTIVE_ENCODING, TYPED_ENCODING)


def input_node_id(index: int) -> int:
    """Permanent branch owner ID for one logical source pad."""
    return -int(index) - 1


def genome_development_version(genome: Genome) -> int:
    # Typed development interprets the same gene fields by a different rule, so
    # it is its own version: a checkpoint written by one must not silently load
    # under the other.
    if is_typed(genome):
        return TYPED_DEVELOPMENT_VERSION
    return (DEVELOPMENT_VERSION if is_constructive(genome)
            else LEGACY_DEVELOPMENT_VERSION)


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
    # Chromosomes remain as hereditary branch containers so the run's existing
    # chromosome-count control and checkpoint metadata retain meaning. Their
    # genes are constructive placements rather than associative CAM rows.
    genome = Genome(
        # The germline telomere is the organism's growth RADIUS, which is what
        # the run's Max telomere control sets. Fresh genomes are born at that
        # ceiling; development refuses any placement deeper than it.
        chromosomes=[Chromosome(
            genes=[], split=0, tag=random.randint(0, 999),
            telomere=max(2, int(max_telomere)))
            for _ in range(int(n_chroms))],
        tag=random.randint(0, 9999),
        input_layout=(
            random_input_layout(n_inputs, max_telomere)
            if n_inputs is not None else None),
        encoding=CONSTRUCTIVE_ENCODING,
        next_gene_id=1,
    )
    # Import lazily to keep the immutable data model independent of its
    # developmental interpreter.
    from .construction import seed_constructive_genome
    target_count = max(6, min(24, 6 * int(n_chroms)))
    basic_logic_body = (
        "LOGIC" in enabled and "DELAY" in enabled
        and enabled.issubset({"LOGIC", "DELAY"}))
    if basic_logic_body and n_inputs is not None and int(n_inputs) > 1:
        # Establish the labelled source cones before filling space at random,
        # so generic bridges stay short and the starting body contains gates
        # rather than spending most of its capacity routing around filler.
        warm_count = min(target_count, max(3, 2 * int(n_inputs)))
        seed_constructive_genome(
            genome, enabled, target_count=warm_count,
            source_fanout=True)
        from .construction_ga import seed_convergent_bridges
        seed_convergent_bridges(
            genome, enabled,
            max_bridges=min(8, max(3, 2 * int(n_inputs))))
        seed_constructive_genome(
            genome, enabled, target_count=target_count,
            source_fanout=False)
    else:
        seed_constructive_genome(
            genome, enabled, target_count=target_count)
    return genome


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
    if is_constructive(genome):
        seen = set()
        typed = is_typed(genome)
        for chromosome in genome.chromosomes:
            # Only typed development reads this: it is the number of rounds the
            # chromosome's rules stay alive for, so zero would be a chromosome
            # that cannot express at all.
            if typed and int(chromosome.telomere) < 1:
                raise ValueError("FNV chromosome telomere must be positive")
            for gene in chromosome.genes:
                if not isinstance(gene, PlacementGene):
                    raise ValueError(
                        "constructive FNV chromosome contains a legacy rule")
                if int(gene.gene_id) <= 0 or int(gene.gene_id) in seen:
                    raise ValueError(
                        "constructive FNV placement IDs must be unique positive integers")
                seen.add(int(gene.gene_id))
                if int(gene.component_id) not in enabled_ids or not int(
                        gene.component_id):
                    raise ValueError(
                        "FNV placement uses a disabled or unknown component")
                if int(gene.branch_id) == 0:
                    raise ValueError("FNV placement branch ID cannot be zero")
                for ref in gene.inputs:
                    if ref.direction not in ("L", "R", "D"):
                        raise ValueError("FNV branch direction is invalid")
                    if int(ref.node_id) == 0:
                        raise ValueError("FNV branch owner ID cannot be zero")
        if len(seen) > MAX_PLACEMENTS:
            raise ValueError("FNV constructive genome exceeds placement limit")
        if int(getattr(genome, "next_gene_id", 1)) <= max(seen, default=0):
            raise ValueError("FNV next placement ID is stale")
        return
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
