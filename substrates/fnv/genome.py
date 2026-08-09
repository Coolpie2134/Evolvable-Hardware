"""Output-rooted branched FNV genome: chromosome arms and context rules.

A chromosome has two arms and therefore two branches. Every gene is a context
rule - the states required of the three neighbours, the state the cell must
already be in, and what it becomes - plus the depth band along its own branch
where it applies. Each target output owns one arm and one genetic site. That
arm starts at its writable OUT site and develops back toward read-only PADs.

The target contributes only stable output-role names. It never contributes a
desired truth table or prescribed coordinate: pad and output geometry, local
component choices, and the neighbourhoods rules react to are all genetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import random

from runtime.limits import MAX_CHROMOSOME_COUNT
from substrates.nervous.hexgrid import hex_frontier_cells, hex_pixel

from .catalogue import (
    BY_NAME, DEFAULT_FAMILIES, FAMILIES, _IDS_BY_FAMILY,
    enabled_component_ids, normalise_families,
)

# Fixed-function phenotypes need many distinct developmental contexts. These are
# ceilings, not initial sizes: a fresh arm holds a spawn and a few context rules
# and may grow by ordinary structural mutation.
MAX_GENES = 64
MAX_PLACEMENTS = 128
MAX_CHROMS = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 32
#: The only FNV encoding. Earlier ones - the associative CAM genome and the
#: dependency-addressed constructive genome - were removed, so a checkpoint from
#: either is rejected by the version stamp rather than silently misread.
BRANCHED_ENCODING = "branched_v6"
DEVELOPMENT_VERSION = 6
SEED_STATE = BY_NAME["DELAY1_D_TO_LR"].id
# Fixed source-interface identities break an otherwise artificial symmetry
# between logical input roles during development. FunctionalSim treats every
# declared input as an external driver regardless of this local component, so
# the palette changes morphogenesis only - not the injected signal or wire rules.
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
def input_ring(distance: int) -> tuple[tuple[int, int], ...]:
    """Cells exactly ``distance`` steps from the origin, in angular order.

    A pad gene names a BEARING and a DISTANCE out from pad zero, so it needs a
    stable ring to index into. Angular order means a bearing mutation slides a
    pad around the anchor and a distance mutation moves it in or out, rather
    than teleporting it.
    """
    reached = {(0, 0)}
    frontier = {(0, 0)}
    for _ in range(max(0, int(distance))):
        nxt = {neighbor
               for cell in frontier
               for neighbor in hex_frontier_cells(*cell)
               if neighbor not in reached}
        reached |= nxt
        frontier = nxt
    return tuple(sorted(
        frontier, key=lambda cell: (math.atan2(*reversed(hex_pixel(*cell))),
                                    cell)))


#: How far out a pad gene may sit from the anchor.
MAX_INPUT_DISTANCE = 8


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


#: A cell with nothing in it. Legal as context and as output, so a rule may
#: react to empty space and may erase a cell back to it.
EMPTY_STATE = 0
#: Reserved context value meaning "an input pad". Distinct from every component
#: id and from EMPTY, so no grown cell can impersonate a pad. PAD is a terminal
#: cue: it can complete a branch but can no longer start one.
PAD_STATE = -1
#: Reserved SELF context meaning an unoccupied genetic output site. OUT is a
#: writable developmental niche: its assigned arm may replace it with a real
#: component, after which that component's ordinary state is visible.
OUT_STATE = -2
#: Depth bands a rule may key on: how far the cell sits along its OWN branch,
#: measured from that branch's spawn. This is the positional half of ontogeny -
#: it is what lets one gene behave differently at the base of a limb than at its
#: tip, so a repeated rule produces graded structure instead of a uniform tile.
#: Banded rather than exact so a rule keeps matching as a limb lengthens.
DEPTH_BANDS = 4
#: A rule that does not care where along its branch it is.
DEPTH_ANY = -1
#: Widest reach an arm may evolve. Distance is the summed difference of node
#: type numbers across the four context slots, so within one catalogue family
#: neighbouring types differ by only a few and across families by tens.
MAX_TOLERANCE = 128
#: Cell changes one arm may make before it dies.
MAX_ARM_TELOMERE = 32


@dataclass
class ContextGene:
    """One rule: a neighbourhood the cell must be in, and what it becomes.

    The same shape the nervous and LUT substrates use. ``ctx_l/ctx_r/ctx_d`` are
    the states required of the three neighbours in the cell's own rotated frame,
    ``self_in`` the state the cell must already be in, and ``self_out`` what it
    becomes. Every field may be EMPTY, so a rule can react to empty space and
    can erase a cell; ``self_out`` of EMPTY kills it.

    Matching is EXACT. A cell no gene matches is left exactly as it is, which is
    what lets structure persist rather than being rewritten every iteration.

    ``branch_id`` records which arm the gene sits in. An arm is the unit
    crossover trades, owns a lifespan, and confines the rule to territory that
    arm already owns or touches (apart from its assigned OUT-root spawn).
    """

    gene_id: int
    ctx_l: int = EMPTY_STATE
    ctx_r: int = EMPTY_STATE
    ctx_d: int = EMPTY_STATE
    self_in: int = EMPTY_STATE
    self_out: int = EMPTY_STATE
    branch_id: int = 0
    #: Which depth band along its own branch this rule applies at, or DEPTH_ANY
    #: for all of them. DEPTH_ANY is the default, so a rule is positionally
    #: agnostic until evolution specialises it.
    depth: int = DEPTH_ANY

    @property
    def context(self):
        return (self.ctx_l, self.ctx_r, self.ctx_d, self.self_in)

    def applies_at(self, band):
        return self.depth == DEPTH_ANY or int(self.depth) == int(band)

    def touches_pad(self):
        return PAD_STATE in (self.ctx_l, self.ctx_r, self.ctx_d)

    def spawns_output(self):
        return int(self.self_in) == OUT_STATE


@dataclass
class InputGene:
    """Where one input pad sits: a bearing and a distance out from pad zero.

    Pad zero itself is not encoded - it stays at the origin as the coordinate
    gauge, so every RELATIVE arrangement is reachable without the organism being
    free to drift sideways into identical copies of itself.
    """

    gene_id: int
    distance: int = 1
    bearing: int = 0
    branch_id: int = 0

    def cell(self):
        ring = input_ring(max(1, min(int(self.distance), MAX_INPUT_DISTANCE)))
        return ring[int(self.bearing) % len(ring)]


@dataclass
class OutputGene:
    """Genetic position and arm ownership of one target output role."""

    gene_id: int
    role: str
    distance: int = 1
    bearing: int = 0
    branch_id: int = 0

    def cell(self):
        ring = input_ring(max(1, min(int(self.distance), MAX_INPUT_DISTANCE)))
        return ring[int(self.bearing) % len(ring)]


@dataclass
class ControlGene:
    """One arm's control gene: how far its rules reach, and how long it lives.

    Both are genetic material rather than fields on the chromosome, so they
    mutate like anything else and travel with their arm through crossover
    without special handling.

    ``tolerance`` is how far a rule of this arm may sit from a neighbourhood and
    still apply - 0 is an exact match. ``telomere`` is how many cell changes the
    arm may make before it dies, checked at the start of an iteration, so a
    burst may carry it past zero and still stand in full.
    """

    gene_id: int
    tolerance: int = 0
    telomere: int = 6
    branch_id: int = 0


@dataclass
class Chromosome:
    """Two arms about a fixed centromere. Each arm carries its own control gene."""

    genes: list = field(default_factory=list)
    #: The centromere. Genes before it form the top arm (read outward, so
    #: reversed) and genes after it the bottom arm. Either arm may be empty.
    split: int = 0
    tag: int = 0


@dataclass
class Genome:
    chromosomes: list[Chromosome] = field(default_factory=list)
    tag: int = 0
    #: The materialised pad cells. Derived from ``input_chromosome`` - keep them
    #: in step with ``sync_input_layout``.
    input_layout: tuple[tuple[int, int], ...] | None = None
    #: An ADDITIONAL chromosome, beyond the growth ones, placing the input pads.
    #: It never grows anything; it only says where the organism is fed.
    input_chromosome: Chromosome | None = None
    #: Materialised ``(role, cell)`` pairs from ``output_chromosome``.
    output_layout: tuple[tuple[str, tuple[int, int]], ...] = ()
    #: One immutable-role gene per target output. Its branch_id binds that
    #: output site to one stable growth-arm slot.
    output_chromosome: Chromosome = field(default_factory=Chromosome)
    encoding: str = BRANCHED_ENCODING
    next_gene_id: int = 1

    def __getstate__(self):
        """Keep local growth caches out of worker and checkpoint payloads."""
        state = dict(self.__dict__)
        state.pop("_fnv_development_cache_key", None)
        state.pop("_fnv_development_cache", None)
        return state


def resolve_input_layout(genome) -> tuple[tuple[int, int], ...]:
    """Pad cells named by the input chromosome, pad zero at the origin.

    Two genes may name the same cell; the later one is nudged outward around
    its ring to the nearest free site, so pads stay distinct without a gene
    silently losing its meaning.
    """
    chromosome = getattr(genome, "input_chromosome", None)
    genes = [gene for gene in (chromosome.genes if chromosome else ())
             if isinstance(gene, InputGene)]
    cells, taken = [(0, 0)], {(0, 0)}
    for gene in genes:
        cell = gene.cell()
        if cell in taken:
            ring = input_ring(max(1, min(int(gene.distance),
                                         MAX_INPUT_DISTANCE)))
            start = int(gene.bearing) % len(ring)
            for step in range(1, len(ring)):
                candidate = ring[(start + step) % len(ring)]
                if candidate not in taken:
                    cell = candidate
                    break
        cells.append(cell)
        taken.add(cell)
    return tuple(cells)


def sync_input_layout(genome):
    """Materialise ``input_layout`` from the input chromosome."""
    if getattr(genome, "input_chromosome", None) is not None:
        genome.input_layout = resolve_input_layout(genome)
    return genome


def _free_output_cell(gene, taken):
    """Deterministically nudge a colliding output without rewriting its gene."""
    distance = max(1, min(int(gene.distance), MAX_INPUT_DISTANCE))
    desired_ring = input_ring(distance)
    start = int(gene.bearing) % len(desired_ring)
    candidates = [desired_ring[(start + step) % len(desired_ring)]
                  for step in range(len(desired_ring))]
    # A completely occupied ring is rare but must not leave an output sitting on
    # a pad. Search nearby rings in a stable order while keeping the allele
    # visible and unchanged for selection.
    for delta in range(1, MAX_INPUT_DISTANCE):
        for radius in (distance + delta, distance - delta):
            if not 1 <= radius <= MAX_INPUT_DISTANCE:
                continue
            ring = input_ring(radius)
            offset = int(gene.bearing) % len(ring)
            candidates.extend(
                ring[(offset + step) % len(ring)] for step in range(len(ring)))
    return next((cell for cell in candidates if cell not in taken),
                gene.cell())


def resolve_output_layout(genome) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Resolve role sites in chromosome order, avoiding pads and earlier roots."""
    chromosome = getattr(genome, "output_chromosome", None)
    genes = [gene for gene in (chromosome.genes if chromosome else ())
             if isinstance(gene, OutputGene)]
    input_layout = getattr(genome, "input_layout", None)
    taken = set(input_layout if input_layout is not None
                else resolve_input_layout(genome))
    layout = []
    for gene in genes:
        cell = _free_output_cell(gene, taken)
        layout.append((str(gene.role), tuple(cell)))
        taken.add(tuple(cell))
    return tuple(layout)


def sync_output_layout(genome):
    """Materialise role positions after either I/O chromosome changes."""
    genome.output_layout = resolve_output_layout(genome)
    return genome


def functional_output_positions(genome: Genome, outputs) -> dict | None:
    """Return the genome's exact role sites, or None for a malformed binding."""
    expected = tuple(str(getattr(output, "role", output)) for output in outputs)
    layout = getattr(genome, "output_layout", ())
    try:
        parsed = tuple(
            (str(role), (int(cell[0]), int(cell[1]))) for role, cell in layout)
    except (TypeError, ValueError, IndexError):
        return None
    roles = tuple(role for role, _cell in parsed)
    cells = tuple(cell for _role, cell in parsed)
    inputs = tuple(getattr(genome, "input_layout", None) or ())
    if (roles != expected or len(set(roles)) != len(roles)
            or len(set(cells)) != len(cells)
            or set(cells).intersection(inputs)):
        return None
    return dict(parsed)


def is_branched(genome: Genome) -> bool:
    return getattr(genome, "encoding", BRANCHED_ENCODING) == BRANCHED_ENCODING


def input_node_id(index: int) -> int:
    """Permanent owner ID for one logical source pad."""
    return -int(index) - 1


def genome_development_version(genome: Genome) -> int:
    return DEVELOPMENT_VERSION


def functional_input_positions(genome: Genome, fallback) -> tuple:
    """The genome's evolved pads, or nothing if the layout is invalid.

    An invalid layout returns no pads rather than being silently repaired during
    evaluation.
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


def random_component_id(families=None, *, empty_probability=0.0) -> int:
    """Family-first draw so large families do not dominate initialization."""
    enabled = normalise_families(families)
    if random.random() < empty_probability:
        return 0
    family = random.choice(tuple(family for family in FAMILIES
                                 if family in enabled))
    return random.choice(_IDS_BY_FAMILY[family])


def random_functional_genome(
        n_chroms=2, max_telomere=MAX_TELOMERE, families=DEFAULT_FAMILIES,
        n_inputs=None, output_roles=None, n_outputs=None, **_legacy) -> Genome:
    """A fresh output-rooted genome with one active arm per output role."""
    if not 1 <= int(n_chroms) <= MAX_CHROMS:
        raise ValueError("FNV chromosome count is out of range")
    enabled = normalise_families(families)
    if not enabled:
        raise ValueError("an FNV run must enable at least one component family")
    # Imported lazily to keep the data model independent of its developmental
    # interpreter and its operators.
    from .construction_ga import random_branched_genome
    roles = tuple(str(role) for role in (output_roles or ()))
    if not roles:
        roles = tuple("out%d" % index
                      for index in range(max(1, int(n_outputs or 1))))
    if len(roles) > 2 * int(n_chroms):
        raise ValueError("FNV needs at least one chromosome arm per output")
    return random_branched_genome(
        int(n_chroms), enabled, max(1, int(n_inputs or 1)), roles,
        max_telomere=max_telomere)


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
    output_layout = getattr(genome, "output_layout", ())
    try:
        output_roles = tuple(str(role) for role, _cell in output_layout)
        output_sites = tuple((int(cell[0]), int(cell[1]))
                             for _role, cell in output_layout)
    except (TypeError, ValueError, IndexError):
        raise ValueError("FNV output layout must contain role/cell pairs")
    if not output_roles or len(set(output_roles)) != len(output_roles):
        raise ValueError("FNV output roles must be present and distinct")
    if len(set(output_sites)) != len(output_sites):
        raise ValueError("FNV outputs must occupy distinct sites")
    if set(output_sites).intersection(sites if layout is not None else ()):
        raise ValueError("FNV inputs and outputs must occupy distinct sites")

    seen = set()
    inputs = getattr(genome, "input_chromosome", None)
    if inputs is not None:
        for gene in inputs.genes:
            if not isinstance(gene, InputGene):
                raise ValueError("FNV input chromosome holds a non-input gene")
            if not 1 <= int(gene.distance) <= MAX_INPUT_DISTANCE:
                raise ValueError("FNV input distance is out of range")
            if int(gene.gene_id) <= 0 or int(gene.gene_id) in seen:
                raise ValueError(
                    "FNV rule IDs must be unique positive integers")
            seen.add(int(gene.gene_id))
    outputs = getattr(genome, "output_chromosome", None)
    if outputs is None or not outputs.genes:
        raise ValueError("FNV genome has no output chromosome")
    output_branches = set()
    chromosome_roles = []
    for gene in outputs.genes:
        if not isinstance(gene, OutputGene):
            raise ValueError("FNV output chromosome holds a non-output gene")
        if not str(gene.role):
            raise ValueError("FNV output role cannot be empty")
        if not 1 <= int(gene.distance) <= MAX_INPUT_DISTANCE:
            raise ValueError("FNV output distance is out of range")
        if not 1 <= int(gene.branch_id) <= 2 * len(genome.chromosomes):
            raise ValueError("FNV output branch is outside the growth genome")
        if int(gene.branch_id) in output_branches:
            raise ValueError("FNV outputs must own distinct branches")
        if int(gene.gene_id) <= 0 or int(gene.gene_id) in seen:
            raise ValueError("FNV rule IDs must be unique positive integers")
        seen.add(int(gene.gene_id))
        output_branches.add(int(gene.branch_id))
        chromosome_roles.append(str(gene.role))
    if tuple(chromosome_roles) != output_roles:
        raise ValueError("FNV output layout is stale")
    for chromosome_index, chromosome in enumerate(genome.chromosomes):
        if not 0 <= int(chromosome.split) <= len(chromosome.genes):
            raise ValueError("FNV centromere is outside its chromosome")
        cut = max(0, min(int(chromosome.split), len(chromosome.genes)))
        for half, arm in enumerate((chromosome.genes[:cut],
                                    chromosome.genes[cut:])):
            label = 2 * chromosome_index + half + 1
            if sum(1 for gene in arm if isinstance(gene, ContextGene)
                   and gene.spawns_output()) > 1:
                raise ValueError(
                    "an FNV arm may hold at most one output-root gene")
            if arm and label not in output_branches:
                raise ValueError("an unassigned FNV arm must remain dormant")
            if arm and sum(1 for gene in arm
                           if isinstance(gene, ControlGene)) != 1:
                raise ValueError(
                    "an occupied FNV arm holds exactly one control gene")
        for gene in chromosome.genes:
            if isinstance(gene, ControlGene):
                if not 0 <= int(gene.tolerance) <= MAX_TOLERANCE:
                    raise ValueError("FNV arm tolerance is out of range")
                if not 0 <= int(gene.telomere) <= MAX_ARM_TELOMERE:
                    raise ValueError("FNV arm telomere is out of range")
                if int(gene.gene_id) <= 0 or int(gene.gene_id) in seen:
                    raise ValueError(
                        "FNV rule IDs must be unique positive integers")
                seen.add(int(gene.gene_id))
                if int(gene.branch_id) == 0:
                    raise ValueError("FNV rule branch ID cannot be zero")
                continue
            if not isinstance(gene, ContextGene):
                raise ValueError("FNV chromosome contains a legacy rule")
            if int(gene.gene_id) <= 0 or int(gene.gene_id) in seen:
                raise ValueError(
                    "FNV rule IDs must be unique positive integers")
            seen.add(int(gene.gene_id))
            if int(gene.branch_id) == 0:
                raise ValueError("FNV rule branch ID cannot be zero")
            for name in ("ctx_l", "ctx_r", "ctx_d"):
                value = int(getattr(gene, name))
                if value not in (PAD_STATE, OUT_STATE) and value not in enabled_ids:
                    raise ValueError(
                        "FNV rule %s uses a disabled or unknown state" % name)
            if (int(gene.self_in) != OUT_STATE
                    and int(gene.self_in) not in enabled_ids):
                raise ValueError(
                    "FNV rule self_in uses a disabled or unknown state")
            if int(gene.self_out) not in enabled_ids:
                raise ValueError(
                    "FNV rule self_out uses a disabled or unknown state")
            if (gene.spawns_output()
                    and int(gene.branch_id) not in output_branches):
                raise ValueError("FNV output-root gene has no assigned output")
            if not (int(gene.depth) == DEPTH_ANY
                    or 0 <= int(gene.depth) < DEPTH_BANDS):
                raise ValueError("FNV rule depth band is out of range")
    if len(seen) > MAX_PLACEMENTS:
        raise ValueError("FNV genome exceeds placement limit")
    if int(getattr(genome, "next_gene_id", 1)) <= max(seen, default=0):
        raise ValueError("FNV next rule ID is stale")
