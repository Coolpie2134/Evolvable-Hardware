"""
substrates/lut/genome.py — genome for the boolean-logic lookup (LUT) array.

Paper Architecture 2 (Edwards EH'02 §5-6; sim6 reference): each cell of a
square, 4-neighbour grid holds four 16-bit lookup tables (one per output
direction). The associative-memory gene is Fig. 10 verbatim: a context of five
16-bit LUT states (the four neighbours' facing LUTs + the cell's own LUT for
that direction) mapped to a new 16-bit LUT, chosen by minimum Hamming distance
over the full 80-bit context. During growth each direction is looked up with
the context rotated to that direction (see lut.py). self_out == 0 (the all-zero
LUT) means that direction is dead; a cell with all four LUTs zero is removed.
Genes are timeless.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from runtime.limits import MAX_CHROMOSOME_COUNT

LUT_BITS     = 16
LUT_STATES   = 1 << LUT_BITS     # 65536 possible cell states
MAX_GENES    = 24
MAX_CHROMS   = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20                # longest evolvable growth phase (iterations)
EDGE_ALLELE_STATES = 1 << 16
LUT_IO_MODES = ('source_pads', 'exterior_edges')

_N4 = ((0, -1), (1, 0), (0, 1), (-1, 0))
_EDGE_DIRECTIONS = (
    ('N', (0, 1)),
    ('E', (1, 0)),
    ('S', (0, -1)),
    ('W', (-1, 0)),
)


@lru_cache(maxsize=None)
def input_layout_domain(radius: int) -> tuple[tuple[int, int], ...]:
    """Square-lattice sites within ``radius`` cardinal steps of the origin."""
    r = max(0, int(radius))
    return tuple(
        (x, y)
        for x in range(-r, r + 1)
        for y in range(-r, r + 1)
        if abs(x) + abs(y) <= r)


@lru_cache(maxsize=None)
def input_layout_radius(max_telomere: int, n_inputs: int) -> int:
    """Compact mutation domain with enough distinct sites for every pad."""
    radius = max(1, min(8, int(max_telomere)))
    while len(input_layout_domain(radius)) < max(1, int(n_inputs)):
        radius += 1
    return radius


def random_input_layout(n_inputs: int,
                        max_telomere: int = MAX_TELOMERE) -> tuple:
    """Draw compact, distinct square-grid pads with input zero as the gauge."""
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


def random_edge_input_layout(n_inputs: int) -> tuple[int, ...]:
    """Legacy point-contact phases retained only for checkpoint compatibility.

    Current ``exterior_edges`` runs use fixed alternating perimeter buses and
    neither create nor read this field.
    """
    count = max(0, int(n_inputs))
    if count > EDGE_ALLELE_STATES:
        raise ValueError('too many LUT exterior inputs')
    return tuple(random.sample(range(EDGE_ALLELE_STATES), count))


@dataclass
class LutGene:
    ctx_n:    int = 0
    ctx_e:    int = 0
    ctx_s:    int = 0
    ctx_w:    int = 0
    self_in:  int = 0
    self_out: int = 0            # 0 = off / dead
    # Heritable developmental node identity. 0 = ordinary body cell,
    # 1 = source-only input terminal, 2 = sink-only output terminal.
    io_kind:  int = 0
    # Body-gene expression priority under tag_rank; desired direction-LUT type
    # or normalised x coordinate on a chromosome-3 port gene.
    tag:      int = 0
    # Deprecated checkpoint field, pinned to 1: every port owns one cell.
    io_limit: int = 1
    # Type-instance selector, or normalised y for spatial wiring.
    io_selector: int = 0


@dataclass
class Chromosome:
    """`telomere`: growth clock — the chromosome's growth rules (self_in == 0)
    only apply while iteration < telomere (sim6's set_limit caps exactly those
    genes); maintenance rules never expire. Evolvable, so the genome bounds its
    own size on the unbounded field."""
    genes: List[LutGene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0
    telomere: int = MAX_TELOMERE
    # Non-developmental port-chromosome marker (see substrates/nervous/genome.py). Its genes
    # hold either type/selector or x/y I/O mappings. Default False leaves every
    # existing genome unchanged.
    wiring:   bool = False


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0
    # Optional heritable polarisation of the developmental seed cell.  ``None``
    # preserves the historical isotropic SEED_STATE exactly.  A four-LUT state
    # lets designed/evolved organisms break the otherwise unavoidable fourfold
    # symmetry of growth from one neutral centre cell.
    seed_state: tuple[int, int, int, int] | None = None
    # Non-behavioural audit label. Compiler rescues retain their origin through
    # mutation/checkpointing so they cannot be mistaken for unaided discoveries.
    provenance: str = ''
    # None marks a legacy fixed-input genome. Native LUT genomes carry exactly
    # one square-lattice source pad per logical input; input zero is pinned only
    # to remove behaviorally meaningless whole-organism translation.
    input_layout: tuple[tuple[int, int], ...] | None = None
    # Deprecated checkpoint field from the former point-contact exterior mode.
    # Current exterior I/O is a fixed alternating bus over every exposed face,
    # so this data round-trips but has no developmental or behavioral effect.
    edge_input_layout: tuple[int, ...] | None = None


def lut_input_positions(genome: Genome, fallback) -> tuple:
    """Resolve native source pads; invalid evolved layouts are unbindable.

    Legacy genomes (``input_layout is None``) retain their target-declared pads.
    An evolved layout is never repaired, clamped, or deduplicated while scoring.
    """
    fallback = tuple(tuple(cell) for cell in fallback)
    layout = getattr(genome, 'input_layout', None)
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


def lut_io_mode(target) -> str:
    """The target-carried LUT I/O architecture (legacy callers use pads)."""
    mode = str(getattr(target, 'lut_io_mode', 'source_pads'))
    if mode not in LUT_IO_MODES:
        raise ValueError(
            "lut_io_mode must be 'source_pads' or 'exterior_edges'")
    return mode


def lut_growth_seeds(genome: Genome, target, strategy='fixed') -> tuple:
    """Developmental germlines for the selected LUT I/O architecture."""
    if lut_io_mode(target) == 'exterior_edges':
        return ((0, 0),)
    layout = getattr(genome, 'input_layout', None)
    if layout is not None:
        return lut_input_positions(genome, target.inputs)
    from substrates.nervous.io_placement import growth_seeds
    return tuple(growth_seeds(target, strategy, genome))


def _outer_empty_sites(grid) -> set:
    """Empty lattice sites connected to infinity around a finite LUT body."""
    if not grid:
        return set()
    xs = [cell[0] for cell in grid]
    ys = [cell[1] for cell in grid]
    low_x, high_x = min(xs) - 1, max(xs) + 1
    low_y, high_y = min(ys) - 1, max(ys) + 1
    start = (low_x, low_y)
    outside = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _N4:
            pos = (x + dx, y + dy)
            if (pos in outside or pos in grid
                    or not low_x <= pos[0] <= high_x
                    or not low_y <= pos[1] <= high_y):
                continue
            outside.add(pos)
            stack.append(pos)
    return outside


def lut_exterior_edges(grid) -> tuple:
    """Ordered exposed outer edges as ``(source_position, cell, direction)``.

    Source positions sit just beyond a cell face. They are real, distinct
    physical ports even when two faces border the same empty lattice site.
    Enclosed holes are not an outside border and are intentionally excluded.
    """
    if not grid:
        return ()
    outside = _outer_empty_sites(grid)
    entries = []
    for x, y in sorted(grid):
        for direction, (dx, dy) in _EDGE_DIRECTIONS:
            if (x + dx, y + dy) not in outside:
                continue
            source = (x + 0.65 * dx, y + 0.65 * dy)
            entries.append((source, (x, y), direction))
    if not entries:
        return ()
    cx = sum(source[0] for source, _, _ in entries) / len(entries)
    cy = sum(source[1] for source, _, _ in entries) / len(entries)
    direction_rank = {name: index for index, (name, _) in
                      enumerate(_EDGE_DIRECTIONS)}
    return tuple(sorted(
        entries,
        key=lambda item: (
            math.atan2(item[0][1] - cy, item[0][0] - cx),
            (item[0][0] - cx) ** 2 + (item[0][1] - cy) ** 2,
            item[1],
            direction_rank[item[2]],
        )))


def lut_exterior_inputs(genome: Genome, grid, n_inputs: int):
    """Build alternating logical-input buses over every exposed outer face.

    ``positions[i]`` is the complete group of taps driven by logical input i.
    The stable cyclic perimeter order is assigned ``A, B, ..., A, B, ...``;
    therefore a two-input target receives A/B alternating around the whole
    organism rather than two movable point contacts. ``links`` maps every tap
    to its one facing cell/direction. Enclosed holes remain excluded by
    :func:`lut_exterior_edges`.

    ``genome`` is retained in the signature for API/checkpoint compatibility;
    exterior attachment is deliberately not genetic.
    """
    count = max(0, int(n_inputs))
    if count == 0:
        return (), {}
    edges = lut_exterior_edges(grid)
    if len(edges) < count:
        return (), {}
    groups = [[] for _ in range(count)]
    for index, (source, _cell, _direction) in enumerate(edges):
        groups[index % count].append(source)
    positions = tuple(tuple(group) for group in groups)
    links = {
        source: (cell, direction) for source, cell, direction in edges
    }
    return positions, links


def random_lut_gene() -> LutGene:
    # self_in == 0 makes a GROWTH rule — the only kind that can bring a dead
    # direction to life under the sim6 empty-cell guard, and the kind expired
    # by telomeres. With uniform 16-bit self_in the chance of one is ~1/65536,
    # so random genomes must be seeded with them explicitly (sim6's
    # table_create manufactures them for every empty context it meets).
    return LutGene(
        ctx_n    = random.randrange(LUT_STATES),
        ctx_e    = random.randrange(LUT_STATES),
        ctx_s    = random.randrange(LUT_STATES),
        ctx_w    = random.randrange(LUT_STATES),
        self_in  = 0 if random.random() < 0.25 else random.randrange(LUT_STATES),
        self_out = random.randrange(LUT_STATES),
    )


def random_lut_chromosome(n_genes=None, wiring=False) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_lut_gene() for _ in range(n_genes)],
        split = (0 if n_genes < 2 else random.randint(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, min(5, MAX_TELOMERE)),
        wiring   = wiring,
    )


def random_lut_genome(n_chroms=2, wiring_chromosome=False, n_ports=None,
                      tag_rank=False, spatial_chromosome=False,
                      terminal_nodes=False, n_inputs=0,
                      n_outputs=0, input_layout=False,
                      edge_input_layout=False,
                      max_telomere=MAX_TELOMERE) -> Genome:
    """Build a fixed genome or seed one of the evolvable I/O strategies."""
    chroms = [random_lut_chromosome() for _ in range(n_chroms)]
    genome = Genome(
        chromosomes = chroms,
        tag = random.randint(0, 9999),
    )
    if wiring_chromosome or spatial_chromosome or tag_rank:
        from substrates.nervous.io_placement import seed_io_metadata
        seed_io_metadata(genome, wiring_chromosome=wiring_chromosome,
                         n_ports=n_ports, tag_rank=tag_rank,
                         spatial_chromosome=spatial_chromosome)
    if terminal_nodes:
        from substrates.nervous.io_placement import seed_terminal_kinds
        seed_terminal_kinds(genome, n_inputs, n_outputs)
    if input_layout:
        genome.input_layout = random_input_layout(n_inputs, max_telomere)
    if edge_input_layout:
        genome.edge_input_layout = random_edge_input_layout(n_inputs)
    return genome
