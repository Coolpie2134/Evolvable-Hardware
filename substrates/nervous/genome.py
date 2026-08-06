"""
substrates/nervous/genome.py - genome native to the hexagonal nervous net.

Fully self-contained: the nervous net shares *no* code with substrates.snn. The SNN's
Gene has four neighbour fields (N/S/E/W) and forbids self_out==0; the hex node
has only three directions (L/R/D) and *wants* to express "off" (a dead /
unrouted cell), so it gets its own gene and its own container types:

    HexGene(ctx_l, ctx_r, ctx_d, self_in, self_out)
        ctx_*     expected neighbour state in each hex direction (context)
        self_in   expected own state (context)
        self_out  state to emit if this gene wins   (0 = off / cell dies)

This mirrors the paper's associative-memory gene exactly: a context (the states
of the neighbouring cells + the cell's own state) mapped to a new state, chosen
by minimum Hamming distance. There is no per-gene time field - the paper's genes
are timeless. Growth is bounded biologically instead of by the paper's "artificial
methods" (section 7): each chromosome carries a telomere (a Hayflick division limit) so
the organism halts its own growth (see Chromosome + substrates/nervous/nervous.py).

The genetic operators (mutation, crossover, selection) live in substrates/nervous/ga.py.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from runtime.limits import MAX_CHROMOSOME_COUNT
from .hexgrid import hex_frontier_cells

MAX_STATE    = 32       # 5-bit cell state: 0-15 = paper AND routing, 16-31 = OR
MAX_GENES    = 24
MAX_CHROMS   = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20        # longest evolvable growth phase (iterations)
MAX_ROUTING_PATCHES = 16  # heritable mature-cell routing overrides

# -- tile architectures ---------------------------------------------------------
# 'single' - the legacy engine: ONE Fig. 3 circuit per tile (5-bit state, one
#            output net that every listening neighbour reads).
# 'tri3'   - the paper's tile layout: THREE independent Fig. 3 circuits per
#            tile, one per output direction (L/R/D). The 15-bit state packs
#            three 5-bit channel configurations (chan L | chan R << 5 |
#            chan D << 10), each indexing ROUTING_HEX exactly as a
#            single-circuit cell state does: 0-15 the paper's AND routings,
#            16-31 their OR twins. (The channels were 4-bit AND-only until the
#            OR half was added; tri tiles could not hold a circulating pulse
#            without it. tritile.widen_legacy_state migrates old states.)
#            Because the channels occupy disjoint bit fields, Hamming context
#            matching and single-bit mutation both act on one channel at a
#            time - the genome exposes three independently mutable 5-bit
#            channels, not one flat 32768-way categorical value.
TILE_ARCHS     = ('single', 'tri3')
TRI_STATE_MAX  = 32768   # 15-bit tri-tile state; 0 = all three channels off (dead)
ARCH_STATE_MAX = {'single': MAX_STATE, 'tri3': TRI_STATE_MAX}
# Random tri genes keep the single-circuit death probability (1/32), not the
# raw alphabet's 1/32768 - otherwise pruning states are essentially unreachable
# in random immigrants (the LUT genome seeds its zero rules explicitly for the
# same reason).
_TRI_DEATH_P   = 1.0 / 32.0


@dataclass
class HexGene:
    ctx_l:    int = 0
    ctx_r:    int = 0
    ctx_d:    int = 0
    self_in:  int = 0
    self_out: int = 0        # 0 = off / dead (native - no shift needed)
    # Heritable developmental node identity. 0 = ordinary body cell,
    # 1 = source-only input terminal, 2 = sink-only output terminal.
    io_kind:  int = 0
    # I/O allele. On a BODY gene this is its expression priority for tag_rank.
    # On chromosome 3 it is either the desired mature node type (type wiring)
    # or the normalised x coordinate (spatial wiring).
    # Defaults preserve old checkpoints and the fixed strategy's RNG stream.
    tag:      int = 0
    # Deprecated checkpoint field, pinned to 1: every port owns one cell.
    io_limit: int = 1
    # On type-wiring genes this selects one matching instance without a
    # left/top bias. On spatial-wiring genes it stores the normalised y.
    io_selector: int = 0


@dataclass
class Chromosome:
    """`telomere` is a HAYFLICK LIMIT - the number of times a lineage may still
    divide, not a global clock. The germline / seed cells start with telomere
    length L (see `germline_telomere`); every time a cell divides (births a live
    frontier cell) the daughter inherits its parent's telomere MINUS ONE. When a
    cell's telomere reaches 0 it is senescent: it stays alive and keeps its
    function (maintenance rules always apply), but it can no longer divide, so no
    new cell is born beyond it. Growth therefore halts on its own at radius L
    from the seeds - the telomere alone bounds the organism's SIZE (replacing the
    old grid_size clip) and its growth DURATION (<= L rings, replacing the old
    fixed iteration cap). Telomere length is evolvable, so the genome itself
    decides how large it grows on the unbounded field."""
    genes: List[HexGene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0
    telomere: int = MAX_TELOMERE
    # Chromosome 3 is marked as the evolvable I/O map under Method B. It is
    # deliberately excluded from development; only its mapping alleles mutate
    # and recombine. Default False keeps old/fixed genomes developmental.
    wiring:   bool = False


@lru_cache(maxsize=None)
def input_layout_domain(radius: int) -> tuple:
    """Honeycomb sites within ``radius`` graph steps of the canonical origin."""
    reached = {(0, 0)}
    frontier = {(0, 0)}
    for _ in range(max(0, int(radius))):
        frontier = {neighbour
                    for cell in frontier
                    for neighbour in hex_frontier_cells(*cell)
                    if neighbour not in reached}
        reached.update(frontier)
    return tuple(sorted(reached))


@lru_cache(maxsize=None)
def input_layout_radius(max_telomere: int, n_inputs: int) -> int:
    """Placement radius with enough distinct sites for every pad."""
    radius = max(1, min(8, int(max_telomere)))
    while len(input_layout_domain(radius)) < max(1, int(n_inputs)):
        radius += 1
    return radius


def random_input_layout(n_inputs: int, max_telomere: int = MAX_TELOMERE):
    """Compact, distinct, collision-free starting pads.

    Input 0 is the origin gauge; the rest are drawn from a small neighbourhood
    so organisms start with their pads close enough to interact. Mutation
    spreads them out later, one honeycomb edge at a time.
    """
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


def nervous_input_positions(genome, fallback) -> tuple:
    """The pads to grow from and inject into: evolved, or the target's own.

    ``input_layout=None`` is a fixed-input genome and keeps the target's
    declared pads. An evolved layout that is malformed, the wrong length,
    self-colliding, or not anchored at the origin returns NO pads, which makes
    the phenotype unbindable. That is deliberate: repairing an invalid layout
    during evaluation - clamping, relocating, deduplicating - would score a
    genome for a body it does not encode, and would hide the very mutations
    that produced the invalid layout from selection.
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


def germline_telomere(genome) -> int:
    """The organism's germline telomere length L: the longest division program
    across its chromosomes. Seed cells start here; growth reaches radius L. A
    genome with no chromosomes cannot divide (L falls back to 1 - seeds only)."""
    body = [c for c in genome.chromosomes if not getattr(c, 'wiring', False)]
    return max((getattr(c, 'telomere', 1) for c in body), default=1)


@dataclass
class RoutingPatch:
    """One phenotype-local edit applied after development has settled.

    Coordinates live in the organism's ordinary hex frame. The patch changes
    only the routing state of an already-living mature cell; it never
    participates in growth, so a useful local circuit edit does not regrow the
    rest of the body.
    """

    x: int = 0
    y: int = 0
    state: int = 1


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0
    # (An evolvable per-node-type pulse-WIDTH vector lived here until it was
    # retired: width evolution is gone from the substrate entirely. Emitted
    # width is now either the run's fixed PulseConfig.width, transported from
    # the input under 'pulse_delay', or emergent under 'paper_analog'.)
    # 'pulse_delay' / width-preserving model ONLY: per-routing-state delay
    # multipliers. Both transported edges use the same selected delay, so width
    # is preserved while propagation speed becomes heritable. ``None`` is the
    # neutral all-1.0 vector and therefore retains the configured fixed delay.
    state_delays: List[float] = None
    # Tile architecture this genome's states are written in (see TILE_ARCHS).
    # Old pickles predate the field: always read via getattr(g,'arch','single').
    arch: str = 'single'
    # Optional somatic routing overlay. Empty preserves every historical
    # genome and evaluation path; populated patches are heritable, mutate and
    # recombine like ordinary alleles, but act only after development settles.
    routing_patches: List[RoutingPatch] = field(default_factory=list)
    # EVOLVED INPUT GEOMETRY: one honeycomb coordinate per logical input, in
    # input order. ``None`` marks a fixed-input genome that reads its pads from
    # the target instead.
    #
    # Position in this tuple IS the logical identity of the input - pad 0 is
    # input 0 - so no per-pad numeric parameter is needed or wanted. Input 0 is
    # pinned to the origin purely as a coordinate gauge: translating a whole
    # organism changes nothing behaviourally, so letting it drift would add a
    # dimension of pure neutral wandering. Every genuine RELATIVE placement is
    # still reachable by moving the other pads.
    #
    # This replaces the grown input-terminal strategies. A grown terminal has a
    # cliff: a genome that fails to express one required terminal gets no
    # meaningful evaluation at all. Discrete pads always preserve the required
    # pad count, so the placement neighbourhood is smooth and local - one pad,
    # one honeycomb edge.
    input_layout: tuple = None


DELAY_MULT_MIN = 0.25    # evolvable propagation-delay bounds (of PulseConfig.delay)
DELAY_MULT_MAX = 4.0
DELAY_LOG_STEP = 0.12    # one fine mutation/tuning step (about +/-12.7 percent)


def default_state_delays():
    """A neutral delay-multiplier vector (all 1.0): fixed-delay transport."""
    return [1.0] * MAX_STATE


def _canonical_draw(terminals=False):
    """A uniformly-drawn CIRCUIT, not a uniformly-drawn bit pattern.

    Drawing over the raw 5-bit register would make every buffer twice as likely
    as every coincidence detector, because the buffers each have two encodings
    (hexgrid.CANONICAL_STATES). Coincidence is the substrate's only
    computational primitive, so that prior worked directly against it.
    """
    from .hexgrid import canonical_states
    return random.choice(canonical_states(terminals))


def _canonical_tri_draw():
    """One tri-tile state: three independently drawn canonical channels."""
    from .tritile import pack_channels
    return pack_channels(_canonical_draw(), _canonical_draw(),
                         _canonical_draw())


def random_hex_gene(arch='single', tag_range=0, terminals=False) -> HexGene:
    # self_in == 0 makes a GROWTH rule: it matches empty cells and is the only
    # kind that can bring an empty cell to life under the sim6 empty-cell guard
    # (division is further gated by the parent's Hayflick telomere). Random
    # genomes need a healthy share of them or nothing ever grows - sim6 gets this
    # for free because table_create manufactures a gene for every empty-cell
    # context it meets.
    #
    if arch not in TILE_ARCHS:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    if arch == 'tri3':
        out = _canonical_tri_draw()
        while out == 0 and random.random() >= _TRI_DEATH_P:
            out = _canonical_tri_draw()
        return HexGene(
            ctx_l    = _canonical_tri_draw(),
            ctx_r    = _canonical_tri_draw(),
            ctx_d    = _canonical_tri_draw(),
            self_in  = 0 if random.random() < 0.25 else _canonical_tri_draw(),
            self_out = out,
        )
    # Context fields are canonical too: they are Hamming-match ANCHORS against
    # real cell states, and a cell can now only ever hold a canonical state, so
    # an alias anchor could never sit at distance 0 from the circuit it is
    # trying to name - a silent handicap on half the drawn contexts.
    return HexGene(
        ctx_l    = _canonical_draw(terminals),
        ctx_r    = _canonical_draw(terminals),
        ctx_d    = _canonical_draw(terminals),
        self_in  = 0 if random.random() < 0.25 else _canonical_draw(terminals),
        self_out = _canonical_draw(terminals),       # 0 = death
    )


def random_hex_chromosome(n_genes=None, max_telomere=MAX_TELOMERE,
                          arch='single', wiring=False,
                          terminals=False) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_hex_gene(arch, terminals=terminals)
                 for _ in range(n_genes)],
        split = random.randint(1, max(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, min(5, max_telomere)),
        wiring   = wiring,
    )


def random_hex_genome(n_chroms=2, max_telomere=MAX_TELOMERE,
                      arch='single', wiring_chromosome=False, n_ports=None,
                      tag_rank=False, spatial_chromosome=False,
                      terminal_nodes=False, n_inputs=0,
                      n_outputs=0, input_layout=False) -> Genome:
    """Random genome with optional evolvable I/O metadata.

    Fixed runs take the original byte-identical path. Method A seeds body-gene
    priorities; Method B reserves chromosome three as a non-developmental port
    map sized to ``n_ports``.
    """
    if arch not in TILE_ARCHS:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    # Under terminal_nodes binding the two dedicated I/O node types are real
    # distinct circuits, so they stay in the drawable alphabet there.
    chroms = [random_hex_chromosome(max_telomere=max_telomere, arch=arch,
                                    terminals=terminal_nodes)
              for _ in range(n_chroms)]
    genome = Genome(
        chromosomes = chroms,
        tag = random.randint(0, 9999),
        arch = arch,
    )
    if wiring_chromosome or spatial_chromosome or tag_rank:
        from .io_placement import seed_io_metadata
        seed_io_metadata(genome, wiring_chromosome=wiring_chromosome,
                         n_ports=n_ports, tag_rank=tag_rank,
                         spatial_chromosome=spatial_chromosome)
    if terminal_nodes:
        from .io_placement import seed_terminal_states
        seed_terminal_states(genome, n_inputs, n_outputs)
    if input_layout:
        genome.input_layout = random_input_layout(n_inputs, max_telomere)
    return genome
