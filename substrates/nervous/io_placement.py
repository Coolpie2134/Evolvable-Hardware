"""Input/output placement shared by the developmental substrates.

The legacy ``fixed`` mode grows from geometrically declared input pads and fits
outputs after simulation. Tag/type evolvable modes grow from one neutral centre
and bind ports without looking at target traces. Spatial mode instead makes its
input alleles the developmental germlines:

``terminal_nodes``
    Body genes carry a heritable ``io_kind`` allele: ordinary, input terminal,
    or output terminal. A mature cell inherits the identity of its winning
    developmental gene. Logical ports bind only to cells that actually express
    the matching terminal kind; target coordinates are never consulted. The
    simulators enforce the directionality: input terminals cannot read the body
    and output terminals cannot drive it.

``tag_rank``
    Body-gene ``tag`` alleles are expression priorities.  A mature cell inherits
    the highest tag of a body gene capable of expressing its node type.  Cells
    compete in descending priority order and the logical ports (inputs first,
    then outputs) claim the highest-ranked still-free cell.  A deterministic,
    genome-keyed spatial tie-break prevents equal tags from always preferring a
    corner or one side of the body.

``wiring_chromosome``
    Chromosome three (index two) is a dedicated, non-developmental I/O
    chromosome.  It is still fully heritable: gene ``i`` evolves the desired
    node type and spatial selector for port ``i``.  The
    selector is an offset into a stable, genotype-keyed permutation of instances
    of the desired type. Every port selects exactly one instance.
    Incrementing/decrementing the selector moves to the adjacent candidate
    instead of globally rehashing every site.

``spatial_chromosome``
    Chromosome three is again a dedicated port map, but each port gene stores a
    normalised ``(x, y)`` anchor instead of a node type. The chromosome does not
    express cell state directly, yet its input loci affect development through
    the germline positions they encode.
    Input anchors are the organism's actual germline positions, so moving an
    input changes development as well as wiring. Output anchors attach to the
    nearest still-free mature cell in a stable target coordinate frame.
    Coordinates mutate locally.

Every physical cell is exclusive to one logical port.  This is a semantic
guardrail, not a geometric one: it prevents an input and output from becoming
the same wire (the shortcut that motivated this rewrite) while leaving evolution
free to place distinct ports beside one another when that is useful.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

from .hexgrid import IO_STATE_INPUT, IO_STATE_OUTPUT

Pos = Tuple[int, int]

IO_STRATEGIES = (
    'fixed', 'terminal_nodes', 'tag_rank', 'wiring_chromosome',
    'spatial_chromosome')
_STRATEGY_ALIASES = {'sex_chromosome': 'wiring_chromosome'}

WIRING_CHROMOSOME_INDEX = 2
IO_PRIORITY_MAX = 65535
SPATIAL_COORD_MAX = 65535
_MASK64 = (1 << 64) - 1
IO_KIND_BODY = 0
IO_KIND_INPUT = 1
IO_KIND_OUTPUT = 2
IO_KINDS = (IO_KIND_BODY, IO_KIND_INPUT, IO_KIND_OUTPUT)


def io_strategy(target) -> str:
    """Return the target's validated I/O strategy (legacy targets are fixed)."""
    strategy = getattr(target, 'io_placement', 'fixed') or 'fixed'
    strategy = _STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in IO_STRATEGIES:
        raise ValueError('unknown io_placement strategy: %r' % (strategy,))
    return strategy


def _spatial_domain(target):
    """Canonical integer field used by heritable spatial input anchors."""
    size = max(1, int(getattr(target, 'grid_size', 5) or 5))
    declared = (
        [tuple(pos) for pos in getattr(target, 'inputs', ())]
        + [tuple(terminal.pos)
           for terminal in getattr(target, 'outputs', ())])
    xs = [int(pos[0]) for pos in declared]
    ys = [int(pos[1]) for pos in declared]
    return (
        min([0] + xs), max([size - 1] + xs),
        min([0] + ys), max([size - 1] + ys),
    )


def spatial_port_sites(genome, target):
    """Decode distinct canonical sites for every spatial port allele.

    Quantisation collisions are resolved by the nearest free canonical lattice
    site. This is deterministic and genotype-only: evaluation never repairs or
    rewrites an allele.
    """
    n_ports = (
        int(getattr(target, 'n_inputs', 0) or 0)
        + len(getattr(target, 'outputs', ()) or ()))
    genes = _wiring_port_genes(genome, n_ports)
    if genes is None or not n_ports:
        return ()
    low_x, high_x, low_y, high_y = _spatial_domain(target)
    candidates = [
        (x, y)
        for x in range(low_x, high_x + 1)
        for y in range(low_y, high_y + 1)]
    if len(candidates) < n_ports:
        return ()
    claimed, sites = set(), []
    for port_index, gene in enumerate(genes):
        anchor = (
            _decode_spatial_coord(_gene_tag(gene), low_x, high_x),
            _decode_spatial_coord(
                _gene_selector(gene), low_y, high_y),
        )
        available = [pos for pos in candidates if pos not in claimed]
        selected = min(
            available,
            key=lambda pos: (
                (float(pos[0]) - anchor[0]) ** 2
                + (float(pos[1]) - anchor[1]) ** 2,
                _site_rank(
                    int(getattr(genome, 'tag', 0)) ^ port_index,
                    pos, port_index + 1),
            ))
        claimed.add(selected)
        sites.append(selected)
    return tuple(sites)


def spatial_input_sites(genome, target):
    """Distinct developmental input sites from the spatial chromosome."""
    n_inputs = int(getattr(target, 'n_inputs', 0) or 0)
    return spatial_port_sites(genome, target)[:n_inputs]


def growth_seeds(target, strategy=None, genome=None):
    """Return germline cells for the selected I/O architecture.

    Fixed binding grows from declared pads. Developmental spatial binding grows
    from its heritable input anchors. Other evolvable strategies, plus
    zero-input spatial organisms, retain the neutral centre seed.
    """
    strategy = strategy or io_strategy(target)
    if strategy == 'fixed':
        return tuple(target.inputs)
    if strategy == 'spatial_chromosome' and genome is not None:
        # The explicit LUT truth-table compiler inverse-develops a one-seed
        # phenotype and labels that provenance. Preserve that verified witness;
        # ordinary evolved spatial genomes use developmental input anchors.
        if getattr(genome, 'provenance', '') == 'truth-table-compiler-v1':
            size = int(getattr(target, 'grid_size', 5) or 5)
            return ((size // 2, size // 2),)
        n_inputs = int(getattr(target, 'n_inputs', 0) or 0)
        sites = spatial_input_sites(genome, target)
        if n_inputs and len(sites) == n_inputs:
            return sites
    size = int(getattr(target, 'grid_size', 5) or 5)
    return ((size // 2, size // 2),)


def uses_port_chromosome(strategy) -> bool:
    """Whether ``strategy`` reserves chromosome three as an I/O map."""
    return strategy in ('wiring_chromosome', 'spatial_chromosome')


def evolves_io(strategy) -> bool:
    """Whether a placement strategy has heritable port-placement alleles."""
    return strategy in (
        'terminal_nodes', 'tag_rank', 'wiring_chromosome',
        'spatial_chromosome')


def wiring_chromosome(genome):
    """Return the explicitly designated I/O chromosome, never a body fallback."""
    for chromosome in getattr(genome, 'chromosomes', None) or ():
        if getattr(chromosome, 'wiring', False):
            return chromosome
    return None


def body_chromosomes(genome):
    """Developmental chromosomes only.

    Growth engines call this rather than iterating the raw chromosome list, so
    the evolvable I/O chromosome cannot accidentally alter the organism it maps.
    """
    return [chromosome for chromosome in
            (getattr(genome, 'chromosomes', None) or ())
            if not getattr(chromosome, 'wiring', False)]


def body_type_pool(genome):
    """Nonzero node types that the developmental program can express."""
    return [int(gene.self_out) for chromosome in body_chromosomes(genome)
            for gene in chromosome.genes if getattr(gene, 'self_out', 0)]


def _body_gene_loci(genome):
    return [
        (chromosome, index)
        for chromosome in body_chromosomes(genome)
        for index in range(len(chromosome.genes))
    ]


def seed_terminal_kinds(genome, n_inputs, n_outputs):
    """Seed heritable terminal alleles on a fresh developmental genome.

    Marking a gene does not guarantee that it wins anywhere; evaluation never
    repairs missing terminals. Gene copies preserve the reproduction invariant
    that shared gene objects are otherwise immutable.
    """
    loci = _body_gene_loci(genome)
    if not loci:
        return genome
    requested = (
        [IO_KIND_INPUT] * max(0, int(n_inputs))
        + [IO_KIND_OUTPUT] * max(0, int(n_outputs)))
    if not requested:
        return genome
    # Dense LUT ontogenies can carry hundreds of mostly neutral genes, while NV
    # genomes are much smaller. Give every required role a modest number of
    # opportunities scaled to genome size; binding still activates exactly one
    # cell per port.
    mandatory = []
    if int(n_inputs) > 0:
        mandatory.append(IO_KIND_INPUT)
    if int(n_outputs) > 0:
        mandatory.append(IO_KIND_OUTPUT)
    remainder = list(requested)
    for kind in mandatory:
        remainder.remove(kind)
    random.shuffle(remainder)
    copies = max(
        2, min(8, len(loci) // max(1, 4 * len(requested))))
    extra = list(requested) * (copies - 1)
    random.shuffle(extra)
    roles = mandatory + remainder + extra
    chosen = random.sample(loci, min(len(loci), len(roles)))
    for (chromosome, index), kind in zip(chosen, roles):
        gene = copy.copy(chromosome.genes[index])
        gene.io_kind = kind
        chromosome.genes[index] = gene
    return genome


def mutate_terminal_kind(genome):
    """Change one body's inherited ordinary/input/output node identity."""
    loci = _body_gene_loci(genome)
    if not loci:
        return False
    chromosome, index = random.choice(loci)
    gene = copy.copy(chromosome.genes[index])
    old = int(getattr(gene, 'io_kind', IO_KIND_BODY))
    gene.io_kind = random.choice([kind for kind in IO_KINDS if kind != old])
    chromosome.genes[index] = gene
    return True


# ── nervous-net dedicated I/O node-type states (16 = input, 17 = output) ─────────
# For the NERVOUS backend, terminal identity is the GROWN STATE, not the io_kind
# tag (which LUT/SNN still use). A gene expresses a terminal by growing a cell
# into the I/O state, so seeding/mutation set ``self_out`` instead of io_kind.

def seed_terminal_states(genome, n_inputs, n_outputs):
    """Nervous: seed genes that grow dedicated I/O node-type states.

    Mirrors :func:`seed_terminal_kinds` but sets ``self_out`` to the input
    (16) / output (17) node-type state — a gene grows a terminal by expressing
    the state, so identity is developmental like every other cell fate. Binding
    still activates exactly one cell per port; extra expressed terminals stay
    ordinary I/O-state cells.
    """
    loci = _body_gene_loci(genome)
    if not loci:
        return genome
    requested = ([IO_STATE_INPUT] * max(0, int(n_inputs))
                 + [IO_STATE_OUTPUT] * max(0, int(n_outputs)))
    if not requested:
        return genome
    mandatory = []
    if int(n_inputs) > 0:
        mandatory.append(IO_STATE_INPUT)
    if int(n_outputs) > 0:
        mandatory.append(IO_STATE_OUTPUT)
    remainder = list(requested)
    for state in mandatory:
        remainder.remove(state)
    random.shuffle(remainder)
    copies = max(2, min(8, len(loci) // max(1, 4 * len(requested))))
    extra = list(requested) * (copies - 1)
    random.shuffle(extra)
    states = mandatory + remainder + extra
    chosen = random.sample(loci, min(len(loci), len(states)))
    for (chromosome, index), state in zip(chosen, states):
        gene = copy.copy(chromosome.genes[index])
        gene.self_out = int(state)
        chromosome.genes[index] = gene
    return genome


def mutate_terminal_state(genome):
    """Nervous: flip one body gene's ``self_out`` to / from a dedicated I/O
    node-type state (16 input, 17 output). A non-terminal gene may become a
    terminal; a terminal gene may switch role or revert to an ordinary cell."""
    loci = _body_gene_loci(genome)
    if not loci:
        return False
    chromosome, index = random.choice(loci)
    gene = copy.copy(chromosome.genes[index])
    io_states = (IO_STATE_INPUT, IO_STATE_OUTPUT)
    current = int(gene.self_out) & 0x1F
    if current in io_states:
        other = next(s for s in io_states if s != current)
        gene.self_out = random.choice((other, 0))     # switch role or retire
    else:
        gene.self_out = random.choice(io_states)       # become a terminal
    chromosome.genes[index] = gene
    return True


def _clone_mapping_gene(template, node_type):
    gene = copy.copy(template)
    gene.tag = int(node_type)
    # Start viable and minimally invasive. Fan-out remains evolvable, but a
    # random immigrant must not let input A greedily consume every instance
    # needed by later ports before selection has seen any behaviour.
    gene.io_limit = 1
    gene.io_selector = 0
    return gene


def _clone_spatial_gene(template):
    gene = copy.copy(template)
    gene.tag = random.randrange(SPATIAL_COORD_MAX + 1)
    gene.io_limit = 1
    gene.io_selector = random.randrange(SPATIAL_COORD_MAX + 1)
    return gene


def seed_io_metadata(genome, wiring_chromosome=False, n_ports=None,
                     tag_rank=False, spatial_chromosome=False):
    """Initialise evolvable I/O metadata in place.

    This is called only for an evolvable strategy, preserving the random stream
    of ordinary fixed runs.  Method A seeds body-gene priorities.  Method B
    reserves chromosome three, sizes it to the target's port count when known,
    and seeds either node-type mappings or normalised spatial anchors.
    """
    if tag_rank:
        for chromosome in body_chromosomes(genome):
            for gene in chromosome.genes:
                gene.tag = random.randrange(IO_PRIORITY_MAX + 1)

    port_chromosome = wiring_chromosome or spatial_chromosome
    if not port_chromosome:
        return genome
    chromosomes = list(getattr(genome, 'chromosomes', None) or ())
    if len(chromosomes) <= WIRING_CHROMOSOME_INDEX:
        raise ValueError(
            'chromosome-based I/O requires at least three chromosomes '
            '(chromosome 3 is the evolvable port map)')
    for index, chromosome in enumerate(chromosomes):
        chromosome.wiring = (index == WIRING_CHROMOSOME_INDEX)
    wiring = chromosomes[WIRING_CHROMOSOME_INDEX]
    pool = body_type_pool(genome)
    if not pool:
        pool = [1]

    requested = len(wiring.genes) if n_ports is None else max(0, int(n_ports))
    templates = list(wiring.genes)
    if not templates:
        template = next((gene for chromosome in body_chromosomes(genome)
                         for gene in chromosome.genes), None)
        if template is None:
            return genome
        templates = [template]
    if spatial_chromosome:
        wiring.genes = [
            _clone_spatial_gene(templates[index % len(templates)])
            for index in range(requested)
        ]
    else:
        wiring.genes = [
            _clone_mapping_gene(templates[index % len(templates)],
                                random.choice(pool))
            for index in range(requested)
        ]
    count = len(wiring.genes)
    wiring.split = 0 if count < 2 else max(1, min(wiring.split, count - 1))
    return genome


def seed_wiring_from_phenotype(genome, grid, target, tags=None):
    """Seed a complete one-site mapping from nodes the mature body expresses.

    Generic genome construction cannot know which producible types will survive
    development. App/GA factories call this after growing a fresh body so each
    port begins on a distinct real cell. The alleles remain ordinary heritable
    genes after initialization; no repair occurs during evaluation.

    Returns the number of ports initialized. A body with fewer eligible physical
    cells than ports is initialized as far as possible and remains honestly
    incomplete.
    """
    specs = _port_specs(target)
    genes = _wiring_port_genes(genome, len(specs))
    if genes is None:
        return 0
    node_types = cell_tags(genome, grid) if tags is None else tags
    eligible = [tuple(pos) for pos in node_types
                if _entry_types(node_types[pos])]
    if not eligible:
        return 0
    chosen = random.sample(eligible, min(len(eligible), len(specs)))
    claimed = set()
    initialized = 0
    wiring = wiring_chromosome(genome)
    mapped = list(wiring.genes)
    for index, desired in enumerate(chosen):
        types = _entry_types(node_types[desired])
        if not types:
            continue
        wanted = random.choice(types)
        candidates = _ordered_wiring_candidates(
            genome, node_types, wanted, claimed)
        if desired not in candidates:
            continue
        gene = copy.copy(mapped[index])
        gene.tag = wanted
        gene.io_limit = 1
        gene.io_selector = candidates.index(desired)
        mapped[index] = gene
        claimed.add(desired)
        initialized += 1
    wiring.genes = mapped
    return initialized


def _spatial_bounds(positions):
    positions = [tuple(pos) for pos in positions]
    if not positions:
        return None
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    return min(xs), max(xs), min(ys), max(ys)


def _encode_spatial_coord(value, low, high):
    if high <= low:
        return SPATIAL_COORD_MAX // 2
    fraction = (float(value) - float(low)) / (float(high) - float(low))
    return max(0, min(
        SPATIAL_COORD_MAX, int(round(fraction * SPATIAL_COORD_MAX))))


def _decode_spatial_coord(code, low, high):
    if high <= low:
        return float(low)
    fraction = (int(code) % (SPATIAL_COORD_MAX + 1)) / SPATIAL_COORD_MAX
    return float(low) + fraction * (float(high) - float(low))


def seed_spatial_from_phenotype(genome, grid, target, tags=None):
    """Seed every port at a distinct random living site on the mature body.

    The stored values are normalised coordinates, so the same alleles remain
    meaningful as the organism grows or shrinks. Evaluation never repairs or
    rewrites them; this is fresh-genome initialisation only.
    """
    specs = _port_specs(target)
    genes = _wiring_port_genes(genome, len(specs))
    if genes is None:
        return 0
    node_types = cell_tags(genome, grid) if tags is None else tags
    eligible = [tuple(pos) for pos in node_types]
    if not eligible:
        return 0
    bounds = _spatial_domain(target)
    chosen = random.sample(eligible, min(len(eligible), len(specs)))
    wiring = wiring_chromosome(genome)
    mapped = list(wiring.genes)
    low_x, high_x, low_y, high_y = bounds
    for index, desired in enumerate(chosen):
        gene = copy.copy(mapped[index])
        gene.tag = _encode_spatial_coord(desired[0], low_x, high_x)
        gene.io_limit = 1
        gene.io_selector = _encode_spatial_coord(
            desired[1], low_y, high_y)
        mapped[index] = gene
    wiring.genes = mapped
    return len(chosen)


def seed_spatial_from_geometry(genome, grid, target):
    """Seed spatial anchors from the target's DECLARED I/O geometry.

    Random phenotype seeding starts every port at an arbitrary living site, so a
    fresh spatial population must rediscover from scratch where inputs and
    outputs belong -- the search-speed gap that leaves spatial trailing fixed at
    a normal generation budget (it only catches up given ~3x the generations).
    This instead initialises each port's normalised anchor at the same location
    fixed binding uses (``target.inputs`` for input ports, each output
    terminal's ``pos``), so a fresh genome already attaches every port to the
    living cell nearest its intended site. The anchors remain ordinary heritable
    alleles; evolution refines or abandons the prior freely.

    Coordinates use the target's stable canonical field rather than a
    particular random body's bounds. Consequently the input alleles can be
    applied before growth and reproduce the declared germline geometry exactly.
    Returns the number of ports initialised, or 0 when the map is incomplete.
    """
    specs = _port_specs(target)
    genes = _wiring_port_genes(genome, len(specs))
    if genes is None:
        return 0
    low_x, high_x, low_y, high_y = _spatial_domain(target)
    declared = ([tuple(pos) for pos in target.inputs]
                + [tuple(terminal.pos) for terminal in target.outputs])
    # Anchor each port at its raw declared coordinate and let the ordinary
    # nearest-free-cell resolution place it. Deliberately NOT forcing distinct
    # cells here: greedy distinct assignment over-constrains the initial
    # population and measurably hurt easy targets, whereas raw anchoring keeps
    # placement diversity and lets the viability tier + mutation resolve the
    # occasional two-ports-one-cell collision.
    wiring = wiring_chromosome(genome)
    mapped = list(wiring.genes)
    for index in range(len(specs)):
        anchor_x, anchor_y = declared[index]
        gene = copy.copy(mapped[index])
        gene.tag = _encode_spatial_coord(anchor_x, low_x, high_x)
        gene.io_limit = 1
        gene.io_selector = _encode_spatial_coord(anchor_y, low_y, high_y)
        mapped[index] = gene
    wiring.genes = mapped
    return len(specs)


def seed_spatial(genome, grid, target, tags=None, geometry_prob=0.75):
    """Initialise a fresh spatial port map for a factory genome.

    Input germlines always start from viable declared geometry. Most output
    anchors do too; a minority is scattered across the canonical field so
    selection sees alternate readouts without target-trace fitting or changing
    the initial body. ``grid`` and ``tags`` remain accepted for caller
    compatibility, but initialisation no longer needs a sacrificial phenotype.
    Returns the number of ports initialised.
    """
    count = seed_spatial_from_geometry(genome, grid, target)
    if not count:
        return 0
    genes = _wiring_port_genes(genome, len(_port_specs(target)))
    if genes is None or random.random() < geometry_prob:
        return count
    wiring = wiring_chromosome(genome)
    mapped = list(wiring.genes)
    for index in range(int(getattr(target, 'n_inputs', 0) or 0), len(genes)):
        gene = copy.copy(mapped[index])
        gene.tag = random.randrange(SPATIAL_COORD_MAX + 1)
        gene.io_selector = random.randrange(SPATIAL_COORD_MAX + 1)
        mapped[index] = gene
    wiring.genes = mapped
    return count


def set_spatial_port_positions(
        genome, positions, assignments, target=None):
    """Encode exact mature-body sites into a spatial port chromosome.

    ``assignments[i]`` is the desired physical cell for logical port ``i``
    (inputs first, then outputs in target order).  This is the deterministic
    counterpart of ``seed_spatial_from_phenotype`` used by plateau-rescue
    search: it creates an ordinary heritable genotype and never changes the
    evaluator or repairs a binding during evaluation.
    """
    positions = [tuple(pos) for pos in positions]
    assignments = [tuple(pos) for pos in assignments]
    genes = _wiring_port_genes(genome, len(assignments))
    # Ordinary spatial genomes use the stable target coordinate frame. The
    # optional target keeps backward compatibility for the inverse-grown LUT
    # compiler, whose verified witness predates developmental spatial inputs
    # and intentionally uses its mature phenotype bounds.
    bounds = (
        _spatial_domain(target) if target is not None
        else _spatial_bounds(positions))
    if genes is None or bounds is None:
        return 0
    eligible = set(positions)
    if (len(set(assignments)) != len(assignments)
            or any(pos not in eligible for pos in assignments)):
        return 0
    low_x, high_x, low_y, high_y = bounds
    wiring = wiring_chromosome(genome)
    mapped = list(wiring.genes)
    for index, desired in enumerate(assignments):
        gene = copy.copy(mapped[index])
        gene.tag = _encode_spatial_coord(desired[0], low_x, high_x)
        gene.io_limit = 1
        gene.io_selector = _encode_spatial_coord(
            desired[1], low_y, high_y)
        mapped[index] = gene
    wiring.genes = mapped
    return len(assignments)


def spatial_output_variants(genome, target, limit=48):
    """Ordinary-genome readout proposals for a stalled spatial run.

    The proposals never inspect expected behavior or simulated traces. They
    keep the complete body program and every developmental input allele fixed,
    then place one output allele at well-spread sites in the target's canonical
    field. Normal evaluation decides whether any proposal is useful. This gives
    selection a discrete readout neighbourhood without granting fixed I/O's
    target-aware trace fitting.
    """
    if io_strategy(target) != 'spatial_chromosome':
        return []
    limit = max(0, int(limit))
    n_inputs = int(getattr(target, 'n_inputs', 0) or 0)
    n_outputs = len(getattr(target, 'outputs', ()) or ())
    genes = _wiring_port_genes(genome, n_inputs + n_outputs)
    if not limit or genes is None or not n_outputs:
        return []

    low_x, high_x, low_y, high_y = _spatial_domain(target)
    remaining = {
        (x, y)
        for x in range(low_x, high_x + 1)
        for y in range(low_y, high_y + 1)}
    declared = [
        tuple(terminal.pos)
        for terminal in getattr(target, 'outputs', ()) or ()]
    ordered = []
    # Start with declared terminals and field corners, then greedily choose the
    # point farthest from those already covered. This gives a useful space-
    # filling prefix even when a small population admits only a few proposals.
    starters = declared + [
        (low_x, low_y), (low_x, high_y),
        (high_x, low_y), (high_x, high_y),
        ((low_x + high_x) // 2, (low_y + high_y) // 2)]
    for site in starters:
        if site in remaining:
            ordered.append(site)
            remaining.remove(site)
    while remaining:
        site = max(
            remaining,
            key=lambda pos: (
                min(
                    (pos[0] - chosen[0]) ** 2
                    + (pos[1] - chosen[1]) ** 2
                    for chosen in ordered),
                _site_rank(int(getattr(genome, 'tag', 0)), pos, len(ordered)),
            ))
        ordered.append(site)
        remaining.remove(site)

    variants, seen = [], set()
    for site in ordered:
        for output_index in range(n_outputs):
            port_index = n_inputs + output_index
            source = genes[port_index]
            encoded = (
                _encode_spatial_coord(site[0], low_x, high_x),
                _encode_spatial_coord(site[1], low_y, high_y))
            current = (_gene_tag(source), _gene_selector(source))
            if encoded == current:
                continue
            candidate = copy.deepcopy(genome)
            wiring = wiring_chromosome(candidate)
            mapped = list(wiring.genes)
            gene = copy.copy(mapped[port_index])
            gene.tag, gene.io_selector = encoded
            gene.io_limit = 1
            mapped[port_index] = gene
            wiring.genes = mapped
            signature = tuple(
                (_gene_tag(item), _gene_selector(item))
                for item in wiring.genes[:n_inputs + n_outputs])
            if signature in seen:
                continue
            seen.add(signature)
            variants.append(candidate)
            if len(variants) >= limit:
                return variants
    return variants


def spatial_routing_variants(genome, target, limit=48):
    """Heritable one-cell routing proposals for a stalled nervous organism.

    Development, germlines and port alleles stay fixed. Each proposal flips one
    bit of one mature cell's routing state through the post-development overlay,
    so selection can assemble a junction or veto locally instead of redrawing a
    many-cell developmental rule. Expected behavior is never inspected.
    """
    if io_strategy(target) != 'spatial_chromosome':
        return []
    if getattr(genome, 'arch', 'single') != 'single':
        return []
    limit = max(0, int(limit))
    if not limit:
        return []

    from .genome import MAX_ROUTING_PATCHES, RoutingPatch
    from .nervous import grow_nervous

    seeds = growth_seeds(target, 'spatial_chromosome', genome)
    grid = grow_nervous(
        genome, seeds=seeds,
        grid_size=getattr(target, 'grid_size', None),
        iters=getattr(target, 'iters', None))
    if not grid:
        return []
    bound = bind_io(genome, grid, target, 'spatial_chromosome')
    occupied_inputs = (
        set(flat_inputs(bound[0])) if bound is not None else set(seeds))
    sites = [site for site in grid if site not in occupied_inputs]
    sites.sort(key=lambda site: (
        _site_rank(int(getattr(genome, 'tag', 0)), site, len(sites)),
        site))

    inherited = list(getattr(genome, 'routing_patches', None) or ())
    inherited_by_site = {
        (int(patch.x), int(patch.y)): patch for patch in inherited}
    variants, seen = [], set()
    bits = (31).bit_length()
    for site in sites:
        old_state = int(grid[site])
        for bit in range(bits):
            new_state = old_state ^ (1 << bit)
            if not 0 < new_state < 32:
                continue
            candidate = copy.deepcopy(genome)
            patches = list(
                getattr(candidate, 'routing_patches', None) or ())
            replacement = RoutingPatch(site[0], site[1], new_state)
            if site in inherited_by_site:
                patches = [
                    replacement
                    if (int(patch.x), int(patch.y)) == site else patch
                    for patch in patches]
            elif len(patches) < MAX_ROUTING_PATCHES:
                patches.append(replacement)
            else:
                continue
            candidate.routing_patches = patches
            signature = tuple(
                (int(patch.x), int(patch.y), int(patch.state))
                for patch in patches)
            if signature in seen:
                continue
            seen.add(signature)
            variants.append(candidate)
            if len(variants) >= limit:
                return variants
    return variants


def _wiring_port_genes(genome, n_ports: int):
    """The first ``n_ports`` mapping genes, or None for an incomplete map."""
    chromosome = wiring_chromosome(genome)
    genes = list(getattr(chromosome, 'genes', ()) or ())
    if len(genes) < n_ports:
        return None
    return genes[:n_ports]


def _gene_tag(gene) -> int:
    """Desired node type on a wiring gene; expression priority on a body gene."""
    return int(getattr(gene, 'tag', 0)) if gene is not None else 0


def _gene_limit(gene) -> int:
    """Compatibility accessor: every logical port owns exactly one cell.

    ``io_limit`` remains present on gene/checkpoint records so older files load,
    but its historical values are intentionally ignored.
    """
    return 1


def _gene_selector(gene) -> int:
    return int(getattr(gene, 'io_selector', 0))


def mutate_io_allele(genome, node_type_max, strategy=None):
    """Mutate one expressed I/O allele in place.

    Type-wiring genomes mutate a mapping's type or selector. Spatial genomes
    mutate one axis of one normalised anchor. Tag-rank genomes mutate one body
    gene's expression priority. The wiring designation itself is structural and
    never migrates to a developmental chromosome.
    """
    if strategy == 'terminal_nodes':
        mutate_terminal_kind(genome)
        return

    wiring = wiring_chromosome(genome)
    if wiring is None:
        loci = [(chromosome, index)
                for chromosome in body_chromosomes(genome)
                for index in range(len(chromosome.genes))]
        if not loci:
            return
        chromosome, index = random.choice(loci)
        gene = copy.copy(chromosome.genes[index])
        old = _gene_tag(gene)
        step = random.choice((-4096, -1024, -257, -17, 17, 257, 1024, 4096))
        gene.tag = (old + step) % (IO_PRIORITY_MAX + 1)
        chromosome.genes[index] = gene
        return

    if not wiring.genes:
        return
    index = random.randrange(len(wiring.genes))
    gene = copy.copy(wiring.genes[index])
    if strategy == 'spatial_chromosome':
        field = random.choice(('tag', 'io_selector'))
        old = int(getattr(gene, field)) % (SPATIAL_COORD_MAX + 1)
        if random.random() < 0.85:
            step = random.choice(
                (-4096, -2048, -1024, 1024, 2048, 4096))
            value = max(0, min(SPATIAL_COORD_MAX, old + step))
            if value == old:
                value = max(0, min(SPATIAL_COORD_MAX, old - step))
        else:
            value = random.randrange(SPATIAL_COORD_MAX)
            if value >= old:
                value += 1
        setattr(gene, field, value)
        wiring.genes[index] = gene
        return

    field = random.choice(('tag', 'io_selector'))
    if field == 'tag':
        pool = body_type_pool(genome)
        old = _gene_tag(gene)
        candidates = list(dict.fromkeys(pool))
        candidates = [value for value in candidates if value != old]
        if candidates and random.random() < 0.7:
            gene.tag = random.choice(candidates)
        else:
            upper = max(2, int(node_type_max))
            gene.tag = 1 + ((old - 1 + random.randrange(1, upper - 1))
                            % (upper - 1))
    else:
        old = _gene_selector(gene)
        # The selector is an ordinal offset, so ±1 replaces only one edge of a
        # selected window. The old bit-flip/hash scheme globally reshuffled the
        # port and provided almost no heritable placement locality.
        gene.io_selector = old + random.choice((-1, 1))
    wiring.genes[index] = gene


def mutate_io_bundle(genome, node_type_max, strategy=None, count=2):
    """Make one coordinated, heritable edit to several I/O loci.

    A spatial bundle fully relocates ``count`` distinct port anchors, changing
    both coordinates together.  Single-axis/single-port edits cannot cross a
    co-adaptation valley where moving either an input or its consumer alone is
    harmful.  Other placement strategies receive several ordinary I/O edits;
    they have no direct per-port x/y representation to relocate.
    """
    count = max(1, int(count))
    wiring = wiring_chromosome(genome)
    if strategy == 'spatial_chromosome' and wiring is not None:
        genes = list(getattr(wiring, 'genes', ()) or ())
        if not genes:
            return 0
        indices = random.sample(
            range(len(genes)), min(count, len(genes)))
        mapped = list(genes)
        for index in indices:
            gene = copy.copy(mapped[index])
            gene.tag = random.randrange(SPATIAL_COORD_MAX + 1)
            gene.io_limit = 1
            gene.io_selector = random.randrange(SPATIAL_COORD_MAX + 1)
            mapped[index] = gene
        wiring.genes = mapped
        return len(indices)
    for _ in range(count):
        mutate_io_allele(genome, node_type_max, strategy=strategy)
    return count


def cell_tags(genome, grid) -> Dict[Pos, int]:
    """Nervous-net node types (kept under the historical helper name)."""
    return {pos: int(state) for pos, state in grid.items()}


def _entry_types(entry):
    values = (entry if isinstance(entry, (tuple, list, set, frozenset))
              else (entry,))
    return tuple(dict.fromkeys(int(value) for value in values if int(value) != 0))


def _cells_by_value(node_types) -> Dict[int, List[Pos]]:
    by_value: Dict[int, List[Pos]] = {}
    for pos in sorted(node_types):
        for value in _entry_types(node_types[pos]):
            by_value.setdefault(value, []).append(tuple(pos))
    return by_value


def _type_priorities(genome):
    """Highest body-gene expression priority for each producible node type."""
    priorities = {}
    for chromosome in body_chromosomes(genome):
        for gene in chromosome.genes:
            node_type = int(getattr(gene, 'self_out', 0))
            if node_type:
                priorities[node_type] = max(
                    priorities.get(node_type, 0), _gene_tag(gene))
    return priorities


def _mix64(value):
    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xbf58476d1ce4e5b9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94d049bb133111eb) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _site_rank(selector, pos, node_type=0):
    """Stable genotype-keyed spatial rank with no Python-hash randomisation."""
    x, y = map(int, pos)
    value = (int(selector) & _MASK64)
    value ^= ((x & 0xffffffff) << 32) | (y & 0xffffffff)
    value ^= (int(node_type) * 0x9e3779b97f4a7c15) & _MASK64
    return _mix64(value)


def _ordered_wiring_candidates(genome, node_types, wanted, claimed=()):
    """Stable unbiased base permutation for one requested node type.

    Selector values do not participate here: the selector rotates this order.
    Consequently a one-step selector mutation changes a limit-N attachment by
    at most one outgoing and one incoming site.
    """
    by_type = _cells_by_value(node_types)
    candidates = [pos for pos in by_type.get(int(wanted), ())
                  if pos not in claimed]
    priorities = _type_priorities(genome)
    seed = int(getattr(genome, 'tag', 0))
    candidates.sort(
        key=lambda pos: (
            _cell_priority(pos, node_types, priorities),
            _site_rank(seed, pos, wanted)),
        reverse=True)
    return candidates


def _rotate(values, offset):
    if not values:
        return []
    index = int(offset) % len(values)
    return values[index:] + values[:index]


def _cell_priority(pos, node_types, type_priorities):
    types = _entry_types(node_types[pos])
    return max((type_priorities.get(value, 0) for value in types), default=0)


def _port_specs(target):
    specs = [('in', index) for index in range(target.n_inputs)]
    specs.extend(('out', terminal.role) for terminal in target.outputs)
    return specs


def _resolve_tag_rank(genome, node_types, target):
    """Ports claim the highest-priority distinct physical nodes, in order."""
    priorities = _type_priorities(genome)
    selector = int(getattr(genome, 'tag', 0))
    ranked = sorted(
        node_types,
        key=lambda pos: (
            _cell_priority(pos, node_types, priorities),
            _site_rank(selector, pos,
                       max(_entry_types(node_types[pos]), default=0))),
        reverse=True)
    specs = _port_specs(target)
    if len(ranked) < len(specs):
        return None
    ports = []
    for (kind, identity), pos in zip(specs, ranked):
        types = _entry_types(node_types[pos])
        ports.append({
            'kind': kind,
            'index': identity if kind == 'in' else None,
            'role': identity if kind == 'out' else None,
            'tag': _cell_priority(pos, node_types, priorities),
            'type': types[0] if len(types) == 1 else types,
            'limit': 1,
            'selector': selector,
            'cells': [tuple(pos)],
        })
    return ports


def _resolve_wiring_partial(genome, node_types, target):
    specs = _port_specs(target)
    chromosome = wiring_chromosome(genome)
    genes = list(getattr(chromosome, 'genes', ()) or ())
    claimed = set()
    ports = []
    for port_index, (kind, identity) in enumerate(specs):
        if port_index >= len(genes):
            continue
        gene = genes[port_index]
        wanted = _gene_tag(gene)
        candidates = _ordered_wiring_candidates(
            genome, node_types, wanted, claimed)
        if not candidates:
            continue
        selector = _gene_selector(gene)
        candidates = _rotate(candidates, selector)
        limit = _gene_limit(gene)
        selected = candidates[:1]
        if not selected:
            continue
        claimed.update(selected)
        ports.append({
            'kind': kind,
            'index': identity if kind == 'in' else None,
            'role': identity if kind == 'out' else None,
            'tag': wanted,
            'type': wanted,
            'limit': limit,
            'selector': selector,
            'cells': list(selected),
        })
    return ports, len(specs)


def _resolve_wiring(genome, node_types, target):
    ports, total = _resolve_wiring_partial(genome, node_types, target)
    return ports if len(ports) == total else None


def _resolve_spatial_partial(genome, node_types, target):
    """Bind germline inputs exactly and outputs by spatial proximity."""
    specs = _port_specs(target)
    chromosome = wiring_chromosome(genome)
    genes = list(getattr(chromosome, 'genes', ()) or ())
    positions = [tuple(pos) for pos in node_types
                 if _entry_types(node_types[pos])]
    developmental = (
        getattr(genome, 'provenance', '') != 'truth-table-compiler-v1')
    bounds = (
        _spatial_domain(target) if developmental
        else _spatial_bounds(positions))
    if bounds is None:
        return [], len(specs)
    low_x, high_x, low_y, high_y = bounds
    germline_inputs = (
        spatial_input_sites(genome, target) if developmental else ())
    claimed = set()
    ports = []
    for port_index, (kind, identity) in enumerate(specs):
        if port_index >= len(genes):
            continue
        gene = genes[port_index]
        exact_input = (
            kind == 'in'
            and int(identity) < len(germline_inputs)
            and germline_inputs[int(identity)] in positions
            and germline_inputs[int(identity)] not in claimed)
        if exact_input:
            anchor = tuple(map(float, germline_inputs[int(identity)]))
        else:
            anchor = (
                _decode_spatial_coord(_gene_tag(gene), low_x, high_x),
                _decode_spatial_coord(
                    _gene_selector(gene), low_y, high_y),
            )
        candidates = [pos for pos in positions if pos not in claimed]
        if not candidates:
            continue
        selected = (
            germline_inputs[int(identity)] if exact_input else
            min(
                candidates,
                key=lambda pos: (
                    (float(pos[0]) - anchor[0]) ** 2
                    + (float(pos[1]) - anchor[1]) ** 2,
                    _site_rank(
                        int(getattr(genome, 'tag', 0)) ^ port_index,
                        pos, port_index + 1),
                )))
        claimed.add(selected)
        types = _entry_types(node_types[selected])
        ports.append({
            'kind': kind,
            'index': identity if kind == 'in' else None,
            'role': identity if kind == 'out' else None,
            'tag': _gene_tag(gene),
            'type': types[0] if len(types) == 1 else types,
            'limit': 1,
            'selector': _gene_selector(gene),
            'anchor': anchor,
            'cells': [selected],
        })
    return ports, len(specs)


def _resolve_spatial(genome, node_types, target):
    ports, total = _resolve_spatial_partial(genome, node_types, target)
    return ports if len(ports) == total else None


def _resolve_terminal_partial(genome, node_types, target):
    """Bind ports only to live cells whose terminal kind was gene-expressed.

    Candidate order is deterministic but genotype-keyed. It never uses target
    coordinates, expected traces, or fitness answers. Extra expressed terminal
    candidates remain ordinary body cells at runtime; exactly one cell is
    activated for each logical port.
    """
    live = {tuple(pos) for pos in node_types}
    expressed = getattr(genome, '_terminal_kinds', None) or {}
    candidates = {
        kind: sorted(
            (tuple(pos) for pos, value in expressed.items()
             if int(value) == kind and tuple(pos) in live),
            key=lambda pos: (
                _site_rank(
                    int(getattr(genome, 'tag', 0)) ^ kind,
                    pos, kind),
                pos))
        for kind in (IO_KIND_INPUT, IO_KIND_OUTPUT)
    }
    ports = []
    for index, selected in enumerate(
            candidates[IO_KIND_INPUT][:int(target.n_inputs)]):
        ports.append({
            'kind': 'in',
            'index': index,
            'role': None,
            'tag': IO_KIND_INPUT,
            'type': 'input_terminal',
            'limit': 1,
            'selector': 0,
            'anchor': None,
            'cells': [selected],
        })
    for terminal, selected in zip(
            target.outputs, candidates[IO_KIND_OUTPUT]):
        ports.append({
            'kind': 'out',
            'index': None,
            'role': terminal.role,
            'tag': IO_KIND_OUTPUT,
            'type': 'output_terminal',
            'limit': 1,
            'selector': 0,
            'anchor': None,
            'cells': [selected],
        })
    return ports, len(_port_specs(target))


def _resolve_terminal(genome, node_types, target):
    ports, total = _resolve_terminal_partial(genome, node_types, target)
    return ports if len(ports) == total else None


def binding_progress(genome, grid, target, tags=None):
    """Return ``(successfully_bound_ports, total_ports)`` without target scoring."""
    specs = _port_specs(target)
    total = len(specs)
    if not total:
        return 0, 0
    strategy = io_strategy(target)
    if strategy == 'fixed':
        return total, total
    node_types = cell_tags(genome, grid) if tags is None else tags
    if strategy == 'terminal_nodes':
        ports, _ = _resolve_terminal_partial(genome, node_types, target)
        return len(ports), total
    if strategy == 'tag_rank':
        return min(len(node_types), total), total
    if strategy == 'spatial_chromosome':
        ports, _ = _resolve_spatial_partial(genome, node_types, target)
        return len(ports), total
    ports, _ = _resolve_wiring_partial(genome, node_types, target)
    return len(ports), total


def record_binding_progress(genome, progress):
    """Attach an evaluation-only viability diagnostic to a local genome copy."""
    count, total = progress
    genome._io_binding_progress = (max(0, int(count)), max(0, int(total)))
    return genome._io_binding_progress


def binding_viability(genome):
    """Selection tier in [0,1]; unevaluated/legacy genomes remain neutral."""
    progress = getattr(genome, '_io_binding_progress', None)
    if progress is None:
        return 1.0
    count, total = progress
    return (float(count) / float(total)) if total else 1.0


def _resolve_ports(genome, grid, target, strategy, node_types):
    if not grid or not _port_specs(target):
        return None
    node_types = cell_tags(genome, grid) if node_types is None else node_types
    if strategy == 'terminal_nodes':
        return _resolve_terminal(genome, node_types, target)
    if strategy == 'tag_rank':
        return _resolve_tag_rank(genome, node_types, target)
    if strategy == 'wiring_chromosome':
        return _resolve_wiring(genome, node_types, target)
    if strategy == 'spatial_chromosome':
        return _resolve_spatial(genome, node_types, target)
    raise ValueError('unknown io_placement strategy: %r' % (strategy,))


def bind_io(genome, grid, target, strategy=None, tags=None):
    """Return ``(input_groups, output_groups)`` for an evolvable strategy.

    ``tags`` is retained as the compatibility keyword for the backend's mature
    node-type map.  Both sides are grouped: inputs fan a stimulus out to every
    selected site; outputs form a deterministic wired-OR readout bus.
    """
    strategy = strategy or io_strategy(target)
    if strategy == 'fixed':
        return None
    ports = _resolve_ports(genome, grid, target, strategy, tags)
    if ports is None:
        return None
    inputs = [list(port['cells']) for port in ports if port['kind'] == 'in']
    outputs = {port['role']: list(port['cells'])
               for port in ports if port['kind'] == 'out'}
    return inputs, outputs


def binding_report(genome, grid, target, tags=None):
    strategy = io_strategy(target)
    if strategy == 'fixed':
        return None
    ports = _resolve_ports(genome, grid, target, strategy, tags)
    if ports is None:
        return None
    return [{
        'port': ('in[%d]' % port['index'] if port['kind'] == 'in'
                 else 'out[%s]' % port['role']),
        'kind': port['kind'],
        'tag': port['tag'],
        'type': port['type'],
        'limit': port['limit'],
        'selector': port['selector'],
        'anchor': port.get('anchor'),
        'cells': list(port['cells']),
    } for port in ports]


def _groups(entries):
    groups = []
    for entry in entries:
        if entry is None:
            groups.append([])
        elif (isinstance(entry, (tuple, list)) and len(entry) == 2
              and all(isinstance(value, (int, float)) for value in entry)):
            groups.append([tuple(entry)])
        else:
            groups.append([tuple(cell) for cell in entry])
    return groups


def input_groups(in_pos):
    return _groups(in_pos)


def output_groups(out_pos):
    return {role: cells for role, cells in
            zip(out_pos, _groups(out_pos.values()))}


def _flat(groups):
    seen, cells = set(), []
    for group in groups:
        for cell in group:
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
    return cells


def flat_inputs(in_pos):
    return _flat(input_groups(in_pos))


def flat_outputs(out_pos):
    return _flat(output_groups(out_pos).values())


def terminal_node_sets(target, in_pos, out_pos):
    """Return ``(source_cells, sink_cells)`` for terminal-node placement.

    Other strategies return two empty sets, keeping every legacy simulation
    byte-for-byte on its original path.
    """
    if io_strategy(target) != 'terminal_nodes':
        return set(), set()
    sources = set(flat_inputs(in_pos))
    sinks = set(flat_outputs(out_pos))
    if sources & sinks:
        raise ValueError('terminal input and output nodes must be distinct')
    return sources, sinks


def merge_intervals(sequences):
    """Union several cells' high intervals into one wired-OR bus waveform."""
    intervals = sorted((float(start), float(end))
                       for sequence in sequences for start, end in sequence
                       if float(end) > float(start))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def describe_binding(genome, grid, target, tags=None):
    strategy = io_strategy(target)
    if strategy == 'fixed':
        return 'io_placement=fixed (geometric seeds + fitted outputs)'
    report = binding_report(genome, grid, target, tags=tags)
    if report is None:
        count, total = binding_progress(
            genome, grid, target, tags=tags)
        reason = (
            'too few gene-expressed input/output terminal cells'
            if strategy == 'terminal_nodes'
            else 'incomplete map or too few exclusive physical sites'
            if strategy == 'spatial_chromosome'
            else 'incomplete map, absent type, or too few exclusive physical sites')
        return ('io_placement=%s (UNBINDABLE: %s; %d/%d ports bound)'
                % (strategy, reason, count, total))
    lines = ['io_placement=%s' % strategy]
    for entry in report:
        if strategy == 'terminal_nodes':
            detail = ('genome-expressed source-only input terminal'
                      if entry['kind'] == 'in'
                      else 'genome-expressed sink-only output terminal')
        elif strategy == 'tag_rank':
            detail = 'priority %d' % entry['tag']
        elif strategy == 'spatial_chromosome':
            detail = ('anchor (%.2f, %.2f)'
                      % tuple(entry['anchor']))
        else:
            detail = ('type %s, selector %d'
                      % (entry['type'], entry['selector']))
        lines.append('  %s <- %s @ %s'
                     % (entry['port'], detail,
                        ', '.join(str(cell) for cell in entry['cells'])))
    return '\n'.join(lines)
