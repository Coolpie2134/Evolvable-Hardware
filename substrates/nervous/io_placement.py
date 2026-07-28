"""Heritable input/output placement shared by all three substrates.

The legacy ``fixed`` mode grows from geometrically declared input pads and fits
outputs after simulation.  The evolvable modes instead grow from one neutral
centre seed and bind ports to the mature body without looking at target traces:

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
    Chromosome three is again a dedicated, non-developmental port map, but each
    port gene stores a normalised ``(x, y)`` anchor instead of a node type.
    After growth the port attaches to the nearest still-free living cell.
    Coordinates mutate locally and a changed body merely slides an attachment
    to its nearest survivor; it cannot invalidate the map by losing a type.

Every physical cell is exclusive to one logical port.  This is a semantic
guardrail, not a geometric one: it prevents an input and output from becoming
the same wire (the shortcut that motivated this rewrite) while leaving evolution
free to place distinct ports beside one another when that is useful.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

Pos = Tuple[int, int]

IO_STRATEGIES = (
    'fixed', 'tag_rank', 'wiring_chromosome', 'spatial_chromosome')
_STRATEGY_ALIASES = {'sex_chromosome': 'wiring_chromosome'}

WIRING_CHROMOSOME_INDEX = 2
IO_PRIORITY_MAX = 65535
SPATIAL_COORD_MAX = 65535
_MASK64 = (1 << 64) - 1


def io_strategy(target) -> str:
    """Return the target's validated I/O strategy (legacy targets are fixed)."""
    strategy = getattr(target, 'io_placement', 'fixed') or 'fixed'
    strategy = _STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in IO_STRATEGIES:
        raise ValueError('unknown io_placement strategy: %r' % (strategy,))
    return strategy


def growth_seeds(target, strategy=None):
    """Pads for fixed binding; one neutral centre seed for evolvable binding."""
    strategy = strategy or io_strategy(target)
    if strategy == 'fixed':
        return tuple(target.inputs)
    size = int(getattr(target, 'grid_size', 5) or 5)
    return ((size // 2, size // 2),)


def uses_port_chromosome(strategy) -> bool:
    """Whether ``strategy`` reserves chromosome three as an I/O map."""
    return strategy in ('wiring_chromosome', 'spatial_chromosome')


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
    bounds = _spatial_bounds(eligible)
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
    """Attach each encoded anchor to its nearest unclaimed living cell."""
    specs = _port_specs(target)
    chromosome = wiring_chromosome(genome)
    genes = list(getattr(chromosome, 'genes', ()) or ())
    positions = [tuple(pos) for pos in node_types
                 if _entry_types(node_types[pos])]
    bounds = _spatial_bounds(positions)
    if bounds is None:
        return [], len(specs)
    low_x, high_x, low_y, high_y = bounds
    claimed = set()
    ports = []
    for port_index, (kind, identity) in enumerate(specs):
        if port_index >= len(genes):
            continue
        gene = genes[port_index]
        anchor = (
            _decode_spatial_coord(_gene_tag(gene), low_x, high_x),
            _decode_spatial_coord(
                _gene_selector(gene), low_y, high_y),
        )
        candidates = [pos for pos in positions if pos not in claimed]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda pos: (
                (float(pos[0]) - anchor[0]) ** 2
                + (float(pos[1]) - anchor[1]) ** 2,
                _site_rank(
                    int(getattr(genome, 'tag', 0)) ^ port_index,
                    pos, port_index + 1),
            ))
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
            'incomplete map or too few exclusive physical sites'
            if strategy == 'spatial_chromosome'
            else 'incomplete map, absent type, or too few exclusive physical sites')
        return ('io_placement=%s (UNBINDABLE: %s; %d/%d ports bound)'
                % (strategy, reason, count, total))
    lines = ['io_placement=%s' % strategy]
    for entry in report:
        if strategy == 'tag_rank':
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
