"""
substrates/lut/branched_ga.py - construction and variation for the branched LUT
encoding.

Mirrors substrates/nervous/branched_ga.py. The one substantive difference is
what a rule may install: ``self_out`` is assembled from the run's ENABLED
FUNCTION BANKS, so evolution picks named gates rather than arbitrary 16-bit
tables. That keeps the alphabet finite and the tolerance metric meaningful.

Every collection feeding a random draw is sorted before use - the FNV hash-order
bug is the reason, and `observed_contexts` is where it would reappear.
"""
from __future__ import annotations

import copy
import random

from .branched import (
    DEPTH_ANY, DEPTH_BANDS, EMPTY_CELL, OUT_CELL, PAD_CELL,
    BranchedLutChromosome, BranchedLutGenome, LutContextGene, LutControlGene,
    LutInputGene, LutOutputGene, _band_of, _cell_of, bearing_cell, catalogue,
    LutIoChromosome, arm_reach, develop_branched_lut, driven_roots,
    growth_candidates, materialise_pads, neighbours, output_root_sites,
    required_output_directions, DIRECTIONS)
from .functions import normalise_function_families

MAX_TELOMERE = 32
#: An arm's match tolerance is normally 1: enough for a rule to fire on a gate
#: that computes something different through the same interface (distance 0) or
#: one wire away from what it saw, and no more. Pads, unbuilt output niches and
#: empty ground are SENTINEL_DISTANCE apart, so no tolerance in this range can
#: confuse the interface with the body.
MAX_TOLERANCE = 3
MAX_IO_DISTANCE = 6
ERASE_PROBABILITY = 0.08
#: How many of a cell's four output directions a fresh rule tends to drive. Most
#: useful cells drive one or two; filling all four makes a body of blobs.
DRIVE_WEIGHTS = (1, 1, 1, 2, 2, 3)


def input_pads(genome):
    """Pad cells, with pad zero pinned to the origin as the coordinate gauge."""
    pads = [(0, 0)]
    for gene in genome.inputs:
        cell = bearing_cell(gene.bearing, gene.distance, pads)
        if cell is not None:
            pads.append(cell)
    return tuple(pads)


def random_cell(families, allow_empty=True, required=()):
    """A cell assembled from named gates in the enabled banks.

    ``required`` are directions the cell must drive to feed the limb it joins
    (see required_output_directions), so the draw comes from parts that can
    actually fire here - the LUT analogue of FNV's ``_compatible_component_id``.
    """
    if allow_empty and random.random() < ERASE_PROBABILITY:
        return EMPTY_CELL
    entries = catalogue(families)
    if not entries:
        return EMPTY_CELL
    tables = [0, 0, 0, 0]
    driven = set(random.sample(range(4), random.choice(DRIVE_WEIGHTS)))
    wanted = [d for d in required if d in DIRECTIONS]
    if wanted:
        driven.add(DIRECTIONS.index(random.choice(sorted(wanted))))
    for index in driven:
        tables[index] = random.choice(entries)[1]
    return tuple(tables)


def observed_contexts(genome, label, trace=None, pads=None):
    """Sorted [(context, band)] the arm can currently act on.

    ``trace`` and ``pads`` are an optional ALREADY-DEVELOPED organism: growing
    one costs far more than evaluating it, so a caller that has just developed
    this genome passes the result in instead of paying for an identical second
    development.

    Sorted, never a raw set: contexts contain the sentinel strings PAD/OUT, and
    iterating a set keyed by strings is hash-order dependent per process.
    """
    pads = set(input_pads(genome) if pads is None else pads)
    roots = output_root_sites(genome, pads)
    if trace is None:
        trace = develop_branched_lut(genome, pads)
    grid, owners, depths = trace.grid, trace.owners, trace.depths
    output_sites = set(roots.values()) - pads

    # EXACTLY the sites development would let this arm write, under the same
    # arm_reach scoping. A wider neighbourhood manufactures rules that can
    # never fire.
    found = set()
    root = roots.get(label)
    for cell in growth_candidates(grid, owners, pads, output_sites):
        # The arm's own unbuilt root is reachable by definition - its root rule
        # is exempt from arm_reach in development, so filtering it here would
        # leave every arm unable to observe the context it needs to start.
        if cell == root and cell not in grid:
            depth = 0
        else:
            depth = arm_reach(cell, label, owners, depths)
            if depth is None:
                continue
        around = neighbours(cell)
        context = tuple(_cell_of(around[d], grid, pads, output_sites)
                        for d in DIRECTIONS) + (
                            _cell_of(cell, grid, pads, output_sites),)
        found.add((context, min(depth, DEPTH_BANDS - 1),
                   required_output_directions(cell, label, depth,
                                              owners, depths)))
    return sorted(found, key=repr)


def random_gene(genome, gene_id, label, *, allow_output=True,
                trace=None, pads=None):
    families = genome.families or None
    seen = observed_contexts(genome, label, trace=trace, pads=pads)
    roots = [entry for entry in seen if entry[0][4] == OUT_CELL]
    if allow_output and roots:
        context, _band, directions = random.choice(roots)
        return LutContextGene(
            gene_id, context[0], context[1], context[2], context[3], OUT_CELL,
            random_cell(families, allow_empty=False, required=directions),
            label, DEPTH_ANY)
    body = [entry for entry in seen if entry[0][4] != OUT_CELL]
    if not body:
        return None
    context, band, directions = random.choice(body)
    depth = band if random.random() < 0.5 else DEPTH_ANY
    return LutContextGene(
        gene_id, context[0], context[1], context[2], context[3], context[4],
        random_cell(families, required=directions), label, depth)


def _arm_has_root(genome, label):
    members, _control = genome.arm(label)
    return any(gene.spawns_output() for gene in (members or ()))


def random_branched_lut_genome(n_chroms=2, n_inputs=2, output_roles=('Q',),
                               families=None, blocks=24,
                               max_telomere=MAX_TELOMERE):
    """A fresh genome whose arms build something, from the enabled banks only."""
    roles = tuple(str(role) for role in output_roles)
    if len(roles) > 2 * int(n_chroms):
        raise ValueError('a branched LUT genome needs one arm per output role')
    enabled = normalise_function_families(families)
    genome = BranchedLutGenome(
        chromosomes=[
            BranchedLutChromosome(controls=[
                LutControlGene(tolerance=random.choice((0, 1, 1, 2)),
                               telomere=random.randint(4, max_telomere))
                for _ in range(2)])
            for _ in range(int(n_chroms))],
        # One dedicated arm per output role, in order: role k gets arm k+1.
        # Leftover arms stay empty - spare capacity, not spare outputs.
        io_chromosome=LutIoChromosome(
            inputs=[LutInputGene(bearing=random.randrange(8),
                                 distance=random.randint(1, 3))
                    for _ in range(max(0, int(n_inputs) - 1))],
            outputs=[LutOutputGene(role=role, bearing=random.randrange(8),
                                   distance=random.randint(1, MAX_IO_DISTANCE),
                                   branch_id=index + 1)
                     for index, role in enumerate(roles)]),
        families=tuple(enabled),
        next_gene_id=1)

    # Development is the expensive step - one growth costs many times what
    # evaluating the grown organism costs - so this loop performs exactly ONE
    # per attempted gene: the pads are fixed for the whole construction, and the
    # accepted organism carries forward instead of being re-grown to inspect it.
    pads = input_pads(genome)
    trace = develop_branched_lut(genome, pads)
    growing = {gene.branch_id for gene in genome.outputs}
    for _ in range(max(1, int(blocks))):
        if not growing:
            break
        order = sorted(growing)
        random.shuffle(order)
        for label in order:
            if label not in growing:
                continue
            chromosome = genome.chromosomes[(label - 1) // 2]
            placed = False
            for _attempt in range(6):
                gene = random_gene(genome, genome.next_gene_id, label,
                                   allow_output=not _arm_has_root(genome, label),
                                   trace=trace, pads=pads)
                if gene is None:
                    break
                chromosome.genes.append(gene)
                grown = develop_branched_lut(genome, pads)
                if grown.grid != trace.grid:
                    genome.next_gene_id += 1
                    trace = grown          # keep the organism it just built
                    placed = True
                    break
                chromosome.genes.pop()
            if not placed:
                growing.discard(label)
    return genome


#: How many random starts to look at before picking one. Same mechanism, and
#: same reason, as the hex port and as FNV.
DEVELOPMENTAL_SEED_CANDIDATES = 6


def select_developmental_seed(make_genome,
                              attempts=DEVELOPMENTAL_SEED_CANDIDATES):
    """Pick the most connected of a few random starts. Target-blind.

    An organism whose output root nothing can drive scores the silent baseline
    whatever else is true of it, and on this substrate almost all random starts
    are that: measured over 40 fresh genomes, only 3 of the 25 viable ones could
    drive an output at all. A population of those is undifferentiated rather
    than merely weak - selection cannot tell its members apart.

    Looks only at whether inputs can reach outputs, never at the target, so it
    moves where search STARTS without changing what counts as fitness.
    """
    def key(genome):
        pads = input_pads(genome)
        grid = materialise_pads(
            develop_branched_lut(genome, pads).grid, pads)
        roots = output_root_sites(genome, pads)
        return (len(driven_roots(grid, pads, roots)), len(grid),
                -sum(len(c.genes) for c in genome.chromosomes))

    best, best_key = None, None
    for _ in range(max(1, int(attempts))):
        candidate = make_genome()
        score = key(candidate)
        if best_key is None or score > best_key:
            best, best_key = candidate, score
        if score[0] >= len(candidate.outputs):
            break                      # every role is driven; stop paying
    return best


# -- variation ------------------------------------------------------------------

def clone_branched_lut(genome):
    return copy.deepcopy(genome)


def _mutate_context_gene(gene, genome, label):
    families = genome.families or None
    choice = random.random()
    if choice < 0.35:
        gene.self_out = random_cell(families)
    elif choice < 0.55 and not gene.spawns_output():
        body = [entry for entry in observed_contexts(genome, label)
                if entry[0][4] != OUT_CELL]
        if body:
            context, _band, _directions = random.choice(body)
            (gene.ctx_n, gene.ctx_s, gene.ctx_e, gene.ctx_w,
             gene.self_in) = context
    elif choice < 0.75:
        gene.depth = (DEPTH_ANY if gene.depth != DEPTH_ANY
                      else random.randrange(DEPTH_BANDS))
    elif not gene.spawns_output():
        field = random.choice(('ctx_n', 'ctx_s', 'ctx_e', 'ctx_w'))
        setattr(gene, field, random.choice(
            (EMPTY_CELL, PAD_CELL, random_cell(families, allow_empty=False))))
    return gene


def mutate_branched_lut(genome, mean_mutations=2.0, max_telomere=MAX_TELOMERE):
    child = clone_branched_lut(genome)
    limit = max(1, int(random.expovariate(1.0 / max(0.5, mean_mutations)) + 1))
    for _ in range(limit):
        roll = random.random()
        labels = [gene.branch_id for gene in child.outputs] or [1]
        label = random.choice(labels)
        index = (label - 1) // 2
        if index >= len(child.chromosomes):
            continue
        chromosome = child.chromosomes[index]
        members = [gene for gene in chromosome.genes
                   if int(gene.branch_id) == label]
        if roll < 0.45 and members:
            _mutate_context_gene(random.choice(members), child, label)
        elif roll < 0.60:
            gene = random_gene(child, child.next_gene_id, label,
                               allow_output=not _arm_has_root(child, label))
            if gene is not None:
                chromosome.genes.append(gene)
                child.next_gene_id += 1
        elif roll < 0.70 and members:
            chromosome.genes.remove(random.choice(members))
        elif roll < 0.85:
            control = chromosome.controls[(label - 1) % 2]
            if random.random() < 0.5:
                control.telomere = max(1, min(
                    max_telomere,
                    control.telomere + random.choice((-4, -1, 1, 4))))
            else:
                control.tolerance = max(0, min(
                    MAX_TOLERANCE,
                    control.tolerance + random.choice((-1, 1))))
        else:
            pool = list(child.inputs) + list(child.outputs)
            if pool:
                allele = random.choice(pool)
                if random.random() < 0.5:
                    allele.bearing = (int(allele.bearing)
                                      + random.choice((-1, 1))) % 8
                else:
                    allele.distance = max(1, min(
                        MAX_IO_DISTANCE,
                        int(allele.distance) + random.choice((-1, 1))))
    return child


def crossover_branched_lut(left, right):
    """Swap whole arms - the unit that owns a territory and a lifespan."""
    child = clone_branched_lut(left)
    for index, chromosome in enumerate(child.chromosomes):
        if index >= len(right.chromosomes) or random.random() < 0.5:
            continue
        donor = right.chromosomes[index]
        half = random.randrange(2)
        label = 2 * index + half + 1
        chromosome.genes = [gene for gene in chromosome.genes
                            if int(gene.branch_id) != label]
        chromosome.genes.extend(
            copy.deepcopy(gene) for gene in donor.genes
            if int(gene.branch_id) == label)
        if half < len(donor.controls):
            chromosome.controls[half] = copy.deepcopy(donor.controls[half])
    child.next_gene_id = max(
        [int(left.next_gene_id), int(right.next_gene_id)]
        + [int(gene.gene_id) + 1
           for chromosome in child.chromosomes for gene in chromosome.genes])
    return child


def prepare_branched_lut(genome, target):
    """(grid, input_pos, output_pos) or None when the organism is incomplete."""
    pads = input_pads(genome)
    n_inputs = int(getattr(target, 'n_inputs', len(pads)))
    if len(pads) < n_inputs:
        return None
    body = develop_branched_lut(genome, pads).grid
    if not body:
        return None
    grid = materialise_pads(body, pads)
    roots = output_root_sites(genome, pads)
    outputs = {}
    for gene in genome.outputs:
        cell = roots.get(int(gene.branch_id))
        if cell is None or cell not in grid:
            return None
        outputs[str(gene.role)] = cell
    if len(set(outputs.values())) != len(outputs):
        return None
    return grid, list(pads[:n_inputs]), outputs


def prepare_branched_lut_net(genome, target):
    """Grow a branched genome and hand it to the existing LUT evaluator.

    Returns ``prepare_lut``'s contract - (grid, out_pos, traces, in_pos) - or
    None for an unusable organism, so scoring, certification and the batch
    evaluator all work unchanged.

    Outputs are the arm ROOTS, traced where they sit, never probes fitted to the
    grown body afterwards. That is what makes this an output-rooted encoding:
    the arm has to know where it is growing from before it grows.
    """
    from .ga import trace_fixed_outputs

    prepared = prepare_branched_lut(genome, target)
    if prepared is None:
        return None
    grid, in_pos, out_pos = prepared
    if len(grid) <= len(in_pos):
        return None
    traces = trace_fixed_outputs(grid, in_pos, out_pos, target,
                                 source_nodes=set(in_pos))
    if traces is None:
        return None
    return grid, out_pos, traces, list(in_pos)
