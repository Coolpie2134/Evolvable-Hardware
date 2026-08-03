"""Dependency-addressed physical construction for fresh FNV genomes.

This is deliberately not inverse development.  A genome names only existing
source/output ports, fixed catalogue components, and their dependencies.  It
never receives a desired truth table, output role, or target coordinate.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from substrates.nervous.hexgrid import hex_dirs

from .catalogue import BY_ID, COMPONENTS, DIRECTIONS, normalise_families
from .genome import (
    BranchRef, MAX_GENES, MAX_PLACEMENTS, PlacementGene, input_node_id,
    input_seed_grid,
)


@dataclass(frozen=True)
class ConstructionTrace:
    grid: dict
    coordinates: dict
    active_ids: frozenset
    dormant_ids: frozenset
    depths: dict
    snapshots: tuple


def _genes(genome):
    return tuple(
        gene
        for chromosome in genome.chromosomes
        for gene in chromosome.genes
        if isinstance(gene, PlacementGene))


def _source_drives(gene_by_id, ref):
    if ref.direction not in DIRECTIONS:
        return False
    if ref.node_id < 0:
        return True
    source = gene_by_id.get(int(ref.node_id))
    return bool(source is not None and ref.direction in
                BY_ID.get(int(source.component_id), BY_ID[0]).outputs)


def _placement_target(gene, coordinates, gene_by_id):
    entry = BY_ID.get(int(gene.component_id))
    refs = tuple(gene.inputs)
    if entry is None or entry.id == 0 or len(refs) != len(entry.inputs):
        return None
    targets = []
    for receiver_direction, ref in zip(entry.inputs, refs):
        source = coordinates.get(int(ref.node_id))
        if source is None or not _source_drives(gene_by_id, ref):
            return None
        target = hex_dirs(*source).get(ref.direction)
        if target is None:
            return None
        # The named source must physically face the declared receiver input.
        if hex_dirs(*target).get(receiver_direction) != source:
            return None
        targets.append(target)
    if not targets or any(target != targets[0] for target in targets[1:]):
        return None
    return targets[0]


def develop_constructive(genome, seeds, *, snapshots=False):
    """Resolve placements by dependency, with strictly local collision loss.

    Ready genes are considered by stable ID. A collision suppresses only that
    placement; an earlier occupant remains and unrelated branches continue.
    Descendants of a suppressed/unresolved gene stay dormant because their
    explicitly named source port never comes into existence.
    """
    seeds = tuple(tuple(seed) for seed in seeds)
    seed_states = input_seed_grid(seeds)
    grid = dict(seed_states)
    coordinates = {
        input_node_id(index): cell for index, cell in enumerate(seeds)}
    depths = {input_node_id(index): 0 for index in range(len(seeds))}
    occupied = {cell: input_node_id(index)
                for index, cell in enumerate(seeds)}
    ordered = sorted(_genes(genome), key=lambda gene: int(gene.gene_id))
    gene_by_id = {}
    duplicate_ids = set()
    for gene in ordered:
        gene_id = int(gene.gene_id)
        if gene_id in gene_by_id:
            duplicate_ids.add(gene_id)
        else:
            gene_by_id[gene_id] = gene
    pending = {
        gene_id for gene_id in gene_by_id if gene_id not in duplicate_ids}
    active, failed = set(), set(duplicate_ids)
    frames = [dict(grid)] if snapshots else []
    while pending:
        ready = [
            gene_by_id[gene_id] for gene_id in sorted(pending)
            if all(int(ref.node_id) in coordinates
                   for ref in gene_by_id[gene_id].inputs)
        ]
        if not ready:
            break
        changed = False
        for gene in ready:
            gene_id = int(gene.gene_id)
            pending.discard(gene_id)
            target = _placement_target(gene, coordinates, gene_by_id)
            if target is None or target in occupied:
                failed.add(gene_id)
                continue
            grid[target] = int(gene.component_id)
            occupied[target] = gene_id
            coordinates[gene_id] = target
            depths[gene_id] = 1 + max(
                depths[int(ref.node_id)] for ref in gene.inputs)
            active.add(gene_id)
            changed = True
        if snapshots and changed:
            frames.append(dict(grid))
    dormant = set(gene_by_id) - active
    if snapshots and (not frames or frames[-1] != grid):
        frames.append(dict(grid))
    return ConstructionTrace(
        grid, coordinates, frozenset(active), frozenset(dormant), depths,
        tuple(frames))


def constructive_depth(genome, seeds):
    trace = develop_constructive(genome, seeds)
    return max(trace.depths.values(), default=0)


def source_ancestry(genome, seeds):
    """Live node ID -> logical source-pad IDs in its dependency cone.

    This is structural provenance, not simulated behavior: it follows only the
    named ``BranchRef`` graph.  It lets generic morphogenesis distinguish a
    real two-input convergence from joining two fan-outs of the same input.
    """
    trace = develop_constructive(genome, seeds)
    genes = {int(gene.gene_id): gene for gene in _genes(genome)}
    ancestry = {
        int(node_id): frozenset((int(node_id),))
        for node_id in trace.coordinates if int(node_id) < 0}
    pending = set(trace.active_ids)
    while pending:
        ready = [
            gene_id for gene_id in sorted(pending)
            if all(int(ref.node_id) in ancestry
                   for ref in genes[gene_id].inputs)]
        if not ready:
            break
        for gene_id in ready:
            ancestry[gene_id] = frozenset().union(*(
                ancestry[int(ref.node_id)]
                for ref in genes[gene_id].inputs))
            pending.discard(gene_id)
    return ancestry


def _receiver_direction(target, source):
    return next((direction for direction, cell in hex_dirs(*target).items()
                 if cell == source), None)


def frontier_references(genome, seeds):
    """Empty sites mapped to the named live output ports that drive them."""
    trace = develop_constructive(genome, seeds)
    genes = {int(gene.gene_id): gene for gene in _genes(genome)}
    driven = {}
    for owner_id, source in trace.coordinates.items():
        outputs = (DIRECTIONS if owner_id < 0 else
                   BY_ID[int(genes[owner_id].component_id)].outputs)
        for output_direction in outputs:
            target = hex_dirs(*source)[output_direction]
            if target in trace.grid:
                continue
            receiver = _receiver_direction(target, source)
            if receiver is not None:
                driven.setdefault(target, {})[receiver] = BranchRef(
                    int(owner_id), output_direction)
    return trace, driven


def feasible_placements(genome, seeds, families, *, component_id=None,
                        target_parity=None, input_count=None):
    """All target-blind physical components attachable to current live tips."""
    enabled = normalise_families(families)
    _trace, driven = frontier_references(genome, seeds)
    choices = []
    for target, by_input in driven.items():
        if target_parity is not None and sum(target) % 2 != target_parity:
            continue
        for entry in COMPONENTS:
            if (entry.id == 0 or entry.family not in enabled
                    or (component_id is not None
                        and entry.id != int(component_id))
                    or (input_count is not None
                        and len(entry.inputs) != int(input_count))):
                continue
            if all(direction in by_input for direction in entry.inputs):
                choices.append((
                    target, entry.id,
                    tuple(by_input[direction] for direction in entry.inputs)))
    return choices


def _choose_family_first(choices):
    families = sorted({BY_ID[item[1]].family for item in choices})
    family = random.choice(families)
    within = [item for item in choices if BY_ID[item[1]].family == family]
    # Within unary routing, physical fan-out creates useful hereditary branch
    # tips. It is a bias among enabled real components, not a virtual fork.
    fanout = [item for item in within if len(BY_ID[item[1]].outputs) == 2]
    if fanout and random.random() < 0.25:
        within = fanout
    return random.choice(within)


def append_placement_choice(genome, choice, *, chromosome=None):
    """Append one already-validated physical frontier placement.

    Keeping this operation separate from candidate selection lets plateau
    rescue enumerate real local joins instead of hoping a random mutation
    samples the one useful facing-port combination.  ``choice`` still comes
    exclusively from :func:`feasible_placements`; there is no target, truth
    table, fitted probe, or desired coordinate in this operation.
    """
    if len(_genes(genome)) >= MAX_PLACEMENTS:
        return False
    _target, component_id, refs = choice
    entry = BY_ID.get(int(component_id))
    if entry is None or entry.id == 0 or len(refs) != len(entry.inputs):
        return False
    by_id = {int(gene.gene_id): gene for gene in _genes(genome)}
    if any(ref.node_id >= 0 and ref.node_id not in by_id for ref in refs):
        return False
    available = [item for item in genome.chromosomes
                 if len(item.genes) < MAX_GENES]
    if not available:
        return False
    gene_id = int(genome.next_gene_id)
    genome.next_gene_id = gene_id + 1
    source_blocks = [
        ref.node_id if ref.node_id < 0 else by_id[ref.node_id].branch_id
        for ref in refs]
    branch_id = (
        (gene_id if refs[0].node_id < 0 else int(source_blocks[0]))
        if len(refs) == 1 else gene_id)
    gene = PlacementGene(gene_id, int(component_id), tuple(refs), branch_id)
    destination = chromosome if chromosome in available else random.choice(
        available)
    destination.genes.append(gene)
    destination.split = (0 if len(destination.genes) < 2 else
                         max(1, len(destination.genes) // 2))
    return True


def append_logic_with_fanout(
        genome, seeds, families, choice, *, chromosome=None):
    """Append a gate and, when physically free, a real two-output delay.

    The gate remains its own permanent component type and the fan-out remains
    its own permanent component type.  Grouping their insertion only makes a
    newly discovered function reusable by later construction; it does not add
    a virtual output or change either node's physics.
    """
    target, component_id, _refs = choice
    entry = BY_ID.get(int(component_id))
    if not append_placement_choice(
            genome, choice, chromosome=chromosome):
        return False
    enabled = normalise_families(families)
    if (entry is None or entry.family != "LOGIC" or "DELAY" not in enabled
            or len(entry.outputs) != 1
            or len(_genes(genome)) >= MAX_PLACEMENTS):
        return True
    gate_id = int(genome.next_gene_id) - 1
    output_direction = entry.outputs[0]
    fanout_cell = hex_dirs(*target)[output_direction]
    trace = develop_constructive(genome, seeds)
    if fanout_cell in trace.grid:
        return True
    receiver = _receiver_direction(fanout_cell, target)
    fanouts = [
        candidate for candidate in COMPONENTS
        if candidate.family == "DELAY"
        and candidate.inputs == (receiver,)
        and len(candidate.outputs) == 2]
    if fanouts:
        append_placement_choice(
            genome,
            (fanout_cell, random.choice(fanouts).id,
             (BranchRef(gate_id, output_direction),)),
            chromosome=chromosome)
    return True


def append_random_placement(genome, seeds, families, *, chromosome=None,
                            input_count=None):
    if len(_genes(genome)) >= MAX_PLACEMENTS:
        return False
    choices = feasible_placements(
        genome, seeds, families, input_count=input_count)
    if not choices:
        return False
    choice = _choose_family_first(choices)
    if BY_ID[int(choice[1])].family == "LOGIC":
        return append_logic_with_fanout(
            genome, seeds, families, choice, chromosome=chromosome)
    return append_placement_choice(genome, choice, chromosome=chromosome)


def append_convergent_placement(genome, seeds, families, *, chromosome=None):
    """Append a two-input component that maximizes new source convergence."""
    choices = feasible_placements(
        genome, seeds, families, input_count=2)
    if not choices:
        return False
    ancestry = source_ancestry(genome, seeds)

    def score(choice):
        cones = [ancestry.get(int(ref.node_id), frozenset())
                 for ref in choice[2]]
        union = frozenset().union(*cones)
        return (len(union) - max((len(cone) for cone in cones), default=0),
                len(union))

    best = max(score(choice) for choice in choices)
    preferred = [choice for choice in choices if score(choice) == best]
    return append_logic_with_fanout(
        genome, seeds, families, _choose_family_first(preferred),
        chromosome=chromosome)


def _seed_source_fanout(genome, seeds, families):
    """Give each input one real two-output transport branch when available."""
    for input_index in range(len(seeds)):
        source_id = input_node_id(input_index)
        choices = [
            choice for choice in feasible_placements(genome, seeds, families)
            if len(BY_ID[choice[1]].inputs) == 1
            and len(BY_ID[choice[1]].outputs) == 2
            and choice[2][0].node_id == source_id
            and BY_ID[choice[1]].family == "DELAY"
        ]
        if not choices or len(_genes(genome)) >= MAX_PLACEMENTS:
            continue
        _target, component_id, refs = random.choice(choices)
        gene_id = int(genome.next_gene_id)
        genome.next_gene_id = gene_id + 1
        available = [item for item in genome.chromosomes
                     if len(item.genes) < MAX_GENES]
        if not available:
            return
        chromosome = random.choice(available)
        chromosome.genes.append(PlacementGene(
            gene_id, int(component_id), tuple(refs), gene_id))
        chromosome.split = (0 if len(chromosome.genes) < 2 else
                            max(1, len(chromosome.genes) // 2))


def seed_constructive_genome(
        genome, families, target_count=12, *, source_fanout=True):
    seeds = tuple(getattr(genome, "input_layout", None) or ((0, 0),))
    if source_fanout:
        _seed_source_fanout(genome, seeds, families)
    # Establish several physical convergence motifs before unconstrained
    # growth. This uses only the number of source pads and live facing ports;
    # gate behavior and every target answer remain unknown.
    desired_joins = min(6, max(0, len(seeds) * (len(seeds) - 1) // 2))
    joins, join_attempts = 0, 18
    while (joins < desired_joins and join_attempts > 0
           and len(_genes(genome)) < int(target_count)):
        join_attempts -= 1
        if append_convergent_placement(genome, seeds, families):
            joins += 1
        else:
            append_random_placement(
                genome, seeds, families, input_count=1)
    attempts = max(16, int(target_count) * 5)
    while len(_genes(genome)) < int(target_count) and attempts > 0:
        attempts -= 1
        if not append_random_placement(genome, seeds, families):
            break
    return genome
