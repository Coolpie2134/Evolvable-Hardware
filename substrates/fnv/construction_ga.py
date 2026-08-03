"""Branch-local genetic operators for constructive FNV genomes."""
from __future__ import annotations

import copy
import math
import random

from substrates.nervous.hexgrid import hex_dirs

from .catalogue import (
    BY_ID, COMPONENTS, behavior_component_ids, local_component_ids,
    normalise_families,
)
from .construction import (
    append_logic_with_fanout, append_placement_choice,
    append_random_placement, develop_constructive, feasible_placements,
    frontier_references, source_ancestry,
)
from .genome import BranchRef, MAX_GENES, MAX_PLACEMENTS, PlacementGene


def clone_constructive(genome):
    clone = copy.copy(genome)
    clone.chromosomes = []
    for chromosome in genome.chromosomes:
        copied = copy.copy(chromosome)
        copied.genes = [copy.copy(gene) for gene in chromosome.genes]
        clone.chromosomes.append(copied)
    layout = getattr(genome, "input_layout", None)
    if layout is not None:
        clone.input_layout = tuple(tuple(cell) for cell in layout)
    return clone


def placement_genes(genome):
    return tuple(
        gene for chromosome in genome.chromosomes
        for gene in chromosome.genes if isinstance(gene, PlacementGene))


def constructive_signature(genome):
    return (
        tuple(tuple(cell) for cell in (getattr(
            genome, "input_layout", None) or ())),
        tuple(
            (int(gene.gene_id), int(gene.component_id), int(gene.branch_id),
             tuple((int(ref.node_id), ref.direction) for ref in gene.inputs))
            for gene in sorted(
                placement_genes(genome), key=lambda item: int(item.gene_id))),
    )


def _seeds(genome, fallback=()):
    return tuple(getattr(genome, "input_layout", None) or fallback or ((0, 0),))


def _poisson(mean):
    if mean <= 0:
        return 0
    limit, count, product = math.exp(-mean), 0, 1.0
    while product > limit:
        count += 1
        product *= random.random()
    return count - 1


def _active_pool(genome, seeds):
    genes = placement_genes(genome)
    active = develop_constructive(genome, seeds).active_ids
    live = [gene for gene in genes if int(gene.gene_id) in active]
    return live if live and random.random() < 0.85 else list(genes)


def _change_component(genome, seeds, families, focused_families=()):
    pool = _active_pool(genome, seeds)
    if not pool:
        return False
    gene = random.choice(pool)
    current = BY_ID[int(gene.component_id)]
    enabled = normalise_families(families)
    focus = normalise_families(focused_families) if focused_families else enabled
    allowed = focus if current.family in focus else enabled
    compatible = [
        entry for entry in COMPONENTS
        if entry.id and entry.id != current.id and entry.family in allowed
        and entry.inputs == current.inputs]
    if not compatible:
        return False
    same_route = [entry for entry in compatible
                  if entry.outputs == current.outputs]
    behavior = set(behavior_component_ids(current.id))
    local = set(local_component_ids(current.id, allowed))
    preferred = [entry for entry in same_route
                 if entry.id in behavior or entry.id in local]
    gene.component_id = int(random.choice(
        preferred or same_route or compatible).id)
    return True


def _logic_cascade_options(genome, seeds, families, focused_families=()):
    """Compatible paired substitutions across the active logic fabric."""
    enabled = normalise_families(families)
    focus = normalise_families(
        focused_families) if focused_families else enabled
    trace = develop_constructive(genome, seeds)
    genes = {int(gene.gene_id): gene for gene in placement_genes(genome)}
    active = set(trace.active_ids)

    def nearest_upstream_logic(gene):
        found, pending, visited = set(), [
            int(ref.node_id) for ref in gene.inputs if int(ref.node_id) >= 0], set()
        while pending:
            node_id = pending.pop()
            if node_id in visited or node_id not in active:
                continue
            visited.add(node_id)
            source = genes.get(node_id)
            if source is None:
                continue
            if BY_ID[int(source.component_id)].family == "LOGIC":
                found.add(node_id)
            else:
                pending.extend(int(ref.node_id) for ref in source.inputs
                               if int(ref.node_id) >= 0)
        return found

    options, covered_pairs = [], set()
    for downstream in genes.values():
        current_down = BY_ID[int(downstream.component_id)]
        if (int(downstream.gene_id) not in active
                or current_down.family != "LOGIC"
                or current_down.family not in focus):
            continue
        down_variants = behavior_component_ids(current_down.id)
        for upstream_id in nearest_upstream_logic(downstream):
            upstream = genes[upstream_id]
            current_up = BY_ID[int(upstream.component_id)]
            if current_up.family not in focus:
                continue
            covered_pairs.add(tuple(sorted((
                int(upstream_id), int(downstream.gene_id)))))
            for up_id in behavior_component_ids(current_up.id):
                for down_id in down_variants:
                    options.append((
                        int(upstream_id), int(up_id),
                        int(downstream.gene_id), int(down_id)))
    active_logic = [
        gene for gene in genes.values()
        if int(gene.gene_id) in active
        and BY_ID[int(gene.component_id)].family == "LOGIC"
        and BY_ID[int(gene.component_id)].family in focus]
    for index, left in enumerate(active_logic):
        left_variants = behavior_component_ids(int(left.component_id))
        for right in active_logic[index + 1:]:
            pair = tuple(sorted((int(left.gene_id), int(right.gene_id))))
            if pair in covered_pairs:
                continue
            for left_id in left_variants:
                for right_id in behavior_component_ids(
                        int(right.component_id)):
                    options.append((
                        int(left.gene_id), int(left_id),
                        int(right.gene_id), int(right_id)))
    return options


def _install_logic_cascade_option(genome, seeds, option):
    upstream_id, upstream_component, downstream_id, downstream_component = option
    trial = clone_constructive(genome)
    genes = {int(gene.gene_id): gene for gene in placement_genes(trial)}
    if upstream_id not in genes or downstream_id not in genes:
        return False
    genes[upstream_id].component_id = int(upstream_component)
    genes[downstream_id].component_id = int(downstream_component)
    active = develop_constructive(trial, seeds).active_ids
    if not {upstream_id, downstream_id}.issubset(active):
        return False
    genome.chromosomes = trial.chromosomes
    return True


def _priority_logic_pairs(genome, seeds):
    """Gate pairs coupled by adjacency or downstream reconvergence."""
    trace = develop_constructive(genome, seeds)
    genes = {int(gene.gene_id): gene for gene in placement_genes(genome)}
    active = set(trace.active_ids)

    def nearest_upstream(gene):
        found, pending, visited = set(), [
            int(ref.node_id) for ref in gene.inputs if int(ref.node_id) >= 0], set()
        while pending:
            node_id = pending.pop()
            if node_id in visited or node_id not in active:
                continue
            visited.add(node_id)
            source = genes.get(node_id)
            if source is None:
                continue
            if BY_ID[int(source.component_id)].family == "LOGIC":
                found.add(node_id)
            else:
                pending.extend(int(ref.node_id) for ref in source.inputs
                               if int(ref.node_id) >= 0)
        return found

    pairs = set()
    for downstream in genes.values():
        if (int(downstream.gene_id) not in active
                or BY_ID[int(downstream.component_id)].family != "LOGIC"):
            continue
        upstream = sorted(nearest_upstream(downstream))
        pairs.update(tuple(sorted((node_id, int(downstream.gene_id))))
                     for node_id in upstream)
        for index, left in enumerate(upstream):
            pairs.update((left, right) for right in upstream[index + 1:])
    return pairs


def _change_logic_cascade(genome, seeds, families, focused_families=()):
    options = _logic_cascade_options(
        genome, seeds, families, focused_families)
    priority = _priority_logic_pairs(genome, seeds)
    preferred = [option for option in options
                 if tuple(sorted((option[0], option[2]))) in priority]
    return bool(options) and _install_logic_cascade_option(
        genome, seeds, random.choice(preferred or options))


def _remove_block(genome, branch_id, *, descendants=False):
    removed_ids = {
        int(gene.gene_id) for gene in placement_genes(genome)
        if int(gene.branch_id) == int(branch_id)}
    if descendants:
        changed = True
        while changed:
            changed = False
            for gene in placement_genes(genome):
                if (int(gene.gene_id) not in removed_ids
                        and any(int(ref.node_id) in removed_ids
                                for ref in gene.inputs)):
                    removed_ids.add(int(gene.gene_id))
                    changed = True
    removed = False
    for chromosome in genome.chromosomes:
        before = len(chromosome.genes)
        chromosome.genes = [
            gene for gene in chromosome.genes
            if not (isinstance(gene, PlacementGene)
                    and int(gene.gene_id) in removed_ids)]
        removed |= len(chromosome.genes) != before
        chromosome.split = (0 if len(chromosome.genes) < 2 else
                            max(1, min(int(chromosome.split),
                                       len(chromosome.genes) - 1)))
    return removed


def _remove_subtree_from_gene(genome, gene_id):
    removed_ids = {int(gene_id)}
    changed = True
    while changed:
        changed = False
        for gene in placement_genes(genome):
            if (int(gene.gene_id) not in removed_ids
                    and any(int(ref.node_id) in removed_ids
                            for ref in gene.inputs)):
                removed_ids.add(int(gene.gene_id))
                changed = True
    removed = False
    for chromosome in genome.chromosomes:
        before = len(chromosome.genes)
        chromosome.genes = [
            gene for gene in chromosome.genes
            if not (isinstance(gene, PlacementGene)
                    and int(gene.gene_id) in removed_ids)]
        removed |= before != len(chromosome.genes)
        chromosome.split = (0 if len(chromosome.genes) < 2 else
                            max(1, min(int(chromosome.split),
                                       len(chromosome.genes) - 1)))
    return removed


def _delete_block(genome, seeds):
    pool = _active_pool(genome, seeds)
    if not pool:
        return False
    return _remove_block(
        genome, random.choice(pool).branch_id, descendants=True)


def _block(genome, branch_id):
    return tuple(sorted(
        (gene for gene in placement_genes(genome)
         if int(gene.branch_id) == int(branch_id)),
        key=lambda gene: int(gene.gene_id)))


def _block_roots(block):
    ids = {int(gene.gene_id) for gene in block}
    return [gene for gene in block
            if all(int(ref.node_id) not in ids for ref in gene.inputs)]


def _install_block(target, source_block, seeds, families, *, source_trace=None):
    """Copy one labelled block and physically re-anchor its sole root."""
    if not source_block or len(placement_genes(target)) + len(
            source_block) > MAX_PLACEMENTS:
        return False
    roots = _block_roots(source_block)
    if len(roots) != 1:
        return False
    root = roots[0]
    old_parity = None
    if source_trace is not None:
        old_cell = source_trace.coordinates.get(int(root.gene_id))
        if old_cell is not None:
            old_parity = sum(old_cell) % 2
    choices = feasible_placements(
        target, seeds, families, component_id=root.component_id,
        target_parity=old_parity)
    if not choices and old_parity is not None:
        choices = feasible_placements(
            target, seeds, families, component_id=root.component_id)
    if not choices:
        return False
    _cell, _component, root_refs = random.choice(choices)
    mapping = {}
    for gene in source_block:
        mapping[int(gene.gene_id)] = int(target.next_gene_id)
        target.next_gene_id += 1
    new_branch = mapping[int(root.gene_id)]
    copies = []
    for gene in source_block:
        refs = tuple(
            type(ref)(mapping.get(int(ref.node_id), int(ref.node_id)),
                      ref.direction)
            for ref in gene.inputs)
        if gene is root:
            refs = tuple(root_refs)
        copies.append(PlacementGene(
            mapping[int(gene.gene_id)], int(gene.component_id), refs,
            new_branch))
    available = [chromosome for chromosome in target.chromosomes
                 if len(chromosome.genes) + len(copies) <= MAX_GENES]
    if not available:
        return False
    chromosome = random.choice(available)
    chromosome.genes.extend(copies)
    chromosome.split = (0 if len(chromosome.genes) < 2 else
                        max(1, len(chromosome.genes) // 2))
    return True


def _dependency_module(genome, seeds):
    """Choose one complete live dependency cone from ``genome``.

    Constructive branch IDs make individual routes hereditary, but a useful
    fixed-function result commonly spans several such branches joined by
    gates.  A crossover unit therefore has to include every positive ancestor
    of one live sink.  Sinks with the broadest source ancestry are preferred
    using physical provenance only; desired outputs and simulated behavior are
    deliberately absent from this choice.
    """
    trace = develop_constructive(genome, seeds)
    active = set(trace.active_ids)
    if not active:
        return ()
    genes = {int(gene.gene_id): gene for gene in placement_genes(genome)}
    consumed = {
        int(ref.node_id)
        for gene_id in active
        for ref in genes[gene_id].inputs
        if int(ref.node_id) in active
    }
    sinks = sorted(active - consumed) or sorted(active)
    ancestry = source_ancestry(genome, seeds)
    widest = max(len(ancestry.get(gene_id, ())) for gene_id in sinks)
    sink = random.choice([
        gene_id for gene_id in sinks
        if len(ancestry.get(gene_id, ())) == widest
    ])
    selected, pending = set(), [sink]
    while pending:
        gene_id = pending.pop()
        if gene_id in selected or gene_id not in active:
            continue
        selected.add(gene_id)
        pending.extend(
            int(ref.node_id) for ref in genes[gene_id].inputs
            if int(ref.node_id) >= 0)
    return tuple(sorted(
        (genes[gene_id] for gene_id in selected),
        key=lambda gene: (trace.depths.get(int(gene.gene_id), 0),
                          int(gene.gene_id))))


def _transplant_dependency_module(target, donor, seeds):
    """Install one whole donor dependency cone at its inherited coordinates.

    This path is valid only when the logical input layouts match.  A donor
    component displaces a different target component at the same physical
    site, together with dependants of that displaced component; identical
    components are safely shared.  Other target branches remain untouched.
    Thus collision resolution is local while the selected donor module is
    inherited as a coherent functional unit.
    """
    target_layout = tuple(tuple(cell) for cell in _seeds(target, seeds))
    donor_layout = tuple(tuple(cell) for cell in _seeds(donor, seeds))
    if target_layout != donor_layout:
        return False
    module = _dependency_module(donor, donor_layout)
    if not module:
        return False

    donor_trace = develop_constructive(donor, donor_layout)
    target_trace = develop_constructive(target, target_layout)
    target_genes = {
        int(gene.gene_id): gene for gene in placement_genes(target)}
    occupied = {
        tuple(cell): int(node_id)
        for node_id, cell in target_trace.coordinates.items()
        if int(node_id) >= 0
    }

    # Let the selected donor module win only its own physical footprint.  The
    # descendant deletion prevents stale target genes from pointing at a node
    # whose component has just been replaced.
    conflicts = []
    for gene in module:
        owner = occupied.get(tuple(
            donor_trace.coordinates[int(gene.gene_id)]))
        if (owner is not None
                and int(target_genes[owner].component_id)
                != int(gene.component_id)):
            conflicts.append(owner)
    changed = False
    for owner in dict.fromkeys(conflicts):
        changed |= _remove_subtree_from_gene(target, owner)

    target_trace = develop_constructive(target, target_layout)
    target_genes = {
        int(gene.gene_id): gene for gene in placement_genes(target)}
    occupied = {
        tuple(cell): int(node_id)
        for node_id, cell in target_trace.coordinates.items()
        if int(node_id) >= 0
    }
    donor_chromosome = {}
    for index, chromosome in enumerate(donor.chromosomes):
        for gene in chromosome.genes:
            if isinstance(gene, PlacementGene):
                donor_chromosome[int(gene.gene_id)] = index

    node_mapping, branch_mapping = {}, {}
    for gene in module:
        donor_id = int(gene.gene_id)
        mapped_refs = []
        for ref in gene.inputs:
            source_id = int(ref.node_id)
            if source_id >= 0:
                if source_id not in node_mapping:
                    return False
                source_id = node_mapping[source_id]
            mapped_refs.append(BranchRef(source_id, ref.direction))

        cell = tuple(donor_trace.coordinates[donor_id])
        owner = occupied.get(cell)
        if owner is not None:
            if int(target_genes[owner].component_id) != int(gene.component_id):
                return False
            node_mapping[donor_id] = owner
            branch_mapping.setdefault(
                int(gene.branch_id), int(target_genes[owner].branch_id))
            continue

        if len(placement_genes(target)) >= MAX_PLACEMENTS:
            return False
        available = [
            (index, chromosome)
            for index, chromosome in enumerate(target.chromosomes)
            if len(chromosome.genes) < MAX_GENES
        ]
        if not available:
            return False
        preferred_index = donor_chromosome.get(donor_id, -1)
        destination = next((
            chromosome for index, chromosome in available
            if index == preferred_index), None)
        if destination is None:
            destination = min(available, key=lambda item: len(
                item[1].genes))[1]

        new_id = int(target.next_gene_id)
        target.next_gene_id = new_id + 1
        node_mapping[donor_id] = new_id
        new_branch = branch_mapping.setdefault(
            int(gene.branch_id), new_id)
        copied = PlacementGene(
            new_id, int(gene.component_id), tuple(mapped_refs), new_branch)
        destination.genes.append(copied)
        destination.split = (0 if len(destination.genes) < 2 else
                             max(1, len(destination.genes) // 2))
        target_genes[new_id] = copied
        occupied[cell] = new_id
        changed = True

    if not changed:
        return False
    final_trace = develop_constructive(target, target_layout)
    return all(mapped_id in final_trace.active_ids
               for mapped_id in node_mapping.values())


def _duplicate_block(genome, seeds, families):
    pool = _active_pool(genome, seeds)
    if not pool:
        return False
    branch_id = random.choice(pool).branch_id
    block = _block(genome, branch_id)
    trace = develop_constructive(genome, seeds)
    return _install_block(
        genome, block, seeds, families, source_trace=trace)


def _reroute_block(genome, seeds, families):
    pool = _active_pool(genome, seeds)
    if not pool:
        return False
    branch_id = random.choice(pool).branch_id
    block = _block(genome, branch_id)
    roots = _block_roots(block)
    if len(roots) != 1:
        return False
    root = roots[0]
    trace = develop_constructive(genome, seeds)
    old_cell = trace.coordinates.get(int(root.gene_id))
    without = clone_constructive(genome)
    _remove_block(without, branch_id)
    choices = feasible_placements(
        without, seeds, families, component_id=root.component_id,
        target_parity=(None if old_cell is None else sum(old_cell) % 2))
    choices = [choice for choice in choices
               if tuple(choice[2]) != tuple(root.inputs)]
    if not choices:
        return False
    old_inputs = tuple(root.inputs)
    root.inputs = tuple(random.choice(choices)[2])
    if int(root.gene_id) not in develop_constructive(
            genome, seeds).active_ids:
        root.inputs = old_inputs
        return False
    return True


def _fanout_options(genome, seeds, families):
    """Neutral one-output -> two-output DELAY type replacements."""
    enabled = normalise_families(families)
    if "DELAY" not in enabled:
        return []
    trace = develop_constructive(genome, seeds)
    options = []
    for gene in placement_genes(genome):
        gene_id = int(gene.gene_id)
        if gene_id not in trace.active_ids:
            continue
        current = BY_ID[int(gene.component_id)]
        if current.family != "DELAY" or len(current.outputs) != 1:
            continue
        coordinate = trace.coordinates[gene_id]
        for replacement in COMPONENTS:
            if (replacement.family != "DELAY"
                    or replacement.inputs != current.inputs
                    or replacement.duration != current.duration
                    or len(replacement.outputs) != 2
                    or not set(current.outputs).issubset(
                        replacement.outputs)):
                continue
            added = set(replacement.outputs) - set(current.outputs)
            if all(hex_dirs(*coordinate)[direction] not in trace.grid
                   for direction in added):
                options.append((gene_id, int(replacement.id)))
    return options


def _install_fanout_option(genome, seeds, option):
    gene_id, replacement_id = option
    gene = next((item for item in placement_genes(genome)
                 if int(item.gene_id) == int(gene_id)), None)
    replacement = BY_ID.get(int(replacement_id))
    if gene is None or replacement is None:
        return False
    trial = clone_constructive(genome)
    trial_gene = next(item for item in placement_genes(trial)
                      if int(item.gene_id) == int(gene_id))
    trial_gene.component_id = int(replacement.id)
    if int(gene_id) not in develop_constructive(trial, seeds).active_ids:
        return False
    genome.chromosomes = trial.chromosomes
    return True


def _replace_consumer_with_fanout(genome, seeds, families):
    """Expose a branch by selecting its real two-output DELAY sibling type."""
    options = _fanout_options(genome, seeds, families)
    return bool(options) and _install_fanout_option(
        genome, seeds, random.choice(options))


def _join_live_tips(genome, seeds, families):
    """Join two facing live output ports with one real two-input component.

    Generic extension previously sampled unary routing and binary convergence
    from one large pool.  On a developed body that made a rare, immediately
    feasible join almost invisible.  This operator changes only that sampling
    probability: every candidate is a catalogue component at an empty cell
    whose declared input pins already face the two named source ports.
    """
    choices = feasible_placements(
        genome, seeds, families, input_count=2)
    if not choices:
        return False
    ancestry = source_ancestry(genome, seeds)

    def convergence(choice):
        cones = [ancestry.get(int(ref.node_id), frozenset())
                 for ref in choice[2]]
        union = frozenset().union(*cones)
        return (len(union) - max((len(cone) for cone in cones), default=0),
                len(union))

    best = max(convergence(choice) for choice in choices)
    independent = [choice for choice in choices
                   if convergence(choice) == best]
    return append_logic_with_fanout(
        genome, seeds, families, random.choice(independent or choices))


def _receiver_direction(target, source):
    return next((direction for direction, cell in hex_dirs(*target).items()
                 if cell == source), None)


def _bridge_options(genome, seeds, families, max_delays=8, limit=96):
    """Find short empty-cell paths that can make two live branches meet.

    Search sees only occupied cells and declared physical ports.  It does not
    score node behavior, inspect probes, or know the target.  A returned option
    is a sequence of real unary DELAY placements followed by one real LOGIC
    placement; it is therefore a heritable multi-gene structural mutation, not
    a virtual wire or an answer-specific gate.
    """
    enabled = normalise_families(families)
    if "LOGIC" not in enabled or "DELAY" not in enabled:
        return []
    trace, driven = frontier_references(genome, seeds)
    ancestry = source_ancestry(genome, seeds)

    def lineage(ref):
        return ancestry.get(int(ref.node_id), frozenset())

    logic_entries = [entry for entry in COMPONENTS
                     if entry.family == "LOGIC"]
    start_entries = [
        (start_cell, incoming, start_ref)
        for start_cell, start_inputs in sorted(driven.items())
        for incoming, start_ref in sorted(start_inputs.items())]
    random.shuffle(start_entries)
    # Ordinary mutation searches a rotating subset; a large diagnostic/rescue
    # request can explicitly cover every live label without exploding routine
    # reproduction cost.
    start_budget = (
        len(start_entries)
        if int(limit) >= len(start_entries) * 20
        else min(len(start_entries), max(8, int(limit) // 4)))
    options_by_start = {}
    for start_cell, incoming, start_ref in start_entries[:start_budget]:
        start_options = []
        # state = empty cell to occupy, receiving pin, path cells/turns.
        queue = [(start_cell, incoming, ())]
        visited = {(start_cell, incoming): 0}
        # Bound work per labelled port rather than globally.  A global cap
        # made bridge eligibility depend on coordinate sort order and
        # silently excluded most of a mature phenotype.
        while queue and len(start_options) < 256:
            cell, receiver, steps = queue.pop(0)
            path_cells = {step[0] for step in steps}
            if steps:
                for other_direction, other_ref in sorted(
                        driven.get(cell, {}).items()):
                    left_cone = lineage(start_ref)
                    right_cone = lineage(other_ref)
                    if (other_direction == receiver
                            or left_cone == right_cone):
                        continue
                    available_inputs = {receiver, other_direction}
                    for entry in logic_entries:
                        if set(entry.inputs) != available_inputs:
                            continue
                        output_cell = hex_dirs(*cell)[entry.outputs[0]]
                        # Make the insertion observationally additive: its
                        # final output cannot drive an existing component.
                        if (output_cell in trace.grid
                                or output_cell in path_cells):
                            continue
                        start_options.append((
                            start_ref, steps, cell, receiver,
                            other_direction, other_ref, entry.id))
                        if len(start_options) >= 256:
                            break
                    if len(start_options) >= 256:
                        break
            if len(steps) >= int(max_delays):
                continue
            for output_direction in ("L", "R", "D"):
                if output_direction == receiver:
                    continue
                next_cell = hex_dirs(*cell)[output_direction]
                if (next_cell in trace.grid or next_cell in path_cells
                        or next_cell == cell):
                    continue
                next_receiver = _receiver_direction(next_cell, cell)
                if next_receiver is None:
                    continue
                depth = len(steps) + 1
                key = (next_cell, next_receiver)
                if visited.get(key, depth + 1) <= depth:
                    continue
                visited[key] = depth
                queue.append((
                    next_cell, next_receiver,
                    steps + ((cell, receiver, output_direction),)))
        if start_options:
            options_by_start[(int(start_ref.node_id),
                              start_ref.direction)] = start_options

    # Round-robin across stable branch labels.  Shuffle within each label so
    # different physical gate variants and endpoints remain reachable without
    # restoring coordinate-order bias.
    groups = list(options_by_start.values())
    for group in groups:
        random.shuffle(group)
        group.sort(key=lambda option: len(option[1]))
    random.shuffle(groups)
    options = []
    while groups and len(options) < int(limit):
        remaining = []
        for group in groups:
            if group and len(options) < int(limit):
                options.append(group.pop(0))
            if group:
                remaining.append(group)
        groups = remaining
    return options


def _install_bridge_option(genome, option):
    (start_ref, steps, endpoint, endpoint_receiver,
     other_direction, other_ref, logic_id) = option
    required = len(steps) + 1
    if (len(placement_genes(genome)) + required > MAX_PLACEMENTS
            or sum(MAX_GENES - len(chromosome.genes)
                   for chromosome in genome.chromosomes) < required):
        return False
    trial = clone_constructive(genome)
    current_ref = start_ref
    for cell, receiver, output_direction in steps:
        delays = [
            entry for entry in COMPONENTS
            if entry.family == "DELAY"
            and entry.inputs == (receiver,)
            and entry.outputs == (output_direction,)]
        if not delays:
            return False
        entry = random.choice(delays)
        if not append_placement_choice(
                trial, (cell, entry.id, (current_ref,))):
            return False
        current_ref = BranchRef(
            int(trial.next_gene_id) - 1, output_direction)
    logic = BY_ID[int(logic_id)]
    refs_by_direction = {
        endpoint_receiver: current_ref,
        other_direction: other_ref,
    }
    refs = tuple(refs_by_direction[direction] for direction in logic.inputs)
    if not append_logic_with_fanout(
            trial, _seeds(trial), ("LOGIC", "DELAY"),
            (endpoint, logic.id, refs)):
        return False
    # Every chosen cell was empty in the original trace and new IDs are later
    # than existing IDs, but verify atomic development before committing.
    new_ids = set(range(int(genome.next_gene_id), int(trial.next_gene_id)))
    trace = develop_constructive(trial, _seeds(trial))
    if not new_ids.issubset(trace.active_ids):
        return False
    genome.chromosomes = trial.chromosomes
    genome.next_gene_id = trial.next_gene_id
    return True


def _bridge_live_tips(genome, seeds, families):
    options = _bridge_options(genome, seeds, families)
    if not options:
        return False
    ancestry = source_ancestry(genome, seeds)
    existing_cones = {
        cone for node_id, cone in ancestry.items()
        if int(node_id) >= 0 and len(cone) >= 2}

    def convergence(option):
        cones = [ancestry.get(int(ref.node_id), frozenset())
                 for ref in (option[0], option[5])]
        union = frozenset().union(*cones)
        return (int(len(union) >= 2 and union not in existing_cones),
                len(union) - max((len(cone) for cone in cones), default=0),
                len(union))

    best = max(convergence(option) for option in options)
    productive = [option for option in options
                  if convergence(option) == best]
    shortest = min(len(option[1]) for option in productive)
    near = [option for option in productive
            if len(option[1]) <= shortest + 1]
    return _install_bridge_option(genome, random.choice(near))


def seed_convergent_bridges(genome, families, max_bridges=2):
    """Give a fresh body a small, random, source-integrating morphology.

    Gate behaviors remain random fixed catalogue IDs.  The only preference is
    structural: a bridge should add a logical source pad to the dependency
    ancestry already reaching its other end.  This is the constructive FNV
    equivalent of starting with connected material rather than isolated dust.
    """
    seeds = _seeds(genome)
    for _ in range(max(0, int(max_bridges))):
        ancestry = source_ancestry(genome, seeds)
        existing_cones = {
            cone for node_id, cone in ancestry.items()
            if int(node_id) >= 0 and len(cone) >= 2}
        options = _bridge_options(
            genome, seeds, families, max_delays=8, limit=5000)
        if not options:
            break

        def convergence(option):
            cones = [ancestry.get(int(ref.node_id), frozenset())
                     for ref in (option[0], option[5])]
            union = frozenset().union(*cones)
            return (int(len(union) >= 2 and union not in existing_cones),
                    len(union) - max(
                        (len(cone) for cone in cones), default=0),
                    len(union), -len(option[1]))

        best = max(convergence(option) for option in options)
        preferred = [option for option in options
                     if convergence(option) == best]
        if not _install_bridge_option(genome, random.choice(preferred)):
            break
    return genome


def mutate_constructive_once(genome, families, growth_seeds=(),
                             focused_families=()):
    seeds = _seeds(genome, growth_seeds)
    occupancy = len(placement_genes(genome)) / float(MAX_PLACEMENTS)
    draw = random.random()
    if occupancy >= 0.75:
        cutoffs = (0.06, 0.13, 0.25, 0.45, 0.63, 0.75, 0.79, 0.91)
    else:
        # Below saturation, coherent additions are much rarer than neutral
        # deletion. Keep removal available, but do not let that asymmetry
        # recreate an accidental small-genome preference.
        cutoffs = (0.15, 0.25, 0.45, 0.65, 0.80, 0.88, 0.91, 0.98)
    if draw < cutoffs[0]:
        return append_random_placement(genome, seeds, families)
    if draw < cutoffs[1]:
        return _join_live_tips(genome, seeds, families)
    if draw < cutoffs[2]:
        return _bridge_live_tips(genome, seeds, families)
    if draw < cutoffs[3]:
        return _change_component(
            genome, seeds, families, focused_families)
    if draw < cutoffs[4]:
        return _change_logic_cascade(
            genome, seeds, families, focused_families)
    if draw < cutoffs[5]:
        return _reroute_block(genome, seeds, families)
    if draw < cutoffs[6]:
        return _duplicate_block(genome, seeds, families)
    if draw < cutoffs[7]:
        return _replace_consumer_with_fanout(genome, seeds, families)
    return _delete_block(genome, seeds)


def mutate_constructive(genome, mean_mutations, families, growth_seeds=(),
                        focused_families=()):
    # One constructive event may append an explicit multi-gene bridge or
    # replace a whole dependency block.  Reusing the associative encoding's
    # event count verbatim stacked several large edits in one birth and erased
    # the locality this representation was introduced to provide.
    requested = 4.0 if mean_mutations is None else max(
        0.0, float(mean_mutations))
    mean = max(0.5, requested * 0.40)
    for _ in range(max(1, _poisson(mean))):
        if not mutate_constructive_once(
                genome, families, growth_seeds, focused_families):
            append_random_placement(
                genome, _seeds(genome, growth_seeds), families)
    return genome


def crossover_constructive(parent_a, parent_b, families):
    child = clone_constructive(parent_a)
    layout_a = tuple(tuple(cell) for cell in _seeds(parent_a))
    layout_b = tuple(tuple(cell) for cell in _seeds(parent_b))
    # Layout and body are a co-adapted physical module.  Swapping only the pad
    # coordinates silently moved the roots out from under an otherwise useful
    # body.  Equal layouts instead permit exact dependency-cone inheritance.
    if layout_a == layout_b:
        trial = clone_constructive(child)
        if _transplant_dependency_module(trial, parent_b, layout_a):
            return trial
    source_genes = placement_genes(parent_b)
    if not source_genes:
        return child
    source = random.choice(source_genes)
    block = _block(parent_b, source.branch_id)
    seeds = layout_a
    trace_b = develop_constructive(parent_b, _seeds(parent_b))
    if len(placement_genes(child)) + len(block) > MAX_PLACEMENTS:
        own = placement_genes(child)
        if own:
            _remove_block(child, random.choice(own).branch_id)
    _install_block(
        child, block, seeds, families, source_trace=trace_b)
    return child


def plateau_candidates_constructive(
        genome, limit, families, growth_seeds=(), focused_families=(),
        layout_mutator=None):
    limit = max(0, int(limit))
    proposals, seen = [], {constructive_signature(genome)}
    seeds = _seeds(genome, growth_seeds)

    cascade_options = _logic_cascade_options(
        genome, seeds, families, focused_families)
    priority_pairs = _priority_logic_pairs(genome, seeds)
    priority_options = [
        option for option in cascade_options
        if tuple(sorted((option[0], option[2]))) in priority_pairs]
    other_options = [
        option for option in cascade_options
        if tuple(sorted((option[0], option[2]))) not in priority_pairs]
    random.shuffle(priority_options)
    random.shuffle(other_options)
    cascade_options = priority_options + other_options
    cascade_quota = max(1, limit // 2)
    for option in cascade_options:
        if len(proposals) >= cascade_quota:
            break
        candidate = clone_constructive(genome)
        if not _install_logic_cascade_option(candidate, seeds, option):
            continue
        signature = constructive_signature(candidate)
        if signature not in seen:
            seen.add(signature)
            proposals.append(candidate)

    # Preserve a bounded share of rescue for exposing consumed signals.  These
    # variants are neutral unless later routing uses the new physical branch,
    # which is exactly the stepping stone a constructive encoding needs.
    fanout_quota = min(limit, len(proposals) + max(1, limit // 6))
    for option in _fanout_options(genome, seeds, families):
        if len(proposals) >= fanout_quota:
            break
        candidate = clone_constructive(genome)
        if not _install_fanout_option(candidate, seeds, option):
            continue
        signature = constructive_signature(candidate)
        if signature not in seen:
            seen.add(signature)
            proposals.append(candidate)

    # Exhaust the immediately feasible binary compositions first.  A mature
    # phenotype commonly has hundreds of unary extensions but only one cell
    # where two useful live ports face each other; purely random rescue almost
    # never exposed that local neighbourhood to selection.
    joins = feasible_placements(
        genome, seeds, families, input_count=2)
    ancestry = source_ancestry(genome, seeds)

    def lineage(ref):
        return ancestry.get(int(ref.node_id), frozenset())

    def convergence(choice):
        cones = [lineage(ref) for ref in choice[2]]
        union = frozenset().union(*cones)
        return (len(union) - max((len(cone) for cone in cones), default=0),
                len(union))

    joins.sort(key=lambda choice: (
        -convergence(choice)[0], -convergence(choice)[1],
        choice[0], int(choice[1])))
    for choice in joins:
        if len(proposals) >= limit:
            break
        candidate = clone_constructive(genome)
        if not append_logic_with_fanout(
                candidate, seeds, families, choice):
            continue
        signature = constructive_signature(candidate)
        if signature not in seen:
            seen.add(signature)
            proposals.append(candidate)

    # Then expose short physical bridges.  One bridge is one structural edit
    # but remains an explicit list of fixed component IDs in the genome.
    if len(proposals) < limit:
        bridges = _bridge_options(
            genome, seeds, families, limit=max(24, limit * 3))
        bridges.sort(key=lambda option: len(option[1]))
        for option in bridges:
            if len(proposals) >= limit:
                break
            candidate = clone_constructive(genome)
            if not _install_bridge_option(candidate, option):
                continue
            signature = constructive_signature(candidate)
            if signature not in seen:
                seen.add(signature)
                proposals.append(candidate)

    attempts = max(24, limit * 8)
    while len(proposals) < limit and attempts > 0:
        attempts -= 1
        candidate = clone_constructive(genome)
        if (layout_mutator is not None and random.random() < 0.12
                and layout_mutator(candidate)):
            changed = True
        else:
            changed = mutate_constructive_once(
                candidate, families, growth_seeds, focused_families)
        signature = constructive_signature(candidate)
        if changed and signature not in seen:
            seen.add(signature)
            proposals.append(candidate)
    return proposals
