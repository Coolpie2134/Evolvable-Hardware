"""Output-rooted context-window development for branched FNV genomes.

This is deliberately not inverse synthesis. A target supplies stable role names,
but never a desired truth table or target coordinate. Each role's genetic OUT
niche gives development a pole; local rules still decide the entire body.

Every gene is a context rule of the same shape the nervous and LUT substrates
use: the states required of the three neighbours, the state the cell must
already be in, and what it becomes. EMPTY is a state like any other, so a rule
may react to empty space, may build into it, may retype a cell, and may erase
one back to empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from substrates.nervous.hexgrid import hex_dirs, hex_frontier_cells

from .catalogue import BY_ID
from .genome import (
    DEPTH_BANDS, EMPTY_STATE, MAX_PLACEMENTS, OUT_STATE, PAD_STATE,
    ContextGene, ControlGene, OutputGene, input_seed_grid,
)
from .simulation import facing_direction, source_for_input


def _developmental_state_distance(left, right):
    """Distance between the PHYSICAL interfaces of two catalogue parts.

    Context genes regulate morphogenesis, not the truth table. Two components
    with the same pins occupy the same developmental niche even if one is AND
    and the other XOR (or one delay lasts a tick longer). That separation lets
    functional mutation tune an already-wired body without invalidating every
    downstream rule that observed the old numeric type.
    """
    a, b = BY_ID[int(left)], BY_ID[int(right)]
    if (a.id == b.id
            or (set(a.inputs) == set(b.inputs)
                and a.outputs == b.outputs)):
        return 0
    distance = 4 * (
        len(set(a.inputs).symmetric_difference(b.inputs))
        + len(set(a.outputs).symmetric_difference(b.outputs)))
    if a.family != b.family:
        distance += 2
    return max(1, distance)


DEVELOPMENTAL_STATE_DISTANCES = tuple(
    tuple(_developmental_state_distance(left, right)
          for right in range(len(BY_ID)))
    for left in range(len(BY_ID))
)


@dataclass(frozen=True)
class ConstructionTrace:
    grid: dict
    coordinates: dict
    active_ids: frozenset
    dormant_ids: frozenset
    depths: dict
    snapshots: tuple
    #: cell -> the branch label of the arm that owns it. Provenance, so a view
    #: can show WHICH branch built what rather than only what it is.
    owners: dict = field(default_factory=dict)
    #: cell -> how far along its OWN branch it sits, counted from that branch's
    #: spawn. This is the positional information rules key on.
    branch_depths: dict = field(default_factory=dict)
    #: cell -> the context gene whose most recent expression created/retyped
    #: it. This lets behavior-space mutation change the corresponding physical
    #: component allele without reverse-engineering a spatial circuit.
    builders: dict = field(default_factory=dict)


def branch_arms(chromosome):
    """Both arms' full gene lists, read outward from the centromere.

    ``split`` is a CENTROMERE: a fixed index with one arm on either side. Either
    arm may be empty. The order matters only for heredity - an arm is what
    crossover trades - never for where a gene applies.
    """
    genes = list(chromosome.genes)
    cut = max(0, min(int(chromosome.split), len(genes)))
    return (tuple(reversed(genes[:cut])), tuple(genes[cut:]))


def branch_growth_order(chromosome):
    """Each arm's CONTEXT rules, outward from the centromere."""
    return tuple(
        tuple(gene for gene in arm if isinstance(gene, ContextGene))
        for arm in branch_arms(chromosome))


def arm_control(chromosome, arm):
    """One arm's control gene, carrying its reach and its lifespan."""
    for gene in branch_arms(chromosome)[int(arm)]:
        if isinstance(gene, ControlGene):
            return gene
    return None


def arm_telomere(chromosome, arm):
    """One arm's lifespan, counted in cell CHANGES its own genes make."""
    control = arm_control(chromosome, arm)
    return 0 if control is None else max(0, int(control.telomere))


def arm_tolerance(chromosome, arm):
    """How far one arm's rules may sit from a neighbourhood and still apply."""
    control = arm_control(chromosome, arm)
    return 0 if control is None else max(0, int(control.tolerance))


def context_distance(wanted, seen):
    """How far a rule's context sits from a neighbourhood, or None if never.

    Summed physical-interface distance across the four slots. Component numbers
    are permanent part labels, not an ordinal scale: changing AND to XOR on the
    same pins is distance zero, while moving a pin is a real morphological
    change. EMPTY, PAD and OUT must match exactly, so tolerance can never invent
    or erase tissue nor confuse a grown cell with a terminal.
    """
    total = 0
    for a, b in zip(wanted, seen):
        if a == b:
            continue
        if a <= EMPTY_STATE or b <= EMPTY_STATE:
            return None
        total += DEVELOPMENTAL_STATE_DISTANCES[int(a)][int(b)]
    return total


def _upstream_parents(cell, label, grid, owners, branch_depths,
                      *, before_depth=None):
    """Owned cells whose declared input port is physically fed from ``cell``."""
    parents = []
    for destination in hex_frontier_cells(*cell):
        if owners.get(destination) != label or destination not in grid:
            continue
        depth = branch_depths.get(destination, 0)
        if before_depth is not None and depth >= before_depth:
            continue
        entry = BY_ID[grid[destination]]
        if any(source_for_input(destination, direction) == cell
               for direction in entry.inputs):
            parents.append(destination)
    return parents


def _reach(cell, label, grid, owners, branch_depths):
    """How deep ``cell`` would sit in ``label``'s branch, or None if out of it.

    A branch may only act on ground it already holds or on ground touching it.
    That is what makes an arm grow as a LIMB rather than as a scatter: without
    it a rule fires wherever its neighbourhood happens to occur, anywhere in the
    organism, and a branch is measured as several disconnected pieces.
    """
    if owners.get(cell) == label:
        return branch_depths.get(cell, 0)
    parents = _upstream_parents(cell, label, grid, owners, branch_depths)
    return min((branch_depths.get(parent, 0) + 1 for parent in parents),
               default=None)


def required_output_directions(cell, label, depth, grid, owners,
                               branch_depths):
    """Source-local ports that would connect this bud toward its arm root."""
    return tuple(sorted({
        facing_direction(cell, parent)
        for parent in _upstream_parents(
            cell, label, grid, owners, branch_depths,
            before_depth=int(depth))
    }))


def _drive_constraints(cell, label, depth, grid, owners, branch_depths):
    """State-independent part of reverse-growth polarity at one bud."""
    required = required_output_directions(
        cell, label, depth, grid, owners, branch_depths)
    for source in hex_frontier_cells(*cell):
        if (owners.get(source) != label
                or branch_depths.get(source, depth) >= depth):
            continue
        direction = facing_direction(source, cell)
        if direction in BY_ID[grid[source]].outputs:
            return required, True
    return required, False


def _drives_toward_root(cell, state, label, depth, grid, owners,
                        branch_depths, constraints=None):
    """Whether ``state`` preserves a directed tree toward its output root."""
    if int(depth) <= 0:
        return True
    required, blocked = (
        _drive_constraints(
            cell, label, depth, grid, owners, branch_depths)
        if constraints is None else constraints)
    if blocked:
        return False
    outputs = BY_ID[int(state)].outputs
    if not any(direction in outputs for direction in required):
        return False
    around = hex_dirs(*cell)
    # Every occupied same-arm output must go to a lower developmental depth.
    # Folding a limb beside its own root otherwise creates an accidental return
    # edge and turns an intended computation tree into a combinational cycle.
    for direction in outputs:
        destination = around[direction]
        if (owners.get(destination) == label
                and branch_depths.get(destination, depth) >= depth):
            return False
    return True


def _state_of(cell, grid, pads, output_sites=()):
    """What a gene sees: PAD, an unwritten OUT niche, a component, or empty.

    OUT is visible only until its writable site is occupied. Once an arm builds
    its root, ordinary component state is exposed so the root cannot be rewritten
    forever by the same spawn rule.
    """
    if cell in pads:
        return PAD_STATE
    if cell in output_sites and cell not in grid:
        return OUT_STATE
    return int(grid.get(cell, EMPTY_STATE))


def output_branch_sites(genome):
    """Stable arm label -> genetic output cell for this organism."""
    role_cells = {
        str(role): tuple(cell)
        for role, cell in (getattr(genome, "output_layout", ()) or ())}
    chromosome = getattr(genome, "output_chromosome", None)
    return {
        int(gene.branch_id): role_cells[str(gene.role)]
        for gene in (chromosome.genes if chromosome is not None else ())
        if isinstance(gene, OutputGene) and str(gene.role) in role_cells
    }


def _develop_branched(genome, seeds, *, snapshots=False):
    """Grow every living arm at once, one synchronous iteration at a time.

    Each iteration every gene of every living arm is matched against the grid as
    it stood at the START of that iteration, and fires at EVERY cell whose
    neighbourhood it matches exactly. Its arm owns both the rule's territory and
    its lifespan.

    Two things scope a gene besides its neighbourhood. It may only act on ground
    its own arm already holds or touches - so a branch grows as a limb, not a
    scatter - except for its one self=OUT rule at its assigned genetic output,
    which is how the branch starts. It may carry a DEPTH BAND, applying that far along
    its own branch, which is what lets one rule build a base and a tip
    differently.

    A cell claimed by several genes goes to the nearest match, and among equally
    near ones to BRANCH PRIORITY - the lowest-numbered arm, then the lowest gene
    id. Leaving contested cells empty instead was tried and it walled branches
    off from each other. A cell no gene matches keeps whatever it already is,
    which is what lets structure persist instead of being rewritten every
    iteration.

    A context gene differentiates one synchronous COHORT and is then spent for
    this organism. Every matching bud in that expression wave may change, but
    the rule cannot fire again on the next wave and extrude an arbitrary chain.
    This preserves the ontogenic amplification that makes a regulatory genome
    useful while bounding it to a spatial branch event. An arm's telomere is
    spent one per changed cell - birth, retype or erasure alike.

    Pads are the input interface and are never written; genes only read them as
    terminal cues. Output sites are writable once development reaches their
    assigned arm's root rule.
    """
    seeds = tuple(tuple(seed) for seed in seeds)
    grid = dict(input_seed_grid(seeds))
    pads = set(seeds)
    branch_roots = output_branch_sites(genome)
    output_sites = set(branch_roots.values()) - pads
    active = set()
    frames = [dict(grid)] if snapshots else []

    arms, owners, branch_depths, builders = [], {}, {}, {}
    for position, chromosome in enumerate(genome.chromosomes):
        for index, members in enumerate(branch_growth_order(chromosome)):
            arms.append({"genes": members,
                         "life": arm_telomere(chromosome, index),
                         "reach": arm_tolerance(chromosome, index),
                         "label": 2 * position + index + 1})

    while True:
        living = [arm for arm in arms if arm["life"] > 0 and arm["genes"]]
        if not living:
            break
        # A root may be genetically distant from the existing body, so output
        # niches must be explicit candidates rather than waiting for a frontier
        # that can never reach them.
        candidates = set(output_sites)
        candidates.update(cell for cell in grid if cell not in pads)
        # A reverse-grown branch buds only at the physical input ports of cells
        # already present. A binary gate therefore exposes two developmental
        # buds, a unary transport exposes one, and irrelevant side-neighbours do
        # not turn into chain-like decoration.
        for destination, state in list(grid.items()):
            if destination in pads:
                continue
            for direction in BY_ID[state].inputs:
                source = source_for_input(destination, direction)
                if source not in grid and source not in pads:
                    candidates.add(source)
        room = MAX_PLACEMENTS - (len(grid) - len(pads))

        proposals = {}
        for cell in candidates:
            around = hex_dirs(*cell)
            context = (
                _state_of(around['L'], grid, pads, output_sites),
                _state_of(around['R'], grid, pads, output_sites),
                _state_of(around['D'], grid, pads, output_sites),
                _state_of(cell, grid, pads, output_sites),
            )
            reach, drive_constraints, matches, best = {}, {}, [], None
            for position, arm in enumerate(living):
                label = arm["label"]
                for gene in arm["genes"]:
                    if int(gene.gene_id) in active:
                        continue
                    if gene.spawns_output():
                        # OUT is not a global wildcard. Only this arm, at the
                        # site its OutputGene owns, can claim initial territory.
                        if (branch_roots.get(label) != cell
                                or context[3] != OUT_STATE):
                            continue
                        depth = 0
                    else:
                        if label not in reach:
                            reach[label] = _reach(
                                cell, label, grid, owners, branch_depths)
                        depth = reach[label]
                        if depth is None:
                            continue
                    if not gene.applies_at(min(depth, DEPTH_BANDS - 1)):
                        continue
                    # OUT already identifies both the developmental role and
                    # its one writable root site. Requiring the root gene to
                    # memorize whatever happened to surround that coordinate
                    # coupled output-position mutation to a simultaneous rule
                    # rewrite: moving a root usually killed the whole arm.
                    # Descendants remain fully context addressed; only the
                    # germline expression is position-relative.
                    distance = (
                        0 if gene.spawns_output()
                        else context_distance(gene.context, context))
                    if distance is None or distance > arm["reach"]:
                        continue
                    # Every non-root birth/retype must remain a physical source
                    # for something nearer this arm's output. This is polarity,
                    # not routing: the local rule still chooses the component,
                    # and nothing chooses a route toward a target pad.
                    if (int(gene.self_out) != EMPTY_STATE
                            and int(depth) > 0):
                        if label not in drive_constraints:
                            drive_constraints[label] = _drive_constraints(
                                cell, label, depth, grid, owners,
                                branch_depths)
                        if not _drives_toward_root(
                                cell, gene.self_out, label, depth, grid,
                                owners, branch_depths,
                                drive_constraints[label]):
                            continue
                    if best is None or distance < best:
                        best = distance
                    matches.append((distance, position, gene, depth))
            # Nearest wins, then branch priority settles it: the lowest arm,
            # then the lowest gene id. Deterministic, and it lets one branch
            # build into ground another is contesting rather than both stalling.
            winner = None
            for distance, position, gene, depth in matches:
                if distance != best:
                    continue
                key = (living[position]["label"], int(gene.gene_id))
                if winner is None or key < winner[0]:
                    winner = (key, position, gene, depth)
            if winner is not None:
                _key, position, gene, depth = winner
                proposals[cell] = {
                    "state": int(gene.self_out), "arm": position,
                    "depth": depth, "gene": int(gene.gene_id),
                    "distance": distance}

        changed, spent = {}, {}
        for cell in sorted(proposals):
            claim = proposals[cell]
            state = claim["state"]
            # Compare with physical storage, not the OUT sentinel. An erase at
            # an empty niche is a no-op and must not burn the arm's lifespan.
            if state == int(grid.get(cell, EMPTY_STATE)):
                continue                   # nothing to do, so nothing spent
            if state and cell not in grid:
                if room <= 0:
                    continue               # body is full; erasures still run
                room -= 1
            changed[cell] = state
            active.add(claim["gene"])
            # Provenance: the arm whose gene won this cell. Only that arm made
            # a change, so only that arm pays for it.
            if state:
                owners[cell] = living[claim["arm"]]["label"]
                branch_depths[cell] = claim["depth"]
                builders[cell] = claim["gene"]
            else:
                owners.pop(cell, None)
                branch_depths.pop(cell, None)
                builders.pop(cell, None)
            spent[claim["arm"]] = spent.get(claim["arm"], 0) + 1
        if not changed:
            break
        for cell, state in changed.items():
            if state:
                grid[cell] = state
            else:
                grid.pop(cell, None)
        for position, arm in enumerate(living):
            # May go negative: every change of the iteration stands, and the arm
            # is simply dead from here.
            arm["life"] -= spent.get(position, 0)
        if snapshots:
            frames.append(dict(grid))

    coordinates = {("cell",) + cell: cell for cell in grid if cell not in pads}
    depths = _radial_depths(grid, pads)
    every = {int(gene.gene_id) for arm in arms for gene in arm["genes"]}
    if snapshots and (not frames or frames[-1] != grid):
        frames.append(dict(grid))
    return ConstructionTrace(
        grid, coordinates, frozenset(active), frozenset(every - active),
        depths, tuple(frames),
        {cell: label for cell, label in owners.items() if cell in grid},
        {cell: value for cell, value in branch_depths.items() if cell in grid},
        {cell: value for cell, value in builders.items() if cell in grid})


def _radial_depths(grid, pads):
    """Steps from the nearest pad, through live cells only.

    Development no longer follows a dependency chain - any cell may be retyped
    or erased - so the settling window is derived from how far a signal has to
    travel, which on this lattice is adjacency distance from an input.
    """
    depths = {}
    frontier = [cell for cell in pads if cell in grid]
    seen = set(frontier)
    for cell in frontier:
        depths[("cell",) + cell] = 0
    step = 0
    while frontier:
        step += 1
        nxt = []
        for cell in frontier:
            for neighbour in hex_frontier_cells(*cell):
                if neighbour in seen or neighbour not in grid:
                    continue
                seen.add(neighbour)
                depths[("cell",) + neighbour] = step
                nxt.append(neighbour)
        frontier = nxt
    return depths


def _development_key(genome, seeds):
    """Immutable key for every field the developmental interpreter reads.

    Construction operators inspect the phenotype several times around one edit
    (observed contexts, open buds, and before/after viability). Re-growing an
    unchanged body for every inspection dominated runtime. This key is cheap
    and, unlike manual invalidation, cannot become stale after in-place edits.
    """
    arms = []
    for chromosome in genome.chromosomes:
        genes = []
        for gene in chromosome.genes:
            if isinstance(gene, ContextGene):
                genes.append((
                    "context", int(gene.gene_id), int(gene.ctx_l),
                    int(gene.ctx_r), int(gene.ctx_d), int(gene.self_in),
                    int(gene.self_out), int(gene.branch_id), int(gene.depth)))
            elif isinstance(gene, ControlGene):
                genes.append((
                    "control", int(gene.gene_id), int(gene.tolerance),
                    int(gene.telomere), int(gene.branch_id)))
            else:
                genes.append((type(gene).__name__, repr(gene)))
        arms.append((int(chromosome.split), tuple(genes)))
    return (
        tuple(tuple(cell) for cell in seeds),
        tuple((str(role), tuple(cell)) for role, cell in
              (getattr(genome, "output_layout", ()) or ())),
        tuple(arms),
    )


def develop_constructive(genome, seeds, *, snapshots=False):
    """Grow one organism. See :func:`_develop_branched` for the rules."""
    normalized_seeds = tuple(tuple(seed) for seed in seeds)
    if snapshots:
        return _develop_branched(genome, normalized_seeds, snapshots=True)
    key = _development_key(genome, normalized_seeds)
    if getattr(genome, "_fnv_development_cache_key", None) == key:
        cached = getattr(genome, "_fnv_development_cache", None)
        if cached is not None:
            return cached
    trace = _develop_branched(genome, normalized_seeds, snapshots=False)
    genome._fnv_development_cache_key = key
    genome._fnv_development_cache = trace
    return trace


def constructive_depth(genome, seeds):
    trace = develop_constructive(genome, seeds)
    return max(trace.depths.values(), default=0)


def grow_functional(genome, seeds, grid_size=None, iters=None):
    """Grown body as a plain ``cell -> component id`` grid.

    ``grid_size`` and ``iters`` are accepted and ignored: growth is bounded by
    each arm's own telomere and by MAX_PLACEMENTS, never by a display grid or an
    iteration count imposed from outside.
    """
    return develop_constructive(genome, tuple(seeds)).grid


def grow_functional_snapshots(genome, seeds, grid_size=None, iters=None):
    """One grid per growth iteration, for the growth view.

    Development is synchronous, so a frame is a whole iteration: every gene that
    fired did so against the previous frame.
    """
    return list(develop_constructive(
        genome, tuple(seeds), snapshots=True).snapshots)
