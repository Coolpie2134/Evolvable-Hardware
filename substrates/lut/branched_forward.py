"""
substrates/lut/branched_forward.py - the SEED-ROOTED branched encoding (variation 2).

Two branched encodings now exist for the LUT array. They differ in which end of
the circuit the genome commits to first, and that single choice changes what a
genome can express:

  VARIATION 1 - output-rooted (substrates/lut/branched.py)
      One arm per output role. Each arm starts at its role's genetic output site
      and grows BACKWARD along the directions its cells read, toward the input
      pads. Every cell an arm places is one its output can already hear.

  VARIATION 2 - seed-rooted (this module)
      2n arms, n = chromosome count, so every arm is a root whether or not the
      target has a matching output. Each root is placed by its own chromosome
      and grows FORWARD along the directions its cells DRIVE, toward the genetic
      output sites. Inputs are not placed at all during growth: they are
      attached afterwards, alternating A, B, C, A, B, C... around the finished
      body's perimeter.

WHY BOTH ARE WORTH HAVING
-------------------------
Variation 1 guarantees every cell is heard by an output but makes each arm
responsible for finding one specific input pad. That is the suspected cause of
the open LUT memory defect, where evolution converges BELOW a hand-built
one-cell circuit: nothing pressures an arm toward the RIGHT pad, so it reliably
binds the wrong one.

Variation 2 inverts the first property. Arms are no longer tied one-to-one to
outputs, so several may cooperate on one role or explore structure that serves
none - and what it gives up is variation 1's guarantee that a placed cell is
heard by anything at all.

THE MULTI-OUTPUT PROBLEM (open)
-------------------------------
Measured on a two-output target under variation 1: only 3 organisms in 35 had
BOTH outputs live - 15 had none, 17 had exactly one. A target whose two outputs
must be computed differently therefore starts from a population where 91% of
organisms cannot score on both at all.

Two attempted fixes were measured and did NOT work, and are recorded here so
they are not retried blindly: giving each arm its own tap of every logical input
moved it 3 -> 4 of 35 (noise), and a chirality field on the gene changed no
bodies whatsoever. What DID move it is selecting fresh organisms on their real
simulated output liveness rather than on a structural reachability proxy -
26% -> 40% both-live - because the proxy reports 'both driven' for organisms the
simulator then finds silent. That costs ~2.9s per genome.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .branched import (DEPTH_ANY, DEPTH_BANDS, EMPTY_CELL, MAX_PLACEMENTS,
                       OUT_CELL, PAD_CELL, SENTINEL_DISTANCE,
                       BranchedLutChromosome, LutContextGene, LutControlGene,
                       LutInputGene, LutIoChromosome, LutOutputGene,
                       BranchedLutTrace, DIRECTIONS, OPPOSITE, bearing_cell,
                       catalogue, cell_distance, cell_outputs, cell_sources,
                       neighbours)

#: A root niche: writable exactly once, by the arm that owns it. The seed-rooted
#: twin of OUT_CELL, and distinct from it so a rule cannot confuse the place an
#: arm STARTS with the place it is trying to REACH.
ROOT_CELL = 'ROOT'

@dataclass
class LutRootGene:
    """Where one arm's seed sits, as a bearing and distance from the origin.

    On its own chromosome rather than alongside the ports: a root is not an
    interface, it is where a limb begins, and moving a seed should not perturb
    the output sites the arms are growing toward.
    """

    bearing: int = 0
    distance: int = 1
    branch_id: int = 1


@dataclass
class LutRootChromosome:
    """The seed-placement chromosome, read before growth."""

    roots: List[LutRootGene] = field(default_factory=list)
    tag: int = 0

    @property
    def genes(self):
        return list(self.roots)


@dataclass
class SeedRootedLutGenome:
    """A seed-rooted organism.

    Deliberately a distinct type from ``BranchedLutGenome`` rather than a mode
    flag on it: the two encodings mean different things by a root, grow in
    opposite directions, and bind their inputs by different mechanisms, so a
    shared type would need a branch at every one of those points.
    """

    chromosomes: List[BranchedLutChromosome] = field(default_factory=list)
    #: Where the seeds go.
    root_chromosome: LutRootChromosome = field(
        default_factory=LutRootChromosome)
    #: Where the ports go, resolved before growth: output 0 at the origin,
    #: every other output and every input pad an evolved bearing and distance
    #: from it.
    io_chromosome: LutIoChromosome = field(default_factory=LutIoChromosome)
    families: Tuple[str, ...] = ()
    next_gene_id: int = 1

    @property
    def outputs(self):
        return self.io_chromosome.outputs

    @property
    def inputs(self):
        return self.io_chromosome.inputs

    @property
    def roots(self):
        return self.root_chromosome.roots

    def arm(self, label):
        index, half = divmod(int(label) - 1, 2)
        if not 0 <= index < len(self.chromosomes):
            return None, None
        chromosome = self.chromosomes[index]
        members = [gene for gene in chromosome.genes
                   if int(gene.branch_id) == int(label)]
        control = (chromosome.controls[half]
                   if half < len(chromosome.controls) else LutControlGene())
        return members, control

    @property
    def arm_labels(self):
        """2n labels - EVERY arm is a root here, not only those with a role."""
        return tuple(range(1, 2 * len(self.chromosomes) + 1))


def seed_root_sites(genome):
    """{branch label: cell} for each arm's seed, placed before growth."""
    taken = set()
    sites = {}
    for gene in genome.roots:
        cell = bearing_cell(gene.bearing, gene.distance, taken)
        if cell is None:
            continue
        sites[int(gene.branch_id)] = cell
        taken.add(cell)
    return sites


def output_sites(genome, taken=()):
    """{role: cell} for the genetic output sites the arms grow toward.

    OUTPUT 0 IS THE ORIGIN. It is the coordinate gauge for this encoding, the
    way input 0 is in the output-rooted twin: with the inputs distributed around
    the perimeter, the centre is the one landmark every arm can address without
    an allele, and pinning it means a genome never has to spend mutations
    discovering where to deliver its first answer. Every other role is an
    evolved bearing and distance from it.
    """
    claimed = set(taken)
    sites = {}
    for index, gene in enumerate(genome.outputs):
        if index == 0:
            cell = (0, 0)
        else:
            cell = bearing_cell(gene.bearing, gene.distance, claimed)
        if cell is None:
            continue
        sites[str(gene.role)] = cell
        claimed.add(cell)
    return sites


def _distance(wanted, found, families=None):
    """Context distance that knows about the ROOT sentinel.

    ``branched.cell_distance`` recognises PAD / OUT / EMPTY. This encoding adds
    a fourth kind of thing, and without teaching the metric about it the string
    'ROOT' reached the table comparison and was parsed character by character.
    A seed niche is a different KIND of site from a built cell, so it sits at
    SENTINEL_DISTANCE like the others rather than on the interface scale.
    """
    if wanted == found:
        return 0
    if wanted == ROOT_CELL or found == ROOT_CELL:
        return SENTINEL_DISTANCE
    return cell_distance(wanted, found, families)


def _is_seed_rule(gene):
    """A rule that opens an arm at its seed.

    Variation 1 marks its root rule with ``self_in == OUT_CELL``; here the
    equivalent marker is ROOT_CELL, because an arm STARTS at a seed and only
    later reaches an output. Sharing ``spawns_output`` would have made the two
    encodings disagree about which niche a rule claims.
    """
    return gene.self_in == ROOT_CELL


def input_pads(genome, taken=()):
    """Pad cells, placed before growth.

    Unlike the output-rooted twin there is no pad pinned at the origin - output
    0 holds that spot here - so every pad is an evolved bearing and distance,
    placed after the seeds and outputs have claimed their sites.
    """
    claimed = set(taken)
    pads = []
    for gene in genome.inputs:
        cell = bearing_cell(gene.bearing, gene.distance, claimed)
        if cell is not None:
            pads.append(cell)
            claimed.add(cell)
    return tuple(pads)


def _cell_of(cell, grid, root_cells, out_cells, pads=()):
    """What a rule sees at ``cell``.

    ``root_cells`` / ``out_cells`` are SETS OF CELLS, not the label->cell and
    role->cell maps: membership-testing a dict checks its keys, so passing the
    maps here silently compared coordinates against arm labels and role names
    and no seed niche was ever visible - every organism came out empty.

    A pad reads PAD, an unbuilt seed reads ROOT, an unbuilt output niche reads
    OUT, everything else is a cell or empty ground.
    """
    if cell in pads:
        return PAD_CELL
    if cell in grid:
        return grid[cell]
    if cell in root_cells:
        return ROOT_CELL
    if cell in out_cells:
        return OUT_CELL
    return EMPTY_CELL


def _upstream_parents(cell, label, owners, depths, before_depth=None):
    """Same-arm neighbours nearer this arm's SEED than ``cell``."""
    parents = []
    for direction, neighbour in neighbours(cell).items():
        if owners.get(neighbour) != label:
            continue
        if before_depth is not None and depths.get(
                neighbour, before_depth) >= before_depth:
            continue
        parents.append((direction, neighbour))
    return parents


def arm_reach(cell, label, owners, depths):
    """How far ``cell`` sits from this arm's seed, or None if outside the arm."""
    if owners.get(cell) == label:
        return depths.get(cell, 0)
    parents = _upstream_parents(cell, label, owners, depths)
    return min((depths.get(n, 0) + 1 for _d, n in parents), default=None)


def required_input_directions(cell, label, depth, owners, depths):
    """Directions a cell here must READ to be fed by the limb it joins.

    The forward-growth mirror of variation 1's ``required_output_directions``.
    There, a new part had to DRIVE toward its parent; here it must be DRIVEN BY
    it, so the requirement is on the part's inputs rather than its outputs.
    """
    return tuple(sorted({
        direction for direction, _n in _upstream_parents(
            cell, label, owners, depths, before_depth=int(depth))}))


def fed_by_limb(cell, state, label, depth, grid, owners, depths):
    """Whether a part placed here is actually driven by the limb it joins.

    Two conditions, mirroring variation 1's polarity rule:

    * the part must READ at least one direction holding a shallower same-arm
      cell, and that neighbour must actually drive back toward this one - a
      part nothing feeds is inert wherever it sits;
    * feedback is allowed, exactly as in variation 1. The LUT array stores
      state topologically, so forbidding cycles would forbid memory (see
      branched.drives_toward_root).
    """
    if int(depth) <= 0:
        return True                       # the seed answers to nobody
    required = set(
        required_input_directions(cell, label, depth, owners, depths))
    reads = cell_sources(state)
    if not any(direction in reads for direction in required):
        return False
    # ...and the parent must genuinely drive this way, or the "connection" is
    # one-sided: this cell listens to a neighbour that never speaks to it.
    around = neighbours(cell)
    for direction in required:
        if direction not in reads:
            continue
        parent = around[direction]
        if parent in grid and OPPOSITE[direction] in cell_outputs(grid[parent]):
            return True
    return False


def _arrival(cell, label, depth, owners, depths, headings):
    """(turn, arrival direction) for a bud, or (None, None) at a limb's start.

    The turn is retained but no longer gates anything: a chirality field on the
    gene was implemented and measured, and over 16 organisms it changed exactly
    zero bodies (1673 gate checks, 0 blocks). At these body sizes a rule already
    fires at essentially one site, so it needs no further discrimination - and
    the bodies were already 62% mirror-asymmetric, so the symmetry the field was
    meant to break was not there to begin with.

    The parent is the shallower same-arm neighbour the limb came from; the
    arrival direction is the step from it to this cell, and the turn compares
    that against the direction the parent itself arrived on. Ties among several
    eligible parents go to the lowest direction name so development stays
    deterministic.
    """
    parents = _upstream_parents(cell, label, owners, depths,
                                before_depth=int(depth))
    if not parents:
        return None, None
    direction, parent = min(parents)
    # The step from parent to cell is the OPPOSITE of the direction the parent
    # lies in when viewed from the cell.
    arrival = OPPOSITE[direction]
    return None, arrival


def growth_candidates(grid, owners, roots, outs):
    """Every site development may write, for any arm.

    The unbuilt seeds and unbuilt output niches, every built cell, and the sites
    each built cell DRIVES into. Forward growth: a cell's frontier is where its
    signal goes, not where it comes from.
    """
    sites = set(roots.values()) - set(grid)      # unbuilt seeds
    sites.update(set(outs.values()) - set(grid))  # unbuilt output niches
    sites.update(grid)                            # retype / erase
    for cell, state in grid.items():
        around = neighbours(cell)
        for direction in cell_outputs(state):
            sites.add(around[direction])
    return sorted(sites)


def develop_seed_rooted_lut(genome, *, snapshots=False):
    """Grow every living arm forward from its seed, one synchronous wave at a time.

    Same contest rules as variation 1 - nearest match, then lowest arm label,
    then lowest gene id; a gene differentiates one cohort and is then spent; an
    arm's telomere is its lifespan in changed cells - with two differences: the
    frontier runs along each cell's OUTPUT directions, and a new part must be
    fed by its parent rather than feed it.
    """
    roots = seed_root_sites(genome)
    outs = output_sites(genome, taken=set(roots.values()))
    root_cells, out_cells = set(roots.values()), set(outs.values())
    pads = set(input_pads(genome, taken=root_cells | out_cells))
    grid: Dict[Tuple[int, int], tuple] = {}
    owners: Dict[Tuple[int, int], int] = {}
    depths: Dict[Tuple[int, int], int] = {}
    #: cell -> the direction the limb was travelling when it reached that cell.
    #: A turn is measured against this, so it has to be remembered per cell
    #: rather than recomputed: the parent's own arrival direction is not
    #: recoverable from the finished grid once several limbs overlap.
    headings: Dict[Tuple[int, int], str] = {}
    frames = [dict(grid)] if snapshots else []
    families = genome.families or None

    arms = []
    for label in genome.arm_labels:
        members, control = genome.arm(label)
        if not members:
            continue
        arms.append({'label': label, 'genes': list(members),
                     'life': int(control.telomere),
                     'reach': max(0, int(control.tolerance)),
                     'spent': set()})

    while True:
        living = [arm for arm in arms
                  if arm['life'] > 0 and len(arm['spent']) < len(arm['genes'])]
        if not living or len(grid) >= MAX_PLACEMENTS:
            break

        proposals = {}
        for cell in growth_candidates(grid, owners, roots, outs):
            around = neighbours(cell)
            context = tuple(
                _cell_of(around[d], grid, root_cells, out_cells, pads)
                for d in DIRECTIONS) + (
                    _cell_of(cell, grid, root_cells, out_cells, pads),)
            reach = {}
            for arm in living:
                label = arm['label']
                for gene in arm['genes']:
                    if gene.gene_id in arm['spent']:
                        continue
                    if _is_seed_rule(gene):
                        # The arm's SEED rule: only this arm, only at the site
                        # its own root gene owns, and matched at distance 0 for
                        # the same reason variation 1's root rule is - so moving
                        # a seed does not also require rewriting the rule.
                        if (roots.get(label) != cell
                                or context[4] != ROOT_CELL):
                            continue
                        depth, distance = 0, 0
                    else:
                        if label not in reach:
                            reach[label] = arm_reach(
                                cell, label, owners, depths)
                        depth = reach[label]
                        if depth is None:
                            continue
                        distance = sum(
                            _distance(wanted, found, families)
                            for wanted, found in zip(gene.context, context))
                        if distance > arm['reach']:
                            continue
                    if not gene.applies_at(min(depth, DEPTH_BANDS - 1)):
                        continue
                    _turn, arrival = _arrival(cell, label, depth, owners,
                                              depths, headings)
                    if (gene.self_out != EMPTY_CELL
                            and not fed_by_limb(cell, gene.self_out, label,
                                                depth, grid, owners, depths)):
                        continue
                    key = (distance, label, gene.gene_id)
                    if cell not in proposals or key < proposals[cell][0]:
                        proposals[cell] = (key, arm, gene, depth, arrival)

        if not proposals:
            break

        changed, fired = False, set()
        for cell, ((_d, label, _gid), arm, gene, depth, arrival) in sorted(
                proposals.items()):
            if arm['life'] <= 0 or gene.gene_id in arm['spent']:
                continue
            new_cell = gene.self_out
            if tuple(new_cell) == tuple(grid.get(cell, EMPTY_CELL)):
                continue
            if new_cell == EMPTY_CELL:
                del grid[cell]
                owners.pop(cell, None)
                depths.pop(cell, None)
                headings.pop(cell, None)
            else:
                if cell not in grid and len(grid) >= MAX_PLACEMENTS:
                    continue
                grid[cell] = tuple(new_cell)
                owners[cell] = label
                depths[cell] = int(depth)
                if arrival is not None:
                    headings[cell] = arrival
            arm['life'] -= 1
            fired.add((label, gene.gene_id))
            changed = True

        for arm in arms:
            for label, gene_id in fired:
                if label == arm['label']:
                    arm['spent'].add(gene_id)
        if snapshots:
            frames.append(dict(grid))
        if not changed:
            break

    trace = BranchedLutTrace(grid=grid, owners=owners, depths=depths,
                             snapshots=frames)
    trace.headings = headings
    return trace




# -- construction and variation -------------------------------------------------

#: Arm lifespan in changed cells. Matches runtime.config's
#: DEFAULT_LUT_MAX_TELOMERE, which is what the controller actually passes; the
#: module default used to say 32 and disagreed with every real run. LUT arms
#: want SHORT lifespans - a body here is a couple of dozen cells across four
#: arms, so a lifespan of 30 only ever buys sprawl.
MAX_TELOMERE = 8
MAX_TOLERANCE = 3
#: How far a seed may sit from the origin. Deliberately tight: the organism is
#: only ever a couple of dozen cells, so a seed parked six steps out simply
#: never joins the rest of the body.
MAX_ROOT_DISTANCE = 3
#: ...and the same ceiling for the output roles that are not pinned at (0, 0).
MAX_IO_DISTANCE = 3
ERASE_PROBABILITY = 0.08
DRIVE_WEIGHTS = (1, 1, 1, 2, 2, 3)


def random_cell(families, allow_empty=True, required=()):
    """A cell from the enabled banks that is fed by its parent.

    ``required`` are directions it must READ - the forward-growth mirror of
    variation 1's draw, where a part had to DRIVE toward its parent. Here it
    must be driven BY its parent instead, so the constraint sits on its inputs.
    """
    if allow_empty and random.random() < ERASE_PROBABILITY:
        return EMPTY_CELL
    entries = catalogue(families)
    if not entries:
        return EMPTY_CELL
    wanted = [d for d in required if d in DIRECTIONS]
    for _attempt in range(12):
        tables = [0, 0, 0, 0]
        for index in random.sample(range(4), random.choice(DRIVE_WEIGHTS)):
            tables[index] = random.choice(entries)[1]
        cell = tuple(tables)
        if not wanted or cell_sources(cell) & set(wanted):
            return cell
    return tuple(tables)


def observed_contexts(genome, label, trace=None):
    """Sorted [(context, band, required_input_dirs)] this arm can act on."""
    if trace is None:
        trace = develop_seed_rooted_lut(genome)
    grid, owners, depths = trace.grid, trace.owners, trace.depths
    roots = seed_root_sites(genome)
    outs = output_sites(genome, taken=set(roots.values()))
    root_cells, out_cells = set(roots.values()), set(outs.values())
    pads = set(input_pads(genome, taken=root_cells | out_cells))
    headings = getattr(trace, 'headings', {}) or {}
    seed = roots.get(label)
    found = set()
    for cell in growth_candidates(grid, owners, roots, outs):
        if cell == seed and cell not in grid:
            depth = 0
        else:
            depth = arm_reach(cell, label, owners, depths)
            if depth is None:
                continue
        around = neighbours(cell)
        context = tuple(_cell_of(around[d], grid, root_cells, out_cells, pads)
                        for d in DIRECTIONS) + (
                            _cell_of(cell, grid, root_cells, out_cells, pads),)
        found.add((context, min(depth, DEPTH_BANDS - 1),
                   required_input_directions(cell, label, depth,
                                             owners, depths)))
    return sorted(found, key=repr)


def _arm_has_seed(genome, label):
    members, _control = genome.arm(label)
    return any(_is_seed_rule(gene) for gene in (members or ()))


def random_gene(genome, gene_id, label, *, allow_seed=True, trace=None):
    families = genome.families or None
    seen = observed_contexts(genome, label, trace=trace)
    seeds = [entry for entry in seen if entry[0][4] == ROOT_CELL]
    if allow_seed and seeds:
        context, _band, _dirs = random.choice(seeds)
        return LutContextGene(
            gene_id, context[0], context[1], context[2], context[3],
            ROOT_CELL, random_cell(families, allow_empty=False),
            label, DEPTH_ANY)
    body = [entry for entry in seen if entry[0][4] != ROOT_CELL]
    if not body:
        return None
    context, band, dirs = random.choice(body)
    depth = band if random.random() < 0.5 else DEPTH_ANY
    return LutContextGene(
        gene_id, context[0], context[1], context[2], context[3], context[4],
        random_cell(families, required=dirs), label, depth)


def random_seed_rooted_lut_genome(n_chroms=2, n_inputs=2, output_roles=('Q',),
                                  families=None, blocks=24,
                                  max_telomere=MAX_TELOMERE):
    """A fresh seed-rooted organism whose arms actually build something.

    EVERY arm gets a seed - 2n of them - regardless of how many output roles
    the target has, which is the point of this variation: arms are not bound
    one-to-one to outputs.
    """
    from .functions import normalise_function_families
    roles = tuple(str(role) for role in output_roles)
    n_arms = 2 * int(n_chroms)
    enabled = normalise_function_families(families)
    genome = SeedRootedLutGenome(
        chromosomes=[
            BranchedLutChromosome(controls=[
                LutControlGene(tolerance=random.choice((0, 1, 1, 2)),
                               telomere=random.randint(2, max_telomere))
                for _ in range(2)])
            for _ in range(int(n_chroms))],
        root_chromosome=LutRootChromosome(roots=[
            LutRootGene(bearing=random.randrange(8),
                        distance=random.randint(1, MAX_ROOT_DISTANCE),
                        branch_id=index + 1)
            for index in range(n_arms)]),
        io_chromosome=LutIoChromosome(
            inputs=[LutInputGene(bearing=random.randrange(8),
                                 distance=random.randint(1, MAX_IO_DISTANCE))
                    for _ in range(max(0, int(n_inputs)))],
            outputs=[LutOutputGene(role=role, bearing=random.randrange(8),
                                   distance=random.randint(1, MAX_IO_DISTANCE),
                                   branch_id=index + 1)
                     for index, role in enumerate(roles)]),
        families=tuple(enabled),
        next_gene_id=1)

    trace = develop_seed_rooted_lut(genome)
    growing = set(genome.arm_labels)
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
                                   allow_seed=not _arm_has_seed(genome, label),
                                   trace=trace)
                if gene is None:
                    break
                chromosome.genes.append(gene)
                grown = develop_seed_rooted_lut(genome)
                if grown.grid != trace.grid:
                    genome.next_gene_id += 1
                    trace = grown
                    placed = True
                    break
                chromosome.genes.pop()
            if not placed:
                growing.discard(label)
    return genome
