"""
substrates/lut/branched.py - FNV's branched, output-rooted encoding on the
four-directional LUT array.

WHY A CATALOGUE, NOT ALL 65536 TABLES
-------------------------------------
A LUT cell is four 16-bit tables, one per output direction, so the raw state
space is 2^64 per cell. Nothing useful can be said about "nearly the same cell"
in that space, and the branched encoding depends on being able to say it: an
arm's tolerance is a budget over how far a context may differ, and a budget is
meaningless without a metric that tracks function.

So this encoding installs only tables from the run's ENABLED FUNCTION BANKS
(substrates/lut/functions.py) - ROUTING, AND, OR, XOR, VETO, THRESHOLD, MUX -
79 named tables in total across the seven banks. That gives the LUT array the
same shape of alphabet FNV has: a finite catalogue of physically meaningful
parts, each with a known interface. Two consequences fall straight out:

  * GROWTH BUDS. A cell's buds are the directions its tables actually depend on.
    A 2-input AND exposes two, a routing table one - exactly as an FNV binary
    gate exposes two and a unary transport one. Reverse growth from the output
    roots works unchanged.
  * TOLERANCE. Distance compares each direction's SUPPORT - which neighbours
    the table reads - so "nearly the same cell" means "a gate in the same place
    wired to nearly the same neighbours", whatever it computes with them.
    Hamming distance over the raw tables would instead call an AND and an XOR
    adjacent whenever their bit patterns happened to be close.

The banks are already quiescent at all-zero input - functions.py refuses to load
otherwise - which is the same power-on property FNV relies on and the reason a
target demanding output from silence is unreachable here too.

WHAT DIFFERS FROM THE HEX PORT
------------------------------
Four neighbours rather than three, so an exact context match is rarer: four
context fields must agree instead of three. Tolerance therefore carries more of
the work here, though it stays in the same small range - an arm's budget is
normally 1.

INTERFACE FIRST
---------------
One arm per output role (two arms per chromosome; spare arms stay empty), each
with its own tolerance and lifespan, plus a separate PLACEMENT chromosome that
fixes every port before growth: input 0 at the origin, every other pad and every
output root an evolved bearing and distance from it.

STATUS
------
This is THE LUT encoding for source-pad runs: the desktop controller and the
headless driver both breed it, and it saves and loads as a `branched_v1`
checkpoint document. Exterior-edge I/O still seeds natively, because its inputs
are drivers outside the body rather than pads an arm can grow toward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

from .functions import (FAMILY_TABLES, INPUT_BITS, INPUTS,
                        normalise_function_families)

#: An absent cell. Matches the native encoding's dead state.
EMPTY_CELL = (0, 0, 0, 0)
#: Read-only input pad, and an unbuilt genetic output niche.
PAD_CELL = 'PAD'
OUT_CELL = 'OUT'

DEPTH_BANDS = 4
DEPTH_ANY = -1
MAX_PLACEMENTS = 128

#: Output directions of a cell, in the order the grid stores them.
DIRECTIONS = ('N', 'S', 'E', 'W')
#: Which neighbour lies in each direction, on the square lattice.
STEP = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


def neighbours(cell):
    """{direction: neighbour cell} - the square analogue of hex_dirs."""
    x, y = cell
    return {name: (x + dx, y + dy) for name, (dx, dy) in STEP.items()}


@lru_cache(maxsize=None)
def table_support(table):
    """Directions a 16-bit table's output actually depends on.

    Computed, not declared: a table depends on an input exactly when flipping
    that input changes the result for some assignment of the others. This is
    what makes a cell's growth buds honest even for a hand-supplied table.
    """
    table = int(table) & 0xFFFF
    support = set()
    for name, bit in INPUT_BITS.items():
        for index in range(16):
            if index & (1 << bit):
                continue
            if ((table >> index) & 1) != ((table >> (index | (1 << bit))) & 1):
                support.add(name)
                break
    return frozenset(support)


def catalogue(families=None):
    """Sorted (family, table) pairs the run's enabled banks allow (cached).

    Cached because ``cell_distance`` asks for a table's bank once per direction
    per candidate cell per developmental iteration: rebuilding and re-sorting
    the catalogue there dominated growth outright.
    """
    return _catalogue(None if families is None else tuple(families))


@lru_cache(maxsize=None)
def _family_index(families):
    """{table: family} for the enabled banks - the lookup cell_distance wants."""
    index = {}
    for family, table in catalogue(families):
        index.setdefault(table, family)
    return index


@lru_cache(maxsize=None)
def _catalogue(families):
    """Sorted (family, table) pairs the run's enabled banks allow.

    UNRESTRICTED means "all 65536 tables" to the native encoding, which this one
    cannot use: tolerance is a budget over how far a context may differ, and
    that budget is meaningless without a metric that tracks function. So here it
    means every NAMED bank instead - the widest alphabet that still has a family
    per table. Selecting it alongside specific banks therefore widens the
    catalogue to all of them rather than silently emptying it.

    Sorted, and returned as a tuple: this feeds random draws during
    construction, and iterating a set keyed by strings is what made FNV runs
    unreproducible for months.
    """
    enabled = normalise_function_families(families)
    if 'UNRESTRICTED' in enabled:
        enabled = tuple(sorted(FAMILY_TABLES))
    entries = set()
    for family in enabled:
        for table in FAMILY_TABLES.get(family, ()):
            entries.add((family, int(table) & 0xFFFF))
    return tuple(sorted(entries))


def table_family(table, families=None):
    """The bank a table belongs to, or '' when it is outside the catalogue."""
    return _family_index(
        None if families is None else tuple(families)).get(
            int(table) & 0xFFFF, '')


def cell_sources(cell):
    """Directions a whole cell draws from - the union of its tables' supports.

    These are its developmental buds. A cell whose tables read N and W exposes
    two places an arm can grow backward into; a dead cell exposes none.
    """
    if cell in (PAD_CELL, OUT_CELL) or cell == EMPTY_CELL:
        return frozenset()
    support = set()
    for table in cell:
        support |= table_support(table)
    return frozenset(support)


#: Distance between two different KINDS of thing - a pad, an unbuilt output
#: niche, empty ground, a cell. Above any tolerance a run sensibly carries, so a
#: rule can never mistake the interface for a piece of body.
SENTINEL_DISTANCE = 4


def cell_distance(wanted, found, families=None):
    """Physical distance between two cells, for tolerant matching.

    Per output direction, compare only what the table READS - its support. What
    the table COMPUTES is deliberately not measured: an XOR where the rule saw
    an AND is the same part in the same place wired to the same neighbours, so a
    rule written for one applies to the other at distance 0, which is what a
    tolerance of 1 is for. Going from a one-input gate to a two-input gate is a
    different interface and costs.

    Deliberately not Hamming over the raw 16-bit values either: an AND and an
    XOR can differ in very few bits while being unrelated gates.
    """
    if wanted == found:
        return 0
    sentinels = (PAD_CELL, OUT_CELL, EMPTY_CELL)
    if wanted in sentinels or found in sentinels:
        return SENTINEL_DISTANCE
    distance = 0
    for a, b in zip(wanted, found):
        if a == b:
            continue
        distance += len(
            table_support(a).symmetric_difference(table_support(b)))
    return distance


# -- genome ---------------------------------------------------------------------

@dataclass
class LutContextGene:
    """One rule over the square lattice's four neighbours.

    ``branch_id`` scopes it to one arm's territory and ``depth`` optionally to a
    band along that arm - the two fields that stop a cell's identity from being
    a pure function of its neighbourhood.
    """

    gene_id: int
    ctx_n: object = EMPTY_CELL
    ctx_s: object = EMPTY_CELL
    ctx_e: object = EMPTY_CELL
    ctx_w: object = EMPTY_CELL
    self_in: object = EMPTY_CELL
    self_out: object = EMPTY_CELL
    branch_id: int = 0
    depth: int = DEPTH_ANY
    @property
    def context(self):
        return (self.ctx_n, self.ctx_s, self.ctx_e, self.ctx_w, self.self_in)

    def applies_at(self, band):
        return self.depth == DEPTH_ANY or int(self.depth) == int(band)

    def spawns_output(self):
        return self.self_in == OUT_CELL


@dataclass
class LutControlGene:
    """One arm's tolerance budget and lifespan.

    The default tolerance is wider than the hex port's. Four context fields must
    agree instead of three, so an exact match is markedly rarer and an arm with
    a tight budget would simply never express.
    """

    tolerance: int = 2
    telomere: int = 24


@dataclass
class LutOutputGene:
    role: str
    bearing: int = 0
    distance: int = 1
    branch_id: int = 1


@dataclass
class LutInputGene:
    bearing: int = 0
    distance: int = 1


@dataclass
class BranchedLutChromosome:
    genes: List[LutContextGene] = field(default_factory=list)
    controls: List[LutControlGene] = field(
        default_factory=lambda: [LutControlGene(), LutControlGene()])


@dataclass
class LutIoChromosome:
    """The placement chromosome, read BEFORE growth. See the hex port's twin.

    Input 0 is the origin at (0, 0); every other pad and every output root is an
    evolved bearing and distance from it, so the interface exists before any arm
    starts growing back toward it.
    """

    inputs: List[LutInputGene] = field(default_factory=list)
    outputs: List[LutOutputGene] = field(default_factory=list)
    tag: int = 0

    @property
    def genes(self):
        return list(self.inputs) + list(self.outputs)


@dataclass
class BranchedLutGenome:
    chromosomes: List[BranchedLutChromosome] = field(default_factory=list)
    #: Where the ports go, resolved before growth.
    io_chromosome: LutIoChromosome = field(default_factory=LutIoChromosome)
    families: Tuple[str, ...] = ()
    next_gene_id: int = 1

    @property
    def inputs(self):
        return self.io_chromosome.inputs

    @property
    def outputs(self):
        return self.io_chromosome.outputs

    @property
    def input_layout(self):
        """Pad cells, as a read-only view under the native encoding's name.

        The native LUT genome stores an explicit pad list; here the pads are
        DERIVED from the input genes. Exposing them under the same name lets the
        shared plumbing (growth seeds, playback, checkpoint) read a branched
        organism's inputs without a branch of its own.
        """
        from .branched_ga import input_pads
        return input_pads(self)

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
        return tuple(range(1, 2 * len(self.chromosomes) + 1))


@dataclass
class BranchedLutTrace:
    grid: Dict[Tuple[int, int], tuple]
    owners: Dict[Tuple[int, int], int]
    depths: Dict[Tuple[int, int], int]
    snapshots: List[dict] = field(default_factory=list)


# -- geometry -------------------------------------------------------------------

def square_ring(distance):
    """Cells exactly ``distance`` steps from the origin, in angular order.

    The square lattice's polar coordinate, so a bearing mutation slides a site
    around the anchor and a distance mutation moves it in or out - the same
    allele behaviour the hex port needs rings for.
    """
    distance = max(0, int(distance))
    if distance == 0:
        return ((0, 0),)
    ring = []
    for step in range(distance):
        ring.append((distance - step, step))
    for step in range(distance):
        ring.append((-step, distance - step))
    for step in range(distance):
        ring.append((-distance + step, -step))
    for step in range(distance):
        ring.append((step, -distance + step))
    return tuple(ring)


def bearing_cell(bearing, distance, taken=()):
    for radius in (int(distance),) + tuple(
            r for delta in range(1, 8)
            for r in (int(distance) + delta, int(distance) - delta)):
        if radius < 1:
            continue
        ring = square_ring(radius)
        start = int(bearing) % len(ring)
        for step in range(len(ring)):
            cell = ring[(start + step) % len(ring)]
            if cell not in taken:
                return cell
    return None


def output_root_sites(genome, pads=()):
    taken = set(tuple(pad) for pad in pads)
    roots = {}
    for gene in genome.outputs:
        cell = bearing_cell(gene.bearing, gene.distance, taken)
        if cell is None:
            continue
        roots[int(gene.branch_id)] = cell
        taken.add(cell)
    return roots


# -- development ----------------------------------------------------------------

def _band_of(cell, around, depths):
    if cell in depths:
        return depths[cell]
    parents = [depths[around[d]] for d in DIRECTIONS if around[d] in depths]
    return (min(parents) + 1) if parents else 0


def _cell_of(cell, grid, pads, output_sites):
    if cell in pads:
        return PAD_CELL
    if cell in grid:
        return grid[cell]
    if cell in output_sites:
        return OUT_CELL
    return EMPTY_CELL


def cell_outputs(cell):
    """Directions a cell DRIVES - the reverse of ``cell_sources``.

    A cell holds one table per output direction, so it drives exactly the
    directions whose table is not constant zero. Reverse-growth polarity is
    stated in terms of this.
    """
    if cell in (PAD_CELL, OUT_CELL) or cell == EMPTY_CELL:
        return ()
    return tuple(direction for direction, table in zip(DIRECTIONS, cell)
                 if table)


def _upstream_parents(cell, label, owners, depths, before_depth=None):
    """Same-arm neighbours nearer this arm's root than ``cell``."""
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
    """How deep ``cell`` sits in ``label``'s branch, or None if outside it.

    An arm may act on ground it holds, or on ground touching that ground. FNV's
    ``_reach``: it is what makes an arm grow as a limb rather than a scatter.
    """
    if owners.get(cell) == label:
        return depths.get(cell, 0)
    parents = _upstream_parents(cell, label, owners, depths)
    return min((depths.get(n, 0) + 1 for _d, n in parents), default=None)


def required_output_directions(cell, label, depth, owners, depths):
    """Source-local ports that would connect this bud toward its arm root."""
    return tuple(sorted({
        direction for direction, _n in _upstream_parents(
            cell, label, owners, depths, before_depth=int(depth))}))


def drives_toward_root(cell, state, label, depth, owners, depths):
    """Whether placing ``state`` here feeds the limb it is joining.

    The part must drive at least one direction reaching a SHALLOWER same-arm
    cell, or the output cannot hear it.

    FEEDBACK IS DELIBERATELY ALLOWED, unlike FNV's ``_drives_toward_root``,
    which also forbids driving a same-arm cell at equal or greater depth so each
    arm stays a strict tree. FNV can afford that because its memory lives inside
    named components; the LUT array's memory is topological - a cross-coupled
    hold - so forbidding cycles forbids state outright. Measured under the tree
    rule: SR latch plateaued at 0.391 and a gated D latch never reached even the
    score of a memoryless pass-through gate.
    """
    if int(depth) <= 0:
        return True
    required = set(
        required_output_directions(cell, label, depth, owners, depths))
    outputs = cell_outputs(state)
    return any(direction in outputs for direction in required)


def growth_candidates(grid, owners, pads, output_sites):
    """Every site development may write this iteration, for any arm.

    Unbuilt output niches (a root can be far from the body, so it must be an
    explicit candidate), every built cell, and the buds at each built cell's
    real input directions. ``arm_reach`` decides which arm may write which.

    Sorted, not a set: deterministic contest, hash-order-independent.
    """
    sites = set(output_sites) - set(grid)
    sites.update(cell for cell in grid if cell not in pads)
    for cell, state in grid.items():
        if cell in pads:
            continue
        around = neighbours(cell)
        for direction in cell_sources(state):
            source = around[direction]
            if source not in pads:
                sites.add(source)
    return sorted(sites)


def develop_branched_lut(genome, seeds, *, snapshots=False):
    """Grow every living arm at once, synchronously.

    Same rules as the hex port and as FNV: arms start at their genetic output
    roots and grow backward along each cell's real input directions; a rule may
    only act on its OWN LIMB; a contested cell goes to the
    nearest match then the lowest arm; a rule differentiates one cohort and is
    then spent; an arm's telomere is spent one per changed cell.
    """
    pads = {tuple(seed) for seed in seeds}
    roots = output_root_sites(genome, pads)
    output_sites = set(roots.values()) - pads
    families = genome.families or None
    grid: Dict[Tuple[int, int], tuple] = {}
    owners: Dict[Tuple[int, int], int] = {}
    depths: Dict[Tuple[int, int], int] = {}
    frames = [dict(grid)] if snapshots else []

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
        for cell in growth_candidates(grid, owners, pads, output_sites):
            around = neighbours(cell)
            context = tuple(
                _cell_of(around[d], grid, pads, output_sites)
                for d in DIRECTIONS) + (
                    _cell_of(cell, grid, pads, output_sites),)
            reach = {}
            for arm in living:
                label = arm['label']
                for gene in arm['genes']:
                    if gene.gene_id in arm['spent']:
                        continue
                    if gene.spawns_output():
                        # OUT is not a wildcard: only this arm, at its own
                        # placement site. The root rule matches at distance 0
                        # WITHOUT comparing its neighbourhood - otherwise moving
                        # a root would require rewriting the rule in the same
                        # mutation, so any placement mutation kills the arm.
                        if (roots.get(label) != cell
                                or context[4] != OUT_CELL):
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
                            cell_distance(wanted, found, families)
                            for wanted, found in zip(gene.context, context))
                        if distance > arm['reach']:
                            continue
                    if not gene.applies_at(min(depth, DEPTH_BANDS - 1)):
                        continue
                    if (gene.self_out != EMPTY_CELL
                            and not drives_toward_root(
                                cell, gene.self_out, label, depth,
                                owners, depths)):
                        continue
                    key = (distance, label, gene.gene_id)
                    if cell not in proposals or key < proposals[cell][0]:
                        proposals[cell] = (key, arm, gene, depth)

        if not proposals:
            break

        changed, fired = False, set()
        for cell, ((_d, label, _gid), arm, gene, depth) in sorted(
                proposals.items()):
            if arm['life'] <= 0 or gene.gene_id in arm['spent']:
                continue
            new_cell = gene.self_out
            # Compare against physical storage: erasing an empty niche is a
            # no-op and must not burn the arm's lifespan.
            if tuple(new_cell) == tuple(grid.get(cell, EMPTY_CELL)):
                continue
            if new_cell == EMPTY_CELL:
                del grid[cell]
                owners.pop(cell, None)
                depths.pop(cell, None)
            else:
                if cell not in grid and len(grid) >= MAX_PLACEMENTS:
                    continue
                grid[cell] = tuple(new_cell)
                owners[cell] = label
                depths[cell] = int(depth)
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

    return BranchedLutTrace(grid=grid, owners=owners, depths=depths,
                            snapshots=frames)


def firing_outputs(grid, pads):
    """(cell, direction) output wires the inputs can ever drive. Target-blind.

    A least-fixed-point over the lattice: every pad drives all four of its
    wires; any other wire can fire once ANY input its table depends on can fire.
    Deliberately optimistic - it does not ask whether a table needs both inputs
    high at once - because an optimistic reachability test can only over-count,
    and over-counting never hides a circuit that really works.

    The wire feeding cell C from direction d is neighbour N's output pointing
    back at C, which is N's table for the OPPOSITE direction. Getting that
    backwards would silently report every organism as connected.
    """
    live = {(cell, direction) for cell in pads for direction in DIRECTIONS}
    changed = True
    while changed:
        changed = False
        for cell, state in grid.items():
            if cell in pads:
                continue
            around = neighbours(cell)
            fed = {
                direction for direction in DIRECTIONS
                if (around[direction], OPPOSITE[direction]) in live}
            if not fed:
                continue
            for index, direction in enumerate(DIRECTIONS):
                wire = (cell, direction)
                if wire in live or not state[index]:
                    continue
                if _can_go_high(state[index], fed):
                    live.add(wire)
                    changed = True
    return live


def _can_go_high(table, fed):
    """Can this table ever output 1, given only ``fed`` inputs can go high?

    Asking "does it read a live input" is not the same question: a two-input AND
    reading one live and one dead neighbour reads a live input and can still
    never go high. The banks are quiescent at all-zero, so this walks the rows
    the live inputs can actually produce.
    """
    live_mask = 0
    for name, bit in INPUT_BITS.items():
        if name in fed:
            live_mask |= 1 << bit
    table = int(table) & 0xFFFF
    for index in range(16):
        if index & ~live_mask:
            continue                   # needs a neighbour that can never rise
        if (table >> index) & 1:
            return True
    return False


def driven_roots(grid, pads, roots):
    """Which output roots the inputs can actually drive."""
    live = firing_outputs(grid, pads)
    return {label for label, cell in roots.items()
            if cell in grid and any((cell, d) in live for d in DIRECTIONS)}


def materialise_pads(grid, pads):
    """Add the input pads to a grown body as live source cells.

    Development treats a pad as read-only CONTEXT with its own identity
    (PAD_CELL), so no rule can overwrite the organism's own input. The simulator
    has no such concept - an input is a cell that exists and conducts - so the
    pads become ordinary SEED_STATE cells here, at the boundary between the
    encoding and the physics.
    """
    from .lut import SEED_STATE
    body = dict(grid)
    for pad in pads:
        body.setdefault(tuple(pad), SEED_STATE)
    return body


def grow_branched_lut(genome, seeds, grid_size=None, iters=None):
    """Grown body as ``cell -> (Ln, Ls, Le, Lw)``, the shape LutSim expects."""
    pads = tuple(tuple(seed) for seed in seeds)
    return materialise_pads(develop_branched_lut(genome, pads).grid, pads)
