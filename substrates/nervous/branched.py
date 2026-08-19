"""
substrates/nervous/branched.py - FNV's branched, output-rooted encoding on the
hex nervous net.

WHY
---
The native hex encoding matches a cell against its genes by MINIMUM HAMMING
DISTANCE, so every cell always matches something and a cell's configuration is a
pure function of its neighbourhood: two cells in identical surroundings are
forced to become identical cells. That is the ceiling recorded in the
`fnv-vs-context-addressed-substrates` note - the genome cannot name a part, only
describe a situation.

FNV escapes it with four mechanisms, and this module ports exactly those onto
the hex grid, changing nothing about the substrate's physics:

  1. BRANCH SCOPING   a rule may only act on its own arm's limb and its buds
  2. DEPTH BANDS      a rule may apply only that far along its own branch
  3. EXACT MATCHING   within an evolvable per-arm tolerance, instead of nearest
  4. SPENT-ONCE       a rule differentiates one synchronous cohort, then retires

Together these mean a cell's identity depends on (neighbourhood, arm, depth,
what has already fired) rather than on neighbourhood alone, which is what lets
one genome build a base and a tip differently.

WHAT IS THE SAME AS FNV
-----------------------
The hex grid is already a degree-3 honeycomb with FNV's exact context shape
(ctx_l / ctx_r / ctx_d / self_in), so the topology needs no change at all.
Growth is reverse: arms start at genetic OUTPUT roots and grow back toward the
input PADS, budding at each cell's physical input directions. On FNV those come
from the component catalogue; here they come from ROUTING_HEX, where a state
names precisely which neighbours it listens to. A node that reads two sources
therefore exposes two developmental buds, exactly as a binary gate does on FNV.

WHAT IS NECESSARILY DIFFERENT
-----------------------------
* CELL ALPHABET. Cells are paper tri-tiles - three ROUTING_HEX circuits, one
  per output direction, packed into a 15-bit state (substrates/nervous/tritile)
  - not catalogue components. That is the one live nervous profile
  (tile_arch='tri3' on the paper_analog node), so it is the only alphabet this
  encoding speaks. Tolerance therefore compares routing INTERFACES per channel
  - which directions each circuit draws from - rather than numeric state
  distance, which would make two tiles that differ only in a low bit look
  adjacent when their L channels are unrelated circuits.
* TELOMERE MEANING. The native hex telomere is a Hayflick division limit that
  provably halts growth at radius L. Here it is FNV's ARM LIFESPAN, spent one
  per changed cell. Size is bounded by MAX_PLACEMENTS and by each arm's
  lifespan instead of by radius. This is a real behavioural change, not a
  reinterpretation, and it is why this encoding is opt-in.

INTERFACE FIRST
---------------
One arm per output role (two arms per chromosome; spare arms stay empty), each
with its own tolerance and its own lifespan. A separate PLACEMENT chromosome
fixes where every port sits before any growth happens - input 0 at the origin,
every other pad and every output root an evolved bearing and distance from it -
because an arm cannot grow backward toward an interface that does not exist yet.

STATUS
------
This is THE nervous encoding: the desktop controller and the headless driver
both breed it, and it saves and loads as a `branched_v1` checkpoint document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

from ..fnv.genome import input_ring   # same honeycomb, same polar geometry
from .hexgrid import ROUTING_HEX, hex_dirs
from .tritile import (TRI_DIRS, TRI_SEED_STATE, channel_configs,
                      pack_channels)

#: A cell that is not there. Same value as the native encoding's dead state, so
#: a genome that never uses branch scoping still reads naturally.
EMPTY_STATE = 0
#: Input pad. Read-only: pads are the interface, never written by development.
PAD_STATE = -1
#: An unbuilt genetic output niche - writable exactly once, by the arm that owns
#: it, which is how a branch starts.
OUT_STATE = -2

#: Depth bands a rule may specialise to, and the value meaning "any depth".
#: Four bands collapsed every site beyond depth three into one address.  The
#: real Full Adder reaches depth ten, where repeated relay contexts are normal;
#: those rules could not distinguish their intended cells and either overgrew
#: or became unreachable. Sixteen covers the placement ceiling's useful limb
#: depths while remaining a small mutation alphabet.
DEPTH_BANDS = 16
DEPTH_ANY = -1

#: Hard ceiling on body size, mirroring FNV. Arms also stop when their own
#: lifespan runs out; this only stops a pathological genome.
MAX_PLACEMENTS = 128

DIRECTIONS = ('L', 'R', 'D')

#: Per-node leak time constants, as multiples of the run's base tau_leak.
#:
#: Deliberately IRRATIONAL-ish ratios rather than a neat 0.5/1.0/2.0 ladder. A
#: node's propagation delay scales with its tau (analog._node_delay), so a
#: ladder of simple multiples would keep every event on a shared grid and the
#: net would stay effectively synchronous - which is exactly the defect this
#: allele exists to remove. Measured before it: 0 of 257 leading edges fell off
#: an integer tick.
#:
#: Index 0 is the base value, so a genome that never mutates this behaves
#: exactly as the single-constant engine did.
TAU_SCALES = (1.0, 0.63, 0.79, 1.27, 1.61, 2.13, 0.41, 3.07)


def tau_of(index, base):
    """The absolute leak time constant an allele names."""
    return float(base) * TAU_SCALES[int(index) % len(TAU_SCALES)]


@lru_cache(maxsize=None)
def _channel_interface(config):
    """(frozenset of source directions, operation) for ONE 5-bit channel."""
    excite_a, excite_b, inhibit, op = ROUTING_HEX[int(config) & 0x1F]
    sources = frozenset(
        d for d in (excite_a, excite_b, inhibit) if d is not None)
    return (sources, 'off' if not sources else op)


@lru_cache(maxsize=None)
def routing_sources(state):
    """Directions a TILE actually draws from - its growth buds.

    This is the hex analogue of an FNV component's ``inputs``. A tri tile holds
    three circuits, one per output direction, and each selects its excitatory /
    inhibitory inputs among the tile's three incoming directions. The tile's
    buds are therefore the UNION of what its three channels listen to: a tile
    whose channels all read 'D' exposes one bud, a tile that coincidence-detects
    L and R exposes two. Growing backward from those buds is what makes an arm a
    limb rather than a scatter.
    """
    if state in (EMPTY_STATE, PAD_STATE, OUT_STATE):
        return ()
    seen = []
    for config in channel_configs(state):
        sources, _op = _channel_interface(config)
        for direction in TRI_DIRS:
            if direction in sources and direction not in seen:
                seen.append(direction)
    return tuple(seen)


@lru_cache(maxsize=None)
def routing_interface(state):
    """Per-channel ((sources, op), (sources, op), (sources, op)) - what a tile IS.

    The physical identity of a tri tile is its three circuits, not the integer
    they happen to pack into, so the interface is reported channel by channel.
    """
    if state in (EMPTY_STATE, PAD_STATE, OUT_STATE):
        marker = (frozenset(), str(state))
        return (marker, marker, marker)
    return tuple(_channel_interface(config)
                 for config in channel_configs(state))


#: Distance between two different KINDS of thing - a pad, an unbuilt output
#: niche, empty ground, a cell. These are not near-misses of each other at any
#: tolerance a run would sensibly carry, so they sit above the usable range
#: rather than being scored on the interface scale below.
SENTINEL_DISTANCE = 4


@lru_cache(maxsize=None)
def state_distance(wanted, found):
    """Physical distance between two tile states, for tolerant matching.

    WHAT COUNTS AS A DIFFERENCE is the point of this function. Distance measures
    the INTERFACE only - which neighbours each circuit draws from - and not what
    the circuit computes with them. Swapping a coincidence AND for its OR twin
    is distance 0: the part sits in the same place in the same limb, wired to
    the same neighbours, so a rule written for one is a rule that applies to the
    other, and a tolerance of 1 is meant to cover exactly that. Going from one
    input to two is a different interface, and costs.

    Deliberately NOT |wanted - found|. A tile state packs three 5-bit routing
    configs, so neighbouring integers are usually unrelated circuits - flipping
    bit 5 rebuilds the R channel while leaving L alone.
    """
    if wanted == found:
        return 0
    sentinels = (EMPTY_STATE, PAD_STATE, OUT_STATE)
    if wanted in sentinels or found in sentinels:
        return SENTINEL_DISTANCE
    total = 0
    for (wanted_sources, _wanted_op), (found_sources, _found_op) in zip(
            routing_interface(wanted), routing_interface(found)):
        total += len(wanted_sources.symmetric_difference(found_sources))
    return total


# -- genome ---------------------------------------------------------------------

@dataclass
class HexContextGene:
    """One rule: a neighbourhood a cell must be in, and what it becomes.

    Identical in shape to the native ``HexGene`` context, plus the two fields
    that do the individuating work: ``branch_id`` scopes the rule to one arm's
    territory, and ``depth`` optionally scopes it to a band along that arm.
    """

    gene_id: int
    ctx_l: int = EMPTY_STATE
    ctx_r: int = EMPTY_STATE
    ctx_d: int = EMPTY_STATE
    self_in: int = EMPTY_STATE
    self_out: int = EMPTY_STATE
    branch_id: int = 0
    depth: int = DEPTH_ANY
    #: Index into TAU_SCALES: the leak time constant of the cell this rule
    #: installs, as a multiple of the run's base tau. One capacitor per node, so
    #: one allele per node - and it is NOT a node type: the routing lives in
    #: ``self_out`` and mutates separately, so timing and function evolve
    #: independently. Default 0 is the base tau, which reproduces the old
    #: single-constant behaviour exactly.
    tau_index: int = 0

    @property
    def context(self):
        return (self.ctx_l, self.ctx_r, self.ctx_d, self.self_in)

    def applies_at(self, band):
        return self.depth == DEPTH_ANY or int(self.depth) == int(band)

    def spawns_output(self):
        return int(self.self_in) == OUT_STATE


@dataclass
class HexControlGene:
    """One arm's two evolvable constants.

    ``tolerance`` is the context distance the arm's rules may match within;
    ``telomere`` is the arm's LIFESPAN in changed cells - births, retypes and
    erasures alike - not a radius.
    """

    tolerance: int = 0
    telomere: int = 24


@dataclass
class HexOutputGene:
    """Where one output role's root sits, as a bearing and distance from origin.

    The nervous net's native path fits output probes to the whole organism after
    growth. That cannot root an arm, because the site does not exist until
    growth has already happened - so this encoding gives outputs genetic sites,
    the way FNV does, and grows backward from them.
    """

    role: str
    bearing: int = 0
    distance: int = 1
    branch_id: int = 1


@dataclass
class HexInputGene:
    """Where one input pad sits relative to pad zero, which holds the origin."""

    bearing: int = 0
    distance: int = 1


@dataclass
class BranchedHexChromosome:
    """Two arms. ``genes`` carries both; ``controls`` is one entry per arm."""

    genes: List[HexContextGene] = field(default_factory=list)
    controls: List[HexControlGene] = field(
        default_factory=lambda: [HexControlGene(), HexControlGene()])


@dataclass
class IoChromosome:
    """The placement chromosome, read BEFORE any growth happens.

    It fixes the organism's interface first - where every input pad and every
    output root sits - and only then do the arms grow backward from those roots
    toward those pads. That ordering is the whole point of an output-rooted
    encoding: an arm cannot grow toward an interface that does not exist yet,
    which is exactly why the native path (probes fitted to the finished body)
    could not root a limb.

    Input 0 is not stored: it IS the origin, (0, 0), and every other pad and
    root is an evolved bearing and distance from it. Anchoring one port makes
    the rest of the interface a set of RELATIVE alleles, so a mutation slides
    one port without translating the whole organism.
    """

    inputs: List[HexInputGene] = field(default_factory=list)
    outputs: List[HexOutputGene] = field(default_factory=list)
    tag: int = 0

    @property
    def genes(self):
        """Both placement gene kinds in one list, for uniform gene handling."""
        return list(self.inputs) + list(self.outputs)


@dataclass
class BranchedHexGenome:
    chromosomes: List[BranchedHexChromosome] = field(default_factory=list)
    #: Where the ports go, resolved before growth. See IoChromosome.
    io_chromosome: IoChromosome = field(default_factory=IoChromosome)
    next_gene_id: int = 1
    #: The one live nervous profile. Carried as a field, not assumed, because
    #: every consumer downstream (interpretation, simulation, viz) reads
    #: ``genome.arch`` to decide whether a cell state is one circuit or three.
    arch: str = 'tri3'

    @property
    def input_layout(self):
        """Pad cells, so the shared nervous plumbing can read them as a layout.

        The native encoding stores an explicit list of pad coordinates; here the
        pads are DERIVED from the input genes, so this is a read-only view of
        the same thing. Exposing it under the native name is what lets
        growth_seeds / layout_pads / the playback code bind a branched organism
        without a branch of their own.
        """
        from .branched_ga import input_pads
        return input_pads(self)

    @property
    def inputs(self):
        """Input placement genes (they live on the I/O chromosome)."""
        return self.io_chromosome.inputs

    @property
    def outputs(self):
        """Output placement genes (they live on the I/O chromosome)."""
        return self.io_chromosome.outputs

    def arm(self, label):
        """(genes, control) for a 1-based branch label, or (None, None)."""
        index, half = divmod(int(label) - 1, 2)
        if not 0 <= index < len(self.chromosomes):
            return None, None
        chromosome = self.chromosomes[index]
        members = [gene for gene in chromosome.genes
                   if int(gene.branch_id) == int(label)]
        control = (chromosome.controls[half]
                   if half < len(chromosome.controls) else HexControlGene())
        return members, control

    @property
    def arm_labels(self):
        return tuple(range(1, 2 * len(self.chromosomes) + 1))


@dataclass
class BranchedTrace:
    grid: Dict[Tuple[int, int], int]
    owners: Dict[Tuple[int, int], int]
    depths: Dict[Tuple[int, int], int]
    snapshots: List[Dict[Tuple[int, int], int]] = field(default_factory=list)
    #: cell -> TAU_SCALES index, inherited from the gene that installed it.
    taus: Dict[Tuple[int, int], int] = field(default_factory=dict)


# -- development ----------------------------------------------------------------

def _band_of(cell, around, depths):
    """How far along its branch a cell sits, for depth-banded rules.

    An UNBUILT bud has no depth of its own yet, so it takes its parent's plus
    one. Defaulting it to 0 instead made every bud look like a root, and no
    depth-banded rule could ever fire - the band that lets one genome build a
    base and a tip differently would have been silently dead.
    """
    if cell in depths:
        return depths[cell]
    parents = [depths[around[direction]] for direction in DIRECTIONS
               if around[direction] in depths]
    return (min(parents) + 1) if parents else 0


def _direction_toward(cell, neighbour):
    """The direction, in ``cell``'s own frame, whose output wire reaches
    ``neighbour`` - or None if they are not neighbours."""
    around = hex_dirs(*cell)
    for direction in DIRECTIONS:
        if around[direction] == neighbour:
            return direction
    return None


def orient_toward(state, directions):
    """Make a tile drive the given directions, keeping the circuit it holds.

    A tri tile is three circuits, one per output wire, and a rule names the
    tile without knowing where development will place it. Left alone, an arm
    would grow a cell into the site its parent READS FROM and the new cell would
    usually drive somewhere else entirely - a limb of unconnected parts, which
    is exactly what the first version of this encoding produced (every organism
    scored the silent baseline).

    So placement ORIENTS the part: any wire that a consumer is listening on, and
    that the tile leaves dead, is given a copy of a live channel. The circuit -
    which neighbours it draws from and whether it ANDs or ORs them - is
    unchanged; only the set of directions it speaks on grows. This is the hex
    analogue of an FNV component being placed with its output facing the parent
    that budded it.
    """
    if state in (EMPTY_STATE, PAD_STATE, OUT_STATE) or not directions:
        return state
    channels = dict(zip(TRI_DIRS, channel_configs(state)))
    live = next((channels[d] for d in TRI_DIRS if channels[d]), 0)
    if not live:
        return state
    for direction in directions:
        if direction in channels and not channels[direction]:
            channels[direction] = live
    return pack_channels(*(channels[d] for d in TRI_DIRS))


def _state_of(cell, grid, pads, output_sites):
    if cell in pads:
        return PAD_STATE
    if cell in grid:
        return grid[cell]
    if cell in output_sites:
        return OUT_STATE
    return EMPTY_STATE


def bearing_cell(bearing, distance, taken=()):
    """The cell a (bearing, distance) allele names, avoiding occupied sites.

    Walking one L/R/D direction repeatedly does NOT travel in a straight line
    here: hex_dirs reports neighbours in the node's own rotated frame, so two
    'L' steps come straight back to where they started. Rings indexed by
    angular bearing are the honeycomb's actual polar coordinate, and they give
    alleles the property selection needs - a bearing mutation slides the site
    around the anchor, a distance mutation moves it in or out.
    """
    for radius in (int(distance),) + tuple(
            r for delta in range(1, 8)
            for r in (int(distance) + delta, int(distance) - delta)):
        if radius < 1:
            continue
        ring = input_ring(radius)
        if not ring:
            continue
        start = int(bearing) % len(ring)
        for step in range(len(ring)):
            cell = ring[(start + step) % len(ring)]
            if cell not in taken:
                return cell
    return None


def output_root_sites(genome, pads=()):
    """{branch label: cell} for each genetic output root.

    Roots are placed in chromosome order and never land on a pad or on an
    earlier root, so two arms cannot start life fighting over one cell.
    """
    taken = set(tuple(pad) for pad in pads)
    roots = {}
    for gene in genome.outputs:
        cell = bearing_cell(gene.bearing, gene.distance, taken)
        if cell is None:
            continue
        roots[int(gene.branch_id)] = cell
        taken.add(cell)
    return roots


def tile_outputs(state):
    """Directions a tile DRIVES - the reverse of ``routing_sources``.

    A tri tile holds one circuit per output direction, so it drives exactly the
    directions whose channel is not the dead config. This is the hex analogue of
    an FNV component's ``outputs``, and reverse-growth polarity is stated in
    terms of it.
    """
    if state in (EMPTY_STATE, PAD_STATE, OUT_STATE):
        return ()
    return tuple(direction for direction, config
                 in zip(TRI_DIRS, channel_configs(state)) if config)


def _upstream_parents(cell, label, owners, depths, before_depth=None):
    """Same-arm neighbours nearer this arm's root than ``cell``.

    "Nearer the root" is what makes the arm a directed tree: a bud's parent is
    the cell it will feed, never a sibling beside it.
    """
    around = hex_dirs(*cell)
    parents = []
    for direction in DIRECTIONS:
        neighbour = around[direction]
        if owners.get(neighbour) != label:
            continue
        if before_depth is not None and depths.get(
                neighbour, before_depth) >= before_depth:
            continue
        parents.append(neighbour)
    return parents


def arm_reach(cell, label, owners, depths):
    """How deep ``cell`` sits in ``label``'s branch, or None if outside it.

    An arm may act on ground it already holds, or on ground TOUCHING that
    ground. This is FNV's ``_reach``, and it is what makes an arm grow as a limb
    rather than as a scatter: without it a rule fires wherever its neighbourhood
    happens to occur, anywhere in the organism.
    """
    if owners.get(cell) == label:
        return depths.get(cell, 0)
    parents = _upstream_parents(cell, label, owners, depths)
    return min((depths.get(parent, 0) + 1 for parent in parents), default=None)


def required_output_directions(cell, label, depth, owners, depths):
    """Source-local ports that connect this bud to its existing arm.

    A part drawn for this site must drive at least one already-built same-arm
    neighbour, or the limb it joins cannot hear it.  Requiring that neighbour
    to have a strictly smaller developmental depth quietly reinstated FNV's
    tree restriction: a useful lateral return in a feedback loop was rejected
    even though its consumer was already present.  Nervous-net state is
    topological, so equal/deeper return edges are part of the expressible
    substrate rather than decorations.
    """
    return tuple(sorted({
        _direction_toward(cell, parent)
        for parent in _upstream_parents(cell, label, owners, depths)}))


def drives_toward_root(cell, state, label, depth, grid, owners, depths):
    """Whether placing ``state`` here keeps the arm a directed tree to its root.

    Reverse growth only means something if each new part actually FEEDS the part
    it grew from, so the part must drive at least one direction reaching a
    SHALLOWER same-arm cell. A part the output cannot hear is a decoration on a
    silent limb.

    FEEDBACK IS DELIBERATELY ALLOWED, and this is where the rule departs from
    FNV's ``_drives_toward_root``. FNV additionally forbids driving a same-arm
    cell at equal or greater depth, keeping each arm a strict directed tree,
    because on that substrate a returning edge is an accidental combinational
    cycle - its memory lives INSIDE named components (HOLD, TOGGLE, C_ELEMENT),
    so it never needs one.

    Nervous and LUT have no such components. Their memory is TOPOLOGICAL: a
    pulse circulating in a loop here, a cross-coupled hold on the LUT array. So
    the tree rule does not merely constrain those substrates, it makes every
    latch, toggle and counter unreachable - measured, an SR latch on the LUT
    array plateaued at 0.391 and a gated D latch never even reached the score of
    a memoryless pass-through gate. Forbidding cycles forbids state.

    What survives from FNV is the part that earned its place: every cell must
    still feed an already-built member of the limb it joined. This is polarity,
    not routing - the local rule still chooses the circuit, and nothing here
    steers growth toward any particular pad.
    """
    if int(depth) <= 0:
        return True                      # the root itself answers to nobody
    required = set(
        required_output_directions(cell, label, depth, owners, depths))
    outputs = tile_outputs(state)
    return any(direction in outputs for direction in required)


def growth_candidates(grid, owners, pads, output_sites):
    """Every site development may write this iteration, for any arm.

    The unbuilt output niches (a root can be genetically distant from the body,
    so it must be an explicit candidate rather than waiting for a frontier that
    never arrives), every built cell, and the BUDS hanging off each built cell's
    physical input directions. A tile that reads two neighbours therefore
    exposes two buds and a buffer exposes one, so irrelevant side-neighbours do
    not turn into decoration.

    Which arm may actually write which of these is decided by ``arm_reach``.

    Sorted, not a set: this feeds a deterministic contest, and iterating a set
    of coordinate tuples is hash-order dependent across processes.
    """
    sites = set(output_sites) - set(grid)
    sites.update(cell for cell in grid if cell not in pads)
    for cell, state in grid.items():
        if cell in pads:
            continue
        around = hex_dirs(*cell)
        for direction in routing_sources(state):
            source = around[direction]
            if source not in pads:
                sites.add(source)
    return sorted(sites)


def develop_branched_hex(genome, seeds, *, snapshots=False):
    """Grow every living arm at once, one synchronous iteration at a time.

    Mirrors ``substrates/fnv/construction._develop_branched``. Each iteration
    every gene of every living arm is matched against the grid AS IT STOOD AT
    THE START of the iteration, and fires at every cell whose neighbourhood it
    matches within its arm's tolerance.

    Scoping beyond the neighbourhood:

    * an arm may only act on ITS OWN LIMB: the cells it has built, and the buds
      hanging off those cells' physical input directions. Its one
      ``self_in == OUT_STATE`` rule at its assigned root is what starts it. So
      every node an arm places is a node the arm's output can already reach
      backward through - the circuit grows from the output toward the inputs,
      and an arm can never scatter cells into ground that merely happens to be
      adjacent to it;
    * a rule may carry a depth band, applying only that far along its own branch;
    * a contested cell goes to the nearest match, then to the lowest arm label,
      then the lowest gene id - never left empty, which walls branches apart;
    * a gene differentiates ONE synchronous cohort and is then spent, which
      bounds ontogenic amplification to a spatial branch event rather than
      letting one rule extrude an unbounded chain.

    Pads are read-only. An arm's telomere is spent one per changed cell.
    """
    pads = {tuple(seed) for seed in seeds}
    roots = output_root_sites(genome, pads)
    output_sites = set(roots.values()) - pads
    grid: Dict[Tuple[int, int], int] = {}
    owners: Dict[Tuple[int, int], int] = {}
    depths: Dict[Tuple[int, int], int] = {}
    taus: Dict[Tuple[int, int], int] = {}
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
        if not living:
            break

        room = MAX_PLACEMENTS - len(grid)
        if room <= 0:
            break

        proposals = {}
        for cell in growth_candidates(grid, owners, pads, output_sites):
            around = hex_dirs(*cell)
            context = (
                _state_of(around['L'], grid, pads, output_sites),
                _state_of(around['R'], grid, pads, output_sites),
                _state_of(around['D'], grid, pads, output_sites),
                _state_of(cell, grid, pads, output_sites),
            )
            reach = {}
            for arm in living:
                label = arm['label']
                for gene in arm['genes']:
                    if gene.gene_id in arm['spent']:
                        continue
                    if gene.spawns_output():
                        # OUT is not a wildcard: only this arm, at the site its
                        # own placement gene owns, may claim initial territory.
                        # The root rule matches at distance 0 WITHOUT comparing
                        # its neighbourhood - OUT already names both the role
                        # and its one writable site. Making it memorise whatever
                        # happened to surround that coordinate would couple
                        # moving a root to rewriting the rule, so any mutation
                        # of the placement chromosome would kill the whole arm.
                        if (roots.get(label) != cell
                                or context[3] != OUT_STATE):
                            continue
                        depth, distance = 0, 0
                    else:
                        if label not in reach:
                            reach[label] = arm_reach(
                                cell, label, owners, depths)
                        depth = reach[label]
                        if depth is None:
                            continue          # outside this arm's territory
                        distance = sum(
                            state_distance(wanted, found)
                            for wanted, found in zip(gene.context, context))
                        if distance > arm['reach']:
                            continue
                    if not gene.applies_at(min(depth, DEPTH_BANDS - 1)):
                        continue
                    # Reverse-growth polarity: a new part must feed the part it
                    # grew from, or the limb it joins cannot hear it.
                    if (int(gene.self_out) != EMPTY_STATE
                            and not drives_toward_root(
                                cell, int(gene.self_out), label, depth,
                                grid, owners, depths)):
                        continue
                    key = (distance, label, gene.gene_id)
                    if cell not in proposals or key < proposals[cell][0]:
                        proposals[cell] = (key, arm, gene, depth)

        if not proposals:
            break

        changed = False
        fired = set()
        for cell, ((_distance, label, gene_id), arm, gene, depth) in sorted(
                proposals.items()):
            if arm['life'] <= 0 or gene.gene_id in arm['spent']:
                continue
            new_state = int(gene.self_out)
            was = grid.get(cell)
            # Compare against physical storage, not the OUT sentinel: erasing an
            # empty niche is a no-op and must not burn the arm's lifespan.
            if new_state == int(grid.get(cell, EMPTY_STATE)):
                continue
            if new_state == EMPTY_STATE:
                del grid[cell]
                owners.pop(cell, None)
                depths.pop(cell, None)
                taus.pop(cell, None)
            else:
                if was is None and len(grid) >= MAX_PLACEMENTS:
                    continue
                # The gene's part is installed AS WRITTEN. Rotating it to face
                # whatever happened to be listening would mean the genome no
                # longer names the circuit it builds - and polarity is already
                # guaranteed, because drives_toward_root rejected any part that
                # does not feed the limb it is joining.
                grid[cell] = new_state
                owners[cell] = label
                depths[cell] = int(depth)
                # The cell inherits its capacitor from the rule that built it.
                taus[cell] = int(getattr(gene, 'tau_index', 0))
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

    return BranchedTrace(grid=grid, owners=owners, depths=depths,
                         snapshots=frames, taus=taus)


def firing_nodes(grid, pads, sink_nodes=None):
    """Sub-nodes that can ever fire, from the pads alone. Target-blind.

    A least-fixed-point over the tri sub-node graph: a pad's IN node fires; any
    other channel fires once its excitatory requirement is met - EITHER source
    for an OR twin, BOTH for a coincidence AND. Inhibitory inputs are ignored,
    because ignoring them can only ever make this over-optimistic, and an
    over-optimistic reachability test never hides a circuit that does work.

    This is the structural question "is this limb connected to an input at all",
    asked without running the physics or looking at the target. It exists
    because more than half of freshly built organisms turn out to have an output
    that nothing can ever drive, and those are all indistinguishable to
    selection.
    """
    from .tritile import interpret_tri
    info = interpret_tri(grid, pads)
    nodes, sources = info['nodes'], info['sources']
    sink_subnodes = {
        node for tile in (sink_nodes or ())
        for node in info['tile_nodes'].get(tuple(tile), ())}
    firing = set(info['in_nodes'].values())
    changed = True
    while changed:
        changed = False
        for node, (excite_a, excite_b, _inhibit, op) in nodes.items():
            if node in firing:
                continue
            feed_a, feed_b, _feed_i = sources.get(node, (None, None, None))
            live_a = (feed_a is not None and feed_a not in sink_subnodes
                      and feed_a in firing)
            live_b = (feed_b is not None and feed_b not in sink_subnodes
                      and feed_b in firing)
            if excite_a is not None and excite_b is not None:
                # Two named excitatory inputs. A coincidence circuit needs both;
                # its OR twin needs either. A config naming the same direction
                # twice resolves to one source and behaves as a buffer.
                ready = (live_a or live_b) if op == 'or' else (
                    live_a and live_b if feed_a != feed_b else live_a)
            else:
                ready = live_a or live_b
            if ready:
                firing.add(node)
                changed = True
    return firing


def driven_roots(grid, pads, roots):
    """Which output roots the inputs can actually drive.

    The readout is a wired-OR of a tile's three output wires, so a root counts
    as driven when ANY of its channels can fire.
    """
    firing = firing_nodes(grid, pads, sink_nodes=set(roots.values()))
    driven = set()
    for label, cell in roots.items():
        if cell in grid and any(
                (cell[0], cell[1], direction) in firing
                for direction in TRI_DIRS):
            driven.add(label)
    return driven


def root_source_counts(grid, pads, roots):
    """Distinct logical pads whose wires can influence each output root.

    This is dependency reachability, so inhibitory and coincidence inputs both
    count even when one source alone cannot make an AND node fire. Declared
    output roots are physical sinks and therefore never relay that influence
    into another role.
    """
    from .tritile import interpret_tri
    info = interpret_tri(grid, pads)
    sink_subnodes = {
        node for tile in roots.values()
        for node in info['tile_nodes'].get(tuple(tile), ())}
    counts = {label: 0 for label in roots}
    for pad in pads:
        source = info['in_nodes'].get(tuple(pad))
        if source is None:
            continue
        live = {source}
        changed = True
        while changed:
            changed = False
            for node, feeders in info['sources'].items():
                if node in live:
                    continue
                if any(feeder is not None and feeder not in sink_subnodes
                       and feeder in live for feeder in feeders):
                    live.add(node)
                    changed = True
        for label, tile in roots.items():
            if any(node in live for node in info['tile_nodes'].get(
                    tuple(tile), ())):
                counts[label] += 1
    return counts


def materialise_pads(grid, pads):
    """Add the input pads to a grown body as live seed tiles.

    Development treats a pad as a read-only piece of CONTEXT - it has its own
    identity, PAD_STATE, and no rule may write it - which is what stops an arm
    from eating its own input. But the simulator has no such concept: an input
    is a cell that exists and conducts. So the pads are materialised here, at
    the boundary between the encoding and the physics, exactly as FNV
    materialises its source pads.
    """
    body = dict(grid)
    for pad in pads:
        body.setdefault(tuple(pad), TRI_SEED_STATE)
    return body


def grow_branched_hex(genome, seeds, grid_size=None, iters=None):
    """Grown body as a plain ``cell -> tile state`` grid, pads included.

    ``grid_size`` and ``iters`` are accepted and ignored: size is bounded by
    each arm's lifespan and by MAX_PLACEMENTS, never by a display grid.
    """
    pads = tuple(tuple(seed) for seed in seeds)
    return materialise_pads(develop_branched_hex(genome, pads).grid, pads)
