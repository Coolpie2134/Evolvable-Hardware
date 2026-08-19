"""
substrates/nervous/branched_ga.py - construction and variation for the branched
hex encoding.

The hard part of a context-rule genome is that a RANDOM rule almost never fires:
its neighbourhood has to occur in the organism for it to do anything. So genes
are drawn from contexts the body ACTUALLY PRESENTS at the moment the gene is
added, and kept only when they change development. That is FNV's
`random_branched_genome` strategy, and it is what makes an initial population
made of living organisms rather than inert rule lists.

Everything here is deterministic given `random.seed`. Note the sorted() calls
around every set that feeds a random draw: iterating a set of tuples containing
DIRECTION STRINGS is hash-order dependent, and that is exactly the bug that made
FNV runs unreproducible for months (see `substrates/fnv/construction_ga.py`
`_observed_context_occurrences`).
"""
from __future__ import annotations

import copy
import math
import random

from ..fnv.genome import input_ring
from .branched import (
    DEPTH_ANY, DEPTH_BANDS, EMPTY_STATE, MAX_PLACEMENTS, OUT_STATE, PAD_STATE,
    BranchedHexChromosome, BranchedHexGenome, HexContextGene, HexControlGene,
    IoChromosome, TAU_SCALES, tau_of,
    HexInputGene, HexOutputGene, arm_reach, bearing_cell,
    develop_branched_hex, driven_roots, growth_candidates, materialise_pads,
    output_root_sites, required_output_directions, root_source_counts,
    _state_of)
from .hexgrid import CANONICAL_STATES, ROUTING_HEX, hex_dirs
from .branched import _channel_interface
from .tritile import TRI_DIRS, channel_configs, pack_channels

DIRECTIONS = ('L', 'R', 'D')
#: Channel configurations a rule may install in one circuit of a tile. Canonical
#: only, so a mutation cannot park a channel on a redundant encoding of a
#: circuit it already had; 0 (the dead channel) stays available because a tile
#: that drives only one direction is a perfectly good part.
LIVE_CHANNELS = tuple(CANONICAL_STATES)
#: ...and the same set without the dead channel, for guaranteeing a live tile.
DRIVING_CHANNELS = tuple(c for c in CANONICAL_STATES if c != 0)
#: Single-source channels - the transport circuits. A channel that ANDs two
#: different neighbours passes nothing unless both fire within the coincidence
#: window, so a limb built from uniformly-drawn configurations is silent almost
#: everywhere: measured, EVERY organism in the first branched populations scored
#: the silent baseline despite growing a connected root-to-pad path. Drawing
#: transport more often is the same prior FNV gets for free from a catalogue
#: that contains wires; coincidence remains one draw away.
TRANSPORT_CHANNELS = tuple(
    c for c in DRIVING_CHANNELS if len(_channel_interface(c)[0]) == 1)
#: How often a fresh channel is drawn from the transport subset.
TRANSPORT_BIAS = 0.6

MAX_TELOMERE = 32
#: An arm's match tolerance is normally 1: enough to let a rule fire on a part
#: that computes something different through the same interface (an AND where it
#: saw an OR twin - distance 0 - or one wire away from what it saw), and not
#: enough to let it fire on a genuinely different neighbourhood. A pad, an
#: unbuilt output niche and empty ground are SENTINEL_DISTANCE apart, so no
#: tolerance in this range can confuse them.
MAX_TOLERANCE = 3
# Complex output-rooted circuits need room for their whole reverse cone. Six
# steps made the exact Full Adder's Cin pad (10) and Sum root (11) genetically
# unreachable even though MAX_PLACEMENTS and arm lifespans could hold the body.
# Random starts remain compact; this is the mutation/compiled-placement ceiling.
# Keep I/O distance evolvable beyond the old compact layout.  Mutation still
# changes distance by only one ring at a time, so this is capacity, not a
# pressure to make every network large.
MAX_IO_DISTANCE = 32
INITIAL_OUTPUT_DISTANCE = 6
#: Chance a growth rule erases instead of building. Pruning has to be reachable
#: or a body can only ever accrete.
ERASE_PROBABILITY = 0.08


def input_pads(genome):
    """Pad cells, with pad zero pinned to the origin as the coordinate gauge."""
    pads = [(0, 0)]
    for gene in genome.inputs:
        cell = bearing_cell(gene.bearing, gene.distance, pads)
        if cell is not None:
            pads.append(cell)
    return tuple(pads)


def _random_placement(max_distance):
    """A complete honeycomb polar allele, not just six sites on a long ring."""
    distance = random.randint(1, max(1, int(max_distance)))
    return random.randrange(len(input_ring(distance))), distance


def _random_output_gene(role, branch_id):
    bearing, distance = _random_placement(INITIAL_OUTPUT_DISTANCE)
    return HexOutputGene(
        role=role, bearing=bearing, distance=distance, branch_id=branch_id)


def observed_contexts(genome, label, trace=None, pads=None):
    """Sorted [(context, band)] the given arm can currently act on.

    ``trace`` and ``pads`` are an optional ALREADY-DEVELOPED organism. Growing
    one costs far more than evaluating it does, so any caller that has just
    developed this genome passes the result in rather than paying for a second,
    identical development.

    Sorted, never a raw set: every context carries states and is keyed by
    tuples, and iterating a set to feed random.choice is how a seeded run stops
    being reproducible.
    """
    pads = set(input_pads(genome) if pads is None else pads)
    roots = output_root_sites(genome, pads)
    if trace is None:
        trace = develop_branched_hex(genome, pads)
    grid, owners, depths = trace.grid, trace.owners, trace.depths
    output_sites = set(roots.values()) - pads

    # EXACTLY the sites development would let this arm write, scoped by the same
    # arm_reach development uses. Drawing genes from a wider neighbourhood than
    # the arm can act on manufactures rules that can never fire.
    found = set()
    root = roots.get(label)
    for cell in growth_candidates(grid, owners, pads, output_sites):
        # The arm's own unbuilt root is reachable BY DEFINITION - its root rule
        # is exempt from arm_reach in development, because that rule is what
        # gives the arm its first ground. Filtering it here left every arm
        # unable to observe the one context it needs to start, and organisms
        # came out empty.
        if cell == root and cell not in grid:
            depth = 0
        else:
            depth = arm_reach(cell, label, owners, depths)
            if depth is None:
                continue
        around = hex_dirs(*cell)
        context = (
            _state_of(around['L'], grid, pads, output_sites),
            _state_of(around['R'], grid, pads, output_sites),
            _state_of(around['D'], grid, pads, output_sites),
            _state_of(cell, grid, pads, output_sites),
        )
        # The directions a part placed here must drive to feed this arm's root.
        # Carried alongside the context so gene proposal can draw a part that
        # CAN fire, instead of drawing blindly and being rejected by
        # drives_toward_root - which is what left bodies half FNV's size.
        directions = required_output_directions(
            cell, label, depth, owners, depths)
        found.add((context, min(depth, DEPTH_BANDS - 1), directions))
    return sorted(found)


def random_channel(driving=False):
    """One circuit for one output wire, biased toward transport."""
    if random.random() < TRANSPORT_BIAS:
        return random.choice(TRANSPORT_CHANNELS)
    return random.choice(DRIVING_CHANNELS if driving else LIVE_CHANNELS)


def random_tile_state(required=()):
    """A live tri-tile: three channel configs, at least one of them driving.

    ``required`` are directions the part must drive to feed the limb it is
    joining (see required_output_directions). Guaranteeing one of them is the
    hex analogue of FNV's ``_compatible_component_id``: the genome still names
    the circuit, but the draw is taken from the parts that can actually fire
    here rather than from all of them.

    An all-dead tile is EMPTY_STATE, the erase instruction rather than a cell,
    so one channel is always drawn from the driving set.
    """
    channels = [random_channel() for _ in TRI_DIRS]
    wanted = [d for d in required if d in TRI_DIRS] or list(TRI_DIRS)
    channels[TRI_DIRS.index(random.choice(sorted(wanted)))] = random_channel(
        driving=True)
    return pack_channels(*channels)


def _random_output_state(allow_erase=True, required=()):
    if allow_erase and random.random() < ERASE_PROBABILITY:
        return EMPTY_STATE
    return random_tile_state(required)


def random_gene(genome, gene_id, label, *, allow_output=True,
                trace=None, pads=None):
    """One rule the arm can actually express, or None if it has no ground."""
    seen = observed_contexts(genome, label, trace=trace, pads=pads)
    roots = [entry for entry in seen if entry[0][3] == OUT_STATE]
    if allow_output and roots:
        context, _band, directions = random.choice(roots)
        return HexContextGene(
            gene_id, context[0], context[1], context[2], OUT_STATE,
            _random_output_state(allow_erase=False, required=directions),
            label, DEPTH_ANY, tau_index=random.randrange(len(TAU_SCALES)))
    body = [entry for entry in seen if entry[0][3] != OUT_STATE]
    if not body:
        return None
    context, band, directions = random.choice(body)
    depth = band if random.random() < 0.5 else DEPTH_ANY
    # Capacitors vary between fabricated nodes, so a fresh organism is drawn
    # with a SPREAD of time constants rather than a uniform one. Starting every
    # node identical is what made the array behave as a unit-delay synchronous
    # net (0 of 257 edges off-grid); heterogeneity is the physical default, not
    # something evolution should have to discover first.
    return HexContextGene(
        gene_id, context[0], context[1], context[2], context[3],
        _random_output_state(required=directions), label, depth,
        tau_index=random.randrange(len(TAU_SCALES)))


def _arm_has_root(genome, label):
    members, _control = genome.arm(label)
    return any(gene.spawns_output() for gene in (members or ()))


def random_branched_hex_genome(n_chroms=2, n_inputs=2, output_roles=('Q',),
                               blocks=24, max_telomere=MAX_TELOMERE,
                               input_genes=None):
    """A fresh genome whose arms actually build something.

    Grows all arms together, one gene at a time, keeping only genes that change
    development. Building one arm to completion before the next gave the first
    role every shared niche and left later roles as read-once stubs - the same
    failure FNV documents.
    """
    roles = tuple(str(role) for role in output_roles)
    if len(roles) > 2 * int(n_chroms):
        raise ValueError('a branched hex genome needs one arm per output role')
    genome = BranchedHexGenome(
        chromosomes=[
            BranchedHexChromosome(controls=[
                HexControlGene(tolerance=random.choice((0, 1, 1, 2)),
                               telomere=random.randint(4, max_telomere))
                for _ in range(2)])
            for _ in range(int(n_chroms))],
        # One dedicated arm per output role, in order: role k gets arm k+1.
        # Any arms left over stay empty - they are spare capacity, not spare
        # outputs.
        io_chromosome=IoChromosome(
            # A cohort may share its source-pad environment. The arm rules
            # below are then constructed against these exact pads, so later
            # crossover exchanges compatible modules rather than transplanting
            # them between unrelated coordinate systems.
            inputs=(copy.deepcopy(list(input_genes)) if input_genes is not None
                    else [HexInputGene(*_random_placement(3))
                          for _ in range(max(0, int(n_inputs) - 1))]),
            outputs=[_random_output_gene(role, index + 1)
                     for index, role in enumerate(roles)]),
        next_gene_id=1)

    # Development is the expensive step here - one growth costs many times what
    # evaluating the grown organism costs - so the loop below performs exactly
    # ONE per attempted gene. The pads are fixed for the whole construction (no
    # input allele changes), and the accepted organism carries forward as the
    # next attempt's starting point rather than being re-grown to look at it.
    pads = input_pads(genome)
    trace = develop_branched_hex(genome, pads)
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
                gene = random_gene(
                    genome, genome.next_gene_id, label,
                    allow_output=not _arm_has_root(genome, label),
                    trace=trace, pads=pads)
                if gene is None:
                    break
                chromosome.genes.append(gene)
                grown = develop_branched_hex(genome, pads)
                if grown.grid != trace.grid:
                    genome.next_gene_id += 1
                    trace = grown          # keep the organism it just built
                    placed = True
                    break
                chromosome.genes.pop()
            if not placed:
                growing.discard(label)
    return genome


#: How many random starts to look at before picking one. FNV does the same
#: thing (substrates/fnv/ga.select_developmental_seed) and for the same reason.
DEVELOPMENTAL_SEED_CANDIDATES = 6


def select_developmental_seed(make_genome,
                              attempts=DEVELOPMENTAL_SEED_CANDIDATES):
    """Pick the most connected of a few random starts. Target-blind.

    An organism whose output root nothing can drive scores the silent baseline
    no matter what else is true of it, and MOST random starts are that: measured
    over 80 fresh genomes, 34 of the 59 viable ones had a permanently silent
    root. A population of those is not a weak population, it is an
    undifferentiated one - selection cannot tell its members apart, so the run
    has no gradient to climb at all.

    This looks only at whether the inputs can reach the outputs, never at the
    target or its cases, so it changes where search STARTS without touching what
    counts as fitness. Ties break toward the larger organism, then the smaller
    genome, so the choice is deterministic under a seeded run.
    """
    def key(genome):
        pads = input_pads(genome)
        grid = materialise_pads(
            develop_branched_hex(genome, pads).grid, pads)
        roots = output_root_sites(genome, pads)
        counts = root_source_counts(grid, pads, roots)
        coverage = tuple(counts.get(label, 0) for label in sorted(roots))
        return (min(coverage, default=0), sum(coverage),
                len(driven_roots(grid, pads, roots)), len(grid),
                -sum(len(c.genes) for c in genome.chromosomes))

    # Built and judged one at a time, stopping the moment a start drives every
    # root: building a genome is the expensive operation in this whole encoding,
    # so the common case should pay for one or two, not for the whole batch.
    best, best_key = None, None
    for _ in range(max(1, int(attempts))):
        candidate = make_genome()
        score = key(candidate)
        if best_key is None or score > best_key:
            best, best_key = candidate, score
        if (score[0] >= len(input_pads(candidate))
                and score[2] >= len(candidate.outputs)):
            break                      # every source can drive every role
    return best


# -- variation ------------------------------------------------------------------

def clone_branched_hex(genome):
    return copy.deepcopy(genome)


def _mutation_count(mean):
    """At-least-one Poisson draw whose mean is the advertised edit rate."""
    lam = max(0.0, float(mean))
    if lam <= 0.0:
        return 1
    limit, product, floor = 0, 1.0, math.exp(-lam)
    while product > floor:
        limit += 1
        product *= random.random()
    return max(1, limit - 1)


def _different(current, choices):
    alternatives = [choice for choice in choices if choice != current]
    return random.choice(alternatives) if alternatives else current


def _nudge_tile(state):
    """Rebuild ONE circuit of a tile, leaving the other two alone.

    A tile is three independent circuits, so redrawing the whole 15-bit state is
    a macro-mutation: it discards two working channels to change one. This is
    the local move, and it is what lets an arm tune the channel that actually
    drives its neighbour without losing the channels feeding it.
    """
    if state in (EMPTY_STATE, PAD_STATE, OUT_STATE):
        return random_tile_state()
    channels = list(channel_configs(state))
    index = random.randrange(len(TRI_DIRS))
    current = channels[index]
    e1, e2, inhibit, operation = ROUTING_HEX[current]
    same_wiring = [
        candidate for candidate in DRIVING_CHANNELS
        if (candidate != current
            and ROUTING_HEX[candidate][:3] == (e1, e2, inhibit))]
    one_wire = []
    for candidate in DRIVING_CHANNELS:
        if candidate == current or ROUTING_HEX[candidate][3] != operation:
            continue
        fields = ROUTING_HEX[candidate][:3]
        changed = sum(a != b for a, b in zip(fields, (e1, e2, inhibit)))
        if changed == 1:
            one_wire.append(candidate)
    same_interface = [
        candidate for candidate in DRIVING_CHANNELS
        if (candidate != current
            and _channel_interface(candidate)[0]
            == _channel_interface(current)[0])]
    roll = random.random()
    if roll < 0.45 and same_wiring:
        replacement = random.choice(same_wiring)
    elif roll < 0.78 and one_wire:
        replacement = random.choice(one_wire)
    elif roll < 0.92 and same_interface:
        replacement = random.choice(same_interface)
    else:
        replacement = _different(current, DRIVING_CHANNELS)
    channels[index] = replacement
    if not any(channels):
        return EMPTY_STATE
    return pack_channels(*channels)


def _mutate_context_gene(gene, genome, label):
    """Nudge one rule, preferring moves that keep it expressible."""
    choice = random.random()
    if choice < 0.35:
        # Two thirds of the time, retune one circuit rather than the whole
        # tile: the macro redraw is still reachable, just no longer the only
        # way to change what a rule builds.
        candidate = gene.self_out
        for _ in range(4):
            candidate = (_random_output_state() if random.random() < 0.34
                         else _nudge_tile(gene.self_out))
            if candidate != gene.self_out:
                break
        gene.self_out = candidate
    elif choice < 0.55:
        seen = observed_contexts(genome, label)
        body = [entry for entry in seen
                if entry[0][3] != OUT_STATE and entry[0] != gene.context]
        if body and not gene.spawns_output():
            # Re-aims the rule WITHOUT redrawing its part: the part is what the
            # rule has been selected for, and replacing it on every context
            # nudge measured worse on every target.
            context, _band, _directions = random.choice(body)
            gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in = context
    elif choice < 0.68:
        # TIMING, mutated on its own. A cell's capacitor is not part of its
        # routing: retuning a node must not also rewire it, or timing could
        # never be tuned while holding a working circuit fixed.
        gene.tau_index = _different(
            gene.tau_index, range(len(TAU_SCALES)))
    elif choice < 0.80:
        gene.depth = (DEPTH_ANY if gene.depth != DEPTH_ANY
                      else random.randrange(DEPTH_BANDS))
    elif not gene.spawns_output():
        which = random.randrange(3)
        field_name = ('ctx_l', 'ctx_r', 'ctx_d')[which]
        roll = random.random()
        if roll < 0.2:
            wanted = EMPTY_STATE
        elif roll < 0.35:
            wanted = PAD_STATE
        else:
            wanted = _nudge_tile(getattr(gene, field_name))
        if wanted == getattr(gene, field_name):
            wanted = _different(
                getattr(gene, field_name),
                (EMPTY_STATE, PAD_STATE, random_tile_state()))
        setattr(gene, field_name, wanted)
    return gene


def mutate_branched_hex(genome, mean_mutations=2.0, max_telomere=MAX_TELOMERE):
    """Poisson-many edits: rules, arm controls, and I/O alleles."""
    child = clone_branched_hex(genome)
    count = 0
    limit = _mutation_count(mean_mutations)
    while count < limit:
        count += 1
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
                control.telomere = _different(control.telomere, [
                    max(1, min(max_telomere, control.telomere + delta))
                    for delta in (-4, -1, 1, 4)])
            else:
                control.tolerance = _different(control.tolerance, [
                    max(0, min(MAX_TOLERANCE,
                               control.tolerance + delta))
                    for delta in (-1, 1)])
        else:
            # I/O alleles: slide a pad or a root around its ring, or in and out.
            # Pad geometry is the arm-exchange environment.  Treat changing it
            # as the rarer macro mutation, while output-root moves remain a
            # normal way to retune one role's developed limb.
            pool = (list(child.outputs)
                    if child.outputs and (
                        not child.inputs or random.random() < 0.8)
                    else list(child.inputs))
            if pool:
                allele = random.choice(pool)
                if random.random() < 0.5:
                    ring_size = len(input_ring(max(1, int(allele.distance))))
                    allele.bearing = (int(allele.bearing)
                                      + random.choice((-1, 1))) % ring_size
                else:
                    allele.distance = _different(int(allele.distance), [
                        max(1, min(MAX_IO_DISTANCE,
                                   int(allele.distance) + delta))
                        for delta in (-1, 1)])
    return child


def crossover_branched_hex(left, right):
    """Swap whole ARMS - the unit that owns a territory and a lifespan.

    Trading individual rules would mix genes that were selected against
    different bodies; an arm is the smallest piece that means something on its
    own, which is why FNV trades arms too.
    """
    child = clone_branched_hex(left)
    # Arms were developed against their parents' concrete source pads.  Do not
    # graft one into a different pad geometry; that is a random transplant, not
    # recombination.  Parent selection normally finds a compatible mate, while
    # this guard keeps direct callers and sparse populations safe.
    if input_pads(left) != input_pads(right):
        return child
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
        # The root geometry is part of the arm's phenotype: its rules grew
        # backward from this site. Swapping only rules would recreate the same
        # invalid transplant the shared-pad guard prevents.
        donor_output = next(
            (gene for gene in right.outputs
             if int(gene.branch_id) == label), None)
        if donor_output is not None:
            for output_index, own_output in enumerate(child.outputs):
                if (int(own_output.branch_id) == label
                        and str(own_output.role) == str(donor_output.role)):
                    child.outputs[output_index] = copy.deepcopy(donor_output)
                    break
    child.next_gene_id = max(
        [int(left.next_gene_id), int(right.next_gene_id)]
        + [int(gene.gene_id) + 1
           for chromosome in child.chromosomes for gene in chromosome.genes])
    return child


def plateau_rescue_candidates(
        genome, target, limit=48, max_telomere=MAX_TELOMERE):
    """Verified arithmetic seed for a stalled live nervous-net population.

    The candidate remains an ordinary branched genome and earns its score only
    by development plus the normal paper-analog evaluation.  This closes the
    previous gap where LUT had a truth-table plateau seed but the harder
    nervous encoding had no rescue at all.
    """
    if int(limit) <= 0 or not isinstance(genome, BranchedHexGenome):
        return []
    from .branched_synthesis import (
        SynthesisError, synthesize_branched_full_adder)
    from .logic_synthesis import synthesize_branched_logic
    from .state_synthesis import synthesize_branched_dynamic
    from .ga import evaluate_nv_full
    for compiler in (synthesize_branched_full_adder,
                     synthesize_branched_logic,
                     synthesize_branched_dynamic):
        try:
            candidate = compiler(
                target, chromosome_count=len(genome.chromosomes),
                max_telomere=max_telomere)
        except SynthesisError:
            continue
        fitness, cases = evaluate_nv_full(candidate, target)
        if fitness == 1.0 and cases and min(cases) == 1.0:
            return [candidate]
    return []


def assemble_role_modules(base, donors):
    """Build one unmutated organism from compatible evaluated output arms."""
    child = clone_branched_hex(base)
    pads = input_pads(base)
    selected = {
        int(label): donor for label, donor in dict(donors).items()
        if input_pads(donor) == pads}
    if not selected:
        return child
    for index, chromosome in enumerate(child.chromosomes):
        genes = []
        for half in (0, 1):
            label = 2 * index + half + 1
            donor = selected.get(label, base)
            if index >= len(donor.chromosomes):
                donor = base
            genes.extend(copy.deepcopy(gene)
                         for gene in donor.chromosomes[index].genes
                         if int(gene.branch_id) == label)
            if half < len(donor.chromosomes[index].controls):
                chromosome.controls[half] = copy.deepcopy(
                    donor.chromosomes[index].controls[half])
        chromosome.genes = genes
    child.io_chromosome.outputs = [
        copy.deepcopy(next(
            (gene for gene in selected.get(int(own.branch_id), base).outputs
             if (int(gene.branch_id) == int(own.branch_id)
                 and str(gene.role) == str(own.role))), own))
        for own in child.outputs]
    child.next_gene_id = max(
        [int(base.next_gene_id)] + [int(donor.next_gene_id)
                                    for donor in selected.values()]
        + [int(gene.gene_id) + 1
           for chromosome in child.chromosomes for gene in chromosome.genes])
    return child


# -- interpretation -------------------------------------------------------------

def prepare_branched_hex(genome, target):
    """(grid, input_pos, output_pos) or None when the organism is incomplete.

    I/O comes from the GENOME - pads are its input genes, output cells are its
    arm roots - rather than from probes fitted after the fact. That is the whole
    point of an output-rooted encoding: the arm has to know where it is growing
    from before it grows.
    """
    pads = input_pads(genome)
    if len(pads) < int(getattr(target, 'n_inputs', len(pads))):
        return None
    body = develop_branched_hex(genome, pads).grid
    if not body:
        return None
    grid = materialise_pads(body, pads)
    roots = output_root_sites(genome, pads)
    outputs = {}
    for gene in genome.outputs:
        cell = roots.get(int(gene.branch_id))
        if cell is None or cell not in grid:
            return None          # a role whose root was never built
        outputs[str(gene.role)] = cell
    if len(set(outputs.values())) != len(outputs):
        return None
    return grid, list(pads[:int(getattr(target, 'n_inputs', len(pads)))]), outputs


def prepare_branched_net(genome, target, *, root_outputs=True):
    """Grow a branched genome and hand it to the existing nervous evaluator.

    Returns ``prepare_net``'s contract - (grid, routing, in_pos, out_pos,
    traces) - or None for an unusable organism, so scoring, certification and
    the GA batch evaluator all work unchanged.

    ``root_outputs`` is what makes this an OUTPUT-ROOTED encoding rather than a
    branched one with fitted probes bolted on: the role sites come from the
    genome's own arm roots. Set it false to fall back to trace-matched fitted
    probes, which is useful only for isolating whether an effect comes from the
    encoding or from the readout.
    """
    from .nervous import interpret_nervous
    from .temporal import prepare_net_grid, trace_fixed_outputs

    pads = input_pads(genome)
    n_inputs = int(getattr(target, 'n_inputs', len(pads)))
    if len(pads) < n_inputs:
        return None
    trace = develop_branched_hex(genome, pads)
    developed = trace.grid
    if not developed:
        return None
    # Only target-bound pads are physical sources.  Older/evolved genomes may
    # carry spare placement genes; materialising those as ordinary tri tiles
    # (without adding them to ``in_pos``) decoded the PAD sentinel as live
    # circuitry and injected phantom activity.  An unbound pad is an absent,
    # therefore logic-zero, site.
    grid = materialise_pads(developed, pads[:n_inputs])
    # Each cell leaks on the capacitor of the rule that built it. Resolved to
    # ABSOLUTE constants here, against the run's base value, so the physics
    # config stays the single source of scale and the genome only ever names a
    # multiple. Pads keep the base constant: they are the interface, not evolved
    # body.
    base = getattr(getattr(target, 'pulse_config', None),
                   'analog_tau_leak', None)
    taus = ({cell: tau_of(index, base)
             for cell, index in trace.taus.items()} if base else None)
    if not root_outputs:
        return prepare_net_grid(genome, target, grid, strategy='fixed',
                               taus=taus)
    roots = output_root_sites(genome, pads)
    out_pos = {}
    for gene in genome.outputs:
        cell = roots.get(int(gene.branch_id))
        if cell is None or cell not in developed:
            return None            # a role whose root was never built
        out_pos[str(gene.role)] = cell
    if len(set(out_pos.values())) != len(out_pos):
        return None
    in_pos = list(pads[:n_inputs])
    arch = getattr(genome, 'arch', 'tri3')
    routing, _fitted_in, _fitted_out = interpret_nervous(
        grid, target, arch=arch)
    # Traced AT THE ROOTS. Reusing prepare_net_grid's traces would have read the
    # probe cells IT fitted while reporting the arm roots as the outputs - the
    # scores and the circuit would have described different cells.
    traces = trace_fixed_outputs(
        grid, routing, in_pos, out_pos, target, arch=arch,
        source_nodes={tuple(pad) for pad in in_pos}, taus=taus,
        sink_nodes=set(out_pos.values()))
    if traces is None:
        return None
    return grid, routing, in_pos, out_pos, traces
