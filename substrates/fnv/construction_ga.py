"""Genetic operators for branched FNV genomes."""
from __future__ import annotations

import copy
import heapq
import itertools
import math
import random

from substrates.nervous.hexgrid import hex_dirs, honeycomb_distance

from .catalogue import (
    BY_ID, FAMILIES, behavior_component_ids, enabled_component_ids,
    normalise_families,
)
from .construction import (
    _reach, _state_of, arm_control, branch_arms, branch_growth_order,
    develop_constructive, output_branch_sites, required_output_directions)
from .genome import (
    DEPTH_ANY, DEPTH_BANDS, EMPTY_STATE, MAX_ARM_TELOMERE, MAX_GENES,
    MAX_INPUT_DISTANCE, MAX_PLACEMENTS, MAX_TOLERANCE, OUT_STATE,
    Chromosome, ContextGene, ControlGene, InputGene, OutputGene, input_ring,
    sync_input_layout, sync_output_layout)
from .simulation import effective_wiring_edges, facing_direction, source_for_input

BRANCHED_MUT_OPS = ["tweak", "add_gene", "connect", "block", "del_rule",
                    "del_branch", "control", "inputs", "outputs"]
#: ``block`` is OFF (weight 0). It is mechanically correct - see _add_blocker -
#: but measured inert, and it cannot work until genes stop going stale: only
#: 10% of banded genes have a context that still occurs at their band, so a
#: blocker almost never meets the rule it was built to disagree with. Half adder
#: over 8 seeds, 200 gens: 6/8 with it against 7/8 without. Re-enable by giving
#: it weight once genes are kept attached to contexts that currently exist.
BRANCHED_MUT_WEIGHTS = [
    0.55, 0.08, 0.15, 0.00, 0.04, 0.01, 0.08, 0.045, 0.045,
]
#: Softmax temperature for the constructive component draw (`_sample_component`).
#: 0 reproduces the old deterministic argmin; larger values flatten the draw
#: toward uniform-over-families. 0.75 was chosen as the smallest value that
#: brought every enabled family into the grown phenotype at a measurable rate
#: while leaving DELAY the plurality part for plain routing.
CONSTRUCTION_TEMPERATURE = 0.75
#: Probability that a constructive step attempts to close a FEEDBACK edge back
#: into the branch it is already growing, instead of extending forward. Growth
#: was strictly output-rooted and acyclic, so 0 of 400 grown bodies contained a
#: cycle - and every stateful target (oscillators, dividers, latches, toggles)
#: needs one. Reserved for the feedback-closing step; not yet consumed.
FEEDBACK_CLOSE_PROBABILITY = 0.25
#: Starting lifespan of a fresh arm.
ARM_TELOMERE_SEED = (24, 32)
#: Fresh output niches start beyond the compact input cluster.  A root only two
#: edges from the pads leaves enough room for a read-once gate, but not for the
#: repeated terminal contacts needed by voting, carry, and multi-bit arithmetic.
#: Distance remains genetic and may evolve across the full domain afterward.
OUTPUT_DISTANCE_SEED = (3, 4)
#: Build a genuinely branching gate crown before terminal tropism takes over.
#: This is target-blind developmental competence: it supplies reusable input
#: limbs, not a desired truth table or a prescribed circuit.
LOGIC_SCAFFOLD_GENES = 6
MAX_LOGIC_SCAFFOLD_GENES = 12
#: Starting reach of a fresh arm. Zero is an exact match, which is where this
#: encoding began; a few units lets a rule apply to neighbouring component types
#: without reaching across catalogue families.
ARM_TOLERANCE_SEED = (0, 12)
#: How often a fresh gene is pinned to ONE depth band rather than applying all
#: along its branch. Positional rules are what give a limb a base and a tip, but
#: a genome made only of them cannot build the limb they sit on.
DEPTH_SPECIFIC_PROBABILITY = 0.15
#: Of those, how often the band is the OUTERMOST one its context occurs at - a
#: TERMINATOR. Retyping the tip is what ends a chain: the rule that was
#: propagating it keyed on the old state, so once the tip changes neither that
#: rule nor the terminator matches again, and the limb stops. Growth otherwise
#: has no gene-level brake at all - measured, 60/60 bodies stopped only because
#: no rule happened to match.
TERMINATOR_BIAS = 0.6
#: How often a fresh gene is keyed on an OCCUPIED cell - a retype or an erasure
#: - rather than on empty space. Without this the sample is dominated by the
#: frontier, because empty neighbours outnumber grown cells: measured 93% of all
#: cell changes were births, 7% retypes and 0% erasures, so a body was written
#: once and then frozen.
OCCUPIED_CONTEXT_PROBABILITY = 0.5
#: How often a fresh gene ERASES rather than builds. Erasure is what lets a body
#: be carved as well as grown, but a genome mostly made of it never accumulates
#: anything.
ERASE_PROBABILITY = 0.12
_BEHAVIOR_CURSORS = {}
_BEHAVIOR_REPERTOIRES = {}
_BEHAVIOR_CURSOR_MAX = 10_000


def clone_constructive(genome):
    cache_key = getattr(genome, "_fnv_development_cache_key", None)
    cache = getattr(genome, "_fnv_development_cache", None)
    clone = copy.copy(genome)
    # ``Genome.__getstate__`` correctly strips caches from multiprocessing and
    # checkpoints; an in-process structural clone is different. It begins with
    # an identical genotype, so sharing the immutable-by-convention trace avoids
    # one otherwise redundant growth pass before its first edit.
    if cache_key is not None and cache is not None:
        clone._fnv_development_cache_key = cache_key
        clone._fnv_development_cache = cache
    inputs = getattr(genome, "input_chromosome", None)
    if inputs is not None:
        clone.input_chromosome = copy.copy(inputs)
        clone.input_chromosome.genes = [copy.copy(g) for g in inputs.genes]
    outputs = getattr(genome, "output_chromosome", None)
    if outputs is not None:
        clone.output_chromosome = copy.copy(outputs)
        clone.output_chromosome.genes = [copy.copy(g) for g in outputs.genes]
    clone.chromosomes = []
    for chromosome in genome.chromosomes:
        copied = copy.copy(chromosome)
        copied.genes = [copy.copy(gene) for gene in chromosome.genes]
        clone.chromosomes.append(copied)
    layout = getattr(genome, "input_layout", None)
    if layout is not None:
        clone.input_layout = tuple(tuple(cell) for cell in layout)
    clone.output_layout = tuple(
        (str(role), tuple(cell)) for role, cell in
        (getattr(genome, "output_layout", ()) or ()))
    clone._sampled_branch_capacities = dict(
        getattr(genome, "_sampled_branch_capacities", {}))
    return clone


def placement_genes(genome):
    return tuple(
        gene for chromosome in genome.chromosomes
        for gene in chromosome.genes if isinstance(gene, ContextGene))


def chromosome_rules(chromosome):
    """Every gene of a chromosome - context rules and control genes alike."""
    return list(chromosome.genes)


def context_rules(chromosome):
    return [gene for gene in chromosome.genes
            if isinstance(gene, ContextGene)]


def branch_cut(chromosome):
    """The centromere: where a chromosome divides into its two arms."""
    return max(0, min(int(chromosome.split),
                      len(chromosome_rules(chromosome))))


def branch_map(genome):
    """``(chromosome index, arm) -> that arm's CONTEXT rules``."""
    branches = {}
    for index, chromosome in enumerate(genome.chromosomes):
        arms = branch_growth_order(chromosome)
        branches[(index, 0)] = list(arms[0])
        branches[(index, 1)] = list(arms[1])
    return branches


def arm_map(genome):
    """``(chromosome index, arm) -> every gene of that arm``."""
    arms = {}
    for index, chromosome in enumerate(genome.chromosomes):
        both = branch_arms(chromosome)
        arms[(index, 0)] = list(both[0])
        arms[(index, 1)] = list(both[1])
    return arms


def new_control_gene(gene_id, max_telomere=None):
    """A fresh arm's control gene: how far it reaches and how long it lives."""
    ceiling = arm_telomere_ceiling(max_telomere)
    low, high = ARM_TELOMERE_SEED
    return ControlGene(
        gene_id,
        tolerance=random.randint(*ARM_TOLERANCE_SEED),
        telomere=random.randint(min(low, ceiling), min(high, ceiling)),
        branch_id=gene_id)


def branch_label(chromosome_index, half):
    """Stable non-zero id for one arm position."""
    return 2 * int(chromosome_index) + int(half) + 1


def relabel_branches(genome):
    """Stamp every gene with the arm it currently sits in."""
    for (index, half), members in arm_map(genome).items():
        label = branch_label(index, half)
        for gene in members:
            gene.branch_id = label
    return genome


def assigned_branch_labels(genome):
    chromosome = getattr(genome, "output_chromosome", None)
    return {
        int(gene.branch_id)
        for gene in (chromosome.genes if chromosome is not None else ())
        if isinstance(gene, OutputGene)}


def branched_signature(genome):
    """Everything that changes what this organism grows."""
    return (
        tuple(tuple(cell) for cell in (getattr(
            genome, "input_layout", None) or ())),
        tuple(
            (int(gene.gene_id), int(gene.ctx_l), int(gene.ctx_r),
             int(gene.ctx_d), int(gene.self_in), int(gene.self_out),
             int(gene.branch_id), int(gene.depth))
            for gene in sorted(
                placement_genes(genome), key=lambda item: int(item.gene_id))),
        tuple(
            (int(gene.gene_id), int(gene.tolerance), int(gene.telomere),
             int(gene.branch_id))
            for gene in sorted(
                (g for c in genome.chromosomes for g in c.genes
                 if isinstance(g, ControlGene)),
                key=lambda item: int(item.gene_id))),
        tuple(
            (int(branch_cut(chromosome)),
             tuple(int(gene.gene_id) for gene in chromosome.genes))
            for chromosome in genome.chromosomes),
        tuple(
            (int(gene.gene_id), int(gene.distance), int(gene.bearing))
            for gene in (getattr(genome, "input_chromosome", None).genes
                         if getattr(genome, "input_chromosome", None) else ())),
        tuple(
            (int(gene.gene_id), str(gene.role), int(gene.distance),
             int(gene.bearing), int(gene.branch_id))
            for gene in (getattr(genome, "output_chromosome", None).genes
                         if getattr(genome, "output_chromosome", None) else ())),
        tuple((str(role), tuple(cell)) for role, cell in
              (getattr(genome, "output_layout", ()) or ())),
    )


def branched_morphology_signature(genome):
    """Developmental body plan with fixed gate behavior factored out."""
    def route(state):
        value = int(state)
        if value <= EMPTY_STATE:
            return value
        entry = BY_ID[value]
        return (tuple(sorted(entry.inputs)), tuple(entry.outputs))

    return (
        tuple(tuple(cell) for cell in (getattr(
            genome, "input_layout", None) or ())),
        tuple((str(role), tuple(cell)) for role, cell in
              (getattr(genome, "output_layout", ()) or ())),
        tuple(
            (int(branch_cut(chromosome)), tuple(
                ("control", int(gene.tolerance), int(gene.telomere))
                if isinstance(gene, ControlGene) else
                ("rule", gene.context, route(gene.self_out), int(gene.depth))
                for gene in chromosome.genes))
            for chromosome in genome.chromosomes),
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


def arm_telomere_ceiling(max_telomere=None):
    """The run's Max telomere, or the built-in default when none is given."""
    if max_telomere is None:
        return MAX_ARM_TELOMERE
    return max(1, int(max_telomere))


def new_input_chromosome(genome, n_inputs):
    """The extra chromosome placing the pads, one gene per pad after the first.

    Pad zero is the coordinate gauge and has no gene: every relative arrangement
    is still reachable, and the organism cannot drift into identical copies of
    itself.
    """
    from .genome import Chromosome
    genes = []
    for index in range(1, max(1, int(n_inputs))):
        gene_id = int(genome.next_gene_id)
        genome.next_gene_id = gene_id + 1
        distance = random.randint(1, min(3, MAX_INPUT_DISTANCE))
        genes.append(InputGene(
            gene_id, distance, random.randrange(len(input_ring(distance))),
            gene_id))
    return Chromosome(genes=genes, split=0, tag=random.randint(0, 999))


def new_output_chromosome(genome, output_roles):
    """One genetic site per role, bound to one stable arm slot."""
    from .genome import Chromosome
    genes = []
    pads = set(getattr(genome, "input_layout", ()) or ())
    occupied = set(pads)
    output_cells = []
    for index, role in enumerate(output_roles):
        gene_id = int(genome.next_gene_id)
        genome.next_gene_id = gene_id + 1
        choices = []
        low, high = OUTPUT_DISTANCE_SEED
        for distance in range(
                min(low, MAX_INPUT_DISTANCE),
                min(high, MAX_INPUT_DISTANCE) + 1):
            for bearing in range(len(input_ring(distance))):
                candidate = OutputGene(
                    gene_id, str(role), distance, bearing, index + 1)
                cell = candidate.cell()
                if (cell not in occupied
                        and all(cell not in hex_dirs(*pad).values()
                                for pad in pads)):
                    choices.append(candidate)
        if choices and output_cells:
            separation = {
                id(candidate): min(
                    honeycomb_distance(candidate.cell(), cell)
                    for cell in output_cells)
                for candidate in choices}
            best = max(separation.values())
            choices = [candidate for candidate in choices
                       if separation[id(candidate)] == best]
        gene = random.choice(choices) if choices else OutputGene(
            gene_id, str(role), 1, random.randrange(len(input_ring(1))),
            index + 1)
        genes.append(gene)
        occupied.add(gene.cell())
        output_cells.append(gene.cell())
    return Chromosome(genes=genes, split=0, tag=random.randint(0, 999))


def _mutate_inputs(genome):
    """Slide one pad around the anchor, or in and out from it.

    Pads are terminal cues. Moving one changes where output-rooted development
    must make contact, while preserving every other genetic coordinate.
    """
    chromosome = getattr(genome, "input_chromosome", None)
    genes = [gene for gene in (chromosome.genes if chromosome else ())
             if isinstance(gene, InputGene)]
    if not genes:
        return False
    gene = random.choice(genes)
    if random.random() < 0.5:
        ring = len(input_ring(max(1, min(int(gene.distance),
                                         MAX_INPUT_DISTANCE))))
        step = random.choice([-2, -1, 1, 2])
        gene.bearing = (int(gene.bearing) + step) % max(1, ring)
    else:
        options = [int(gene.distance) + delta for delta in (-1, 1)
                   if 1 <= int(gene.distance) + delta <= MAX_INPUT_DISTANCE]
        if not options:
            return False
        gene.distance = random.choice(options)
        gene.bearing = int(gene.bearing) % len(input_ring(gene.distance))
    sync_input_layout(genome)
    sync_output_layout(genome)
    return True


def _mutate_outputs(genome):
    """Slide one role's output root around the shared coordinate frame."""
    chromosome = getattr(genome, "output_chromosome", None)
    genes = [gene for gene in (chromosome.genes if chromosome else ())
             if isinstance(gene, OutputGene)]
    if not genes:
        return False
    gene = random.choice(genes)
    if random.random() < 0.5:
        ring = len(input_ring(max(1, min(int(gene.distance),
                                         MAX_INPUT_DISTANCE))))
        gene.bearing = (int(gene.bearing) + random.choice((-2, -1, 1, 2))) \
            % max(1, ring)
    else:
        options = [int(gene.distance) + delta for delta in (-1, 1)
                   if 1 <= int(gene.distance) + delta <= MAX_INPUT_DISTANCE]
        if not options:
            return False
        gene.distance = random.choice(options)
        gene.bearing = int(gene.bearing) % len(input_ring(gene.distance))
    sync_output_layout(genome)
    return True


def new_branched_chromosome(max_telomere=None):
    """An empty chromosome. Its arms gain control genes when they gain rules."""
    from .genome import Chromosome
    return Chromosome(genes=[], split=0, tag=random.randint(0, 999))


def observed_contexts(genome, seeds, label=None):
    """Neighbourhoods that occur where a gene could actually act.

    Matching is EXACT, so a gene drawn from thin air almost never fires. Drawing
    its context from one the body really presents is what makes a new gene
    expressible. Passing ``label`` narrows that to the ground ONE arm can reach -
    its own territory and the ring around it - because a gene only acts on its
    own branch. Sampling the whole organism instead keys new genes on contexts
    that only exist in someone else's limb, where they can never fire.

    Developmental state only: no target, truth table or output role.
    """
    return {context for context, _band, _directions
            in _observed_context_occurrences(genome, seeds, label)}


def observed_context_bands(genome, seeds, label=None):
    """``(context, depth band)`` pairs occurring where a gene could act.

    The band matters as much as the context. A depth-specific rule drawn with a
    RANDOM band almost never matches the band its context actually occurs at -
    measured, such genes were a third of the genome and 2.1% of them ever fired.
    """
    return {(context, band) for context, band, _directions
            in _observed_context_occurrences(genome, seeds, label)}


def _observed_context_occurrences(genome, seeds, label=None):
    """Current expressible sites, including the wire direction a bud must drive."""
    trace = develop_constructive(genome, seeds)
    grid, pads = trace.grid, set(seeds)
    branch_roots = output_branch_sites(genome)
    output_sites = set(branch_roots.values()) - pads
    if label is None:
        reachable = {cell for cell in grid if cell not in pads}
    else:
        reachable = {cell for cell, owner in trace.owners.items()
                     if owner == label}
    cells = set(reachable)
    for destination in list(reachable):
        state = grid.get(destination)
        if state is None:
            continue
        for direction in BY_ID[state].inputs:
            source = source_for_input(destination, direction)
            if source in pads:
                continue
            if (label is None or source not in grid
                    or trace.owners.get(source) == label):
                cells.add(source)
    if label is not None and label in branch_roots:
        # A branch that owns nothing still has one reachable place: its own
        # writable output niche. No PAD grants initial territory anymore.
        cells.add(branch_roots[label])
    found = set()
    for cell in cells:
        around = hex_dirs(*cell)
        context = (
            _state_of(around['L'], grid, pads, output_sites),
            _state_of(around['R'], grid, pads, output_sites),
            _state_of(around['D'], grid, pads, output_sites),
            _state_of(cell, grid, pads, output_sites),
        )
        depth = trace.branch_depths.get(cell)
        if depth is None:
            if label is not None and branch_roots.get(label) == cell:
                depth = 0
            elif label is not None:
                depth = _reach(
                    cell, label, grid, trace.owners, trace.branch_depths)
            else:
                depth = 0
        if depth is None:
            continue
        band = min(int(depth), DEPTH_BANDS - 1)
        directions = (
            required_output_directions(
                cell, label, depth, grid, trace.owners, trace.branch_depths)
            if label is not None else ())
        found.add((context, band, directions))
    # SORTED, not the raw set. Every element carries `directions`, a tuple of
    # direction STRINGS, and CPython randomises string hashing per process - so
    # iterating the set yields a different order in every run. That order
    # becomes the candidate list handed to random.choice in _random_gene, which
    # made a fixed seed produce a different evolution on each invocation
    # (Majority-3 solved at generation 7, 24, 17, 22 and 30 on one seed).
    # The contents were always identical; only the order moved.
    return sorted(found)


def _compatible_component_id(families, directions=(), min_inputs=0):
    """Family-first component draw that can physically feed a downstream bud."""
    enabled = normalise_families(families)
    required = set(directions)
    by_family = {
        family: [
            state for state in enabled_component_ids((family,))
            if state != EMPTY_STATE
            and len(BY_ID[state].inputs) >= int(min_inputs)
            and (not required or required.intersection(BY_ID[state].outputs))
        ]
        for family in FAMILIES if family in enabled
    }
    available = [family for family, states in by_family.items() if states]
    if not available and min_inputs:
        return _compatible_component_id(families, directions, 0)
    if not available:
        return None
    return random.choice(by_family[random.choice(available)])


def _random_output(families, directions=(), min_inputs=0, *, allow_erase=True):
    """A physically compatible component, or empty when pruning is allowed."""
    if allow_erase and random.random() < ERASE_PROBABILITY:
        return EMPTY_STATE
    return _compatible_component_id(families, directions, min_inputs)


def _random_gene(genome, gene_id, families, seeds, allow_output, label=None,
                 prefer_growth=False):
    """One rule, drawn from a neighbourhood its own arm can actually reach.

    A positional rule takes a band the context REALLY occurs at, usually the
    outermost one, so it lands as a terminator at the tip of a limb rather than
    on a band that never comes round.
    """
    seen = [(context, band, directions)
            for context, band, directions in _observed_context_occurrences(
                genome, seeds, label)
            if (allow_output or context[3] != OUT_STATE)
            and OUT_STATE not in context[:3]]
    if not seen:
        return None
    bands = {}
    for context, band, directions in seen:
        bands.setdefault(context, []).append(band)
    # Deliberately prefer a neighbourhood sitting on an OCCUPIED cell when one
    # is offered, so retype and erase rules get made at all. Empty frontier
    # cells otherwise dominate the sample and every gene is a birth rule.
    occupied = [row for row in seen if row[0][3] != EMPTY_STATE]
    empty = [row for row in seen if row[0][3] == EMPTY_STATE]
    if prefer_growth and empty:
        # Build fresh organisms breadth-first. Choosing only the shallowest open
        # physical input ports stops one lucky rule at a tip from consuming the
        # whole initialization budget as a long wire before its sibling port is
        # ever represented in the genome.
        shallowest = min(row[1] for row in empty)
        context, _site_band, directions = random.choice(
            [row for row in empty if row[1] == shallowest])
    elif occupied and (not empty
                     or random.random() < OCCUPIED_CONTEXT_PROBABILITY):
        context, _site_band, directions = random.choice(occupied)
    else:
        context, _site_band, directions = random.choice(empty or occupied)
    ctx_l, ctx_r, ctx_d, self_in = context
    occurs = sorted(bands[(ctx_l, ctx_r, ctx_d, self_in)])
    if random.random() < DEPTH_SPECIFIC_PROBABILITY:
        depth = (occurs[-1] if random.random() < TERMINATOR_BIAS
                 else random.choice(occurs))
    else:
        depth = DEPTH_ANY
    for _ in range(6):
        self_out = _random_output(
            families, directions,
            min_inputs=(
                2 if self_in == OUT_STATE
                or (prefer_growth and self_in == EMPTY_STATE) else 0),
            allow_erase=(self_in != OUT_STATE))
        if self_out is None:
            return None
        if self_out != self_in:
            return ContextGene(gene_id, ctx_l, ctx_r, ctx_d, self_in,
                               self_out, gene_id, depth)
    return None


def _add_blocker(genome, families, n_inputs):
    """Add a gene that STOPS a limb, by disagreeing with the rule extending it.

    A terminator that retypes the tip cannot work: development is synchronous,
    so the extending rule fires at the cell beyond the tip in the very same
    iteration, and the terminator is permanently one step behind.

    A blocker acts on the growth site instead. It carries the same context as a
    rule that builds, pinned to the outermost band that context reaches, but
    proposes a DIFFERENT state - so the conflict rule leaves that cell empty,
    every iteration, at no telomere cost. That is a gene-level brake on growth
    built from machinery already present.
    """
    seeds = _seeds(genome)
    branches = branch_map(genome)
    candidates = []
    for (index, half), members in branches.items():
        label = branch_label(index, half)
        bands = {}
        for context, band in observed_context_bands(genome, seeds, label):
            bands.setdefault(context, []).append(band)
        for gene in members:
            # Only a rule that BUILDS and applies at every depth can be blocked
            # this way; a banded one may simply never share a cell with us.
            if (gene.self_out == EMPTY_STATE or gene.depth != DEPTH_ANY
                    or gene.context not in bands):
                continue
            candidates.append((index, half, gene, max(bands[gene.context])))
    if not candidates:
        return False
    index, half, gene, band = random.choice(candidates)
    chromosome = genome.chromosomes[index]
    if len(chromosome.genes) >= MAX_GENES or len(
            placement_genes(genome)) >= MAX_PLACEMENTS:
        return False
    enabled = [state for state in enabled_component_ids(families)
               if state != int(gene.self_out) and state != int(gene.self_in)]
    if not enabled:
        return False
    gene_id = int(genome.next_gene_id)
    genome.next_gene_id = gene_id + 1
    ctx_l, ctx_r, ctx_d, self_in = gene.context
    _extend_arm(chromosome, ContextGene(
        gene_id, ctx_l, ctx_r, ctx_d, self_in, random.choice(enabled),
        gene_id, band), top=(half == 0))
    _repair_genome(genome, families, n_inputs)
    return True


def _sample_component(viable, families=None):
    """Family-first softmax draw over equally-routable catalogue parts.

    ``viable`` rows are the candidate tuples built in `_connect_terminal_step`,
    already filtered to one routability class, ending in the component id.

    Two properties matter and neither is optional:

    * **Family-first.** GATED_OSCILLATOR ships 36 entries and C_ELEMENT ships
      3. Drawing uniformly over *states* would hand the oscillator family a
      12x prior for no physical reason. Pick the family first, then a part
      inside it - the same correction `catalogue.random_component_id` already
      makes for mutation, which the constructive path never applied.
    * **Preference is a bias, not a veto.** The remaining keys (pin count,
      output-arity fit) still favour the tidy part, but through a softmax
      weight rather than a lexicographic gate, so the untidy part that happens
      to hold state stays reachable.

    ``CONSTRUCTION_TEMPERATURE == 0`` restores the historical argmin exactly.
    """
    if not viable:
        raise ValueError("_sample_component needs at least one candidate")
    by_family = {}
    for row in viable:
        state = row[-1]
        # Lower is better, matching the original lexicographic ordering.
        cost = float(row[6] + row[7])
        by_family.setdefault(BY_ID[state].family, []).append((cost, state))
    temperature = float(CONSTRUCTION_TEMPERATURE)
    if temperature <= 0:
        best = min(row[6:-1] + (row[-1],) for row in viable)
        return int(best[-1])

    def _draw(pairs):
        floor = min(cost for cost, _state in pairs)
        weights = [math.exp(-(cost - floor) / temperature) for cost, _s in pairs]
        total = sum(weights)
        if total <= 0:
            return random.choice([state for _c, state in pairs])
        cut = random.random() * total
        for (_cost, state), weight in zip(pairs, weights):
            cut -= weight
            if cut <= 0:
                return state
        return pairs[-1][1]

    family_pairs = [
        (min(cost for cost, _s in parts), family)
        for family, parts in by_family.items()]
    family = _draw([(cost, name) for cost, name in family_pairs])
    return int(_draw(by_family[family]))


def _arm_has_output_gene(members, exclude=None):
    return any(gene.spawns_output() for gene in members if gene is not exclude)


def _repair_genome(genome, families=None, n_inputs=1):
    """Keep output ownership, one root per arm, and arm controls coherent.

    PAD is an ordinary terminal cue and may occur in several rules. OUT is the
    spawn privilege, so an assigned arm may have at most one self=OUT rule and
    an unassigned arm remains dormant.
    """
    enabled = enabled_component_ids(families) if families else None
    assigned = assigned_branch_labels(genome)
    # One active developmental module per output. Spare arms remain hereditary
    # capacity, but are not allowed to create unrooted phenotype fragments.
    for index, chromosome in enumerate(genome.chromosomes):
        top, bottom = branch_arms(chromosome)
        if branch_label(index, 0) not in assigned:
            top = ()
        if branch_label(index, 1) not in assigned:
            bottom = ()
        chromosome.genes = list(reversed(top)) + list(bottom)
        chromosome.split = len(top)
    for chromosome in genome.chromosomes:
        chromosome.split = max(
            0, min(int(chromosome.split), len(chromosome_rules(chromosome))))
    # Drop rules an arm already holds a copy of. Within ONE arm an exact
    # duplicate can never do anything the original does not - it only bloats the
    # genome. Across arms it is left alone: territorial matching means the same
    # rule on another limb builds somewhere else, which is a real difference and
    # is how a repeated motif gets made.
    # An occupied arm holds exactly one control gene: it is what says how far
    # that branch reaches and how long it lives.
    for (index, half), members in arm_map(genome).items():
        controls = [g for g in members if isinstance(g, ControlGene)]
        chromosome = genome.chromosomes[index]
        if not members:
            continue
        if not controls:
            gene_id = int(genome.next_gene_id)
            genome.next_gene_id = gene_id + 1
            _attach(chromosome, new_control_gene(gene_id), top=(half == 0))
        elif len(controls) > 1:
            extra = {id(g) for g in controls[1:]}
            before = branch_cut(chromosome)
            kept, dropped = [], 0
            for position, gene in enumerate(chromosome.genes):
                if id(gene) in extra:
                    dropped += position < before
                    continue
                kept.append(gene)
            chromosome.genes = kept
            chromosome.split = max(0, before - dropped)
    for (index, half), members in branch_map(genome).items():
        seen, doomed = set(), set()
        for gene in members:
            key = (gene.context, int(gene.self_out), int(gene.depth))
            if key in seen:
                doomed.add(id(gene))
            else:
                seen.add(key)
        if doomed:
            chromosome = genome.chromosomes[index]
            before = branch_cut(chromosome)
            kept, removed_before_cut = [], 0
            for position, gene in enumerate(chromosome.genes):
                if id(gene) in doomed:
                    removed_before_cut += position < before
                    continue
                kept.append(gene)
            chromosome.genes = kept
            chromosome.split = max(0, before - removed_before_cut)
    for members in branch_map(genome).values():
        seen_output = False
        for gene in members:
            if not gene.spawns_output():
                continue
            if not seen_output:
                seen_output = True
                continue
            replacement = (EMPTY_STATE if enabled is None
                           else random.choice(enabled))
            gene.self_in = replacement
    return relabel_branches(genome)


def _attach(chromosome, gene, top):
    """Put one gene on the outer end of an arm, centromere held in place."""
    if top:
        chromosome.genes.insert(0, gene)
        chromosome.split = int(chromosome.split) + 1
    else:
        chromosome.genes.append(gene)


def _extend_arm(chromosome, gene, top):
    """Attach one gene to the outer end of an arm."""
    if top:
        chromosome.genes.insert(0, gene)
        chromosome.split = int(chromosome.split) + 1
    else:
        chromosome.genes.append(gene)


def _reverse_cone(root, edges):
    reverse = {}
    for source, destination in edges:
        reverse.setdefault(destination, set()).add(source)
    reached, pending = {root}, [root]
    while pending:
        destination = pending.pop()
        for source in reverse.get(destination, ()):
            if source not in reached:
                reached.add(source)
                pending.append(source)
    return reached


def _shared_open_buds(genome, label):
    """Open source sites feeding two or more cells of one output arm."""
    trace = develop_constructive(genome, _seeds(genome))
    pads = set(_seeds(genome))
    consumers = {}
    for destination, owner in trace.owners.items():
        if owner != int(label) or destination not in trace.grid:
            continue
        for direction in BY_ID[trace.grid[destination]].inputs:
            source = source_for_input(destination, direction)
            if source not in trace.grid and source not in pads:
                consumers.setdefault(source, set()).add(destination)
    return sum(len(destinations) >= 2 for destinations in consumers.values())


def _connect_terminal_step(genome, families, n_inputs, label=None,
                           required_inputs=None):
    """Grow one local bud toward an under-connected source pad.

    This is terminal tropism, not circuit synthesis: it reads only the current
    physical body and terminal coordinates, never input values or desired
    outputs. The resulting component is still encoded as an ordinary context
    rule and must regrow synchronously with the rest of the organism.
    """
    if len(placement_genes(genome)) >= MAX_PLACEMENTS:
        return False
    seeds = _seeds(genome)
    pads = set(seeds)
    if required_inputs is None:
        goal_pads = pads
    else:
        goal_pads = {
            seeds[int(index)] for index in required_inputs
            if 0 <= int(index) < len(seeds)}
    if not goal_pads:
        return False
    unwanted_pads = pads - goal_pads
    trace = develop_constructive(genome, seeds)
    roots = output_branch_sites(genome)
    edges = effective_wiring_edges(trace.grid, pads)
    options = []
    labels = [int(label)] if label is not None else sorted(roots)
    for branch_id in labels:
        root = roots.get(branch_id)
        if root not in trace.grid:
            continue
        cone = _reverse_cone(root, edges)
        toward_root = {}
        for source, destination in edges:
            if source in cone and destination in cone:
                toward_root.setdefault(source, set()).add(destination)
        pad_edges = {
            pad: sum(source == pad and destination in cone
                     for source, destination in edges)
            for pad in pads}
        buds = {}
        for destination, owner in trace.owners.items():
            if owner != branch_id or destination not in trace.grid:
                continue
            for direction in BY_ID[trace.grid[destination]].inputs:
                source = source_for_input(destination, direction)
                if source not in trace.grid and source not in pads:
                    # Which first-level input limb of the output root owns this
                    # bud? Grow the least terminal-rich limb first, so a binary
                    # output does not degenerate into one deep computation arm
                    # and one direct-input shortcut.
                    limb = destination
                    seen = set()
                    while (limb != root and limb not in seen
                           and root not in toward_root.get(limb, ())):
                        seen.add(limb)
                        destinations = sorted(toward_root.get(limb, ()))
                        if not destinations:
                            break
                        limb = destinations[0]
                    limb_cone = (
                        set() if limb == root else _reverse_cone(limb, edges))
                    limb_edges = sum(
                        edge_source in pads and edge_destination in limb_cone
                        for edge_source, edge_destination in edges)
                    buds.setdefault(source, []).append((
                        limb_edges, limb_cone.intersection(pads), destination))
        for pad in goal_pads:
            for bud, consumers in buds.items():
                limb_edges = min(row[0] for row in consumers)
                limb_pads = set().union(*(row[1] for row in consumers))
                options.append((
                    int(pad_edges[pad] > 0), int(pad in limb_pads),
                    limb_edges, -len(consumers), honeycomb_distance(bud, pad),
                    branch_id, pad, bud))
    if not options:
        return False
    # Keep the route on the strict local terminal gradient. A wider near-best
    # fringe was measured here and made fresh genome construction materially
    # slower without increasing terminal coverage.
    ranked_shapes = sorted(set(row[:5] for row in options))
    best_shape = ranked_shapes[0]
    (_already_global, _already_limb, _limb_edges, _fanout, _distance,
     branch_id, pad, bud) = random.choice(
        [row for row in options if row[:5] == best_shape])
    depth = _reach(
        bud, branch_id, trace.grid, trace.owners, trace.branch_depths)
    if depth is None:
        return False
    required = set(required_output_directions(
        bud, branch_id, depth, trace.grid, trace.owners,
        trace.branch_depths))
    candidates = []
    for state in enabled_component_ids(families):
        entry = BY_ID[state]
        covered = required.intersection(entry.outputs)
        if not covered or not entry.inputs:
            continue
        input_sources = [
            (direction, source_for_input(bud, direction))
            for direction in entry.inputs]
        live_inputs = sum(
            source in pads
            or (source in trace.grid
                and direction in BY_ID[trace.grid[source]].outputs)
            for direction, source in input_sources)
        unwanted_inputs = sum(
            source in unwanted_pads for _direction, source in input_sources)
        relevant_live_inputs = sum(
            source in goal_pads
            or (source in trace.grid
                and direction in BY_ID[trace.grid[source]].outputs)
            for direction, source in input_sources)
        next_distance = min(
            honeycomb_distance(source, pad)
            for _direction, source in input_sources)
        candidates.append((
            int(not required.issubset(entry.outputs)),
            -len(covered),
            unwanted_inputs,
            -relevant_live_inputs,
            -live_inputs,
            next_distance,
            # PREFERENCE keys. Formerly strict, and that was the single most
            # damaging line in the substrate: `int(entry.family != "DELAY")`
            # gave DELAY absolute lexicographic priority over every other
            # family, and `min()` below made the choice a deterministic argmin
            # rather than a draw. Measured on 400 grown bodies: DELAY 64.6%,
            # LOGIC 17.5%, C_ELEMENT 17.9%, and NORMALIZER / HOLD / TOGGLE /
            # GATED_OSCILLATOR 0.0% - 81 of 117 catalogue entries were enabled
            # and never once expressed. With no state-holding part and no
            # feedback edge, every grown circuit was a combinational delay
            # chain and every stateful target was UNREACHABLE, not merely hard.
            len(entry.inputs),
            abs(len(entry.outputs) - len(required)),
            state))
    if not candidates:
        return False
    # Routability is still decided strictly: anything that cannot be wired at
    # this bud loses outright. Among the parts that CAN be wired, choose by
    # softmax sampling over the preference keys instead of argmin, so a
    # marginally-less-convenient part stays reachable at initialization. This
    # is what puts HOLD / TOGGLE / GATED_OSCILLATOR / NORMALIZER back into the
    # developmental repertoire; without them the search cannot express memory.
    routable = min(row[:6] for row in candidates)
    viable = [row for row in candidates if row[:6] == routable]
    state = _sample_component(viable, families)

    around = hex_dirs(*bud)
    output_sites = set(roots.values()) - pads
    context = (
        _state_of(around['L'], trace.grid, pads, output_sites),
        _state_of(around['R'], trace.grid, pads, output_sites),
        _state_of(around['D'], trace.grid, pads, output_sites),
        _state_of(bud, trace.grid, pads, output_sites),
    )
    chromosome_index, half = divmod(branch_id - 1, 2)
    if chromosome_index >= len(genome.chromosomes):
        return False
    chromosome = genome.chromosomes[chromosome_index]
    if len(chromosome.genes) >= MAX_GENES:
        return False
    gene_id = int(genome.next_gene_id)
    gene = ContextGene(
        gene_id, *context, state, gene_id,
        min(int(depth), DEPTH_BANDS - 1))
    genome.next_gene_id = gene_id + 1
    _extend_arm(chromosome, gene, top=(half == 0))
    _repair_genome(genome, families, n_inputs)
    # Do not develop the proposed child to decide whether the gene survives.
    # The context and bud came from the current body; evaluation and selection
    # own the consequences of adding it.
    return True


def _add_gene(genome, families, n_inputs):
    """Grow one arm by a gene, at the top or the bottom of a chromosome."""
    assigned = assigned_branch_labels(genome)
    available = [
        (index, half)
        for index, chromosome in enumerate(genome.chromosomes)
        for half in (0, 1)
        if branch_label(index, half) in assigned
        and len(chromosome.genes) < MAX_GENES]
    if not available or len(placement_genes(genome)) >= MAX_PLACEMENTS:
        return False
    index, half = random.choice(available)
    chromosome = genome.chromosomes[index]
    top = half == 0
    arm = branch_growth_order(chromosome)[0 if top else 1]
    gene_id = int(genome.next_gene_id)
    gene = _random_gene(genome, gene_id, families, _seeds(genome),
                         allow_output=not _arm_has_output_gene(arm),
                         label=branch_label(index, half), prefer_growth=True)
    if gene is None:
        return False
    genome.next_gene_id = gene_id + 1
    _extend_arm(chromosome, gene, top)
    _repair_genome(genome, families, n_inputs)
    return True


def _adopt_branch(genome, members):
    """Renumbered copies of an arm's genes, control gene included."""
    copies = []
    for gene in members:
        new_id = int(genome.next_gene_id)
        genome.next_gene_id = new_id + 1
        if isinstance(gene, ControlGene):
            copies.append(ControlGene(
                new_id, gene.tolerance, gene.telomere, new_id))
        else:
            copies.append(ContextGene(
                new_id, gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in,
                gene.self_out, new_id, gene.depth))
    return copies


def _function_compatible_arms(left, right):
    """Whether two arms share one morphology and differ only in component use."""
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if type(a) is not type(b):
            return False
        if isinstance(a, ControlGene):
            if (int(a.tolerance), int(a.telomere)) != (
                    int(b.tolerance), int(b.telomere)):
                return False
            continue
        if (a.context != b.context or int(a.depth) != int(b.depth)
                or a.spawns_output() != b.spawns_output()):
            return False
        entry_a, entry_b = BY_ID[int(a.self_out)], BY_ID[int(b.self_out)]
        if (set(entry_a.inputs) != set(entry_b.inputs)
                or entry_a.outputs != entry_b.outputs):
            return False
    return True


def _recombine_functions(left, right):
    """Uniform gate crossover on an already co-adapted developmental arm."""
    mixed = []
    used_left = used_right = False
    for a, b in zip(left, right):
        source = a
        if isinstance(a, ContextGene) and int(a.self_out) != int(b.self_out):
            if random.random() < 0.5:
                source = b
                used_right = True
            else:
                used_left = True
        copied = copy.copy(source)
        mixed.append(copied)
    # If a draw accidentally chose one parent at every differing locus, force
    # one allele from the missing parent so this operator is real recombination.
    differing = [index for index, (a, b) in enumerate(zip(left, right))
                 if isinstance(a, ContextGene)
                 and int(a.self_out) != int(b.self_out)]
    if len(differing) >= 2 and not used_right:
        index = random.choice(differing)
        mixed[index] = copy.copy(right[index])
    elif len(differing) >= 2 and not used_left:
        index = random.choice(differing)
        mixed[index] = copy.copy(left[index])
    return mixed


def _tweak_gene(gene, genome, families, n_inputs):
    """Move one field of a rule: where it applies, what it reacts to, or what
    it makes."""
    current = int(gene.self_out)
    route_preserving = tuple(
        state for state in behavior_component_ids(current)
        if BY_ID[state].family in normalise_families(families))
    if route_preserving and random.random() < 0.70:
        # Functional tuning is a first-class point mutation. It needs no body
        # reconstruction because physical pins and developmental context stay
        # fixed, making it both cheap and non-destructive.
        gene.self_out = random.choice(route_preserving)
        return True
    seeds = _seeds(genome)
    if random.random() < 0.18:
        # Slide the rule along its branch, or free it from position entirely.
        choices = [DEPTH_ANY] + list(range(DEPTH_BANDS))
        choices = [value for value in choices if value != int(gene.depth)]
        gene.depth = random.choice(choices)
        return True
    if random.random() < 0.55:
        # Retarget the context, preferring a neighbourhood the body presents so
        # the rule stays expressible.
        contexts = {
            context for context, _band, directions
            in _observed_context_occurrences(
                genome, seeds, int(gene.branch_id) or None)
            if ((context[3] == OUT_STATE) == gene.spawns_output())
            and OUT_STATE not in context[:3]
            and (int(gene.self_out) == EMPTY_STATE or not directions
                 or set(directions).intersection(
                     BY_ID[int(gene.self_out)].outputs))}
        if not contexts:
            return False
        chosen = random.choice(sorted(contexts))
        if chosen == gene.context:
            return False
        gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in = chosen
        return True
    occurrences = [
        (band, directions)
        for context, band, directions in _observed_context_occurrences(
            genome, seeds, int(gene.branch_id) or None)
        if context == gene.context and gene.applies_at(band)]
    directions = random.choice(occurrences)[1] if occurrences else ()
    # Most component edits change only the fixed function while retaining the
    # same physical pins.  Once a reverse-grown cone has found useful geometry,
    # an AND -> XOR substitution should test a different computation without
    # moving both upstream buds and amputating the branch that feeds them.
    # Route-changing edits remain available below for genuine morphogenesis.
    if route_preserving and random.random() < 0.80:
        gene.self_out = random.choice(route_preserving)
        return True
    for _ in range(6):
        value = _random_output(
            families, directions,
            min_inputs=(2 if gene.spawns_output() else 0),
            allow_erase=not gene.spawns_output())
        if value is None:
            return False
        if value != current and value != int(gene.self_in):
            gene.self_out = value
            return True
    return False


def mutate_branched_once(genome, families, n_inputs, max_telomere=None):
    """One state-changing edit, chosen from the shared weighted menu."""
    _repair_genome(genome, families, n_inputs)
    genes = placement_genes(genome)
    branches = branch_map(genome)
    populated = sorted(key for key, members in branches.items() if members)
    options, weights = [], []
    for name, weight in zip(BRANCHED_MUT_OPS, BRANCHED_MUT_WEIGHTS):
        if name == "tweak" and not genes:
            continue
        if name == "add_gene" and len(genes) >= MAX_PLACEMENTS:
            continue
        if name == "block" and not genes:
            continue
        if name == "del_rule" and len(genes) <= 1:
            continue
        if name == "del_branch" and not any(
                0 < len(branches[key]) < len(genes) for key in populated):
            continue
        if name == "inputs" and not getattr(genome, "input_chromosome", None):
            continue
        if name == "outputs" and not getattr(genome, "output_chromosome", None):
            continue
        options.append(name)
        weights.append(weight)
    if not options:
        return False
    op = random.choices(options, weights=weights)[0]

    if op == "tweak":
        changed = _tweak_gene(
            random.choice(genes), genome, families, n_inputs)
        _repair_genome(genome, families, n_inputs)
        return changed
    if op == "add_gene":
        return _add_gene(genome, families, n_inputs)
    if op == "connect":
        return _connect_terminal_step(genome, families, n_inputs)
    if op == "block":
        return _add_blocker(genome, families, n_inputs)
    if op == "del_rule":
        victim = random.choice(genes)
        for chromosome in genome.chromosomes:
            for position, gene in enumerate(chromosome.genes):
                if gene is victim:
                    chromosome.genes.pop(position)
                    if position < branch_cut(chromosome):
                        chromosome.split = max(0, int(chromosome.split) - 1)
                    _repair_genome(genome, families, n_inputs)
                    return True
        return False
    if op == "del_branch":
        index, half = random.choice([
            key for key in populated if len(branches[key]) < len(genes)])
        doomed = {id(gene) for gene in branches[(index, half)]}
        chromosome = genome.chromosomes[index]
        if half == 0:
            chromosome.split = 0
        chromosome.genes = [gene for gene in chromosome.genes
                            if id(gene) not in doomed]
        _repair_genome(genome, families, n_inputs)
        return True
    if op == "inputs":
        return _mutate_inputs(genome)
    if op == "outputs":
        return _mutate_outputs(genome)
    # control: an arm's REACH - how far its rules may sit from a neighbourhood
    # and still apply - or its LIFESPAN. Both are genetic material on the arm's
    # own control gene, so they mutate like anything else and cross with it.
    controls = [gene for chromosome in genome.chromosomes
                for gene in chromosome.genes
                if isinstance(gene, ControlGene)]
    if not controls:
        return False
    gene = random.choice(controls)
    if random.random() < 0.5:
        base = max(0, min(int(gene.tolerance), MAX_TOLERANCE))
        values = [base + delta for delta in (-8, -4, -1, 1, 4, 8)
                  if 0 <= base + delta <= MAX_TOLERANCE]
        if not values:
            return False
        gene.tolerance = random.choice(values)
        return True
    ceiling = arm_telomere_ceiling(max_telomere)
    base = max(0, min(int(gene.telomere), ceiling))
    values = [base + delta for delta in (-4, -2, -1, 1, 2, 4)
              if 0 <= base + delta <= ceiling]
    if not values:
        return False
    gene.telomere = random.choice(values)
    return True


def mutate_branched(genome, mean_mutations, families, n_inputs,
                    max_telomere=None, focus_families=()):
    """Poisson-many edits, with at least one guaranteed to land."""
    lam = 4.0 if mean_mutations is None else max(0.0, float(mean_mutations))
    # A genome from a run with a looser ceiling must be brought under this run's
    # control, not left above it forever.
    ceiling = arm_telomere_ceiling(max_telomere)
    for chromosome in genome.chromosomes:
        for gene in chromosome.genes:
            if isinstance(gene, ControlGene):
                gene.telomere = max(0, min(int(gene.telomere), ceiling))
    enabled = normalise_families(families)
    focused = frozenset(
        family for family in focus_families if family in enabled)
    for _ in range(max(1, _poisson(lam))):
        # Contract classes may bias the mutation supply without banning any
        # user-enabled component.  Static truth-table targets spend most edits
        # on gates and transport; one edit in five still samples the full bank,
        # so temporal hardware remains reachable and this is not a truth-table
        # hint.
        palette = focused if focused and random.random() < 0.80 else enabled
        if not mutate_branched_once(
                genome, palette, n_inputs, max_telomere):
            break
    return genome


def randomize_branch_functions(genome, probability=0.70, branch_id=None):
    """Resample basic gate components on one arm without changing its body."""
    candidates = []
    for (index, half), members in branch_map(genome).items():
        if (branch_id is not None
                and branch_label(index, half) != int(branch_id)):
            continue
        if any(behavior_component_ids(gene.self_out) for gene in members):
            candidates.append(members)
    if not candidates:
        return False
    genes = [gene for gene in random.choice(candidates)
             if behavior_component_ids(gene.self_out)]
    changed = False
    for gene in genes:
        if random.random() <= float(probability):
            options = behavior_component_ids(gene.self_out)
            gene.self_out = random.choice(options)
            changed = True
    if not changed and genes:
        gene = random.choice(genes)
        gene.self_out = random.choice(behavior_component_ids(gene.self_out))
        changed = True
    return changed


def mutate_branch_function(genome, branch_id=None):
    """Change exactly one fixed gate while preserving every physical pin."""
    genes = [
        gene
        for (index, half), members in branch_map(genome).items()
        if (branch_id is None
            or branch_label(index, half) == int(branch_id))
        for gene in members
        if behavior_component_ids(gene.self_out)
    ]
    if not genes:
        return False
    gene = random.choice(genes)
    gene.self_out = random.choice(behavior_component_ids(gene.self_out))
    return True


def randomize_branch_behavior(genome, branch_id, n_inputs, *, limit=100_000,
                              preferred_signature=None, input_patterns=None):
    """Sample an attainable arm behavior, not a biased gate bit string.

    Gate assignments are a badly biased coordinate system: exponentially many
    assignments collapse onto constant or one-input functions.  This operator
    enumerates (or boundedly samples) the signatures the CURRENT physical arm
    can express and maps one signature back to ordinary fixed component alleles.
    With ``preferred_signature`` it chooses the attainable table with minimum
    Hamming distance to that contract role; without one it cycles the repertoire.
    ``input_patterns`` supplies the contract's actual row order.  This matters
    for tables such as the 2-bit adder, whose tuple is ordered A0,A1,B0,B1 while
    its rows are enumerated with A0 as the least-significant counter bit.
    Pins, cells, contexts, pads, output positions, and every other arm remain
    unchanged.
    """
    label = int(branch_id)
    count = max(1, int(n_inputs))
    if input_patterns is None:
        patterns = tuple(
            tuple((truth_row >> (count - index - 1)) & 1
                  for index in range(count))
            for truth_row in range(1 << count))
    else:
        patterns = tuple(
            tuple(int(bit) & 1 for bit in pattern)
            for pattern in input_patterns)
        if not patterns or any(len(pattern) != count for pattern in patterns):
            return False
    rows = len(patterns)
    sample_limit = max(1, int(limit))
    if count >= 4:
        sample_limit = min(sample_limit, 20_000)
    trace = develop_constructive(genome, _seeds(genome))
    root = output_branch_sites(genome).get(label)
    if root not in trace.grid:
        return False
    pads = tuple(_seeds(genome))
    pad_set = set(pads)
    genes = {int(gene.gene_id): gene for gene in placement_genes(genome)}
    variable_ids = sorted({
        int(trace.builders[cell])
        for cell, owner in trace.owners.items()
        if owner == label and cell in trace.grid
        and trace.builders.get(cell) in genes
        and behavior_component_ids(genes[trace.builders[cell]].self_out)
    })
    if not variable_ids:
        return False
    options = {
        gene_id: tuple(sorted({
            int(genes[gene_id].self_out),
            *behavior_component_ids(genes[gene_id].self_out)}))
        for gene_id in variable_ids}

    def route_token(cell, state):
        builder = trace.builders.get(cell)
        if trace.owners.get(cell) == label and builder in options:
            entry = BY_ID[state]
            return ("variable", int(builder),
                    tuple(sorted(entry.inputs)), tuple(entry.outputs))
        return ("fixed", int(state))

    route_key = (
        label, count, sample_limit,
        None if preferred_signature is None else int(preferred_signature),
        patterns, pads, tuple(root), tuple(sorted(
            (cell, route_token(cell, state))
            for cell, state in trace.grid.items())))
    repertoire = _BEHAVIOR_REPERTOIRES.get(route_key)
    if repertoire is None:
        # Route-preserving alternatives have identical dependency edges, so one
        # topological order serves every assignment.
        dependents = {cell: [] for cell in trace.grid if cell not in pad_set}
        indegree = {cell: 0 for cell in dependents}
        for cell in dependents:
            entry = BY_ID[trace.grid[cell]]
            for direction in entry.inputs:
                source = source_for_input(cell, direction)
                driven = (
                    source in pad_set
                    or (source in trace.grid
                        and direction in BY_ID[trace.grid[source]].outputs))
                if driven and source in indegree:
                    dependents[source].append(cell)
                    indegree[cell] += 1
        pending = sorted(cell for cell, degree in indegree.items()
                         if degree == 0)
        order = []
        while pending:
            cell = pending.pop()
            order.append(cell)
            for destination in dependents[cell]:
                indegree[destination] -= 1
                if indegree[destination] == 0:
                    pending.append(destination)
        if len(order) != len(indegree):
            return False

        banks = [options[gene_id] for gene_id in variable_ids]
        combination_count = math.prod(map(len, banks))
        if combination_count <= sample_limit:
            sampled = list(itertools.product(*banks))
        else:
            sampled = [tuple(int(genes[gene_id].self_out)
                             for gene_id in variable_ids)]
            sampled.extend(
                tuple(random.choice(bank) for bank in banks)
                for _ in range(sample_limit - 1))
        sample_count = len(sampled)
        assignment_mask = (1 << sample_count) - 1
        byte_count = (sample_count + 7) // 8
        allele_bytes = {
            gene_id: {state: bytearray(byte_count)
                      for state in options[gene_id]}
            for gene_id in variable_ids}
        for assignment_index, values in enumerate(sampled):
            byte_index, bit_index = divmod(assignment_index, 8)
            for gene_id, state in zip(variable_ids, values):
                allele_bytes[gene_id][state][byte_index] |= 1 << bit_index
        allele_masks = {
            gene_id: {
                state: int.from_bytes(bits, "little")
                for state, bits in states.items()}
            for gene_id, states in allele_bytes.items()}

        # Every allele is route-preserving, so physical sources can be resolved
        # once per (cell, state) instead of once per truth row, assignment, and
        # beam neighbour.
        state_sources = {}
        for cell in order:
            builder = trace.builders.get(cell)
            states = (options[builder]
                      if builder in options else (trace.grid[cell],))
            for state in states:
                resolved = []
                for direction in BY_ID[state].inputs:
                    source = source_for_input(cell, direction)
                    driven = (
                        source in pad_set
                        or (source in trace.grid
                            and direction in BY_ID[trace.grid[source]].outputs))
                    resolved.append(source if driven else None)
                state_sources[(cell, state)] = tuple(resolved)

        root_rows = []
        for truth_row, pattern in enumerate(patterns):
            tables = {
                pad: (assignment_mask
                      if pattern[index] else 0)
                for index, pad in enumerate(pads)}
            for cell in order:
                builder = trace.builders.get(cell)
                states = (options[builder]
                          if builder in options else (trace.grid[cell],))
                result = 0
                for state in states:
                    entry = BY_ID[state]
                    args = [0 if source is None else tables.get(source, 0)
                            for source in state_sources[(cell, state)]]
                    if entry.behavior == "AND":
                        value = args[0] & args[1]
                    elif entry.behavior == "OR":
                        value = args[0] | args[1]
                    elif entry.behavior == "XOR":
                        value = args[0] ^ args[1]
                    elif entry.behavior == "VETO":
                        value = args[0] & (~args[1] & assignment_mask)
                    else:
                        value = args[0] if args else 0
                    select = (allele_masks[builder][state]
                              if builder in allele_masks
                              else assignment_mask)
                    result |= value & select
                tables[cell] = result & assignment_mask
            root_rows.append(tables.get(root, 0))

        repertoire_by_signature = {}

        # A requested four-input function is generally a needle in raw gate
        # assignment space: most random assignments collapse to constants or
        # one-input behavior. Starting from the closest packed samples, search
        # a small beam of route-preserving allele neighbours. This is still an
        # assignment on the EXISTING physical body; it cannot add a component,
        # change a pin, or make an inexpressible morphology capable.
        beam_choice = None
        if preferred_signature is not None and count >= 4:
            preferred = int(preferred_signature) & ((1 << rows) - 1)
            sampled_signatures = [
                sum(((root_rows[truth_row] >> assignment_index) & 1)
                    << truth_row for truth_row in range(rows))
                for assignment_index in range(sample_count)]
            seed_indices = heapq.nsmallest(
                min(16, sample_count), range(sample_count),
                key=lambda index: (
                    (sampled_signatures[index] ^ preferred).bit_count(),
                    index))

            pattern_tables = {
                pad: sum((int(pattern[index]) & 1) << truth_row
                         for truth_row, pattern in enumerate(patterns))
                for index, pad in enumerate(pads)}

            def assignment_signature(assignment):
                allele = dict(zip(variable_ids, assignment))
                tables = dict(pattern_tables)
                for cell in order:
                    builder = trace.builders.get(cell)
                    state = allele.get(builder, trace.grid[cell])
                    entry = BY_ID[state]
                    args = [
                        0 if source is None else tables.get(source, 0)
                        for source in state_sources[(cell, state)]]
                    if entry.behavior == "AND":
                        value = args[0] & args[1]
                    elif entry.behavior == "OR":
                        value = args[0] | args[1]
                    elif entry.behavior == "XOR":
                        value = args[0] ^ args[1]
                    elif entry.behavior == "VETO":
                        value = args[0] & (~args[1] & ((1 << rows) - 1))
                    else:
                        value = args[0] if args else 0
                    tables[cell] = value
                return tables.get(root, 0) & ((1 << rows) - 1)

            beam = {
                sampled[index]: sampled_signatures[index]
                for index in seed_indices}
            seen_assignments = set(beam)
            for _round in range(10):
                best_assignment, best_signature = min(
                    beam.items(), key=lambda row:
                    ((row[1] ^ preferred).bit_count(), row[0]))
                if best_signature == preferred:
                    beam_choice = (best_signature, best_assignment)
                    break
                expanded = dict(beam)
                for assignment in tuple(beam):
                    for position, bank in enumerate(banks):
                        for state in bank:
                            if state == assignment[position]:
                                continue
                            neighbour = (
                                assignment[:position] + (int(state),)
                                + assignment[position + 1:])
                            if neighbour in seen_assignments:
                                continue
                            seen_assignments.add(neighbour)
                            expanded[neighbour] = assignment_signature(neighbour)
                beam = dict(heapq.nsmallest(
                    min(16, len(expanded)), expanded.items(),
                    key=lambda row: (
                        (row[1] ^ preferred).bit_count(), row[0])))
            if beam_choice is None and beam:
                assignment, signature = min(
                    beam.items(), key=lambda row:
                    ((row[1] ^ preferred).bit_count(), row[0]))
                beam_choice = (signature, assignment)

        def matching_assignments(signature):
            matching = assignment_mask
            for truth_row, values in enumerate(root_rows):
                matching &= (values if (signature >> truth_row) & 1
                             else assignment_mask ^ values)
                if not matching:
                    break
            return matching

        if rows <= 8:
            candidate_signatures = range(1 << rows)
        else:
            # Four logical inputs already imply 65,536 possible truth tables.
            # Extract a bounded, deterministic assignment sample for diversity;
            # the requested contract signature below is still checked against
            # ALL packed assignments in one bitset operation per truth row.
            stride = max(1, sample_count // min(sample_count, 5000))
            candidate_signatures = {
                sum(((root_rows[truth_row] >> assignment_index) & 1)
                    << truth_row for truth_row in range(rows))
                for assignment_index in range(0, sample_count, stride)
            }
        for signature in candidate_signatures:
            matching = matching_assignments(signature)
            if not matching:
                continue
            assignment_index = (matching & -matching).bit_length() - 1
            repertoire_by_signature[signature] = sampled[assignment_index]
        if preferred_signature is not None:
            preferred = int(preferred_signature)
            matching = matching_assignments(preferred)
            if matching:
                assignment_index = (matching & -matching).bit_length() - 1
                repertoire_by_signature[preferred] = sampled[assignment_index]
            if beam_choice is not None:
                signature, assignment = beam_choice
                repertoire_by_signature[int(signature)] = tuple(assignment)
        repertoire = tuple(sorted(repertoire_by_signature.items()))
        if len(_BEHAVIOR_REPERTOIRES) >= _BEHAVIOR_CURSOR_MAX:
            _BEHAVIOR_REPERTOIRES.clear()
        _BEHAVIOR_REPERTOIRES[route_key] = repertoire
    if not repertoire:
        return False
    sampled_capacities = dict(
        getattr(genome, "_sampled_branch_capacities", {}))
    sampled_capacities[label] = len(repertoire)
    genome._sampled_branch_capacities = sampled_capacities

    if len(_BEHAVIOR_CURSORS) >= _BEHAVIOR_CURSOR_MAX:
        _BEHAVIOR_CURSORS.clear()
    start = _BEHAVIOR_CURSORS.setdefault(
        route_key, random.randrange(len(repertoire))) % len(repertoire)
    ordered = repertoire[start:] + repertoire[:start]
    if preferred_signature is not None:
        preferred = int(preferred_signature)
        ordered = tuple(
            sorted(ordered, key=lambda row:
                   (row[0] ^ preferred).bit_count()))
    for offset, (_signature, assignment) in enumerate(ordered):
        changed = False
        for gene_id, state in zip(variable_ids, assignment):
            gene = genes[gene_id]
            if int(gene.self_out) == int(state):
                continue
            gene.self_out = int(state)
            changed = True
        if changed:
            _BEHAVIOR_CURSORS[route_key] = (
                start + offset + 1) % len(repertoire)
            return True
    _BEHAVIOR_CURSORS[route_key] = start
    return False


def regrow_branch(genome, branch_id, families, n_inputs,
                  *, blocks=8, max_telomere=None, required_inputs=None):
    """Replace one role arm with a fresh output-rooted developmental germline.

    The input chromosome, every other output module, and both terminal layouts
    stay fixed.  This is the structural counterpart to function resampling: it
    can cross a dead-morphology valley without erasing a solved sibling output.
    The regrowth sees only physical pads, occupied tissue, and local contexts.
    """
    label = int(branch_id)
    chromosome_index, half = divmod(label - 1, 2)
    if not (0 <= chromosome_index < len(genome.chromosomes)):
        return False
    if label not in assigned_branch_labels(genome):
        return False
    chromosome = genome.chromosomes[chromosome_index]
    cut = branch_cut(chromosome)
    if half == 0:
        chromosome.genes = list(chromosome.genes[cut:])
        chromosome.split = 0
    else:
        chromosome.genes = list(chromosome.genes[:cut])
        chromosome.split = len(chromosome.genes)
    _repair_genome(genome, families, n_inputs)

    seeds = _seeds(genome)
    top = half == 0
    life_ceiling = arm_telomere_ceiling(max_telomere)
    role_arity = (
        int(n_inputs) if required_inputs is None
        else len(tuple(required_inputs)))
    if required_inputs is None:
        scaffold_floor = LOGIC_SCAFFOLD_GENES
        scaffold_ceiling = MAX_LOGIC_SCAFFOLD_GENES
    else:
        # A role known to depend on N pads needs more than N-1 arbitrary gates:
        # functions such as majority/carry reuse upstream signals, so the
        # developmental crown must have time to expose shared buds.  Start
        # checking at the binary-tree minimum, but keep growing until either
        # those fan-out niches exist or the generic two-gates-per-added-input
        # ceiling is reached.  The previous random floor was also the ceiling,
        # which made the shared-bud test vacuous and stranded Carry at 7/8.
        minimum_gates = max(1, role_arity - 1)
        maximum_gates = max(minimum_gates, 2 * role_arity - 2)
        scaffold_floor = minimum_gates
        # Use the same bounded crown window as fresh ontogeny. Four arbitrary
        # gates are the Boolean minimum for some three-input functions but are
        # rarely enough spatial trials to expose the two shared source niches
        # those functions physically require on a degree-three lattice.
        scaffold_ceiling = max(
            maximum_gates,
            min(MAX_LOGIC_SCAFFOLD_GENES, maximum_gates + 4))
    changed = False
    for block_index in range(max(1, int(blocks))):
        arm = branch_growth_order(chromosome)[0 if top else 1]
        needed_fanouts = max(0, role_arity - 1) if role_arity >= 3 else 0
        scaffold_ready = (
            block_index >= scaffold_floor
            and (_shared_open_buds(genome, label) >= needed_fanouts
                 or block_index >= scaffold_ceiling))
        if scaffold_ready and _arm_has_output_gene(arm):
            connected = False
            for _attempt in range(6):
                if _connect_terminal_step(
                        genome, families, n_inputs, label,
                        required_inputs=required_inputs):
                    connected = changed = True
                    break
            if not connected:
                break
            continue
        arm = branch_growth_order(chromosome)[0 if top else 1]
        gene_id = int(genome.next_gene_id)
        gene = _random_gene(
            genome, gene_id, families, seeds,
            allow_output=not _arm_has_output_gene(arm),
            label=label, prefer_growth=True)
        if gene is None:
            break
        genome.next_gene_id = gene_id + 1
        _extend_arm(chromosome, gene, top)
        _repair_genome(genome, families, n_inputs)
        if int(n_inputs) >= 4:
            control = arm_control(chromosome, half)
            if control is not None:
                control.telomere = life_ceiling
        changed = True
    _repair_genome(genome, families, n_inputs)
    return changed


def random_branched_genome(chromosome_count, families, n_inputs,
                           output_roles=("out0",), input_layout=None,
                           blocks=8, max_telomere=None):
    """A fresh genome: one output-rooted developmental arm per target role.

    Each gene is drawn from a neighbourhood the organism presents at the moment
    it is added. The constructor never develops a proposed child to decide
    whether that gene survives; ordinary evaluation and selection do that.
    """
    from .genome import BRANCHED_ENCODING, Genome
    count = max(1, int(chromosome_count))
    roles = tuple(str(role) for role in output_roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("FNV output roles must be present and distinct")
    if len(roles) > 2 * count:
        raise ValueError("FNV needs at least one chromosome arm per output")
    genome = Genome(
        chromosomes=[new_branched_chromosome(max_telomere)
                     for _ in range(count)],
        encoding=BRANCHED_ENCODING,
        next_gene_id=1)
    genome.input_chromosome = new_input_chromosome(genome, n_inputs)
    sync_input_layout(genome)
    if input_layout is not None:
        # An explicit layout still wins, for the callers that pin pads.
        genome.input_layout = tuple(tuple(cell) for cell in input_layout)
    genome.output_chromosome = new_output_chromosome(genome, roles)
    sync_output_layout(genome)
    seeds = _seeds(genome)
    assigned = assigned_branch_labels(genome)
    slots = [
        (chromosome_index, half, branch_label(chromosome_index, half),
         half == 0)
        for chromosome_index in range(len(genome.chromosomes))
        for half in (0, 1)
        if branch_label(chromosome_index, half) in assigned
    ]
    growing = {label for _index, _half, label, _top in slots}
    life_ceiling = arm_telomere_ceiling(max_telomere)
    # Grow all role germlines together. Building one complete output before the
    # next gave the first role every shared/fanout niche and systematically
    # reduced later outputs to read-once trees. The developmental interpreter
    # is already synchronous; initialization now respects that same biology.
    for block_index in range(max(1, int(blocks))):
        round_slots = list(slots)
        random.shuffle(round_slots)
        for chromosome_index, half, label, top in round_slots:
            if label not in growing:
                continue
            chromosome = genome.chromosomes[chromosome_index]
            arm = branch_growth_order(chromosome)[0 if top else 1]
            needed_fanouts = (
                max(0, int(n_inputs) - 1) if n_inputs >= 3 else 0)
            scaffold_ready = (
                block_index >= LOGIC_SCAFFOLD_GENES
                and (_shared_open_buds(genome, label) >= needed_fanouts
                     or block_index >= MAX_LOGIC_SCAFFOLD_GENES))
            if scaffold_ready and _arm_has_output_gene(arm):
                connected = False
                for _attempt in range(6):
                    if _connect_terminal_step(
                            genome, families, n_inputs, label):
                        connected = True
                        break
                if not connected:
                    growing.discard(label)
                continue
            arm = branch_growth_order(chromosome)[0 if top else 1]
            gene_id = int(genome.next_gene_id)
            gene = _random_gene(
                genome, gene_id, families, seeds,
                allow_output=not _arm_has_output_gene(arm),
                label=label, prefer_growth=True)
            if gene is None:
                growing.discard(label)
                continue
            genome.next_gene_id = gene_id + 1
            _extend_arm(chromosome, gene, top)
            _repair_genome(genome, families, n_inputs)
            if int(n_inputs) >= 4:
                control = arm_control(chromosome, half)
                if control is not None:
                    control.telomere = life_ceiling
    return _repair_genome(genome, families, n_inputs)


def crossover_branched(parent_a, parent_b, families=None):
    """Cross role modules: each output gene travels with its assigned arm.

    Output geometry is selected per role together with the rules, reach, and
    lifespan that grew from it.  The input pads are the shared environment of
    *all* those modules, so a layout mismatch is not a legal arm graft: retain
    parent A intact and let ordinary mutation explore from it instead.
    """
    child = clone_constructive(parent_a)
    shared = min(len(child.chromosomes), len(parent_b.chromosomes))
    if shared < 1:
        return child
    own_layout = tuple(getattr(child, "input_layout", ()) or ())
    donor_layout = tuple(getattr(parent_b, "input_layout", ()) or ())
    if own_layout != donor_layout:
        return child
    donors = arm_map(parent_b)
    mine = arm_map(child)
    own_outputs = {
        int(gene.branch_id): gene
        for gene in getattr(child, "output_chromosome", Chromosome()).genes
        if isinstance(gene, OutputGene)}
    donor_outputs = {
        int(gene.branch_id): gene
        for gene in getattr(parent_b, "output_chromosome", Chromosome()).genes
        if isinstance(gene, OutputGene)}
    choose_donor = {
        label: (
            label in donor_outputs
            and label in own_outputs
            and str(donor_outputs[label].role) == str(own_outputs[label].role)
            and random.random() < 0.5)
        for label in own_outputs}
    for index in range(shared):
        target = child.chromosomes[index]
        chosen = []
        for half in (0, 1):
            label = branch_label(index, half)
            own_arm = list(mine.get((index, half), ()))
            donor_arm = list(donors.get((index, half), ()))
            if (_function_compatible_arms(own_arm, donor_arm)
                    and any(isinstance(a, ContextGene)
                            and int(a.self_out) != int(b.self_out)
                            for a, b in zip(own_arm, donor_arm))
                    and random.random() < 0.70):
                chosen.append(_recombine_functions(own_arm, donor_arm))
                choose_donor[label] = False
            else:
                source = donors if choose_donor.get(label, False) else mine
                chosen.append(list(source.get((index, half), ())))
        top, bottom = chosen
        room = min(MAX_GENES,
                   MAX_PLACEMENTS - (len(placement_genes(child))
                                     - len(chromosome_rules(target))))
        if len(top) + len(bottom) > room:
            # Trim the outer ends, where an arm is youngest.
            bottom = bottom[:max(0, room - len(top))]
            top = top[:max(0, room - len(bottom))]
        adopted_top = _adopt_branch(child, top)
        adopted_bottom = _adopt_branch(child, bottom)
        target.genes = list(reversed(adopted_top)) + adopted_bottom
        target.split = len(adopted_top)
    selected_output_genes = []
    for label, own_gene in sorted(
            own_outputs.items(), key=lambda item: int(item[0])):
        source = (donor_outputs[label]
                  if choose_donor.get(label, False) else own_gene)
        selected_output_genes.append(copy.copy(source))
    child.output_chromosome.genes = selected_output_genes
    sync_output_layout(child)
    n_inputs = len(getattr(child, "input_layout", None) or (1,))
    return _repair_genome(child, families, n_inputs)


def assemble_role_modules(base, donors, families=None):
    """Join already-evaluated output modules without mutating the join.

    ``donors`` maps stable branch labels to genomes.  Every donor must use the
    same input layout as ``base`` because an output arm is adapted to that
    shared physical environment.  The role's OutputGene and complete arm move
    together, exactly as in crossover; unlike ordinary offspring, this one
    assembly candidate is not immediately damaged by unrelated mutation.
    """
    child = clone_constructive(base)
    base_layout = tuple(getattr(base, "input_layout", ()) or ())
    selected = {
        int(label): donor for label, donor in dict(donors).items()
        if tuple(getattr(donor, "input_layout", ()) or ()) == base_layout}
    if not selected:
        return child

    own_arms = arm_map(child)
    donor_arms = {label: arm_map(donor) for label, donor in selected.items()}
    for index, target in enumerate(child.chromosomes):
        chosen = []
        for half in (0, 1):
            label = branch_label(index, half)
            source = donor_arms.get(label, own_arms)
            chosen.append(list(source.get((index, half), ())))
        top, bottom = chosen
        room = min(
            MAX_GENES,
            MAX_PLACEMENTS - (
                len(placement_genes(child)) - len(chromosome_rules(target))))
        if len(top) + len(bottom) > room:
            bottom = bottom[:max(0, room - len(top))]
            top = top[:max(0, room - len(bottom))]
        adopted_top = _adopt_branch(child, top)
        adopted_bottom = _adopt_branch(child, bottom)
        target.genes = list(reversed(adopted_top)) + adopted_bottom
        target.split = len(adopted_top)

    output_genes = []
    for own_gene in getattr(
            child, "output_chromosome", Chromosome()).genes:
        label = int(own_gene.branch_id)
        donor = selected.get(label)
        source = own_gene
        if donor is not None:
            source = next((
                gene for gene in getattr(
                    donor, "output_chromosome", Chromosome()).genes
                if (isinstance(gene, OutputGene)
                    and int(gene.branch_id) == label
                    and str(gene.role) == str(own_gene.role))
            ), own_gene)
        output_genes.append(copy.copy(source))
    child.output_chromosome.genes = output_genes
    sync_output_layout(child)
    return _repair_genome(child, families, len(base_layout) or 1)
