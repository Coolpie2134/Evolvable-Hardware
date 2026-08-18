"""
substrates/lut/branched_repair.py - variation 3: output-rooted growth with a
deterministic connectivity repair.

WHAT THIS VARIATION IS FOR
--------------------------
Solve rate and speed, both limited by the same thing: most fresh organisms
cannot drive their outputs at all, so a run spends its budget on genomes that
score the silent baseline whatever their rules say. Measured on a two-output
target under variation 1, only 3 organisms in 35 had BOTH outputs live - 15 had
none and 17 had exactly one.

The existing answer is REJECTION SAMPLING: build six organisms, keep the most
connected (`select_developmental_seed`). It costs six builds per genome - 0.54s
against 0.09s for one - and it plateaus at 26% both-live no matter how many
attempts it is given, because it selects on a structural reachability proxy that
reports "both driven" for organisms the simulator then finds silent.

This variation replaces rejection with CONSTRUCTION. Grow exactly once, then lay
a minimal transport path from any undriven output root back to the nearest input
pad. Connectivity stops being something a genome must be lucky enough to have
and becomes something every organism is guaranteed, so selection is spent on
FUNCTION rather than on wiring.

MEASURED OUTCOME: THE IDEA DOES NOT WORK
----------------------------------------
Kept because the mechanism is sound and the measurement is worth not repeating,
NOT because it is the variation to use. Head to head against variation 1 with
six-way seed selection:

    two-output (Half adder), 30 organisms
        v1 + selection   0 live:8   1 live:16  2 live:1   unusable:5   1.26s
        v3 repair        0 live:16  1 live:13  2 live:1   unusable:0   0.46s

    single-output (AND), 30 organisms
        v1 + selection   live 15/29 (52%)   0.22s
        v3 repair        live  8/30 (27%)   0.25s

So the repair guarantees STRUCTURAL connectivity - unusable organisms drop to
zero, and on the two-output target it is 2.8x faster - but it produces FEWER
organisms whose outputs actually do anything. On the single-output target it is
half as good and no faster.

The reason is the thing this variation was built to route around, and it turns
out to bite here too: rejection sampling picks organisms whose OWN evolved
structure already conducts, while repair takes a body that does not conduct and
bolts a wire onto it. A wire that satisfies the structural reachability test
still frequently produces nothing under simulation - the same optimistic proxy
that made seed selection plateau at 26%. Verifying the repair against the
simulator and repairing again (the second pass in prepare_repaired_lut_net)
changed the numbers not at all.

The honest lesson: connectivity is necessary but nowhere near sufficient, and
the gap between "the inputs can reach this output" and "this output does
something" is where the multi-output problem actually lives. Fixing THAT is what
would raise solve rates; neither construction nor rejection addresses it.

WHY THIS IS NOT CHEATING
------------------------
The repair is target-blind. It reads the grown body and the port positions and
nothing else - never the truth table, never a trial, never a score. It is the
same class of intervention as developmental seed selection, which the project
already accepts, only constructive rather than rejective.

It IS a real bias and worth stating plainly: every organism now contains at
least a wire from input to output, so a "connect the input straight to the
output" solution is always one mutation away. `tools/probe_temporal_baselines.py`
exists to check whether that wire is enough to score well on a given target; if
it is, the target is measuring the repair rather than the genome.

WHAT IT KEEPS FROM THE OTHER TWO
--------------------------------
Output-rooted growth from variation 1, which is the direction that actually
certifies (AND / XOR / Half adder, held out). Feedback edges stay legal, because
LUT memory is topological and forbidding cycles forbids state. Interface-only
tolerance, sentinel distances, small telomeres. What it drops is the six-way
seed selection, which the repair makes redundant.
"""
from __future__ import annotations

import collections

from .branched import (DIRECTIONS, EMPTY_CELL, OPPOSITE, STEP,
                       cell_outputs, cell_sources, develop_branched_lut,
                       driven_roots, materialise_pads, neighbours,
                       output_root_sites, table_support)

#: Tables that copy one neighbour straight through, indexed by the direction
#: they READ. These are the transport parts the repair lays down; they are
#: ordinary ROUTING-bank tables, not a privileged cell type.
COPY_FROM = {'N': 0xAAAA, 'S': 0xCCCC, 'E': 0xF0F0, 'W': 0xFF00}

#: How far the repair will search for a pad before giving up. A body is a couple
#: of dozen cells inside a small lattice, so a path longer than this is not a
#: repair, it is a new organism.
MAX_REPAIR_PATH = 24


def _route(root, pads, grid, limit=MAX_REPAIR_PATH):
    """Shortest lattice path from ``root`` back to any pad, or None.

    Breadth-first so the path is minimal, and it prefers to travel through
    CELLS THAT ALREADY EXIST: reusing the organism's own body keeps the repair
    as small as possible and leaves the evolved structure carrying the signal
    wherever it already can.
    """
    start = tuple(root)
    pads = set(map(tuple, pads))
    if start in pads:
        return [start]
    seen = {start: None}
    queue = collections.deque([(start, 0)])
    while queue:
        cell, depth = queue.popleft()
        if depth >= limit:
            continue
        # Existing cells first, so an equal-length route through live body wins
        # over one that has to build new transport.
        neighbourhood = sorted(
            neighbours(cell).items(),
            key=lambda item: (item[1] not in grid, item[0]))
        for _direction, nxt in neighbourhood:
            if nxt in seen:
                continue
            seen[nxt] = cell
            if nxt in pads:
                path = [nxt]
                while path[-1] != start:
                    path.append(seen[path[-1]])
                return path                      # pad -> ... -> root
            queue.append((nxt, depth + 1))
    return None


def repair_output_paths(genome, pads, trace, force=()):
    """Guarantee every output root can be driven by an input. Target-blind.

    Returns a new grid. For each root the inputs cannot already reach, a minimal
    path is routed back to the nearest pad and transport is installed ALONG it:

    * an empty cell on the path becomes a pure copy part, reading from the way
      the signal came and driving the way it is going;
    * an occupied cell is left alone unless the direction it must now drive is
      unused, in which case that one table is filled in. Evolved structure is
      never overwritten - a repair that clobbered working parts would be
      undoing the search it exists to help.
    """
    grid = dict(trace.grid)
    pads = tuple(map(tuple, pads))
    if not pads:
        return grid
    roots = output_root_sites(genome, pads)
    already = driven_roots(materialise_pads(grid, pads), pads, roots)
    # ``force`` names roots the SIMULATOR found silent. They are repaired even
    # when the structural test believes they are already driven, because that
    # test is optimistic by construction - it asks whether a table CAN go high
    # given reachable inputs, not whether one ever does. Trusting it is what
    # left 16 organisms in 30 silent after a repair had supposedly run.
    already = set(already) - set(force)
    for label, root in sorted(roots.items()):
        if label in already:
            continue
        path = _route(root, pads, grid)
        if path is None:
            continue
        # path runs pad -> ... -> root; walk it forward installing transport.
        for index in range(1, len(path)):
            cell = path[index]
            previous = path[index - 1]
            arrive = next(d for d, step in STEP.items()
                          if (cell[0] + step[0], cell[1] + step[1]) == previous)
            if index + 1 < len(path):
                nxt = path[index + 1]
                drive = next(d for d, step in STEP.items()
                             if (cell[0] + step[0], cell[1] + step[1]) == nxt)
            else:
                drive = None                     # the root itself drives nothing
            state = list(grid.get(cell, EMPTY_CELL))
            if drive is not None:
                slot = DIRECTIONS.index(drive)
                # The chain only conducts if the table driving onward actually
                # READS the direction the signal arrives from. An existing table
                # that drives the right way but listens elsewhere breaks the
                # path just as surely as an empty one - that was the difference
                # between "repair ran" and "repair worked".
                if state[slot] == 0 or arrive not in table_support(state[slot]):
                    state[slot] = COPY_FROM[arrive]
            else:
                # The root. It has to LISTEN to the arriving signal, or nothing
                # the readout observes there will ever change.
                if not any(arrive in table_support(table)
                           for table in state if table):
                    slot = DIRECTIONS.index(OPPOSITE[arrive])
                    state[slot] = COPY_FROM[arrive]
            grid[cell] = tuple(state)
    return grid


def grow_repaired_lut(genome, seeds, force=()):
    """Grow, then repair. The variation-3 replacement for plain development."""
    pads = tuple(tuple(seed) for seed in seeds)
    trace = develop_branched_lut(genome, pads)
    return repair_output_paths(genome, pads, trace, force=force)


def prepare_repaired_lut_net(genome, target):
    """``prepare_branched_lut_net``'s contract, over the REPAIRED body.

    Mirrors variation 1 exactly except for the one substitution this variation
    exists for: the grown grid is repaired before it is interpreted, so an
    organism is never discarded merely for failing to wire itself up.
    """
    from .branched_ga import input_pads
    from .ga import trace_fixed_outputs

    pads = input_pads(genome)
    n_inputs = int(getattr(target, 'n_inputs', len(pads)))
    if len(pads) < n_inputs:
        return None
    roots = output_root_sites(genome, pads)
    in_pos = list(pads[:n_inputs])

    def build(force=()):
        grid = grow_repaired_lut(genome, pads, force=force)
        if not grid:
            return None
        body = materialise_pads(grid, pads)
        out_pos = {}
        for gene in genome.outputs:
            cell = roots.get(int(gene.branch_id))
            if cell is None or cell not in grid:
                return None
            out_pos[str(gene.role)] = cell
        if len(set(out_pos.values())) != len(out_pos):
            return None
        if len(body) <= len(in_pos):
            return None
        traces = trace_fixed_outputs(body, in_pos, out_pos, target,
                                     source_nodes=set(in_pos))
        if traces is None:
            return None
        return body, out_pos, traces, in_pos

    first = build()
    if first is None:
        return None
    # SECOND PASS. Ask the simulator which roles actually produced anything and
    # repair the ones that did not, then rebuild. One extra simulation per
    # organism buys what six extra GENOME BUILDS could not: seed selection
    # plateaued at 26% both-live because it was choosing between organisms on
    # the same optimistic proxy this pass replaces.
    silent = []
    intervals = getattr(first[2], 'intervals', {}) or {}
    for gene in genome.outputs:
        if not any(intervals.get(str(gene.role), [])):
            silent.append(int(gene.branch_id))
    if not silent:
        return first
    second = build(force=silent)
    return second if second is not None else first
