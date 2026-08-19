"""Verified Full Adder synthesis for the live branched tri-tile genome.

The result is an ordinary :class:`BranchedHexGenome`: context rules grow it,
the paper-analog simulator runs it, and the unchanged target contract scores
it.  This module supplies a reachable plateau seed, not a fitness shortcut.
"""
from __future__ import annotations

from ..fnv.genome import input_ring
from .branched import (
    DEPTH_BANDS, OUT_STATE, BranchedHexChromosome, BranchedHexGenome,
    HexContextGene, HexControlGene, HexInputGene, HexOutputGene, IoChromosome,
    _state_of, arm_reach, develop_branched_hex, drives_toward_root, growth_candidates,
    output_root_sites, routing_sources)
from .branched_ga import input_pads
from .hexgrid import hex_dirs, honeycomb_distance


class SynthesisError(ValueError):
    """The requested target or run limits cannot hold this witness."""


FULL_ADDER_CASES = (
    ((0, 0, 0), (0, 0)), ((0, 0, 1), (1, 0)),
    ((0, 1, 0), (1, 0)), ((0, 1, 1), (0, 1)),
    ((1, 0, 0), (1, 0)), ((1, 0, 1), (0, 1)),
    ((1, 1, 0), (0, 1)), ((1, 1, 1), (1, 1)),
)

# Minimal live cone of the timing-balanced circuit derived in
# tools/handbuild_tri_logic.py.  Coordinates are retained in the derivation's
# frame, then mapped to the genome's input-0 origin below.  The one tile outside
# both output cones is deliberately absent.
_BODY = dict((
    ((-7, -7), 32), ((-7, -6), 3072), ((-6, -8), 32),
    ((-6, -7), 2048), ((-6, -6), 33), ((-6, -5), 3072),
    ((-5, -8), 2), ((-5, -7), 1024), ((-5, -6), 3074),
    ((-5, -5), 4), ((-4, -8), 3072), ((-4, -7), 1),
    ((-4, -6), 98), ((-4, -4), 96), ((-3, -9), 96),
    ((-3, -7), 2144), ((-3, -6), 34), ((-3, -5), 3072),
    ((-3, -4), 20), ((-2, -9), 2), ((-2, -8), 32),
    ((-2, -7), 3744), ((-2, -6), 256), ((-2, -5), 3072),
    ((-2, -4), 96), ((-1, -9), 96), ((-1, -8), 2),
    ((-1, -7), 4), ((-1, -6), 2050), ((-1, -5), 1),
    ((-1, -4), 2048), ((0, -9), 2), ((0, -8), 2144),
    ((0, -7), 33), ((0, -6), 96), ((1, -9), 3072),
    ((1, -8), 34), ((1, -7), 3072), ((1, -6), 20),
    ((2, -8), 256), ((2, -7), 3072), ((2, -6), 96),
    ((3, -7), 1), ((3, -6), 2048),
))
_PADS = ((-5, -4), (-4, -5), (-4, -9))
_SUM_ROOT = (2, -8)
_CARRY_ROOT = (-5, -7)
_TAUS = {(-7, -6): 6, (-7, -7): 6, (-6, -7): 6}

# A connected 23/21 partition of the two reverse-growth territories.  The
# nearest-root split was 29/15 and exceeded the normal arm lifespan despite
# using the same physical circuit.  Jointly assigning the shared cone removes
# that purely developmental obstruction.
_CARRY_OWNED = frozenset({
    (-7, -7), (-7, -6), (-6, -8), (-6, -7), (-6, -6), (-6, -5),
    (-5, -8), (-5, -7), (-5, -6), (-5, -5), (-4, -8), (-4, -7),
    (-4, -6), (-3, -7), (-3, -6), (-2, -8), (-2, -7), (-1, -8),
    (-1, -7), (0, -8), (0, -7),
})


def _full_adder_roles(target):
    cases = (getattr(target, 'temporal_logic_cases', ())
             or getattr(target, 'combinational_cases', ())
             or getattr(target, 'cases', ()))
    cases = tuple((tuple(map(int, inputs)), tuple(map(int, outputs)))
                  for inputs, outputs in cases)
    roles = tuple(str(output.role) for output in getattr(target, 'outputs', ()))
    if len(roles) != 2 or len(cases) != 8:
        raise SynthesisError('branched nervous rescue supports Full Adder')
    ordered = sorted(cases)
    if tuple(inputs for inputs, _ in ordered) != tuple(
            inputs for inputs, _ in FULL_ADDER_CASES):
        raise SynthesisError('rescue requires the exhaustive 3-input table')
    expected = tuple(outputs for _inputs, outputs in FULL_ADDER_CASES)
    columns = [tuple(outputs[index] for _inputs, outputs in ordered)
               for index in range(2)]
    xor_column = tuple(outputs[0] for outputs in expected)
    carry_column = tuple(outputs[1] for outputs in expected)
    try:
        xor_index = columns.index(xor_column)
        carry_index = columns.index(carry_column)
    except ValueError as exc:
        raise SynthesisError('truth table is not a Full Adder') from exc
    if xor_index == carry_index:
        raise SynthesisError('Full Adder outputs must be distinct')
    return roles[xor_index], roles[carry_index]


def _to_origin(cell):
    """Direction-preserving half-turn sending the first pad to (0, 0)."""
    return (_PADS[0][0] - cell[0], _PADS[0][1] - cell[1])


def _placement(cell):
    distance = honeycomb_distance((0, 0), cell)
    ring = input_ring(distance)
    try:
        return ring.index(cell), distance
    except ValueError as exc:
        raise SynthesisError('port is not representable on its honeycomb ring') from exc


def _dependency_owners(grid, pads, output_cells):
    """Partition a physical multi-output cone into developmental arms."""
    pad_set = set(pads)
    owners = {}
    for label, root in enumerate(output_cells, 1):
        stack = [root]
        while stack:
            cell = stack.pop()
            if cell in pad_set or cell not in grid or cell in owners:
                continue
            owners[cell] = label
            around = hex_dirs(*cell)
            for direction in sorted(routing_sources(grid[cell]), reverse=True):
                source = around[direction]
                if source in grid and source not in pad_set:
                    stack.append(source)
    return owners


def synthesize_branched_grid(
        target, grid, pads, outputs, chromosome_count=2, max_telomere=24,
        tau_indices=None, owners=None):
    """Reverse-compile a verified tri-tile phenotype into a live genome."""
    chromosome_count = max(1, int(chromosome_count))
    roles = tuple(str(output.role) for output in getattr(target, 'outputs', ()))
    if not roles or len(roles) > 2 * chromosome_count:
        raise SynthesisError('not enough nervous arms for every output')
    pads = tuple(map(tuple, pads))
    if not pads:
        raise SynthesisError('compiled nervous circuit needs an input anchor')

    # The live genome pins pad zero at (0, 0).  Honeycomb direction labels
    # depend on coordinate parity: an even-parity translation preserves them,
    # while moving an odd anchor to the origin needs the compensating
    # half-turn used by the Full Adder.
    anchor = pads[0]
    if (anchor[0] + anchor[1]) & 1:
        transform = lambda cell: (anchor[0] - cell[0], anchor[1] - cell[1])
    else:
        transform = lambda cell: (cell[0] - anchor[0], cell[1] - anchor[1])
    pad_sites = tuple(transform(cell) for cell in pads)
    source_grid = {tuple(cell): int(state) for cell, state in grid.items()
                   if tuple(cell) not in set(pads)}
    wanted_all = {transform(cell): state for cell, state in source_grid.items()}
    output_cells = tuple(transform(outputs[role]) for role in roles)
    owner_map = ({transform(cell): int(label) for cell, label in owners.items()}
                 if owners is not None else
                 _dependency_owners(wanted_all, pad_sites, output_cells))
    if any(root not in owner_map for root in output_cells):
        raise SynthesisError('compiled nervous output is not dependency-reachable')
    wanted = {cell: state for cell, state in wanted_all.items()
              if cell in owner_map}
    transformed_taus = {
        transform(cell): int(index)
        for cell, index in dict(tau_indices or {}).items()
        if transform(cell) in wanted}
    owner_sizes = {
        label: sum(owner == label for owner in owner_map.values())
        for label in range(1, len(roles) + 1)}
    limit = max(1, int(max_telomere))
    if any(size > limit for size in owner_sizes.values()):
        raise SynthesisError(
            'compiled nervous arm needs more tiles than configured lifespan')

    chromosomes = [BranchedHexChromosome(
        controls=[HexControlGene(tolerance=0, telomere=limit),
                  HexControlGene(tolerance=0, telomere=limit)])
        for _ in range(chromosome_count)]
    next_gene_id = 1
    for label, root in enumerate(output_cells, 1):
        chromosomes[(label - 1) // 2].genes.append(HexContextGene(
            next_gene_id, self_in=OUT_STATE, self_out=wanted[root],
            branch_id=label, tau_index=transformed_taus.get(root, 0)))
        next_gene_id += 1
    for label, size in owner_sizes.items():
        chromosomes[(label - 1) // 2].controls[(label - 1) % 2] = (
            HexControlGene(tolerance=0, telomere=size))

    input_genes = [HexInputGene(*_placement(cell)) for cell in pad_sites[1:]]
    output_genes = []
    for label, (role, root) in enumerate(zip(roles, output_cells), 1):
        bearing, distance = _placement(root)
        output_genes.append(HexOutputGene(role, bearing, distance, label))
    genome = BranchedHexGenome(
        chromosomes=chromosomes,
        io_chromosome=IoChromosome(inputs=input_genes, outputs=output_genes),
        next_gene_id=next_gene_id, arch='tri3')
    trace = _compile_body(genome, wanted, owner_map, transformed_taus)
    if trace.grid != wanted:
        raise SynthesisError('compiled nervous circuit did not replay exactly')
    return genome


def _compile_body(genome, wanted, owners, wanted_taus):
    pads = input_pads(genome)
    roots = output_root_sites(genome, pads)
    output_sites = set(roots.values()) - set(pads)

    def is_prefix(trace):
        return all(
            cell in wanted
            and wanted[cell] == state
            and owners[cell] == trace.owners.get(cell)
            and wanted_taus.get(cell, 0) == trace.taus.get(cell, 0)
            for cell, state in trace.grid.items())

    for _round in range(len(wanted) + 6):
        trace = develop_branched_hex(genome, pads)
        if not is_prefix(trace):
            raise SynthesisError('compiled nervous genome grew an unintended tile')
        if trace.grid == wanted:
            return trace
        candidates = set(growth_candidates(
            trace.grid, trace.owners, set(pads), output_sites))
        accepted = False
        for cell in sorted(set(wanted) - set(trace.grid)):
            if cell not in candidates:
                continue
            label = owners[cell]
            depth = arm_reach(cell, label, trace.owners, trace.depths)
            if depth is None:
                continue
            around = hex_dirs(*cell)
            context = tuple(
                _state_of(around[direction], trace.grid, set(pads), output_sites)
                for direction in ('L', 'R', 'D')) + (
                    _state_of(cell, trace.grid, set(pads), output_sites),)
            chromosome = genome.chromosomes[(label - 1) // 2]
            gene = HexContextGene(
                genome.next_gene_id,
                context[0], context[1], context[2], context[3],
                wanted[cell], label, min(int(depth), DEPTH_BANDS - 1),
                tau_index=wanted_taus.get(cell, 0))
            chromosome.genes.append(gene)
            trial = develop_branched_hex(genome, pads)
            if (is_prefix(trial)
                    and len(trial.grid) > len(trace.grid)):
                genome.next_gene_id += 1
                accepted = True
                break
            chromosome.genes.pop()
        if not accepted:
            missing = sorted(set(wanted) - set(trace.grid))
            reachable = [
                (cell, owners[cell],
                 arm_reach(cell, owners[cell], trace.owners, trace.depths),
                 drives_toward_root(
                     cell, wanted[cell], owners[cell],
                     arm_reach(cell, owners[cell], trace.owners, trace.depths),
                     trace.grid, trace.owners, trace.depths))
                for cell in missing if cell in candidates]
            raise SynthesisError(
                'no exact context extends the nervous genome after %d/%d tiles; '
                'missing=%r reachable=%r'
                % (len(trace.grid), len(wanted), missing[:6], reachable[:6]))
    raise SynthesisError('branched nervous Full Adder compiler did not converge')


def synthesize_branched_full_adder(
        target, chromosome_count=2, max_telomere=24):
    """Compile a timing-balanced Full Adder into the live nervous encoding."""
    sum_role, carry_role = _full_adder_roles(target)
    chromosome_count = max(1, int(chromosome_count))
    if chromosome_count < 1:
        raise SynthesisError('one chromosome is required for two output arms')

    wanted = {_to_origin(cell): state for cell, state in _BODY.items()}
    owners = {_to_origin(cell): (2 if cell in _CARRY_OWNED else 1)
              for cell in _BODY}
    wanted_taus = {_to_origin(cell): index for cell, index in _TAUS.items()}
    owner_sizes = [sum(label == owner for owner in owners.values())
                   for label in (1, 2)]
    limit = int(max_telomere)
    if limit < max(owner_sizes):
        raise SynthesisError(
            'Full Adder needs arm lifespan %d (configured %d)'
            % (max(owner_sizes), limit))

    pad_sites = tuple(_to_origin(cell) for cell in _PADS)
    input_genes = [HexInputGene(*_placement(cell)) for cell in pad_sites[1:]]
    sum_root, carry_root = _to_origin(_SUM_ROOT), _to_origin(_CARRY_ROOT)
    sum_bearing, sum_distance = _placement(sum_root)
    carry_bearing, carry_distance = _placement(carry_root)
    chromosomes = [BranchedHexChromosome() for _ in range(chromosome_count)]
    chromosomes[0] = BranchedHexChromosome(
        genes=[
            HexContextGene(
                1, self_in=OUT_STATE, self_out=wanted[sum_root],
                branch_id=1, tau_index=wanted_taus.get(sum_root, 0)),
            HexContextGene(
                2, self_in=OUT_STATE, self_out=wanted[carry_root],
                branch_id=2, tau_index=wanted_taus.get(carry_root, 0)),
        ],
        controls=[HexControlGene(tolerance=0, telomere=owner_sizes[0]),
                  HexControlGene(tolerance=0, telomere=owner_sizes[1])])
    genome = BranchedHexGenome(
        chromosomes=chromosomes,
        io_chromosome=IoChromosome(
            inputs=input_genes,
            outputs=[
                HexOutputGene(sum_role, sum_bearing, sum_distance, 1),
                HexOutputGene(carry_role, carry_bearing, carry_distance, 2),
            ]),
        next_gene_id=3, arch='tri3')
    trace = _compile_body(genome, wanted, owners, wanted_taus)
    if trace.grid != wanted or trace.taus != {
            cell: wanted_taus.get(cell, 0) for cell in wanted}:
        raise SynthesisError('compiled nervous Full Adder did not replay exactly')
    return genome
