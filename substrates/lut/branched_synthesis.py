"""Verified truth-table rescue for the live output-rooted LUT encoding.

This is a compiler into the same ``BranchedLutGenome`` population evolution
uses.  It does not grant fitness or bypass simulation: the candidate is grown
from its context rules and must score under the ordinary target contract.
"""
from __future__ import annotations

from .branched import (
    DEPTH_BANDS, DIRECTIONS, OUT_CELL,
    BranchedLutChromosome, BranchedLutGenome, LutContextGene, LutControlGene,
    LutInputGene, LutIoChromosome, LutOutputGene, _cell_of, arm_reach,
    bearing_cell, catalogue, cell_distance, cell_sources,
    develop_branched_lut, drives_toward_root, growth_candidates, neighbours,
    output_root_sites, table_support)
from .branched_ga import input_pads
from .functions import FAMILY_TABLES, INPUT_TABLES, normalise_function_families
from .synthesis import SynthesisError, synthesize_grid


FULL_ADDER_CASES = (
    ((0, 0, 0), (0, 0)), ((0, 0, 1), (1, 0)),
    ((0, 1, 0), (1, 0)), ((0, 1, 1), (0, 1)),
    ((1, 0, 0), (1, 0)), ((1, 0, 1), (0, 1)),
    ((1, 1, 0), (0, 1)), ((1, 1, 1), (1, 1)),
)


def _full_adder_roles(target):
    """Return (xor_role, majority_role), recognizing behavior rather than name."""
    cases = tuple(
        (tuple(int(v) for v in inputs), tuple(int(v) for v in outputs))
        for inputs, outputs in (getattr(target, 'combinational_cases', ()) or ()))
    roles = tuple(str(output.role) for output in getattr(target, 'outputs', ()))
    if len(roles) != 2 or len(cases) != 8:
        raise SynthesisError('branched rescue currently supports Full Adder')
    ordered = sorted(cases)
    if tuple(inputs for inputs, _outputs in ordered) != tuple(
            inputs for inputs, _outputs in FULL_ADDER_CASES):
        raise SynthesisError('branched rescue requires the exhaustive 3-input table')
    xor_index = majority_index = None
    for output_index in range(2):
        column = tuple(outputs[output_index] for _inputs, outputs in ordered)
        if column == tuple(outputs[0] for _inputs, outputs in FULL_ADDER_CASES):
            xor_index = output_index
        if column == tuple(outputs[1] for _inputs, outputs in FULL_ADDER_CASES):
            majority_index = output_index
    if xor_index is None or majority_index is None or xor_index == majority_index:
        raise SynthesisError('truth table is not a Full Adder')
    return roles[xor_index], roles[majority_index]


def _compile_body(genome, wanted, wanted_owners):
    """Emit exact depth/context rules until replay grows ``wanted`` only."""
    pads = input_pads(genome)
    roots = output_root_sites(genome, pads)
    output_sites = set(roots.values()) - set(pads)

    def is_prefix(trace):
        return all(cell in wanted and wanted[cell] == state
                   for cell, state in trace.grid.items())

    for _round in range(len(wanted) + 4):
        trace = develop_branched_lut(genome, pads)
        if not is_prefix(trace):
            raise SynthesisError('compiled LUT grew an unintended cell')
        if trace.grid == wanted:
            return trace
        candidates = set(growth_candidates(
            trace.grid, trace.owners, set(pads), output_sites))
        accepted = False
        last_rejection = None
        for cell in sorted(set(wanted) - set(trace.grid)):
            if cell not in candidates:
                continue
            label = wanted_owners[cell]
            depth = arm_reach(cell, label, trace.owners, trace.depths)
            if depth is None:
                continue
            around = neighbours(cell)
            context = tuple(
                _cell_of(around[direction], trace.grid, set(pads), output_sites)
                for direction in DIRECTIONS) + (
                    _cell_of(cell, trace.grid, set(pads), output_sites),)
            chromosome = genome.chromosomes[(label - 1) // 2]
            gene = LutContextGene(
                genome.next_gene_id,
                context[0], context[1], context[2], context[3], context[4],
                wanted[cell], label, min(int(depth), DEPTH_BANDS - 1))
            chromosome.genes.append(gene)
            trial = develop_branched_lut(genome, pads)
            if is_prefix(trial) and len(trial.grid) > len(trace.grid):
                genome.next_gene_id += 1
                accepted = True
                break
            wrong = next((
                (placed, state, wanted.get(placed))
                for placed, state in sorted(trial.grid.items())
                if placed not in wanted or wanted.get(placed) != state), None)
            last_rejection = (cell, len(trial.grid), wrong)
            chromosome.genes.pop()
        if not accepted:
            remaining = sorted(set(wanted) - set(trace.grid))
            diagnostic = None
            if remaining:
                cell = remaining[0]
                label = wanted_owners[cell]
                around = neighbours(cell)
                context = tuple(
                    _cell_of(around[direction], trace.grid, set(pads), output_sites)
                    for direction in DIRECTIONS) + (
                        _cell_of(cell, trace.grid, set(pads), output_sites),)
                depth = arm_reach(cell, label, trace.owners, trace.depths)
                diagnostic = {
                    'candidate': cell in candidates,
                    'depth': depth,
                    'drive': (depth is not None and drives_toward_root(
                        cell, wanted[cell], label, depth,
                        trace.owners, trace.depths)),
                    'context': context,
                }
            raise SynthesisError(
                'no exact context extends the compiled LUT '
                '(grown %d/%d; next %r owner %r; %d candidates; '
                'reject %r; diagnostic %r)'
                % (len(trace.grid), len(wanted),
                   remaining[0] if remaining else None,
                   wanted_owners.get(remaining[0]) if remaining else None,
                   len(candidates), last_rejection, diagnostic))
    raise SynthesisError('branched LUT compiler did not converge')


def _placement_for(cell, taken, max_distance=32):
    """Return a placement allele that resolves to exactly ``cell``."""
    cell = tuple(cell)
    for distance in range(1, max(1, int(max_distance)) + 1):
        for bearing in range(4 * distance):
            if bearing_cell(bearing, distance, taken) == cell:
                return bearing, distance
    raise SynthesisError('compiled port lies outside the placement domain')


def _dependency_owners(grid, inputs, output_cells):
    """Partition a shared LUT circuit into output-rooted developmental arms.

    A later arm stops when it reaches cells already grown by an earlier arm.
    That boundary is still a real physical wire; ownership is developmental,
    not an electrical insulator.  This lets several output roots reuse one
    crossbar without cloning the whole input network.
    """
    pads = set(map(tuple, inputs))
    owners = {}
    for label, root in enumerate(output_cells, 1):
        stack = [tuple(root)]
        while stack:
            cell = stack.pop()
            if cell in pads or cell not in grid or cell in owners:
                continue
            owners[cell] = label
            around = neighbours(cell)
            for direction in sorted(cell_sources(grid[cell]), reverse=True):
                source = around[direction]
                if source in grid and source not in pads:
                    stack.append(source)
    return owners


def synthesize_branched_truth_table(
        target, chromosome_count=2, max_telomere=32,
        function_families=None):
    """Compile any retained <=4-input, <=4-output table into the live genome.

    The phenotype comes from the same compact directional crossbar used by the
    older spatial-I/O compiler.  Here it is partitioned into ordinary
    output-rooted arms and then reverse-compiled into context rules.  Five-port
    targets (four data lanes plus case-valid strobe) use that compiler's
    verified two-stage cascade.
    """
    grid, inputs, outputs = synthesize_grid(target, seed_pos=(0, 0))
    return synthesize_branched_grid(
        target, grid, inputs, outputs,
        chromosome_count=chromosome_count, max_telomere=max_telomere,
        function_families=function_families)


def synthesize_branched_grid(
        target, grid, inputs, outputs, chromosome_count=2,
        max_telomere=32, function_families=None):
    """Reverse-compile an explicit LUT phenotype into the live genome."""
    chromosome_count = max(1, int(chromosome_count))
    roles = tuple(str(output.role) for output in getattr(target, 'outputs', ()))
    if not roles or len(roles) > 2 * chromosome_count:
        raise SynthesisError('not enough developmental arms for every output')
    families = normalise_function_families(function_families)
    if 'UNRESTRICTED' not in families:
        raise SynthesisError(
            'general truth-table rescue requires the UNRESTRICTED LUT bank')
    grid = {tuple(cell): tuple(state) for cell, state in grid.items()}
    inputs = tuple(map(tuple, inputs))
    if not inputs or inputs[0] != (0, 0):
        raise SynthesisError('live branched LUT fixes input zero at the origin')
    output_cells = tuple(outputs[role] for role in roles)
    owners = _dependency_owners(grid, inputs, output_cells)
    if any(root not in owners for root in output_cells):
        raise SynthesisError('compiled output is not dependency-reachable')

    # Source pads are materialised by the live evaluator, not grown by arms.
    wanted = {cell: tuple(state) for cell, state in grid.items()
              if cell in owners and cell not in set(inputs)}
    owner_counts = {
        label: sum(1 for owner in owners.values() if owner == label)
        for label in range(1, len(roles) + 1)}
    telomere = max(1, int(max_telomere or 1))
    if any(count > telomere for count in owner_counts.values()):
        raise SynthesisError(
            'compiled arm needs more cells than the configured telomere')

    chromosomes = [BranchedLutChromosome(
        controls=[LutControlGene(tolerance=0, telomere=telomere),
                  LutControlGene(tolerance=0, telomere=telomere)])
        for _ in range(chromosome_count)]
    next_gene_id = 1
    for label, root in enumerate(output_cells, 1):
        chromosomes[(label - 1) // 2].genes.append(LutContextGene(
            next_gene_id, self_in=OUT_CELL, self_out=wanted[root],
            branch_id=label))
        next_gene_id += 1

    taken = {(0, 0)}
    input_genes = []
    for cell in inputs[1:]:
        bearing, distance = _placement_for(cell, taken)
        input_genes.append(LutInputGene(bearing, distance))
        taken.add(tuple(cell))
    output_genes = []
    for label, (role, cell) in enumerate(zip(roles, output_cells), 1):
        bearing, distance = _placement_for(cell, taken)
        output_genes.append(LutOutputGene(role, bearing, distance, label))
        taken.add(tuple(cell))

    genome = BranchedLutGenome(
        chromosomes=chromosomes,
        io_chromosome=LutIoChromosome(
            inputs=input_genes, outputs=output_genes),
        families=families, next_gene_id=next_gene_id)
    trace = _compile_body(genome, wanted, owners)
    if trace.grid != wanted:
        raise SynthesisError('compiled LUT did not replay exactly')
    return genome


def synthesize_branched_full_adder(
        target, chromosome_count=2, max_telomere=8,
        function_families=None):
    """Compile an exhaustive Full Adder into a six-cell live branched genome."""
    xor_role, majority_role = _full_adder_roles(target)
    chromosome_count = max(1, int(chromosome_count))
    telomere = min(8, max(1, int(
        8 if max_telomere is None else max_telomere)))
    if telomere < 3:
        raise SynthesisError('branched Full Adder requires telomere >= 3')
    families = normalise_function_families(function_families)

    xor3 = next(table for table in FAMILY_TABLES['XOR']
                if table_support(table) == {'N', 'S', 'W'})
    majority3 = next(table for table in FAMILY_TABLES['THRESHOLD']
                     if table_support(table) == {'N', 'S', 'E'})
    relay_e = INPUT_TABLES['E']
    required = {xor3, majority3, relay_e}
    available = {int(table) for _family, table in catalogue(families)}
    if not required.issubset(available):
        raise SynthesisError(
            'enabled LUT function banks cannot express the compiled Full Adder')

    sum_state = (xor3, 0, 0, 0)
    carry_state = (majority3, 0, 0, 0)
    relay_w_from_e = (0, 0, 0, relay_e)
    relay_s_from_e = (0, relay_e, 0, 0)
    relay_n_from_e = (relay_e, 0, 0, 0)
    chromosomes = [BranchedLutChromosome() for _ in range(chromosome_count)]
    chromosomes[0] = BranchedLutChromosome(
        genes=[
            LutContextGene(
                1, self_in=OUT_CELL, self_out=sum_state, branch_id=1),
            LutContextGene(
                2, self_in=OUT_CELL, self_out=carry_state, branch_id=2),
        ],
        controls=[LutControlGene(tolerance=0, telomere=telomere),
                  LutControlGene(tolerance=0, telomere=telomere)])
    genome = BranchedLutGenome(
        chromosomes=chromosomes,
        io_chromosome=LutIoChromosome(
            inputs=[LutInputGene(bearing=1, distance=2),
                    LutInputGene(bearing=7, distance=2)],
            outputs=[LutOutputGene(xor_role, 0, 1, 1),
                     LutOutputGene(majority_role, 2, 1, 2)]),
        families=families, next_gene_id=3)
    wanted = {
        (1, 0): sum_state,
        (-1, 0): carry_state,
        (-1, 1): relay_s_from_e,
        (0, 1): relay_w_from_e,
        (-1, -1): relay_n_from_e,
        (0, -1): relay_w_from_e,
    }
    owners = {(1, 0): 1, **{
        cell: 2 for cell in wanted if cell != (1, 0)}}
    trace = _compile_body(genome, wanted, owners)
    if trace.grid != wanted:
        raise SynthesisError('compiled LUT did not replay exactly')
    return genome
