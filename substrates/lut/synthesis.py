"""Exact feed-forward synthesis for small periodic truth-table targets.

The asynchronous LUT substrate has four independent inputs and four independent
directional output tables per cell. A compact crossbar therefore needs one
logic cell: equal-length input routes approach it from N/S/E/W, and each output
leaves on its own directional wire. Crossings are safe because a LUT cell
carries four distinct directed channels.

The phenotype is converted back into a normal developmental genome. A
polarised, heritable seed breaks the otherwise unavoidable fourfold symmetry of
growth from one centre cell. Truth-table bits that the circuit never addresses
serve as deterministic developmental labels, distinguishing relay cells without
changing observed behaviour.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .genome import Chromosome, Genome, LutGene
from .lut import SEED_STATE, grow_lut
from .pulse import AsyncLutSim
from .reverse import grid_to_genome_lut


Pos = tuple[int, int]

DIRECTIONS = ('N', 'S', 'E', 'W')
DIR_INDEX = {direction: index
             for index, direction in enumerate(DIRECTIONS)}
DELTA = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
RELAY = {'N': 0xAAAA, 'S': 0xCCCC, 'E': 0xF0F0, 'W': 0xFF00}

# Directionally distinct and quiet at dynamic input zero. Wired-OR injection
# still drives all four directions exactly as it does for the historical seed.
POLARISED_SEED = (RELAY['N'], RELAY['S'], RELAY['E'], RELAY['W'])

# Input k reaches hub index bit k because the physical N/S/E/W inputs are bits
# 0/1/2/3. Every route is four edges long.
_HUB = (0, 0)
_INPUT_PATHS = (
    ((-3, 1), (-2, 1), (-1, 1), (0, 1), _HUB),
    ((-3, -1), (-2, -1), (-1, -1), (0, -1), _HUB),
    ((1, 3), (1, 2), (1, 1), (1, 0), _HUB),
    ((-1, 3), (-1, 2), (-1, 1), (-1, 0), _HUB),
)
_OUTPUT_PORTS = {
    'N': (0, 2), 'S': (0, -2), 'E': (2, 0), 'W': (-2, 0)}


class SynthesisError(ValueError):
    """The target or growth bound cannot be compiled exactly."""


@dataclass(frozen=True)
class SynthesisResult:
    genome: Genome
    grid: dict[Pos, tuple[int, int, int, int]]
    inputs: tuple[Pos, ...]
    outputs: dict[str, Pos]
    inverse_report: dict
    label_seed: int


def _direction(start, end):
    delta = end[0] - start[0], end[1] - start[1]
    for direction, candidate in DELTA.items():
        if delta == candidate:
            return direction
    raise ValueError('non-adjacent route step: %r -> %r' % (start, end))


def _truth_table(target):
    source_rows = (
        getattr(target, 'combinational_cases', ())
        or getattr(target, 'temporal_logic_cases', ()))
    rows = [
        (tuple(map(int, input_bits)), tuple(map(int, output_bits)))
        for input_bits, output_bits in
        source_rows]
    if not rows:
        raise SynthesisError('target does not retain a truth table')
    n_inputs = int(
        getattr(target, 'combinational_data_inputs', 0)
        or getattr(target, 'temporal_logic_data_inputs', 0)
        or target.n_inputs)
    n_outputs = len(target.outputs)
    if n_inputs > 4 or n_outputs > 4:
        raise SynthesisError(
            'direct LUT synthesis supports at most four inputs and four outputs')
    if len(rows) != 1 << n_inputs or len({row[0] for row in rows}) != len(rows):
        raise SynthesisError('truth table must contain every input row exactly once')
    if any(len(input_bits) != n_inputs or len(output_bits) != n_outputs
           for input_bits, output_bits in rows):
        raise SynthesisError('truth-table row width does not match target ports')
    zero = next(
        output_bits for input_bits, output_bits in rows
        if not any(input_bits))
    has_strobe = bool(
        getattr(target, 'combinational_strobe', False)
        or int(target.n_inputs) > n_inputs)
    if any(zero) and not has_strobe:
        raise SynthesisError(
            'the all-zero row requests an output event but supplies no input '
            'event; add an explicit case-valid/strobe lane to make silence '
            'observable')
    return rows


def _translate(pos, offset):
    return pos[0] + offset[0], pos[1] + offset[1]


def synthesize_grid(target, seed_pos=(0, 0)):
    """Return a compact perfect phenotype and its explicit I/O cells."""
    rows = _truth_table(target)
    n_inputs = int(target.n_inputs)
    if n_inputs > 4:
        return _synthesize_strobed_four_input_grid(
            target, rows, seed_pos)
    data_inputs = int(
        getattr(target, 'combinational_data_inputs', 0)
        or getattr(target, 'temporal_logic_data_inputs', 0)
        or n_inputs)
    has_strobe = bool(
        getattr(target, 'combinational_strobe', False)
        or n_inputs > data_inputs)
    states = {}
    inputs = []

    for path in _INPUT_PATHS[:n_inputs]:
        inputs.append(path[0])
        for previous, cell, following in zip(path, path[1:], path[2:]):
            incoming = _direction(cell, previous)
            outgoing = _direction(cell, following)
            state = states.setdefault(cell, [0, 0, 0, 0])
            existing = state[DIR_INDEX[outgoing]]
            if existing not in (0, RELAY[incoming]):
                raise SynthesisError('route channel collision at %r' % (cell,))
            state[DIR_INDEX[outgoing]] = RELAY[incoming]

    for source in inputs:
        states.setdefault(source, list(SEED_STATE))
    states[inputs[0]] = list(POLARISED_SEED)

    table = dict(rows)
    hub_state = []
    for output_index in range(4):
        mask = 0
        if output_index < len(target.outputs):
            for dynamic_index in range(16):
                bits = tuple(
                    (dynamic_index >> bit) & 1
                    for bit in range(data_inputs))
                enabled = (
                    not has_strobe
                    or bool((dynamic_index >> data_inputs) & 1))
                if enabled and table[bits][output_index]:
                    mask |= 1 << dynamic_index
        hub_state.append(mask)
    states[_HUB] = hub_state

    outputs = {}
    for output_index, terminal in enumerate(target.outputs):
        direction = DIRECTIONS[output_index]
        side = DELTA[direction]
        port = _OUTPUT_PORTS[direction]
        incoming = _direction(side, _HUB)
        # The side cell is full duplex: one table carries an input into the
        # hub, while another carries this output away from it.
        states.setdefault(side, [0, 0, 0, 0])[
            DIR_INDEX[direction]] = RELAY[incoming]
        states.setdefault(port, [0, 0, 0, 0])[
            DIR_INDEX[direction]] = RELAY[incoming]
        outputs[terminal.role] = port

    offset = seed_pos[0] - inputs[0][0], seed_pos[1] - inputs[0][1]
    return (
        {_translate(pos, offset): tuple(state)
         for pos, state in states.items()},
        tuple(_translate(pos, offset) for pos in inputs),
        {role: _translate(pos, offset)
         for role, pos in outputs.items()},
    )


_QUIET = (0x8000, 0x8000, 0x8000, 0x8000)


def _route(states, path):
    """Configure every interior cell of a directed relay path."""
    for previous, cell, following in zip(path, path[1:], path[2:]):
        incoming = _direction(cell, previous)
        outgoing = _direction(cell, following)
        state = states.setdefault(cell, [0, 0, 0, 0])
        existing = state[DIR_INDEX[outgoing]]
        if existing not in (0, RELAY[incoming]):
            raise SynthesisError('route channel collision at %r' % (cell,))
        state[DIR_INDEX[outgoing]] = RELAY[incoming]


def _logic_mask(predicate):
    return sum(1 << index for index in range(16) if predicate(index))


def _synthesize_strobed_four_input_grid(target, rows, seed_pos):
    """Five-port cascade for four data bits plus a case-valid event.

    The first hub emits up to three nonzero-row result functions plus ANY(data).
    A second gate computes strobe AND NOT ANY, and a merge adds that zero-row
    event to the one output that requests it. All three readouts are delay
    matched.
    """
    data_inputs = int(
        getattr(target, 'combinational_data_inputs', 0)
        or getattr(target, 'temporal_logic_data_inputs', 0))
    if (data_inputs != 4
            or not (getattr(target, 'combinational_strobe', False)
                    or int(target.n_inputs) > data_inputs)
            or int(target.n_inputs) != 5 or len(target.outputs) > 3):
        raise SynthesisError(
            'five-port synthesis supports four data inputs, one strobe, '
            'and at most three outputs')
    table = dict(rows)
    zero_outputs = table[(0, 0, 0, 0)]
    zero_roles = [
        index for index, value in enumerate(zero_outputs) if value]
    if len(zero_roles) != 1:
        raise SynthesisError(
            'five-port synthesis requires exactly one zero-row-high output')
    zero_role = zero_roles[0]
    direct_roles = [
        index for index in range(len(target.outputs))
        if index != zero_role]

    states = {}
    data_sources = []
    for path in _INPUT_PATHS:
        data_sources.append(path[0])
        _route(states, path)
    for source in data_sources:
        states.setdefault(source, list(SEED_STATE))
    states[data_sources[0]] = list(POLARISED_SEED)

    def role_mask(role):
        return _logic_mask(
            lambda index: (
                index != 0
                and bool(table[tuple(
                    (index >> bit) & 1 for bit in range(4))][role])))

    # N and E are the two direct result channels, S is the zero-row merge
    # base, and W carries ANY(data) to the zero detector.
    hub_state = [0, role_mask(zero_role), 0, 0xFFFE]
    for direction, role in zip(('N', 'E'), direct_roles):
        hub_state[DIR_INDEX[direction]] = role_mask(role)
    states[_HUB] = hub_state

    # ANY(data): hub -> zero gate.
    zero_gate = (-4, 0)
    _route(states, [
        _HUB, (-1, 0), (-2, 0), (-3, 0), zero_gate, (-4, -1)])

    # The fifth source is kept close to the body for growth, but its signal
    # takes eight relay stages so it meets ANY at the zero gate.
    strobe_source = (-3, 3)
    strobe_path = [
        strobe_source, (-4, 3), (-5, 3), (-6, 3), (-6, 2),
        (-6, 1), (-6, 0), (-5, 0), zero_gate]
    _route(states, strobe_path)
    states.setdefault(strobe_source, list(SEED_STATE))

    # zero = strobe(W/bit3) AND NOT any(E/bit2), emitted south to merge.
    states[zero_gate][DIR_INDEX['S']] = _logic_mask(
        lambda index: bool(index & (1 << DIR_INDEX['W']))
        and not bool(index & (1 << DIR_INDEX['E'])))
    states.setdefault((-4, -1), [0, 0, 0, 0])[
        DIR_INDEX['S']] = RELAY['N']

    merge = (-4, -2)
    # The nonzero-row base for the zero-high role turns below the opposing input
    # lane. The extra zero relay makes both values reach the merge together.
    _route(states, [
        _HUB, (0, -1), (0, -2), (-1, -2), (-2, -2), (-3, -2),
        merge, (-4, -3), (-4, -4)])
    states[merge][DIR_INDEX['S']] = _logic_mask(
        lambda index: bool(index & (1 << DIR_INDEX['N']))
        or bool(index & (1 << DIR_INDEX['E'])))

    outputs = {
        target.outputs[zero_role].role: (-4, -3)}

    direct_paths = (
        [_HUB, (0, 1), (0, 2), (-1, 2), (-2, 2),
         (-3, 2), (-4, 2), (-4, 1), (-5, 1)],
        [_HUB, (1, 0), (2, 0), (2, 1), (2, 2),
         (3, 2), (3, 1), (3, 0), (4, 0)],
    )
    for role, path in zip(direct_roles, direct_paths):
        _route(states, path)
        outputs[target.outputs[role].role] = path[-2]

    # Quiet structural bridges keep every compiled cell within the default
    # eight-step developmental radius; their index-0 output is zero. Routes
    # that legitimately cross these positions have already claimed their
    # directional tables.
    for pos in ((-3, 2), (-4, 1), (-5, 1)):
        states.setdefault(pos, list(_QUIET))

    inputs = tuple(data_sources + [strobe_source])
    offset = (
        seed_pos[0] - inputs[0][0],
        seed_pos[1] - inputs[0][1])
    return (
        {_translate(pos, offset): tuple(state)
         for pos, state in states.items()},
        tuple(_translate(pos, offset) for pos in inputs),
        {role: _translate(pos, offset)
         for role, pos in outputs.items()},
    )


class _RecordingSim(AsyncLutSim):
    """AsyncLutSim that records only LUT indexes actually addressed."""

    def __init__(self, *args, **kwargs):
        self.visited = {}
        super().__init__(*args, **kwargs)

    def _eval(self):
        wires = self._wirepad
        indexes = (
            ((wires[self._nN] & 0x4) >> 2)
            | ((wires[self._nS] & 0x8) >> 2)
            | ((wires[self._nE] & 0x1) << 2)
            | ((wires[self._nW] & 0x2) << 2)
        )
        for index, value in enumerate(indexes):
            self.visited.setdefault(
                self._cells[index], set()).add(int(value))
        return super()._eval()


def _visited_indexes(grid, inputs, target):
    visited = {pos: set() for pos in grid}
    observation_ticks = (
        int(target.T)
        + int(getattr(target, 'event_max_shift', 12)) + 2)
    for trial in target.trials:
        if getattr(trial, 'input_events', None) is not None:
            raise SynthesisError(
                'periodic combinational synthesis expects tick-lattice trials')
        simulator = _RecordingSim(grid)
        simulator.run_bits(trial.streams, inputs, observation_ticks)
        for pos, indexes in simulator.visited.items():
            visited[pos].update(indexes)
    return visited


def _label_unreachable_bits(grid, visited, label_seed):
    """Use unreachable LUT entries as developmental cell labels."""
    rng = random.Random(int(label_seed))
    labelled = {}
    for pos in sorted(grid):
        state = grid[pos]
        if state == POLARISED_SEED:
            labelled[pos] = state
            continue
        new_state = []
        for mask in state:
            candidate = rng.randrange(1 << 16)
            for index in visited[pos]:
                if (mask >> index) & 1:
                    candidate |= 1 << index
                else:
                    candidate &= ~(1 << index)
            new_state.append(candidate)
        labelled[pos] = tuple(new_state)
    return labelled


def _split_body(genome, chromosome_count, n_ports):
    if chromosome_count < 3:
        raise SynthesisError(
            'spatially mapped synthesis requires at least three chromosomes')
    body_slots = [index for index in range(chromosome_count) if index != 2]
    source = genome.chromosomes[0]
    genes = list(source.genes)
    if len(genes) < len(body_slots):
        raise SynthesisError('compiled body has too few rules to split')
    chromosomes = []
    cursor = 0
    remaining_slots = len(body_slots)
    for index in range(chromosome_count):
        if index == 2:
            chromosomes.append(Chromosome(
                genes=[LutGene() for _ in range(n_ports)],
                split=(0 if n_ports < 2 else n_ports // 2),
                tag=-1, telomere=source.telomere, wiring=True))
            continue
        take = (
            len(genes) - cursor + remaining_slots - 1) // remaining_slots
        chunk = genes[cursor:cursor + take]
        cursor += take
        remaining_slots -= 1
        chromosomes.append(Chromosome(
            genes=chunk,
            split=(0 if len(chunk) < 2 else len(chunk) // 2),
            tag=source.tag, telomere=source.telomere))
    return Genome(
        chromosomes=chromosomes, tag=genome.tag,
        seed_state=genome.seed_state,
        provenance='truth-table-compiler-v1')


def synthesize_combinational_genome(
        target, chromosome_count=3, max_telomere=8, label_attempts=16):
    """Compile a supported target into a verified grown spatial-I/O genome."""
    from substrates.nervous.io_placement import (
        bind_io, growth_seeds, set_spatial_port_positions)
    from .lut import cell_io_tags

    if getattr(target, 'io_placement', 'fixed') != 'spatial_chromosome':
        raise SynthesisError(
            'compiled LUT rescue currently requires spatial_chromosome I/O')
    seeds = tuple(growth_seeds(target, 'spatial_chromosome'))
    if len(seeds) != 1:
        raise SynthesisError('compiled LUT rescue requires one growth seed')
    grid, inputs, outputs = synthesize_grid(target, seed_pos=seeds[0])
    visited = _visited_indexes(grid, inputs, target)

    last_report = None
    for label_seed in range(max(1, int(label_attempts))):
        labelled = _label_unreachable_bits(grid, visited, label_seed)
        body, report = grid_to_genome_lut(
            labelled, seeds, grid_size=target.grid_size, iters=target.iters,
            telomere_cap=max_telomere, repair_rounds=24,
            seed_state=POLARISED_SEED)
        last_report = report
        if not report['exact']:
            continue
        genome = _split_body(
            body, int(chromosome_count),
            int(target.n_inputs) + len(target.outputs))
        assignments = list(inputs) + [
            outputs[terminal.role] for terminal in target.outputs]
        if set_spatial_port_positions(
                genome, sorted(labelled), assignments) != len(assignments):
            continue
        grown = grow_lut(
            genome, seeds=seeds,
            grid_size=target.grid_size, iters=target.iters)
        if grown != labelled:
            continue
        bound = bind_io(
            genome, grown, target, 'spatial_chromosome',
            tags=cell_io_tags(genome, grown))
        if bound is None:
            continue
        return SynthesisResult(
            genome=genome, grid=grown, inputs=inputs, outputs=outputs,
            inverse_report=report, label_seed=label_seed)

    detail = '' if last_report is None else (
        ' (matched %d/%d cells, %d conflicts)' % (
            last_report['matched'], last_report['target'],
            len(last_report['conflicts'])))
    raise SynthesisError(
        'could not encode the phenotype within the configured growth bound'
        + detail)
