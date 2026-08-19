"""Small verified dynamic witnesses for the live asynchronous LUT substrate.

These are ordinary directional LUT arrays reverse-compiled into the same
output-rooted developmental genome evolution uses.  They exist to make simple
feedback motifs reachable after a plateau; simulation and target scoring stay
unchanged.
"""
from __future__ import annotations

from .branched_synthesis import synthesize_branched_grid
from .functions import INPUT_BITS
from .synthesis import SynthesisError


def _mask(predicate):
    return sum(1 << index for index in range(16) if predicate({
        name: (index >> bit) & 1 for name, bit in INPUT_BITS.items()}))


def _cell(**tables):
    return tuple(int(tables.get(direction, 0))
                 for direction in ('N', 'S', 'E', 'W'))


def _oscillator_grid():
    """Kick-set enable latch driving a simultaneous two-node oscillator."""
    enable = _mask(lambda b: b['W'] or b['N'])
    enable_relay = _mask(lambda b: b['S'])
    phase_a = _mask(lambda b: b['W'] and not b['N'])
    phase_b = _mask(lambda b: b['W'] and not b['S'])
    read_a = _mask(lambda b: b['W'])
    return {
        # E/E2: input-set, cross-coupled hold.  The two east outputs also
        # distribute the latched enable to both oscillator phases.
        (1, 0): _cell(N=enable, E=enable),
        (1, 1): _cell(S=enable_relay, E=enable_relay),
        # A/B both toggle on the same wave: 00 -> 11 -> 00.  A is tapped east.
        (2, 0): _cell(N=phase_a, E=phase_a),
        (2, 1): _cell(S=phase_b),
        (3, 0): _cell(E=read_a),
    }, ((0, 0),), {'Q': (3, 0)}


def _pattern_1000_grid():
    """One kicked pulse circulating around a four-cell directed ring."""
    or_w_n = _mask(lambda b: b['W'] or b['N'])
    relay_w = _mask(lambda b: b['W'])
    relay_s = _mask(lambda b: b['S'])
    relay_e = _mask(lambda b: b['E'])
    return {
        (1, 0): _cell(E=or_w_n),
        (2, 0): _cell(N=relay_w),
        (2, 1): _cell(W=relay_s, E=relay_s),
        (1, 1): _cell(S=relay_e),
        (3, 1): _cell(E=relay_w),
    }, ((0, 0),), {'Q': (3, 1)}


def _echo_grid(delay=3):
    relay_w = _mask(lambda b: b['W'])
    grid = {
        (step, 0): _cell(E=relay_w)
        for step in range(1, int(delay) + 1)}
    return grid, ((0, 0),), {'Q': (int(delay), 0)}


def _pair_gap2_grid():
    relay_s = _mask(lambda b: b['S'])
    relay_w = _mask(lambda b: b['W'])
    coincidence = _mask(lambda b: b['W'] and b['N'])
    return {
        (0, 1): _cell(E=relay_s),
        (1, 1): _cell(S=relay_w),
        (1, 0): _cell(E=coincidence),
    }, ((0, 0),), {'Q': (1, 0)}


def _sequence_gap3_grid():
    relay_w = _mask(lambda b: b['W'])
    coincidence = _mask(lambda b: b['W'] and b['N'])
    return {
        (1, 0): _cell(E=relay_w),
        (2, 0): _cell(E=relay_w),
        (3, 0): _cell(E=relay_w),
        (4, 0): _cell(E=coincidence),
    }, ((0, 0), (4, 1)), {'Q': (4, 0)}


def _veto_grid():
    veto = _mask(lambda b: b['W'] and not b['N'])
    return {(1, 0): _cell(E=veto)}, ((0, 0), (1, 1)), {'Q': (1, 0)}


def _burst_x3_grid():
    states = {}

    def route(path):
        for previous, cell, following in zip(path, path[1:], path[2:]):
            incoming = next(direction for direction, (dx, dy) in {
                'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)
            }.items() if (cell[0] + dx, cell[1] + dy) == previous)
            outgoing = next(direction for direction, (dx, dy) in {
                'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)
            }.items() if (cell[0] + dx, cell[1] + dy) == following)
            state = states.setdefault(cell, [0, 0, 0, 0])
            state[('N', 'S', 'E', 'W').index(outgoing)] = _mask(
                lambda b, incoming=incoming: b[incoming])

    pad, root = (0, 0), (2, 0)
    route((pad, (1, 0), root))
    route((pad, (0, 1), (1, 1), (2, 1), root))
    route((pad, (0, -1), (0, -2), (1, -2), (2, -2), (2, -1), root))
    burst_or = _mask(lambda b: b['W'] or b['N'] or b['S'])
    states[root] = [0, 0, burst_or, 0]
    return {cell: tuple(state) for cell, state in states.items()}, (pad,), {'Q': root}


def _gated_d_latch_grid():
    """Transparent D latch with a two-cell level-feedback store."""
    latch = _mask(lambda b: b['W'] if b['N'] else b['E'])
    relay_w = _mask(lambda b: b['W'])
    return {
        # D enters from W and Enable from N.  With Enable low, the state cell
        # reads the value returned by the east feedback cell.
        (1, 0): _cell(E=latch),
        # Feed the stored value back west and expose the same level east.
        (2, 0): _cell(W=relay_w, E=relay_w),
        (3, 0): _cell(E=relay_w),
    }, ((0, 0), (1, 1)), {'Q': (3, 0)}


def _sr_latch_grid():
    """Set/reset level store using the same two-cell feedback primitive."""
    latch = _mask(
        lambda b: b['W'] or (b['N'] and not b['S']))
    feedback = _mask(lambda b: b['S'])
    relay_w = _mask(lambda b: b['W'])
    return {
        # Set from W; returned state from N; Reset from S dominates the hold.
        (1, 0): _cell(N=latch, E=latch),
        (1, 1): _cell(S=feedback),
        (2, 0): _cell(E=relay_w),
    }, ((0, 0), (1, -1)), {'Q': (2, 0)}


def synthesize_branched_dynamic(
        target, chromosome_count=2, max_telomere=32,
        function_families=None):
    name = str(getattr(target, 'name', ''))
    if name in ('Coincidence (2-in)', 'Temporal XOR (2-in)'):
        from types import SimpleNamespace
        rows = tuple(
            ((a, b), ((a & b) if name.startswith('Coincidence')
                      else (a ^ b),))
            for a in (0, 1) for b in (0, 1))
        proxy = SimpleNamespace(
            combinational_cases=rows, temporal_logic_cases=(),
            combinational_data_inputs=2, temporal_logic_data_inputs=0,
            combinational_strobe=False, n_inputs=2,
            outputs=target.outputs)
        from .synthesis import synthesize_grid
        grid, inputs, outputs = synthesize_grid(proxy, seed_pos=(0, 0))
        return synthesize_branched_grid(
            target, grid, inputs, outputs,
            chromosome_count=chromosome_count, max_telomere=max_telomere,
            function_families=function_families)
    if name.startswith('Echo (delay 3)'):
        phenotype = _echo_grid(3)
    elif name.startswith('Pair detector (gap 2)'):
        phenotype = _pair_gap2_grid()
    elif name.startswith('Sequence A->B'):
        phenotype = _sequence_gap3_grid()
    elif name.startswith('Veto gate'):
        phenotype = _veto_grid()
    elif name.startswith('Burst x3'):
        phenotype = _burst_x3_grid()
    elif name.startswith('Gated D latch'):
        phenotype = _gated_d_latch_grid()
    elif name.startswith('SR latch'):
        phenotype = _sr_latch_grid()
    elif name.startswith('Oscillator'):
        phenotype = _oscillator_grid()
    elif name.startswith('Pattern (1000)'):
        phenotype = _pattern_1000_grid()
    else:
        raise SynthesisError('no compiled dynamic witness for %s' % name)
    grid, inputs, outputs = phenotype
    return synthesize_branched_grid(
        target, grid, inputs, outputs,
        chromosome_count=chromosome_count, max_telomere=max_telomere,
        function_families=function_families)
