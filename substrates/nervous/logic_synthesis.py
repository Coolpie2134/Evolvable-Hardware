"""Verified temporal-logic witnesses for the paper tri-tile substrate."""
from __future__ import annotations

from .branched_synthesis import SynthesisError, synthesize_branched_grid


def _rows(target):
    return tuple(
        (tuple(map(int, inputs)), tuple(map(int, outputs)))
        for inputs, outputs in (
            getattr(target, 'temporal_logic_cases', ())
            or getattr(target, 'combinational_cases', ())
            or ()))


def _columns(target):
    rows = sorted(_rows(target))
    return rows, tuple(str(output.role) for output in target.outputs)


def _matches(rows, function):
    return tuple(outputs[0] for _inputs, outputs in rows) == tuple(
        int(function(*inputs)) for inputs, _outputs in rows)


def _tap(grid, pads, source, extra_delay=0):
    """Add a terminal relay without turning an internal logic tile into a sink."""
    from tools.handbuild_tri_logic import _config, _direction
    from .hexgrid import hex_dirs
    from .tritile import TRI_DIRS, channel_configs, pack_channels

    channels = list(channel_configs(grid[source]))
    live = next((config for config in channels if config), 0)
    if not live:
        raise SynthesisError('cannot tap a dead tri-tile')
    occupied = set(grid) | set(pads)
    for index, direction in enumerate(TRI_DIRS):
        first = hex_dirs(*source)[direction]
        if channels[index] or first in occupied:
            continue
        path = None

        def extend(candidate, remaining):
            nonlocal path
            if path is not None:
                return
            if remaining == 0:
                path = tuple(candidate)
                return
            for neighbour in sorted(hex_dirs(*candidate[-1]).values()):
                if neighbour in occupied or neighbour in candidate:
                    continue
                extend(candidate + [neighbour], remaining - 1)
                if path is not None:
                    return

        extend([first], max(0, int(extra_delay)))
        if path is None:
            continue
        channels[index] = live
        grid[source] = pack_channels(*channels)
        previous = source
        for step, cell in enumerate(path):
            incoming = _direction(cell, previous)
            outgoing = (incoming if step == len(path) - 1 else
                        _direction(cell, path[step + 1]))
            cell_channels = [0, 0, 0]
            cell_channels[TRI_DIRS.index(outgoing)] = _config(incoming, incoming)
            grid[cell] = pack_channels(*cell_channels)
            previous = cell
        return path[-1]
    raise SynthesisError('logic tile has no free terminal direction')


def _inverted_gate_grid(operation):
    """Strobed NAND/NOR as a balanced logic pulse inhibiting a strobe."""
    from tools.handbuild_tri_logic import _config, _direction
    from .tritile import TRI_DIRS, pack_channels

    pad_a, pad_b = (0, 0), (2, 0)
    logic, root = (1, 0), (1, -1)
    strobe_delay, strobe_pad = (2, -1), (2, -2)

    logic_channels = {direction: 0 for direction in TRI_DIRS}
    from_a = _direction(logic, pad_a)
    from_b = _direction(logic, pad_b)
    toward_root = _direction(logic, root)
    logic_channels[toward_root] = _config(
        from_a, from_b, operation=operation)

    delay_channels = {direction: 0 for direction in TRI_DIRS}
    from_strobe = _direction(strobe_delay, strobe_pad)
    toward_root = _direction(strobe_delay, root)
    delay_channels[toward_root] = _config(from_strobe, from_strobe)

    root_channels = {direction: 0 for direction in TRI_DIRS}
    excitation = _direction(root, strobe_delay)
    inhibition = _direction(root, logic)
    outward = next(direction for direction in TRI_DIRS
                   if direction not in (excitation, inhibition))
    root_channels[outward] = _config(
        excitation, excitation, inhibit=inhibition)

    grid = {
        logic: pack_channels(*(
            logic_channels[direction] for direction in TRI_DIRS)),
        strobe_delay: pack_channels(*(
            delay_channels[direction] for direction in TRI_DIRS)),
        root: pack_channels(*(
            root_channels[direction] for direction in TRI_DIRS)),
    }
    return grid, (pad_a, pad_b, strobe_pad), root


def _xnor_grid():
    """Strobed XNOR by timing a strobe against the proven theta XOR."""
    from tools.handbuild_tri_logic import (
        _config, _direction, xor2_theta)
    from .tritile import TRI_DIRS, pack_channels

    grid, data_pads, xor_node = xor2_theta()
    xor_bridge, root = (-1, -6), (0, -6)
    # XOR needs seven analog stages.  Its bridge adds the eighth; match that
    # with an eight-relay strobe route before the final veto/output stage.
    strobe_path = (
        (5, -10), (4, -10), (4, -9), (3, -9), (3, -8),
        (2, -8), (2, -7), (1, -7), (1, -6))

    channels = [0, 0, 0]
    incoming = _direction(xor_bridge, xor_node)
    outgoing = _direction(xor_bridge, root)
    channels[TRI_DIRS.index(outgoing)] = _config(incoming, incoming)
    grid[xor_bridge] = pack_channels(*channels)

    for index, cell in enumerate(strobe_path[1:], 1):
        channels = [0, 0, 0]
        incoming = _direction(cell, strobe_path[index - 1])
        outgoing = (_direction(cell, root)
                    if index == len(strobe_path) - 1 else
                    _direction(cell, strobe_path[index + 1]))
        channels[TRI_DIRS.index(outgoing)] = _config(incoming, incoming)
        grid[cell] = pack_channels(*channels)

    channels = [0, 0, 0]
    excitation = _direction(root, strobe_path[-1])
    inhibition = _direction(root, xor_bridge)
    outward = next(direction for direction in TRI_DIRS
                   if direction not in (excitation, inhibition))
    channels[TRI_DIRS.index(outward)] = _config(
        excitation, excitation, inhibit=inhibition)
    grid[root] = pack_channels(*channels)
    return grid, data_pads + (strobe_path[0],), root


def _mux_grid():
    """Balanced ``A unless S else B`` veto/AND branches into one OR."""
    from tools.handbuild_tri_logic import _config, _direction
    from .tritile import TRI_DIRS, pack_channels

    pad_a, pad_b, pad_s = (-8, 0), (8, 0), (0, -2)
    gate_a, gate_b, root = (-3, 0), (3, 0), (0, 0)
    paths = (
        (pad_a, (-7, 0), (-6, 0), (-5, 0), (-4, 0), gate_a),
        (pad_b, (7, 0), (6, 0), (5, 0), (4, 0), gate_b),
        (pad_s, (0, -1), (-1, -1), (-2, -1), (-3, -1), gate_a),
        (pad_s, (0, -1), (1, -1), (2, -1), (3, -1), gate_b),
        (gate_a, (-2, 0), (-1, 0), root),
        (gate_b, (2, 0), (1, 0), root),
    )
    channels = {}

    def mapping(cell):
        return channels.setdefault(
            cell, {direction: 0 for direction in TRI_DIRS})

    for path in paths:
        for index, cell in enumerate(path[1:-1], 1):
            incoming = _direction(cell, path[index - 1])
            outgoing = _direction(cell, path[index + 1])
            mapping(cell)[outgoing] = _config(incoming, incoming)

    excitation = _direction(gate_a, (-4, 0))
    inhibition = _direction(gate_a, (-3, -1))
    outgoing = _direction(gate_a, (-2, 0))
    mapping(gate_a)[outgoing] = _config(
        excitation, excitation, inhibit=inhibition)

    from_b = _direction(gate_b, (4, 0))
    from_s = _direction(gate_b, (3, -1))
    outgoing = _direction(gate_b, (2, 0))
    mapping(gate_b)[outgoing] = _config(from_b, from_s)

    from_a = _direction(root, (-1, 0))
    from_b = _direction(root, (1, 0))
    outward = next(direction for direction in TRI_DIRS
                   if direction not in (from_a, from_b))
    mapping(root)[outward] = _config(
        from_a, from_b, operation='or')

    grid = {
        cell: pack_channels(*(
            cell_channels[direction] for direction in TRI_DIRS))
        for cell, cell_channels in channels.items()}
    return grid, (pad_a, pad_b, pad_s), root


def synthesize_branched_logic(target, chromosome_count=2, max_telomere=24):
    """Compile gates already demonstrated by the physical theta motifs."""
    rows, roles = _columns(target)
    name = str(getattr(target, 'name', ''))
    if name in ('Coincidence (2-in)', 'Temporal XOR (2-in)'):
        from tools.handbuild_tri_logic import xor2_theta, find_theta
        grid, pads, xor_node = xor2_theta()
        output = (find_theta()[0][0]
                  if name == 'Coincidence (2-in)' else xor_node)
        return synthesize_branched_grid(
            target, grid, pads, {roles[0]: output},
            chromosome_count=chromosome_count, max_telomere=max_telomere)
    if not rows or len(roles) not in (1, 2):
        raise SynthesisError('no tri-tile logic witness for this target')

    # The derivation module constructs actual routing tiles; this module only
    # selects which proven node(s) are exposed as target roots.
    from tools.handbuild_tri_logic import xor2_theta, xor3_chain, find_theta

    if len(rows[0][0]) == 2:
        # The temporal wrapper adds a third, always-present case strobe for
        # inverted truth tables.  Delay it by the same one physical stage as
        # the positive AND/OR signal, then use that signal as inhibition.
        if _matches(rows, lambda a, b: not (a & b)):
            grid, pads, output = _inverted_gate_grid('and')
            return synthesize_branched_grid(
                target, grid, pads, {roles[0]: output},
                chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        if _matches(rows, lambda a, b: not (a | b)):
            grid, pads, output = _inverted_gate_grid('or')
            return synthesize_branched_grid(
                target, grid, pads, {roles[0]: output},
                chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        if _matches(rows, lambda a, b: not (a ^ b)):
            grid, pads, output = _xnor_grid()
            return synthesize_branched_grid(
                target, grid, pads, {roles[0]: output},
                chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        grid, pads, xor_node = xor2_theta()
        path_a, _path_b, path_logic, _out, _input_index, _logic_index = find_theta()
        and_node, or_node = path_a[0], path_a[-1]
        if len(roles) == 2:
            expected = tuple(outputs for _inputs, outputs in rows)
            xor_col = tuple(a ^ b for (a, b), _outputs in rows)
            and_col = tuple(a & b for (a, b), _outputs in rows)
            columns = [tuple(outputs[i] for _inputs, outputs in rows)
                       for i in range(2)]
            if sorted(columns) != sorted((xor_col, and_col)):
                raise SynthesisError('two-output target is not a Half Adder')
            carry_tap = _tap(grid, pads, path_logic[1], extra_delay=4)
            outputs = {
                roles[index]: (xor_node if column == xor_col else carry_tap)
                for index, column in enumerate(columns)}
        elif _matches(rows, lambda a, b: a ^ b):
            outputs = {roles[0]: xor_node}
        elif _matches(rows, lambda a, b: a & b):
            outputs = {roles[0]: and_node}
        elif _matches(rows, lambda a, b: a | b):
            outputs = {roles[0]: or_node}
        else:
            raise SynthesisError('two-input function has no theta witness')
        return synthesize_branched_grid(
            target, grid, pads, outputs,
            chromosome_count=chromosome_count, max_telomere=max_telomere)

    if len(rows[0][0]) == 3 and len(roles) == 1:
        if _matches(rows, lambda a, b, select: b if select else a):
            grid, pads, output = _mux_grid()
            return synthesize_branched_grid(
                target, grid, pads, {roles[0]: output},
                chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        grid, pads, sum_node, metadata = xor3_chain()
        if _matches(rows, lambda a, b, c: a ^ b ^ c):
            # The compact exact Full Adder already partitions this 43-tile
            # shared cone over two ordinary arms.  Keep its unused Carry pin as
            # an unscored physical sink so the Sum timing and development stay
            # exactly the verified circuit rather than forcing the whole cone
            # through one arm.
            from types import SimpleNamespace
            from .branched_synthesis import (
                FULL_ADDER_CASES, synthesize_branched_full_adder)
            proxy = SimpleNamespace(
                outputs=(SimpleNamespace(role=roles[0]),
                         SimpleNamespace(role='__carry_helper__')),
                temporal_logic_cases=FULL_ADDER_CASES)
            return synthesize_branched_full_adder(
                proxy, chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        elif _matches(rows, lambda a, b, c: a + b + c >= 2):
            from types import SimpleNamespace
            from .branched_synthesis import (
                FULL_ADDER_CASES, synthesize_branched_full_adder)
            proxy = SimpleNamespace(
                outputs=(SimpleNamespace(role='__sum_helper__'),
                         SimpleNamespace(role=roles[0])),
                temporal_logic_cases=FULL_ADDER_CASES)
            return synthesize_branched_full_adder(
                proxy, chromosome_count=chromosome_count,
                max_telomere=max_telomere)
        else:
            raise SynthesisError('three-input function has no theta witness')
        return synthesize_branched_grid(
            target, grid, pads, {roles[0]: output},
            chromosome_count=chromosome_count, max_telomere=max_telomere,
            tau_indices=metadata['tau_indices'])
    raise SynthesisError('no tri-tile logic witness for this arity')
