"""Construct timing-balanced Boolean motifs on the real tri-tile honeycomb."""
from __future__ import annotations

from runtime.config import nv_run_config
from substrates.nervous.branched import materialise_pads, tau_of
from substrates.nervous.hexgrid import CANONICAL_STATES, ROUTING_HEX, hex_dirs
from substrates.nervous.nervous import interpret_nervous
from substrates.nervous.scoring import score_contract
from substrates.nervous.targets import coincident_temporal_target
from substrates.nervous.temporal import trace_fixed_outputs
from substrates.nervous.tritile import TRI_DIRS, pack_channels
from substrates.snn.targets import get_target


def _direction(source, destination):
    return next(direction for direction, cell in hex_dirs(*source).items()
                if cell == destination)


def _config(e1, e2, inhibit=None, operation='and'):
    return next(index for index in CANONICAL_STATES
                if (ROUTING_HEX[index] == (e1, e2, inhibit, operation)
                    or ROUTING_HEX[index] == (
                        e2, e1, inhibit, operation)))


def find_theta():
    """Three internally-disjoint paths with a balanced, free XOR output."""
    # Smallest motif found by exhaustive search that cascades by a
    # pure translation without overlapping its predecessor (apart from the
    # intended bridge/fanout tile).
    return (
        ((-5, -5), (-5, -4), (-4, -4), (-3, -4)),
        ((-5, -5), (-4, -5), (-3, -5), (-3, -4)),
        ((-5, -5), (-6, -5), (-6, -6), (-5, -6), (-4, -6),
         (-3, -6), (-2, -6), (-2, -5), (-1, -5), (-1, -4),
         (-2, -4), (-3, -4)),
        (-1, -6), 1, 6)


def xor2_theta():
    """Return (grid, pads, output) for a hand-routed temporal XOR2."""
    (path_a, path_b, path_logic, output_neighbour,
     input_index, logic_index) = find_theta()
    left, right = path_a[0], path_a[-1]
    pad_a, pad_b = path_a[input_index], path_b[input_index]
    veto = path_logic[logic_index]
    channels = {cell: {direction: 0 for direction in TRI_DIRS}
                for path in (path_a, path_b, path_logic)
                for cell in path}
    channels.setdefault(output_neighbour, {direction: 0 for direction in TRI_DIRS})

    def route(path, source_index, destination_index):
        step = 1 if destination_index > source_index else -1
        for index in range(source_index + step, destination_index, step):
            cell = path[index]
            source = path[index - step]
            destination = path[index + step]
            incoming = _direction(cell, source)
            outgoing = _direction(cell, destination)
            channels[cell][outgoing] = _config(incoming, incoming)

    route(path_a, input_index, 0)
    route(path_a, input_index, len(path_a) - 1)
    route(path_b, input_index, 0)
    route(path_b, input_index, len(path_b) - 1)

    a_left = _direction(left, path_a[1])
    b_left = _direction(left, path_b[1])
    logic_left = _direction(left, path_logic[1])
    channels[left][logic_left] = _config(a_left, b_left, operation='and')
    a_right = _direction(right, path_a[-2])
    b_right = _direction(right, path_b[-2])
    logic_right = _direction(right, path_logic[-2])
    channels[right][logic_right] = _config(
        a_right, b_right, operation='or')

    route(path_logic, 0, logic_index)
    route(path_logic, len(path_logic) - 1, logic_index)
    from_left = _direction(veto, path_logic[logic_index - 1])
    from_right = _direction(veto, path_logic[logic_index + 1])
    outward = _direction(veto, output_neighbour)
    channels[veto][outward] = _config(
        from_right, from_right, inhibit=from_left, operation='and')

    grid = {
        cell: pack_channels(*(mapping[direction] for direction in TRI_DIRS))
        for cell, mapping in channels.items()}
    grid = materialise_pads(grid, (pad_a, pad_b))
    return grid, (pad_a, pad_b), veto


def _shift_path(path, shift):
    return tuple((x + shift[0], y + shift[1]) for x, y in path)


def _xor_module(shift=(0, 0), feeder_neighbours=(None, None)):
    """Channel map for one theta XOR; feeders replace its external pads."""
    (base_a, base_b, base_logic, base_out,
     input_index, logic_index) = find_theta()
    path_a = _shift_path(base_a, shift)
    path_b = _shift_path(base_b, shift)
    path_logic = _shift_path(base_logic, shift)
    output_neighbour = (base_out[0] + shift[0], base_out[1] + shift[1])
    left, right = path_a[0], path_a[-1]
    pads = (path_a[input_index], path_b[input_index])
    veto = path_logic[logic_index]
    channels = {cell: {direction: 0 for direction in TRI_DIRS}
                for path in (path_a, path_b, path_logic)
                for cell in path}
    channels.setdefault(output_neighbour, {direction: 0 for direction in TRI_DIRS})

    def route(path, source_index, destination_index):
        step = 1 if destination_index > source_index else -1
        for index in range(source_index + step, destination_index, step):
            cell = path[index]
            source = path[index - step]
            destination = path[index + step]
            incoming = _direction(cell, source)
            outgoing = _direction(cell, destination)
            channels[cell][outgoing] = _config(incoming, incoming)

    for path in (path_a, path_b):
        route(path, input_index, 0)
        route(path, input_index, len(path) - 1)
    for pad, feeder, path in zip(pads, feeder_neighbours, (path_a, path_b)):
        if feeder is None:
            continue
        incoming = _direction(pad, feeder)
        toward_left = _direction(pad, path[input_index - 1])
        toward_right = _direction(pad, path[input_index + 1])
        channels[pad][toward_left] = _config(incoming, incoming)
        channels[pad][toward_right] = _config(incoming, incoming)

    a_left = _direction(left, path_a[1])
    b_left = _direction(left, path_b[1])
    logic_left = _direction(left, path_logic[1])
    channels[left][logic_left] = _config(a_left, b_left, operation='and')
    a_right = _direction(right, path_a[-2])
    b_right = _direction(right, path_b[-2])
    logic_right = _direction(right, path_logic[-2])
    channels[right][logic_right] = _config(a_right, b_right, operation='or')
    route(path_logic, 0, logic_index)
    route(path_logic, len(path_logic) - 1, logic_index)
    from_left = _direction(veto, path_logic[logic_index - 1])
    from_right = _direction(veto, path_logic[logic_index + 1])
    outward = _direction(veto, output_neighbour)
    channels[veto][outward] = _config(
        from_right, from_right, inhibit=from_left, operation='and')
    return channels, pads, veto, output_neighbour, left


def xor3_chain():
    """Two physical theta motifs cascaded into a timing-balanced XOR3."""
    first, pads_ab, xor_ab, bridge, and_ab = _xor_module()
    template = find_theta()
    base_a, base_b, _base_logic, _base_out, input_index, logic_index = template
    base_pad_a = base_a[input_index]
    base_pad_b = base_b[input_index]
    shift = (bridge[0] - base_pad_a[0], bridge[1] - base_pad_a[1])
    base_cin_feeder = next(iter(
        set(hex_dirs(*base_pad_b).values()) - set(base_b)))
    cin_feeder = (base_cin_feeder[0] + shift[0],
                  base_cin_feeder[1] + shift[1])
    second, internal_pads, sum_node, _sum_bridge, and_xc = _xor_module(
        shift=shift, feeder_neighbours=(xor_ab, cin_feeder))
    channels = first
    for cell, mapping in second.items():
        own = channels.setdefault(
            cell, {direction: 0 for direction in TRI_DIRS})
        for direction, config in mapping.items():
            if own[direction] and config and own[direction] != config:
                raise ValueError('cascaded XOR motifs conflict')
            own[direction] = own[direction] or config
    # XOR(A,B) has ``input_index + logic_index`` node delays before it reaches
    # the second motif's
    # fanout. Delay Cin by the same amount so every AND/OR coincidence in the
    # second theta sees simultaneous edges.
    occupied = set(channels) | {internal_pads[1]}
    forbidden = set(pads_ab) | set(internal_pads) | {xor_ab, sum_node}
    delayed = None

    def extend(path, remaining):
        nonlocal delayed
        if delayed is not None:
            return
        if remaining == 0:
            candidate = tuple(reversed(path))  # actual Cin pad -> feeder
            if candidate[0] in occupied:
                return
            for index in range(1, len(candidate)):
                cell, source = candidate[index], candidate[index - 1]
                destination = (internal_pads[1]
                               if index == len(candidate) - 1
                               else candidate[index + 1])
                outgoing = _direction(cell, destination)
                if channels.get(cell, {}).get(outgoing, 0):
                    return
            delayed = candidate
            return
        for neighbour in sorted(hex_dirs(*path[-1]).values()):
            if neighbour in path or neighbour in forbidden:
                continue
            extend(path + [neighbour], remaining - 1)
            if delayed is not None:
                return

    extend([cin_feeder], input_index + logic_index)
    if delayed is None:
        raise ValueError('could not route the Cin delay line')
    for cell in delayed:
        channels.setdefault(cell, {direction: 0 for direction in TRI_DIRS})
    for index in range(1, len(delayed)):
        cell, source = delayed[index], delayed[index - 1]
        destination = (internal_pads[1]
                       if index == len(delayed) - 1
                       else delayed[index + 1])
        incoming = _direction(cell, source)
        outgoing = _direction(cell, destination)
        channels[cell][outgoing] = _config(incoming, incoming)

    # Carry = (A AND B) OR (Cin AND XOR(A,B)).  Both conjunctions already
    # exist as the left endpoints of the two theta motifs.  Route the early
    # A&B signal through nine buffers so it reaches the first second-motif
    # logic-path tile alongside the later Cin&XOR signal.
    path_logic_second = _shift_path(template[2], shift)
    carry_gate = path_logic_second[1]
    next_sum_path = path_logic_second[2]
    arrival_options = sorted(
        set(hex_dirs(*carry_gate).values()) - {and_xc, next_sum_path})
    if len(arrival_options) != 1:
        raise ValueError('carry gate has no unique third input')
    carry_arrival = arrival_options[0]
    channels.setdefault(
        carry_arrival, {direction: 0 for direction in TRI_DIRS})
    sum_cells = set(channels)
    ab_path = None
    carry_path = None

    and_config = next(
        config for config in channels[and_ab].values() if config)

    def ab_writes_for(path):
        writes = {}
        for index, cell in enumerate(path):
            if index == len(path) - 1:
                source, destination = path[index - 1], carry_gate
                incoming = _direction(cell, source)
                outgoing = _direction(cell, destination)
                desired = _config(incoming, incoming)
            elif index == 0:
                outgoing = _direction(cell, path[1])
                desired = and_config
            else:
                incoming = _direction(cell, path[index - 1])
                outgoing = _direction(cell, path[index + 1])
                desired = _config(incoming, incoming)
            present = channels.get(
                cell, {direction: 0 for direction in TRI_DIRS})[outgoing]
            if present not in (0, desired):
                return None
            writes[(cell, outgoing)] = desired
        return writes

    def carry_for(candidate_ab, ab_writes):
        """Five-node return compatible with the proposed A&B channel writes."""
        found = None
        previous_ab = candidate_ab[-2]

        def visit(path):
            nonlocal found
            if found is not None:
                return
            if len(path) == 5:
                if path[-1] in sum_cells or path[-1] in set(candidate_ab):
                    return
                for index, cell in enumerate(path):
                    previous = carry_gate if index == 0 else path[index - 1]
                    outgoing = (_direction(cell, previous)
                                if index == len(path) - 1 else
                                _direction(cell, path[index + 1]))
                    incoming = _direction(cell, previous)
                    desired = _config(incoming, incoming)
                    present = ab_writes.get(
                        (cell, outgoing),
                        channels.get(
                            cell,
                            {direction: 0 for direction in TRI_DIRS})[outgoing])
                    if present not in (0, desired):
                        return
                found = tuple(path)
                return
            for neighbour in sorted(hex_dirs(*path[-1]).values()):
                if neighbour in path or neighbour in set(pads_ab) \
                        or neighbour in {sum_node, xor_ab, carry_gate}:
                    continue
                visit(path + [neighbour])

        first_options = sorted(
            set(hex_dirs(*carry_arrival).values())
            - {carry_gate, previous_ab})
        for first in first_options:
            visit([carry_arrival, first])
            if found is not None:
                break
        return found

    def route_ab(path, remaining):
        nonlocal ab_path, carry_path
        if ab_path is not None:
            return
        if remaining == 0:
            if (path[-1] == carry_arrival
                    and any(cell not in sum_cells for cell in path[1:-1])):
                candidate = tuple(path)
                writes = ab_writes_for(candidate)
                private = [cell for cell in candidate[1:-1]
                           if cell not in sum_cells]
                excess = (len(candidate) - 1) - 8
                if writes is not None and len(private) * 0.59 >= excess - 0.5:
                    tail = carry_for(candidate, writes)
                    if tail is not None:
                        ab_path = candidate
                        carry_path = tail
            return
        for neighbour in sorted(hex_dirs(*path[-1]).values()):
            if neighbour in path or neighbour in set(pads_ab) \
                    or neighbour in {xor_ab, sum_node, carry_gate}:
                continue
            # Leave exactly enough steps to reach the fixed arrival tile.
            from substrates.nervous.hexgrid import honeycomb_distance
            if honeycomb_distance(neighbour, carry_arrival) > remaining - 1:
                continue
            route_ab(path + [neighbour], remaining - 1)
            if ab_path is not None:
                return

    for edge_count in range(8, 17):
        route_ab([and_ab], edge_count)
        if ab_path is not None:
            break
    if ab_path is None:
        raise ValueError('could not jointly route A&B and Carry')
    private_ab = [cell for cell in ab_path[1:-1] if cell not in sum_cells]
    fast_ab = min(
        len(private_ab),
        max(0, round(((len(ab_path) - 1) - 8) / 0.59)))
    tau_indices = {cell: 6 for cell in private_ab[:fast_ab]}
    for index, cell in enumerate(ab_path):
        channels.setdefault(cell, {direction: 0 for direction in TRI_DIRS})
        if index == len(ab_path) - 1:
            source, destination = ab_path[index - 1], carry_gate
            incoming = _direction(cell, source)
            outgoing = _direction(cell, destination)
            desired = _config(incoming, incoming)
        elif index == 0:
            outgoing = _direction(cell, ab_path[1])
            desired = and_config
        else:
            incoming = _direction(cell, ab_path[index - 1])
            outgoing = _direction(cell, ab_path[index + 1])
            desired = _config(incoming, incoming)
        if channels[cell][outgoing] not in (0, desired):
            raise ValueError('A&B delay route conflicts with a live channel')
        channels[cell][outgoing] = desired

    from_xc = _direction(carry_gate, and_xc)
    from_ab = _direction(carry_gate, carry_arrival)
    carry_config = _config(from_xc, from_ab, operation='or')
    if channels[carry_gate][from_ab] not in (0, carry_config):
        raise ValueError('Carry OR conflicts with the Sum route')
    channels[carry_gate][from_ab] = carry_config

    # The jointly selected five-node return may cross occupied tiles through
    # independent spare channels, but its terminal is private and cannot sever
    # either logic path when output-root sink semantics are applied.
    carry_path = list(carry_path)
    for cell in carry_path:
        channels.setdefault(cell, {direction: 0 for direction in TRI_DIRS})
    incoming_carry = _direction(carry_arrival, carry_gate)
    outgoing_carry = _direction(carry_arrival, carry_path[1])
    channels[carry_arrival][outgoing_carry] = _config(
        incoming_carry, incoming_carry)
    for index in range(1, len(carry_path)):
        cell, previous = carry_path[index], carry_path[index - 1]
        incoming = _direction(cell, previous)
        outgoing = (incoming if index == len(carry_path) - 1 else
                    _direction(cell, carry_path[index + 1]))
        channels[cell][outgoing] = _config(incoming, incoming)
    carry_node = carry_path[-1]
    grid = {
        cell: pack_channels(*(mapping[direction] for direction in TRI_DIRS))
        for cell, mapping in channels.items()}
    pads = (pads_ab[0], pads_ab[1], delayed[0])
    grid = materialise_pads(grid, pads)
    return grid, pads, sum_node, {
        'xor_ab': xor_ab, 'and_ab': and_ab, 'and_xc': and_xc,
        'internal_pads': internal_pads, 'bridge': bridge,
        'carry': carry_node, 'carry_gate': carry_gate,
        'tau_indices': tau_indices, 'ab_path': ab_path,
        'carry_path': tuple(carry_path)}


def main():
    grid, pads, output = xor2_theta()
    target = coincident_temporal_target(get_target('XOR'), schedules=12)
    target.pulse_config = nv_run_config().pulse
    role = target.outputs[0].role
    routing, _inputs, _outputs = interpret_nervous(grid, target, arch='tri3')
    traces = trace_fixed_outputs(
        grid, routing, list(pads), {role: output}, target, arch='tri3',
        source_nodes=set(pads), sink_nodes={output})
    score, cases, alignment = score_contract(traces, target)
    print('theta XOR2', 'tiles', len(grid), 'pads', pads, 'output', output,
          'score', score, 'worst', min(cases), 'alignment', alignment,
          'overflow', traces.overflow)
    for cell in sorted(grid):
        print(cell, grid[cell])

    grid, pads, sum_output, metadata = xor3_chain()
    target = coincident_temporal_target(
        get_target('Full adder'), schedules=12)
    target.pulse_config = nv_run_config().pulse
    outputs = {
        terminal.role: (sum_output if terminal.role == 'Sum'
                        else metadata['carry'])
        for terminal in target.outputs}
    routing, _inputs, _outputs = interpret_nervous(grid, target, arch='tri3')
    base_tau = target.pulse_config.analog_tau_leak
    taus = {cell: tau_of(index, base_tau)
            for cell, index in metadata['tau_indices'].items()}
    traces = trace_fixed_outputs(
        grid, routing, list(pads), outputs, target, arch='tri3',
        source_nodes=set(pads), sink_nodes=set(outputs.values()), taus=taus)
    score, cases, alignment = score_contract(traces, target)
    print('theta Full Adder', 'tiles', len(grid), 'pads', pads,
          'outputs', outputs,
          'score', score, 'worst', min(cases), 'alignment', alignment,
          'overflow', traces.overflow, 'metadata', metadata)


if __name__ == '__main__':
    main()
