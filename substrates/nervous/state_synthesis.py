"""Verified small feedback witnesses for the paper tri-tile substrate."""
from __future__ import annotations

from .branched_synthesis import SynthesisError, synthesize_branched_grid
from .hexgrid import hex_dirs
from .tritile import TRI_DIRS, pack_channels


def _direction(source, destination):
    return next(direction for direction, cell in hex_dirs(*source).items()
                if cell == destination)


def _config(e1, e2, operation='and', inhibit=None):
    from .hexgrid import CANONICAL_STATES, ROUTING_HEX
    return next(index for index in CANONICAL_STATES
                if (ROUTING_HEX[index] == (e1, e2, inhibit, operation)
                    or ROUTING_HEX[index] == (e2, e1, inhibit, operation)))


def _two_tile_oscillator():
    pad, a, b, root = (0, 0), (1, 0), (1, -1), (2, 0)
    channels = {
        cell: {direction: 0 for direction in TRI_DIRS}
        for cell in (a, b, root)}
    from_pad = _direction(a, pad)
    from_b = _direction(a, b)
    to_b = _direction(a, b)
    to_root = _direction(a, root)
    ring_gate = _config(from_pad, from_b, operation='or')
    channels[a][to_b] = ring_gate
    channels[a][to_root] = ring_gate
    from_a = _direction(b, a)
    channels[b][from_a] = _config(from_a, from_a)
    root_in = _direction(root, a)
    channels[root][root_in] = _config(root_in, root_in)
    grid = {
        cell: pack_channels(*(mapping[direction] for direction in TRI_DIRS))
        for cell, mapping in channels.items()}
    return grid, (pad,), {'Q': root}


def _relay_path(path, output_role='Q'):
    channels = {
        cell: {direction: 0 for direction in TRI_DIRS}
        for cell in path[1:]}
    for index, cell in enumerate(path[1:], 1):
        previous = path[index - 1]
        incoming = _direction(cell, previous)
        outgoing = (incoming if index == len(path) - 1 else
                    _direction(cell, path[index + 1]))
        channels[cell][outgoing] = _config(incoming, incoming)
    grid = {
        cell: pack_channels(*(mapping[direction] for direction in TRI_DIRS))
        for cell, mapping in channels.items()}
    return grid, (path[0],), {output_role: path[-1]}


def _echo_grid():
    return _relay_path(((0, 0), (1, 0), (1, -1), (2, -1)))


def _pair_gap2_grid():
    pad, root = (0, 0), (1, 0)
    relays = ((-1, 0), (-1, -1), (0, -1), (1, -1))
    grid, _pads, outputs = _relay_path((pad,) + relays + (root,))
    from_delayed = _direction(root, relays[-1])
    from_current = _direction(root, pad)
    gate = _config(from_delayed, from_current)
    root_channels = [0, 0, 0]
    root_channels[0] = gate
    grid[root] = pack_channels(*root_channels)
    return grid, (pad,), outputs, {cell: 6 for cell in relays}


def _sequence_gap3_grid():
    pad_a = (0, 0)
    path = (pad_a, (1, 0), (1, -1), (2, -1), (2, -2))
    root = path[-1]
    # _relay_path treats the last path cell as the output root, so use the
    # first three relays then replace that root with a two-input coincidence.
    grid, _pads, outputs = _relay_path(path)
    delayed = path[-2]
    pad_b = next(cell for cell in sorted(hex_dirs(*root).values())
                 if cell != delayed and cell not in grid and cell != pad_a)
    from_a = _direction(root, delayed)
    from_b = _direction(root, pad_b)
    gate = _config(from_a, from_b)
    root_channels = [0, 0, 0]
    root_channels[0] = gate
    grid[root] = pack_channels(*root_channels)
    return grid, (pad_a, pad_b), outputs


def _veto_grid():
    root = (1, 0)
    neighbours = sorted(hex_dirs(*root).values())
    pad_a, pad_b = (0, 0), next(cell for cell in neighbours if cell != (0, 0))
    from_a, from_b = _direction(root, pad_a), _direction(root, pad_b)
    gate = _config(from_a, from_a, inhibit=from_b)
    channels = [0, 0, 0]
    channels[0] = gate
    return {root: pack_channels(*channels)}, (pad_a, pad_b), {'Q': root}


def _refractory_grid():
    """Non-paralyzable refractory gate for the three-second oracle.

    A lone slow relay gets close, but it is the wrong physical primitive:
    every rejected input pushes its analog voltage down again and extends the
    recovery interval.  The reference machine deliberately does not restart
    its cooldown on rejected inputs.  Put a fast feedback stage in front of the
    slow output so inputs arriving while the accepted pulse is in flight are
    clamped at the front gate and never reach the slow capacitor.
    """
    pad, gate_cell, feedback, root = (0, 0), (1, 0), (1, -1), (2, 0)
    from_pad = _direction(gate_cell, pad)
    from_feedback = _direction(gate_cell, feedback)
    gated_buffer = _config(
        from_pad, from_pad, inhibit=from_feedback)

    gate_channels = {direction: 0 for direction in TRI_DIRS}
    gate_channels[_direction(gate_cell, feedback)] = gated_buffer
    gate_channels[_direction(gate_cell, root)] = gated_buffer

    feedback_channels = {direction: 0 for direction in TRI_DIRS}
    from_gate = _direction(feedback, gate_cell)
    feedback_channels[from_gate] = _config(from_gate, from_gate)

    root_channels = {direction: 0 for direction in TRI_DIRS}
    from_gate = _direction(root, gate_cell)
    root_channels[from_gate] = _config(from_gate, from_gate)

    grid = {
        gate_cell: pack_channels(*(
            gate_channels[direction] for direction in TRI_DIRS)),
        feedback: pack_channels(*(
            feedback_channels[direction] for direction in TRI_DIRS)),
        root: pack_channels(*(
            root_channels[direction] for direction in TRI_DIRS)),
    }
    # The output capacitor supplies the declared three-second dead time.  The
    # gate and feedback stages stay fast so a rejected edge cannot leak through
    # while inhibition is being established.
    return grid, (pad,), {'Q': root}, {
        gate_cell: 0, feedback: 0, root: 7}


def _burst_x3_grid():
    """Three physical delay taps with approximately two-second separation."""
    pad, root = (0, 0), (1, 0)
    paths = (
        (pad, (-1, 0), (-1, -1), (0, -1), (1, -1), root),
        (pad, (0, 1), (1, 1), (2, 1), (2, 0), root),
    )
    channels = {
        cell: {direction: 0 for direction in TRI_DIRS}
        for path in paths for cell in path[1:-1]}
    tau_indices = {}
    for path_index, path in enumerate(paths):
        for index, cell in enumerate(path[1:-1], 1):
            incoming = _direction(cell, path[index - 1])
            outgoing = _direction(cell, path[index + 1])
            channels[cell][outgoing] = _config(incoming, incoming)
            # Four fast relays give the middle tap a 1.64-second offset;
            # four base relays give the last tap a four-second offset.  Both
            # are inside the event contract's half-second timing tolerance.
            tau_indices[cell] = 6 if path_index == 0 else 0

    root_channels = {direction: 0 for direction in TRI_DIRS}
    for source in (pad, paths[0][-2], paths[1][-2]):
        incoming = _direction(root, source)
        root_channels[incoming] = _config(incoming, incoming)
    channels[root] = root_channels
    grid = {
        cell: pack_channels(*(
            mapping[direction] for direction in TRI_DIRS))
        for cell, mapping in channels.items()}
    tau_indices[root] = 0
    return grid, (pad,), {'Q': root}, tau_indices


def synthesize_branched_dynamic(
        target, chromosome_count=2, max_telomere=24):
    name = str(getattr(target, 'name', ''))
    if not (name.startswith('Oscillator')
            or name.startswith('Pattern (1000)')
            or name.startswith('Burst x3')
            or name.startswith('Echo (delay 3)')
            or name.startswith('Pair detector (gap 2)')
            or name.startswith('Refractory filter (3 seconds)')
            or name.startswith('Sequence A->B')
            or name.startswith('Veto gate')):
        raise SynthesisError('no compiled nervous dynamic witness for %s' % name)
    tau_indices = None
    if name.startswith('Echo (delay 3)'):
        grid, pads, outputs = _echo_grid()
    elif name.startswith('Burst x3'):
        grid, pads, outputs, tau_indices = _burst_x3_grid()
    elif name.startswith('Pair detector (gap 2)'):
        grid, pads, outputs, tau_indices = _pair_gap2_grid()
    elif name.startswith('Refractory filter (3 seconds)'):
        grid, pads, outputs, tau_indices = _refractory_grid()
    elif name.startswith('Sequence A->B'):
        grid, pads, outputs = _sequence_gap3_grid()
    elif name.startswith('Veto gate'):
        grid, pads, outputs = _veto_grid()
    else:
        grid, pads, outputs = _two_tile_oscillator()
    if name.startswith('Pattern (1000)'):
        # One base and one slow physical stage make one lap 4.07 seconds,
        # inside the target's 4 +/- 0.5 cadence tolerance.  The same circulating-pulse
        # topology is used; only its heritable capacitor sizes differ.
        tau_indices = {(1, 0): 0, (1, -1): 7}
    return synthesize_branched_grid(
        target, grid, pads, outputs,
        chromosome_count=chromosome_count, max_telomere=max_telomere,
        tau_indices=tau_indices)
