"""Continuous-time directed-wire simulator for the Functional NV Net.

The honeycomb edge between two cells carries two independent, antiparallel
binary wires.  A component only drives the directions listed by its permanent
catalogue entry, and only consumes its listed input directions.  Undriven and
off-grid inputs read zero.
"""
from __future__ import annotations

import heapq
from math import inf

from substrates.nervous.hexgrid import hex_dirs

from .catalogue import BY_ID, ComponentType, component

TICK = 1.0
PROPAGATION_DELAY = 1.0


def facing_direction(source, destination) -> str | None:
    """Return the source-local direction whose edge reaches destination."""
    for direction, neighbor in hex_dirs(*source).items():
        if neighbor == destination:
            return direction
    return None


def source_for_input(cell, direction):
    return hex_dirs(*cell)[direction]


def _wiring_maps(grid, input_nodes):
    """Return the receiver inputs and source watchers used by FNV physics."""
    sources_by_cell = {}
    watchers = {cell: set() for cell in grid}
    for cell, state in grid.items():
        entry = BY_ID[state]
        sources = []
        for direction in entry.inputs:
            source = source_for_input(cell, direction)
            if source not in grid:
                sources.append(None)
                continue
            facing = facing_direction(source, cell)
            source_entry = BY_ID[grid[source]]
            driven = (
                source in input_nodes or facing in source_entry.outputs)
            if driven:
                sources.append(source)
                watchers[source].add(cell)
            else:
                sources.append(None)
        sources_by_cell[cell] = tuple(sources)
    return sources_by_cell, watchers


def effective_wiring_edges(grid, input_nodes=()):
    """The directed connections capable of influencing non-source components.

    Incoming physical wires to an input pad are deliberately omitted because
    FNV pads are source-only: ``FunctionalSim._evaluate`` ignores them.
    """
    normalized = {
        tuple(position): int(state)
        for position, state in grid.items()
        if int(state) != 0
    }
    inputs = {tuple(cell) for cell in (input_nodes or ())}
    _, watchers = _wiring_maps(normalized, inputs)
    return tuple(sorted(
        (source, destination)
        for source, destinations in watchers.items()
        for destination in destinations
        if destination not in inputs
    ))


class FunctionalSim:
    """Event-driven simulation of one grown ``{position: component_id}`` grid."""

    def __init__(self, grid, *, input_nodes=None, output_nodes=None,
                 max_events=2048):
        self.grid = {tuple(pos): int(state) for pos, state in grid.items()
                     if int(state) != 0}
        for state in self.grid.values():
            component(state)
        self.input_nodes = set(input_nodes or ())
        self.output_nodes = set(output_nodes or ())
        self.max_events = None if max_events is None else max(1, int(max_events))
        self._build_wiring()
        self.reset()

    def _build_wiring(self):
        self._inputs, self._watch = _wiring_maps(
            self.grid, self.input_nodes)

    def reset(self):
        self.levels = {cell: 0 for cell in self.grid}
        self._external_depth = {cell: 0 for cell in self.input_nodes
                                if cell in self.grid}
        self._input_cache = {
            cell: tuple(0 for _ in self._inputs.get(cell, ()))
            for cell in self.grid}
        self._generation = {cell: 0 for cell in self.grid}
        self._hold_release = {cell: 0 for cell in self.grid}
        self._osc_generation = {cell: 0 for cell in self.grid}
        self._normalizer_armed = {cell: True for cell in self.grid}
        self._normalizer_active = {cell: False for cell in self.grid}
        self._heap = []
        self._seq = 0
        self._tick = 0
        self._previous_inputs = {}
        self.event_count = 0
        self.overflow = False
        self.rise_times = {cell: [] for cell in self.grid}
        self.fall_times = {cell: [] for cell in self.grid}
        self.ever = {cell: 0 for cell in self.grid}

    def _push(self, when, kind, cell, value=0, token=0):
        self._seq += 1
        heapq.heappush(self._heap, (
            float(when), self._seq, kind, cell, int(value), int(token)))

    def inject_pulse(self, cell, time, width=TICK):
        if cell not in self._external_depth:
            return
        time, width = float(time), float(width)
        if width <= 0:
            raise ValueError("FNV input pulse width must be positive")
        self._push(time, "external", cell, 1)
        self._push(time + width, "external", cell, -1)

    def inject_level(self, cell, time, value):
        """Set a source terminal's external level at an absolute time."""
        if cell not in self._external_depth:
            return
        value = 1 if value else 0
        # Always queue the assignment.  Several future level changes may be
        # scheduled before the first is processed, so the present level cannot
        # be used to decide whether a future assignment is redundant.
        self._push(time, "external_set", cell, value)

    def _input_values(self, cell):
        return tuple(0 if source is None else self.levels[source]
                     for source in self._inputs.get(cell, ()))

    def _schedule_inertial(self, cell, value, delay=PROPAGATION_DELAY):
        self._generation[cell] += 1
        token = self._generation[cell]
        if int(value) != self.levels[cell]:
            self._push(self._time + delay, "inertial", cell, value, token)

    def _evaluate(self, cell):
        if cell in self.input_nodes or cell not in self.grid:
            return
        entry = BY_ID[self.grid[cell]]
        values = self._input_values(cell)
        previous = self._input_cache[cell]
        self._input_cache[cell] = values
        behavior = entry.behavior

        if behavior in ("AND", "OR", "XOR", "VETO"):
            a, b = values
            desired = {
                "AND": a & b,
                "OR": a | b,
                "XOR": a ^ b,
                "VETO": a & (not b),
            }[behavior]
            self._schedule_inertial(cell, int(desired), entry.delay)
            return

        if behavior == "DELAY":
            if values != previous:
                self._push(self._time + entry.duration, "transport", cell,
                           values[0])
            return

        if behavior == "NORMALIZER":
            old, new = previous[0], values[0]
            if new and not old and self._normalizer_armed[cell]:
                self._normalizer_armed[cell] = False
                self._normalizer_active[cell] = True
                token = self._generation[cell] = self._generation[cell] + 1
                start = self._time + entry.delay
                self._push(start, "normalizer_set", cell, 1, token)
                self._push(start + entry.duration, "normalizer_end", cell, 0,
                           token)
            elif old and not new and not self._normalizer_active[cell]:
                self._normalizer_armed[cell] = True
            return

        if behavior == "HOLD":
            old, new = previous[0], values[0]
            if new and not old:
                self._hold_release[cell] += 1
                self._push(self._time + entry.delay, "transport", cell, 1)
            elif old and not new:
                self._hold_release[cell] += 1
                token = self._hold_release[cell]
                self._push(self._time + entry.delay + entry.duration,
                           "hold_release", cell, 0, token)
            return

        if behavior == "C_ELEMENT":
            if values == (1, 1):
                self._schedule_inertial(cell, 1, entry.delay)
            elif values == (0, 0):
                self._schedule_inertial(cell, 0, entry.delay)
            else:
                # Disagreement means hold the state that exists now, including
                # cancelling a not-yet-applied agreement transition.
                self._generation[cell] += 1
            return

        if behavior == "TOGGLE":
            if previous == (0,) and values == (1,):
                self._push(self._time + entry.delay, "toggle", cell)
            return

        if behavior == "GATED_OSCILLATOR":
            old, new = previous[0], values[0]
            if new and not old:
                self._osc_generation[cell] += 1
                token = self._osc_generation[cell]
                self._push(self._time + entry.delay, "oscillator", cell, 1,
                           token)
            elif old and not new:
                self._osc_generation[cell] += 1
                token = self._osc_generation[cell]
                self._push(self._time + entry.delay, "oscillator_stop", cell,
                           0, token)

    def _apply_event(self, kind, cell, value, token):
        if cell not in self.grid:
            return
        entry: ComponentType = BY_ID[self.grid[cell]]
        if kind == "external":
            self._external_depth[cell] = max(
                0, self._external_depth.get(cell, 0) + value)
            self.levels[cell] = int(self._external_depth[cell] > 0)
        elif kind == "external_set":
            self._external_depth[cell] = value
            self.levels[cell] = value
        elif kind == "inertial":
            if token == self._generation[cell]:
                self.levels[cell] = value
        elif kind == "transport":
            self.levels[cell] = value
        elif kind == "hold_release":
            if token == self._hold_release[cell] and not self._input_values(cell)[0]:
                self.levels[cell] = 0
        elif kind == "normalizer_set":
            if token == self._generation[cell]:
                self.levels[cell] = 1
        elif kind == "normalizer_end":
            if token == self._generation[cell]:
                self.levels[cell] = 0
                self._normalizer_active[cell] = False
                if not self._input_values(cell)[0]:
                    self._normalizer_armed[cell] = True
        elif kind == "toggle":
            self.levels[cell] ^= 1
        elif kind == "oscillator":
            if (token == self._osc_generation[cell]
                    and self._input_values(cell)[0]):
                self.levels[cell] = value
                duration = entry.high_time if value else entry.low_time
                self._push(self._time + duration, "oscillator", cell,
                           0 if value else 1, token)
        elif kind == "oscillator_stop":
            if token == self._osc_generation[cell]:
                self.levels[cell] = 0

    def _process_time(self, when):
        self._time = when
        before = {}
        while self._heap and self._heap[0][0] == when:
            _, _, kind, cell, value, token = heapq.heappop(self._heap)
            before.setdefault(cell, self.levels.get(cell, 0))
            self._apply_event(kind, cell, value, token)
        changed = {
            cell for cell, old in before.items()
            if self.levels.get(cell, 0) != old
        }
        for cell in changed:
            if self.levels[cell]:
                self.rise_times[cell].append(when)
                self.ever[cell] = 1
                self.event_count += 1
            else:
                self.fall_times[cell].append(when)
        if (self.max_events is not None
                and self.event_count > self.max_events):
            self.overflow = True
            self._heap.clear()
            return
        affected = set()
        for source in changed:
            affected.update(self._watch.get(source, ()))
        for cell in sorted(affected):
            self._evaluate(cell)

    def advance_to(self, when):
        when = float(when)
        while self._heap and not self.overflow and self._heap[0][0] <= when:
            self._process_time(self._heap[0][0])
        self._time = when

    def activity_at(self, _when=None):
        return dict(self.levels)

    @property
    def pulse_intervals(self):
        result = {}
        for cell in self.grid:
            falls = self.fall_times[cell]
            result[cell] = [
                [start, falls[index] if index < len(falls) else inf]
                for index, start in enumerate(self.rise_times[cell])
            ]
        return result

    def rise_trains(self, cells):
        return {cell: list(self.rise_times.get(cell, ())) for cell in cells}

    @property
    def out(self):
        return dict(self.levels)

    def step(self, input_levels=None):
        input_levels = input_levels or {}
        start = float(self._tick)
        for cell in self.input_nodes:
            value = 1 if input_levels.get(cell, 0) else 0
            previous = self._previous_inputs.get(cell, 0)
            if value != previous:
                self._push(start, "external_set", cell, value)
                self._previous_inputs[cell] = value
        self.advance_to(start + 0.5)
        self._tick += 1
        return self.activity_at()

    def run_bits(self, input_streams, steps):
        history = []
        for tick in range(int(steps)):
            levels = {
                cell: stream[tick] if tick < len(stream) else 0
                for cell, stream in input_streams.items()
            }
            history.append(self.step(levels))
        return history
