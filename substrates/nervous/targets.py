"""
substrates/nervous/targets.py - temporal targets for the hex nervous net.

A TemporalTarget scores behavior over time instead of a truth table.  Each
Trial drives a stimulus stream and may define raw expected point events,
sampled state/persistence windows, or a target-specific cadence invariant.

Every preset carries SEVERAL trials with different pulse timings. A net that
merely matches one fixed schedule (a lucky delay chain) fails the shifted
trials; only genuine state - a loop holding a circulating value - passes all
of them. That is what makes these targets select for memory.

The nervous backend also runs the combinational targets registered in
substrates.snn.targets (gates, adders, custom tables); those are plain data objects
passed in by the GUI, so nothing here needs to import them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
import random
from typing import Dict, List, Optional, Tuple

from .contracts import (BehaviorContract, cadence_contract,
                        combinational_contract, event_contract,
                        state_contract, toggle_contract)

Pos = Tuple[int, int]


@dataclass
class OutputTerminal:
    role: str            # unique label, e.g. "Q"
    pos:  Pos            # terminal cell the output is read nearest to


@dataclass
class Trial:
    streams:  List[Tuple[int, ...]]                 # streams[t] = input bits at tick t
    expected: Dict[str, List[Optional[int]]]        # role -> expected bit per tick (None = don't score)
    # Optional ground-truth point events.  Event-semantic targets populate this
    # directly so consecutive events are not confused with one held-high run.
    expected_events: Dict[str, List[float]] = field(default_factory=dict)
    # Optional physical input schedule, one list per input. Each event is
    # ``(start_time, pulse_width)`` and may start between ticks. Nervous
    # evaluation uses this instead of ``streams``; the sampled stream remains as
    # a compatibility/display fallback for clocked backends.
    input_events: Optional[List[List[Tuple[float, float]]]] = None
    # Optional complete output waveforms. Each interval is ``(rise, fall)``;
    # waveform targets score both boundaries so equal rise times with different
    # pulse widths are no longer treated as equivalent.
    expected_intervals: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict)
    # Explicit periodic truth-table windows. Each entry is
    # ``(start, end, data_input_bits)``. Unlike reconstructing windows from
    # input onsets, this represents a silent all-zero row without adding an
    # unnecessary physical strobe port to tables whose zero row expects quiet.
    case_windows: List[Tuple[float, float, Tuple[int, ...]]] = field(
        default_factory=list)


@dataclass
class TemporalTarget:
    name:            str
    inputs:          List[Pos]
    outputs:         List[OutputTerminal]
    T:               int
    trials:          List[Trial]
    grid_size:       int = 7
    iters:           int = 30      # safety CAP - growth stops at its attractor
    output_strategy: str = "terminals"
    # Compatibility I/O strategy consumed by SNN/programmatic LUT experiments.
    # Current Nervous/FNV runs require 'fixed' here while resolving their native
    # genome layouts and fitted probes elsewhere.
    #   'fixed'          - use the backend's native/default binding.
    #   'tag_rank'       - ports bind to the highest-tagged cells, in order
    #                      (Method A). Placement becomes an evolvable genome trait.
    #   'wiring_chromosome' - chromosome 3 maps each port to a desired node type,
    #                      then selects one matching instance (Method B).
    #   'spatial_chromosome': chromosome 3 maps each port to an evolvable
    #                      normalised (x, y) anchor. Input anchors seed
    #                      development; output anchors select the nearest
    #                      unclaimed live cell.
    io_placement:    str = "fixed"
    temporal:        bool = True          # marker so the GUI/GA can dispatch
    description:     str = ''             # human explanation, shown in the GUI
    # Executable statement of the target idea. Every backend supplies raw
    # observations to the same contract evaluator; targets no longer choose a
    # scoring pipeline.
    contract:        BehaviorContract = field(default_factory=state_contract)
    max_events:      int = 2048
    # Nominal input->output latency baked into the expected traces (oracle
    # `latency`). Kept as explicit metadata so semantic scoring does not have to
    # rediscover it; the fitted alignment is a SEPARATE, additional offset.
    latency:         int = 0
    # Optional physical-waveform relation. Width-sensitive targets derive their
    # expected intervals from explicit input pulses and the run's base delay.
    waveform_contract: str = ''
    waveform_delay_multiplier: float = 1.0
    waveform_width_multiplier: float = 1.0
    # Empty means no additional restriction. Physical floating-time targets can
    # opt into the asynchronous backends (nervous, lut) only, so clocked
    # backends do not silently quantize them.
    supported_backends: Tuple[str, ...] = ()
    # Empty means every nervous node-timing model (see substrates/nervous/pulse.NODE_MODELS)
    # can attempt the target. Waveform-contract targets demand input-DEPENDENT
    # output durations, which the fixed-width 'uniform' node physically
    # cannot emit (regenerated widths, single-driver wires) - they declare
    # ('pulse_delay',) so a run under the wrong model is filtered out instead
    # of silently capping below 1.0. Only consulted for the nervous backend.
    supported_models: Tuple[str, ...] = ()
    # Optional explicit GUI folder (see target_ui.target_category). Empty means
    # "derive from score-mode semantics". Set where semantics would mislead:
    # periodic truth-table wrappers are combinational despite scoring as
    # events, and pulse-width targets are about durations, not edge timing.
    category: str = ''
    # Original rows for a periodic combinational wrapper. Temporal trials do
    # not otherwise identify the all-zero row (silence has no timestamp), so
    # retaining it is necessary for exact feasibility checks and synthesis.
    combinational_cases: List[
        Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)
    combinational_data_inputs: int = 0
    combinational_strobe: bool = False
    # Ticks each active input LEVEL is held inside its row window (0 for the
    # edge-timed twins, which present a row as one point event), and how much of
    # that hold is a settling allowance before the output level is read.
    combinational_hold: int = 0
    combinational_settle: int = 0
    # Edge-timed truth-table twins deliberately do not use ``combinational_*``:
    # their contract scores one pulse per asserted row, not a held level.  They
    # retain the source table separately so certification can replay genuinely
    # fresh row orders without changing their event-scoring semantics.
    temporal_logic_cases: List[
        Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)
    temporal_logic_data_inputs: int = 0
    temporal_logic_gap: int = 0
    temporal_logic_schedules: int = 0
    temporal_logic_tail: int = 0

    def __post_init__(self):
        # The established scoring primitives still read these names. They are
        # projections of contract data, never a second source of target truth.
        aliases = {
            'tolerance': ('event_tolerance', 'waveform_tolerance'),
            'max_shift': ('event_max_shift',),
            'role_window': ('event_role_window',),
            'fit_latency': ('fit_latency',),
            'period': ('cadence_period',),
            'settle': ('cadence_settle', 'stepper_settle'),
            'min_events': ('cadence_min_events', 'stepper_min_events'),
            'min_period': ('stepper_min_period',),
            'max_period': ('stepper_max_period',),
            'max_delay': ('stepper_max_delay',),
        }
        defaults = {
            'event_tolerance': 0.5, 'waveform_tolerance': 0.25,
            'event_max_shift': 12.0, 'fit_latency': True,
            # How far apart the outputs of a MULTI-output target may drift from
            # each other, in ticks, before it costs score. Different physical
            # path lengths to different output pads are routing, not a logic
            # error. Ignored by single-output targets.
            'event_role_window': 1.0, 'event_role_slack': None,
            'cadence_period': 0.0, 'cadence_tolerance': 0.5,
            'cadence_settle': 5.0, 'cadence_min_events': 4,
            'stepper_min_period': 2, 'stepper_max_period': 6,
            'stepper_settle': 2, 'stepper_min_events': 4,
            'stepper_max_delay': 8,
        }
        for name, value in defaults.items():
            setattr(self, name, value)
        for clause in self.contract.constraints:
            params = clause.parameters
            for key, names in aliases.items():
                if key in params:
                    for name in names:
                        setattr(self, name, params[key])
            if 'tolerance' in params and clause.relation == 'sustained_cadence':
                self.cadence_tolerance = params['tolerance']

    @property
    def n_inputs(self):  return len(self.inputs)
    @property
    def n_outputs(self): return len(self.outputs)
    # so interactive playback can drive it like any nervous target
    @property
    def cases(self):     return []
    high = 1.0


# -- trace-building helpers ------------------------------------------------------

# Unscored grace ticks after an input event: the output has this long to
# respond (and settle) before it is scored, so a valid circuit is not required
# to react at one exact tick - a Set that takes 2 ticks and a Reset that takes
# 2-5 are all accepted. Too small over-penalises legitimate response-latency
# variation (measured: raising 3 -> 5 lifted the SR-latch ceiling ~0.96 -> ~0.98
# by no longer marking slightly-slow resets wrong); too large leaves too few
# scored ticks. 5 is the balance for the T=20..24 traces here.
SETTLE = 5


def describe_target(goal, tests):
    """The static half of a target's description: what the circuit must do, and
    how it is exercised.

    There is deliberately no 'Scoring:' section. How a target is scored is not
    prose chosen when the target is written - it is the executable contract, and
    every report renders that through contracts.behavior_contract_lines(). A
    hand-written copy here could drift from the contract, and it did: after the
    contract rewrite the GUI printed the old mode-based scoring description
    directly above the contract that actually did the scoring.
    """
    return 'Goal: %s\nTests: %s' % (goal.strip(), tests.strip())


def _zero_row_holds_sole_negative(cases, zero_outputs, n_outputs):
    """True when dropping the all-zero row would leave an output with no 0 case.

    Periodic targets now retain a scoring window even for a silent row. This
    compatibility check still preserves the historical physical case-valid
    lane for tables whose zero row was the sole negative evidence.

    OR is exactly this shape - its only 0 is ``00 -> 0`` - and it "solved" for a
    blanket-firing circuit until the strobe was forced here. The check is
    deliberately narrow: it fires only when the zero row actually SUPPLIES the
    missing level, so a genuinely constant output is left alone.
    """
    presented = [output_bits for input_bits, output_bits in cases
                 if any(input_bits)]
    if not presented:
        return False
    return any(
        len({output_bits[index] for output_bits in presented}) < 2
        and zero_outputs[index] not in {output_bits[index]
                                        for output_bits in presented}
        for index in range(n_outputs))


def periodic_combinational_target(target, spacing=None, repeats=2, latency=1,
                                  hold=None):
    """Encode a static binary truth table as widely-spaced asynchronous cases.

    Every input combination in the truth table (all 2**n_inputs rows) is
    presented as its own isolated test window: a 1 is HELD HIGH for ``hold``
    ticks, a 0 stays silent, and the expected output emits an event ``latency``
    ticks after the row's onset for the rows whose output is 1. Consecutive
    windows are separated by a generous settle gap - ``spacing`` ticks between
    onsets, several times the few-tick transient of these small grids - so any
    circulating pulse from one case dies out before the next begins and no test
    can contaminate the next.

    The held level is what makes this a COMBINATIONAL presentation rather than a
    temporal one. A combinational function is a function of a level that is
    present while the circuit computes: FNV holds its inputs for the whole
    settling horizon, SNN holds an input current for the whole run, and the
    native LUT/nervous static scorers hold their case levels too. This wrapper
    used to be the odd one out - a single-tick point event per row - which asked
    the asynchronous backends to REMEMBER the row while computing it. That is a
    memory task, and it is exactly what the ``<name> (temporal)`` twin
    (:func:`coincident_temporal_target`) exists to pose; the two encodings are
    only different problems if this one holds.

    ``hold`` defaults to a full grid diameter, so the level is still applied
    when the far side of the array settles, and the settle gap after release
    keeps its previously justified width. The whole table then repeats
    ``repeats`` times (so a circuit must re-arm, not fire once and stick) under
    alternate row orders and two phases, which keeps a fixed oscillator from
    passing without doing input-dependent logic.

    ``spacing`` defaults to a comfortable multiple of the grid so it scales with
    the substrate size (a 2-bit adder's larger grid gets proportionally wider
    windows than a two-input gate's).
    """
    repeats = int(repeats)
    latency = int(latency)

    data_inputs = int(target.n_inputs)
    n_outputs = int(target.n_outputs)
    cases = list(target.cases)
    if not cases:
        raise ValueError('periodic combinational targets require truth-table cases')
    zero_outputs = next(
        (output_bits for input_bits, output_bits in cases
         if not any(input_bits)), (0,) * n_outputs)
    # A physical case-valid pulse is necessary when the otherwise-silent zero
    # row must produce an event or supplies an output's only negative example.
    # Quiet zero rows still receive an explicit *scoring* window below, without
    # needlessly changing the circuit's data-input interface.
    has_strobe = (any(zero_outputs)
                  or _zero_row_holds_sole_negative(
                      cases, zero_outputs, n_outputs))
    n_inputs = data_inputs + int(has_strobe)
    span = max(n_inputs, n_outputs, 2)
    grid_size = max(5, 2 * span + 1)
    # The row only needs one crossing of the grid before it is read.  The old
    # 2G settle plus another full-G measurement held every input level for
    # 3G, then added a 4G release gap.  That was excessive for the nervous
    # combinational encoding: it multiplied every truth-table row and repeat
    # into long mostly-idle traces, especially for decoder/multiplier targets.
    # Keep a full-G propagation margin and a half-G read window instead.
    #
    # The margin is G+1, not G: grid width is a LOWER bound on propagation, not
    # an upper one. Measured, a compiled LUT witness raises its output 6 ticks
    # after the row is applied on every grid size (the path wanders, and the
    # inertial delay is per cell, not per column), so a bare G budget is one
    # tick short at G=5. Since the read window is only ``measure`` ticks wide,
    # that one tick cost a third of every asserting row: AND, XOR and the half
    # adder capped at 0.8333 with a PERFECT circuit, which is exactly what a
    # bare wire echoing one input also scored - the contract could not tell a
    # correct gate from a wire. ``hold`` follows settle, so the read window is
    # unchanged in width and simply opens one tick later.
    settle = grid_size + 1
    measure = max(3, (grid_size + 1) // 2)
    if hold is None:
        hold = settle + measure
    hold = max(2, int(hold))
    settle = min(settle, hold - 1)
    if spacing is None:
        # Leave a 2G release gap, enough for a pulse to clear the body without
        # making each truth-table schedule dominated by idle ticks.
        spacing = max(hold + 2 * grid_size, hold + latency + 8)
    spacing = int(spacing)
    if spacing < hold + latency + 2 or repeats < 1 or latency < 0:
        # A row must fall silent before the next one rises, or two rows that
        # share an input lane would merge into one uninterrupted level and the
        # second row would never present an edge.
        raise ValueError(
            'spacing/repeats/latency must be >= hold+latency+2, >= 1, and >= 0')
    inputs = [(0, 2 * index + 1) for index in range(n_inputs)]
    output_ys = ([grid_size // 2] if n_outputs == 1 else [
        int(round(1 + index * (grid_size - 3) / (n_outputs - 1)))
        for index in range(n_outputs)
    ])
    outputs = [OutputTerminal(term.role, (2, output_ys[index]))
               for index, term in enumerate(target.outputs)]

    def active_first(rows):
        """Rotate, without dropping any row, so activity seeds each schedule."""
        start = next((index for index, (bits, _) in enumerate(rows)
                      if any(bits)), 0)
        return rows[start:] + rows[:start]

    schedules = [active_first(cases)]
    alternate = active_first(list(reversed(cases)))
    if alternate != schedules[0]:
        schedules.append(alternate)

    phases = (2, 3)
    cycle = len(cases) * spacing
    T = max(phases) + repeats * cycle + latency + 2
    trials = []
    for schedule in schedules:
        for phase in phases:
            streams = [[0] * n_inputs for _ in range(T)]
            expected = {}
            expected_events = {}
            case_windows = [
                (
                    float(phase + (
                        repetition * len(schedule) + slot) * spacing),
                    float(phase + (
                        repetition * len(schedule) + slot + 1) * spacing),
                    tuple(input_bits),
                )
                for repetition in range(repeats)
                for slot, (input_bits, _output_bits) in enumerate(schedule)
            ]
            expected_intervals = {}
            for output_index, terminal in enumerate(outputs):
                events = []
                held_levels = []
                for repetition in range(repeats):
                    for slot, (input_bits, output_bits) in enumerate(schedule):
                        tick = phase + (repetition * len(schedule) + slot) * spacing
                        # Hold the row's levels: contiguous 1s are ONE physical
                        # pulse of that width on every backend (nervous
                        # temporal._inject_stream_edges, LUT AsyncLutSim), so
                        # this is a level applied and later released, not a
                        # burst of re-triggering edges.
                        held = range(tick, min(tick + hold, T))
                        for lane, bit in enumerate(input_bits):
                            if bit:
                                for step in held:
                                    streams[step][lane] = 1
                        if has_strobe:
                            for step in held:
                                streams[step][data_inputs] = 1
                        if output_bits[output_index]:
                            # The row asserts: the output must SIT high across
                            # the settled read window, not merely edge once.
                            # The event marks which rows assert (and keeps the
                            # older event readers working); the interval is the
                            # level the contract actually judges.
                            events.append(float(tick + latency))
                            held_levels.append(
                                (float(tick + settle), float(tick + hold)))
                expected[terminal.role] = [
                    1 if any(low <= tick < high for low, high in held_levels)
                    else 0
                    for tick in range(T)]
                expected_events[terminal.role] = events
                expected_intervals[terminal.role] = held_levels
            trials.append(Trial(
                [tuple(row) for row in streams], expected, expected_events,
                expected_intervals=expected_intervals,
                case_windows=case_windows))

    return TemporalTarget(
        target.name, inputs, outputs, T, trials,
        grid_size=grid_size, iters=30, contract=combinational_contract(),
        latency=latency,
        # LUT ONLY, and not because the nervous net is too weak to compute the
        # table - it is because a HELD LEVEL IS NOT A SIGNAL to that substrate.
        # The paper's node couples its inputs CAPACITIVELY (substrates/nervous/
        # analog.py): a series capacitor has no DC path, so holding an input
        # high delivers exactly one edge and then nothing. Measured: a
        # hand-built coincidence tile - a perfect AND gate, firing on precisely
        # the right rows - emits a 0.53-tick spike where this contract demands
        # the output SIT high for 5 ticks, and scores 0.5000, the silent
        # baseline. The same tile scores 1.0000 on the edge-timed twin.
        #
        # So a nervous run on this target measures the encoding of the stimulus,
        # not the circuit. Every truth table has a "<name> (temporal)" twin
        # (coincident_temporal_target) that poses the same function in edges,
        # which is the form this hardware actually reads; that is what nervous
        # runs instead.
        #
        # This is a STIMULUS-ENCODING restriction, not a claim about capability.
        # The alternative - delivering the held level as a chopped carrier, the
        # way real AC-coupled links carry DC - was measured to work at stock
        # physics (a 1-tick chop holds the output high across the whole 15-tick
        # window). It needs a per-substrate stimulus builder, so it is not what
        # this target does today.
        supported_backends=('lut',),
        category='Combinational logic',
        combinational_cases=[
            (tuple(input_bits), tuple(output_bits))
            for input_bits, output_bits in cases],
        combinational_data_inputs=data_inputs,
        combinational_strobe=has_strobe,
        combinational_hold=hold,
        combinational_settle=settle,
        description=describe_target(
            'Compute the %s binary truth table one case at a time: each input '
            'combination gets its own %d-second test window (input 1 is HELD '
            'high for %d seconds, input 0 is silent%s), spaced far enough apart '
            'that the circuit settles between cases. Each output must then SIT '
            'at its required level for the last %d seconds the row is applied.'
            % (target.name, spacing, hold,
               '; a case-valid lane is held on every row'
               if has_strobe else '', hold - settle),
            'All %d truth-table rows are tested in isolated, widely-spaced '
            'windows and repeated in full cycles under alternate row orders and '
            'two phases, so one case cannot contaminate the next and a fixed '
            'oscillator cannot replace input-dependent logic. Held level in, '
            'held level out, as combinational logic means everywhere else in '
            'this project - the edge-timed version of the same table is the '
            'separate "%s (temporal)" target.'
            % (len(target.cases), target.name)))


def with_io_placement(target, strategy):
    """Return a copy of ``target`` that binds its I/O ports with the given
    strategy (directional ``terminal_nodes``; evolvable ``tag_rank``,
    ``wiring_chromosome``, or ``spatial_chromosome``; ``fixed`` is the default
    geometric/trace-fitted binding). Stimulus, contract, and grid are
    preserved. Spatial input anchors also become developmental germlines, so
    that strategy intentionally changes both port placement and ontogeny.
    Handy for A/B comparisons."""
    import dataclasses
    from .io_placement import IO_STRATEGIES
    if strategy not in IO_STRATEGIES:
        raise ValueError('unknown io_placement strategy: %r' % (strategy,))
    return dataclasses.replace(target, io_placement=strategy)


def _pulse_streams(T, n_inputs, pulses):
    """streams[t] for `pulses` = {input_index: [ticks it is 1]}."""
    ons = {i: set(ts) for i, ts in pulses.items()}
    return [tuple(1 if t in ons.get(i, ()) else 0 for i in range(n_inputs))
            for t in range(T)]


def _hold_trace(T, events):
    """Expected trace from `events` = [(tick, level)]: after each event's tick
    + SETTLE the output must hold `level` until the next event. Ticks inside a
    settle window are None (unscored)."""
    exp = [None] * T
    events = sorted(events) + [(T, None)]
    for (tick, level), (nxt, _) in zip(events, events[1:]):
        start = 0 if tick == 0 else tick + SETTLE
        for t in range(start, min(nxt, T)):
            exp[t] = level
    return exp


# -- general spike-event target builder ------------------------------------------

def spike_target(name, cases, T, n_inputs=None, output_role='Q', latency=1,
                 inputs=None, out_pos=(2, 2), grid_size=5, iters=30,
                 description='', outputs=None, event_max_shift=12.0):
    """Describe a temporal function purely as SPIKE EVENTS - the easy path to a
    new target. Each case in `cases` is ``(input_spikes, output_spikes)``:

        input_spikes  - {input_index: [ticks]}, or a list-of-lists (one ticks
                        list per input), giving the ticks each input pulses on.
        output_spikes - the list of ticks the output should fire on. For a
                        MULTI-OUTPUT target pass ``{role: [ticks]}`` and name
                        the roles through ``outputs``; a role left out of one
                        case's dict is simply silent for that case.

    ``outputs`` is an optional list of ``(role, pos)`` terminals. Supplying it
    is what makes a target multi-output; leaving it None keeps the historical
    single-``output_role`` behaviour byte for byte.

    Expected outputs are stored as point-event timestamps. A net that stays
    silent is penalised for every missing event and a net that fires spuriously
    is penalised for every extra one (one-to-one event F1). The first `latency`
    ticks remain unscored startup grace for compatibility with the trace view.

    `latency` is now just a nominal minimum-causal offset (default 1), NOT a delay
    the circuit must match: scoring is latency-invariant (substrates.nervous.temporal, best
    global continuous-time shift), so a circuit that produces the right spikes
    at ANY consistent delay scores the same. Describe the RELATIVE event
    structure; the absolute input->output delay is free.

    Example - a coincidence detector (fires one tick after A and B coincide)::

        spike_target('Coincidence', [
            ({0: [20],     1: [20]},     [21]),   # A & B on tick 20 -> spike @21
            ({0: [5],      1: [12]},     []),     # never coincide   -> silent
            ({0: [8],      1: []},       []),     # lone A           -> silent
            ({0: [3, 15],  1: [3, 15]},  [4, 16]),# coincide twice   -> two spikes
        ], T=24, n_inputs=2)
    """
    def _norm(input_spikes):
        if isinstance(input_spikes, dict):
            return {int(i): list(ts) for i, ts in input_spikes.items()}
        return {i: list(ts) for i, ts in enumerate(input_spikes)}

    def _norm_out(output_spikes):
        if isinstance(output_spikes, dict):
            return {str(role): set(ticks)
                    for role, ticks in output_spikes.items()}
        return {output_role: set(output_spikes)}

    norm_cases = [(_norm(isp), _norm_out(osp)) for isp, osp in cases]
    if n_inputs is None:
        n_inputs = max([0] + [i + 1 for isp, _ in norm_cases for i in isp])
    if inputs is None:
        inputs = [(0, min(grid_size - 1, 1 + 2 * i)) for i in range(n_inputs)]
    if outputs is None:
        terminals = [OutputTerminal(output_role, out_pos)]
    else:
        terminals = [OutputTerminal(str(role), tuple(pos))
                     for role, pos in outputs]
    roles = [terminal.role for terminal in terminals]
    trials = []
    for pulse_dict, out_sets in norm_cases:
        unknown = set(out_sets) - set(roles)
        if unknown:
            raise ValueError('%s: case names unknown output role(s) %s'
                             % (name, ', '.join(sorted(unknown))))
        streams = _pulse_streams(T, n_inputs, pulse_dict)
        expected, events = {}, {}
        for role in roles:
            # A role absent from this case is silent, not unscored: a spurious
            # edge on it still has to cost something.
            out_set = out_sets.get(role, set())
            expected[role] = [
                1 if t in out_set else (None if t < latency else 0)
                for t in range(T)]
            events[role] = sorted(float(t) for t in out_set)
        trials.append(Trial(streams, expected, events))
    return TemporalTarget(name, list(inputs), terminals, T, trials,
                          grid_size=grid_size, iters=iters,
                          # The expected edges above were generated at
                          # input + `latency`, so that offset is part of the
                          # target, not something the scorer should rediscover.
                          # Storing it is what lets event scoring impose a
                          # causal floor on the fitted shift (an output may be
                          # earlier than nominal, but never earlier than its
                          # own cause).
                          latency=int(latency),
                          contract=event_contract(max_shift=event_max_shift),
                          description=description or describe_target(
        'Produce exactly the requested output-edge pattern.',
        'Each supplied case contributes its input schedule, required edges, and '
        'the surrounding silence window.'))


# -- preset temporal targets -----------------------------------------------------
# Inputs/outputs kept close together (nervous nets need short signal paths).

# Trial banks are deliberately DIVERSE in pulse count, spacing and gap parity:
# with only a couple of schedules the GA memorises the timings (measured: a
# "solved" toggle locked up on pulse gaps it never saw). A net can only score
# 1.0 across the whole bank by implementing the actual rule.

def sr_latch(grid_size=5):
    """Set/Reset latch: a Set pulse drives Q to 1 (and it holds); a Reset pulse
    drives it to 0. Five trials with shifted timings and mixed set->reset gap
    parities - including a long hold with no reset - so only a real, timing-
    independent latch scores 1.0."""
    T = 20
    Set, Reset = (0, 1), (0, 3)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for set_t, reset_t in ((2, 10), (5, 14), (3, 9), (5, 12), (2, None)):
        pulses = {0: [set_t]}
        events = [(0, 0), (set_t, 1)]
        if reset_t is not None:
            pulses[1] = [reset_t]
            events.append((reset_t, 0))
        trials.append(Trial(_pulse_streams(T, 2, pulses),
                            {'Q': _hold_trace(T, events)}))
    return TemporalTarget('SR latch', [Set, Reset], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          description=describe_target(
        'Input A sets one stored bit; input B resets it. Q must remain in the '
        'selected state without continued input.',
        'Five Set/Reset schedules vary event phase and hold length, including a '
        'Set-only test that must retain the bit through the horizon.'))


def _toggle_trial(T, pulses):
    events, q = [(0, 0)], 0
    for p in pulses:
        q ^= 1
        events.append((p, q))
    return Trial(_pulse_streams(T, 1, {0: pulses}), {'Q': _hold_trace(T, events)})


def toggle_ff(grid_size=5):
    """T flip-flop: each input pulse flips the output (period-2 memory).
    Six trials spanning 2/3-pulse schedules with odd AND even gaps at varied
    phases - a phase-locked ring that only toggles for one spacing fails."""
    T = 24
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    banks = ([3, 9, 14], [4, 12], [2, 7, 12], [3, 10, 15], [5, 11], [2, 8, 15])
    return TemporalTarget('Toggle flip-flop', [In], [out], T,
                          [_toggle_trial(T, p) for p in banks],
                          grid_size=grid_size, iters=30,
                          contract=toggle_contract(),
                          description=describe_target(
        'Each input edge flips the stored bit: quiet to active, then active to '
        'quiet.',
        'Six pulse trains vary phase, count, and odd/even spacing so a '
        'phase-locked shortcut cannot pass.'))


def oscillator(grid_size=5, period=2):
    """Kicked oscillator: a startup pulse injects a value into a loop, which then
    rings on its own (no input needed to sustain - but, correctly, an input IS
    needed to start it: nothing comes from nothing). Output should thereafter
    keep toggling; the exact phase is not scored (only that Q alternates and is
    never stuck), so any circulating loop of the right period qualifies.

    Two trials (kick at different ticks) so a fixed one-shot pulse chain that
    just happens to blip once can't pass - only a genuinely ringing loop does.
    """
    T = 20
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for kick in (0, 3, 5):
        streams = _pulse_streams(T, 1, {0: [kick]})
        # after the kick + settle, require a strict alternation of some phase.
        # score BOTH phases loosely by leaving the absolute phase free: we mark
        # expected as the alternation that starts 1 at the first scored tick;
        # the balanced GA scorer rewards matching the toggling, and a stuck
        # output (all 0 or all 1) can match at most half.
        start = kick + SETTLE + 1
        exp = [None] * T
        phase = 0
        for t in range(start, T):
            exp[t] = 1 if (phase // (period // 2)) % 2 == 0 else 0
            phase += 1
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('Oscillator (period %d)' % period, [In], [out], T,
                          trials, grid_size=grid_size, iters=30,
                          contract=cadence_contract(period, settle=SETTLE),
                          description=describe_target(
        'One input edge starts a free-running output cadence with period %d.'
        % period,
        'Three kick times require silence before the trigger and sustained '
        'oscillation through the end; a finite burst fails.'))


def pattern_generator(grid_size=5, pattern=(1, 0, 0, 0)):
    """Kicked pattern generator (paper section 3: "simple pattern generation circuits
    can be built from these circuits, connected in loops"): one kick pulse must
    start the output repeating `pattern` indefinitely - a loop whose length and
    loading encode the bit sequence.

    The honeycomb is BIPARTITE - every edge joins an (x+y)-even node to an odd
    one (see hexgrid.hex_dirs), so every cycle has even length and a single
    circulating pulse can only produce an EVEN period. An ODD-period pattern
    like 100 is therefore geometrically out of reach on the cheap one-pulse
    route: it would need a SECOND pulse injected half a loop away (output_period
    = loop_length / n_pulses), a conjunction the GA path dips through lower
    fitness to reach and empirically never crosses - neither a bigger grid nor
    200 generations moved it, it just parked on local optima (a one-pulse
    period-6 loop, F1 ~0.67; a 3-spike burst that then dies, F1 ~0.76). So the
    default pattern is 1000: period 4, a single pulse circulating a length-4
    loop read at one cell, which the parity-legal route reaches directly. The
    kick tick and absolute phase are free (phase/latency-invariant scoring)."""
    T = 28
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for kick in (2, 5, 7):
        streams = _pulse_streams(T, 1, {0: [kick]})
        exp = [None] * T
        for i, t in enumerate(range(kick + SETTLE + 1, T)):
            exp[t] = pattern[i % len(pattern)]
        trials.append(Trial(streams, {'Q': exp}))
    pat = ''.join(map(str, pattern))
    cyclic_rises = [i for i, bit in enumerate(pattern)
                    if bit and not pattern[i - 1]]
    # One pulse per cycle is a pure cadence invariant.  Patterns with several
    # uneven pulse offsets still need their richer phase-relative trace scorer.
    cadence_semantics = len(cyclic_rises) == 1
    contract = (cadence_contract(len(pattern), settle=SETTLE)
                if cadence_semantics else state_contract())
    return TemporalTarget('Pattern (%s)' % pat, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          contract=contract,
                          description=describe_target(
        'One input edge starts the repeating output pattern %s (period %d).'
        % (pat, len(pattern)),
        'Three kick times require pre-trigger silence and repetition through the '
        'full observation window.'))


def echo(grid_size=5, delay=3):
    """Echo: output reproduces the input pulse train `delay` ticks later - the
    simplest temporal target (a delay line), a stepping stone to real memory.
    Four pulse trains with varied spacing.  Distinct pulse edges are separated
    by at least one low tick; adjacent high samples would be one held pulse in
    the physical input model, not two events."""
    T = 20
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for pulses in ([2, 7, 9, 13], [4, 10, 16], [3, 5, 9, 14], [6, 12, 14]):
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None] * delay + [streams[t - delay][0] for t in range(delay, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(p + delay) for p in pulses
                                   if p + delay < T]}))
    return TemporalTarget('Echo (delay %d)' % delay, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          contract=event_contract(fit_latency=False),
                          description=describe_target(
        'Reproduce every input edge at Q exactly %d seconds later.' % delay,
        'Four schedules vary pulse count and spacing. A direct input-to-output '
        'connection fails because no additional latency offset is fitted.'))


def coincidence_detector(grid_size=5, latency=1):
    """Two-input coincidence detector - the paper's marquee node capability
    lifted to a circuit: output pulses iff BOTH inputs pulse at the same tick.
    Trials mix simultaneous pairs (fire), pulses staggered by 1-2 ticks (must
    NOT fire - the async coincidence window is what enforces this physically),
    and lone single-channel pulses (must not fire)."""
    T = 20
    A, B = (0, 1), (0, 3)
    out  = OutputTerminal('Q', (2, 2))
    # (pulses_on_A, pulses_on_B) per trial
    banks = [
        ([3, 9, 14], [3, 9, 14]),          # all coincident -> fire each time
        ([3, 10],    [4, 10]),             # stagger 1 then coincide
        ([5, 12],    [7, 13]),             # never coincident
        ([2, 8, 13], [2, 9, 13]),          # coincide, miss, coincide
        ([4],        []),                  # lone pulses -> silence
        ([],         [6]),
    ]
    trials = []
    for pa, pb in banks:
        streams = _pulse_streams(T, 2, {0: pa, 1: pb})
        exp = [None] * latency + [
            1 if (streams[t - latency][0] and streams[t - latency][1]) else 0
            for t in range(latency, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(t + latency)
                                   for t in range(T - latency)
                                   if streams[t][0] and streams[t][1]]}))
    return TemporalTarget('Coincidence (2-in)', [A, B], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          latency=int(latency),
                          contract=event_contract(),
                          description=describe_target(
        'Emit one Q edge only when inputs A and B arrive together.',
        'Six schedules mix coincident pairs, one- and two-second offsets, and '
        'single-input events. Offset and lone events must remain silent.'))


def one_shot(grid_size=5, width=3, latency=3):
    """One-shot / monostable: each input pulse triggers a fixed `width`-tick
    burst at the output, which must then self-terminate - a loop that loads
    itself AND cuts itself off (delayed self-inhibition). Hold windows are
    phase-tolerant (a ringing burst counts); the silence after must be real.
    One unscored tick after each burst tolerates the turn-off transient."""
    T = 24
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    trials = []
    for pulses in ([3], [2, 11], [5, 14], [4, 12, 20]):   # spacing >= width + 4
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None if t < latency else 0 for t in range(T)]
        for p in pulses:
            for t in range(p + latency, min(p + latency + width, T)):
                exp[t] = 1
            if p + latency + width < T:
                exp[p + latency + width] = None            # turn-off transient
        trials.append(Trial(streams, {'Q': exp}))
    return TemporalTarget('One-shot (%d seconds)' % width, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          description=describe_target(
        'Each input edge starts a self-terminating active interval lasting %d '
        'seconds, after which Q must return quiet.' % width,
        'Four schedules include isolated and repeated triggers. Every interval '
        'must terminate and the circuit must re-arm for later input.'))


def pair_detector(grid_size=5, gap=2, latency=3):
    """Double-pulse detector: output fires iff two input pulses arrive exactly
    `gap` ticks apart (out at second pulse + latency) - a delay line feeding a
    coincidence node, timing used as computation. Wrong gaps and lone pulses
    must stay silent."""
    T = 22
    In  = (0, 2)
    out = OutputTerminal('Q', (2, 2))
    banks = [
        [3, 5],                 # gap 2 -> fire
        [4, 8],                 # gap 4 -> silent
        [2, 4, 11, 13],         # two gap-2 pairs -> fire twice
        [6],                    # lone pulse -> silent
        [3, 4, 9, 11],          # gap 1 (no) then gap 2 (yes)
    ]
    trials = []
    for pulses in banks:
        streams = _pulse_streams(T, 1, {0: pulses})
        exp = [None] * (latency + gap) + [
            1 if (streams[t - latency][0] and streams[t - latency - gap][0]) else 0
            for t in range(latency + gap, T)]
        trials.append(Trial(streams, {'Q': exp},
                            {'Q': [float(t + latency)
                                   for t in range(gap, T - latency)
                                   if streams[t][0] and streams[t - gap][0]]}))
    return TemporalTarget('Pair detector (gap %d)' % gap, [In], [out], T, trials,
                          grid_size=grid_size, iters=30,
                          contract=event_contract(),
                          description=describe_target(
        'Emit Q when two input edges are separated by exactly %d seconds.' % gap,
        'Five schedules include valid pairs, wrong gaps, multiple pairs, and a '
        'single edge. Only correctly spaced pairs may produce output.'))


# -- more complex temporal functions (point-event targets; Nervous + LUT) --------
# Built straight from spike_target: each is a handful of (input_spikes,
# output_spikes) cases, so the behaviour is transparent and easy to tweak. Cases
# mix positive and negative examples at varied timings so a net can't pass by
# memorising one schedule. These point-event targets penalise both missing and
# spurious output edges.

def temporal_xor(grid_size=5, latency=1):
    """Temporal XOR - the complement of coincidence: fire iff EXACTLY ONE of the
    two inputs pulses on a tick (both-or-neither -> silent). Needs each input to
    excite the output while the pair mutually inhibits."""
    T = 22
    return spike_target('Temporal XOR (2-in)', [
        ({0: [4],        1: []},        [4 + latency]),          # A only  -> fire
        ({0: [],         1: [7]},       [7 + latency]),          # B only  -> fire
        ({0: [11],       1: [11]},      []),                     # both    -> silent
        ({0: [3, 15],    1: [8, 15]},   [3 + latency, 8 + latency]),  # A,B,then both
        ({0: [5, 9],     1: [9, 18]},   [5 + latency, 18 + latency]), # overlap @9 -> silent
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Emit one Q edge when exactly one of A or B arrives; simultaneous A+B '
        'must cancel.',
        'Five schedules cover A-only, B-only, simultaneous inputs, and mixed '
        'trains with both positive and suppressed events.'))


def ordered_sequence(grid_size=5, gap=3, latency=1):
    """Ordered two-input sequence detector: fire only when A pulses and THEN B
    pulses exactly `gap` ticks later (B-before-A, wrong gaps and lone pulses stay
    silent). Order matters - a delay line on A must meet B at a coincidence node,
    so the reverse order misses."""
    T = 24
    return spike_target('Sequence A->B (gap %d)' % gap, [
        ({0: [4],        1: [4 + gap]},     [4 + gap + latency]),   # A then B  -> fire
        ({0: [10 + gap], 1: [10]},          []),                    # B then A  -> silent
        ({0: [6],        1: [6 + gap + 2]}, []),                    # gap wrong -> silent
        ({0: [3, 14],    1: [3 + gap, 14 + gap]},
                                       [3 + gap + latency, 14 + gap + latency]),
        ({0: [8],        1: []},            []),                    # lone A    -> silent
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Emit Q only for the ordered sequence A then B, separated by %d seconds.'
        % gap,
        'Five schedules include correct order, reverse order, wrong spacing, '
        'repeated valid sequences, and an incomplete sequence.'))


def veto_gate(grid_size=5, latency=1):
    """Inhibited echo: output echoes input A after `latency` ticks UNLESS input B
    pulses on the same tick, which vetoes that echo. The inhibitory routing used
    as a real gate - 'pass A, but B can suppress it'."""
    T = 22
    return spike_target('Veto gate (B blocks A)', [
        ({0: [3, 9, 15],  1: []},        [3 + latency, 9 + latency, 15 + latency]),
        ({0: [4, 10],     1: [10]},      [4 + latency]),          # B blocks the 2nd
        ({0: [6],         1: [6]},       []),                     # B blocks the only A
        ({0: [5, 12, 18], 1: [12]},      [5 + latency, 18 + latency]),
        ({0: [],          1: [7, 14]},   []),                     # B alone -> nothing
    ], T=T, n_inputs=2, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Pass each A edge to Q unless B arrives simultaneously; B alone must '
        'produce nothing.',
        'Five schedules mix unopposed A, vetoed A+B, and B-only events.'))


def burst_generator(grid_size=5, n=3, spacing=2, latency=1):
    """Fan-out / burst: a single input kick produces a fixed BURST of `n` evenly
    spaced output spikes, then silence until the next kick. One edge in, several
    edges out - a delay-line tap or a short re-triggerable ring."""
    T = 22
    def burst(k):
        return [k + latency + i * spacing for i in range(n)]
    return spike_target('Burst x%d' % n, [
        ({0: [3]},         burst(3)),
        ({0: [5]},         burst(5)),
        ({0: [2, 13]},     burst(2) + burst(13)),   # re-triggers
        ({0: [4]},         burst(4)),
    ], T=T, n_inputs=1, latency=latency, grid_size=grid_size,
       description=describe_target(
        'Convert each input edge into %d Q edges separated by %d seconds, then '
        'return to silence.' % (n, spacing),
        'Four schedules vary trigger time and include a two-trigger test that '
        'requires the generator to re-arm.'))


def divide_by_3(grid_size=5, latency=1):
    """Divide-by-3 counter: the output fires on every THIRD input pulse and stays
    silent on the other two - a modulo-3 counter, harder than the toggle (/2)
    because it needs two bits of state, not one."""
    T = 26
    def every3(pulses):
        return [pulses[i] + latency for i in range(2, len(pulses), 3)]
    trains = [[3, 6, 9, 12, 15, 18], [2, 5, 8, 11, 14, 17, 20], [4, 8, 12, 16, 20]]
    return spike_target('Divide-by-3', [({0: tr}, every3(tr)) for tr in trains],
                        T=T, n_inputs=1, latency=latency, grid_size=grid_size,
                        description=describe_target(
        'Emit Q on every third input edge and remain silent on the other two.',
        'Three trains vary phase, spacing, and length so the circuit must retain '
        'the modulo-3 count rather than memorize one schedule.'))


def coincident_temporal_target(target, name=None, gap=None, latency=1,
                               schedules=12, tail=None):
    """Turn a static truth table into a COINCIDENT-EDGE temporal target.

    This is the event-timed twin of `periodic_combinational_target`, and the
    difference between them is the whole point:

    * the periodic wrapper gives every row its own widely-spaced window, several
      grid-widths long, so the circuit settles before it is read. That measures
      combinational logic on a substrate that happens to signal with pulses.
    * this wrapper packs every row into ONE trial at half that spacing, and
      scores exact edge correspondence rather than a settled level. A circuit
      only passes if it also RECOVERS between rows, so lingering state, a stuck
      output, or a slowly ringing path all cost score even when the logic
      itself is right.

    Encoding follows the hand-built `Coincidence (2-in)` / `Temporal XOR (2-in)`
    pair already in this module: an input bit of 1 is one pulse at the row's
    tick, a 0 is silence, and each output bit of 1 is one expected edge
    `latency` later. Row order is permuted per schedule so a fixed output rhythm
    cannot pass.

    A case-valid strobe lane is added under exactly the rule
    `periodic_combinational_target` uses. It is needed more sharply here: with
    no strobe the all-zero row delivers NO input events at all, and these
    substrates are quiescent, so a row that must fire would be physically
    unrepresentable and a row that must stay silent could not be distinguished
    from a circuit that simply never fires.
    """
    cases = list(getattr(target, 'cases', ()) or ())
    if not cases:
        raise ValueError('%s has no truth table to convert'
                         % getattr(target, 'name', target))
    data_inputs = target.n_inputs
    roles = [terminal.role for terminal in target.outputs]
    n_outputs = len(roles)
    zero_outputs = next(
        (output_bits for input_bits, output_bits in cases
         if not any(input_bits)), (0,) * n_outputs)
    has_strobe = (any(zero_outputs)
                  or _zero_row_holds_sole_negative(
                      cases, zero_outputs, n_outputs))
    n_inputs = data_inputs + int(has_strobe)

    span = max(n_inputs, n_outputs, 2)
    grid_size = max(5, 2 * span + 1)
    # The gap has a hard physical floor: a response still crossing the body when
    # the next row lands cannot be attributed to either row, so a gap shorter
    # than the settling transient makes the target unsolvable rather than hard.
    # `periodic_combinational_target` puts that transient at roughly 2 grid
    # widths and leaves 4 for safety. Sit at the 2G diameter: half the periodic
    # spacing, so rows really do crowd each other and a circuit has to recover
    # between them, but not so tight that recovery is physically impossible.
    if gap is None:
        gap = max(4, 2 * grid_size)
    if tail is None:
        tail = gap

    # Present enough NEGATIVE evidence that a blanket responder cannot score
    # well. Event scoring is one-to-one F1, so a circuit that simply fires on
    # every row earns precision equal to the fraction of asserted output bits.
    # On a table like OR or NAND that fraction is 3/4, which is F1 0.857 - high
    # enough to look like a solution. Repeating the quietest rows until at most
    # half of all (row, output) slots are asserted drops that to about 0.67.
    # This weights rows only by their own output bits; no circuit is consulted.
    #
    # Without a strobe a row whose inputs are all 0 delivers no edge at all, so
    # the circuit never physically sees it and it cannot be scored. Drop it
    # rather than counting a phantom presentation in the balance. When such a
    # row carries real evidence, has_strobe is already True and it stays.
    visible = [row for row, (input_bits, _out) in enumerate(cases)
               if has_strobe or any(input_bits)]
    asserted = {row: sum(1 for bit in cases[row][1] if bit) for row in visible}
    weights = {row: 1 for row in visible}
    # Bounded: where the quietest row still asserts a bit (a full adder's
    # rows all do), the ratio only creeps toward 0.5 and chasing it exactly
    # would multiply trial length several-fold for a fraction of a point. Stop
    # at twice the row count; the residual imbalance is far below the level a
    # blanket responder needs to look convincing.
    while len(weights) and sum(weights.values()) < 2 * len(visible):
        total_slots = n_outputs * sum(weights.values())
        filled = sum(asserted[row] * weight for row, weight in weights.items())
        if filled * 2 <= total_slots:
            break
        # Certification rebuilds the same table after shuffling its source
        # rows.  A row INDEX is therefore not a stable tie-break: Full Adder
        # has several equally quiet rows, and choosing the first index made
        # train repeat 001->Sum while a holdout might repeat 011->Carry.  That
        # silently changed the class/input distribution instead of merely the
        # schedule.  Break ties on the Boolean row itself so every rebuild has
        # exactly the same multiset of presentations.
        quietest = min(
            visible,
            key=lambda row: (
                asserted[row], tuple(cases[row][0]), tuple(cases[row][1])))
        weights[quietest] += 1

    ordered = [row for row in visible for _ in range(weights[row])]
    # A rotated/reversed row list still exposes a short, fixed clock pattern:
    # the old temporal AND "solvers" fit those three schedules perfectly and
    # scored 0.0 under one actually shuffled order. Use distinct permutations
    # instead.  Small tables enumerate their available orders directly; wider
    # tables use a deterministic local RNG so target construction remains
    # reproducible across processes without leaking a global random state.
    wanted_schedules = max(1, int(schedules))
    schedule_orders, seen_orders = [], set()

    def add_order(candidate):
        frozen = tuple(candidate)
        if frozen not in seen_orders and len(schedule_orders) < wanted_schedules:
            seen_orders.add(frozen)
            schedule_orders.append(list(frozen))

    if len(ordered) <= 6:
        for candidate in permutations(ordered):
            add_order(candidate)
            if len(schedule_orders) >= wanted_schedules:
                break
    seed = 0xC0FFEE
    for index, row in enumerate(ordered):
        seed = (seed * 1_103_515_245 + (index + 1) * (int(row) + 17)) & 0xFFFFFFFF
    for attempt in range(max(32, wanted_schedules * 16)):
        if len(schedule_orders) >= wanted_schedules:
            break
        candidate = list(ordered)
        random.Random(seed + attempt).shuffle(candidate)
        add_order(candidate)
    # Constant-output tables can have fewer unique permutations than requested.
    # Duplicating the sole schedule there is harmless; it does not create a new
    # timing recipe or conceal a missing input dependency.
    while len(schedule_orders) < wanted_schedules:
        schedule_orders.append(list(ordered))

    T = latency + gap * len(ordered) + tail
    spike_cases = []
    for order in schedule_orders:
        pulses = {index: [] for index in range(n_inputs)}
        events = {role: [] for role in roles}
        for slot, row in enumerate(order):
            tick = latency + slot * gap
            input_bits, output_bits = cases[row]
            for lane, bit in enumerate(input_bits):
                if bit:
                    pulses[lane].append(tick)
            if has_strobe:
                pulses[data_inputs].append(tick)
            for index, role in enumerate(roles):
                if output_bits[index]:
                    events[role].append(tick + latency)
        spike_cases.append((pulses, events))

    ys = [min(grid_size - 1, 1 + 2 * index) for index in range(n_outputs)]
    outputs = [(role, (grid_size - 1, y)) for role, y in zip(roles, ys)]
    base = getattr(target, 'name', 'target')
    strobe_note = (
        ' A case-valid strobe lane pulses on every row, because the all-zero '
        'row carries evidence that silence alone cannot express.'
        if has_strobe else '')
    entry = spike_target(
        name or ('%s (temporal)' % base),
        spike_cases, T=T, n_inputs=n_inputs, latency=latency,
        grid_size=grid_size, outputs=outputs,
        # A valid combinational pipeline may traverse almost one whole row
        # interval before emitting its answer. The generic 12-tick cap made
        # the real 13.5-tick Full Adder physically exact yet unscorable. Fresh
        # row permutations still reject a whole-row phase slip.
        event_max_shift=float(gap),
        description=describe_target(
            'Compute %s from coincident input edges, emitting one output edge '
            'per asserted bit.' % base,
            'All %d row presentations share one trial, %d seconds apart, under '
            '%d row orders, so the circuit must settle and recover between rows '
            'rather than hold one answer. Quiet rows repeat where a table is '
            'lopsided, so firing on everything cannot score well.%s'
            % (len(ordered), gap, len(schedule_orders), strobe_note)))
    entry.temporal_logic_cases = [
        (tuple(input_bits), tuple(output_bits))
        for input_bits, output_bits in cases]
    entry.temporal_logic_data_inputs = data_inputs
    entry.temporal_logic_gap = int(gap)
    entry.temporal_logic_schedules = len(schedule_orders)
    entry.temporal_logic_tail = int(tail)
    return entry


def register_temporal_logic_targets():
    """Add a coincident-edge temporal twin for every combinational table.

    Idempotent, and safe to call from either import order. The truth tables
    live in ``substrates.snn.targets``, which imports this package back through
    ``nervous.contracts`` - so whichever module is imported FIRST reaches the
    other while it is still half-built. Registering from both ends, and giving
    up quietly when the other side is not ready, keeps either order working.

    Returns True when the twins are registered.
    """
    try:
        from substrates.snn.targets import TARGETS as _COMBINATIONAL
    except ImportError:
        # snn.targets is mid-import; it calls us again once TARGETS exists.
        return False
    for target in _COMBINATIONAL.values():
        entry = coincident_temporal_target(target)
        # setdefault so a hand-built or oracle-backed entry always wins a name
        # clash: these are derived twins, not replacements.
        TEMPORAL_TARGETS.setdefault(entry.name, entry)
    return True


# -- registry: one entry per function, best-measuring style (metric shootout) ---
# Judged on HELD-OUT schedules (fresh random timings), oracle-trained circuits
# generalise far better for most input-driven functions (echo hand-trained:
# held-out 0.55 vs oracle 1.00; latch 0.74 vs 0.93) - so those use the oracle
# spec. Coincidence measured BETTER hand-built (held-out f1 0.94 vs 0.68), so it
# stays hand-built. Oscillator / Pattern are autonomous behaviours (no
# input->output relation to sample) and stay hand-built by necessity.
TEMPORAL_TARGETS = {
    'Oscillator':            oscillator(),
    'Pattern (1000)':        pattern_generator(),
    'Coincidence (2-in)':    coincidence_detector(),
    'Temporal XOR (2-in)':   temporal_xor(),
    'Sequence A->B':         ordered_sequence(),
    'Veto gate':             veto_gate(),
    'Burst x3':              burst_generator(),
    'Divide-by-3':           divide_by_3(),
}

# Registry display name -> its reference-oracle spec name (in oracle.ORACLE_SPECS).
# Exposed so held-out certification can recover the reference state machine that
# defines a given target and re-sample fresh validation schedules from it.
ORACLE_KEY_TO_SPEC = {
    'Pulse width sum (A+B)': 'Pulse width sum (oracle)',
    'Odd pulse selector':    'Odd pulse selector (oracle)',
    'A-count parity queried by B': 'A parity query (oracle)',
    'A-count multiple-of-3 queried by B': 'A modulo-3 query (oracle)',
    'Odd A batch closed by B': 'A batch parity query (oracle)',
    'SR latch':              'SR latch (oracle)',
    'Rhythm cascade': 'Rhythm cascade (oracle)',
    'Ring pattern': 'Ring pattern (oracle)',
    'Count to 4': 'Count to 4 (oracle)',
    # One target per substrate, each sited where that substrate's physics is an
    # advantage rather than an obstacle. See the docstrings in oracle.py for why
    # each belongs where it does.
    'Gap band-pass (A->B gap 2-4)': 'Gap band-pass (oracle)',
    'Resettable divide-by-4': 'Resettable divide-by-4 (oracle)',
    'Gated D latch': 'Gated D latch (oracle)',
    'C-element (2-in join)': 'C-element (oracle)',
    'Refractory filter (3 seconds)': 'Refractory filter (oracle)',
    'A-first rendezvous':     'A-first rendezvous (oracle)',
    'Collision serializer (2-to-1)': 'Collision serializer (oracle)',
    'Watchdog timeout (5 seconds)': 'Watchdog timeout (oracle)',
    'Toggle flip-flop':      'Toggle (oracle)',
    'Echo (delay 3)':        'Echo (oracle)',
    'One-shot (12 seconds)': 'One-shot (oracle)',
    'Period doubler (2x)':   'Period doubler (oracle)',
    'Period tripler (3x)':   'Period tripler (oracle)',
    'Period halver (1/2x)':  'Period halver (oracle)',
    'Temporal sum (deltaA + deltaB)': 'Temporal sum (oracle)',
    'Pair detector (gap 2)': 'Pair detector (oracle)',
    'Pair detection gap (2x pulse width)': 'Pair gap 2x width (oracle)',
    'Period stepper':        'Period stepper (oracle)',
    'Gated oscillator':      'Gated oscillator (oracle)',
    'Resettable toggle':     'Resettable toggle (oracle)',
}

# Checkpoints created before time units were presented as seconds retain these
# names. They remain valid for certification, but are not registered as duplicate
# entries in the target picker.
LEGACY_ORACLE_KEY_TO_SPEC = {
    'Refractory filter (3 ticks)': 'Refractory filter (oracle)',
    'Watchdog timeout (5 ticks)': 'Watchdog timeout (oracle)',
    'One-shot (5 ticks)': 'One-shot (oracle)',
    # The one-shot hold was widened to 12 so a single pulse can no longer cover
    # it (see oracle.one_shot_oracle). Checkpoints saved under the old 5-second
    # name still certify against the current spec.
    'One-shot (5 seconds)': 'One-shot (oracle)',
    # This target was displayed with Greek capital delta until the source tree
    # was made pure ASCII. Written as an escape so the name still MATCHES what
    # older checkpoints stored while this file stays ASCII-only.
    'Temporal sum (\u0394A + \u0394B)': 'Temporal sum (oracle)',
}


def _register_oracle_targets():
    import dataclasses
    from .oracle import ORACLE_SPECS
    for key, spec_name in ORACLE_KEY_TO_SPEC.items():
        t = ORACLE_SPECS[spec_name]()
        TEMPORAL_TARGETS[key] = dataclasses.replace(t, name=key)

_register_oracle_targets()

# Registered last so a hand-built or oracle-backed entry always wins a name
# clash. `Coincidence (2-in)`, `Temporal XOR (2-in)` and `Veto gate` stay the
# measured-best versions of temporal AND / XOR / veto; the derived twins give
# every other truth table the same treatment. This call is a no-op when
# snn.targets is the module currently mid-import - it calls back when ready.
register_temporal_logic_targets()
