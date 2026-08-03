"""
substrates/nervous/temporal.py — the temporal evaluation HARNESS over the pulse engine:
running trials, placing outputs, and preparing score bundles.

The nervous net is a temporal system: a pulse injected by an input circulates
around a loop of buffers (a stored bit = a circulating pulse) until inhibition
stops it. The dynamics themselves are the asynchronous edge-triggered pulse
simulation in substrates/nervous/pulse.py (PulseSim). Event/cadence fitness schedules input
edges directly in continuous time; only playback and coverage-scored targets
request sampled tick states (scoring.needs_samples decides).

Core (shared by interactive playback and temporal scoring):
    run_nervous(grid, routing, in_pos, out_pos, streams, T) -> (states, traces)
    prepare_net(genome, ttarget)  -> grown, interpreted, output-placed bundle
    score_temporal(genome, ttarget)                 -> behavioural fitness [0,1]
    loop_profile(grid, routing, in_pos, out_pos)    -> feedback-loop stats

All scoring MATH lives in substrates/nervous/scoring.py — the single scoring contract
(relation registry, matchers, alignment discipline, report body). This module
re-exports every scorer name for the historical `from substrates.nervous.temporal import`
path; new code should import from substrates.nervous.scoring directly.

Loop analysis feeds the GA's fitness shaping (substrates/nervous/ga.py): memory needs a
directed cycle in the signal graph, and a cycle only matters if the inputs can
write into it and it can drive an output.

The target types (Trial, TemporalTarget, presets) live in substrates/nervous/targets.py;
they are re-exported here for back-compat with older imports and pickles.
"""
from __future__ import annotations

from .nervous import (grow_nervous, interpret_nervous, node_delays)
from .io_placement import (io_strategy, bind_io, input_groups, output_groups,
                           flat_inputs, flat_outputs, layout_pads,
                           merge_intervals,
                           growth_seeds, binding_progress,
                           record_binding_progress, terminal_node_sets)
from .hexgrid import hex_dirs
from .pulse import TICK
from .simulation import create_simulator
from .targets import (OutputTerminal, Trial, TemporalTarget,  # noqa: F401
                      TEMPORAL_TARGETS,
                      sr_latch, toggle_ff, oscillator, echo)

# ── scoring re-exports ───────────────────────────────────────────────────────
# All scoring math lives in substrates/nervous/scoring.py (the single scoring contract:
# relation registry, matchers, alignment discipline, report body). The names
# are re-exported here because every historical consumer imports them from
# substrates.nervous.temporal; new code should import from substrates.nervous.scoring directly.
from .scoring import (                                          # noqa: F401
    PhysicalEvents, TemporalTraces,
    best_distinct_assignment, behavior_representatives,
    contract_relations, has_relation, needs_samples,
    METRIC, _trace_metric, _obs_len,
    _expected_windows, _window_score, _role_trace_score,
    _pr_counts, _f1, _pr_score, _pooled_f1, _cand_shifts, _target_pairs,
    _best_shift, _placement_score,
    sampled_events, _role_events, _expected_events, _scored_ranges,
    _event_counts, _event_pairs, _event_candidate_shifts, _pooled_event_f1,
    _best_event_shift, _event_case_score, event_score,
    _role_intervals, _interval_case_score, _waveform_expected,
    _waveform_at_shift, _best_waveform_shift, waveform_score,
    _input_edges, _display_time, event_list_summary, trial_input_summary,
    expected_window_summary,
    _cadence_trial_score, _cadence_at_latency, _best_cadence_latency,
    cadence_score,
    _pulse_events, _stepper_epoch, _stepper_trial_score, _stepper_at_shift,
    _best_stepper_shift, period_stepper_score,
    windowed_score, _REFIT_ALIGNMENT, score_contract, contract_case_count,
    exact_tick_accuracy, _score_output_candidate,
    NV_REPORT_NOTES, LUT_REPORT_NOTES, score_report_lines)
from .contracts import behavior_contract_lines


# ── dynamics (asynchronous pulse engine, sampled per tick) ──────────────────────
# The actual dynamics are event-driven edge-triggered pulses (see pulse.py).
# Playback samples once per tick; event-semantic scoring reads raw timestamps.

def input_cone(grid, routing, in_pos):
    """The cells whose value can be influenced by an input — everything forward-
    reachable from the resolved source pads or compatibility attachments in the
    signal graph. On the nervous net there is no spontaneous activity, so every
    cell OUTSIDE this cone is silent for all time; simulating only the cone is
    exact and, on the unbounded field where growth may leave a large off-cone
    blob, far cheaper. ``in_pos`` may carry per-input attachment GROUPS; the cone
    starts from every attachment cell."""
    edges = signal_graph(grid, routing)          # u -> cells that read u
    seen  = set(p for p in flat_inputs(in_pos) if p in grid)
    stack = list(seen)
    while stack:
        for v in edges.get(stack.pop(), ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return seen


def _inject_stream_edges(sim, in_pos, streams, T):
    """Queue each contiguous high run as one physical wired-OR pulse. An input
    with several attachment cells (evolvable binding) injects the same pulse at
    every site; shared cells wired-OR naturally in the pulse engine."""
    stop = min(int(T), len(streams))
    for input_index, cells in enumerate(input_groups(in_pos)):
        tick = 0
        while tick < stop:
            if input_index >= len(streams[tick]) or not streams[tick][input_index]:
                tick += 1
                continue
            start = tick
            tick += 1
            while (tick < stop and input_index < len(streams[tick])
                   and streams[tick][input_index]):
                tick += 1
            duration = max(sim.config.width, (tick - start) * TICK)
            for cell in cells:
                sim.inject_pulse(cell, start * TICK, duration)


def _inject_physical_events(sim, in_pos, input_events):
    """Queue an explicit floating-time stimulus schedule onto the input nets
    (every attachment cell of an input receives the input's schedule)."""
    for input_index, cells in enumerate(input_groups(in_pos)):
        events = input_events[input_index] if input_index < len(input_events) else ()
        for start, width in events:
            for cell in cells:
                sim.inject_pulse(cell, float(start), float(width))


def _run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True,
                 max_events=None, sample=True, config=None, input_events=None,
                 delays=None, arch='single', terminal_inputs=None,
                 terminal_outputs=None):
    """
    Run T ticks of the asynchronous pulse simulation. streams[t] = tuple of
    input bits (one per in_pos); a 0->1 transition injects a pulse edge onto
    that input cell's net (held 1s are one long pulse). out_pos may contain one
    cell or a wired-OR cell group per role. Returns (states, traces):
        states : list[T] of sampled {(x,y):0/1} activity maps
        traces : {role: [0/1 over T]}
    `prune` restricts the simulation to the input cone (exact for the nervous
    net); off-cone cells are absent from `states` and read as 0 by callers.
    `delays` ({cell: delay}) drives width-preserving transport, which derives
    each output width from its input.
    `arch='tri3'` runs the three-circuit-per-tile substrate on TriSim, which
    presents the same tile-keyed surface so the rest of this function — stimulus
    injection, sampling, event capture — is architecture-agnostic.
    """
    if arch == 'tri3':
        from .tritile import TriSim
        # Tri sub-node fan-out makes routing/hex_dirs pruning inapplicable and
        # the graph is small anyway, so simulate the whole grown organism.
        sim = TriSim(grid, flat_inputs(in_pos), config=config,
                     max_events=max_events, outputs=terminal_outputs)
    else:
        sub = grid
        if prune:
            cone = input_cone(grid, routing, in_pos)
            if len(cone) < len(grid):
                sub = {c: grid[c] for c in cone}
        sim    = create_simulator(sub, routing, max_events=max_events,
                                  config=config, delays=delays,
                                  input_nodes=terminal_inputs,
                                  output_nodes=terminal_outputs)
    # Queue one physical schedule for both sampled and event scoring. The
    # width-preserving engine transports rising and falling edges causally, so
    # it also agrees with incremental ``step`` input; pre-queuing here simply
    # avoids maintaining two stimulus paths.
    if input_events is not None:
        _inject_physical_events(sim, in_pos, input_events)
    else:
        _inject_stream_edges(sim, in_pos, streams, T)
    if not sample:
        # Event/cadence fitness needs edges, not O(T*cells) display snapshots.
        sim.advance_to(T * TICK)
        events = PhysicalEvents(
            sim.rise_times,
            intervals={cell: [tuple(v) for v in values]
                       for cell, values in sim.pulse_intervals.items()})
        return [], {role: [] for role in out_pos}, events, sim.overflow
    states = []
    traces = {role: [] for role in out_pos}
    for t in range(T):
        # sample past the end of the input streams (padding), so a circuit that
        # responds at a DELAY is still observed — its late events fall in
        # [len(streams), T) instead of off the end (see _obs_len).
        sample_time = (t + 0.5) * TICK
        sim.advance_to(sample_time)
        state = sim.activity_at(sample_time)
        states.append(state)
        for role, cells in output_groups(out_pos).items():
            traces[role].append(
                1 if any(state.get(cell, 0) for cell in cells) else 0)
        if sim.overflow:
            break
    # Flush the remaining physical horizon so scoring retains every edge.
    if not sim.overflow:
        sim.advance_to(T * TICK)
    events = PhysicalEvents(
        sim.rise_times,
        intervals={cell: [tuple(v) for v in values]
                   for cell, values in sim.pulse_intervals.items()})
    return states, traces, events, sim.overflow


def run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True,
                config=None):
    """Backward-compatible sampled playback API.

    Dynamics remain asynchronous; these tick samples are for display and legacy
    state-window targets.  Use :func:`run_nervous_events` for behavioral timing.
    """
    states, traces, _, _ = _run_nervous(
        grid, routing, in_pos, out_pos, streams, T, prune=prune,
        config=config)
    return states, traces


def run_nervous_events(grid, routing, in_pos, out_pos, streams, T, prune=True,
                       max_events=None, sample=True, config=None,
                       input_events=None, delays=None,
                       arch='single', terminal_inputs=None,
                       terminal_outputs=None):
    """Run once and return ``(states, traces, rise_times, overflow)``.

    ``rise_times`` maps every simulated cell to its continuous leading-edge
    timestamps.  No tick quantisation is applied to this event record.
    """
    return _run_nervous(grid, routing, in_pos, out_pos, streams, T,
                        prune=prune, max_events=max_events, sample=sample,
                        config=config, input_events=input_events, delays=delays,
                        arch=arch, terminal_inputs=terminal_inputs,
                        terminal_outputs=terminal_outputs)


def _sample_intervals(intervals, ticks):
    """Reconstruct the engine's half-tick samples from physical intervals.

    PulseSim, AnalogPulseSim, and TriSim all define a wire as high on the
    half-open interval ``[start, end)``. Temporal scoring samples at
    ``(tick + 0.5) * TICK``. Replaying those immutable intervals after the run
    is therefore exactly equivalent to building a full-grid state dictionary
    at every tick, while avoiding that allocation in the fitness hot path.
    """
    ordered = tuple(intervals)
    values = [0] * int(ticks)
    interval_index = 0
    for tick in range(int(ticks)):
        when = (tick + 0.5) * TICK
        while (interval_index < len(ordered)
               and ordered[interval_index][1] <= when):
            interval_index += 1
        if (interval_index < len(ordered)
                and ordered[interval_index][0] <= when
                < ordered[interval_index][1]):
            values[tick] = 1
    return tuple(values)


# Outputs are NON-HERITABLE PROBES over the WHOLE organism.
#
# This used to examine only the twelve cells nearest the target's declared
# output coordinate, on the reasoning that scoring every cell costs O(cells) and
# lets a lucky far-off cell inflate fitness without a real mechanism.
#
# The declared coordinate is a label on the target, not a property of the
# organism, so restricting to its neighbourhood asks "did you deliver the answer
# HERE" when the honest question is "where does this organism actually produce
# each answer" — a genome computing the right thing four cells too far away was
# scored as though it had computed nothing.
#
# The inflation worry was the right thing to check, and it was checked rather
# than argued away: raising scores is what CORRECT crediting and lucky-cell
# inflation would BOTH look like, and held-out certification separates them.
# Measured under analog_tri at equal budget, against the same experiment run
# with the local probe:
#
#     Toggle    train 1.000 -> held-out 1.000,  3/3 CERTIFIED  (was 5/5)
#     SR latch  train 0.924 -> held-out 0.906,  2/3 CERTIFIED  (was 3/5)
#
# No OVERFIT verdicts, and a train->held-out gap of 0.018 and 0.000. Circuits
# selected this way generalise, so the extra candidates are finding real
# mechanisms rather than lucky cells. The other half of the original worry —
# that the chosen cell JITTERS between genomes and roughens the gradient — has
# a non-mutating diagnostic in tools/probe_gradient_jitter.py; no full-bank
# result is claimed here yet.
#
# So every mature non-input component is eligible for every role. The cost is
# controlled by collapsing cells that respond IDENTICALLY (they are
# interchangeable as probes) rather than by geography, and compactness survives
# only as a deterministic tie-break inside best_distinct_assignment.
def _output_candidates(grid, in_set):
    """Every mature non-input component is eligible as an output probe."""
    return tuple(sorted(cell for cell in grid if cell not in in_set))


# The retired local probe. SNN still owns a backend-specific local readout;
# Nervous, FNV, and LUT now use whole-organism global fitting. This helper
# remains for legacy callers and controlled comparisons.
LEGACY_OUT_RADIUS = 12


def _local_output_candidates(grid, in_set, term, radius=LEGACY_OUT_RADIUS):
    tx, ty = term.pos
    cands = [c for c in grid if c not in in_set]
    cands.sort(key=lambda c: (abs(c[0] - tx) + abs(c[1] - ty), c))
    return cands[:radius]


def place_outputs_by_trace(grid, routing, in_pos, ttarget, delays=None,
                           arch='single', source_nodes=None,
                           sink_nodes=None):
    """Assign every role jointly over all mature non-input cells.

    Candidate scores use the expected behavior across ALL trials. The global
    injective assignment maximizes total score while keeping roles on distinct
    cells; compact cell order is only a deterministic tie-break.

    ``delays`` ({cell: delay}) drives width-preserving transport; None leaves
    the engine at its configured fixed delay. Returns (out_pos
    {role: (x,y)|None}, traces {role: [trace per trial]}).
    """
    in_set = set(flat_inputs(in_pos))
    out_pos = {term.role: None for term in ttarget.outputs}
    traces  = TemporalTraces()
    if len(grid) <= len(in_set):
        return out_pos, traces
    # One dynamics run per trial, observed to _obs_len (past T) so a delayed
    # output's late events are captured. Sampled traces are reconstructed from
    # the engine's complete physical intervals after each run.
    obs = _obs_len(ttarget)
    relations = set(contract_relations(ttarget))
    need_samples = needs_samples(ttarget)
    config = getattr(ttarget, 'pulse_config', None)
    # Topology is identical across trials.  Computing the exact reachable cone
    # once avoids rebuilding the signal graph for every stimulus schedule.
    # The tri substrate resolves its own sub-node graph, so it simulates the full
    # tile grid (its hex_dirs routing here is not the real signal graph).
    if arch == 'tri3':
        sub = grid
    else:
        cone = input_cone(grid, routing, in_pos)
        sub = grid if len(cone) == len(grid) else {c: grid[c] for c in cone}
    # Probes must be FITTED under the same physics they will be SCORED under.
    # Running placement without the terminal sets and then tracing with them
    # selects a cell for behaviour the final run never reproduces.
    runs = []
    for trial in ttarget.trials:
        run = run_nervous_events(
            sub, routing, in_pos, {}, trial.streams, obs, prune=False,
            max_events=getattr(ttarget, 'max_events', 2048),
            sample=False, config=config,
            input_events=getattr(trial, 'input_events', None), delays=delays,
            arch=arch, terminal_inputs=source_nodes,
            terminal_outputs=sink_nodes)
        runs.append(run)
        # Overflow invalidates the complete behavioral contract. Continuing
        # the remaining schedules and then fitting every output candidate can
        # only spend more time to return the same zero fitness.
        if run[3]:
            traces.overflow = True
            return out_pos, traces
    trial_events = [run[2] for run in runs]
    trial_intervals = [getattr(run[2], 'intervals', {}) for run in runs]
    traces.overflow = False

    def cell_response(c):
        """This cell's complete observed response, across every trial."""
        return (
            tuple(
                _sample_intervals(
                    trial_intervals[ti].get(c, ()), obs)
                if need_samples else ()
                for ti in range(len(ttarget.trials))),
            tuple(tuple(trial_events[ti].get(c, ()))
                  for ti in range(len(ttarget.trials))),
            tuple(tuple(tuple(pair) for pair in trial_intervals[ti].get(c, ()))
                  for ti in range(len(ttarget.trials))))

    # Cells that respond identically are interchangeable as probes, so only
    # enough of each distinct response to still allow one cell per role are
    # scored. This is what keeps whole-organism candidacy affordable.
    response_cache = {}
    def cached_response(cell):
        response = response_cache.get(cell)
        if response is None:
            response = cell_response(cell)
            response_cache[cell] = response
        return response

    candidates = behavior_representatives(
        _output_candidates(grid, in_set), cached_response,
        len(ttarget.outputs))
    candidate_responses = {
        cell: response_cache[cell] for cell in candidates}

    scores = {term.role: {} for term in ttarget.outputs}
    for term in ttarget.outputs:
        for c in candidates:
            ctr, cevents, cintervals, cexp = [], [], [], []
            sampled_response, event_response, interval_response = \
                candidate_responses[c]
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                ctr.append(list(sampled_response[ti])
                           if need_samples else [])
                cevents.append(list(event_response[ti]))
                cintervals.append(list(interval_response[ti]))
                cexp.append(exp)
            if not ctr:
                scores[term.role][c] = 0.0
                continue
            scores[term.role][c] = _score_output_candidate(
                ctr, cevents, cexp, term.role, ttarget,
                traces.overflow, intervals=cintervals)[0]

    # Globally best injective assignment, not role-by-role greedy: a greedy
    # first role can spend the only strong cell and strand a later one.
    assignment = best_distinct_assignment(
        tuple(out_pos), candidates, scores,
        balance_worst=bool(getattr(ttarget, 'combinational_cases', ())))
    if assignment is None:
        return out_pos, traces
    out_pos.update(assignment)
    for term in ttarget.outputs:
        best = out_pos[term.role]
        traces[term.role] = [
            list(candidate_responses[best][0][ti])
            if need_samples else []
            for ti in range(len(ttarget.trials))]
        traces.events[term.role] = [list(trial_events[ti].get(best, ()))
                                    for ti in range(len(ttarget.trials))]
        traces.intervals[term.role] = [
            list(trial_intervals[ti].get(best, ()))
            for ti in range(len(ttarget.trials))]
    return out_pos, traces


def trace_fixed_outputs(grid, routing, in_pos, out_pos, ttarget,
                        delays=None, arch='single', source_nodes=None):
    """Run target trials at already-selected nervous-net output cells.

    Unlike :func:`place_outputs_by_trace`, this function performs no search.
    It is the evaluation path for validation schedules: output identity is a
    fitted model parameter and must remain unchanged after training.
    """
    groups = output_groups(out_pos)
    if any(pos not in grid for pos in flat_outputs(out_pos)):
        return None
    obs = _obs_len(ttarget)
    need_samples = needs_samples(ttarget)
    config = getattr(ttarget, 'pulse_config', None)
    if source_nodes is None:
        terminal_inputs, terminal_outputs = terminal_node_sets(
            ttarget, in_pos, out_pos)
    else:
        # A FROZEN set supplied by the caller. Held-out validation passes the
        # pads it fitted with; re-resolving them here would let validation
        # quietly choose a different input binding from training.
        terminal_inputs = set(source_nodes)
        # The selected outputs are observation probes, not physical sink
        # terminals. They must retain their ordinary outgoing connections,
        # exactly as they did while place_outputs_by_trace fitted them.
        terminal_outputs = set()
    if arch == 'tri3':
        sub = grid
    else:
        cone = input_cone(grid, routing, in_pos)
        sub = grid if len(cone) == len(grid) else {c: grid[c] for c in cone}
    runs = []
    for trial in ttarget.trials:
        run = run_nervous_events(
            sub, routing, in_pos, out_pos, trial.streams, obs,
            prune=False, max_events=getattr(ttarget, 'max_events', 2048),
            sample=False, config=config,
            input_events=getattr(trial, 'input_events', None),
            delays=delays, arch=arch,
            terminal_inputs=terminal_inputs,
            terminal_outputs=terminal_outputs)
        if run[3]:
            return TemporalTraces(overflow=True)
        runs.append(run)
    role_intervals = {
        role: [merge_intervals(
                   [getattr(run[2], 'intervals', {}).get(cell, ())
                    for cell in cells])
               for run in runs]
        for role, cells in groups.items()}
    traces = TemporalTraces(
        {
            role: [list(_sample_intervals(intervals, obs))
                   if need_samples else []
                   for intervals in role_intervals[role]]
            for role in groups
        },
        events={
            role: [[start for start, _ in intervals]
                   for intervals in role_intervals[role]]
            for role in groups},
        intervals=role_intervals,
        overflow=False)
    return traces


def prepare_net(genome, ttarget):
    """Grow + interpret a genome for a temporal target, placing outputs by
    trace match. Returns (grid, routing, in_pos, out_pos, traces) — where
    traces[role][i] is the chosen cell's trace in trial i — or None if the net
    is unusable (too small, no candidate output cells, or an input seed dead).

    Current genomes grow from ``input_layout`` and fit global probes. Legacy
    fixed-input genomes use target pads; direct compatibility strategies may
    instead bind cells through ``substrates.nervous.io_placement``."""
    strategy = io_strategy(ttarget)
    # Native layouts take precedence inside growth_seeds. Without one, fixed
    # binding uses target pads, spatial compatibility uses inherited anchors,
    # and the other compatibility strategies retain one neutral centre.
    grid = grow_nervous(genome, seeds=growth_seeds(
                            ttarget, strategy, genome),
                        grid_size=ttarget.grid_size, iters=ttarget.iters)
    return prepare_net_grid(genome, ttarget, grid, strategy=strategy)


def prepare_net_grid(genome, ttarget, grid, strategy=None,
                     record_progress=True):
    """``prepare_net`` for a body that has ALREADY been grown.

    Split out so lifespan scoring (runtime/escape.py) can interpret several
    developmental snapshots of one organism through exactly this code path
    instead of a parallel copy of it. ``record_progress`` is false for juvenile
    bodies: I/O binding progress describes the ADULT organism, and letting a
    half-grown snapshot overwrite it would corrupt the selection-only wiring
    viability that rank_key reads."""
    if strategy is None:
        strategy = io_strategy(ttarget)
    if record_progress and strategy in (
            'terminal_nodes', 'wiring_chromosome', 'spatial_chromosome'):
        record_binding_progress(
            genome, binding_progress(genome, grid, ttarget))
    if len(grid) <= ttarget.n_inputs:
        return None
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, _ = interpret_nervous(grid, ttarget, arch=arch)
    # An evolved layout replaces the target's declared pads outright. Every pad
    # must have survived development: a pad that failed to grow is a genuinely
    # unbindable phenotype, not something to relocate to the nearest live cell.
    pads = layout_pads(genome, ttarget)
    if pads is not None:
        if not pads or any(cell not in grid for cell in pads):
            return None
        in_pos = list(pads)
    elif strategy == 'fixed' and any(p not in grid for p in in_pos):
        return None                     # a dead seed pad (fixed binding only)
    # Width-preserving transport: build the per-cell delays from the genome's
    # node-type delay vector (node_delays returns None for every other model,
    # and that model derives each output width from its incoming waveform).
    # The tri substrate runs only uniform/analog physics (the delay vector is a
    # single-tile node-type feature), so it never consults it.
    config = getattr(ttarget, 'pulse_config', None)
    delays = None if arch == 'tri3' else node_delays(genome, grid, config)

    if pads is None and strategy != 'fixed':
        bound = bind_io(genome, grid, ttarget, strategy)
        if bound is None:
            return None
        in_pos, out_pos = bound          # in_pos: attachment GROUPS per input
        if any(cell not in grid for cell in flat_inputs(in_pos)):
            return None
        traces = trace_fixed_outputs(grid, routing, in_pos, out_pos, ttarget,
                                     delays=delays, arch=arch)
        if traces is None:
            return None
        return grid, routing, in_pos, out_pos, traces

    # Explicit source membership, resolved ONCE and used for both fitting and
    # tracing. Empty for a fixed-input genome, which keeps the legacy wired-OR
    # input semantics untouched.
    source_nodes = ({tuple(cell) for cell in pads} if pads else None)
    out_pos, traces = place_outputs_by_trace(grid, routing, in_pos, ttarget,
                                             delays=delays,
                                             arch=arch,
                                             source_nodes=source_nodes)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, routing, in_pos, out_pos, traces


def score_temporal(genome, ttarget):
    """Behavioural fitness in [0,1] through the shared target contract."""
    prep = prepare_net(genome, ttarget)
    if prep is None:
        return 0.0
    _, _, _, _, traces = prep
    return score_contract(traces, ttarget)[0]


def score_temporal_plastic(genome, ttarget, samples=8, seed=0, step=None,
                           return_settings=False):
    """Locally tune heritable propagation delays without changing the circuit.

    Growth and the trace-fitted output readout happen once. Starting at the
    genome's inherited ``state_delays``, each tuning step nudges ONE routing state
    up or down by ``exp(step)`` and keeps the change only when it improves the
    score. Only routing states present in the grown body are considered. This is
    deliberately a fine, topology-preserving coordinate search: it never redraws
    the whole delay vector and never changes the output cell while judging a
    delay adjustment.

    The function itself does not mutate ``genome``. ``return_settings=True``
    reports the locally improved vector so the GA can copy it into a breeder and
    make the adjustment heritable. Returns ``(best_score, best_cases)`` — or
    ``(best_score, best_cases, {'state_delays': vector|None})`` when settings are
    requested — or None if the target does not use supported fixed binding.
    """
    import math as _math
    import random as _random
    from .genome import (DELAY_MULT_MIN, DELAY_MULT_MAX, DELAY_LOG_STEP, MAX_STATE,
                         default_state_delays)
    step = DELAY_LOG_STEP if step is None else float(step)

    if io_strategy(ttarget) != 'fixed':
        return None                      # prototype: fixed-I/O growth only
    n_cases = contract_case_count(ttarget)
    arch = getattr(genome, 'arch', 'single')

    def _ret(score, cases, mult):
        if return_settings:
            return score, cases, {'state_delays': mult}
        return score, cases

    grid = grow_nervous(
        genome, seeds=growth_seeds(ttarget, 'fixed', genome),
        grid_size=ttarget.grid_size, iters=ttarget.iters)
    if len(grid) <= ttarget.n_inputs:
        return _ret(0.0, (0.0,) * n_cases, None)
    routing, in_pos, _ = interpret_nervous(grid, ttarget, arch=arch)
    if any(pos not in grid for pos in in_pos):
        return _ret(0.0, (0.0,) * n_cases, None)
    config = getattr(ttarget, 'pulse_config', None)

    # Establish the inherited phenotype and choose its readout once. The readout
    # stays fixed below so an apparent timing improvement cannot actually be a
    # lucky jump to a different output cell.
    base_delays = None if arch == 'tri3' else node_delays(genome, grid, config)
    best_score, best_cases, best_mult = -1.0, None, None
    out0, traces0 = place_outputs_by_trace(
        grid, routing, in_pos, ttarget, delays=base_delays, arch=arch)
    if all(out0.get(t.role) is not None for t in ttarget.outputs):
        best_score, best_cases, _ = score_contract(traces0, ttarget)
    tune_delays = (arch != 'tri3' and config is not None
                   and getattr(config, 'model', 'uniform') == 'pulse_delay')
    if best_cases is None or not tune_delays or step <= 0:
        return _ret(best_score if best_cases is not None else 0.0,
                    best_cases or (0.0,) * n_cases, None)

    inherited = getattr(genome, 'state_delays', None)
    current_mult = default_state_delays()
    if inherited:
        copied = min(len(inherited), MAX_STATE)
        current_mult[:copied] = list(inherited[:copied])
    active_states = sorted({
        state & 0x1F for state in grid.values()
        if 0 < (state & 0x1F) < MAX_STATE
    })
    if not active_states:
        return _ret(best_score, best_cases, None)

    rng = _random.Random(seed)
    for _ in range(max(0, int(samples))):
        state_index = rng.choice(active_states)
        direction = -1.0 if rng.random() < 0.5 else 1.0
        candidate = list(current_mult)
        candidate[state_index] = min(
            DELAY_MULT_MAX,
            max(DELAY_MULT_MIN,
                candidate[state_index] * _math.exp(direction * step)))
        if candidate[state_index] == current_mult[state_index]:
            continue
        delays = {
            pos: config.delay * candidate[state & 0x1F]
            for pos, state in grid.items()
        }
        traces = trace_fixed_outputs(
            grid, routing, in_pos, out0, ttarget, delays=delays, arch=arch)
        if traces is None or getattr(traces, 'overflow', False):
            continue
        score, cases, _ = score_contract(traces, ttarget)
        if score > best_score:
            current_mult = candidate
            best_score, best_cases, best_mult = score, cases, list(candidate)
    return _ret(best_score, best_cases, best_mult)


def temporal_report(ttarget, genome=None):
    """Human-readable explanation of a temporal target for the GUI: what the
    circuit must do, its stimulus tests, and — when a genome is given — the
    evolved net's actual behaviour with a score for each test. The report body
    is the shared scoring.score_report_lines; only growth/prep is
    nervous-specific here."""
    lines = ['Target: %s   [temporal nervous net]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    relations = set(contract_relations(ttarget))
    prep = None if genome is None else prepare_net(genome, ttarget)
    if relations & {'sustained_cadence', 'commanded_cadence'}:
        # rhythm modes measure real output; no expectation-only preview exists
        note = NV_REPORT_NOTES.get(next(iter(relations), 'logical_state'))
        pre = ['', note] if note else []
        if genome is None:
            return '\n'.join(lines + [''] + behavior_contract_lines(ttarget) + pre + [
                '', '(run the GA or Load Saved to inspect a circuit)'])
        if prep is None:
            return '\n'.join(lines + [''] + behavior_contract_lines(ttarget) + pre + [
                '', '(circuit incomplete - grew too little or inputs dead)'])
    if genome is not None and prep is None:
        lines += ['', '(circuit incomplete — grew too little or inputs dead)']
    traces = prep[4] if prep is not None else None
    out_pos = prep[3] if prep is not None else None
    _, body = score_report_lines(ttarget, traces, out_pos,
                                 notes=NV_REPORT_NOTES)
    lines += [''] + body        # blank line between the static description and
                                # the contract, matching the cadence path above
    if traces is None and genome is None:
        lines += ['', '(run the GA or Load Saved to inspect a circuit)']
    return '\n'.join(lines)


# ── feedback-loop analysis (for the GA's loop-aware shaping) ────────────────────

def signal_graph(grid, routing):
    """Directed signal graph as {node: set(readers)}: u -> v iff node v's
    routing reads neighbour u (excitatory or inhibitory)."""
    edges = {c: set() for c in grid}
    for v, entry in routing.items():
        e1, e2, i1 = entry[0], entry[1], entry[2]
        nb = hex_dirs(*v)
        for d in (e1, e2, i1):
            if d is None:
                continue
            u = nb[d]
            if u in edges:
                edges[u].add(v)
    return edges


def _reachable(edges, sources):
    seen, stack = set(), [s for s in sources if s in edges]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(edges[u])
    return seen


def cycle_nodes(edges):
    """Nodes that lie on a directed cycle (can reach themselves). Grids are
    tiny (<= ~50 nodes), so per-node reachability is plenty fast."""
    on_cycle = set()
    for s in edges:
        seen, stack = set(), list(edges[s])
        while stack:
            u = stack.pop()
            if u == s:
                on_cycle.add(s)
                break
            if u in seen or u not in edges:
                continue
            seen.add(u)
            stack.extend(edges[u])
    return on_cycle


def loop_profile(grid, routing, in_pos, out_pos):
    """Feedback-loop stats of a grown net:
        n_cycle    — nodes on any directed cycle
        n_relevant — cycle nodes both writable from an input and driving an output
    A latch is exactly a relevant cycle: inputs can set/clear the circulating
    value and the output can read it."""
    edges = signal_graph(grid, routing)
    cyc   = cycle_nodes(edges)
    if not cyc:
        return {'n_cycle': 0, 'n_relevant': 0}
    from_in = _reachable(edges, flat_inputs(in_pos))
    rev = {c: set() for c in edges}
    for u, vs in edges.items():
        for v in vs:
            rev[v].add(u)
    to_out = _reachable(rev, flat_outputs(out_pos))
    return {'n_cycle': len(cyc), 'n_relevant': len(cyc & from_in & to_out)}


# ── structural topology (selection tie-break, target-agnostic) ─────────────────

def nervous_topology(grid, routing, in_pos, arch='single'):
    """Measure this organism's EFFECTIVE routing as a directed graph.

    The graph is the physical wiring the engine will actually run, not the
    genome and not the target:

    * single tile — one output net per cell, so an edge ``u -> v`` exists iff
      v's routing reads neighbour u (excitatory or inhibitory);
    * tri3 — the CHANNEL/sub-node graph from ``interpret_tri``. Measuring tri
      at tile level would fuse three electrically separate circuits into one
      node and invent paths and loops that no signal can take.

    Input pads have outgoing edges but no incoming effective edge (they are
    source-only), so a pad can never be counted inside a cycle.
    """
    from substrates.topology import EMPTY, measure
    if not grid:
        return EMPTY
    sources = [tuple(cell) for cell in flat_inputs(in_pos)]
    if arch == 'tri3':
        from .tritile import interpret_tri
        info = interpret_tri(grid, sources)
        # sources[v] = the sub-nodes feeding v; invert into (u -> v) wires.
        edges = [(feeder, node)
                 for node, feeders in info['sources'].items()
                 for feeder in feeders
                 if feeder is not None]
        pads = [node for tile, node in info['in_nodes'].items()]
        return measure(edges, pads, nodes=info['nodes'])
    readers = signal_graph(grid, routing)
    edges = [(u, v) for u, listeners in readers.items() for v in listeners]
    return measure(edges, sources, nodes=grid)
