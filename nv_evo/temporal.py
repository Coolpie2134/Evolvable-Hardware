"""
nv_evo/temporal.py — the temporal evaluation HARNESS over the pulse engine:
running trials, placing outputs, and preparing score bundles.

The nervous net is a temporal system: a pulse injected by an input circulates
around a loop of buffers (a stored bit = a circulating pulse) until inhibition
stops it. The dynamics themselves are the asynchronous edge-triggered pulse
simulation in nv_evo/pulse.py (PulseSim). Event/cadence fitness schedules input
edges directly in continuous time; only playback and coverage-scored targets
request sampled tick states (scoring.needs_samples decides).

Core (shared by interactive playback and temporal scoring):
    run_nervous(grid, routing, in_pos, out_pos, streams, T) -> (states, traces)
    prepare_net(genome, ttarget)  -> grown, interpreted, output-placed bundle
    score_temporal(genome, ttarget)                 -> behavioural fitness [0,1]
    loop_profile(grid, routing, in_pos, out_pos)    -> feedback-loop stats

All scoring MATH lives in nv_evo/scoring.py — the single scoring contract
(relation registry, matchers, alignment discipline, report body). This module
re-exports every scorer name for the historical `from nv_evo.temporal import`
path; new code should import from nv_evo.scoring directly.

Loop analysis feeds the GA's fitness shaping (nv_evo/ga.py): memory needs a
directed cycle in the signal graph, and a cycle only matters if the inputs can
write into it and it can drive an output.

The target types (Trial, TemporalTarget, presets) live in nv_evo/targets.py;
they are re-exported here for back-compat with older imports and pickles.
"""
from __future__ import annotations

from .nervous import (grow_nervous, interpret_nervous, node_delays)
from .hexgrid import hex_dirs
from .pulse import TICK
from .simulation import create_simulator
from .targets import (OutputTerminal, Trial, TemporalTarget,  # noqa: F401
                      TEMPORAL_TARGETS,
                      sr_latch, toggle_ff, oscillator, echo)

# ── scoring re-exports ───────────────────────────────────────────────────────
# All scoring math lives in nv_evo/scoring.py (the single scoring contract:
# relation registry, matchers, alignment discipline, report body). The names
# are re-exported here because every historical consumer imports them from
# nv_evo.temporal; new code should import from nv_evo.scoring directly.
from .scoring import (                                          # noqa: F401
    PhysicalEvents, TemporalTraces,
    RELATIONS, RelationSpec, relation_spec, needs_samples,
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
    windowed_score, _REFIT_ALIGNMENT, score_temporal_bundle,
    exact_tick_accuracy, _score_output_candidate,
    NV_REPORT_NOTES, LUT_REPORT_NOTES, score_report_lines)


# ── dynamics (asynchronous pulse engine, sampled per tick) ──────────────────────
# The actual dynamics are event-driven edge-triggered pulses (see pulse.py).
# Playback samples once per tick; event-semantic scoring reads raw timestamps.

def input_cone(grid, routing, in_pos):
    """The cells whose value can be influenced by an input — everything forward-
    reachable from the input seeds in the signal graph. On the nervous net there
    is no spontaneous activity, so every cell OUTSIDE this cone is silent for all
    time; simulating only the cone is exact and, on the unbounded field where
    growth may leave a large off-cone blob, far cheaper."""
    edges = signal_graph(grid, routing)          # u -> cells that read u
    seen  = set(p for p in in_pos if p in grid)
    stack = list(seen)
    while stack:
        for v in edges.get(stack.pop(), ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return seen


def _inject_stream_edges(sim, in_pos, streams, T):
    """Queue each contiguous high run as one physical wired-OR pulse."""
    stop = min(int(T), len(streams))
    for input_index, cell in enumerate(in_pos):
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
            sim.inject_pulse(cell, start * TICK, duration)


def _inject_physical_events(sim, in_pos, input_events):
    """Queue an explicit floating-time stimulus schedule onto the input nets."""
    for input_index, cell in enumerate(in_pos):
        events = input_events[input_index] if input_index < len(input_events) else ()
        for start, width in events:
            sim.inject_pulse(cell, float(start), float(width))


def _run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True,
                 max_events=None, sample=True, config=None, input_events=None,
                 delays=None, arch='single'):
    """
    Run T ticks of the asynchronous pulse simulation. streams[t] = tuple of
    input bits (one per in_pos); a 0->1 transition injects a pulse edge onto
    that input cell's net (held 1s are one long pulse). out_pos is
    {role: (x,y)}. Returns (states, traces):
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
        sim = TriSim(grid, in_pos, config=config, max_events=max_events)
    else:
        sub = grid
        if prune:
            cone = input_cone(grid, routing, in_pos)
            if len(cone) < len(grid):
                sub = {c: grid[c] for c in cone}
        sim    = create_simulator(sub, routing, max_events=max_events,
                                  config=config, delays=delays)
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
        for role, p in out_pos.items():
            traces[role].append(state.get(p, 0) if p else 0)
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
                       arch='single'):
    """Run once and return ``(states, traces, rise_times, overflow)``.

    ``rise_times`` maps every simulated cell to its continuous leading-edge
    timestamps.  No tick quantisation is applied to this event record.
    """
    return _run_nervous(grid, routing, in_pos, out_pos, streams, T,
                        prune=prune, max_events=max_events, sample=sample,
                        config=config, input_events=input_events, delays=delays, arch=arch)


# The output is read at a cell NEAR the target's terminal, not anywhere in the
# organism. On the unbounded field a net can be hundreds of cells, and scoring
# every one both costs O(cells) per genome and lets a lucky far-off cell inflate
# the fitness without a real mechanism (the output location then jitters between
# genomes, wrecking the gradient). Restricting to the OUT_RADIUS-nearest cells
# to the terminal keeps placement cheap and the fitness signal about a fixed,
# local output — the terminal is a designated read-out point, after all.
OUT_RADIUS = 12


def _output_candidates(grid, in_set, term):
    tx, ty = term.pos
    cands = [c for c in grid if c not in in_set]
    cands.sort(key=lambda c: (abs(c[0] - tx) + abs(c[1] - ty), c))
    return cands[:OUT_RADIUS]


def place_outputs_by_trace(grid, routing, in_pos, ttarget, delays=None, arch='single'):
    """Assign each output role the live non-input cell — among those nearest its
    terminal (see OUT_RADIUS) — whose activity trace best matches the expected
    trace across ALL trials (ties broken by distance to the terminal, then cell
    order). Evolution builds the computation and lands it near the read-out.

    ``delays`` ({cell: delay}) drives width-preserving transport; None leaves
    the engine at its configured fixed delay. Returns (out_pos
    {role: (x,y)|None}, traces {role: [trace per trial]}).
    """
    in_set = set(in_pos)
    out_pos = {term.role: None for term in ttarget.outputs}
    traces  = TemporalTraces()
    if len(grid) <= len(in_set):
        return out_pos, traces
    # one dynamics run per trial, observed to _obs_len (past T) so a delayed
    # output's late events are captured; every cell's trace falls out of the states
    obs = _obs_len(ttarget)
    mode = getattr(ttarget, 'score_mode', 'trace')
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
    runs = [run_nervous_events(
                sub, routing, in_pos, {}, tr.streams, obs, prune=False,
                max_events=getattr(ttarget, 'max_events', 2048),
                sample=need_samples, config=config,
                input_events=getattr(tr, 'input_events', None), delays=delays, arch=arch)
            for tr in ttarget.trials]
    trial_states = [run[0] for run in runs]
    trial_events = [run[2] for run in runs]
    trial_intervals = [getattr(run[2], 'intervals', {}) for run in runs]
    traces.overflow = any(run[3] for run in runs)
    used = set()
    for term in ttarget.outputs:
        best, best_key, best_aux = None, None, None
        score_cache = {}
        for c in _output_candidates(grid, in_set, term):
            if c in used:
                continue
            ctr, cevents, cintervals, cexp = [], [], [], []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                # a trial that overflowed breaks its sampling loop early, so its
                # state list can be shorter than obs — pad the missing ticks with
                # 0 (overflow already forces this candidate's score to 0) instead
                # of indexing off the end.
                si = trial_states[ti]
                ctr.append([si[t].get(c, 0) if t < len(si) else 0
                            for t in range(obs)] if need_samples else [])
                cevents.append(list(trial_events[ti].get(c, ())))
                cintervals.append(list(trial_intervals[ti].get(c, ())))
                cexp.append(exp)
            if not ctr:
                s, aux = 0.0, None
            else:
                source = (cintervals if mode == 'waveform' else
                          cevents if mode in ('events', 'cadence') else ctr)
                signature = tuple(tuple(seq) for seq in source)
                cached = score_cache.get(signature)
                s, aux = cached if cached is not None else (None, None)
            if ctr and s is None:
                s, aux = _score_output_candidate(
                    ctr, cevents, cexp, term.role, ttarget,
                    traces.overflow, intervals=cintervals)
            if ctr:
                score_cache[signature] = (s, aux)
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best, best_aux = key, c, aux
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        traces[term.role] = [
            [trial_states[ti][t].get(best, 0) if t < len(trial_states[ti]) else 0
             for t in range(obs)]
            if need_samples else []
            for ti in range(len(ttarget.trials))]
        traces.events[term.role] = [list(trial_events[ti].get(best, ()))
                                    for ti in range(len(ttarget.trials))]
        traces.intervals[term.role] = [
            list(trial_intervals[ti].get(best, ()))
            for ti in range(len(ttarget.trials))]
        if len(ttarget.outputs) == 1 and mode == 'events':
            traces._event_result = best_aux
        elif len(ttarget.outputs) == 1 and mode == 'cadence':
            traces._cadence_result = best_aux
    return out_pos, traces


def trace_fixed_outputs(grid, routing, in_pos, out_pos, ttarget, delays=None, arch='single'):
    """Run target trials at already-selected nervous-net output cells.

    Unlike :func:`place_outputs_by_trace`, this function performs no search.
    It is the evaluation path for validation schedules: output identity is a
    fitted model parameter and must remain unchanged after training.
    """
    if any(pos not in grid for pos in out_pos.values()):
        return None
    obs = _obs_len(ttarget)
    need_samples = needs_samples(ttarget)
    config = getattr(ttarget, 'pulse_config', None)
    if arch == 'tri3':
        sub = grid
    else:
        cone = input_cone(grid, routing, in_pos)
        sub = grid if len(cone) == len(grid) else {c: grid[c] for c in cone}
    runs = [run_nervous_events(
                sub, routing, in_pos, out_pos, trial.streams, obs,
                prune=False, max_events=getattr(ttarget, 'max_events', 2048),
                sample=need_samples, config=config,
                input_events=getattr(trial, 'input_events', None), delays=delays, arch=arch)
            for trial in ttarget.trials]
    traces = TemporalTraces(
        {role: [run[1].get(role, []) for run in runs] for role in out_pos},
        events={role: [list(run[2].get(pos, ())) for run in runs]
                for role, pos in out_pos.items()},
        intervals={role: [list(getattr(run[2], 'intervals', {}).get(pos, ()))
                          for run in runs]
                   for role, pos in out_pos.items()},
        overflow=any(run[3] for run in runs))
    return traces


def prepare_net(genome, ttarget):
    """Grow + interpret a genome for a temporal target, placing outputs by
    trace match. Returns (grid, routing, in_pos, out_pos, traces) — where
    traces[role][i] is the chosen cell's trace in trial i — or None if the net
    is unusable (too small, no candidate output cells, or an input seed dead)."""
    grid = grow_nervous(genome, seeds=tuple(ttarget.inputs),
                        grid_size=ttarget.grid_size, iters=ttarget.iters)
    if len(grid) <= ttarget.n_inputs:
        return None
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, _ = interpret_nervous(grid, ttarget, arch=arch)
    if any(p not in grid for p in in_pos):
        return None
    # Width-preserving transport: build the per-cell delays from the genome's
    # node-type delay vector (node_delays returns None for every other model,
    # and that model derives each output width from its incoming waveform).
    # The tri substrate runs only uniform/analog physics (the delay vector is a
    # single-tile node-type feature), so it never consults it.
    config = getattr(ttarget, 'pulse_config', None)
    delays = None if arch == 'tri3' else node_delays(genome, grid, config)
    out_pos, traces = place_outputs_by_trace(grid, routing, in_pos, ttarget,
                                             delays=delays,
                                             arch=arch)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, routing, in_pos, out_pos, traces


def score_temporal(genome, ttarget):
    """Behavioural fitness in [0,1] using the target's declared score mode."""
    prep = prepare_net(genome, ttarget)
    if prep is None:
        return 0.0
    _, _, _, _, traces = prep
    return score_temporal_bundle(traces, ttarget)[0]


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
    spec = relation_spec(ttarget)
    prep = None if genome is None else prepare_net(genome, ttarget)
    if spec.family == 'rhythm':
        # rhythm modes measure real output; no expectation-only preview exists
        note = NV_REPORT_NOTES.get(getattr(ttarget, 'score_mode', 'trace'))
        pre = ['', note] if note else []
        if genome is None:
            return '\n'.join(lines + pre + [
                '', '(run the GA or Load Saved to inspect a circuit)'])
        if prep is None:
            return '\n'.join(lines + pre + [
                '', '(circuit incomplete - grew too little or inputs dead)'])
    if genome is not None and prep is None:
        lines += ['', '(circuit incomplete — grew too little or inputs dead)']
    traces = prep[4] if prep is not None else None
    out_pos = prep[3] if prep is not None else None
    _, body = score_report_lines(ttarget, traces, out_pos,
                                 notes=NV_REPORT_NOTES)
    lines += body
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
    from_in = _reachable(edges, in_pos)
    rev = {c: set() for c in edges}
    for u, vs in edges.items():
        for v in vs:
            rev[v].add(u)
    to_out = _reachable(rev, [p for p in out_pos.values() if p])
    return {'n_cycle': len(cyc), 'n_relevant': len(cyc & from_in & to_out)}
