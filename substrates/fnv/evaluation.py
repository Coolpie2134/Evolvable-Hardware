"""Target adapters and readout fitting for the Functional NV Net."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from substrates.nervous.scoring import (
    PhysicalEvents, TemporalTraces, _obs_len, _score_output_candidate,
    best_distinct_assignment, behavior_representatives,
    contract_case_count, needs_samples, score_contract,
)

from .catalogue import BY_ID
from .genome import (
    functional_input_positions, germline_telomere, is_constructive,
)
from .growth import grow_functional
from .simulation import (
    FunctionalSim, TICK, compile_functional_grid, effective_wiring_edges,
)


@dataclass(frozen=True)
class FunctionalTopology:
    """Target-agnostic computational structure reachable from input pads."""

    reachable_nodes: int = 0
    reachable_edges: int = 0
    integrating_nodes: int = 0
    max_input_convergence: int = 0
    fully_integrating_nodes: int = 0
    distinct_convergence_cones: int = 0
    distinct_behaviors: int = 0
    multi_input_behaviors: int = 0
    fully_input_dependent_behaviors: int = 0
    max_behavioral_inputs: int = 0
    cyclic_nodes: int = 0
    loop_rank: int = 0
    loop_regions: int = 0

    @property
    def connectivity(self):
        return (
            math.log1p(self.reachable_nodes)
            + math.log1p(self.reachable_edges)
            + math.log1p(self.integrating_nodes)
            + math.log1p(self.max_input_convergence)
            + math.log1p(self.fully_integrating_nodes)
        )

    @property
    def feedback(self):
        return (
            math.log1p(self.cyclic_nodes)
            + math.log1p(self.loop_rank)
            + math.log1p(self.loop_regions)
        )

    @property
    def score(self):
        return self.connectivity + self.feedback


def _reachable_from(start, adjacency):
    reached, pending = {start}, [start]
    while pending:
        source = pending.pop()
        for destination in adjacency.get(source, ()):
            if destination not in reached:
                reached.add(destination)
                pending.append(destination)
    return reached


def _strong_components(nodes, adjacency):
    """Iterative Kosaraju decomposition; large telomeres cannot hit recursion."""
    nodes = set(nodes)
    ordered_adjacency = {
        node: tuple(sorted(
            destination for destination in adjacency.get(node, ())
            if destination in nodes))
        for node in nodes
    }
    visited, finish_order = set(), []
    for start in sorted(nodes):
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for destination in reversed(ordered_adjacency[node]):
                if destination not in visited:
                    stack.append((destination, False))

    reverse = {node: set() for node in nodes}
    for source, destinations in ordered_adjacency.items():
        for destination in destinations:
            reverse[destination].add(source)
    assigned, components = set(), []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component, pending = set(), [start]
        assigned.add(start)
        while pending:
            node = pending.pop()
            component.add(node)
            for source in reverse[node]:
                if source not in assigned:
                    assigned.add(source)
                    pending.append(source)
        components.append(component)
    return components


def functional_topology(grid, inputs, *, _compiled=None):
    """Measure connected and cyclic FNV hardware without inspecting behavior."""
    input_set = set(map(tuple, inputs))
    edges = effective_wiring_edges(
        grid, input_set, compiled=_compiled)
    adjacency = {cell: set() for cell in grid}
    for source, destination in edges:
        adjacency.setdefault(source, set()).add(destination)
        adjacency.setdefault(destination, set())

    per_input = [
        _reachable_from(tuple(cell), adjacency)
        for cell in inputs
        if tuple(cell) in grid
    ]
    reachable = set().union(*per_input) if per_input else set()
    reachable_nodes = reachable - input_set
    reachable_edges = tuple(
        (source, destination)
        for source, destination in edges
        if source in reachable and destination in reachable_nodes
    )
    convergence = {
        cell: sum(cell in reached for reached in per_input)
        for cell in reachable_nodes}
    max_input_convergence = max(convergence.values(), default=0)
    incoming = {cell: set() for cell in reachable_nodes}
    for source, destination in reachable_edges:
        incoming.setdefault(destination, set()).add(source)
    # Count physical convergence junctions, not every unary delay downstream
    # of a junction.  The old definition let one gate followed by a long wire
    # masquerade as a rich integrating morphology.
    junctions = {
        cell for cell, sources in incoming.items()
        if len(sources) >= 2 and convergence.get(cell, 0) >= 2}
    integrating_nodes = len(junctions)
    distinct_convergence_cones = len({
        tuple(index for index, reached in enumerate(per_input)
              if cell in reached)
        for cell in junctions})
    fully_integrating_nodes = sum(
        bool(per_input) and convergence[cell] == len(per_input)
        for cell in junctions)

    cyclic_nodes, loop_rank, loop_regions = 0, 0, 0
    for component in _strong_components(reachable_nodes, adjacency):
        internal_edges = sum(
            destination in component
            for source in component
            for destination in adjacency.get(source, ())
        )
        cyclic = (
            len(component) > 1
            or any(node in adjacency.get(node, ()) for node in component)
        )
        if not cyclic:
            continue
        loop_regions += 1
        cyclic_nodes += len(component)
        loop_rank += max(1, internal_edges - len(component) + 1)

    return FunctionalTopology(
        reachable_nodes=len(reachable_nodes),
        reachable_edges=len(reachable_edges),
        integrating_nodes=integrating_nodes,
        max_input_convergence=max_input_convergence,
        fully_integrating_nodes=fully_integrating_nodes,
        distinct_convergence_cones=distinct_convergence_cones,
        cyclic_nodes=cyclic_nodes,
        loop_rank=loop_rank,
        loop_regions=loop_regions,
    )


def logic_behavior_diversity(cases, target):
    """Diagnostic computational repertoire of a static circuit.

    A signature is one node's settled truth table over the exhaustive input
    rows. We count distinct nonconstant signatures and how many logical input
    variables can actually change each one. Desired outputs are deliberately
    never consulted. The result is telemetry only: although it does not inspect
    desired answers, it still describes behavior under the target's input bank
    and therefore must not enter FNV selection.
    """
    rows = [tuple(map(int, input_bits)) for input_bits, _ in target.cases]
    if not rows or not cases:
        return 0, 0, 0, 0
    cells = set().union(*(case.get("node_levels", {}) for case in cases))
    signatures = {
        tuple(int(case.get("node_levels", {}).get(cell, 0))
              for case in cases)
        for cell in cells}
    nonconstant = {signature for signature in signatures
                   if len(set(signature)) > 1}
    row_index = {bits: index for index, bits in enumerate(rows)}
    arities = []
    for signature in nonconstant:
        influence = 0
        for bit in range(len(rows[0])):
            changes = False
            for input_bits, index in row_index.items():
                other = list(input_bits)
                other[bit] ^= 1
                other_index = row_index.get(tuple(other))
                if (other_index is not None
                        and signature[index] != signature[other_index]):
                    changes = True
                    break
            influence += int(changes)
        arities.append(influence)
    n_inputs = len(rows[0])
    return (
        len(nonconstant),
        sum(arity >= 2 for arity in arities),
        sum(arity == n_inputs for arity in arities),
        max(arities, default=0),
    )


def native_component_baseline(target, families=None):
    """Return an informational one-component match, never a run restriction."""
    from .catalogue import normalise_families
    enabled = normalise_families(families)
    cases = (
        getattr(target, "combinational_cases", ())
        or (getattr(target, "cases", ()) if not getattr(
            target, "temporal", False) else ()))
    if cases and len(getattr(target, "outputs", ())) == 1:
        n_inputs = len(cases[0][0])
        table = {
            tuple(input_bits): int(output_bits[0])
            for input_bits, output_bits in cases
        }
        if n_inputs == 2 and "LOGIC" in enabled:
            functions = {
                "AND": lambda a, b: a & b,
                "OR": lambda a, b: a | b,
                "XOR": lambda a, b: a ^ b,
                "VETO(A,!B)": lambda a, b: a & (not b),
                "VETO(B,!A)": lambda a, b: b & (not a),
            }
            for name, function in functions.items():
                if all(table.get((a, b)) == int(function(a, b))
                       for a, b in table):
                    return name
        if (n_inputs == 1 and "DELAY" in enabled
                and all(table.get((bit,)) == bit for bit in (0, 1)
                        if (bit,) in table)):
            return "DELAY"

    name = str(getattr(target, "name", ""))
    if name == "Toggle flip-flop" and "TOGGLE" in enabled:
        return "TOGGLE"
    if name.startswith("Echo (delay ") and "DELAY" in enabled:
        try:
            duration = int(name.removeprefix("Echo (delay ").split(")")[0])
        except ValueError:
            duration = 0
        if duration in (1, 2):
            return f"DELAY{duration}"
    if name.startswith("One-shot (") and "NORMALIZER" in enabled:
        try:
            duration = int(name.removeprefix("One-shot (").split()[0])
        except ValueError:
            duration = 0
        if duration in (1, 2):
            return f"NORMALIZER{duration}"
    if name in ("Coincidence (2-in)", "Temporal XOR", "Veto gate"):
        if "LOGIC" in enabled:
            return {
                "Coincidence (2-in)": "AND",
                "Temporal XOR": "XOR",
                "Veto gate": "VETO",
            }[name]
    return None


def _output_candidates(grid, inputs):
    """Every mature non-source component is eligible as an output probe."""
    return tuple(sorted(cell for cell in grid if cell not in inputs))


def _inject_streams(sim, input_positions, streams, horizon):
    stop = min(int(horizon), len(streams))
    for input_index, cell in enumerate(input_positions):
        tick = 0
        while tick < stop:
            if (input_index >= len(streams[tick])
                    or not streams[tick][input_index]):
                tick += 1
                continue
            start = tick
            tick += 1
            while (tick < stop and input_index < len(streams[tick])
                   and streams[tick][input_index]):
                tick += 1
            sim.inject_pulse(cell, start * TICK, (tick - start) * TICK)


def _inject_events(sim, input_positions, input_events):
    for input_index, cell in enumerate(input_positions):
        events = (input_events[input_index]
                  if input_index < len(input_events) else ())
        for start, width in events:
            sim.inject_pulse(cell, float(start), float(width))


def run_functional_events(grid, input_positions, output_positions, streams,
                          horizon, *, max_events=2048, sample=True,
                          input_events=None, _compiled=None):
    """Run one trial and return snapshots, role traces, physical events, flag."""
    inputs = [tuple(cell) for cell in input_positions]
    sim = FunctionalSim(
        grid, input_nodes=inputs, output_nodes=set(output_positions.values()),
        max_events=max_events, compiled=_compiled)
    if input_events is None:
        _inject_streams(sim, inputs, streams, horizon)
    else:
        _inject_events(sim, inputs, input_events)
    if not sample:
        sim.advance_to(float(horizon) * TICK)
        events = PhysicalEvents(
            sim.rise_times,
            intervals={
                cell: [tuple(interval) for interval in intervals]
                for cell, intervals in sim.pulse_intervals.items()
            })
        return [], {role: [] for role in output_positions}, events, sim.overflow

    states = []
    traces = {role: [] for role in output_positions}
    for tick in range(int(horizon)):
        sample_time = (tick + 0.5) * TICK
        sim.advance_to(sample_time)
        state = sim.activity_at(sample_time)
        states.append(state)
        for role, cell in output_positions.items():
            traces[role].append(state.get(cell, 0) if cell is not None else 0)
        if sim.overflow:
            break
    if not sim.overflow:
        sim.advance_to(float(horizon) * TICK)
    events = PhysicalEvents(
        sim.rise_times,
        intervals={
            cell: [tuple(interval) for interval in intervals]
            for cell, intervals in sim.pulse_intervals.items()
        })
    return states, traces, events, sim.overflow


def _temporal_runs(grid, inputs, target, *, _compiled=None):
    horizon = _obs_len(target)
    sample = needs_samples(target)
    circuit = (
        compile_functional_grid(grid, inputs)
        if _compiled is None else _compiled)
    return [
        run_functional_events(
            grid, inputs, {}, trial.streams, horizon,
            max_events=getattr(target, "max_events", 2048),
            sample=sample,
            input_events=getattr(trial, "input_events", None),
            _compiled=circuit,
        )
        for trial in target.trials
    ], horizon, sample


def place_temporal_outputs(grid, inputs, target, *, _compiled=None):
    out_positions = {terminal.role: None for terminal in target.outputs}
    traces = TemporalTraces()
    input_set = set(inputs)
    if len(grid) <= len(input_set):
        return out_positions, traces
    runs, horizon, sample = _temporal_runs(
        grid, inputs, target, _compiled=_compiled)
    states = [run[0] for run in runs]
    events = [run[2] for run in runs]
    intervals = [getattr(run[2], "intervals", {}) for run in runs]
    traces.overflow = any(run[3] for run in runs)
    candidates = _output_candidates(grid, input_set)
    candidates = behavior_representatives(
        candidates,
        lambda cell: tuple(
            (
                tuple(
                    states[trial_index][tick].get(cell, 0)
                    if tick < len(states[trial_index]) else 0
                    for tick in range(horizon)
                ) if sample else (),
                tuple(events[trial_index].get(cell, ())),
                tuple(tuple(interval) for interval in
                      intervals[trial_index].get(cell, ())),
            )
            for trial_index in range(len(target.trials))
        ),
        len(target.outputs),
    )
    scores = {terminal.role: {} for terminal in target.outputs}
    for terminal in target.outputs:
        for cell in candidates:
            sampled, rises, waves, expected = [], [], [], []
            for trial_index, trial in enumerate(target.trials):
                role_expected = trial.expected.get(terminal.role)
                if role_expected is None:
                    continue
                sampled.append([
                    (states[trial_index][tick].get(cell, 0)
                     if tick < len(states[trial_index]) else 0)
                    for tick in range(horizon)
                ] if sample else [])
                rises.append(list(events[trial_index].get(cell, ())))
                waves.append(list(intervals[trial_index].get(cell, ())))
                expected.append(role_expected)
            score, _ = _score_output_candidate(
                sampled, rises, expected, terminal.role, target,
                traces.overflow, intervals=waves)
            scores[terminal.role][cell] = score

    assignment = best_distinct_assignment(
        tuple(out_positions), candidates, scores,
        balance_worst=bool(getattr(target, "combinational_cases", ())))
    if assignment is None:
        return out_positions, traces
    out_positions.update(assignment)
    for terminal in target.outputs:
        best_cell = out_positions[terminal.role]
        traces[terminal.role] = [
            [
                (states[trial_index][tick].get(best_cell, 0)
                 if tick < len(states[trial_index]) else 0)
                for tick in range(horizon)
            ] if sample else []
            for trial_index in range(len(target.trials))
        ]
        traces.events[terminal.role] = [
            list(events[trial_index].get(best_cell, ()))
            for trial_index in range(len(target.trials))
        ]
        traces.intervals[terminal.role] = [
            list(intervals[trial_index].get(best_cell, ()))
            for trial_index in range(len(target.trials))
        ]
    return out_positions, traces


def trace_fixed_outputs(grid, inputs, output_positions, target,
                        *, _compiled=None):
    """Trace an already-fitted FNV readout on fresh temporal trials."""
    runs, horizon, sample = _temporal_runs(
        grid, inputs, target, _compiled=_compiled)
    traces = TemporalTraces(overflow=any(run[3] for run in runs))
    for role, cell in output_positions.items():
        traces[role] = [
            [
                (run[0][tick].get(cell, 0)
                 if tick < len(run[0]) else 0)
                for tick in range(horizon)
            ] if sample else []
            for run in runs
        ]
        traces.events[role] = [
            list(run[2].get(cell, ()))
            for run in runs
        ]
        traces.intervals[role] = [
            list(getattr(run[2], "intervals", {}).get(cell, ()))
            for run in runs
        ]
    return traces


_UNSET = object()


def _develop_functional(genome, target):
    inputs = list(functional_input_positions(genome, target.inputs))
    if len(inputs) != int(target.n_inputs):
        return None
    grid = grow_functional(
        genome, inputs, grid_size=target.grid_size, iters=target.iters)
    return grid, inputs


def prepare_functional(genome, target, *, _developed=_UNSET,
                       _compiled=None):
    developed = (
        _develop_functional(genome, target)
        if _developed is _UNSET else _developed)
    if developed is None:
        return None
    grid, inputs = developed
    if len(grid) <= len(inputs) or any(cell not in grid for cell in inputs):
        return None
    circuit = (
        compile_functional_grid(grid, inputs)
        if _compiled is None else _compiled)
    if getattr(target, "temporal", False):
        output_positions, traces = place_temporal_outputs(
            grid, inputs, target, _compiled=circuit)
        if any(output_positions[terminal.role] is None
               for terminal in target.outputs):
            return None
        return grid, inputs, output_positions, traces
    output_positions, cases = place_logic_outputs(
        genome, grid, inputs, target, _compiled=circuit)
    if any(output_positions[terminal.role] is None
           for terminal in target.outputs):
        return None
    return grid, inputs, output_positions, cases


def functional_logic_horizon(genome):
    """Continuous-time settling window used by FNV logic fitness.

    It is genotype-derived because FNV growth is bounded by the germline
    telomere rather than by the target's display grid.
    """
    if is_constructive(genome):
        from .construction import constructive_depth
        seeds = tuple(getattr(genome, "input_layout", None) or ((0, 0),))
        return 2 * max(1, constructive_depth(genome, seeds)) + 4
    return 2 * germline_telomere(genome) + 4


def _run_logic_case(genome, grid, inputs, bits, target, *, _compiled=None):
    # FNV growth is unbounded and target.grid_size is ignored. A signal crosses
    # at most the germline telomere in two-tick components, so this is the
    # genotype-derived settling bound rather than a display-grid-derived one.
    horizon = functional_logic_horizon(genome)
    sim = FunctionalSim(grid, input_nodes=inputs,
                        max_events=getattr(target, "max_events", 2048),
                        compiled=_compiled)
    for index, cell in enumerate(inputs):
        if index < len(bits) and bits[index]:
            sim.inject_pulse(cell, 0.0, float(horizon))
    sim.advance_to(float(horizon))
    return dict(sim.ever), dict(sim.levels), sim.overflow


def _bit_parallel_logic_runs(grid, inputs, target, circuit):
    """Exact steady truth tables for acyclic stateless FNV hardware.

    One Python integer packs all declared rows. Delays are identity at settled
    level and fixed gates become bit operations. Stateful components and any
    directed cycle fall back to the continuous-time simulator, preserving the
    physical semantics for precisely the cases where timing/history matters.
    """
    supported = {"AND", "OR", "XOR", "VETO", "DELAY"}
    input_set = set(inputs)
    nodes = [cell for cell in grid if cell not in input_set]
    if any(BY_ID[grid[cell]].behavior not in supported for cell in nodes):
        return None

    dependents = {cell: [] for cell in nodes}
    indegree = {cell: 0 for cell in nodes}
    for cell in nodes:
        for source in circuit.inputs.get(cell, ()):
            if source is not None and source not in input_set:
                if source not in indegree:
                    return None
                dependents[source].append(cell)
                indegree[cell] += 1
    pending = sorted(cell for cell, degree in indegree.items() if degree == 0)
    order = []
    while pending:
        cell = pending.pop()
        order.append(cell)
        for destination in dependents[cell]:
            indegree[destination] -= 1
            if indegree[destination] == 0:
                pending.append(destination)
    if len(order) != len(nodes):
        return None

    row_count = len(target.cases)
    mask = (1 << row_count) - 1
    tables = {}
    for input_index, cell in enumerate(inputs):
        tables[cell] = sum(
            (1 << row_index)
            for row_index, (bits, _expected) in enumerate(target.cases)
            if input_index < len(bits) and bits[input_index])
    for cell in order:
        entry = BY_ID[grid[cell]]
        values = tuple(
            0 if source is None else tables.get(source, 0)
            for source in circuit.inputs.get(cell, ()))
        if entry.behavior == "AND":
            value = values[0] & values[1]
        elif entry.behavior == "OR":
            value = values[0] | values[1]
        elif entry.behavior == "XOR":
            value = values[0] ^ values[1]
        elif entry.behavior == "VETO":
            value = values[0] & (~values[1] & mask)
        else:  # settled DELAY
            value = values[0]
        tables[cell] = value & mask

    return [
        (
            {cell: int(bool(table & (1 << row_index)))
             for cell, table in tables.items()},
            {cell: int(bool(table & (1 << row_index)))
             for cell, table in tables.items()},
            False,
        )
        for row_index in range(row_count)
    ]


def place_logic_outputs(genome, grid, inputs, target, *, _compiled=None):
    input_set = set(inputs)
    if _compiled is None:
        # Keep the public helper's historical call seam: diagnostics/tests may
        # replace _run_logic_case with a five-argument physical oracle.
        runs = [
            _run_logic_case(genome, grid, inputs, bits, target)
            for bits, _ in target.cases
        ]
    else:
        runs = _bit_parallel_logic_runs(grid, inputs, target, _compiled)
        if runs is None:
            runs = [
                _run_logic_case(
                    genome, grid, inputs, bits, target, _compiled=_compiled)
                for bits, _ in target.cases
            ]
    positions = {terminal.role: None for terminal in target.outputs}
    candidates = _output_candidates(grid, input_set)
    candidates = behavior_representatives(
        candidates,
        lambda cell: tuple(run[1].get(cell, 0) for run in runs),
        len(target.outputs),
    )
    scores = {terminal.role: {} for terminal in target.outputs}
    for output_index, terminal in enumerate(target.outputs):
        for cell in candidates:
            correct = sum(
                int(run[1].get(cell, 0) == expected[output_index])
                for run, (_, expected) in zip(runs, target.cases)
            )
            scores[terminal.role][cell] = correct
    assignment = best_distinct_assignment(
        tuple(positions), candidates, scores, balance_worst=True)
    if assignment is not None:
        positions.update(assignment)
    cases = []
    for run, (input_bits, expected) in zip(runs, target.cases):
        acts = {
            terminal.role: (
                run[1].get(positions[terminal.role], 0)
                if positions[terminal.role] is not None else 0)
            for terminal in target.outputs
        }
        cases.append({
            "in_bits": input_bits,
            "out_bits": expected,
            "node_outputs": run[0],
            "node_levels": run[1],
            "acts": acts,
            "overflow": run[2],
        })
    return positions, cases


def score_fixed_logic_outputs(genome, grid, inputs, output_positions, target):
    """Return static case rows for an already-fitted FNV readout."""
    circuit = compile_functional_grid(grid, inputs)
    rows = []
    for bits, _ in target.cases:
        _, levels, overflow = _run_logic_case(
            genome, grid, inputs, bits, target, _compiled=circuit)
        if overflow:
            return None
        rows.append([
            float(levels.get(output_positions[terminal.role], 0))
            for terminal in target.outputs
        ])
    return rows


def evaluate_functional_full(genome, target, *, include_topology=False):
    developed = _develop_functional(genome, target)
    circuit = (
        compile_functional_grid(*developed)
        if developed is not None else None)
    topology = (
        functional_topology(*developed, _compiled=circuit)
        if include_topology and developed is not None
        else FunctionalTopology())
    prepared = prepare_functional(
        genome, target, _developed=developed, _compiled=circuit)
    if prepared is None:
        result = 0.0, (0.0,) * contract_case_count(target)
    else:
        _, _, _, observations = prepared
        if getattr(target, "temporal", False):
            score, cases, _ = score_contract(observations, target)
        else:
            rows = [
                [float(case["acts"][terminal.role])
                 for terminal in target.outputs]
                for case in observations
            ]
            score, cases, _ = score_contract(rows, target)
            diversity = logic_behavior_diversity(observations, target)
            topology = replace(
                topology,
                distinct_behaviors=diversity[0],
                multi_input_behaviors=diversity[1],
                fully_input_dependent_behaviors=diversity[2],
                max_behavioral_inputs=diversity[3])
        result = score, cases
    return (*result, topology) if include_topology else result


def score_functional(genome, target):
    return evaluate_functional_full(genome, target)[0]


def functional_case_outputs(genome, target):
    developed = _develop_functional(genome, target)
    prepared = prepare_functional(
        genome, target, _developed=developed)
    if prepared is None:
        if developed is None:
            return {}, [], {
                terminal.role: None for terminal in target.outputs
            }, []
        grid, inputs = developed
        return grid, inputs, {
            terminal.role: None for terminal in target.outputs
        }, []
    grid, inputs, outputs, observations = prepared
    return grid, inputs, outputs, observations


def functional_report(target, genome=None):
    from substrates.nervous.contracts import behavior_contract_lines
    lines = [f"Target: {target.name}   [Functional NV Net]", ""]
    lines.extend(behavior_contract_lines(target))
    baseline = native_component_baseline(
        target, getattr(target, "_fnv_families", None))
    lines.extend([
        "",
        ("Native single-component baseline: " + baseline
         if baseline else
         "Emergent target: no enabled single FNV component is a full solution."),
    ])
    if genome is None:
        return "\n".join(lines)
    prepared = prepare_functional(genome, target)
    if prepared is None:
        return "\n".join(lines + ["", "(circuit incomplete)"])
    grid, inputs, outputs, observations = prepared
    topology = functional_topology(grid, inputs)
    behavior_diagnostic = (
        logic_behavior_diversity(observations, target)
        if not getattr(target, "temporal", False) else None)
    family_counts = {}
    from .catalogue import BY_ID
    for state in grid.values():
        family = BY_ID[state].family
        family_counts[family] = family_counts.get(family, 0) + 1
    lines.extend([
        "",
        "Circuit: %d components  (%s)" % (
            len(grid),
            ", ".join(f"{family}={count}"
                      for family, count in sorted(family_counts.items()))),
        ("Inputs (%s source pads): " % (
            "evolved relative"
            if getattr(genome, "input_layout", None) is not None
            else "legacy fixed"
        )) + ", ".join(map(str, inputs)),
        "Outputs (globally fitted probes): " + ", ".join(
            f"{role}={cell}" for role, cell in outputs.items()),
        (
            "Topology: score=%.3f, reachable=%d nodes/%d edges, "
            "multi-input=%d, cyclic=%d, loop-rank=%d in %d regions"
        ) % (
            topology.score,
            topology.reachable_nodes,
            topology.reachable_edges,
            topology.integrating_nodes,
            topology.cyclic_nodes,
            topology.loop_rank,
            topology.loop_regions,
        ),
    ])
    if behavior_diagnostic is not None:
        distinct, multi, full, max_inputs = behavior_diagnostic
        lines.append(
            "Behavior diagnostic (not selected): distinct=%d, multi-input=%d, "
            "fully-input-dependent=%d, max-input-dependence=%d" % (
                distinct, multi, full, max_inputs))
    if not getattr(target, "temporal", False):
        lines.append("")
        for case in observations:
            actual = tuple(
                case["acts"][terminal.role] for terminal in target.outputs)
            lines.append(
                "%s -> %s  expected %s" %
                (case["in_bits"], actual, case["out_bits"]))
    return "\n".join(lines)
