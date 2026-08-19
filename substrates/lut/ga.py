"""
substrates/lut/ga.py - scoring and GA for the LUT array.

Problem definitions and executable behavior contracts are shared with the rest
of the project; only the substrate-to-observation adapter differs. Selection uses the same recipe as the
nervous GA - elites + random immigrants + tournament, NO early stop at 1.0 -
but WITHOUT the parsimony tie-break: seeds are dense sim6-style ontogeny
biomorphs (rich morphology), and parsimony pruned them back to sparse diamonds.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from runtime.cache import LRUCache
from runtime.mutation import adaptive_mutation_rate, STRESS_PATIENCE
from runtime.parallel import map_ordered

from substrates.nervous.temporal import (_obs_len, TemporalTraces,
                             _score_output_candidate)
from substrates.nervous.scoring import (contract_relations, needs_samples, score_contract,
                            score_report_lines, LUT_REPORT_NOTES,
                            behavior_representatives, best_distinct_assignment)
from substrates.nervous.contracts import behavior_contract_lines
from .genome import (LUT_STATES, MAX_CHROMS, MAX_TELOMERE,
                     Genome, Chromosome,
                     input_layout_domain, input_layout_radius,
                     random_input_layout,
                     lut_input_positions, lut_exterior_inputs,
                     lut_growth_seeds, lut_io_mode,
                     random_lut_gene, random_lut_chromosome)
from .functions import (
    UNRESTRICTED, allowed_function_table, mutate_function_table,
    normalise_function_families, project_function_table, unrestricted_only,
)
from .lut import SEED_LUT, grow_lut, grow_lut_tracked
from .pulse import AsyncLutSim, TICK


def is_branched(genome):
    """True for the branched, output-rooted encoding (substrates/lut/branched).

    The population machinery is encoding-agnostic; only copy, signature,
    mutate, cross and grow are not. Those dispatch on this.
    """
    from .branched import BranchedLutGenome
    return isinstance(genome, BranchedLutGenome)


def clone_genome(genome):
    """Fast structural copy (shared, never-in-place-mutated gene objects; fresh
    structure) - an identical-behaviour, ~10x cheaper replacement for
    copy.deepcopy on the reproduction hot path. See substrates.nervous.ga.clone_genome."""
    if is_branched(genome):
        # Branched genes ARE edited in place by mutation, so the shared-gene
        # trick would corrupt the parent; this encoding pays for a real copy.
        from .branched_ga import clone_branched_lut
        clone = clone_branched_lut(genome)
        for attribute in ('_io_binding_progress', '_mut_rate'):
            if hasattr(genome, attribute):
                setattr(clone, attribute, getattr(genome, attribute))
        return clone
    clone = Genome(
        chromosomes=[Chromosome(genes=c.genes[:], split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE),
                                wiring=getattr(c, 'wiring', False))
                     for c in genome.chromosomes],
        tag=genome.tag,
        seed_state=getattr(genome, 'seed_state', None),
        provenance=getattr(genome, 'provenance', ''),
        input_layout=(
            None if getattr(genome, 'input_layout', None) is None
            else tuple(tuple(cell) for cell in genome.input_layout)),
        edge_input_layout=(
            None if getattr(genome, 'edge_input_layout', None) is None
            else tuple(int(value) for value in genome.edge_input_layout)))
    if hasattr(genome, '_io_binding_progress'):
        clone._io_binding_progress = genome._io_binding_progress
    if hasattr(genome, '_mut_rate'):
        # Heritable self-adaptive mutation rate; see substrates.nervous.ga.
        clone._mut_rate = genome._mut_rate
    return clone


def constrain_genome_functions(genome, function_families=None):
    """Project every executable LUT state into the selected physical banks.

    Context and ``self_in`` fields remain unrestricted 16-bit CAM patterns;
    only the tables that can actually be installed at runtime are constrained.
    The unrestricted default is a strict no-op.
    """
    families = normalise_function_families(function_families)
    if unrestricted_only(families):
        return genome
    for chromosome in genome.chromosomes:
        if getattr(chromosome, 'wiring', False):
            continue
        for index, gene in enumerate(chromosome.genes):
            value = int(gene.self_out) & 0xFFFF
            if allowed_function_table(value, families):
                continue
            edited = copy.copy(gene)
            edited.self_out = project_function_table(value, families)
            chromosome.genes[index] = edited
    seed = (
        tuple(int(value) & 0xFFFF for value in genome.seed_state)
        if getattr(genome, 'seed_state', None) is not None
        else (SEED_LUT,) * 4)
    genome.seed_state = tuple(
        value if allowed_function_table(value, families)
        else project_function_table(value, families)
        for value in seed)
    return genome

POPSIZE        = 120
ELITE_FRAC     = 0.10        # elites = this fraction of pop, UNLESS ELITE_COUNT set
ELITE_COUNT    = None        # exact elite count (GUI override); None = use ELITE_FRAC
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
EXPLORATION_PARENT_FRAC = 0.30
# Let a small cohort prove the inherited crossover before the normal mutation
# transaction changes it.  These are evaluated offspring, not preserved elites:
# they still survive only if their declared behavior earns a place next round.
RECOMBINATION_EVALUATION_FRACTION = 0.10
MEAN_MUTATIONS = 4.0         # hot-start rate for annealing (see substrates.nervous.ga)
MUT_DECAY      = 0.997       # slow cooldown: 4.0 -> ~0.89 by gen 500
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))  # see substrates.nervous.ga
FITNESS_CACHE_MAX = 200_000  # cap the fitness cache on very long runs

# LUT temporal search has the same flat recurrent landscapes as the nervous
# net; reheat after a plateau instead of cooling indefinitely.
# -- running trials / placing outputs (trace-matched, as in nv) ------------------
# Native LUT I/O follows the NV/FNV probe rule: every mature non-source cell is
# eligible, behaviorally identical candidates are compressed, and all output
# roles are assigned together to distinct non-invasive probes.


def _expand_input_lanes(in_pos):
    """(flat_cells, lane_map) for an in_pos that may carry per-input attachment
    GROUPS (evolvable binding): lane_map[k] is the logical input that drives
    flat_cells[k]. A plain flat in_pos maps 1:1 and is returned as-is, so the
    fixed path is untouched."""
    from substrates.nervous.io_placement import input_groups
    groups = input_groups(in_pos)
    if all(len(g) == 1 for g in groups):
        return [g[0] for g in groups], list(range(len(groups)))
    flat, lanes = [], []
    for index, group in enumerate(groups):
        for cell in group:
            flat.append(cell)
            lanes.append(index)
    return flat, lanes


def _run_lut_trials(grid, in_pos, ttarget, watch_cells,
                    source_nodes=None, sink_nodes=None,
                    external_inputs=None):
    """Run every trial once on the asynchronous engine (substrates.lut.pulse).

    Returns (trial_B, trial_events, overflow): the [obs, ncells] mid-tick
    sample matrix per trial (empty when the score mode consumes edges, not
    samples), the continuous leading-edge trains of the ``watch_cells`` per
    trial, and whether any run blew its event budget. A trial that carries a
    physical ``input_events`` schedule is injected at its real (possibly
    sub-tick) times - the LUT backend no longer quantises such targets.

    ``in_pos`` may carry per-input attachment groups (evolvable binding): each
    logical input's stimulus is replicated onto every attachment cell; a cell
    shared by inputs receives the wired-OR (the sim's wire counters merge
    overlapping injections)."""
    obs = _obs_len(ttarget)                  # observe past T to catch delayed events
    need_samples = needs_samples(ttarget)
    flat_pos, lanes = _expand_input_lanes(in_pos)
    expand = lanes != list(range(len(flat_pos)))
    if source_nodes is None and sink_nodes is None:
        from substrates.nervous.io_placement import terminal_node_sets
        terminal_inputs, terminal_outputs = terminal_node_sets(
            ttarget, in_pos, {'watch': list(watch_cells)})
    else:
        terminal_inputs = set(source_nodes or ())
        terminal_outputs = set(sink_nodes or ())
    sim = AsyncLutSim(
        grid, config=getattr(ttarget, 'lut_config', None),
        input_nodes=terminal_inputs, output_nodes=terminal_outputs,
        external_inputs=external_inputs)
    trial_B, trial_events, trial_intervals, overflow = [], [], [], False
    first = True
    for tr in ttarget.trials:
        if not first:
            sim.reset()
        first = False
        events = getattr(tr, 'input_events', None)
        if events is not None:
            if expand:
                events = [events[i] if i < len(events) else []
                          for i in lanes]
            B = sim.run_input_events(events, flat_pos, obs, sample=need_samples)
        else:
            streams = tr.streams
            if expand:
                streams = [tuple(row[i] if i < len(row) else 0 for i in lanes)
                           for row in tr.streams]
            B = sim.run_bits(streams, flat_pos, obs)
        trial_B.append(B)
        trial_events.append(sim.rise_trains(watch_cells))
        intervals = sim.pulse_intervals
        trial_intervals.append({
            cell: list(intervals.get(cell, ())) for cell in watch_cells})
        overflow = overflow or sim.overflow
    return sim, trial_B, trial_events, trial_intervals, overflow


def place_outputs_by_trace(grid, in_pos, ttarget, source_nodes=None,
                           external_inputs=None):
    """Globally assign roles to distinct mature non-input cells.

    Every candidate is scored across all trials and a joint injective assignment
    maximizes total role score, matching the current Nervous/FNV probe rule.

    The dynamics are the asynchronous engine (substrates.lut.pulse.AsyncLutSim):
    trace/persistence modes score the mid-tick sample matrix (a column slice
    per candidate cell - no per-tick dicts), while event/cadence modes read
    the wires' raw continuous leading-edge timestamps, exactly as the nervous
    backend does."""
    from substrates.nervous.io_placement import flat_inputs
    in_set = set(flat_inputs(in_pos))
    out_pos = {t.role: None for t in ttarget.outputs}
    traces  = TemporalTraces()
    # LUT holds a level (a latch), it does not ring like the nervous net - so a
    # commanded HOLD must be genuinely high on every tick, not satisfied by a
    # sparse 2-3 tick burst under the nervous net's +/-1 ring tolerance. Mark these
    # traces strict (hold_tol=0) so scoring + placement demand the full hold.
    traces.hold_tol = 0
    if len(grid) <= len(in_set.intersection(grid)):
        return out_pos, traces
    watch = tuple(sorted(cell for cell in grid if cell not in in_set))
    if source_nodes is None:
        sim, trial_B, trial_events, trial_intervals, overflow = \
            _run_lut_trials(grid, in_pos, ttarget, watch)
    else:
        sim, trial_B, trial_events, trial_intervals, overflow = \
            _run_lut_trials(
                grid, in_pos, ttarget, watch, source_nodes=source_nodes,
                sink_nodes=set(), external_inputs=external_inputs)
    traces.overflow = overflow
    cidx = sim._cidx
    need_samples = needs_samples(ttarget)
    def cell_response(cell):
        col = cidx[cell]
        return tuple(
            (
                tuple(trial_B[index][:, col].tolist())
                if need_samples else (),
                tuple(trial_events[index].get(cell, ())),
                tuple(tuple(pair)
                      for pair in trial_intervals[index].get(cell, ())),
            )
            for index in range(len(ttarget.trials)))

    candidates = behavior_representatives(
        watch, cell_response, len(ttarget.outputs))
    scores = {term.role: {} for term in ttarget.outputs}
    for term in ttarget.outputs:
        for cell in candidates:
            col = cidx[cell]
            sampled, rises, intervals, expected = [], [], [], []
            for trial_index, trial in enumerate(ttarget.trials):
                role_expected = trial.expected.get(term.role)
                if role_expected is None:
                    continue
                sampled.append(
                    trial_B[trial_index][:, col].tolist()
                    if need_samples else [])
                rises.append(list(trial_events[trial_index].get(cell, ())))
                intervals.append(list(
                    trial_intervals[trial_index].get(cell, ())))
                expected.append(role_expected)
            scores[term.role][cell] = (
                _score_output_candidate(
                    sampled, rises, expected, term.role, ttarget,
                    traces.overflow, tol=0, intervals=intervals)[0]
                if sampled else 0.0)

    assignment = best_distinct_assignment(
        tuple(out_pos), candidates, scores,
        balance_worst=bool(getattr(ttarget, 'combinational_cases', ())))
    if assignment is None:
        return out_pos, traces
    out_pos.update(assignment)
    for term in ttarget.outputs:
        best = out_pos[term.role]
        col = cidx[best]
        traces[term.role] = [
            trial_B[index][:, col].tolist() if need_samples else []
            for index in range(len(ttarget.trials))]
        traces.events[term.role] = [
            list(trial_events[index].get(best, ()))
            for index in range(len(ttarget.trials))]
        traces.intervals[term.role] = [
            list(trial_intervals[index].get(best, ()))
            for index in range(len(ttarget.trials))]
    return out_pos, traces


def trace_fixed_outputs(grid, in_pos, out_pos, ttarget, source_nodes=None,
                        sink_nodes=None):
    """Run LUT trials at pre-selected output cells without validation search."""
    from substrates.nervous.io_placement import (output_groups, flat_outputs,
                                     merge_intervals)
    groups = output_groups(out_pos)
    if any(pos not in grid for pos in flat_outputs(out_pos)):
        return None
    need_samples = needs_samples(ttarget)
    watch = sorted(set(flat_outputs(out_pos)))
    # Periodic truth tables are independent, integer-lattice trials over one
    # immutable topology.  The ordinary path below rebuilt/reset the simulator
    # and executed one NumPy grid update per tick *per trial*.  Put trials on a
    # leading batch axis instead: the same recurrence is already equivalence-
    # tested in AsyncLutSim, and Full Adder has four trials to amortise at once.
    can_batch = (
        bool(getattr(ttarget, 'combinational_cases', ()))
        and 'combinational_level' in contract_relations(ttarget)
        and all(getattr(trial, 'input_events', None) is None
                for trial in ttarget.trials))
    if can_batch:
        import numpy as np
        flat_pos, lanes = _expand_input_lanes(in_pos)
        obs = _obs_len(ttarget)
        values = np.zeros(
            (len(ttarget.trials), obs, len(flat_pos)), dtype=np.uint8)
        for trial_index, trial in enumerate(ttarget.trials):
            count = min(obs, len(trial.streams))
            if count:
                source = np.asarray(trial.streams[:count], dtype=np.uint8)
                for attachment, lane in enumerate(lanes):
                    if lane < source.shape[1]:
                        values[trial_index, :count, attachment] = source[:, lane]
        if source_nodes is None:
            from substrates.nervous.io_placement import terminal_node_sets
            terminal_inputs, terminal_outputs = terminal_node_sets(
                ttarget, in_pos, {'watch': watch})
        else:
            terminal_inputs = set(source_nodes)
            terminal_outputs = set(sink_nodes or ())
        sim = AsyncLutSim(
            grid, config=getattr(ttarget, 'lut_config', None),
            input_nodes=terminal_inputs, output_nodes=terminal_outputs)
        if sim.config.delay == TICK:
            levels = sim.run_bits_batch_lattice(values, flat_pos)
            traces = TemporalTraces(overflow=False)
            traces.hold_tol = 0

            def intervals_of(bits):
                spans, start = [], None
                for tick, active in enumerate(bits):
                    if active and start is None:
                        start = tick
                    elif not active and start is not None:
                        spans.append((start * TICK, tick * TICK))
                        start = None
                if start is not None:
                    # The event engine reports a level that never fell as an
                    # open-ended physical interval. Preserve that observable;
                    # scoring clips it to each finite read window.
                    spans.append((start * TICK, float('inf')))
                return spans

            for role, cells in groups.items():
                columns = [sim._cidx[cell] for cell in cells]
                role_bits = levels[:, :, columns].max(axis=2)
                traces[role] = (
                    [row.tolist() for row in role_bits]
                    if need_samples else [[] for _row in role_bits])
                traces.intervals[role] = [
                    intervals_of(row) for row in role_bits]
                traces.events[role] = [
                    [start for start, _end in spans]
                    for spans in traces.intervals[role]]
            return traces
    if source_nodes is None:
        sim, trial_B, _, trial_intervals, overflow = _run_lut_trials(
            grid, in_pos, ttarget, watch)
    else:
        sim, trial_B, _, trial_intervals, overflow = _run_lut_trials(
            grid, in_pos, ttarget, watch, source_nodes=source_nodes,
            sink_nodes=set(sink_nodes or ()))
    traces = TemporalTraces(overflow=overflow)
    traces.hold_tol = 0                        # LUT holds strictly (see place_outputs_by_trace)
    for role, cells in groups.items():
        columns = [sim._cidx[cell] for cell in cells]
        traces[role] = [
            B[:, columns].max(axis=1).tolist() if need_samples else []
            for B in trial_B]
        merged = [
            merge_intervals([intervals.get(cell, ()) for cell in cells])
            for intervals in trial_intervals]
        traces.intervals[role] = merged
        traces.events[role] = [[start for start, _ in intervals]
                               for intervals in merged]
    return traces


def prepare_lut(genome, ttarget):
    """Grow + place a genome for a temporal target.

    Returns ``(grid, out_pos, traces, in_pos)`` - ``in_pos`` is the driven input
    sites (native evolved internal pads, native exterior drivers, legacy target
    pads, or compatibility tag-selected cells) - or None if unusable.

    Native internal layouts and compatibility spatial genomes nucleate from
    heritable input coordinates. Exterior mode and the tag/type compatibility
    strategies use a neutral germline."""
    from substrates.nervous.io_placement import io_strategy, growth_seeds
    if is_branched(genome):
        # Output-rooted: the role sites are the genome's own arm roots, not
        # probes fitted to the grown body. Same return contract.
        from .branched_ga import prepare_branched_lut_net
        return prepare_branched_lut_net(genome, ttarget)
    strategy = io_strategy(ttarget)
    mode = lut_io_mode(ttarget)
    if mode == 'exterior_edges':
        seeds = lut_growth_seeds(genome, ttarget, strategy)
        grid = grow_lut(genome, seeds=seeds,
                        grid_size=ttarget.grid_size, iters=ttarget.iters)
        return prepare_lut_grid(
            genome, ttarget, grid, strategy=strategy)
    evolved_layout = getattr(genome, 'input_layout', None) is not None
    in_pos = lut_input_positions(genome, ttarget.inputs)
    if evolved_layout and len(in_pos) != ttarget.n_inputs:
        return None
    seeds = in_pos if evolved_layout else growth_seeds(
        ttarget, strategy, genome)
    grid = grow_lut(genome, seeds=seeds,
                    grid_size=ttarget.grid_size, iters=ttarget.iters)
    return prepare_lut_grid(genome, ttarget, grid, strategy=strategy)


def prepare_lut_grid(genome, ttarget, grid, strategy=None,
                     record_progress=True):
    """``prepare_lut`` for an already-grown body - the LUT twin of
    ``substrates.nervous.temporal.prepare_net_grid``. Lifespan scoring
    interprets developmental snapshots through this, so juvenile and adult
    bodies share one code path. ``record_progress`` is false for juveniles: the
    binding-progress record describes the ADULT organism."""
    from substrates.nervous.io_placement import (
        io_strategy, bind_io, binding_progress, record_binding_progress)
    if strategy is None:
        strategy = io_strategy(ttarget)
    if lut_io_mode(ttarget) == 'exterior_edges':
        in_pos, external_inputs = lut_exterior_inputs(
            genome, grid, ttarget.n_inputs)
        if (len(in_pos) != ttarget.n_inputs
                or len(grid) < len(ttarget.outputs)):
            return None
        out_pos, traces = place_outputs_by_trace(
            grid, in_pos, ttarget, source_nodes=set(),
            external_inputs=external_inputs)
        if any(out_pos[terminal.role] is None
               for terminal in ttarget.outputs):
            return None
        return grid, out_pos, traces, list(in_pos)
    evolved_layout = getattr(genome, 'input_layout', None) is not None
    if evolved_layout:
        in_pos = lut_input_positions(genome, ttarget.inputs)
        if (len(in_pos) != ttarget.n_inputs
                or any(cell not in grid for cell in in_pos)
                or len(grid) <= len(in_pos)):
            return None
        source_nodes = set(in_pos)
        out_pos, traces = place_outputs_by_trace(
            grid, in_pos, ttarget, source_nodes=source_nodes)
        if any(out_pos[terminal.role] is None
               for terminal in ttarget.outputs):
            return None
        return grid, out_pos, traces, list(in_pos)
    node_types = None
    if strategy != 'fixed':
        from .lut import cell_io_tags
        node_types = cell_io_tags(genome, grid)
        if record_progress and strategy in (
                'terminal_nodes', 'wiring_chromosome',
                'spatial_chromosome'):
            record_binding_progress(
                genome,
                binding_progress(genome, grid, ttarget, tags=node_types))
    if len(grid) <= ttarget.n_inputs:
        return None
    if strategy == 'fixed' and any(p not in grid for p in ttarget.inputs):
        return None                     # a dead seed pad (fixed binding only)

    if strategy != 'fixed':
        bound = bind_io(genome, grid, ttarget, strategy,
                        tags=node_types)
        if bound is None:
            return None
        in_pos, out_pos = bound
        traces = trace_fixed_outputs(grid, in_pos, out_pos, ttarget)
        if traces is None:
            return None
        return grid, out_pos, traces, in_pos

    in_pos = list(ttarget.inputs)
    out_pos, traces = place_outputs_by_trace(grid, in_pos, ttarget)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, out_pos, traces, in_pos


def score_lut_temporal(genome, ttarget):
    prep = prepare_lut(genome, ttarget)
    if prep is None:
        return 0.0
    return score_contract(prep[2], ttarget)[0]


def _place_outputs_combinational(grid, target, in_pos=None, source_nodes=None,
                                 external_inputs=None):
    """{role: cell|None} - where each combinational output is read.

    Prefers the FITTED placement the scorer uses (the cell that best computes
    each role, see _fit_combinational_outputs) so every view - growth tab,
    interactive playback, truth-table report - reads the same cell that decides
    fitness. Falls back to nearest-cell proximity when there is no truth table
    to fit against or the fit cannot fill every role."""
    from substrates.nervous.io_placement import flat_inputs
    actual_inputs = target.inputs if in_pos is None else in_pos
    internal_inputs = set(flat_inputs(actual_inputs)).intersection(grid)
    if getattr(target, 'cases', None) and len(grid) > len(internal_inputs):
        try:
            fitted, _ = _fit_combinational_outputs(
                grid, target, in_pos=in_pos, source_nodes=source_nodes,
                external_inputs=external_inputs)
            if all(fitted.get(t.role) is not None for t in target.outputs):
                return fitted
        except Exception:
            pass
    in_set = set(flat_inputs(actual_inputs))
    non_input = [p for p in sorted(grid) if p not in in_set]
    out_pos, used = {}, set()
    for term in target.outputs:
        cands = [p for p in non_input if p not in used]
        if not cands:
            out_pos[term.role] = None
            continue
        best = min(cands, key=lambda p: abs(p[0] - term.pos[0]) + abs(p[1] - term.pos[1]))
        used.add(best)
        out_pos[term.role] = best
    return out_pos


# The LUT array is a SYNCHRONOUS RECURRENT cellular automaton: every cell both
# reads and drives all four neighbours, so with inputs held it generally does NOT
# come to rest - it oscillates or goes chaotic ("most circuits produce immediate
# sustained activity ... chaotic behavior", paper section 6). Reading a combinational
# answer at one fixed tick therefore samples a single phase of that oscillation
# and rewards phase-luck, not computation. A real combinational result requires
# the output to reach a FIXED POINT: hold the same value over a settling window.
#
# An output that never settles is not simply discarded, though. The array is a
# deterministic finite machine, so a held input drives every output into a
# repeating cycle; the honest read of a cycling output is the DUTY CYCLE over one
# period - the fraction of the period it sits high. A fixed point is the period-1
# case (duty 0.0 or 1.0), so the two live on one continuous scale. Scoring feeds
# that duty as a confidence to the contract, which turns it into "fraction of the
# period the output is correct": a circuit right nine-tenths of its cycle scores
# 0.9, giving evolution a smooth gradient toward a stable fixed point instead of
# the old settle/no-settle cliff (non-settling was a hard 0, so a nearly-solved
# oscillator looked no better than noise). The averaging is over one detected
# period, not a raw tail window, so it is phase-invariant, not phase-luck.
SETTLE_WINDOW = 5


def _steady_duty(seq):
    """Fraction of the output's settled CYCLE that sits high, phase-invariantly.

    ``seq`` is the recorded output-bit tail. For each candidate period ``p`` the
    periodicity is measured back from the end (how long ``seq[i] == seq[i-p]``
    keeps holding); ``p`` is accepted only if that periodic suffix covers a solid
    stretch - at least half the tail and several whole periods - so a chaotic run
    cannot fluke a short period from its last few equal samples (the earlier
    version checked only ``2p`` samples and so declared any tail ending in three
    equal bits 'settled', crediting oscillators as solved). The duty is the mean
    over the periodic suffix's whole periods; a fixed point is ``p == 1`` and
    returns exactly 0.0/1.0. A tail with no such period falls back to its whole
    mean - a chaotic output lands near 0.5 and earns only chance credit, which is
    correct: it is not computing a stable function."""
    n = len(seq)
    if n == 0:
        return 0.0
    for p in range(1, n // 2 + 1):
        length = 0                                   # periodic suffix length
        for i in range(n - 1, p - 1, -1):
            if seq[i] == seq[i - p]:
                length += 1
            else:
                break
        if length >= max(3 * p, (n - p + 1) // 2):
            reps = length // p                       # whole periods only
            return sum(seq[-reps * p:]) / (reps * p)
    return sum(seq) / n


def _steady_duties(matrix):
    """Vectorized :func:`_steady_duty` for every column of a sample matrix."""
    import numpy as np
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError('steady-duty samples must be a 2-D matrix')
    n, count = values.shape
    if n == 0:
        return np.zeros(count, dtype=float)
    result = values.mean(axis=0, dtype=float)
    unresolved = np.ones(count, dtype=bool)
    for period in range(1, n // 2 + 1):
        equal = values[period:] == values[:-period]
        reversed_false = ~equal[::-1]
        has_false = reversed_false.any(axis=0)
        trailing = np.where(
            has_false, reversed_false.argmax(axis=0), equal.shape[0])
        eligible = unresolved & (
            trailing >= max(3 * period, (n - period + 1) // 2))
        if not eligible.any():
            continue
        # The scalar rule averages whole periods only. Group columns by the
        # accepted suffix length so each group remains one NumPy reduction.
        whole = (trailing // period) * period
        for length in np.unique(whole[eligible]):
            columns = eligible & (whole == length)
            result[columns] = values[-int(length):, columns].mean(
                axis=0, dtype=float)
        unresolved[eligible] = False
        if not unresolved.any():
            break
    return result
# Combinational inputs are presented as PULSES, not a level held from t=0. Each
# trial gives the active inputs a common rising edge after a random delay (so the
# case's onset is one clean coincident edge) and a random per-line hold width,
# and several such trials are averaged - the array must compute the function
# robustly across arrival times and hold durations rather than exploiting one
# fixed, clean, power-on-aligned presentation. Widths sit above a settling floor
# so the output can reach a fixed point WHILE the whole pattern is held, and the
# duty is read only over that held, settled window. Timings are seeded-fixed per
# target (a stable battery, deterministic across genomes and worker processes so
# the fitness cache stays valid), varied across trials.
N_COMB_TRIALS = 3
_COMB_SEED    = 0xC0FFEE


def _comb_timing(grid_size):
    """(settle_lead, measure_span) in ticks for a combinational presentation:
    time for the pattern to propagate and settle, then the window over which the
    held output's duty is read."""
    return 2 * grid_size, max(2 * SETTLE_WINDOW, grid_size)


def _combinational_schedule(target, n_trials=N_COMB_TRIALS):
    """Seeded-fixed pulse timings [(delay, [width per input line]), ...].

    Same battery for every genome (so fitness is deterministic and cacheable),
    varied across trials. A trial's delay is the common rising edge of that
    trial's active inputs; each input line holds for its own random width, all
    at or above the settling floor so the full pattern outlasts the read window.
    """
    lead, measure = _comb_timing(target.grid_size)
    floor = lead + measure
    rng   = random.Random(_COMB_SEED)
    n     = len(target.inputs)
    span  = max(2, target.grid_size)
    return [(rng.randint(1, span),
             [floor + rng.randint(0, target.grid_size) for _ in range(n)])
            for _ in range(n_trials)]


def _all_cell_duties(grid, target, in_bits, schedule, in_pos=None,
                     watch_groups=(), source_nodes=None,
                     external_inputs=None):
    """Average, over the schedule's pulse trials, each live non-input cell's
    steady duty read during the held window - {cell: mean duty}.

    The active inputs (bits set in ``in_bits``) rise together at each trial's
    delay and hold for their per-line widths; the duty is measured over
    ``[delay + lead, delay + min active width]`` - the full pattern present and
    settled. The all-zero case has no pulses, so it reads a fixed lead..measure
    window of the array's no-input response (the function's value there). Uses
    the vectorised ``run_bits`` matrix so the whole-grid read is not per-tick
    Python.

    ``in_pos`` overrides the driven input cells (the native or compatibility
    binding and possibly per-input attachment groups); None keeps the target's
    legacy pads. ``external_inputs`` maps each outside bus tap to the one
    perimeter cell/direction it drives."""
    lead, measure = _comb_timing(target.grid_size)
    in_pos  = list(target.inputs) if in_pos is None else list(in_pos)
    flat_pos, lanes = _expand_input_lanes(in_pos)
    in_set  = set(flat_pos)
    if source_nodes is None:
        from substrates.nervous.io_placement import terminal_node_sets
        terminal_inputs, terminal_outputs = terminal_node_sets(
            target, in_pos,
            {index: list(group) for index, group in enumerate(watch_groups)})
    else:
        terminal_inputs = set(source_nodes)
        terminal_outputs = set()
    active  = [i for i, b in enumerate(in_bits) if b]
    totals  = None
    sim = AsyncLutSim(
        grid, input_nodes=terminal_inputs,
        output_nodes=terminal_outputs,
        external_inputs=external_inputs)
    for trial_index, (delay, widths) in enumerate(schedule):
        if trial_index:
            # Grid topology, LUT columns and terminal masks are identical for
            # every timing replicate. Reset only dynamic state instead of
            # rebuilding those arrays and neighbour-index columns each time.
            sim.reset()
        # Read the LAST ``measure`` ticks the full pattern is held, so the output
        # has the maximum settling time (>= lead, since widths >= lead+measure)
        # before it is sampled and the input-arrival transient has cleared.
        if active:
            hi = delay + min(widths[i] for i in active)
            lo = max(delay + lead, hi - measure)
        else:
            lo, hi = delay + lead, delay + lead + measure
        # One stream lane per ATTACHMENT CELL: lane k carries logical input
        # lanes[k]'s bit - a shared cell wired-ORs in the sim's injection.
        streams = [tuple((1 if (in_bits[lanes[k]]
                                and delay <= t < delay + widths[lanes[k]])
                          else 0)
                         for k in range(len(flat_pos)))
                   for t in range(hi)]
        levels = sim.run_bits(streams, flat_pos, hi)
        window = levels[lo:hi]
        cells  = sim._cells
        column_duties = _steady_duties(window)
        duties = {cells[j]: float(column_duties[j])
                  for j in range(len(cells)) if cells[j] not in in_set}
        # A multi-site output is one physical wired-OR bus, so score the union
        # waveform rather than cherry-picking or averaging its member cells.
        for group in watch_groups:
            key = tuple(group)
            columns = [sim._cidx[cell] for cell in key
                       if cell in sim._cidx and cell not in in_set]
            if len(columns) == len(key) and columns:
                duties[key] = _steady_duty(
                    window[:, columns].max(axis=1).tolist())
        if totals is None:
            totals = duties
        else:
            for c in totals:
                totals[c] += duties.get(c, 0.0)
    n = len(schedule) or 1
    return {c: v / n for c, v in (totals or {}).items()}


def _all_case_duties(grid, target, case_inputs, schedule, in_pos=None,
                     watch_groups=(), source_nodes=None,
                     external_inputs=None):
    """Batched equivalent of one :func:`_all_cell_duties` call per case.

    Truth-table cases and timing replicates are independent power-on trials
    with identical topology.  Running them as the leading NumPy dimension
    removes thousands of tiny whole-grid operations on hard arithmetic targets
    while preserving every pulse width, settling window and periodic-duty
    calculation.
    """
    import numpy as np
    lead, measure = _comb_timing(target.grid_size)
    in_pos = list(target.inputs) if in_pos is None else list(in_pos)
    flat_pos, lanes = _expand_input_lanes(in_pos)
    in_set = set(flat_pos)
    watch_groups = tuple(tuple(group) for group in watch_groups)
    if source_nodes is None:
        from substrates.nervous.io_placement import terminal_node_sets
        terminal_inputs, terminal_outputs = terminal_node_sets(
            target, in_pos,
            {index: list(group)
             for index, group in enumerate(watch_groups)})
    else:
        terminal_inputs = set(source_nodes)
        terminal_outputs = set()
    cases = tuple(tuple(bits) for bits in case_inputs)
    schedule = tuple(schedule)
    if not cases or not schedule:
        return [{} for _ in cases]

    records = []
    max_hi = 0
    for case_index, in_bits in enumerate(cases):
        active = [index for index, bit in enumerate(in_bits) if bit]
        for trial_index, (delay, widths) in enumerate(schedule):
            if active:
                hi = delay + min(widths[index] for index in active)
                lo = max(delay + lead, hi - measure)
            else:
                lo, hi = delay + lead, delay + lead + measure
            records.append((case_index, trial_index, int(lo), int(hi)))
            max_hi = max(max_hi, int(hi))

    streams = np.zeros(
        (len(records), max_hi, len(flat_pos)), dtype=np.uint8)
    for record_index, (case_index, trial_index, _lo, _hi) in enumerate(records):
        delay, widths = schedule[trial_index]
        bits = cases[case_index]
        for attachment, lane in enumerate(lanes):
            if bits[lane]:
                streams[
                    record_index,
                    int(delay):int(delay + widths[lane]),
                    attachment,
                ] = 1

    sim = AsyncLutSim(
        grid, input_nodes=terminal_inputs,
        output_nodes=terminal_outputs,
        external_inputs=external_inputs)
    levels = sim.run_bits_batch_lattice(streams, flat_pos)
    starts = np.asarray([record[2] for record in records], dtype=np.intp)
    lengths = np.asarray(
        [record[3] - record[2] for record in records], dtype=np.intp)
    if not np.all(lengths == measure):
        raise ValueError('combinational duty windows must have fixed length')
    rows = starts[:, None] + np.arange(measure, dtype=np.intp)[None, :]
    windows = levels[np.arange(len(records))[:, None], rows]

    # _steady_duties treats columns independently. Flatten trial and cell into
    # one column axis, then restore [case, timing trial, cell].
    count = len(sim._cells)
    steady = _steady_duties(
        windows.transpose(1, 0, 2).reshape(measure, -1)
    ).reshape(len(cases), len(schedule), count).mean(axis=1)
    duty_by_case = [
        {cell: float(steady[case_index, column])
         for column, cell in enumerate(sim._cells) if cell not in in_set}
        for case_index in range(len(cases))
    ]

    for group in watch_groups:
        columns = [sim._cidx[cell] for cell in group
                   if cell in sim._cidx and cell not in in_set]
        if len(columns) != len(group) or not columns:
            continue
        group_windows = windows[:, :, columns].max(axis=2)
        group_duties = _steady_duties(
            group_windows.T
        ).reshape(len(cases), len(schedule)).mean(axis=1)
        for case_index, duty in enumerate(group_duties):
            duty_by_case[case_index][group] = float(duty)
    return duty_by_case


def _balanced_match(duties, expected):
    """Balanced per-level match of one cell's per-case duty to an expected bit
    vector - the single-output form of the contract's logic aggregation, used
    only to RANK candidate output cells (the reported fitness comes from
    score_contract on the chosen cells)."""
    groups = {0: [], 1: []}
    for duty, want in zip(duties, expected):
        groups[1 if want else 0].append(duty if want else 1.0 - duty)
    parts = [sum(v) / len(v) for v in groups.values() if v]
    return sum(parts) / len(parts) if parts else 0.0


def _fit_combinational_outputs(grid, target, in_pos=None, source_nodes=None,
                               external_inputs=None):
    """Globally assign output roles to distinct cells from per-case duty.

    Output identity is a FITTED parameter, exactly as on the temporal path
    (``place_outputs_by_trace``): rather than reading a cell chosen by mere
    proximity to the terminal, the array is treated as a POOL of candidate
    functions. One distinct cell is fixed per role across every case (never a
    different cell per case), and the joint assignment maximizes total score so
    a greedy early role cannot consume the only strong candidate for a later
    one. ``duty_by_case[i][cell]`` is phase-invariant steady duty: a settled
    cell contributes its exact bit and a cycling cell the fraction of its
    period it is correct."""
    schedule = _combinational_schedule(target)   # one shared pulse battery
    duty_by_case = _all_case_duties(
        grid, target, [in_bits for in_bits, _ in target.cases], schedule,
        in_pos=in_pos, source_nodes=source_nodes,
        external_inputs=external_inputs)
    cells = list(duty_by_case[0]) if duty_by_case else []
    candidates = behavior_representatives(
        cells,
        lambda cell: tuple(case.get(cell, 0.0) for case in duty_by_case),
        len(target.outputs))
    out_pos = {term.role: None for term in target.outputs}
    scores = {term.role: {} for term in target.outputs}
    for oi, term in enumerate(target.outputs):
        expected = [out_bits[oi] for _, out_bits in target.cases]
        for cell in candidates:
            scores[term.role][cell] = _balanced_match(
                [duty_by_case[index][cell]
                 for index in range(len(target.cases))],
                expected)
    assignment = best_distinct_assignment(
        tuple(out_pos), candidates, scores, balance_worst=True)
    if assignment is not None:
        out_pos.update(assignment)
    return out_pos, duty_by_case


def score_lut_combinational_full(genome, target):
    """Return fitness plus the exact row/output correctness vector.

    Hold each input pattern, let the array settle, and read globally fitted
    distinct cells for the output roles (see _fit_combinational_outputs).
    A settled cell scores its exact bit; a cycling cell scores the fraction of
    its period it is correct.

    Under an evolvable io_placement strategy the driven input cells AND the read
    output cells come from the genome's tags instead (no duty-fitting search -
    the binding is heritable, not fitted). Spatial input anchors also serve as
    germline positions; other evolvable modes nucleate from one neutral centre."""
    from substrates.nervous.scoring import contract_case_count
    from substrates.nervous.io_placement import (
        io_strategy, bind_io, growth_seeds, binding_progress,
        record_binding_progress)
    failed = (0.0, (0.0,) * contract_case_count(target))
    strategy = io_strategy(target)
    mode = lut_io_mode(target)
    if mode == 'exterior_edges':
        grid = grow_lut(
            genome, seeds=lut_growth_seeds(genome, target, strategy),
            grid_size=target.grid_size, iters=target.iters)
        in_pos, external_inputs = lut_exterior_inputs(
            genome, grid, target.n_inputs)
        if (len(in_pos) != target.n_inputs or not target.cases
                or not target.outputs):
            return failed
        out_pos, duty_by_case = _fit_combinational_outputs(
            grid, target, in_pos=in_pos, source_nodes=set(),
            external_inputs=external_inputs)
        if any(out_pos[term.role] is None for term in target.outputs):
            return failed
        observations = [
            [duty_by_case[index][out_pos[term.role]]
             for term in target.outputs]
            for index in range(len(target.cases))]
        score, cases, _ = score_contract(observations, target)
        return score, cases
    evolved_layout = getattr(genome, 'input_layout', None) is not None
    in_pos = lut_input_positions(genome, target.inputs)
    if evolved_layout and len(in_pos) != target.n_inputs:
        return failed
    seeds = in_pos if evolved_layout else growth_seeds(
        target, strategy, genome)
    grid = grow_lut(genome, seeds=seeds,
                    grid_size=target.grid_size, iters=target.iters)
    node_types = None
    if strategy != 'fixed':
        from .lut import cell_io_tags
        node_types = cell_io_tags(genome, grid)
        if strategy in (
                'terminal_nodes', 'wiring_chromosome',
                'spatial_chromosome'):
            record_binding_progress(
                genome,
                binding_progress(genome, grid, target, tags=node_types))
    if len(grid) <= target.n_inputs:
        return failed
    if len(target.cases) == 0 or len(target.outputs) == 0:
        return failed

    if evolved_layout:
        if any(cell not in grid for cell in in_pos):
            return failed
        source_nodes = set(in_pos)
        out_pos, duty_by_case = _fit_combinational_outputs(
            grid, target, in_pos=in_pos, source_nodes=source_nodes)
        if any(out_pos[term.role] is None for term in target.outputs):
            return failed
        observations = [
            [duty_by_case[index][out_pos[term.role]]
             for term in target.outputs]
            for index in range(len(target.cases))]
        score, cases, _ = score_contract(observations, target)
        return score, cases

    if strategy != 'fixed':
        from substrates.nervous.io_placement import output_groups
        bound = bind_io(genome, grid, target, strategy,
                        tags=node_types)
        if bound is None:
            return failed
        in_pos, out_pos = bound
        out_groups = output_groups(out_pos)
        schedule = _combinational_schedule(target)
        duty_by_case = _all_case_duties(
            grid, target, [in_bits for in_bits, _ in target.cases], schedule,
            in_pos=in_pos, watch_groups=out_groups.values())
        # A bound output that is also a driven input has no duty entry (inputs
        # are excluded from the read); such a binding simply scores 0.
        if any(tuple(out_groups[t.role]) not in duty_by_case[0]
               for t in target.outputs):
            return failed
        observations = [[duty_by_case[i][tuple(out_groups[t.role])]
                         for t in target.outputs]
                        for i in range(len(target.cases))]
        score, cases, _ = score_contract(observations, target)
        return score, cases

    out_pos, duty_by_case = _fit_combinational_outputs(grid, target)
    if any(out_pos[t.role] is None for t in target.outputs):
        return failed
    observations = [[duty_by_case[i][out_pos[t.role]] for t in target.outputs]
                    for i in range(len(target.cases))]
    score, cases, _ = score_contract(observations, target)
    return score, cases


def score_lut_combinational(genome, target):
    """Scalar compatibility wrapper for combinational LUT evaluation."""
    return score_lut_combinational_full(genome, target)[0]


def lut_case_outputs(genome, target):
    """Grow, place outputs, and run each combinational case to the horizon.
    Returns (grid, out_pos, cases); each case carries the settled node activity
    (`node_outputs` 0/1 and `node_nibbles` 4-bit N/S/E/W) so the GUI can draw the
    array's response. The LUT analogue of substrates.nervous.nervous_case_outputs."""
    from substrates.nervous.io_placement import io_strategy, bind_io, growth_seeds
    strategy = io_strategy(target)
    exterior = lut_io_mode(target) == 'exterior_edges'
    evolved_layout = (
        not exterior and getattr(genome, 'input_layout', None) is not None)
    if exterior:
        seeds = lut_growth_seeds(genome, target, strategy)
        in_pos = ()
    else:
        in_pos = lut_input_positions(genome, target.inputs)
        if evolved_layout and len(in_pos) != target.n_inputs:
            return {}, {term.role: None for term in target.outputs}, []
        seeds = in_pos if evolved_layout else growth_seeds(
            target, strategy, genome)
    grid = grow_lut(genome, seeds=seeds,
                    grid_size=target.grid_size, iters=target.iters)
    external_inputs = None
    if exterior:
        in_pos, external_inputs = lut_exterior_inputs(
            genome, grid, target.n_inputs)
        if len(in_pos) != target.n_inputs:
            return grid, {term.role: None for term in target.outputs}, []
    cases = []
    if ((not exterior and len(grid) <= target.n_inputs)
            or (exterior and len(grid) < len(target.outputs))
            or not target.cases or not target.outputs):
        return grid, {t.role: None for t in target.outputs}, cases
    # Read at the SAME fitted cells and pulsed trials the scorer uses, so the
    # report reflects the circuit's actual fitness. Node activity for the drawing
    # comes from one representative trial's held window; the reported duty is the
    # trial-averaged value the scorer uses, and a role counts 'settled' only when
    # that averaged duty is an exact 0/1 (a fixed point in every trial).
    bound_outputs = (
        strategy != 'fixed' and not evolved_layout and not exterior)
    source_nodes = set() if exterior else (
        set(in_pos) if evolved_layout else None)
    if exterior:
        out_pos, duty_by_case = _fit_combinational_outputs(
            grid, target, in_pos=in_pos, source_nodes=source_nodes,
            external_inputs=external_inputs)
        if any(out_pos[term.role] is None for term in target.outputs):
            return grid, out_pos, cases
    elif evolved_layout:
        if any(cell not in grid for cell in in_pos):
            return grid, {term.role: None for term in target.outputs}, cases
        out_pos, duty_by_case = _fit_combinational_outputs(
            grid, target, in_pos=in_pos, source_nodes=source_nodes)
        if any(out_pos[term.role] is None for term in target.outputs):
            return grid, out_pos, cases
    elif strategy != 'fixed':
        from .lut import cell_io_tags
        from substrates.nervous.io_placement import output_groups
        bound = bind_io(genome, grid, target, strategy,
                        tags=cell_io_tags(genome, grid))
        if bound is None:
            return grid, {t.role: None for t in target.outputs}, cases
        in_pos, out_pos = bound
        out_groups = output_groups(out_pos)
        schedule = _combinational_schedule(target)
        duty_by_case = _all_case_duties(
            grid, target, [in_bits for in_bits, _ in target.cases], schedule,
            in_pos=in_pos, watch_groups=out_groups.values())
        if any(tuple(out_groups[t.role]) not in duty_by_case[0]
               for t in target.outputs):
            return grid, out_pos, cases
    else:
        in_pos = list(target.inputs)
        out_pos, duty_by_case = _fit_combinational_outputs(grid, target)
        if any(out_pos[t.role] is None for t in target.outputs):
            return grid, out_pos, cases
    schedule      = _combinational_schedule(target)
    lead, measure = _comb_timing(target.grid_size)
    delay, widths = schedule[0]
    flat_pos, lanes = _expand_input_lanes(in_pos)
    for ci, (in_bits, out_bits) in enumerate(target.cases):
        active = [i for i, b in enumerate(in_bits) if b]
        if active:
            hi = delay + min(widths[i] for i in active)
        else:
            hi = delay + lead + measure
        streams = [tuple((1 if (in_bits[lanes[k]]
                                and delay <= t < delay + widths[lanes[k]])
                          else 0)
                         for k in range(len(flat_pos)))
                   for t in range(hi)]
        if source_nodes is None:
            from substrates.nervous.io_placement import terminal_node_sets
            terminal_inputs, terminal_outputs = terminal_node_sets(
                target, in_pos, out_pos)
        else:
            terminal_inputs, terminal_outputs = source_nodes, set()
        sim = AsyncLutSim(
            grid, input_nodes=terminal_inputs,
            output_nodes=terminal_outputs,
            external_inputs=external_inputs)
        sim.run_bits(streams, flat_pos, hi)
        node_out = {c: (1 if sim.out[c] else 0) for c in grid}
        duty = {
            t.role: duty_by_case[ci][
                tuple(out_groups[t.role]) if bound_outputs
                else out_pos[t.role]]
            for t in target.outputs}
        acts = {r: (int(round(d)) if (d <= 1e-9 or d >= 1 - 1e-9) else None)
                for r, d in duty.items()}
        cases.append({'in_bits': in_bits, 'out_bits': out_bits,
                      'node_outputs': node_out, 'node_nibbles': dict(sim.out),
                      'acts': acts, 'duty': duty,
                      'stable': all(v is not None for v in acts.values())})
    return grid, out_pos, cases


def lut_truth_table(genome, target):
    """Per-case truth-table report for a combinational target on the LUT array
    (mirrors substrates.nervous.nervous_truth_table)."""
    grid, out_pos, cases = lut_case_outputs(genome, target)
    lines = ['Target: %s   [LUT array]' % target.name,
             'Circuit: %d live cells  (square array, four 16-bit LUTs/cell,'
             % len(grid),
             '          asynchronous level logic - paper Architecture 2)']
    from substrates.nervous.io_placement import output_groups
    groups = output_groups(out_pos)
    for term in target.outputs:
        cells = groups.get(term.role, [])
        lines.append("  out '%s': %s" % (
            term.role,
            ('wired-OR at %s' % ', '.join(map(str, cells)))
            if cells else '(not found)'))
    if not cases:
        lines += ['', '(circuit incomplete - grew too little or outputs missing)']
        return '\n'.join(lines)
    lines += ['', 'Inputs held; each output settles to a fixed point (full',
              "credit) or cycles - then it scores the fraction of its period it",
              "is correct. Cell shows expected / actual, actual = settled bit,",
              "or ~NN% duty when still oscillating.", '']
    in_hdr  = ' '.join('i%d' % i for i in range(len(target.inputs)))
    out_hdr = ' '.join('%s:e/a' % t.role for t in target.outputs)
    lines += ['  %s | %s | result' % (in_hdr, out_hdr),
              '  ' + '-' * (len(in_hdr) + len(out_hdr) + 14)]
    for case in cases:
        cells, row_ok, cycling = [], True, False
        for i, term in enumerate(target.outputs):
            act = case['acts'][term.role]; exp = case['out_bits'][i]
            duty = case['duty'][term.role]
            row_ok = row_ok and (act == exp)
            if act is None:                    # cycling: show duty toward expected
                cycling = True
                shown = '~%d%%' % round((duty if exp else 1.0 - duty) * 100)
            else:
                shown = str(act)
            cells.append('%d/%s' % (exp, shown))
        in_str  = ' '.join(str(b) for b in case['in_bits']).ljust(len(in_hdr))
        out_str = ' '.join(c.ljust(len('%s:e/a' % t.role))
                           for c, t in zip(cells, target.outputs))
        # PASS = every output settled correct; a settled-but-wrong output is a
        # clean FAIL; only a still-cycling output is 'partial' (graded credit).
        verdict = 'PASS' if row_ok else ('partial' if cycling else 'FAIL')
        lines.append('  %s | %s | %s' % (in_str, out_str, verdict))
    # Report the exact fitness the GA optimises, not a separate settled-only
    # tally, so the readout can never disagree with selection.
    fit = score_lut_combinational(genome, target)
    n_settled = sum(1 for c in cases if c['stable'])
    lines += ['', '  => fitness = %.4f   (%d/%d cases settled to a fixed point)%s'
              % (fit, n_settled, len(cases),
                 '   ALL PASS' if fit >= 0.9999 else '')]
    return '\n'.join(lines)


def evaluate_lut_full(genome, target, _return_prepared=False):
    """(scalar fitness, per-case vector) - cases are the units epsilon-lexicase
    streams over. Static truth tables retain one case per row/output check.

    Under LIFESPAN SCORING the vector is extended by one entry per
    developmental checkpoint; the scalar stays the ADULT organism's score (see
    substrates/nervous/objectives.py)."""
    if not getattr(target, 'temporal', False):
        result = score_lut_combinational_full(genome, target)
        return result + (None,) if _return_prepared else result
    from substrates.nervous.objectives import (
        grown_snapshots, juvenile_scores, prepare_grid, total_case_count)
    escape = getattr(target, '_escape', None)
    lifespan = escape is not None and escape.lifespan_scoring
    n_cases = total_case_count(target)
    snapshots, strategy = None, None
    if lifespan:
        # One growth pass serves both the adult evaluation and every juvenile
        # checkpoint: the final snapshot is bit-identical to grow_lut's result.
        from substrates.nervous.io_placement import io_strategy
        strategy = io_strategy(target)
        snapshots = grown_snapshots(genome, target, 'lut', strategy)
        prep = prepare_grid(genome, target, 'lut', snapshots[-1], strategy)
    else:
        prep = prepare_lut(genome, target)
    if prep is None:
        result = (0.0, (0.0,) * n_cases)
        return result + (None,) if _return_prepared else result
    traces = prep[2]
    score, cases, _ = score_contract(traces, target)
    if lifespan:
        cases = tuple(cases or ()) + juvenile_scores(
            genome, target, 'lut', snapshots, strategy,
            escape.lifespan_checkpoints, score)
    result = (score, cases)
    return result + (prep,) if _return_prepared else result


def _lut_structural_topology(prepared):
    """Target-blind source-reachable topology of one prepared LUT body.

    Directional LUT outputs are distinct graph nodes.  An edge enters output
    wire ``d`` only from the neighbour directions that that wire's table really
    depends on, so two logical inputs count as integrated only after their
    influence reaches the same directional computation.  Virtual source nodes
    keep a pad's four injected wires grouped as ONE logical input.
    """
    from substrates.topology import EMPTY, measure
    from .branched import DIRECTIONS, OPPOSITE, neighbours, table_support

    if prepared is None:
        return EMPTY
    grid, out_pos, _traces, in_pos = prepared
    if not grid or not in_pos:
        return EMPTY
    pads = {tuple(cell) for cell in in_pos}
    from substrates.nervous.io_placement import flat_outputs
    sinks = {tuple(cell) for cell in flat_outputs(out_pos)}
    nodes = {(cell, direction) for cell in grid for direction in DIRECTIONS}
    sources, edges = [], []
    for index, pad in enumerate(in_pos):
        source = ('LUT_INPUT', int(index))
        sources.append(source)
        for direction in DIRECTIONS:
            edges.append((source, (tuple(pad), direction)))
    for cell, state in grid.items():
        if cell in pads:                 # source-only under AsyncLutSim
            continue
        around = neighbours(cell)
        for output_direction, table in zip(DIRECTIONS, state):
            destination = (cell, output_direction)
            for input_direction in table_support(table):
                source_cell = around[input_direction]
                if source_cell in grid and source_cell not in sinks:
                    edges.append((
                        (source_cell, OPPOSITE[input_direction]),
                        destination))
    return measure(edges, sources, nodes=nodes)


def _output_module_scores(cases, target):
    """Balanced per-output contract scores for retaining useful LUT arms."""
    base = tuple(float(value) for value in (cases or ()))
    outputs = tuple(getattr(target, 'outputs', ()) or ())
    if len(outputs) < 2:
        return ()
    if (getattr(target, 'temporal', False)
            and getattr(target, 'combinational_cases', ())
            and 'combinational_level' in contract_relations(target)):
        # score_contract emits one cell per (trial, role, row), in that order.
        # Recover the desired level from the same read windows so an arm is
        # retained with the same 0/1-balanced statistic used by the scalar
        # fitness.  Previously all temporal targets returned (), which meant
        # the live periodic Full Adder never exposed its useful Sum and Carry
        # modules to role-aware selection/recombination.
        from substrates.nervous.scoring import _combinational_read_windows
        roles = tuple(str(output.role) for output in outputs)
        per_role = {role: {0: [], 1: []} for role in roles}
        cursor = 0
        for trial in getattr(target, 'trials', ()):
            windows = _combinational_read_windows(target, trial)
            for role in trial.expected:
                expected = tuple(sorted(
                    trial.expected_events.get(role, ())))
                for start, end, _read_low, _read_high in windows:
                    if cursor >= len(base):
                        return ()
                    wanted = any(start <= event < end for event in expected)
                    if str(role) in per_role:
                        per_role[str(role)][int(wanted)].append(base[cursor])
                    cursor += 1
        if not cursor:
            return ()
        scores = []
        for role in roles:
            means = [sum(values) / len(values)
                     for values in per_role[role].values() if values]
            scores.append(sum(means) / len(means) if means else 0.0)
        return tuple(scores)
    if getattr(target, 'temporal', False):
        return ()
    rows = tuple(getattr(target, 'cases', ()) or ())
    if len(base) != len(rows) * len(outputs):
        return ()
    scores = []
    for output_index in range(len(outputs)):
        levels = {0: [], 1: []}
        for row_index, (_inputs, expected) in enumerate(rows):
            if len(expected) <= output_index:
                return ()
            levels[int(bool(expected[output_index]))].append(
                base[row_index * len(outputs) + output_index])
        means = [sum(values) / len(values)
                 for values in levels.values() if values]
        scores.append(sum(means) / len(means) if means else 0.0)
    return tuple(scores)


def _selection_case_vector(cases, target):
    """Retain coherent role and row views for multi-output lexicase."""
    base = tuple(float(value) for value in (cases or ()))
    outputs = tuple(getattr(target, 'outputs', ()) or ())
    rows = tuple(getattr(target, 'cases', ()) or ())
    if (getattr(target, 'temporal', False)
            and getattr(target, 'combinational_cases', ())):
        output_scores = _output_module_scores(base, target)
        return (base + output_scores + (min(output_scores),)
                if output_scores else base)
    if (getattr(target, 'temporal', False) or len(outputs) < 2
            or len(base) != len(rows) * len(outputs)):
        return base
    output_scores = _output_module_scores(base, target)
    joint_rows = tuple(
        min(base[row * len(outputs):(row + 1) * len(outputs)])
        for row in range(len(rows)))
    return base + output_scores + joint_rows + (min(output_scores),)


def _evaluate_lut_selection_record(genome, target):
    fitness, cases, prepared = evaluate_lut_full(
        genome, target, _return_prepared=True)
    from substrates.nervous.io_placement import io_strategy
    from substrates.nervous.objectives import escape_objectives
    total = target.n_inputs + len(target.outputs)
    juvenile, robust = escape_objectives(genome, target, 'lut', cases)
    topology = _lut_structural_topology(prepared)
    output_scores = _output_module_scores(cases, target)
    selection_cases = _selection_case_vector(cases, target)
    if io_strategy(target) not in (
            'terminal_nodes', 'wiring_chromosome', 'spatial_chromosome'):
        return (fitness, selection_cases, (total, total), juvenile, robust,
                topology, topology.score, output_scores)
    progress = getattr(genome, '_io_binding_progress', (total, total))
    return (fitness, selection_cases, progress, juvenile, robust,
            topology, topology.score, output_scores)


def branched_signature(genome):
    """Cache identity of a branched LUT genome.

    Gene IDs are included because development breaks a contested cell by
    (distance, arm label, gene id): two rule sets differing only in their IDs
    really can grow different bodies. The enabled banks are included because
    they are what the tolerance metric measures against.
    """
    chroms = tuple(
        (tuple((gene.gene_id, gene.ctx_n, gene.ctx_s, gene.ctx_e, gene.ctx_w,
                gene.self_in, gene.self_out, gene.branch_id, gene.depth)
               for gene in chromosome.genes),
         tuple((control.tolerance, control.telomere)
               for control in chromosome.controls))
        for chromosome in genome.chromosomes)
    return ('branched', chroms,
            tuple((gene.bearing, gene.distance) for gene in genome.inputs),
            tuple((gene.role, gene.bearing, gene.distance, gene.branch_id)
                  for gene in genome.outputs),
            tuple(genome.families or ()))


def genome_signature(genome):
    if is_branched(genome):
        return branched_signature(genome)
    layout = (
        None if getattr(genome, 'input_layout', None) is None
        else tuple(tuple(cell) for cell in genome.input_layout))
    return (getattr(genome, 'seed_state', None), layout) + tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0), getattr(c, 'wiring', False),
         tuple((g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in, g.self_out,
                getattr(g, 'tag', 0), getattr(g, 'io_selector', 0),
                getattr(g, 'io_kind', 0))
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_cases(genomes, target, cache=None, executor=None,
                     should_stop=None, on_progress=None):
    """(fitnesses, case_vectors) in parallel; cache holds (fit, cases). A
    persistent `executor` is reused instead of spawning a fresh pool per call
    (large speed-up on Windows); omitting it keeps the one-shot-pool behaviour.
    `should_stop`/`on_progress` are threaded to map_ordered so a run stays
    cancellable without a chunk barrier."""
    out  = [None] * len(genomes)
    todo = list(range(len(genomes)))
    if cache is not None and len(cache) > FITNESS_CACHE_MAX:
        cache.clear()                      # bound memory on very long runs
    if cache is not None:
        sigs = [genome_signature(g) for g in genomes]
        todo = [i for i in todo if sigs[i] not in cache]
        for i in range(len(genomes)):
            if sigs[i] in cache:
                out[i] = cache[sigs[i]]
    if todo:
        fn = partial(_evaluate_lut_selection_record, target=target)
        if cache is not None:
            unique = {}
            for i in todo:
                unique.setdefault(sigs[i], []).append(i)
            representatives = [(sig, indices[0]) for sig, indices in unique.items()]
        else:
            representatives = [(i, i) for i in todo]
            unique = {i: [i] for i in todo}
        subset = [genomes[i] for _, i in representatives]
        if executor is not None:
            results = map_ordered(executor, fn, subset, should_stop, on_progress)
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                results = map_ordered(ex, fn, subset, should_stop, on_progress)
        for (sig, _), r in zip(representatives, results):
            for i in unique[sig]:
                out[i] = r
            if cache is not None:
                cache[sig] = r
    from substrates.nervous.io_placement import record_binding_progress
    from substrates.nervous.ga import record_escape_objectives
    for genome, record in zip(genomes, out):
        progress = record[2] if len(record) > 2 else (
            target.n_inputs + len(target.outputs),) * 2
        record_binding_progress(genome, progress)
        record_escape_objectives(genome, record)
        genome._output_scores = (
            tuple(record[7]) if len(record) > 7 else ())
    return [r[0] for r in out], [r[1] for r in out]


# -- genetic operators ------------------------------------------------------------

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_n", "ctx_e", "ctx_s", "ctx_w", "self_in", "self_out"]


def _normalize_split(chromosome):
    """Keep split on a real between-gene boundary (or zero for one gene)."""
    count = len(chromosome.genes)
    chromosome.split = (0 if count < 2 else
                        max(1, min(int(chromosome.split), count - 1)))


def _recombine_gene_fields(gene_a, gene_b, fields=None):
    """Uniformly recombine a single LUT rule's active alleles."""
    fields = (
        tuple(_GENE_FIELDS) + ('tag', 'io_kind')
        if fields is None else fields)
    differing = [field for field in fields
                  if getattr(gene_a, field) != getattr(gene_b, field)]
    if len(differing) < 2:
        return gene_a, gene_b
    exchanged = set(random.sample(
        differing, random.randint(1, len(differing) - 1)))
    child_a, child_b = copy.copy(gene_a), copy.copy(gene_b)
    for field in exchanged:
        value_a, value_b = getattr(gene_a, field), getattr(gene_b, field)
        setattr(child_a, field, value_b)
        setattr(child_b, field, value_a)
    return child_a, child_b


def _recombination_signature(genome):
    """Alleles crossover can actually exchange (not slot/object identity)."""
    layout = (
        None if getattr(genome, 'input_layout', None) is None
        else tuple(tuple(cell) for cell in genome.input_layout))
    return (getattr(genome, 'seed_state', None), layout) + tuple(
        (getattr(chromosome, 'wiring', False),
         tuple((*(getattr(gene, field) for field in _GENE_FIELDS),
                getattr(gene, 'tag', 0), getattr(gene, 'io_selector', 0),
                getattr(gene, 'io_kind', 0))
               for gene in chromosome.genes))
        for chromosome in genome.chromosomes)


def _recombination_environment(genome):
    """Physical context that branched arms must share before they can mix."""
    if not is_branched(genome):
        return None
    from .branched_ga import input_pads
    return tuple(input_pads(genome))


def _other_lut(value):
    """Choose a genuinely different 16-bit LUT value."""
    return (value + random.randrange(1, LUT_STATES)) % LUT_STATES


_SOFT_LUT_MASKS = tuple(
    sum(1 << bit for bit in bits)
    for width in (1, 2, 3)
    for bits in combinations(range(16), width))
_SOFT_LUT_MASK_INDEX = {mask: index for index, mask in enumerate(_SOFT_LUT_MASKS)}


def _soft_lut_excluding(value, parent_value):
    """Flip 1-3 bits without returning either current or parental LUT."""
    forbidden = (None if parent_value is None else value ^ parent_value)
    forbidden_index = _SOFT_LUT_MASK_INDEX.get(forbidden)
    if forbidden_index is None:
        return value ^ random.choice(_SOFT_LUT_MASKS)
    pick = random.randrange(len(_SOFT_LUT_MASKS) - 1)
    if pick >= forbidden_index:
        pick += 1
    return value ^ _SOFT_LUT_MASKS[pick]


def _force_nonparent_tweak(genome, parent, function_families=None):
    """End a multi-edit transaction at an allele distinct from its parent."""
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes and not getattr(chromosome, 'wiring', False)]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(random_lut_chromosome(
                function_families=function_families))
        else:
            random.choice(genome.chromosomes).genes.append(
                random_lut_gene(function_families))
        with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                      if chromosome.genes]
    ci = random.choice(with_genes)
    gi = random.randrange(len(genome.chromosomes[ci].genes))
    field = random.choice(_GENE_FIELDS)
    gene = copy.copy(genome.chromosomes[ci].genes[gi])
    parent_value = None
    if (ci < len(parent.chromosomes)
            and gi < len(parent.chromosomes[ci].genes)):
        parent_value = getattr(parent.chromosomes[ci].genes[gi], field)
    if field == 'self_out' and not unrestricted_only(function_families):
        gene.self_out = mutate_function_table(
            gene.self_out, function_families,
            forbidden=(() if parent_value is None else (parent_value,)))
    else:
        setattr(gene, field, _soft_lut_excluding(
            getattr(gene, field), parent_value))
    genome.chromosomes[ci].genes[gi] = gene


def _tweak_gene(gene, function_families=None):
    """Soft mutation: usually flip 1-3 bits of one 16-bit field (a nearby
    boolean function), sometimes replace the field entirely. self_in has an
    extra chance of snapping to 0 - growth rules must stay reachable."""
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_out' and not unrestricted_only(function_families):
        g.self_out = mutate_function_table(g.self_out, function_families)
    elif fld == 'self_in' and g.self_in != 0 and random.random() < 0.2:
        g.self_in = 0
    elif random.random() < 0.5:
        v = getattr(g, fld)
        for bit in random.sample(range(16), random.randint(1, 3)):
            v ^= 1 << bit
        setattr(g, fld, v)
    else:
        setattr(g, fld, _other_lut(getattr(g, fld)))
    return g


def _mutate_io_tag(genome, io_placement=None):
    """Mutate one body priority, type/selector pair, or spatial anchor."""
    from substrates.nervous.io_placement import mutate_io_allele
    mutate_io_allele(genome, LUT_STATES, strategy=io_placement)


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
# I/O alleles mutate separately when an evolvable placement strategy is active.
# (evolve_io) - off by default, so the ordinary mutation stream is unchanged.
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]
IO_MUTATION_PROB = 0.20


def mutate_input_layout(genome, max_telomere=MAX_TELOMERE):
    """Move one non-anchor source pad by one cardinal square-lattice edge."""
    layout = getattr(genome, 'input_layout', None)
    if layout is None or len(layout) < 2:
        return False
    try:
        sites = [tuple(map(int, cell)) for cell in layout]
    except (TypeError, ValueError):
        return False
    domain = set(input_layout_domain(
        input_layout_radius(max_telomere, len(sites))))
    occupied = set(sites)
    indices = list(range(1, len(sites)))
    random.shuffle(indices)
    for index in indices:
        x, y = sites[index]
        options = [
            (x + dx, y + dy)
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
            if (x + dx, y + dy) in domain
            and (x + dx, y + dy) not in occupied
        ]
        if not options:
            continue
        sites[index] = random.choice(options)
        genome.input_layout = tuple(sites)
        return True
    return False


def _mutate_once_lut(genome, max_telomere=MAX_TELOMERE,
                     chromosome_count=None, evolve_io=False,
                     io_placement=None, function_families=None):
    """Apply one feasible, state-changing mutation to a LUT genome.

    ``evolve_io`` adds the I/O-tag and wiring-chromosome mutations that let an
    evolvable io_placement strategy re-wire its ports; off by default so the
    ordinary mutation stream is byte-identical."""
    if not genome.chromosomes:
        genome.chromosomes.append(random_lut_chromosome(
            function_families=function_families))
        return

    chroms = genome.chromosomes
    has_wiring = any(getattr(chromosome, 'wiring', False)
                     for chromosome in chroms)
    body = [chromosome for chromosome in chroms
            if not getattr(chromosome, 'wiring', False)]
    with_genes = [chromosome for chromosome in body if chromosome.genes]
    options = []
    if with_genes:
        options.append('tweak')
    if any(chromosome.genes and len(chromosome.genes) < _gene_cap()
           for chromosome in body):
        options.append('duplicate')
    if any(len(chromosome.genes) < _gene_cap() for chromosome in body):
        options.append('add_gene')
    if any(len(chromosome.genes) > 1 for chromosome in body):
        options.append('del_gene')
    if chromosome_count is None and not has_wiring and len(chroms) < MAX_CHROMS:
        options.append('add_chrom')
    if chromosome_count is None and not has_wiring and len(body) > 1:
        options.append('del_chrom')
    split_options = []
    for chromosome in body:
        values = [value for value in range(1, len(chromosome.genes))
                  if value != chromosome.split]
        if values:
            split_options.append((chromosome, values))
    if split_options:
        options.append('split')
    telomere_options = []
    for chromosome in body:
        base = getattr(chromosome, 'telomere', 10)
        values = [base + delta for delta in (-3, -2, -1, 1, 2, 3)
                  if 1 <= base + delta <= max_telomere]
        if values:
            telomere_options.append((chromosome, values))
    if telomere_options:
        options.append('telomere')

    weights = [_MUT_WEIGHTS[_MUT_OPS.index(op)] for op in options]
    op = random.choices(options, weights=weights)[0]
    if op == 'tweak':
        chromosome = random.choice(with_genes)
        index = random.randrange(len(chromosome.genes))
        chromosome.genes[index] = _tweak_gene(
            chromosome.genes[index], function_families)
    elif op == 'duplicate':
        chromosome = random.choice([
            item for item in body
            if item.genes and len(item.genes) < _gene_cap()
        ])
        chromosome.genes.insert(
            random.randrange(len(chromosome.genes) + 1),
            _tweak_gene(
                random.choice(chromosome.genes), function_families))
    elif op == 'add_gene':
        chromosome = random.choice([
            item for item in body if len(item.genes) < _gene_cap()
        ])
        chromosome.genes.append(random_lut_gene(function_families))
    elif op == 'del_gene':
        chromosome = random.choice([
            item for item in body if len(item.genes) > 1
        ])
        chromosome.genes.pop(random.randrange(len(chromosome.genes)))
    elif op == 'add_chrom':
        chroms.append(random_lut_chromosome(
            function_families=function_families))
    elif op == 'del_chrom':
        chroms.remove(min(body, key=lambda item: len(item.genes)))
    elif op == 'split':
        chromosome, values = random.choice(split_options)
        chromosome.split = random.choice(values)
    else:  # telomere
        chromosome, values = random.choice(telomere_options)
        chromosome.telomere = random.choice(values)


def mutate_lut(genome, mean_mutations=None, max_telomere=MAX_TELOMERE,
               chromosome_count=None, evolve_io=False, io_placement=None,
               io_mutations=1, coordinated_io=False,
               function_families=None):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    if is_branched(genome):
        # The branched operator owns its own I/O alleles and arm controls; the
        # native layout / tag arguments have nothing to act on and are accepted
        # and ignored, because the breeding loop passes them for every encoding.
        from .branched_ga import MAX_TELOMERE as BRANCHED_MAX_TELOMERE
        from .branched_ga import mutate_branched_lut
        child = mutate_branched_lut(
            genome, MEAN_MUTATIONS if mean_mutations is None else mean_mutations,
            max_telomere=min(int(max_telomere), BRANCHED_MAX_TELOMERE))
        if hasattr(genome, '_mut_rate'):
            child._mut_rate = genome._mut_rate
        return child
    g = clone_genome(genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    events = max(1, _poisson(lam))
    for _ in range(events - 1):
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=io_placement,
            function_families=function_families)
    if events == 1:
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=io_placement,
            function_families=function_families)
    else:
        _force_nonparent_tweak(g, genome, function_families)
    if evolve_io and random.random() < IO_MUTATION_PROB:
        if coordinated_io:
            from substrates.nervous.io_placement import mutate_io_bundle
            mutate_io_bundle(
                g, LUT_STATES, strategy=io_placement,
                count=max(2, int(io_mutations)))
        else:
            for _ in range(max(1, int(io_mutations))):
                _mutate_io_tag(g, io_placement=io_placement)
    if (getattr(g, 'input_layout', None) is not None
            and len(g.input_layout) > 1
            and random.random() < IO_MUTATION_PROB):
        mutate_input_layout(g, max_telomere)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    # Also makes switching an existing/legacy population to a restricted
    # inventory safe: inherited executable alleles are projected even when
    # this particular mutation event touched some other gene field.
    return constrain_genome_functions(g, function_families)


def crossover_lut(pa, pb):
    """Tag-matched hierarchical crossover, including one-rule chromosomes."""
    if is_branched(pa) or is_branched(pb):
        if not (is_branched(pa) and is_branched(pb)):
            # The two encodings do not share a meaningful crossover unit.
            # Preserve both parents and leave a subsequent mutation as the
            # first legal variation instead of forcing an arm transplant.
            return clone_genome(pa), clone_genome(pb)
        # Branched recombination trades whole ARMS and returns one child, so the
        # reciprocal cross supplies the second.
        from .branched_ga import crossover_branched_lut
        return (crossover_branched_lut(pa, pb),
                crossover_branched_lut(pb, pa))
    ca, cb = clone_genome(pa), clone_genome(pb)
    layout_a = getattr(pa, 'input_layout', None)
    layout_b = getattr(pb, 'input_layout', None)
    if (layout_a is not None and layout_b is not None
            and len(layout_a) == len(layout_b) and random.random() < 0.5):
        # The pad arrangement is one co-adapted physical module. Exchange it
        # whole; per-coordinate crossover could manufacture collisions.
        ca.input_layout = tuple(tuple(cell) for cell in layout_b)
        cb.input_layout = tuple(tuple(cell) for cell in layout_a)
    used_b = set()
    for i, chrom_a in enumerate(ca.chromosomes):
        best_j, best_dist = None, float("inf")
        for j, chrom_b in enumerate(cb.chromosomes):
            if j in used_b:
                continue
            if (getattr(chrom_a, 'wiring', False)
                    != getattr(chrom_b, 'wiring', False)):
                continue
            d = abs(chrom_a.tag - chrom_b.tag)
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is None:
            continue
        used_b.add(best_j)
        chrom_b = cb.chromosomes[best_j]
        # Snapshot both inputs: chrom_a aliases child A, so assigning A before
        # reading its reciprocal suffix used to make child B a parent-B clone.
        genes_a, genes_b = chrom_a.genes[:], chrom_b.genes[:]
        common = min(len(genes_a), len(genes_b))
        if common >= 2:
            sp = max(1, min(int(chrom_a.split), common - 1))
            ca.chromosomes[i].genes = genes_a[:sp] + genes_b[sp:]
            cb.chromosomes[best_j].genes = genes_b[:sp] + genes_a[sp:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = sp
        elif common == 1:
            fields = (('tag', 'io_selector')
                      if getattr(chrom_a, 'wiring', False) else None)
            gene_a, gene_b = _recombine_gene_fields(
                genes_a[0], genes_b[0], fields=fields)
            ca.chromosomes[i].genes = [gene_a] + genes_b[1:]
            cb.chromosomes[best_j].genes = [gene_b] + genes_a[1:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = 0
        else:
            ca.chromosomes[i].genes = genes_b
            cb.chromosomes[best_j].genes = genes_a
        _normalize_split(ca.chromosomes[i])
        _normalize_split(cb.chromosomes[best_j])
    for chromosome in ca.chromosomes + cb.chromosomes:
        _normalize_split(chromosome)
    return ca, cb


def n_genes(genome):
    return sum(len(c.genes) for c in genome.chromosomes)


# -- neutral compaction (gene expression) -----------------------------------------
# A dense ontogeny genome carries genes that win NO growth lookup - they are
# unexpressed, and (proven) removing a never-winning gene cannot change any argmin,
# so the grown organism is bit-identical and the fitness is unchanged. Compaction
# drops exactly those: the threshold is zero expression (a neutrality boundary, not
# a tuned cutoff), leaving the genome's functional core with behaviour untouched.
# Biologically: pseudogene loss. NB measured only a modest shrink on fresh ontogeny
# seeds (~7-14% unexpressed - the density is mostly FUNCTIONAL morphology, not junk),
# so this is a human-facing cleanup for inspecting/refining an evolved genome
# (the Designer), NOT an evolution driver - it did not improve solving in tests.
def compact_genome(genome, seeds, grid_size=7, iters=30):
    """Return a copy of `genome` with all never-expressed genes removed (neutral:
    same grown organism, same fitness). `seeds` must be the SAME seeds the genome
    is grown with, or the expression tally is for a different organism."""
    _, counts = grow_lut_tracked(genome, seeds=tuple(seeds),
                                 grid_size=grid_size, iters=iters)
    new_chroms, k = [], 0
    for c in genome.chromosomes:
        if getattr(c, 'wiring', False):
            new_chroms.append(Chromosome(
                genes=list(c.genes), split=c.split, tag=c.tag,
                telomere=getattr(c, 'telomere', MAX_TELOMERE), wiring=True))
            continue
        kept = [g for gi, g in enumerate(c.genes) if counts[k + gi] > 0]
        k += len(c.genes)
        if kept:
            split = (0 if len(kept) < 2 else
                     max(1, min(int(c.split), len(kept) - 1)))
            new_chroms.append(Chromosome(
                genes=kept, split=split,
                tag=c.tag, telomere=getattr(c, 'telomere', MAX_TELOMERE),
                wiring=False))
    if not any(not getattr(c, 'wiring', False) for c in new_chroms):
        # Every BODY gene was unexpressed: keep one developmental rule. The
        # wiring chromosome, if present, is metadata and cannot be that rule.
        c0 = next(c for c in genome.chromosomes
                  if not getattr(c, 'wiring', False))
        new_chroms.insert(0, Chromosome(
            genes=c0.genes[:1], split=0, tag=c0.tag,
            telomere=getattr(c0, 'telomere', MAX_TELOMERE), wiring=False))
    return Genome(
        chromosomes=new_chroms, tag=genome.tag,
        seed_state=getattr(genome, 'seed_state', None),
        provenance=getattr(genome, 'provenance', ''),
        input_layout=(
            None if getattr(genome, 'input_layout', None) is None
            else tuple(tuple(cell) for cell in genome.input_layout)),
        edge_input_layout=(
            None if getattr(genome, 'edge_input_layout', None) is None
            else tuple(int(value) for value in genome.edge_input_layout)))


# -- ontogeny seeding is THE seed path (see substrates/lut/ontogeny.py) ------------------
# Sparse random genomes grow near-uniform DIAMONDS - the "uninteresting" LUTs.
# Richness tracks gene density (sim6's table_create invents a gene per unseen
# context), so the population/immigrants are seeded with sim6-style biomorph
# genomes, and the smaller-genome parsimony tie-break is dropped: it pruned the
# dense genomes straight back to sparse (re-diamonding them). This is a user
# decision (2026-07-07): rich morphology over solve speed, no random/ontogeny
# switch. Size is bounded by ONTOGENY_CAP + telomeres instead of parsimony.
ONTOGENY_CAP  = 350               # max genes kept from an ontogeny biomorph


# Growing one sim6 ontogeny seed is expensive (measured ~0.3 s: up to 60
# biomorph growths per accepted genome), and make_seed_genome is called for
# EVERY immigrant EVERY generation - that alone was a large share of ontogeny
# mode's ~50x slowdown. So the factory grows only the first _ONTO_POOL_SIZE
# genomes per chromosome count and caches them; later calls return a mutated
# variant of a random pool member (still fresh genetic material, ~free).
_ONTO_POOL      = {}     # n_chroms -> [cached seed genomes]
_ONTO_POOL_SIZE = 24


def make_seed_genome(n_chroms=2):
    """The population/immigrant factory: a dense sim6-style biomorph genome
    (the rich-morphology seeds - see the ONTOGENY_CAP note above)."""
    from .ontogeny import random_ontogeny_genome
    pool = _ONTO_POOL.setdefault(n_chroms, [])
    if len(pool) < _ONTO_POOL_SIZE:
        g = random_ontogeny_genome(n_chroms, cap_genes=ONTOGENY_CAP)
        pool.append(g)
        return clone_genome(g)     # the pool master stays pristine
    return mutate_lut(random.choice(pool), chromosome_count=n_chroms)


def _tiebreak(genome):
    """Tie-break term for equal fitness: NEUTRAL - the smaller-genome parsimony
    pressure was what pruned dense ontogeny genomes back to sparse diamonds
    (re-diamonding). Size is bounded by ONTOGENY_CAP + telomeres instead."""
    return 0


def rank_key(genome, fitness):
    """Selection-only wiring viability, then honest behavior, then the escape
    objectives, then tie-break.

    Robustness and juvenile (lifespan) credit sit strictly BELOW fitness, so
    neither can trade against correctness; both are 0.0 when their mechanism is
    off, which collapses this to the original three-tier key. See
    substrates/nervous/ga.rank_key for the full argument."""
    from substrates.nervous.io_placement import binding_viability
    return (binding_viability(genome), fitness,
            getattr(genome, '_robustness', 0.0) or 0.0,
            getattr(genome, '_juvenile_score', 0.0) or 0.0,
            getattr(genome, '_topology_score', 0.0) or 0.0,
            _tiebreak(genome))


def _gene_cap():
    """Per-chromosome gene cap for add/duplicate mutations - sized for dense
    ontogeny genomes (MAX_GENES was the sparse random path's tight cap)."""
    return ONTOGENY_CAP


def plateau_rescue_candidates(
        genome, target, limit=48, max_telomere=MAX_TELOMERE,
        function_families=None):
    """Legitimate rescue genomes for a stalled spatial-I/O LUT run.

    Three failure modes are covered:

    * A retained periodic truth table with at most four inputs/outputs is
      compiled into a compact crossbar, inverse-grown into an ordinary genome,
      and verified. This is an explicit feasibility witness and gives genuinely
      hard arithmetic targets a reachable rescue seed.

    * For a two-input/two-output body, enumerate compact 2x2 port motifs.  The
      four ports are moved together, allowing selection to cross valleys where
      every one-port intermediate is worse.
    * Flip one bit of a rule whose output LUT is expressed at a currently bound
      output cell.  Maintenance rules are tried before growth rules so a local
      logic repair need not disturb morphology.

    Every proposal remains an ordinary genome and is evaluated by the unchanged
    target contract. The compiler uses the target's declared truth table; the
    local motif/rule neighbourhood does not.
    """
    limit = max(0, int(limit))
    if limit == 0:
        return []
    # The live fixed-pad encoding used to fall through here with no rescue at
    # all: the existing truth-table compiler below only speaks the retired
    # spatial-I/O genome. Compile the hard arithmetic witness into an ordinary
    # output-rooted genome instead. It is still grown and evaluated normally;
    # this merely makes the already-proven circuit reachable after a plateau.
    if is_branched(genome):
        from .branched_synthesis import synthesize_branched_truth_table
        from .state_synthesis import synthesize_branched_dynamic
        from .synthesis import SynthesisError
        for compiler in (
                synthesize_branched_truth_table,
                synthesize_branched_dynamic):
            try:
                candidate = compiler(
                    target, chromosome_count=len(genome.chromosomes),
                    max_telomere=max_telomere,
                    function_families=function_families)
            except SynthesisError:
                continue
            fitness, cases = evaluate_lut_full(candidate, target)
            if fitness == 1.0 and cases and min(cases) == 1.0:
                return [candidate]
        return []
    from substrates.nervous.io_placement import (
        bind_io, flat_inputs, flat_outputs, growth_seeds, io_strategy,
        set_spatial_port_positions)
    if io_strategy(target) != 'spatial_chromosome':
        return []
    seen = {_recombination_signature(genome)}

    def unique(candidate, destination):
        signature = _recombination_signature(candidate)
        if signature in seen:
            return
        seen.add(signature)
        destination.append(candidate)

    families = normalise_function_families(function_families)
    compiled = []
    if UNRESTRICTED in families:
        from .synthesis import (
            SynthesisError, synthesize_combinational_genome)
        try:
            result = synthesize_combinational_genome(
                target, chromosome_count=len(genome.chromosomes),
                max_telomere=max_telomere)
            unique(result.genome, compiled)
        except SynthesisError:
            pass

    if len(compiled) >= limit:
        return compiled[:limit]
    from substrates.nervous.io_placement import spatial_output_variants
    readouts = []
    for candidate in spatial_output_variants(
            genome, target,
            limit=max(1, (limit - len(compiled)) // 2)):
        unique(candidate, readouts)
    grid = grow_lut(
        genome, seeds=growth_seeds(
            target, io_strategy(target), genome),
        grid_size=target.grid_size, iters=target.iters)
    if not grid:
        return (compiled + readouts)[:limit]
    from .lut import cell_io_tags
    bound = bind_io(
        genome, grid, target, 'spatial_chromosome',
        tags=cell_io_tags(genome, grid))
    if bound is None:
        return (compiled + readouts)[:limit]
    in_pos, out_pos = bound
    current = flat_inputs(in_pos) + flat_outputs(out_pos)
    positions = sorted(grid)
    motif = []
    if target.n_inputs == 2 and len(target.outputs) == 2 and len(current) == 4:
        from itertools import permutations
        cells = set(grid)
        assignments = []
        for x, y in positions:
            square = {
                (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
            if not square.issubset(cells):
                continue
            diagonals = (
                ([(x, y), (x + 1, y + 1)],
                 [(x + 1, y), (x, y + 1)]),
                ([(x + 1, y), (x, y + 1)],
                 [(x, y), (x + 1, y + 1)]),
            )
            for inputs, outputs in diagonals:
                for input_order in permutations(inputs):
                    for output_order in permutations(outputs):
                        assignment = list(input_order + output_order)
                        distance = sum(
                            abs(old[0] - new[0]) + abs(old[1] - new[1])
                            for old, new in zip(current, assignment))
                        assignments.append((distance, assignment))
        assignments.sort(key=lambda item: (item[0], item[1]))
        for _distance, assignment in assignments:
            candidate = clone_genome(genome)
            if set_spatial_port_positions(
                    candidate, positions, assignment,
                    target=target) == len(assignment):
                unique(candidate, motif)

    rule = []
    output_values = {
        int(value)
        for pos in flat_outputs(out_pos)
        for value in grid.get(pos, ())
        if int(value)
    }
    loci = []
    for chromosome_index, chromosome in enumerate(genome.chromosomes):
        if getattr(chromosome, 'wiring', False):
            continue
        for gene_index, gene in enumerate(chromosome.genes):
            if int(gene.self_out) in output_values:
                # Maintenance before growth, then deterministic locus order.
                loci.append((
                    0 if int(gene.self_in) else 1,
                    chromosome_index, gene_index))
    loci.sort()
    for _kind, chromosome_index, gene_index in loci:
        for bit in range(16):
            candidate = clone_genome(genome)
            chromosome = candidate.chromosomes[chromosome_index]
            gene = copy.copy(chromosome.genes[gene_index])
            if unrestricted_only(families):
                gene.self_out ^= 1 << bit
            else:
                gene.self_out = mutate_function_table(
                    gene.self_out, families)
            chromosome.genes[gene_index] = gene
            unique(candidate, rule)

    # Reserve room for both rescue families.  Unused quota from one is filled
    # by the other, so small/simple bodies do not waste evaluation slots.
    remaining = limit - len(compiled) - len(readouts)
    motif_quota = remaining // 2
    rule_quota = remaining - motif_quota
    selected = (
        compiled + readouts + motif[:motif_quota] + rule[:rule_quota])
    if len(selected) < limit:
        selected += motif[motif_quota:motif_quota + limit - len(selected)]
    if len(selected) < limit:
        selected += rule[rule_quota:rule_quota + limit - len(selected)]
    return selected[:limit]


def tournament_lut(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(
        idx, key=lambda i: rank_key(population[i], fitnesses[i]))]


def select_parent(population, fitnesses, case_vecs=None):
    import substrates.nervous.ga as _nga
    if (_nga.SELECTION == 'lexicase' and case_vecs is not None
            and case_vecs[0] is not None):
        return _nga._lexicase_parent(population, case_vecs)
    return tournament_lut(population, fitnesses)


def consolidate_population(parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
    """Terminal parent/offspring selection after fitness 1.0 is first reached."""
    pop = len(parents)
    if (len(parent_fitnesses) != pop or len(offspring) != pop
            or len(offspring_fitnesses) != pop):
        raise ValueError('parent/offspring population sizes must match')
    case_vectors = parent_cases is not None or offspring_cases is not None
    if case_vectors and (parent_cases is None or offspring_cases is None):
        raise ValueError('parent and offspring case vectors must both be present')
    if case_vectors and (len(parent_cases) != pop or len(offspring_cases) != pop):
        raise ValueError('case-vector count must match population size')

    genomes = list(parents) + list(offspring)
    fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
    cases = ((list(parent_cases) + list(offspring_cases))
             if case_vectors else None)
    order = list(range(len(genomes)))
    random.shuffle(order)
    order.sort(key=lambda i: rank_key(genomes[i], fitnesses[i]), reverse=True)
    keep = order[:pop]
    return ([genomes[i] for i in keep],
            [fitnesses[i] for i in keep],
            ([cases[i] for i in keep] if cases is not None else None))


def _append_role_module_candidates(children, population, fitnesses,
                                   recombination):
    """Keep and join measured output specialists in compatible pad cohorts."""
    if (not population or not is_branched(population[0])
            or len(getattr(population[0], 'outputs', ())) < 2):
        return
    from .branched_ga import assemble_role_modules, input_pads
    roles = tuple(getattr(population[0], 'outputs', ()))
    groups = {}
    for index, genome in enumerate(population):
        groups.setdefault(tuple(input_pads(genome)), []).append(index)

    assemblies = []
    if recombination:
        for indices in groups.values():
            donors, donor_scores = {}, []
            for role_index, role in enumerate(roles):
                eligible = [index for index in indices
                            if len(getattr(population[index], '_output_scores', ()))
                            > role_index]
                if not eligible:
                    break
                best = max(eligible, key=lambda index: (
                    population[index]._output_scores[role_index],
                    rank_key(population[index], fitnesses[index])))
                donors[int(role.branch_id)] = population[best]
                donor_scores.append(population[best]._output_scores[role_index])
            else:
                if len(set(id(donor) for donor in donors.values())) > 1:
                    base = max(indices, key=lambda index: rank_key(
                        population[index], fitnesses[index]))
                    assemblies.append(((min(donor_scores), sum(donor_scores)),
                                       assemble_role_modules(
                                           population[base], donors)))
    if assemblies and len(children) < len(population):
        children.append(max(assemblies, key=lambda item: item[0])[1])

    seen = {genome_signature(genome) for genome in children}
    for role_index, _role in enumerate(roles):
        if len(children) >= len(population):
            break
        eligible = [index for index, genome in enumerate(population)
                    if len(getattr(genome, '_output_scores', ())) > role_index]
        if not eligible:
            continue
        best = max(eligible, key=lambda index: (
            population[index]._output_scores[role_index],
            rank_key(population[index], fitnesses[index])))
        signature = genome_signature(population[best])
        if signature not in seen:
            children.append(clone_genome(population[best]))
            seen.add(signature)


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None, ga_config=None,
                    chromosome_count=None, recombination=True,
                    evolve_io=False, io_placement=None, archive_parent=None,
                    stagnation=0, rescue_candidates=None, escape=None,
                    mutation_limit=None, function_families=None):
    """One generation of a steady, exploratory GA - elitism + immigrants +
    recombination/mutation, parents via epsilon-lexicase when per-case vectors are
    available (temporal targets), else tournament. Elites are breeders only;
    this operator always returns an entirely new generation. `mean_mutations`
    overrides the mutation rate for this generation (used by the annealing
    schedule)."""
    pop = len(population)
    elite_count = (ELITE_COUNT if ga_config is None else ga_config.elite_count)
    immigrant_fraction = (IMMIGRANT_FRAC if ga_config is None
                          else ga_config.immigrant_fraction)
    tournament_size = (TOURNAMENT_K if ga_config is None
                       else ga_config.tournament_size)
    # Escape mechanisms (runtime/escape.py). Resolved the same way the nervous
    # breeder resolves them, so both substrates and both drive paths breed
    # under one set of rules.
    if escape is None:
        from runtime.escape import OFF
        escape = (getattr(ga_config, 'escape', None) if ga_config is not None
                  else None) or OFF
    if mutation_limit is None:
        mutation_limit = (getattr(ga_config, 'mutation_limit', 8.0)
                          if ga_config is not None else 8.0)
    recombination = (
        recombination and
        (getattr(ga_config, 'recombination_enabled', True)
         if ga_config is not None else True))
    if ga_config is not None and ga_config.chromosome_count is not None:
        chromosome_count = ga_config.chromosome_count
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    import substrates.nervous.ga as _nga
    selection = (_nga.SELECTION if ga_config is None else ga_config.selection)
    strategy = (io_placement or (
        getattr(ga_config, 'io_placement', 'fixed')
        if ga_config is not None else 'fixed'))
    if function_families is None:
        function_families = (
            getattr(ga_config, 'lut_function_families', None)
            if ga_config is not None else None)
    function_families = normalise_function_families(function_families)
    evolve_io = evolve_io or strategy in (
        'terminal_nodes', 'tag_rank', 'wiring_chromosome',
        'spatial_chromosome')
    if make_genome is None:
        if evolve_io:
            from substrates.nervous.io_placement import (
                seed_io_metadata, seed_terminal_kinds)
            inferred_ports = 0
            if strategy in (
                    'wiring_chromosome', 'spatial_chromosome') and population:
                from substrates.nervous.io_placement import wiring_chromosome
                mapping = wiring_chromosome(population[0])
                inferred_ports = len(mapping.genes) if mapping is not None else 0
            def make_genome():
                genome = seed_io_metadata(
                    constrain_genome_functions(
                        make_seed_genome(chromosome_count or 2),
                        function_families),
                    wiring_chromosome=(strategy == 'wiring_chromosome'),
                    spatial_chromosome=(strategy == 'spatial_chromosome'),
                    n_ports=inferred_ports,
                    tag_rank=(strategy == 'tag_rank'))
                if strategy == 'terminal_nodes':
                    # Direct next_population callers do not provide a target.
                    # Preserve the population's approximate role inventory.
                    reference = population[0] if population else None
                    kinds = [
                        int(getattr(gene, 'io_kind', 0))
                        for chromosome in (
                            reference.chromosomes if reference else ())
                        for gene in chromosome.genes]
                    seed_terminal_kinds(
                        genome, max(1, kinds.count(1) // 2),
                        max(1, kinds.count(2) // 2))
                return genome
        else:
            reference_layout = (
                getattr(population[0], 'input_layout', None)
                if population else None)
            if reference_layout is None:
                make_genome = lambda: constrain_genome_functions(
                    make_seed_genome(chromosome_count or 2),
                    function_families)
            else:
                def make_genome():
                    genome = constrain_genome_functions(
                        make_seed_genome(chromosome_count or 2),
                        function_families)
                    genome.input_layout = random_input_layout(
                        len(reference_layout),
                        (MAX_TELOMERE if ga_config is None
                         else ga_config.max_telomere))
                    return genome
    n_elite = elite_count if elite_count is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = min(int(round(pop * immigrant_fraction)), pop)
    order   = sorted(
        range(pop),
        key=lambda i: rank_key(population[i], fitnesses[i]),
        reverse=True)
    # Elites choose parents here but are never copied into the next generation.
    # Plateau-rescue proposals and archived-champion descendants are still new,
    # mutated genomes; no evaluated parent survives into this generation.
    new_pop = [
        constrain_genome_functions(
            clone_genome(candidate), function_families)
        for candidate in list(rescue_candidates or ())[:pop]]
    remaining = pop - len(new_pop)
    new_pop += [
        constrain_genome_functions(make_genome(), function_families)
        for _ in range(min(n_imm, remaining))]
    if (archive_parent is not None and stagnation >= STRESS_PATIENCE
            and len(new_pop) < pop):
        archive_count = min(
            max(1, int(round(pop * 0.10))), pop - len(new_pop))
        for index in range(archive_count):
            new_pop.append(mutate_lut(
                archive_parent, mean_mutations, max_telomere=(
                    MAX_TELOMERE if ga_config is None
                    else ga_config.max_telomere),
                chromosome_count=chromosome_count, evolve_io=evolve_io,
                io_placement=strategy,
                io_mutations=2,
                coordinated_io=(evolve_io and index % 2 == 0),
                function_families=function_families))
    _append_role_module_candidates(new_pop, population, fitnesses,
                                   recombination)
    use_lexicase = (selection == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    # One downsampled case set per generation, shared by every selection event.
    case_subset = None
    if use_lexicase:
        from runtime.escape import lexicase_case_subset
        case_subset = lexicase_case_subset(len(case_vecs[0]), escape)
    residual = order[n_elite:] if n_elite < pop else order
    elite_pool = order[:n_elite] if n_elite else order
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]
    recombination_environments = [
        _recombination_environment(genome)
        for genome in population]

    def pick_index(candidates):
        if use_lexicase:
            local_population = [population[index] for index in candidates]
            local_cases = [case_vecs[index] for index in candidates]
            chosen = _nga._lexicase_parent(
                local_population, local_cases, case_subset)
            return candidates[next(
                index for index, genome in enumerate(local_population)
                if genome is chosen)]
        k = min(tournament_size, len(candidates))
        return max(
            random.sample(candidates, k),
            key=lambda index: rank_key(
                population[index], fitnesses[index]))

    def parent_pair():
        if pop == 1:
            return population[0], population[0]

        def mate_pool(first, candidates):
            distinct = [
                index for index in candidates
                if recombination_signatures[index]
                != recombination_signatures[first]]
            pool = distinct or candidates
            environment = recombination_environments[first]
            if environment is not None:
                compatible = [
                    index for index in pool
                    if recombination_environments[index] == environment]
                if compatible:
                    pool = compatible
            return pool

        def choose(exclude=None):
            if use_lexicase:
                candidates = [index for index in range(pop) if index != exclude]
                return pick_index(candidates)
            candidates = [index for index in elite_pool if index != exclude]
            exploratory = [index for index in residual if index != exclude]
            if exploratory and random.random() < EXPLORATION_PARENT_FRAC:
                candidates = exploratory
            if not candidates:
                candidates = [index for index in range(pop) if index != exclude]
            return pick_index(candidates)

        first = choose()
        candidates = [index for index in range(pop) if index != first]
        parent_pool = mate_pool(first, candidates)
        if use_lexicase:
            from runtime.escape import complementary_parent_index
            second = complementary_parent_index(
                first, parent_pool, case_vecs, fitnesses,
                case_subset)
        else:
            second = choose(first)
            if second not in parent_pool:
                second = pick_index(parent_pool)
        return population[first], population[second]

    # mean_mutations is None on the direct-API path (mutate_lut then uses its
    # own default); self-adaptation needs a real number to perturb.
    adaptive_base = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    if escape.self_adaptive_mutation:
        # Immigrants have no lineage to inherit a rate from; start them on a
        # randomised spread around the run rate (see substrates.nervous.ga).
        from runtime.escape import seed_mutation_rate
        for genome in new_pop:
            if not hasattr(genome, '_mut_rate'):
                seed_mutation_rate(genome, adaptive_base, mutation_limit)

    def child_rate(child):
        if not escape.self_adaptive_mutation:
            return mean_mutations
        from runtime.escape import mutation_rate_of
        return mutation_rate_of(child, adaptive_base)

    # Mutation remains the main generator of local variants.  This bounded
    # cohort answers a different question: did crossover itself join two useful
    # inherited modules?  Without it, every such join is obscured before the
    # evaluator can ever select for it.
    crossover_slots = min(
        pop - len(new_pop),
        max(1, int(round(pop * RECOMBINATION_EVALUATION_FRACTION))))
    while recombination and pop > 1 and crossover_slots > 0:
        pa, pb = parent_pair()
        ca, cb = crossover_lut(pa, pb)
        if escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate
            inherit_mutation_rate(ca, pa, pb, escape, adaptive_base,
                                  mutation_limit)
            inherit_mutation_rate(cb, pb, pa, escape, adaptive_base,
                                  mutation_limit)
        new_pop.append(constrain_genome_functions(ca, function_families))
        crossover_slots -= 1
        if crossover_slots > 0 and len(new_pop) < pop:
            new_pop.append(constrain_genome_functions(cb, function_families))
            crossover_slots -= 1

    while len(new_pop) < pop:
        pa, pb = parent_pair()
        ca, cb = (crossover_lut(pa, pb) if recombination else
                  (clone_genome(pa), clone_genome(pb)))
        if escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate
            inherit_mutation_rate(ca, pa, pb, escape, adaptive_base,
                                  mutation_limit)
            inherit_mutation_rate(cb, pb, pa, escape, adaptive_base,
                                  mutation_limit)
        max_telomere = (MAX_TELOMERE if ga_config is None
                        else ga_config.max_telomere)
        new_pop.append(mutate_lut(
            ca, child_rate(ca), max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=strategy,
            function_families=function_families))
        if len(new_pop) < pop:
            new_pop.append(mutate_lut(
                cb, child_rate(cb), max_telomere,
                chromosome_count=chromosome_count, evolve_io=evolve_io,
                io_placement=strategy,
                function_families=function_families))
    if (chromosome_count is not None
            and any(len(genome.chromosomes) != chromosome_count
                    for genome in new_pop)):
        raise ValueError('genome factory violated configured chromosome count')
    return new_pop[:pop]


def evolve_lut(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True,
               seed=None, escape=None, ga_config=None):
    import dataclasses as _dc
    from runtime.config import GAConfig
    from runtime.escape import build_escape_state, OFF
    if ga_config is None:
        ga_config = GAConfig(chromosome_count=n_chroms)
    if escape is not None:
        ga_config = _dc.replace(ga_config, escape=escape)
    if (not getattr(target, 'temporal', False)
            or getattr(target, 'combinational_cases', ())
            or getattr(target, 'temporal_logic_cases', ())):
        ga_config = _dc.replace(ga_config, selection='lexicase')
    function_families = normalise_function_families(
        getattr(ga_config, 'lut_function_families', None))
    setattr(
        target, 'lut_io_mode',
        getattr(ga_config, 'lut_io_mode', 'source_pads'))
    setattr(target, '_lut_function_families', function_families)
    if seed is not None:
        random.seed(seed)
        # _ONTO_POOL survives for the life of the PROCESS, so without this a
        # seeded run is only reproducible when it happens to run first. The
        # first call into a fresh process grows 24 seed genomes and caches
        # them; every later call draws its whole population from that cache
        # instead - masters grown under a PREVIOUS target's RNG stream. Same
        # seed, same config, different answer depending on what ran before:
        # LUT AND scored 1.000 run first and 0.750 run after Half adder, which
        # silently contaminated rows 2..46 of every multi-target sweep.
        # Clearing it costs ~24 ontogeny growths (~7s) against 100-600s rows,
        # and only on the seeded path, so interactive runs keep the cache.
        _ONTO_POOL.clear()
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    from substrates.nervous.io_placement import (
        growth_seeds, io_strategy, seed_io_metadata,
        seed_spatial, seed_wiring_from_phenotype,
        uses_port_chromosome)
    strategy = io_strategy(target)
    if uses_port_chromosome(strategy) and n_chroms < 3:
        raise ValueError('chromosome-based I/O requires n_chroms >= 3')
    evolve_io = strategy in (
        'terminal_nodes', 'tag_rank', 'wiring_chromosome',
        'spatial_chromosome')
    n_ports = target.n_inputs + len(target.outputs)
    if evolve_io:
        def make_genome():
            genome = seed_io_metadata(
                constrain_genome_functions(
                    make_seed_genome(n_chroms), function_families),
                wiring_chromosome=(strategy == 'wiring_chromosome'),
                spatial_chromosome=(strategy == 'spatial_chromosome'),
                n_ports=n_ports, tag_rank=(strategy == 'tag_rank'))
            if strategy == 'terminal_nodes':
                from substrates.nervous.io_placement import seed_terminal_kinds
                seed_terminal_kinds(
                    genome, target.n_inputs, len(target.outputs))
            if strategy == 'spatial_chromosome':
                seed_spatial(genome, None, target)
            elif uses_port_chromosome(strategy):
                from .lut import grow_lut, cell_io_tags
                grid = grow_lut(
                    genome, seeds=growth_seeds(target, strategy, genome),
                    grid_size=target.grid_size, iters=target.iters)
                tags = cell_io_tags(genome, grid)
                seed_wiring_from_phenotype(
                    genome, grid, target, tags=tags)
            return genome
    else:
        def make_genome(input_genes=None):
            if lut_io_mode(target) != 'exterior_edges':
                # The live encoding: branched and output-rooted, exactly what
                # the desktop controller breeds. Keeping both drive paths on one
                # encoding is the point (see the two-GA-drive-paths note);
                # exterior-edge I/O still seeds natively because its inputs are
                # drivers outside the body, not pads the arms grow toward.
                from .branched_ga import (random_branched_lut_genome,
                                           select_developmental_seed)
                return select_developmental_seed(
                    lambda: random_branched_lut_genome(
                        n_chroms, max_telomere=ga_config.max_telomere,
                        n_inputs=target.n_inputs,
                        output_roles=tuple(terminal.role
                                           for terminal in target.outputs),
                        families=function_families,
                        input_genes=input_genes),
                    attempts=make_genome.developmental_seed_candidates)
            return constrain_genome_functions(
                make_seed_genome(n_chroms), function_families)
    if strategy == 'fixed' and lut_io_mode(target) != 'exterior_edges':
        make_genome.developmental_seed_candidates = 6
    # Escape mechanisms, resolved and attached exactly as the desktop
    # controller does it, so this headless driver and the app agree.
    escape_cfg = ga_config.escape or OFF
    setattr(target, '_escape', escape_cfg)
    cache       = LRUCache(FITNESS_CACHE_MAX)
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:
        if strategy == 'fixed' and lut_io_mode(target) != 'exterior_edges':
            population, cohort_inputs = [], []
            cohort_count = min(4, max(1, pop // 8))
            for index in range(pop):
                if index < cohort_count:
                    genome = make_genome()
                    cohort_inputs.append(list(genome.inputs))
                else:
                    genome = make_genome(
                        cohort_inputs[(index - cohort_count) % cohort_count])
                population.append(genome)
            make_genome.developmental_seed_candidates = 2
        else:
            population = [make_genome() for _ in range(pop)]
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        escape_state = build_escape_state(
            'lut', ga_config, chromosome_count=n_chroms,
            io_placement=strategy, evolve_io=evolve_io,
            lut_function_families=function_families)
        escape_state.apply_robustness_blend(population, max(fitnesses))
        bi = max(
            range(pop),
            key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        best_rank = rank_key(best_genome, best_fitness)
        escape_state.record_champion(0, best_genome, best_fitness)
        escape_state.note_contract_progress(cases, fitnesses)
        stagnation   = 0
        mut_rate     = MEAN_MUTATIONS        # annealing schedule (see MUT_DECAY)
        for gen in range(generations):
            mut_rate *= MUT_DECAY
            mm = adaptive_mutation_rate(mut_rate, stagnation,
                                        solved=best_fitness >= 1.0)
            parents, parent_fitnesses, parent_cases = population, fitnesses, cases
            rescue = (
                plateau_rescue_candidates(
                    best_genome, target, limit=min(48, max(1, pop // 2)),
                    function_families=function_families)
                if (stagnation >= STRESS_PATIENCE
                    and best_fitness < 1.0) else ())
            offspring = next_population(
                parents, parent_fitnesses, make_genome, parent_cases, mm,
                chromosome_count=n_chroms, evolve_io=evolve_io,
                io_placement=strategy, archive_parent=best_genome,
                stagnation=stagnation, rescue_candidates=rescue,
                escape=escape_cfg, mutation_limit=ga_config.mutation_limit,
                function_families=function_families)
            offspring_fitnesses, offspring_cases = eval_batch_cases(
                offspring, target, cache, ex)
            escape_state.apply_robustness_blend(
                list(parents) + list(offspring),
                max(best_fitness, max(offspring_fitnesses)))
            # Terminal consolidation once solved, otherwise optional crowding
            # plus the baseline rotating contract-elite reserve.
            population, fitnesses, cases = escape_state.merge_generation(
                parents, parent_fitnesses, parent_cases,
                offspring, offspring_fitnesses, offspring_cases,
                consolidate=consolidate_population,
                solved=max(best_fitness, max(offspring_fitnesses)) >= 1.0)
            gi = max(
                range(pop),
                key=lambda i: rank_key(population[i], fitnesses[i]))
            case_progress = escape_state.note_contract_progress(
                cases, fitnesses)
            if (fitnesses[gi] > best_fitness + 1e-12
                    or case_progress):
                stagnation = 0
            else:
                stagnation += 1
            generation_rank = rank_key(population[gi], fitnesses[gi])
            if escape_state.accepts(generation_rank, best_rank):
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
                best_rank = generation_rank
            escape_state.record_champion(gen, best_genome, best_fitness)
            population, fitnesses, cases, rebirth_info = \
                escape_state.maybe_rebirth(
                    gen, population, fitnesses, cases, mm, stagnation,
                    best_fitness,
                    lambda genomes: eval_batch_cases(
                        genomes, target, cache, ex))
            if rebirth_info is not None:
                stagnation = 0
                escape_state.note_contract_progress(cases, fitnesses)
                if verbose:
                    print('Rebirth at generation %d: %d genomes from ancestors '
                          '%s at rate %.2f'
                          % (gen, rebirth_info['reborn'],
                             rebirth_info['ancestors'], rebirth_info['rate']))
            escape_state.tick()
            if verbose and gen % 10 == 0:
                print("%5d  %6.4f  %6.4f" % (gen, best_fitness,
                                             sum(fitnesses) / pop))
        return best_genome, best_fitness
    finally:
        ex.shutdown()


def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None, executor=None, should_stop=None, on_progress=None,
              max_telomere=MAX_TELOMERE, chromosome_count=None,
              evolve_io=False, io_placement=None, function_families=None):
    """Build evaluated, rule-distinct valid offspring; see NV's implementation."""
    if cache is None:
        cache = LRUCache(FITNESS_CACHE_MAX)
    if batch is None:
        batch = max(48, pop_size)
    should_stop = should_stop or (lambda: False)
    seeds = list(seeds)
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count for genome in seeds):
            raise ValueError('seed violates configured chromosome count')
    _eval = lambda gs: eval_batch_cases(gs, target, cache, executor)[0]   # reuse pool
    pool, pool_signatures, seen = [], [], set()
    for g, f in zip(seeds, _eval(seeds)):
        s = _recombination_signature(g)
        if f >= valid and s not in seen:
            pool.append(g); pool_signatures.append(s); seen.add(s)
    if not pool:
        return pool
    for round_i in range(rounds):
        if len(pool) >= pop_size or should_stop():
            break
        if on_progress is not None:
            on_progress(round_i + 1, rounds, len(pool))
        cands, candidate_signatures = [], []
        for _ in range(batch):
            parent_index = random.randrange(len(pool))
            parent_a = pool[parent_index]
            signature_a = pool_signatures[parent_index]
            mates = [index for index, signature in enumerate(pool_signatures)
                     if signature != signature_a]
            if mates:
                base = random.choice(crossover_lut(
                    parent_a, pool[random.choice(mates)]))
            else:
                base = parent_a
            c = mutate_lut(base, max_telomere=max_telomere,
                           chromosome_count=chromosome_count,
                           evolve_io=evolve_io,
                           io_placement=io_placement,
                           function_families=function_families)
            s = _recombination_signature(c)
            if s not in seen:
                seen.add(s)
                cands.append(c); candidate_signatures.append(s)
        if should_stop():
            break
        for c, s, f in zip(cands, candidate_signatures, _eval(cands)):
            if f >= valid and len(pool) < pop_size:
                pool.append(c); pool_signatures.append(s)
    return pool[:pop_size]


# -- GUI report -------------------------------------------------------------------

def lut_report(ttarget, genome=None):
    """Temporal-target report for the LUT model. The report body is the shared
    scoring.score_report_lines; only growth/prep is LUT-specific here."""
    lines = ['Target: %s   [LUT array]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    relations = set(contract_relations(ttarget))
    prep = None if genome is None else prepare_lut(genome, ttarget)
    if relations & {'sustained_cadence', 'commanded_cadence'}:
        # rhythm modes measure real output; no expectation-only preview exists
        note = LUT_REPORT_NOTES.get(next(iter(relations), 'logical_state'))
        pre = ['', note] if note else []
        if genome is None:
            return '\n'.join(lines + [''] + behavior_contract_lines(ttarget) + pre + [
                '', '(run the GA or Load Saved to inspect a circuit)'])
        if prep is None:
            return '\n'.join(lines + [''] + behavior_contract_lines(ttarget) + pre + [
                '', '(circuit incomplete - grew too little or inputs dead)'])
    if genome is not None and prep is None:
        lines += ['', '(circuit incomplete - grew too little or inputs dead)']
    traces = prep[2] if prep is not None else None
    out_pos = prep[1] if prep is not None else None
    _, body = score_report_lines(ttarget, traces, out_pos,
                                 notes=LUT_REPORT_NOTES)
    lines += body
    if traces is None and genome is None:
        lines += ['', '(run the GA or Load Saved to inspect a circuit)']
    return '\n'.join(lines)
