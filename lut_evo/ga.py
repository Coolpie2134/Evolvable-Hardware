"""
lut_evo/ga.py — scoring and GA for the LUT array.

Problem definitions and executable behavior contracts are shared with the rest
of the project; only the substrate-to-observation adapter differs. Selection uses the same recipe as the
nervous GA — elites + random immigrants + tournament, NO early stop at 1.0 —
but WITHOUT the parsimony tie-break: seeds are dense sim6-style ontogeny
biomorphs (rich morphology), and parsimony pruned them back to sparse diamonds.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from evo_runtime.cache import LRUCache
from evo_runtime.mutation import adaptive_mutation_rate
from evo_runtime.parallel import map_ordered

from nv_evo.temporal import (_obs_len, _output_candidates, TemporalTraces,
                             _score_output_candidate)
from nv_evo.scoring import (contract_relations, needs_samples, score_contract,
                            score_report_lines, LUT_REPORT_NOTES)
from nv_evo.contracts import behavior_contract_lines
from .genome import (LUT_STATES, MAX_CHROMS, MAX_TELOMERE,
                     Genome, Chromosome,
                     random_lut_gene, random_lut_chromosome)
from .lut import grow_lut, grow_lut_tracked
from .pulse import AsyncLutSim


def clone_genome(genome):
    """Fast structural copy (shared, never-in-place-mutated gene objects; fresh
    structure) — an identical-behaviour, ~10x cheaper replacement for
    copy.deepcopy on the reproduction hot path. See nv_evo.ga.clone_genome."""
    clone = Genome(
        chromosomes=[Chromosome(genes=c.genes[:], split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE),
                                wiring=getattr(c, 'wiring', False))
                     for c in genome.chromosomes],
        tag=genome.tag)
    if hasattr(genome, '_io_binding_progress'):
        clone._io_binding_progress = genome._io_binding_progress
    return clone

POPSIZE        = 120
ELITE_FRAC     = 0.10        # elites = this fraction of pop, UNLESS ELITE_COUNT set
ELITE_COUNT    = None        # exact elite count (GUI override); None = use ELITE_FRAC
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
EXPLORATION_PARENT_FRAC = 0.30
MEAN_MUTATIONS = 4.0         # hot-start rate for annealing (see nv_evo.ga)
MUT_DECAY      = 0.997       # slow cooldown: 4.0 -> ~0.89 by gen 500
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))  # see nv_evo.ga
FITNESS_CACHE_MAX = 200_000  # cap the fitness cache on very long runs

# LUT temporal search has the same flat recurrent landscapes as the nervous
# net; reheat after a plateau instead of cooling indefinitely.
# ── running trials / placing outputs (trace-matched, as in nv) ──────────────────
# Output candidates are the OUT_RADIUS cells nearest each terminal, via the
# shared nv_evo.temporal._output_candidates (same radius, same tie-breaks).


def _expand_input_lanes(in_pos):
    """(flat_cells, lane_map) for an in_pos that may carry per-input attachment
    GROUPS (evolvable binding): lane_map[k] is the logical input that drives
    flat_cells[k]. A plain flat in_pos maps 1:1 and is returned as-is, so the
    fixed path is untouched."""
    from nv_evo.io_placement import input_groups
    groups = input_groups(in_pos)
    if all(len(g) == 1 for g in groups):
        return [g[0] for g in groups], list(range(len(groups)))
    flat, lanes = [], []
    for index, group in enumerate(groups):
        for cell in group:
            flat.append(cell)
            lanes.append(index)
    return flat, lanes


def _run_lut_trials(grid, in_pos, ttarget, watch_cells):
    """Run every trial once on the asynchronous engine (lut_evo.pulse).

    Returns (trial_B, trial_events, overflow): the [obs, ncells] mid-tick
    sample matrix per trial (empty when the score mode consumes edges, not
    samples), the continuous leading-edge trains of the ``watch_cells`` per
    trial, and whether any run blew its event budget. A trial that carries a
    physical ``input_events`` schedule is injected at its real (possibly
    sub-tick) times — the LUT backend no longer quantises such targets.

    ``in_pos`` may carry per-input attachment groups (evolvable binding): each
    logical input's stimulus is replicated onto every attachment cell; a cell
    shared by inputs receives the wired-OR (the sim's wire counters merge
    overlapping injections)."""
    obs = _obs_len(ttarget)                  # observe past T to catch delayed events
    need_samples = needs_samples(ttarget)
    flat_pos, lanes = _expand_input_lanes(in_pos)
    expand = lanes != list(range(len(flat_pos)))
    sim = AsyncLutSim(grid, config=getattr(ttarget, 'lut_config', None))
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


def place_outputs_by_trace(grid, in_pos, ttarget):
    """{role: cell}, traces — each role read at the live non-input cell nearest
    its terminal whose trace best matches the expectation across all trials
    (terminal-distance tie-break), exactly like the nervous backend.

    The dynamics are the asynchronous engine (lut_evo.pulse.AsyncLutSim):
    trace/persistence modes score the mid-tick sample matrix (a column slice
    per candidate cell — no per-tick dicts), while event/cadence modes read
    the wires' raw continuous leading-edge timestamps, exactly as the nervous
    backend does."""
    in_set = set(in_pos)
    out_pos = {t.role: None for t in ttarget.outputs}
    traces  = TemporalTraces()
    # LUT holds a level (a latch), it does not ring like the nervous net — so a
    # commanded HOLD must be genuinely high on every tick, not satisfied by a
    # sparse 2-3 tick burst under the nervous net's ±1 ring tolerance. Mark these
    # traces strict (hold_tol=0) so scoring + placement demand the full hold.
    traces.hold_tol = 0
    if len(grid) <= len(in_set):
        return out_pos, traces
    relations = set(contract_relations(ttarget))
    cands_by_role = {term.role: _output_candidates(grid, in_set, term)
                     for term in ttarget.outputs}
    watch = sorted(set().union(*cands_by_role.values()))
    sim, trial_B, trial_events, _, overflow = _run_lut_trials(
        grid, in_pos, ttarget, watch)
    traces.overflow = overflow
    cidx = sim._cidx
    need_samples = needs_samples(ttarget)
    used = set()
    for term in ttarget.outputs:
        best, best_key, best_aux = None, None, None
        score_cache = {}
        for c in cands_by_role[term.role]:
            if c in used:
                continue
            col = cidx[c]
            ctr, cevents, cexp = [], [], []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                ctr.append(trial_B[ti][:, col].tolist() if need_samples else [])
                cevents.append(list(trial_events[ti].get(c, ())))
                cexp.append(exp)
            if not ctr:
                s, aux = 0.0, None
            else:
                signature = (tuple(tuple(seq) for seq in ctr),
                             tuple(tuple(seq) for seq in cevents))
                cached = score_cache.get(signature)
                s, aux = cached if cached is not None else (None, None)
            if ctr and s is None:
                s, aux = _score_output_candidate(
                    ctr, cevents, cexp, term.role, ttarget, traces.overflow,
                    tol=0)                              # LUT: strict hold
            if ctr:
                score_cache[signature] = (s, aux)
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best, best_aux = key, c, aux
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        col = cidx[best]
        traces[term.role]  = [trial_B[ti][:, col].tolist() if need_samples else []
                              for ti in range(len(ttarget.trials))]
        traces.events[term.role] = [list(trial_events[ti].get(best, ()))
                                    for ti in range(len(ttarget.trials))]
    return out_pos, traces


def trace_fixed_outputs(grid, in_pos, out_pos, ttarget):
    """Run LUT trials at pre-selected output cells without validation search."""
    from nv_evo.io_placement import (output_groups, flat_outputs,
                                     merge_intervals)
    groups = output_groups(out_pos)
    if any(pos not in grid for pos in flat_outputs(out_pos)):
        return None
    need_samples = needs_samples(ttarget)
    watch = sorted(set(flat_outputs(out_pos)))
    sim, trial_B, _, trial_intervals, overflow = _run_lut_trials(
        grid, in_pos, ttarget, watch)
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

    Returns ``(grid, out_pos, traces, in_pos)`` — ``in_pos`` is the driven input
    cells (the target's seed pads under the default 'fixed' binding; the
    genome's tag-chosen cells under an evolvable io_placement strategy, see
    nv_evo/io_placement.py) — or None if the net is unusable.

    Under an evolvable strategy the organism nucleates from ONE neutral center
    cell (growth_seeds) instead of the input pads — the body is not anchored to
    the declared I/O positions; the tags wire it afterward."""
    from nv_evo.io_placement import (
        io_strategy, bind_io, growth_seeds, binding_progress,
        record_binding_progress)
    strategy = io_strategy(ttarget)
    grid = grow_lut(genome, seeds=growth_seeds(ttarget, strategy),
                    grid_size=ttarget.grid_size, iters=ttarget.iters)
    node_types = None
    if strategy != 'fixed':
        from .lut import cell_io_tags
        node_types = cell_io_tags(genome, grid)
        if strategy in ('wiring_chromosome', 'spatial_chromosome'):
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


def _place_outputs_combinational(grid, target):
    """{role: cell|None} — where each combinational output is read.

    Prefers the FITTED placement the scorer uses (the cell that best computes
    each role, see _fit_combinational_outputs) so every view — growth tab,
    interactive playback, truth-table report — reads the same cell that decides
    fitness. Falls back to nearest-cell proximity when there is no truth table
    to fit against or the fit cannot fill every role."""
    if getattr(target, 'cases', None) and len(grid) > target.n_inputs:
        try:
            fitted, _ = _fit_combinational_outputs(grid, target)
            if all(fitted.get(t.role) is not None for t in target.outputs):
                return fitted
        except Exception:
            pass
    in_set    = set(target.inputs)
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
# come to rest — it oscillates or goes chaotic ("most circuits produce immediate
# sustained activity ... chaotic behavior", paper §6). Reading a combinational
# answer at one fixed tick therefore samples a single phase of that oscillation
# and rewards phase-luck, not computation. A real combinational result requires
# the output to reach a FIXED POINT: hold the same value over a settling window.
#
# An output that never settles is not simply discarded, though. The array is a
# deterministic finite machine, so a held input drives every output into a
# repeating cycle; the honest read of a cycling output is the DUTY CYCLE over one
# period — the fraction of the period it sits high. A fixed point is the period-1
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
    stretch — at least half the tail and several whole periods — so a chaotic run
    cannot fluke a short period from its last few equal samples (the earlier
    version checked only ``2p`` samples and so declared any tail ending in three
    equal bits 'settled', crediting oscillators as solved). The duty is the mean
    over the periodic suffix's whole periods; a fixed point is ``p == 1`` and
    returns exactly 0.0/1.0. A tail with no such period falls back to its whole
    mean — a chaotic output lands near 0.5 and earns only chance credit, which is
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
# Combinational inputs are presented as PULSES, not a level held from t=0. Each
# trial gives the active inputs a common rising edge after a random delay (so the
# case's onset is one clean coincident edge) and a random per-line hold width,
# and several such trials are averaged — the array must compute the function
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
                     watch_groups=()):
    """Average, over the schedule's pulse trials, each live non-input cell's
    steady duty read during the held window — {cell: mean duty}.

    The active inputs (bits set in ``in_bits``) rise together at each trial's
    delay and hold for their per-line widths; the duty is measured over
    ``[delay + lead, delay + min active width]`` — the full pattern present and
    settled. The all-zero case has no pulses, so it reads a fixed lead..measure
    window of the array's no-input response (the function's value there). Uses
    the vectorised ``run_bits`` matrix so the whole-grid read is not per-tick
    Python.

    ``in_pos`` overrides the driven input cells (the evolvable io_placement
    binding — may carry per-input attachment groups); None keeps the target's
    seed pads — the legacy default."""
    lead, measure = _comb_timing(target.grid_size)
    in_pos  = list(target.inputs) if in_pos is None else list(in_pos)
    flat_pos, lanes = _expand_input_lanes(in_pos)
    in_set  = set(flat_pos)
    active  = [i for i, b in enumerate(in_bits) if b]
    totals  = None
    for delay, widths in schedule:
        # Read the LAST ``measure`` ticks the full pattern is held, so the output
        # has the maximum settling time (>= lead, since widths >= lead+measure)
        # before it is sampled and the input-arrival transient has cleared.
        if active:
            hi = delay + min(widths[i] for i in active)
            lo = max(delay + lead, hi - measure)
        else:
            lo, hi = delay + lead, delay + lead + measure
        # One stream lane per ATTACHMENT CELL: lane k carries logical input
        # lanes[k]'s bit — a shared cell wired-ORs in the sim's injection.
        streams = [tuple((1 if (in_bits[lanes[k]]
                                and delay <= t < delay + widths[lanes[k]])
                          else 0)
                         for k in range(len(flat_pos)))
                   for t in range(hi)]
        sim    = AsyncLutSim(grid)
        levels = sim.run_bits(streams, flat_pos, hi)
        window = levels[lo:hi]
        cells  = sim._cells
        duties = {cells[j]: _steady_duty(window[:, j].tolist())
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


def _balanced_match(duties, expected):
    """Balanced per-level match of one cell's per-case duty to an expected bit
    vector — the single-output form of the contract's logic aggregation, used
    only to RANK candidate output cells (the reported fitness comes from
    score_contract on the chosen cells)."""
    groups = {0: [], 1: []}
    for duty, want in zip(duties, expected):
        groups[1 if want else 0].append(duty if want else 1.0 - duty)
    parts = [sum(v) / len(v) for v in groups.values() if v]
    return sum(parts) / len(parts) if parts else 0.0


def _fit_combinational_outputs(grid, target):
    """(out_pos, duty_by_case) — assign each output role to the live cell whose
    per-case duty best computes that role's truth-table column.

    Output identity is a FITTED parameter, exactly as on the temporal path
    (``place_outputs_by_trace``): rather than reading a cell chosen by mere
    proximity to the terminal, the array is treated as a POOL of candidate
    functions and each role reads the cell that best implements it — its whole
    point is lookup. One cell is fixed per role across every case (never a
    different cell per case), so a full score means a single grown cell really
    computes the function, not a cherry-picked phase. ``duty_by_case[i][cell]``
    is the phase-invariant steady duty, so a settled cell contributes its exact
    bit and a cycling cell the fraction of its period it is correct."""
    schedule = _combinational_schedule(target)   # one shared pulse battery
    duty_by_case = [_all_cell_duties(grid, target, in_bits, schedule)
                    for in_bits, _ in target.cases]
    cells = list(duty_by_case[0]) if duty_by_case else []
    out_pos, used = {}, set()
    for oi, term in enumerate(target.outputs):
        expected = [out_bits[oi] for _, out_bits in target.cases]
        free = [c for c in cells if c not in used]
        if not free:
            out_pos[term.role] = None
            continue
        best = max(free, key=lambda c: (
            _balanced_match([duty_by_case[i][c] for i in range(len(target.cases))],
                            expected),
            -(abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]))))
        used.add(best)
        out_pos[term.role] = best
    return out_pos, duty_by_case


def score_lut_combinational(genome, target):
    """Hold each input pattern, let the array settle, and read the output at the
    fitted cell that best computes each role (see _fit_combinational_outputs).
    A settled cell scores its exact bit; a cycling cell scores the fraction of
    its period it is correct.

    Under an evolvable io_placement strategy the driven input cells AND the read
    output cells come from the genome's tags instead (no duty-fitting search —
    the binding is heritable, not fitted), and the organism nucleates from ONE
    neutral center cell rather than the input pads."""
    from nv_evo.io_placement import (
        io_strategy, bind_io, growth_seeds, binding_progress,
        record_binding_progress)
    strategy = io_strategy(target)
    grid = grow_lut(genome, seeds=growth_seeds(target, strategy),
                    grid_size=target.grid_size, iters=target.iters)
    node_types = None
    if strategy != 'fixed':
        from .lut import cell_io_tags
        node_types = cell_io_tags(genome, grid)
        if strategy in ('wiring_chromosome', 'spatial_chromosome'):
            record_binding_progress(
                genome,
                binding_progress(genome, grid, target, tags=node_types))
    if len(grid) <= target.n_inputs:
        return 0.0
    if len(target.cases) == 0 or len(target.outputs) == 0:
        return 0.0

    if strategy != 'fixed':
        from nv_evo.io_placement import output_groups
        bound = bind_io(genome, grid, target, strategy,
                        tags=node_types)
        if bound is None:
            return 0.0
        in_pos, out_pos = bound
        out_groups = output_groups(out_pos)
        schedule = _combinational_schedule(target)
        duty_by_case = [
            _all_cell_duties(
                grid, target, in_bits, schedule, in_pos=in_pos,
                watch_groups=out_groups.values())
            for in_bits, _ in target.cases]
        # A bound output that is also a driven input has no duty entry (inputs
        # are excluded from the read); such a binding simply scores 0.
        if any(tuple(out_groups[t.role]) not in duty_by_case[0]
               for t in target.outputs):
            return 0.0
        observations = [[duty_by_case[i][tuple(out_groups[t.role])]
                         for t in target.outputs]
                        for i in range(len(target.cases))]
        return score_contract(observations, target)[0]

    out_pos, duty_by_case = _fit_combinational_outputs(grid, target)
    if any(out_pos[t.role] is None for t in target.outputs):
        return 0.0
    observations = [[duty_by_case[i][out_pos[t.role]] for t in target.outputs]
                    for i in range(len(target.cases))]
    return score_contract(observations, target)[0]


def lut_case_outputs(genome, target):
    """Grow, place outputs, and run each combinational case to the horizon.
    Returns (grid, out_pos, cases); each case carries the settled node activity
    (`node_outputs` 0/1 and `node_nibbles` 4-bit N/S/E/W) so the GUI can draw the
    array's response. The LUT analogue of nv_evo.nervous_case_outputs."""
    from nv_evo.io_placement import io_strategy, bind_io, growth_seeds
    strategy = io_strategy(target)
    grid = grow_lut(genome, seeds=growth_seeds(target, strategy),
                    grid_size=target.grid_size, iters=target.iters)
    cases = []
    if len(grid) <= target.n_inputs or not target.cases or not target.outputs:
        return grid, {t.role: None for t in target.outputs}, cases
    # Read at the SAME fitted cells and pulsed trials the scorer uses, so the
    # report reflects the circuit's actual fitness. Node activity for the drawing
    # comes from one representative trial's held window; the reported duty is the
    # trial-averaged value the scorer uses, and a role counts 'settled' only when
    # that averaged duty is an exact 0/1 (a fixed point in every trial).
    if strategy != 'fixed':
        from .lut import cell_io_tags
        from nv_evo.io_placement import output_groups
        bound = bind_io(genome, grid, target, strategy,
                        tags=cell_io_tags(genome, grid))
        if bound is None:
            return grid, {t.role: None for t in target.outputs}, cases
        in_pos, out_pos = bound
        out_groups = output_groups(out_pos)
        schedule = _combinational_schedule(target)
        duty_by_case = [
            _all_cell_duties(
                grid, target, in_bits, schedule, in_pos=in_pos,
                watch_groups=out_groups.values())
            for in_bits, _ in target.cases]
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
        sim = AsyncLutSim(grid)
        sim.run_bits(streams, flat_pos, hi)
        node_out = {c: (1 if sim.out[c] else 0) for c in grid}
        duty = {
            t.role: duty_by_case[ci][
                tuple(out_groups[t.role]) if strategy != 'fixed'
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
    (mirrors nv_evo.nervous_truth_table)."""
    grid, out_pos, cases = lut_case_outputs(genome, target)
    lines = ['Target: %s   [LUT array]' % target.name,
             'Circuit: %d live cells  (square array, four 16-bit LUTs/cell,'
             % len(grid),
             '          asynchronous level logic — paper Architecture 2)']
    from nv_evo.io_placement import output_groups
    groups = output_groups(out_pos)
    for term in target.outputs:
        cells = groups.get(term.role, [])
        lines.append("  out '%s': %s" % (
            term.role,
            ('wired-OR at %s' % ', '.join(map(str, cells)))
            if cells else '(not found)'))
    if not cases:
        lines += ['', '(circuit incomplete — grew too little or outputs missing)']
        return '\n'.join(lines)
    lines += ['', 'Inputs held; each output settles to a fixed point (full',
              "credit) or cycles — then it scores the fraction of its period it",
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


def evaluate_lut_full(genome, target):
    """(scalar fitness, per-case vector) — cases are per (trial, role) traces,
    the units ε-lexicase streams over (None for combinational targets)."""
    if not getattr(target, 'temporal', False):
        return score_lut_combinational(genome, target), None
    from nv_evo.scoring import contract_case_count
    n_cases = contract_case_count(target)
    prep = prepare_lut(genome, target)
    if prep is None:
        return 0.0, (0.0,) * n_cases
    traces = prep[2]
    score, cases, _ = score_contract(traces, target)
    return score, cases


def evaluate_lut(genome, target):
    return evaluate_lut_full(genome, target)[0]


def _evaluate_lut_selection_record(genome, target):
    fitness, cases = evaluate_lut_full(genome, target)
    from nv_evo.io_placement import io_strategy
    total = target.n_inputs + len(target.outputs)
    if io_strategy(target) not in (
            'wiring_chromosome', 'spatial_chromosome'):
        return fitness, cases, (total, total)
    progress = getattr(genome, '_io_binding_progress', (total, total))
    return fitness, cases, progress


def genome_signature(genome):
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0), getattr(c, 'wiring', False),
         tuple((g.ctx_n, g.ctx_e, g.ctx_s, g.ctx_w, g.self_in, g.self_out,
                getattr(g, 'tag', 0), getattr(g, 'io_selector', 0))
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
    from nv_evo.io_placement import record_binding_progress
    for genome, record in zip(genomes, out):
        progress = record[2] if len(record) > 2 else (
            target.n_inputs + len(target.outputs),) * 2
        record_binding_progress(genome, progress)
    return [r[0] for r in out], [r[1] for r in out]


def eval_batch_lut(genomes, target, cache=None):
    return eval_batch_cases(genomes, target, cache)[0]


# ── genetic operators ────────────────────────────────────────────────────────────

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
    fields = tuple(_GENE_FIELDS) + ('tag',) if fields is None else fields
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
    return tuple(
        (getattr(chromosome, 'wiring', False),
         tuple((*(getattr(gene, field) for field in _GENE_FIELDS),
                getattr(gene, 'tag', 0), getattr(gene, 'io_selector', 0))
               for gene in chromosome.genes))
        for chromosome in genome.chromosomes)


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


def _force_nonparent_tweak(genome, parent):
    """End a multi-edit transaction at an allele distinct from its parent."""
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes and not getattr(chromosome, 'wiring', False)]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(random_lut_chromosome())
        else:
            random.choice(genome.chromosomes).genes.append(random_lut_gene())
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
    setattr(gene, field, _soft_lut_excluding(
        getattr(gene, field), parent_value))
    genome.chromosomes[ci].genes[gi] = gene


def _tweak_gene(gene):
    """Soft mutation: usually flip 1-3 bits of one 16-bit field (a nearby
    boolean function), sometimes replace the field entirely. self_in has an
    extra chance of snapping to 0 — growth rules must stay reachable."""
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and g.self_in != 0 and random.random() < 0.2:
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
    from nv_evo.io_placement import mutate_io_allele
    mutate_io_allele(genome, LUT_STATES, strategy=io_placement)


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
# I/O alleles mutate separately when an evolvable placement strategy is active.
# (evolve_io) — off by default, so the ordinary mutation stream is unchanged.
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]
IO_MUTATION_PROB = 0.20


def _mutate_once_lut(genome, max_telomere=MAX_TELOMERE,
                     chromosome_count=None, evolve_io=False,
                     io_placement=None):
    """Apply one feasible, state-changing mutation to a LUT genome.

    ``evolve_io`` adds the I/O-tag and wiring-chromosome mutations that let an
    evolvable io_placement strategy re-wire its ports; off by default so the
    ordinary mutation stream is byte-identical."""
    if not genome.chromosomes:
        genome.chromosomes.append(random_lut_chromosome())
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
        chromosome.genes[index] = _tweak_gene(chromosome.genes[index])
    elif op == 'duplicate':
        chromosome = random.choice([
            item for item in body
            if item.genes and len(item.genes) < _gene_cap()
        ])
        chromosome.genes.insert(
            random.randrange(len(chromosome.genes) + 1),
            _tweak_gene(random.choice(chromosome.genes)))
    elif op == 'add_gene':
        chromosome = random.choice([
            item for item in body if len(item.genes) < _gene_cap()
        ])
        chromosome.genes.append(random_lut_gene())
    elif op == 'del_gene':
        chromosome = random.choice([
            item for item in body if len(item.genes) > 1
        ])
        chromosome.genes.pop(random.randrange(len(chromosome.genes)))
    elif op == 'add_chrom':
        chroms.append(random_lut_chromosome())
    elif op == 'del_chrom':
        chroms.remove(min(body, key=lambda item: len(item.genes)))
    elif op == 'split':
        chromosome, values = random.choice(split_options)
        chromosome.split = random.choice(values)
    else:  # telomere
        chromosome, values = random.choice(telomere_options)
        chromosome.telomere = random.choice(values)


def mutate_lut(genome, mean_mutations=None, max_telomere=MAX_TELOMERE,
               chromosome_count=None, evolve_io=False, io_placement=None):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    g = clone_genome(genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    events = max(1, _poisson(lam))
    for _ in range(events - 1):
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=io_placement)
    if events == 1:
        _mutate_once_lut(
            g, max_telomere=max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=io_placement)
    else:
        _force_nonparent_tweak(g, genome)
    if evolve_io and random.random() < IO_MUTATION_PROB:
        _mutate_io_tag(g, io_placement=io_placement)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    return g


def crossover_lut(pa, pb):
    """Tag-matched hierarchical crossover, including one-rule chromosomes."""
    ca, cb = clone_genome(pa), clone_genome(pb)
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


# ── neutral compaction (gene expression) ─────────────────────────────────────────
# A dense ontogeny genome carries genes that win NO growth lookup — they are
# unexpressed, and (proven) removing a never-winning gene cannot change any argmin,
# so the grown organism is bit-identical and the fitness is unchanged. Compaction
# drops exactly those: the threshold is zero expression (a neutrality boundary, not
# a tuned cutoff), leaving the genome's functional core with behaviour untouched.
# Biologically: pseudogene loss. NB measured only a modest shrink on fresh ontogeny
# seeds (~7-14% unexpressed — the density is mostly FUNCTIONAL morphology, not junk),
# so this is a human-facing cleanup for inspecting/refining an evolved genome
# (the Designer), NOT an evolution driver — it did not improve solving in tests.
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
    return Genome(chromosomes=new_chroms, tag=genome.tag)


# ── ontogeny seeding is THE seed path (see lut_evo/ontogeny.py) ──────────────────
# Sparse random genomes grow near-uniform DIAMONDS — the "uninteresting" LUTs.
# Richness tracks gene density (sim6's table_create invents a gene per unseen
# context), so the population/immigrants are seeded with sim6-style biomorph
# genomes, and the smaller-genome parsimony tie-break is dropped: it pruned the
# dense genomes straight back to sparse (re-diamonding them). This is a user
# decision (2026-07-07): rich morphology over solve speed, no random/ontogeny
# switch. Size is bounded by ONTOGENY_CAP + telomeres instead of parsimony.
ONTOGENY_CAP  = 350               # max genes kept from an ontogeny biomorph


# Growing one sim6 ontogeny seed is expensive (measured ~0.3 s: up to 60
# biomorph growths per accepted genome), and make_seed_genome is called for
# EVERY immigrant EVERY generation — that alone was a large share of ontogeny
# mode's ~50x slowdown. So the factory grows only the first _ONTO_POOL_SIZE
# genomes per chromosome count and caches them; later calls return a mutated
# variant of a random pool member (still fresh genetic material, ~free).
_ONTO_POOL      = {}     # n_chroms -> [cached seed genomes]
_ONTO_POOL_SIZE = 24


def make_seed_genome(n_chroms=2):
    """The population/immigrant factory: a dense sim6-style biomorph genome
    (the rich-morphology seeds — see the ONTOGENY_CAP note above)."""
    from .ontogeny import random_ontogeny_genome
    pool = _ONTO_POOL.setdefault(n_chroms, [])
    if len(pool) < _ONTO_POOL_SIZE:
        g = random_ontogeny_genome(n_chroms, cap_genes=ONTOGENY_CAP)
        pool.append(g)
        return clone_genome(g)     # the pool master stays pristine
    return mutate_lut(random.choice(pool), chromosome_count=n_chroms)


def _tiebreak(genome):
    """Tie-break term for equal fitness: NEUTRAL — the smaller-genome parsimony
    pressure was what pruned dense ontogeny genomes back to sparse diamonds
    (re-diamonding). Size is bounded by ONTOGENY_CAP + telomeres instead."""
    return 0


def rank_key(genome, fitness):
    """Selection-only wiring viability, then honest behavior, then tie-break."""
    from nv_evo.io_placement import binding_viability
    return (binding_viability(genome), fitness, _tiebreak(genome))


def _gene_cap():
    """Per-chromosome gene cap for add/duplicate mutations — sized for dense
    ontogeny genomes (MAX_GENES was the sparse random path's tight cap)."""
    return ONTOGENY_CAP


def tournament_lut(population, fitnesses):
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(
        idx, key=lambda i: rank_key(population[i], fitnesses[i]))]


def select_parent(population, fitnesses, case_vecs=None):
    import nv_evo.ga as _nga
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


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None, ga_config=None,
                    chromosome_count=None, recombination=True,
                    evolve_io=False, io_placement=None):
    """One generation of a steady, exploratory GA — elitism + immigrants +
    recombination/mutation, parents via ε-lexicase when per-case vectors are
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
    if ga_config is not None and ga_config.chromosome_count is not None:
        chromosome_count = ga_config.chromosome_count
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    import nv_evo.ga as _nga
    selection = (_nga.SELECTION if ga_config is None else ga_config.selection)
    strategy = (io_placement or (
        getattr(ga_config, 'io_placement', 'fixed')
        if ga_config is not None else 'fixed'))
    evolve_io = evolve_io or strategy != 'fixed'
    if make_genome is None:
        if evolve_io:
            from nv_evo.io_placement import seed_io_metadata
            inferred_ports = 0
            if strategy in (
                    'wiring_chromosome', 'spatial_chromosome') and population:
                from nv_evo.io_placement import wiring_chromosome
                mapping = wiring_chromosome(population[0])
                inferred_ports = len(mapping.genes) if mapping is not None else 0
            make_genome = lambda: seed_io_metadata(
                make_seed_genome(chromosome_count or 2),
                wiring_chromosome=(strategy == 'wiring_chromosome'),
                spatial_chromosome=(strategy == 'spatial_chromosome'),
                n_ports=inferred_ports, tag_rank=(strategy == 'tag_rank'))
        else:
            make_genome = lambda: make_seed_genome(chromosome_count or 2)
    n_elite = elite_count if elite_count is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = min(int(round(pop * immigrant_fraction)), pop)
    order   = sorted(
        range(pop),
        key=lambda i: rank_key(population[i], fitnesses[i]),
        reverse=True)
    # Elites choose parents here but are never copied into the next generation.
    new_pop = [make_genome() for _ in range(n_imm)]                  # immigrants
    use_lexicase = (selection == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    residual = order[n_elite:] if n_elite < pop else order
    elite_pool = order[:n_elite] if n_elite else order
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]

    def pick_index(candidates):
        if use_lexicase:
            local_population = [population[index] for index in candidates]
            local_cases = [case_vecs[index] for index in candidates]
            chosen = _nga._lexicase_parent(local_population, local_cases)
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
        second = choose(first)
        if recombination_signatures[second] == recombination_signatures[first]:
            distinct = [
                index for index in range(pop)
                if index != first
                and recombination_signatures[index]
                != recombination_signatures[first]
            ]
            if distinct:
                second = pick_index(distinct)
        return population[first], population[second]

    while len(new_pop) < pop:
        pa, pb = parent_pair()
        ca, cb = (crossover_lut(pa, pb) if recombination else
                  (clone_genome(pa), clone_genome(pb)))
        max_telomere = (MAX_TELOMERE if ga_config is None
                        else ga_config.max_telomere)
        new_pop.append(mutate_lut(
            ca, mean_mutations, max_telomere,
            chromosome_count=chromosome_count, evolve_io=evolve_io,
            io_placement=strategy))
        if len(new_pop) < pop:
            new_pop.append(mutate_lut(
                cb, mean_mutations, max_telomere,
                chromosome_count=chromosome_count, evolve_io=evolve_io,
                io_placement=strategy))
    if (chromosome_count is not None
            and any(len(genome.chromosomes) != chromosome_count
                    for genome in new_pop)):
        raise ValueError('genome factory violated configured chromosome count')
    return new_pop[:pop]


def evolve_lut(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True,
               seed=None):
    if seed is not None:
        random.seed(seed)
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    from nv_evo.io_placement import (
        growth_seeds, io_strategy, seed_io_metadata,
        seed_spatial_from_phenotype, seed_wiring_from_phenotype,
        uses_port_chromosome)
    strategy = io_strategy(target)
    if uses_port_chromosome(strategy) and n_chroms < 3:
        raise ValueError('chromosome-based I/O requires n_chroms >= 3')
    evolve_io = strategy != 'fixed'
    n_ports = target.n_inputs + len(target.outputs)
    if evolve_io:
        def make_genome():
            genome = seed_io_metadata(
                make_seed_genome(n_chroms),
                wiring_chromosome=(strategy == 'wiring_chromosome'),
                spatial_chromosome=(strategy == 'spatial_chromosome'),
                n_ports=n_ports, tag_rank=(strategy == 'tag_rank'))
            if uses_port_chromosome(strategy):
                from .lut import grow_lut, cell_io_tags
                grid = grow_lut(
                    genome, seeds=growth_seeds(target),
                    grid_size=target.grid_size, iters=target.iters)
                tags = cell_io_tags(genome, grid)
                if strategy == 'spatial_chromosome':
                    seed_spatial_from_phenotype(
                        genome, grid, target, tags=tags)
                else:
                    seed_wiring_from_phenotype(
                        genome, grid, target, tags=tags)
            return genome
    else:
        make_genome = lambda: make_seed_genome(n_chroms)  # ontogeny biomorph seeds
    cache       = LRUCache(FITNESS_CACHE_MAX)
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:
        population  = [make_genome() for _ in range(pop)]
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        bi = max(
            range(pop),
            key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        best_rank = rank_key(best_genome, best_fitness)
        stagnation   = 0
        mut_rate     = MEAN_MUTATIONS        # annealing schedule (see MUT_DECAY)
        for gen in range(generations):
            mut_rate *= MUT_DECAY
            mm = adaptive_mutation_rate(mut_rate, stagnation,
                                        solved=best_fitness >= 1.0)
            parents, parent_fitnesses, parent_cases = population, fitnesses, cases
            offspring = next_population(
                parents, parent_fitnesses, make_genome, parent_cases, mm,
                chromosome_count=n_chroms, evolve_io=evolve_io,
                io_placement=strategy)
            offspring_fitnesses, offspring_cases = eval_batch_cases(
                offspring, target, cache, ex)
            if max(best_fitness, max(offspring_fitnesses)) >= 1.0:
                # Once solved, accumulate perfect circuits so the mean can
                # converge; pre-solve evolution remains offspring-only.
                population, fitnesses, cases = consolidate_population(
                    parents, parent_fitnesses, parent_cases,
                    offspring, offspring_fitnesses, offspring_cases)
            else:
                population, fitnesses, cases = (
                    offspring, offspring_fitnesses, offspring_cases)
            gi = max(
                range(pop),
                key=lambda i: rank_key(population[i], fitnesses[i]))
            if fitnesses[gi] > best_fitness + 1e-12:
                stagnation = 0
            else:
                stagnation += 1
            generation_rank = rank_key(population[gi], fitnesses[gi])
            if generation_rank > best_rank:
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
                best_rank = generation_rank
            if verbose and gen % 10 == 0:
                print("%5d  %6.4f  %6.4f" % (gen, best_fitness,
                                             sum(fitnesses) / pop))
        return best_genome, best_fitness
    finally:
        ex.shutdown()


def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None, executor=None, should_stop=None, on_progress=None,
              max_telomere=MAX_TELOMERE, chromosome_count=None,
              evolve_io=False, io_placement=None):
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
                           io_placement=io_placement)
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


# ── GUI report ───────────────────────────────────────────────────────────────────

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
        lines += ['', '(circuit incomplete — grew too little or inputs dead)']
    traces = prep[2] if prep is not None else None
    out_pos = prep[1] if prep is not None else None
    _, body = score_report_lines(ttarget, traces, out_pos,
                                 notes=LUT_REPORT_NOTES)
    lines += body
    if traces is None and genome is None:
        lines += ['', '(run the GA or Load Saved to inspect a circuit)']
    return '\n'.join(lines)
