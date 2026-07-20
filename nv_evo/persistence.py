"""
nv_evo/persistence.py — retention (persistent-memory) target + fitness.

The ordinary toggle oracle only observes ~T=24 ticks, so a circuit whose "hold"
is a finite ring burst that decays after ~10 units scores as well as a genuine
memory (finite-hold discovery, see results/async_probe_toggle_v1). To EVOLVE
real memory we must demand RETENTION: a set bit must stay set across a long gap,
and a cleared bit must stay clear — verified at multiple horizons.

This target is scored on the event-native path (nv_evo.simulation), so a long
horizon costs O(events), not O(T x cells) tick sampling. Fitness is staged:
short schedules gate the expensive long-horizon sweeps.

Claim discipline: finite simulation cannot prove infinite persistence. The
defensible claim is BOUNDED RETENTION through a predeclared, demanding duration
sweep, supported by a sustained feedback loop — not mathematical infinity.
"""
from __future__ import annotations
import random

from . import pulse
from . import simulation as ae
from .targets import TemporalTarget, Trial, OutputTerminal
from .nervous import grow_nervous, interpret_nervous, node_widths, node_delays
from .temporal import _output_candidates
from .scoring import (parity_intervals, score_retention_graded,
                      score_retention, score_interval_graded,
                      score_reset_influence)

RETENTION_HORIZONS = (32, 64, 128, 256)     # x DELAY (predeclared duration sweep)
INPUT = (0, 2)
LATENCY = 2


def _stream(edges, horizon):
    """Length-`horizon` integer command stream with a 1 at each toggle tick."""
    s = [(0,) for _ in range(horizon)]
    for e in edges:
        if 0 <= e < horizon:
            s[e] = (1,)
    return s


def retention_oracle(seed=20260711):
    """Trials that demand bounded retention: hold-1 across each horizon, a quiet
    (never-set) trial, and toggle-back / late-second-toggle trials that prove the
    loop can still CLEAR after holding. Random phases/gaps make held-out variants
    when re-seeded. Each trial's stream length IS its horizon."""
    rng = random.Random(seed)
    trials, meta = [], []
    Hmax = max(RETENTION_HORIZONS)
    # a short-parity LADDER first: gives a gradient random -> parity -> short hold
    # before the demanding long-horizon holds (climbing, not a cliff).
    for H in (16, 24):
        t0 = 5 + rng.randint(0, 2)
        trials.append(Trial(_stream([t0], H), {'Q': [0] * H}))
        meta.append(('hold', H))
        t0 = 4 + rng.randint(0, 2)
        g = 6 + rng.randint(0, 3)
        trials.append(Trial(_stream([t0, t0 + g], H + t0 + g), {'Q': [0] * (H + t0 + g)}))
        meta.append(('clear', H + t0 + g))
    for H in RETENTION_HORIZONS:                       # hold a set bit to horizon H
        t0 = 5 + rng.randint(0, 3)
        trials.append(Trial(_stream([t0], H), {'Q': [0] * H}))
        meta.append(('hold', H))
    trials.append(Trial(_stream([], Hmax), {'Q': [0] * Hmax}))   # quiet: stays clear
    meta.append(('quiet', Hmax))
    for G in (32, 64, 128):                            # toggle-back: hold G then clear
        t0 = 5 + rng.randint(0, 3)
        H = t0 + 2 * G + 8
        trials.append(Trial(_stream([t0, t0 + G], H), {'Q': [0] * H}))
        meta.append(('clear', H))
    t0 = 5 + rng.randint(0, 3)                          # late second toggle
    H = t0 + 200
    trials.append(Trial(_stream([t0, t0 + 150], H), {'Q': [0] * H}))
    meta.append(('late-clear', H))
    tgt = TemporalTarget('Retention (oracle)', [INPUT], [OutputTerminal('Q', (2, 2))],
                         Hmax, trials, grid_size=7, iters=30, score_mode='retention',
                         latency=LATENCY,
                         description='Bounded persistent memory: a set bit must be '
                                     'retained across long gaps and cleared on the '
                                     'next command, verified to %d x DELAY.' % Hmax)
    tgt._retention_meta = meta
    return tgt


def _event_cap(n_cells, horizon, config=None):
    """Event cap PROPORTIONAL to horizon so a legitimate sustained ring is not
    misclassified as runaway; a dense whole-grid oscillation still overflows."""
    delay = pulse.DELAY if config is None else config.delay
    width = pulse.WIDTH if config is None else config.width
    return int(3 * n_cells * horizon / (delay + width)) + 256


_OFFSETS = (0, 1, 2, 3)          # small residual-latency search on top of target.latency


def evaluate_retention(genome, target):
    """(worst-trial retention fitness, per-trial case vector). Grows once, runs
    each trial on the event-native path shortest-horizon first, and short-circuits
    a hopeless genome before the long sweeps. The readout cell and a single global
    offset are fitted on these training trials (frozen only for held-out scoring)."""
    n = len(target.trials)
    zero = (0.0,) * n
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0, zero
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, _ = interpret_nervous(grid, target, arch=arch)
    if any(p not in grid for p in in_pos):
        return 0.0, zero
    cands = _output_candidates(grid, set(in_pos), target.outputs[0])
    if not cands:
        return 0.0, zero
    config = getattr(target, 'pulse_config', None)
    D = pulse.DELAY if config is None else config.delay
    W = pulse.WIDTH if config is None else config.width
    widths = None if arch == 'tri3' else node_widths(genome, grid, config)
    delays = None if arch == 'tri3' else node_delays(genome, grid, config)
    offsets = [target.latency + k for k in _OFFSETS]

    order = sorted(range(n), key=lambda i: len(target.trials[i].streams))
    # full per-(cell, offset) trial vector, so the reported cases are CONSISTENT
    # with the single (cell, offset) we ultimately select (no per-trial cherry-pick)
    scmat = {(c, o): [0.0] * n for c in cands for o in offsets}
    for i in order:
        tr = target.trials[i]
        H = len(tr.streams)
        sched = ae.streams_to_schedule(tr.streams, 1, H, config=config)
        edges = ae.effective_edges(sched)[0]
        intervals = parity_intervals(edges, H)
        run_h = H + max(offsets) + 4 * (D + W)
        res, overflow = ae.run_schedule(grid, routing, in_pos, sched, run_h, cands,
                                        max_events=_event_cap(len(grid), H, config),
                                        config=config, widths=widths,
                                        delays=delays, arch=arch)
        for c in cands:
            for o in offsets:
                s = 0.0 if overflow else score_retention_graded(res[c], intervals, o)
                scmat[(c, o)][i] = 0.0 if s is None else s

    # pick the readout (cell, offset) lexicographically: best worst-trial, then
    # best overall profile — so among the (common, early) all-worst-0 ties we keep
    # the genuine memory cell, giving lexicase a real per-case gradient to climb.
    best_key = max(scmat, key=lambda k: (min(scmat[k]), sum(scmat[k])))
    return min(scmat[best_key]), tuple(scmat[best_key])


# â”€â”€ persistent SR-latch curriculum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SR_INPUTS = ((0, 1), (0, 3))


def _sr_stream(set_edges, reset_edges, horizon):
    sets, resets = set(set_edges), set(reset_edges)
    return [(1 if t in sets else 0, 1 if t in resets else 0)
            for t in range(horizon)]


def _sr_target(phase, seed=20260711, strict=False):
    """Build one behavioral curriculum phase for persistent SR memory.

    Runs may contribute several *separate* cases. Lexicase therefore sees Set
    activation, long retention, Reset suppression, cleared quiet, and re-Set
    recovery independently instead of hiding a missing behavior in an average.
    """
    if phase not in ('set_hold', 'reset', 'full'):
        raise ValueError('unknown SR curriculum phase: %s' % phase)
    rng, trials, runs = random.Random(seed), [], []

    def add(name, set_edges, reset_edges, horizon, cases):
        trials.append(Trial(
            _sr_stream(set_edges, reset_edges, horizon), {'Q': [0] * horizon}))
        runs.append({'name': name, 'cases': tuple(cases)})

    quiet_h = 64 if phase != 'full' else 128
    add('quiet', [], [], quiet_h,
        [('quiet', 'state', 0, 0.0, float(quiet_h))])

    holds = ((16, 32, 64, 128) if phase == 'set_hold'
             else ((32, 64, 128) if phase == 'reset'
                   else (32, 64, 128, 256)))
    for duration in holds:
        start = 5 + rng.randint(0, 3)
        horizon = start + duration
        add('hold-%d' % duration, [start], [], horizon,
            [('hold-%d' % duration, 'state', 1,
              float(start), float(horizon))])

    if phase in ('reset', 'full'):
        # Paired odd/even gaps prevent a phase-locked Reset from looking like an
        # asynchronous clear. A small even seed-dependent displacement changes
        # absolute gaps without destroying the phase contrast.
        bases = ((23, 24, 47, 48, 95, 96) if phase == 'reset'
                 else (31, 32, 63, 64, 127, 128))
        gaps = tuple(base + 2 * rng.randint(-1, 1) for base in bases)
        for gap in gaps:
            start = 5 + rng.randint(0, 3)
            reset = start + gap
            horizon = reset + max(32, gap // 2)
            cases = [
                ('pre-reset-hold-%d' % gap, 'state', 1,
                 float(start), float(reset)),
                ('clear-%d' % gap, 'state', 0,
                 float(reset), float(horizon)),
            ]
            if phase == 'reset':
                cases.insert(1, ('reset-influence-%d' % gap,
                                 'reset_influence', float(reset)))
            add('set-reset-%d' % gap, [start], [reset], horizon, cases)

    if phase == 'full':
        # Reset-only must not create activity.
        reset = 7 + rng.randint(0, 2)
        add('reset-only', [], [reset], 80,
            [('reset-only-quiet', 'state', 0, 0.0, 80.0)])
        # Set -> long hold -> Reset -> quiet -> Set again. This is the complete
        # conjunction, and every epoch remains a separate selection case.
        reload_gaps = tuple(base + 2 * rng.randint(-1, 1)
                            for base in (47, 48, 95, 96))
        for gap in reload_gaps:
            start = 5 + rng.randint(0, 3)
            reset = start + gap
            # Re-Set is also tried at both ring phases across the paired cases.
            restart = reset + (31 if gap % 2 else 32)
            horizon = restart + 64
            add('reload-%d' % gap, [start, restart], [reset], horizon, [
                ('reload-first-hold-%d' % gap, 'state', 1,
                 float(start), float(reset)),
                ('reload-clear-%d' % gap, 'state', 0,
                 float(reset), float(restart)),
                ('reload-second-hold-%d' % gap, 'state', 1,
                 float(restart), float(horizon)),
            ])

    target = TemporalTarget(
        'Persistent SR latch (%s)' % phase, list(SR_INPUTS),
        [OutputTerminal('Q', (2, 2))], max(len(t.streams) for t in trials),
        trials, grid_size=7, iters=30, score_mode='sr_retention',
        latency=LATENCY,
        description=('Behavioral curriculum phase %s for bounded asynchronous '
                     'SR memory; final retention is tested through 256 x DELAY.'
                     % phase))
    target._sr_phase = phase
    target._sr_runs = tuple(runs)
    target._sr_strict = bool(strict)
    return target


def sr_set_hold_oracle(seed=20260711, strict=False):
    return _sr_target('set_hold', seed, strict)


def sr_reset_oracle(seed=20260711, strict=False):
    return _sr_target('reset', seed, strict)


def sr_full_oracle(seed=20260711, strict=False):
    return _sr_target('full', seed, strict)


def _sr_case_score(rise, case, offset, strict):
    kind = case[1]
    if kind == 'state':
        _, _, state, start, end = case
        if strict:
            score = score_retention(rise, [(state, start, end)], offset)
            return 0.0 if score is None else score
        return score_interval_graded(rise, state, start, end, offset)
    if kind == 'reset_influence':
        return score_reset_influence(rise, case[2], offset)
    raise ValueError('unknown SR retention case: %s' % (kind,))


def _evaluate_sr_details(genome, target, fitted=None):
    """Return ``(score, cases, cell, residual_alignment)`` for one SR target."""
    n_cases = sum(len(run['cases']) for run in target._sr_runs)
    zero = (0.0,) * n_cases
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0, zero, None, 0.0
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, _ = interpret_nervous(grid, target, arch=arch)
    if any(pos not in grid for pos in in_pos):
        return 0.0, zero, None, 0.0

    if fitted is None:
        candidates = _output_candidates(
            grid, set(in_pos), target.outputs[0])
        offsets = [target.latency + k for k in _OFFSETS]
    else:
        cell = fitted.output_positions.get('Q')
        if cell not in grid:
            return 0.0, zero, None, 0.0
        candidates = [cell]
        offsets = [target.latency + float(fitted.alignment or 0.0)]
    if not candidates:
        return 0.0, zero, None, 0.0

    profiles = {(cell, offset): [] for cell in candidates for offset in offsets}
    config = getattr(target, 'pulse_config', None)
    D = pulse.DELAY if config is None else config.delay
    W = pulse.WIDTH if config is None else config.width
    widths = None if arch == 'tri3' else node_widths(genome, grid, config)
    delays = None if arch == 'tri3' else node_delays(genome, grid, config)
    for trial, run in zip(target.trials, target._sr_runs):
        horizon = len(trial.streams)
        schedule = ae.streams_to_schedule(
            trial.streams, 2, horizon, config=config)
        run_horizon = horizon + max(offsets) + 4 * (D + W)
        traces, overflow = ae.run_schedule(
            grid, routing, in_pos, schedule, run_horizon, candidates,
            max_events=_event_cap(len(grid), horizon, config),
            config=config, widths=widths, delays=delays, arch=arch)
        for cell in candidates:
            for offset in offsets:
                profiles[(cell, offset)].extend(
                    0.0 if overflow else _sr_case_score(
                        traces[cell], case, offset,
                        bool(getattr(target, '_sr_strict', False)))
                    for case in run['cases'])

    best = max(profiles, key=lambda key: (
        min(profiles[key]), sum(profiles[key]), profiles[key]))
    cases = tuple(profiles[best])
    cell, total_offset = best
    return min(cases), cases, cell, total_offset - target.latency


def evaluate_sr_retention(genome, target):
    score, cases, _, _ = _evaluate_sr_details(genome, target)
    return score, cases


def fit_sr_readout(genome, target):
    from .evaluation import FittedReadout
    score, _, cell, alignment = _evaluate_sr_details(genome, target)
    if cell is None:
        return None
    return FittedReadout('nervous', (('Q', cell),), alignment, score)


def score_sr_frozen(genome, target, fitted):
    score, _, _, _ = _evaluate_sr_details(genome, target, fitted=fitted)
    return score


def sr_case_names(target):
    return tuple(case[0] for run in target._sr_runs for case in run['cases'])


def evolve_sr_curriculum(generations=(30, 40, 60), pop=80, n_chroms=2,
                         seed=20260711, verbose=True):
    """Evolve SR memory in behavioral stages without a designed warm start.

    The naturally evolved population (plus its phase champion) becomes the next
    phase's starting population. Selection is explicitly lexicase over the
    separate behavioral cases; the scalar minimum is used only for champion
    reporting and the final bounded-retention claim.
    """
    from .ga import evolve_nervous
    builders = (sr_set_hold_oracle, sr_reset_oracle, sr_full_oracle)
    if len(generations) != len(builders):
        raise ValueError('generations must contain set-hold, reset, and full budgets')
    population, summaries, champion = None, [], None
    for index, (builder, budget) in enumerate(zip(builders, generations)):
        target = builder(seed=seed + index)
        result = evolve_nervous(
            target, generations=int(budget), pop=pop, n_chroms=n_chroms,
            verbose=verbose, seed=seed + index,
            seed_genomes=population, selection='lexicase',
            return_population=True)
        champion, fitness, population, fitnesses, case_vectors = result
        # Keep the all-time phase champion in the naturally evolved lineage even
        # though the exploratory GA intentionally has no survival elitism.
        population = [champion] + list(population[:max(0, pop - 1)])
        names = sr_case_names(target)
        maxima = [max(vector[index] for vector in case_vectors)
                  for index in range(len(names))]
        solved_per_genome = [sum(value >= 0.999 for value in vector)
                             for vector in case_vectors]
        summaries.append({
            'phase': target._sr_phase,
            'generations': int(budget),
            'best_training_min': fitness,
            'population_mean_min': sum(fitnesses) / len(fitnesses),
            'case_maxima': tuple(zip(names, maxima)),
            'cases_solved_somewhere': sum(value >= 0.999 for value in maxima),
            'case_count': len(names),
            'max_cases_solved_by_one_genome': max(solved_per_genome),
        })

    strict_train = sr_full_oracle(seed=seed + 2, strict=True)
    fitted = fit_sr_readout(champion, strict_train)
    strict_train_score = 0.0 if fitted is None else fitted.training_score
    strict_holdout = sr_full_oracle(seed=seed + 1002, strict=True)
    strict_holdout_score = (0.0 if fitted is None else
                            score_sr_frozen(champion, strict_holdout, fitted))
    return champion, {
        'seed': seed,
        'population': pop,
        'generations': tuple(int(value) for value in generations),
        'phases': tuple(summaries),
        'strict_training_min': strict_train_score,
        'strict_frozen_holdout_min': strict_holdout_score,
        'fitted': fitted,
    }
