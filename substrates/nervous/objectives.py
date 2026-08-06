"""
substrates/nervous/objectives.py - the two EXTRA objectives an escape run adds.

Both are opt-in (runtime/escape.EscapeConfig) and both are shared by the
nervous and LUT substrates, which is why they live here alongside the shared
scoring contract rather than being duplicated in each ga.py.

LIFESPAN SCORING scores the organism at several points along its DEVELOPMENT,
not only as a finished adult. The motivation is that a temporal plateau is
usually flat rather than rugged: a genome whose stage-6 body half-works but
whose stage-12 body is broken scores zero, dies, and leaves no gradient behind.
Its juvenile scores give it somewhere to climb from. Two rules keep this
honest:

  * the REPORTED fitness is always the adult score. A run that reads 1.0 still
    means the fully grown circuit does the job - juvenile credit never inflates
    it;
  * juvenile scores enter selection only, as extra epsilon-lexicase cases and as a
    rank_key tier BELOW behavioral fitness.

ROBUSTNESS re-scores the adult under jittered physics. It is a second
objective, never a substitute for the first: it is aggregated by WORST case
(so "perfect on three trials, dead on the fourth" cannot look good) and ranked
strictly below nominal fitness (so a robust wrong circuit can never outrank a
correct one). Its purpose is landscape shaping - brittle solutions sit on
narrow spikes, and preferring the broader basin among EQUALLY CORRECT circuits
both makes the optimum easier to find and generalises better.
"""
from __future__ import annotations

import copy

from .scoring import contract_case_count


def total_case_count(target):
    """Length of the case vector a genome must produce for this target.

    The contract's own cases, plus one per lifespan checkpoint when lifespan
    scoring is on. epsilon-lexicase requires every population member to present the
    same number of cases, so this is the single source of truth for that
    length - including for the all-zero vector a dead genome returns.
    """
    escape = getattr(target, '_escape', None)
    base = contract_case_count(target)
    if escape is not None and escape.lifespan_scoring:
        return base + escape.lifespan_checkpoints
    return base


def checkpoint_indices(n_snapshots, count):
    """Interior developmental stages to score, oldest first.

    Index 0 is the bare seed pads (nothing has grown yet) and the last index is
    the adult, which is scored anyway - so only the interior is useful. Fewer
    than ``count`` distinct stages come back for an organism that matures
    quickly; the caller pads.
    """
    if n_snapshots <= 2 or count < 1:
        return ()
    picks = []
    for i in range(count):
        index = int(round((i + 1) * (n_snapshots - 1) / float(count + 1)))
        index = max(1, min(n_snapshots - 2, index))
        if index not in picks:
            picks.append(index)
    return tuple(picks)


def grown_snapshots(genome, target, backend, strategy):
    """Every developmental stage of one organism, adult last.

    The final snapshot is bit-identical to what ``grow_nervous`` / ``grow_lut``
    would have returned, so a lifespan run's adult body - and therefore its
    reported fitness - is exactly the body an ordinary run would have grown.
    """
    if backend == 'lut':
        from substrates.lut.genome import lut_growth_seeds
        from substrates.lut.lut import grow_lut_snapshots
        seeds = lut_growth_seeds(genome, target, strategy)
        return grow_lut_snapshots(genome, seeds=seeds,
                                  grid_size=target.grid_size,
                                  iters=target.iters)
    from .io_placement import growth_seeds
    seeds = growth_seeds(target, strategy, genome)
    from .nervous import grow_nervous_snapshots
    return grow_nervous_snapshots(genome, seeds=seeds,
                                  grid_size=target.grid_size,
                                  iters=target.iters)


def prepare_grid(genome, target, backend, grid, strategy,
                 record_progress=True):
    """Interpret one already-grown body; returns the backend's prep tuple."""
    if backend == 'lut':
        from substrates.lut.ga import prepare_lut_grid
        return prepare_lut_grid(genome, target, grid, strategy=strategy,
                                record_progress=record_progress)
    from .temporal import prepare_net_grid
    return prepare_net_grid(genome, target, grid, strategy=strategy,
                            record_progress=record_progress)


def _traces_of(prep, backend):
    if prep is None:
        return None
    return prep[2] if backend == 'lut' else prep[4]


def juvenile_scores(genome, target, backend, snapshots, strategy, count,
                    adult_score):
    """Behavioural score at each lifespan checkpoint, length exactly ``count``.

    Padding uses the ADULT score, not zero. An organism that matures in three
    steps has no distinct juvenile stage - at that age it already WAS its adult
    self, and scoring it as a failure would penalise compact development, which
    is not what this objective is measuring.
    """
    from .scoring import score_contract
    scores = []
    for index in checkpoint_indices(len(snapshots), count):
        grid = dict(snapshots[index])
        if backend == 'nervous':
            # Routing patches are heritable mature-cell overrides; apply them to
            # juveniles too so every stage is interpreted under one rule.
            from .nervous import apply_routing_patches
            grid = apply_routing_patches(genome, grid)
        try:
            prep = prepare_grid(genome, target, backend, grid, strategy,
                                record_progress=False)
        except Exception:
            # A half-grown body can fail binding in ways an adult cannot. That
            # is a zero for this checkpoint, not a failed run.
            prep = None
        traces = _traces_of(prep, backend)
        if traces is None or getattr(traces, 'overflow', False):
            scores.append(0.0)
            continue
        scores.append(float(score_contract(traces, target)[0]))
    while len(scores) < count:
        scores.append(float(adult_score))
    return tuple(scores[:count])


def juvenile_mean(scores):
    """Scalar summary of the juvenile vector, for the rank_key tie-break."""
    values = [float(v) for v in (scores or ())]
    return sum(values) / len(values) if values else 0.0


def escape_objectives(genome, target, backend, cases):
    """``(juvenile_mean, robust_case_vector)`` for one evaluated genome.

    Runs inside the evaluation worker. The juvenile mean is recovered from the
    tail of the case vector rather than recomputed - those trailing entries ARE
    the per-checkpoint scores, so there is exactly one measurement and no way
    for the two to disagree.
    """
    escape = getattr(target, '_escape', None)
    if escape is None:
        return 0.0, None
    juvenile = 0.0
    if escape.lifespan_scoring and cases:
        count = escape.lifespan_checkpoints
        if len(cases) >= count:
            juvenile = juvenile_mean(tuple(cases)[-count:])
    return juvenile, robust_case_vector(genome, target, backend, escape)


def robust_case_vector(genome, target, backend, escape):
    """Per-case scores under jittered physics, WORST over the jitter variants.

    Returns None when robustness is off or the run's physics cannot be
    perturbed. Worst-over-variants (rather than mean) is deliberate: a circuit
    that survives a 15% slow-down but dies on a 15% speed-up is not robust, and
    averaging the two would hide exactly that.
    """
    from runtime.escape import jitter_physics
    from .scoring import score_contract
    if escape is None or not escape.robustness:
        return None
    attribute = 'lut_config' if backend == 'lut' else 'pulse_config'
    variants = jitter_physics(getattr(target, attribute, None), escape)
    if not variants:
        return None
    n_cases = contract_case_count(target)
    worst = None
    for config in variants:
        probe = copy.copy(target)
        setattr(probe, attribute, config)
        # Never let a jitter probe recurse into lifespan scoring or into
        # another robustness pass - this measures the ADULT under new physics.
        setattr(probe, '_escape', None)
        try:
            from .temporal import prepare_net
            if backend == 'lut':
                from substrates.lut.ga import prepare_lut
                prep = prepare_lut(genome, probe)
            else:
                prep = prepare_net(genome, probe)
        except Exception:
            prep = None
        traces = _traces_of(prep, backend)
        if traces is None or getattr(traces, 'overflow', False):
            return (0.0,) * n_cases
        cases = score_contract(traces, probe)[1]
        if cases is None:
            return None
        cases = tuple(float(v) for v in cases)
        if len(cases) != n_cases:
            return None
        worst = cases if worst is None else tuple(
            min(a, b) for a, b in zip(worst, cases))
    return worst


def structural_topology(genome, target, *, _developed=None):
    """This organism's reachable computational structure.

    Deliberately target-AGNOSTIC: it grows and interprets the body, then
    measures wiring. No truth table, expected trace, target name, fitted output,
    gene count or telomere reaches it - the target is used only to find the
    growth seeds and the tile architecture.
    """
    from substrates.topology import EMPTY
    from .io_placement import io_strategy, growth_seeds, layout_pads
    from .nervous import grow_nervous, interpret_nervous
    from .temporal import nervous_topology
    try:
        if _developed is None:
            strategy = io_strategy(target)
            grid = grow_nervous(
                genome, seeds=growth_seeds(target, strategy, genome),
                grid_size=target.grid_size, iters=target.iters)
        else:
            grid, strategy = _developed
        if not grid:
            return EMPTY
        arch = getattr(genome, 'arch', 'single')
        routing, in_pos, _ = interpret_nervous(grid, target, arch=arch)
        pads = layout_pads(genome, target)
        if pads is not None:
            if not pads or any(cell not in grid for cell in pads):
                return EMPTY
            in_pos = list(pads)
        return nervous_topology(grid, routing, in_pos, arch=arch)
    except Exception:
        return EMPTY
