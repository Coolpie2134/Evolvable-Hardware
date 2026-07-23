"""
nv_evo/robustness.py — semantic asynchronous-robustness harness.

Judges an evolved circuit's SEMANTIC correctness (was the commanded behaviour
produced?) when its inputs are perturbed off the integer lattice, on the audited
floating-time path (nv_evo.simulation). It is deliberately NOT the structural
event-count probe: it scores whether the commanded stored state is held.

Rules (per the review):
  * readout cell and temporal alignment are FROZEN from training and never
    refitted on perturbed output;
  * the commanded state is computed from the EFFECTIVE (wired-OR merged) input
    edge train, not the pre-merge schedule;
  * nominal input->output latency is explicit target metadata; the fitted
    alignment is a separate, additional offset;
  * each commanded interval is scored separately and the WORST interval is the
    trial's score, so one long hold cannot mask a failed transition;
  * a transition guard band (±DELAY) is excluded — transitions are fuzzy;
  * overflow ⇒ zero; the target's event cap is enforced;
  * thresholds are predeclared here, before any perturbation is inspected.
"""
from __future__ import annotations

from . import pulse
from . import simulation as ae
from .nervous import grow_nervous, interpret_nervous, node_delays

# The predeclared coverage constants and every interval/retention scorer now
# live in nv_evo/scoring.py (the coverage relation of the scoring contract).
# Re-exported here for the historical import path.
from .scoring import (                                          # noqa: F401
    GUARD_FRAC, LEAD_FRAC, MININT_FRAC, GAP_TOL_FRAC, RING_COV, QUIET_COV,
    PASS, _cov_gap, _windows_worst, _obs_len,
    parity_intervals, sr_intervals,
    score_state_intervals, score_retention, score_retention_graded,
    score_interval_graded, score_reset_influence)


def total_offset(target, fitted):
    """Frozen input->output offset in TIME: nominal latency (target metadata) +
    fitted alignment, both in ticks, converted by TICK. Never refitted."""
    align = fitted.alignment or 0.0
    return (float(target.latency) + float(align)) * pulse.TICK


def semantic_trial_scores(genome, target, fitted, schedules):
    """Frozen-readout semantic score per trial for a list of (possibly perturbed)
    float `schedules`. Grows once; injects each schedule; scores worst-interval."""
    from .io_placement import growth_seeds
    grid = grow_nervous(genome, seeds=growth_seeds(target),
                        grid_size=target.grid_size, iters=target.iters)
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, _ = interpret_nervous(grid, target, arch=arch)
    # Drive the FITTED binding (the genome's evolved attachment groups under an
    # io_placement strategy); the pads under fixed binding.
    in_pos = fitted.input_positions(target)
    from .io_placement import output_groups, merge_intervals
    cells = output_groups(fitted.output_positions)[target.outputs[0].role]
    if any(cell not in grid for cell in cells):
        return [0.0] * len(schedules)
    delays = (None if arch == 'tri3' else
              node_delays(genome, grid, getattr(target, 'pulse_config', None)))
    offset = total_offset(target, fitted)
    # Judge over the TRAINED observation window (_obs_len): this is a JITTER-
    # robustness test, not a hold-duration test. (Separately noted: this winner's
    # "hold" is a finite ~10-unit ring burst, not indefinite persistence — a real
    # limitation the short training window hid, but out of scope for jitter here.)
    horizon = _obs_len(target) * pulse.TICK
    out = []
    for sched in schedules:
        res, overflow = ae.run_schedule(
            grid, routing, in_pos, sched, horizon, cells,
            max_events=target.max_events,
            config=getattr(target, 'pulse_config', None),
            delays=delays, return_intervals=True, arch=arch)
        if overflow:
            out.append(0.0)
            continue
        edges = ae.effective_edges(sched)[0]        # single-input parity target
        intervals = parity_intervals(edges, horizon)
        rises = [start for start, _ in
                 merge_intervals([res[cell] for cell in cells])]
        out.append(score_state_intervals(rises, intervals, offset))
    return out
