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
from .nervous import grow_nervous, interpret_nervous, node_widths, node_delays
from .temporal import _obs_len

# ── predeclared scoring constants (a state-1 hold rings with period ~D+W) ────────
# The nervous-net memory holds a bit as a pulse CIRCULATING (reads as ~50%-duty
# ringing), and a 1->0 flip leaves a short decay tail before the ring is vetoed.
# So we (a) drop a transition guard band, (b) exempt a leading decay/ramp window,
# (c) skip intervals too short to judge after those, and (d) score by sustained
# occupancy: a held bit rings (coverage present, no long silent gap); a cleared
# bit stays quiet (low coverage after the decay window).
GUARD_FRAC   = 1.0     # transition guard band  = GUARD_FRAC * DELAY (each side)
LEAD_FRAC    = 1.0     # leading decay/ramp exemption = LEAD_FRAC * (D+W)
MININT_FRAC  = 1.0     # need interior >= MININT_FRAC*(D+W) to judge, else skip
RING_COV     = 0.30    # state-1 interior coverage giving full credit (ring ~0.5)
QUIET_COV    = 0.30    # state-0 interior coverage tolerated before score 0
GAP_TOL_FRAC = 2.0     # a state-1 silent gap beyond (D+W)+this*(D+W) => bit lost
PASS         = 0.90    # a case passes iff worst-interval score >= PASS


def _cov_gap(pulses, lo, hi):
    """Covered length and largest uncovered gap of `pulses` within [lo, hi]."""
    segs = []
    for (s, e) in pulses:
        s2, e2 = (s if s > lo else lo), (e if e < hi else hi)
        if e2 > s2:
            segs.append((s2, e2))
    segs.sort()
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], e if e > merged[-1][1] else merged[-1][1])
        else:
            merged.append((s, e))
    covered = sum(e - s for s, e in merged)
    cur, max_gap = lo, 0.0
    for s, e in merged:
        if s - cur > max_gap:
            max_gap = s - cur
        cur = e
    if hi - cur > max_gap:
        max_gap = hi - cur
    return covered, max_gap


def parity_intervals(edges, horizon):
    """Commanded (state, start, end) intervals from effective input edges: state
    starts 0 and flips at each edge (toggle / T-flip-flop semantics)."""
    bounds = [0.0] + sorted(edges) + [float(horizon)]
    return [(i % 2, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def sr_intervals(set_edges, reset_edges, horizon):
    """SR-latch command intervals from effective float edges.

    Reset wins an exact-time tie, matching the nervous net's inhibitory veto.
    Repeated Set or Reset edges leave the commanded state unchanged.
    """
    grouped = {}
    for t in set_edges:
        grouped.setdefault(float(t), [False, False])[0] = True
    for t in reset_edges:
        grouped.setdefault(float(t), [False, False])[1] = True
    state, start, out = 0, 0.0, []
    for t in sorted(grouped):
        if t < 0.0 or t > horizon:
            continue
        has_set, has_reset = grouped[t]
        new_state = 0 if has_reset else (1 if has_set else state)
        if new_state != state:
            if t > start:
                out.append((state, start, t))
            state, start = new_state, t
    if horizon > start:
        out.append((state, start, float(horizon)))
    return out


def score_interval_graded(rise, state, start, end, offset):
    """Graded score for one independently selectable SR behavior case.

    Active intervals receive the fraction of consecutive windows retained
    before the first loss. Quiet intervals receive mean quietness, which gives
    Reset a gradient as ringing is progressively suppressed. Final claims use
    :func:`score_retention`, not this training-only gradient.
    """
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead = GUARD_FRAC * D, LEAD_FRAC * (D + W)
    lo = start + offset + guard + (0.0 if state == 1 else lead)
    hi = end + offset - guard
    win = 2.0 * (D + W)
    if hi - lo < D + W:
        return 0.0
    pulses = [(t, t + W) for t in sorted(rise)]
    scores, t = [], lo
    while t + 1e-9 < hi:
        wh = min(t + win, hi)
        if wh - t >= D + W:
            cov, _ = _cov_gap(pulses, t, wh)
            frac = cov / (wh - t)
            scores.append(min(1.0, frac / RING_COV) if state == 1
                          else max(0.0, 1.0 - frac / QUIET_COV))
        t += win
    if not scores:
        return 0.0
    if state == 0:
        return sum(scores) / len(scores)
    retained = 0.0
    for score in scores:
        if score < PASS:
            retained += score / PASS
            break
        retained += 1.0
    return retained / len(scores)


def score_reset_influence(rise, reset_time, offset):
    """Behavioral gradient for Reset: reward reduced activity after its arrival.

    A candidate gets no credit unless it was active before Reset. This remains
    a curriculum aid only; the full phase requires actual sustained quiet.
    """
    W, D = pulse.WIDTH, pulse.DELAY
    span = 4.0 * (D + W)
    arrival = reset_time + offset
    pulses = [(t, t + W) for t in sorted(rise)]
    pre, _ = _cov_gap(pulses, max(0.0, arrival - span), arrival - D)
    post_lo = arrival + LEAD_FRAC * (D + W)
    post, _ = _cov_gap(pulses, post_lo, post_lo + span)
    pre_frac = pre / max(D + W, span - D)
    post_frac = post / span
    if pre_frac <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (pre_frac - post_frac) / pre_frac))


def score_state_intervals(rise, intervals, offset):
    """Worst-interval semantic score, or None if no interval is long enough to
    judge. `intervals` are (state,a,b) in INPUT time; `offset` maps input time to
    output time (frozen). A state-1 interval must ring (sustained coverage, no
    silent gap beyond tolerance); a state-0 interval must stay quiet after its
    decay window."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    gap_lost = (D + W) + GAP_TOL_FRAC * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        # ringing starts immediately at the offset (no ramp), so state-1 gets only
        # the transition guard; a cleared bit's ring takes a moment to die, so
        # state-0 additionally exempts a leading decay window.
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:                # too short to judge after exemptions
            continue
        cov, gap = _cov_gap(pulses, lo, hi)
        frac = cov / (hi - lo)
        if state == 1:                      # held bit: ringing, no long silence
            s = min(1.0, frac / RING_COV)
            s = min(s, 1.0 - min(1.0, max(0.0, gap - (D + W)) / (gap_lost - (D + W))))
        else:                               # cleared bit: quiet
            s = max(0.0, 1.0 - frac / QUIET_COV)
        if s < worst:
            worst = s
        scored += 1
    return worst if scored else None


def _windows_worst(pulses, lo, hi, state):
    """Slide a window across [lo, hi]; return the WORST window's correctness. A
    held (state-1) interval must stay active in EVERY window (a sustained ring,
    not a burst that dies); a cleared (state-0) interval must stay quiet in every
    window. This is what turns 'held for a moment' into 'retained across the gap'."""
    W, D = pulse.WIDTH, pulse.DELAY
    win = 2.0 * (D + W)
    worst, t = 1.0, lo
    while t + 1e-9 < hi:
        wh = min(t + win, hi)
        if wh - t >= (D + W):                 # judge only near-full windows
            cov, _ = _cov_gap(pulses, t, wh)
            frac = cov / (wh - t)
            s = min(1.0, frac / RING_COV) if state == 1 else max(0.0, 1.0 - frac / QUIET_COV)
            if s < worst:
                worst = s
        t += win
    return worst


def score_retention(rise, intervals, offset):
    """Worst-window-worst-interval retention score (or None if unjudgeable).
    Unlike score_state_intervals (which checks a hold over one window), this
    requires the commanded state to be sustained across the WHOLE interval — so a
    finite ring burst that decays before the next command fails at the horizon
    where it dies."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:
            continue
        s = _windows_worst(pulses, lo, hi, state)
        if s < worst:
            worst = s
        scored += 1
    return worst if scored else None


def score_retention_graded(rise, intervals, offset):
    """Gradient version of score_retention for EVOLUTION: each interval is scored
    by the FRACTION of its windows correct BEFORE the first sustained failure
    (0..1), worst interval. Strict score_retention has no gradient (0 until a bit
    holds the WHOLE horizon); this rewards holding longer, so selection can climb
    from a short burst toward a sustained loop. Use the strict version for claims."""
    W, D = pulse.WIDTH, pulse.DELAY
    guard, lead, minint = GUARD_FRAC * D, LEAD_FRAC * (D + W), MININT_FRAC * (D + W)
    win = 2.0 * (D + W)
    pulses = [(t, t + W) for t in sorted(rise)]
    worst, scored = 1.0, 0
    for (state, a, b) in intervals:
        lo = a + offset + guard + (0.0 if state == 1 else lead)
        hi = b + offset - guard
        if hi - lo < minint:
            continue
        total, correct, failed, t = 0, 0, False, lo
        while t + 1e-9 < hi:
            wh = min(t + win, hi)
            if wh - t >= (D + W):
                total += 1
                cov, _ = _cov_gap(pulses, t, wh)
                frac = cov / (wh - t)
                s = min(1.0, frac / RING_COV) if state == 1 else max(0.0, 1.0 - frac / QUIET_COV)
                if not failed and s >= PASS:
                    correct += 1
                else:
                    failed = True
            t += win
        frac_ret = (correct / total) if total else 0.0
        if frac_ret < worst:
            worst = frac_ret
        scored += 1
    return worst if scored else None


def total_offset(target, fitted):
    """Frozen input->output offset in TIME: nominal latency (target metadata) +
    fitted alignment, both in ticks, converted by TICK. Never refitted."""
    align = fitted.alignment or 0.0
    return (float(target.latency) + float(align)) * pulse.TICK


def semantic_trial_scores(genome, target, fitted, schedules):
    """Frozen-readout semantic score per trial for a list of (possibly perturbed)
    float `schedules`. Grows once; injects each schedule; scores worst-interval."""
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    routing, in_pos, _ = interpret_nervous(grid, target)
    cell = fitted.output_positions[target.outputs[0].role]
    if cell not in grid:
        return [0.0] * len(schedules)
    widths = node_widths(genome, grid, getattr(target, 'pulse_config', None))
    delays = node_delays(genome, grid, getattr(target, 'pulse_config', None))
    offset = total_offset(target, fitted)
    # Judge over the TRAINED observation window (_obs_len): this is a JITTER-
    # robustness test, not a hold-duration test. (Separately noted: this winner's
    # "hold" is a finite ~10-unit ring burst, not indefinite persistence — a real
    # limitation the short training window hid, but out of scope for jitter here.)
    horizon = _obs_len(target) * pulse.TICK
    out = []
    for sched in schedules:
        res, overflow = ae.run_schedule(grid, routing, in_pos, sched, horizon,
                                        [cell], max_events=target.max_events,
                                        config=getattr(target, 'pulse_config', None),
                                        widths=widths, delays=delays)
        if overflow:
            out.append(0.0)
            continue
        edges = ae.effective_edges(sched)[0]        # single-input parity target
        intervals = parity_intervals(edges, horizon)
        out.append(score_state_intervals(res[cell], intervals, offset))
    return out
