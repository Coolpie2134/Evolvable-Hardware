"""Leak-free train/validation evaluation for temporal developmental circuits.

Output-cell identity and temporal alignment are fitted parameters.  Evolution
may select them on its training schedules, but validation must reuse them
unchanged; otherwise validation silently searches for a new readout and delay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Pos = Tuple[int, int]


@dataclass(frozen=True)
class FittedReadout:
    """The complete fitted evaluation state carried into fresh schedules."""

    backend: str
    outputs: Tuple[Tuple[str, Pos], ...]
    alignment: Optional[float]
    training_score: float

    @property
    def output_positions(self):
        return dict(self.outputs)


def fit_readout(genome, target, backend='nervous'):
    """Fit output cells and one shared alignment on training schedules."""
    if (backend == 'nervous'
            and getattr(target, 'score_mode', '') == 'sr_retention'):
        from .persistence import fit_sr_readout
        return fit_sr_readout(genome, target)
    if backend == 'nervous':
        from .temporal import prepare_net, score_temporal_bundle
        prep = prepare_net(genome, target)
        if prep is None:
            return None
        out_pos, traces = prep[3], prep[4]
    elif backend == 'lut':
        from lut_evo.ga import prepare_lut
        from .temporal import score_temporal_bundle
        prep = prepare_lut(genome, target)
        if prep is None:
            return None
        out_pos, traces = prep[1], prep[2]
    else:
        raise ValueError("unknown temporal backend: %s" % backend)
    if getattr(traces, 'overflow', False):
        return None
    score, _, alignment = score_temporal_bundle(traces, target)
    outputs = tuple(sorted(out_pos.items()))
    return FittedReadout(backend, outputs, alignment, score)


def score_frozen(genome, target, fitted):
    """Score fresh schedules without changing the fitted cell or alignment."""
    if (fitted.backend == 'nervous'
            and getattr(target, 'score_mode', '') == 'sr_retention'):
        from .persistence import score_sr_frozen
        return score_sr_frozen(genome, target, fitted)
    out_pos = fitted.output_positions
    expected_roles = {terminal.role for terminal in target.outputs}
    if set(out_pos) != expected_roles:
        raise ValueError('fitted output roles do not match validation target')

    if fitted.backend == 'nervous':
        from .nervous import (grow_nervous, interpret_nervous, node_widths,
                              node_delays)
        from .temporal import trace_fixed_outputs, score_temporal_bundle
        grid = grow_nervous(genome, seeds=tuple(target.inputs),
                            grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs:
            return 0.0
        routing, in_pos, _ = interpret_nervous(grid, target)
        if any(pos not in grid for pos in in_pos):
            return 0.0
        # carry the evolved per-node pulse widths into validation too, so a
        # fitted 'evolved_width' champion is scored on the same physics
        # (node_widths returns None off that model).
        config = getattr(target, 'pulse_config', None)
        widths = node_widths(genome, grid, config)
        delays = node_delays(genome, grid, config)
        traces = trace_fixed_outputs(
            grid, routing, in_pos, out_pos, target, widths=widths,
            delays=delays)
    elif fitted.backend == 'lut':
        from lut_evo.lut import grow_lut
        from lut_evo.ga import trace_fixed_outputs
        from .temporal import score_temporal_bundle
        grid = grow_lut(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs or any(
                pos not in grid for pos in target.inputs):
            return 0.0
        traces = trace_fixed_outputs(
            grid, list(target.inputs), out_pos, target)
    else:
        raise ValueError("unknown fitted backend: %s" % fitted.backend)

    if traces is None or getattr(traces, 'overflow', False):
        return 0.0
    return score_temporal_bundle(
        traces, target, alignment=fitted.alignment)[0]
