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
    # Native fitted probes store one Pos per role. Compatibility bindings may
    # store a tuple of Pos values: an immutable wired-OR output group.
    outputs: Tuple
    alignment: Optional[float]
    training_score: float
    # Input sites frozen at fit time. Native layouts use a flat tuple; legacy
    # tag/wiring strategies may use attachment groups. Empty means use the
    # target's declared pads for an old fixed-input document. Defaulted so
    # legacy 4-arg construction still works.
    inputs: Tuple = ()

    @property
    def output_positions(self):
        return dict(self.outputs)

    def input_positions(self, target):
        """The frozen input sites, or target pads for a legacy fixed document."""
        return list(self.inputs) if self.inputs else list(target.inputs)


def fit_readout(genome, target, backend='nervous'):
    """Fit probes/alignment and freeze any genome-selected input binding."""
    def _freeze_inputs(entries):
        """Nested-tuple form of an in_pos (groups stay groups, cells stay cells)."""
        from .io_placement import input_groups
        return tuple(tuple(group) for group in input_groups(entries))

    in_pos = ()
    if backend == 'nervous':
        from .temporal import prepare_net
        from .scoring import score_contract
        prep = prepare_net(genome, target)
        if prep is None:
            return None
        in_pos, out_pos, traces = _freeze_inputs(prep[2]), prep[3], prep[4]
    elif backend == 'lut':
        from substrates.lut.ga import prepare_lut
        from .scoring import score_contract
        prep = prepare_lut(genome, target)
        if prep is None:
            return None
        out_pos, traces = prep[1], prep[2]
        in_pos = _freeze_inputs(prep[3]) if len(prep) > 3 else ()
    elif backend == 'fnv':
        from substrates.fnv.evaluation import prepare_functional
        from .scoring import score_contract
        prep = prepare_functional(genome, target)
        if prep is None:
            return None
        _, inputs, out_pos, observations = prep
        if getattr(target, 'temporal', False):
            traces = observations
        else:
            traces = [
                [float(case['acts'][terminal.role])
                 for terminal in target.outputs]
                for case in observations
            ]
        # FNV input_layout is a flat tuple of physical source pads.
        in_pos = tuple(tuple(cell) for cell in inputs)
    else:
        raise ValueError("unknown temporal backend: %s" % backend)
    if getattr(traces, 'overflow', False):
        return None
    score, _, alignment = score_contract(traces, target)
    if getattr(target, 'io_placement', 'fixed') == 'fixed':
        outputs = tuple(sorted(out_pos.items()))
    else:
        from .io_placement import output_groups
        outputs = tuple(sorted(
            (role, tuple(cells))
            for role, cells in output_groups(out_pos).items()))
    # Only carry the input binding for genuinely evolvable strategies; leave it
    # empty for 'fixed' so input_positions() falls back to the seed pads and
    # nothing about the legacy path changes.
    # A genome carrying an evolved input_layout freezes its EXACT flat pad
    # tuple. Validation must reuse these coordinates verbatim: re-resolving the
    # layout there would let held-out silently pick a different input binding
    # from the one training was scored on.
    evolved_layout = getattr(genome, 'input_layout', None) is not None
    if evolved_layout:
        # The EXACT FLAT pad tuple, one coordinate per logical input. A layout
        # pad is a single cell - there is no fan-out to represent - so the
        # grouped form would only add a level of nesting for validation to
        # unwrap, and flat is already the shape the fixed path hands on.
        from .io_placement import flat_inputs as _flat_inputs
        fitted_inputs = tuple(tuple(cell) for cell in _flat_inputs(in_pos))
    else:
        fitted_inputs = (
            in_pos
            if getattr(target, 'io_placement', 'fixed') != 'fixed' else ())
    return FittedReadout(backend, outputs, alignment, score, inputs=fitted_inputs)


def score_frozen(genome, target, fitted):
    """Score fresh schedules without changing the fitted cell or alignment."""
    out_pos = fitted.output_positions
    expected_roles = {terminal.role for terminal in target.outputs}
    if set(out_pos) != expected_roles:
        raise ValueError('fitted output roles do not match validation target')
    from .io_placement import io_strategy
    strategy = io_strategy(target)

    if fitted.backend == 'nervous':
        from .nervous import (grow_nervous, interpret_nervous, node_delays)
        from .temporal import trace_fixed_outputs
        from .scoring import score_contract
        from .io_placement import growth_seeds
        # Use the same genome-aware developmental origin as training; otherwise
        # validation could grow a different organism from the fitted one.
        grid = grow_nervous(genome, seeds=growth_seeds(
                                target, strategy, genome),
                            grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs:
            return 0.0
        arch = getattr(genome, 'arch', 'single')
        routing, _, _ = interpret_nervous(grid, target, arch=arch)
        # Drive the FITTED input cells (the genome's evolved binding under an
        # io_placement strategy); fall back to the seed pads for fixed binding.
        from .io_placement import flat_inputs
        in_pos = fitted.input_positions(target)
        if any(pos not in grid for pos in flat_inputs(in_pos)):
            return 0.0
        # carry the evolved per-node delays into validation too, so a fitted
        # width-preserving champion is scored on the same physics
        # (node_delays returns None off that model).
        config = getattr(target, 'pulse_config', None)
        delays = None if arch == 'tri3' else node_delays(genome, grid, config)
        # Source membership comes from the FROZEN pads, not from re-resolving
        # the genome's layout, so validation runs the same terminal physics on
        # the same coordinates that training was scored under.
        source_nodes = (
            {tuple(cell) for cell in flat_inputs(in_pos)}
            if getattr(genome, 'input_layout', None) is not None else None)
        traces = trace_fixed_outputs(
            grid, routing, in_pos, out_pos, target, delays=delays, arch=arch,
            source_nodes=source_nodes)
    elif fitted.backend == 'lut':
        from substrates.lut.lut import grow_lut
        from substrates.lut.ga import trace_fixed_outputs
        from .scoring import score_contract
        from .io_placement import growth_seeds
        # Drive the FITTED input cells (the genome's evolved binding under an
        # io_placement strategy); the seed pads for fixed binding.
        from .io_placement import flat_inputs
        in_pos = fitted.input_positions(target)
        evolved_layout = getattr(genome, 'input_layout', None) is not None
        seeds = (tuple(flat_inputs(in_pos)) if evolved_layout
                 else growth_seeds(target, strategy, genome))
        grid = grow_lut(
            genome, seeds=seeds,
            grid_size=target.grid_size, iters=target.iters)
        if len(grid) <= target.n_inputs or any(
                pos not in grid for pos in flat_inputs(in_pos)):
            return 0.0
        traces = trace_fixed_outputs(
            grid, list(in_pos), out_pos, target,
            source_nodes=(
                {tuple(cell) for cell in flat_inputs(in_pos)}
                if evolved_layout else None))
    elif fitted.backend == 'fnv':
        from substrates.fnv.construction import grow_functional
        from substrates.fnv.evaluation import (
            score_fixed_logic_outputs, trace_fixed_outputs,
        )
        from .scoring import score_contract
        in_pos = fitted.input_positions(target)
        grid = grow_functional(
            genome, in_pos, grid_size=target.grid_size, iters=target.iters)
        if (len(grid) <= target.n_inputs
                or any(pos not in grid for pos in in_pos)):
            return 0.0
        if getattr(target, 'temporal', False):
            traces = trace_fixed_outputs(
                grid, list(in_pos), out_pos, target)
        else:
            traces = score_fixed_logic_outputs(
                genome, grid, list(in_pos), out_pos, target)
    else:
        raise ValueError("unknown fitted backend: %s" % fitted.backend)

    if traces is None or getattr(traces, 'overflow', False):
        return 0.0
    return score_contract(
        traces, target, alignment=fitted.alignment)[0]
