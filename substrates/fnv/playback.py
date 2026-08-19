"""Scorer-faithful continuous-time playback for Functional NV Nets."""
from __future__ import annotations

from substrates.nervous.playback import AsyncPlayer
from substrates.nervous.scoring import _obs_len

from .evaluation import functional_logic_horizon, prepare_functional
from .simulation import FunctionalSim, TICK


class FunctionalPlayer(AsyncPlayer):
    """Interactive player over the same ``FunctionalSim`` used by fitness."""

    def __init__(self, grid, inputs, outputs=(), *, horizon=40.0, dt=0.5,
                 max_events=2048):
        self.grid = grid
        self.inputs = [tuple(cell) for cell in inputs]
        self.outputs = [tuple(cell) for cell in outputs]
        self.max_events = max_events
        super().__init__(
            horizon=horizon, dt=dt, pulse_width=TICK, default_width=TICK)

    def _make_sim(self):
        return FunctionalSim(
            self.grid, input_nodes=self.inputs, output_nodes=self.outputs,
            max_events=self.max_events)


def prepare_functional_playback(genome, target):
    """The exact grown pads/probes and horizon used by FNV fitness, plus the
    branch that built each cell so a view can colour by provenance."""
    prepared = prepare_functional(genome, target)
    if prepared is None:
        return None
    grid, inputs, outputs, _observations = prepared
    horizon = (
        _obs_len(target)
        if getattr(target, "temporal", False)
        else functional_logic_horizon(genome, grid=grid, inputs=inputs)
    )
    from .construction import develop_constructive
    owners = develop_constructive(genome, tuple(inputs)).owners
    return grid, inputs, outputs, float(horizon), owners


def functional_case_pulses(target, n_inputs, horizon, case_index=0):
    """FNV's exact combinational stimulus: asserted inputs held from time 0."""
    rows = getattr(target, "cases", None) or ()
    pulses = [[] for _ in range(int(n_inputs))]
    if not rows:
        return pulses
    bits = rows[max(0, min(int(case_index), len(rows) - 1))][0]
    for index, bit in enumerate(bits[:len(pulses)]):
        if bit:
            pulses[index] = [(0.0, float(horizon))]
    return pulses
