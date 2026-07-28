"""
substrates/lut/playback.py — continuous-time playback of a grown LUT array: the LUT
twin of substrates.nervous.playback.NervousPlayer, driving the asynchronous level-logic
engine (substrates.lut.pulse.AsyncLutSim) from the same clickable pulse timeline.

The Interactive and Designer tabs drive BOTH substrates identically now: input
pulses are placed on the shared PulseLaneEditor at real (possibly sub-tick)
times and played through a physical-time cursor. Only the drawn phenotype
differs (hex wires pulsing vs LUT wedges emitting).

No pulse-event cap is imposed by default: a chaotic LUT organism legitimately
produces thousands of wire rises per run (unlike the quiescent nervous net),
and the engine's structural wave cap already guards true runaways.
"""
from __future__ import annotations

from substrates.nervous.pulse import TICK
from substrates.nervous.playback import AsyncPlayer, DEFAULT_DT, DEFAULT_HORIZON

from .pulse import AsyncLutSim


class LutPlayer(AsyncPlayer):
    """Continuous-time playback of one grown LUT array."""

    def __init__(self, grid, horizon=DEFAULT_HORIZON, dt=DEFAULT_DT,
                 pulse_width=None, max_events=None, config=None,
                 inputs=None, outputs=None):
        self.grid       = grid
        self.config     = config
        self.max_events = max_events
        self.inputs     = list(inputs or ())
        self.outputs    = list(outputs or ())
        super().__init__(horizon=horizon, dt=dt, pulse_width=pulse_width,
                         default_width=TICK)

    def _make_sim(self):
        return AsyncLutSim(self.grid, config=self.config,
                           max_events=self.max_events,
                           input_nodes=self.inputs,
                           output_nodes=self.outputs)

    def nibbles(self):
        """{cell: 4-bit N/S/E/W nibble} at the cursor — the emission map the
        LUT wedge view draws (Fig. 14 red/green)."""
        return self.sim.out
