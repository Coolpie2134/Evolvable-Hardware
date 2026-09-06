"""Shared stimulus parameters and analog nervous-net run configuration.

The digital uniform and width-preserving engines have been removed. Their model
identifiers remain readable metadata for other backends; Nervous execution
accepts only paper_analog. WIDTH is external stimulus duration, not an analog
output-width parameter. COINC is retained in serialized run configurations.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

DELAY = 1.0
WIDTH = 1.0
COINC = 0.5
TICK = 1.0
NODE_MODELS = ('uniform', 'pulse_delay', 'paper_analog')

@dataclass(frozen=True)
class PulseConfig:
    """Immutable paper-model timing parameters carried with one run."""
    delay: float = DELAY
    width: float = WIDTH
    coincidence: float = COINC
    event_cap: int = 2048
    # Runnable NV physics; old identifiers are retained only as metadata.
    model: str = 'paper_analog'
    # Fixed physical constants for the paper_analog model.  They live on the
    # run configuration (rather than being hidden AnalogConfig defaults) so a
    # checkpoint reproduces the same circuit.
    analog_threshold: float = 0.5
    analog_step: float = 0.34
    analog_tau_leak: float = 1.10
    analog_hysteresis: float = 0.08

    def __post_init__(self):
        analog_values = (self.analog_threshold, self.analog_step,
                         self.analog_tau_leak, self.analog_hysteresis)
        if (not all(math.isfinite(v) for v in
                    (self.delay, self.width, self.coincidence) + analog_values)
                or self.delay <= 0 or self.width <= 0 or self.coincidence < 0
                or self.event_cap < 1):
            raise ValueError('delay/width/event_cap must be positive and coincidence non-negative')
        if self.model not in NODE_MODELS:
            raise ValueError('model must be one of %s' % (NODE_MODELS,))
        if not (0 < self.analog_threshold < 1.0):
            raise ValueError('analog_threshold must lie strictly between 0 and 1')
        gap = 1.0 - self.analog_threshold
        if not (gap / 2.0 < self.analog_step < gap):
            raise ValueError('analog_step must let two terminal edges fire but not one')
        if self.analog_tau_leak <= 0 or self.analog_hysteresis < 0:
            raise ValueError('analog_tau_leak must be positive and analog_hysteresis non-negative')
        if self.analog_threshold + self.analog_hysteresis >= 1.0:
            raise ValueError('analog threshold plus hysteresis must remain below rest (1.0)')
