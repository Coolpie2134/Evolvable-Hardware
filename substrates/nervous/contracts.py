"""Declarative behavioral contracts shared by every hardware backend.

A target describes *what must be true* as a collection of constraints.  The
simulators remain substrate-specific, but they all return observations to the
single :func:`nv_evo.scoring.score_contract` entry point.

The constraint relation names are deliberately data, not evaluator modes.  A
contract may combine any number of them, which lets a target require, for
example, both a stateful hold and a precisely timed acknowledgement without a
new scoring pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


RELATION_PRESENTATION = {
    'truth_table': (
        'Truth-table correspondence',
        'Every declared output bit must match for every input row.'),
    'event_correspondence': (
        'One-to-one timed events',
        'Required and produced leading edges are paired; misses and extras both cost.'),
    'logical_state': (
        'Phase-invariant logical state',
        'Active and quiet epochs must hold without requiring an arbitrary pulse phase.'),
    'pulse_intervals': (
        'Complete pulse intervals',
        'Each pulse must have the required rise time and physical duration.'),
    'sustained_cadence': (
        'Sustained cadence',
        'The output must maintain the required rhythm, coverage, and quiet startup.'),
    'commanded_cadence': (
        'Commanded cadence',
        'Each command-delimited dwell must sustain and correctly change its rhythm.'),
    'bounded_state': (
        'Bounded persistent state',
        'Long-horizon hold, quiet, clear, reset, and reload restrictions are tested.'),
}

OBSERVABLE_PRESENTATION = {
    'logic': 'logical output values',
    'rises': 'continuous-time leading edges',
    'samples': 'state epochs reconstructed from physical output',
    'intervals': 'complete rise/fall intervals',
}

PARAMETER_PRESENTATION = {
    'tolerance': 'tolerance',
    'max_shift': 'maximum shared latency',
    'fit_latency': 'fit shared latency',
    'period': 'required period',
    'settle': 'settling allowance',
    'min_events': 'minimum events',
    'min_period': 'minimum period',
    'max_period': 'maximum period',
    'max_delay': 'maximum response delay',
    'strict': 'strict level hold',
    'reset_influence': 'require reset influence',
}


@dataclass
class Constraint:
    """One independently testable behavioral restriction."""

    relation: str
    observable: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class BehaviorContract:
    """An executable target idea.

    ``mean_worst`` retains a useful average gradient while ensuring one badly
    violated restriction cannot disappear inside many easy ones.  A perfect
    score is possible only when every constraint is perfect.
    """

    constraints: List[Constraint]
    aggregation: str = 'mean_worst'
    version: int = 1

    @property
    def observables(self):
        return frozenset(c.observable for c in self.constraints)


def _display_value(value):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return '%g' % value
    return str(value)


def behavior_contract_lines(target_or_contract):
    """Render the executable contract for reports and visual surfaces.

    The presentation is generated from the same data passed to
    ``score_contract`` so the GUI cannot silently describe a different method.
    """
    contract = getattr(target_or_contract, 'contract', target_or_contract)
    lines = [
        'Behavior Contract v%d  |  evaluator: score_contract (shared by every backend)'
        % contract.version,
        'Aggregation: mean + worst restriction; fitness 1.0 requires every restriction to pass.',
        'Restrictions:',
    ]
    for index, clause in enumerate(contract.constraints, 1):
        label, explanation = RELATION_PRESENTATION.get(
            clause.relation,
            (clause.relation.replace('_', ' ').title(),
             'The declared behavioral restriction must be satisfied.'))
        observable = OBSERVABLE_PRESENTATION.get(
            clause.observable, clause.observable.replace('_', ' '))
        weight = '' if abs(float(clause.weight) - 1.0) < 1e-12 \
            else '; weight %g' % float(clause.weight)
        lines.append('  %d. %s  [observes %s%s]' %
                     (index, label, observable, weight))
        lines.append('     ' + explanation)
        if clause.parameters:
            settings = ', '.join(
                '%s=%s' % (PARAMETER_PRESENTATION.get(key, key.replace('_', ' ')),
                           _display_value(value))
                for key, value in clause.parameters.items())
            lines.append('     Settings: ' + settings)
    return lines


def behavior_contract_text(target_or_contract):
    return '\n'.join(behavior_contract_lines(target_or_contract))


def behavior_contract_badge(target_or_contract):
    """Compact summary for the Interactive playback tab."""
    contract = getattr(target_or_contract, 'contract', target_or_contract)
    labels = [RELATION_PRESENTATION.get(
        clause.relation, (clause.relation.replace('_', ' ').title(), ''))[0]
              for clause in contract.constraints]
    return 'Behavior Contract v%d  |  %s  |  mean + worst' % (
        contract.version, ' + '.join(labels))


def logic_contract():
    return BehaviorContract([Constraint('truth_table', 'logic')])


def event_contract(tolerance=0.5, max_shift=12.0, fit_latency=True):
    return BehaviorContract([Constraint(
        'event_correspondence', 'rises', {
            'tolerance': float(tolerance),
            'max_shift': float(max_shift),
            'fit_latency': bool(fit_latency),
        })])


def state_contract(max_shift=12.0, fit_latency=True):
    return BehaviorContract([Constraint(
        'logical_state', 'samples', {
            'max_shift': float(max_shift),
            'fit_latency': bool(fit_latency),
        })])


def interval_contract(tolerance=0.25, max_shift=12.0, fit_latency=True):
    return BehaviorContract([Constraint(
        'pulse_intervals', 'intervals', {
            'tolerance': float(tolerance),
            'max_shift': float(max_shift),
            'fit_latency': bool(fit_latency),
        })])


def cadence_contract(period, tolerance=0.5, settle=5.0, min_events=4,
                     max_shift=12.0):
    return BehaviorContract([Constraint(
        'sustained_cadence', 'rises', {
            'period': float(period),
            'tolerance': float(tolerance),
            'settle': float(settle),
            'min_events': int(min_events),
            'max_shift': float(max_shift),
        })])


def cadence_step_contract(min_period=2, max_period=6, settle=2,
                          min_events=4, max_delay=8):
    return BehaviorContract([Constraint(
        'commanded_cadence', 'samples', {
            'min_period': int(min_period),
            'max_period': int(max_period),
            'settle': int(settle),
            'min_events': int(min_events),
            'max_delay': int(max_delay),
        })])


def bounded_state_contract(strict=False, reset_influence=False):
    return BehaviorContract([Constraint(
        'bounded_state', 'rises', {
            'strict': bool(strict),
            'reset_influence': bool(reset_influence),
        })])


def contract_from_dict(value):
    """Restore a contract embedded in a JSON checkpoint."""
    if isinstance(value, BehaviorContract):
        return value
    if value is None:
        return state_contract()
    return BehaviorContract(
        constraints=[c if isinstance(c, Constraint) else Constraint(**c)
                     for c in value.get('constraints', ())],
        aggregation=value.get('aggregation', 'mean_worst'),
        version=int(value.get('version', 1)))


def legacy_contract(mode, data=None):
    """One-way checkpoint migration for files written before contract v1."""
    data = data or {}
    if mode == 'events':
        return event_contract(data.get('event_tolerance', 0.5),
                              data.get('event_max_shift', 12.0),
                              data.get('fit_latency', True))
    if mode == 'waveform':
        return interval_contract(data.get('waveform_tolerance', 0.25),
                                 data.get('event_max_shift', 12.0),
                                 data.get('fit_latency', True))
    if mode == 'cadence':
        return cadence_contract(data.get('cadence_period', 0.0),
                                data.get('cadence_tolerance', 0.5),
                                data.get('cadence_settle', 5.0),
                                data.get('cadence_min_events', 4),
                                data.get('event_max_shift', 12.0))
    if mode == 'period_stepper':
        return cadence_step_contract(
            data.get('stepper_min_period', 2),
            data.get('stepper_max_period', 6),
            data.get('stepper_settle', 2),
            data.get('stepper_min_events', 4),
            data.get('stepper_max_delay', 8))
    if mode in ('retention', 'sr_retention'):
        return bounded_state_contract(
            strict=bool(data.get('_sr_strict', False)),
            reset_influence=(mode == 'sr_retention'))
    return state_contract(data.get('event_max_shift', 12.0),
                          data.get('fit_latency', True))
