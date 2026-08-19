"""Score TRIVIAL behaviours against a TEMPORAL target.

tools/probe_trivial_baselines.py does this for combinational truth tables by
enumerating boolean readouts. Temporal targets need the same guard for the same
reason - every degeneracy this project has found was a trivial circuit scoring
near 1.0, and none of them were visible in a solve-rate table - but their
observations are traces over time, not truth-table rows.

So this synthesises the observation a trivial circuit WOULD produce and scores
it through the real contract:

  silent      never fires
  blanket     fires on every tick
  echo A      copies input A (a bare wire from the first input)
  echo B      copies input B
  echo A|B    fires whenever either input does
  echo A&B    fires when both inputs are high on the same tick

A target is well-posed when its own oracle scores 1.0 and every one of these
sits far below it. A trivial behaviour at or near the score search reaches means
search is parked on a decoy.

Usage:
    py tools/probe_temporal_baselines.py --target "Gap band-pass (oracle)"
    py tools/probe_temporal_baselines.py            # all oracle targets
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from substrates.nervous.oracle import ORACLE_TARGETS          # noqa: E402
from substrates.nervous.scoring import (TemporalTraces,        # noqa: E402
                                        score_contract)


def _streams(trial):
    return [tuple(row) for row in trial.streams]


def _behaviours(streams):
    """{name: per-tick 0/1 output} for each trivial circuit, on one trial."""
    n = len(streams[0]) if streams else 0
    out = {
        'silent': [0] * len(streams),
        'blanket': [1] * len(streams),
    }
    for index in range(n):
        out['echo in%d' % index] = [int(bool(row[index])) for row in streams]
    if n >= 2:
        out['echo A|B'] = [int(bool(row[0]) or bool(row[1])) for row in streams]
        out['echo A&B'] = [int(bool(row[0]) and bool(row[1]))
                           for row in streams]
    return out


def _as_traces(target, per_trial):
    """Wrap per-trial 0/1 sequences as the observation object scorers expect.

    EVERY role gets the series, not just the first. Scoring only
    ``outputs[0]`` and leaving the rest absent made a multi-output target's own
    ORACLE score 0.7398 and 0.5333 - the probe reporting the perfect answer as
    wrong, which looks exactly like a broken target and is not.

    Events and intervals are derived from the same sequence so event-relation
    and level-relation contracts both see a consistent circuit: a run of 1s is
    one held interval, and its leading edge is one point event.
    """
    roles = [terminal.role for terminal in target.outputs]
    events, intervals = [], []
    for series in per_trial:
        rises, spans, start = [], [], None
        for tick, value in enumerate(series):
            if value and start is None:
                start = tick
                rises.append(float(tick))
            elif not value and start is not None:
                spans.append((float(start), float(tick)))
                start = None
        if start is not None:
            spans.append((float(start), float(len(series))))
        events.append(rises)
        intervals.append(spans)
    return TemporalTraces({role: list(per_trial) for role in roles},
                          events={role: events for role in roles},
                          intervals={role: intervals for role in roles})


def _oracle_traces(target):
    """The target's OWN expected output, per role - the perfect-circuit control."""
    roles = [terminal.role for terminal in target.outputs]
    series = {role: [] for role in roles}
    for trial in target.trials:
        for role in roles:
            expected = trial.expected.get(role, [])
            series[role].append(
                [0 if value is None else int(value) for value in expected])
    events, intervals = {}, {}
    for role in roles:
        role_events, role_spans = [], []
        for run in series[role]:
            rises, spans, start = [], [], None
            for tick, value in enumerate(run):
                if value and start is None:
                    start = tick
                    rises.append(float(tick))
                elif not value and start is not None:
                    spans.append((float(start), float(tick)))
                    start = None
            if start is not None:
                spans.append((float(start), float(len(run))))
            role_events.append(rises)
            role_spans.append(spans)
        events[role] = role_events
        intervals[role] = role_spans
    return TemporalTraces(series, events=events, intervals=intervals)


def probe(target):
    names = None
    collected = {}
    for trial in target.trials:
        streams = _streams(trial)
        behaviours = _behaviours(streams)
        if names is None:
            names = list(behaviours)
            collected = {name: [] for name in names}
        for name in names:
            collected[name].append(behaviours[name])

    rows = []
    for name, per_trial in collected.items():
        score, _cases, _align = score_contract(_as_traces(target, per_trial),
                                               target)
        rows.append((name, score))
    # The target's own oracle output, as a sanity check that a PERFECT circuit
    # scores 1.0 through this same path. If it does not, the probe is wrong,
    # not the target.
    score, _cases, _align = score_contract(_oracle_traces(target), target)
    rows.append(('ORACLE (perfect)', score))
    rows.sort(key=lambda row: -row[1])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', action='append', default=None)
    args = parser.parse_args()
    names = args.target or sorted(ORACLE_TARGETS)
    for name in names:
        target = ORACLE_TARGETS.get(name)
        if target is None:
            print('unknown target: %s' % name)
            continue
        print('\n%s' % name)
        rows = probe(target)
        best_wrong = max((s for n, s in rows if n != 'ORACLE (perfect)'),
                         default=0.0)
        for label, score in rows:
            flag = ''
            if label != 'ORACLE (perfect)' and score >= 0.85:
                flag = '   <-- DECOY'
            print('   %-18s %.4f%s' % (label, score, flag))
        oracle = dict(rows).get('ORACLE (perfect)', 0.0)
        print('   gap oracle - best trivial: %.4f' % (oracle - best_wrong))


if __name__ == '__main__':
    main()
