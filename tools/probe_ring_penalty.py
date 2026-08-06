"""
tools/probe_ring_penalty.py - can a PHYSICALLY HONEST ring score 1.0?

The nervous net stores a bit as a pulse circulating a loop. Such a ring cannot
stop faster than its own circulation period: when a reset arrives, the pulse
already in flight completes its lap and lands AFTER the commanded transition.
That trailing pulse is correct behaviour, not a fault.

This probe builds exactly that circuit for every state-contract target - the
ideal solution a human would draw, including the unavoidable trailing pulse -
sweeps plausible ring periods, and reports any target where it fails to score
1.0. A failure means the SCORER is punishing a correct circuit, which is the
same family of fault as the degenerate scorers in tools/probe_trivial_baselines
but pointing the other way: there a wrong circuit passed, here a right one fails.

    py tools/probe_ring_penalty.py [--periods 2,3,4,5,6] [--verbose]

Exit code is non-zero if any target penalises an honest ring, so this can gate a
change to the scoring contract.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous import pulse as physics                    # noqa: E402
from substrates.nervous.scoring import (_expected_windows,          # noqa: E402
                                        has_relation, score_contract)
from substrates.nervous.targets import TEMPORAL_TARGETS            # noqa: E402
from substrates.nervous.temporal import TemporalTraces             # noqa: E402


DEFAULT_PERIODS = (2.0, 3.0, 4.0, 5.0, 6.0)


def state_targets():
    """Targets judged by held logical state - where a ring's stop time matters."""
    out = []
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        if not getattr(target, 'temporal', False):
            continue
        if not hasattr(target, 'contract'):
            continue
        try:
            if has_relation(target, 'logical_state'):
                out.append((name, target))
        except Exception:
            continue
    return out


def honest_ring(target, role, trial, period, width, trailing=True):
    """Held-bit intervals for an ideal ring, including its trailing pulse.

    During each commanded ACTIVE epoch a pulse circulates every ``period``.
    When the epoch ends, the pulse already in flight lands one period after the
    last emission - inside the newly-quiet window. A real loop cannot avoid
    this, so an honest probe must include it.
    """
    intervals = []
    for state, ticks in _expected_windows(trial.expected.get(role, ())):
        if state != 1 or not ticks:
            continue
        start, end = float(min(ticks)), float(max(ticks) + 1)
        when = start
        while when < end:
            intervals.append((when, when + width))
            when += period
        if trailing and when < float(target.T):
            # The lap already under way when the reset arrived.
            intervals.append((when, min(when + width, float(target.T))))
    return intervals


def score_ring(target, period, width, trailing=True):
    roles = [terminal.role for terminal in target.outputs]
    per_role = {}
    for role in roles:
        per_role[role] = [honest_ring(target, role, trial, period, width,
                                      trailing)
                          for trial in target.trials]
    traces = TemporalTraces(
        {role: [[] for _ in target.trials] for role in roles},
        events={role: [[start for start, _ in ivs] for ivs in per_role[role]]
                for role in roles},
        intervals=per_role)
    return score_contract(traces, target)


def probe(periods=DEFAULT_PERIODS, verbose=False):
    width = physics.WIDTH
    budget = 2.0 * (physics.DELAY + physics.WIDTH)
    print('ring width %.1f, scorer circulation budget 2*(delay+width) = %.1f'
          % (width, budget))
    print()
    header = '%-34s %s' % ('target', '  '.join('P=%g' % p for p in periods))
    print(header)
    print('-' * len(header))
    failures = []
    for name, target in state_targets():
        cells, worst = [], None
        for period in periods:
            score, cases, _align = score_ring(target, period, width)
            cells.append('%5.3f' % score)
            if score < 0.999:
                bad = [i for i, c in enumerate(cases) if c < 0.999]
                if worst is None or score < worst[1]:
                    worst = (period, score, bad)
        print('%-34s %s%s' % (name, '  '.join(cells),
                              '   <-- PENALISED' if worst else ''))
        if worst:
            failures.append((name, worst))
            if verbose:
                period, score, bad = worst
                print('        worst at period %g: %.4f, trials %s'
                      % (period, score, bad))
    print()
    if not failures:
        print('OK: an honest ring scores 1.000 on every state target.')
        return 0
    print('%d target(s) penalise a physically honest ring:' % len(failures))
    for name, (period, score, bad) in failures:
        print('  %-32s worst %.4f at period %g (trials %s)'
              % (name, score, period, bad))
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--periods', default=None,
                        help='comma-separated ring periods to sweep')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    periods = (DEFAULT_PERIODS if not args.periods else
               tuple(float(p) for p in args.periods.split(',')))
    return probe(periods, args.verbose)


if __name__ == '__main__':
    raise SystemExit(main())
