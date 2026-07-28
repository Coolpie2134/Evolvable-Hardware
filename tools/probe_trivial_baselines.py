"""Score TRIVIAL behaviours against every combinational target.

The diagnostic premise: if a bare wire — or silence, or firing on every window
— scores what a 100-generation search achieves, the target is not measuring the
function it names. Two real scoring bugs in this project were found exactly this
way (the One-shot echo degeneracy and the periodic-combinational blanket-fire
degeneracy), and both were invisible to a solve-rate table.

Rather than guess which trivial behaviours matter, this enumerates EVERY boolean
readout of the target's inputs (all 2**(2**n) of them for small n) and reports:

  correct    the score of the target's own truth table  (must be 1.0)
  runner-up  the best score reachable by any WRONG function

A large gap means the target discriminates. A runner-up at or above what search
actually reaches means search is parked on a decoy, not on a hard problem.

Scores here are BEHAVIOURAL (substrates.nervous.scoring.score_contract). They exclude the
GA's loop-structure shaping bonus, which adds (1 - s) * LOOP_WEIGHT * bonus on
top — so a reported fitness of 0.8219 corresponds to s = 0.8125 here.

Usage:
    py tools/probe_trivial_baselines.py [--target NAME] [--max-inputs 3]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from substrates.snn.targets import TARGETS                          # noqa: E402
from substrates.nervous.targets import periodic_combinational_target     # noqa: E402
from substrates.nervous.scoring import score_contract, _combinational_windows  # noqa: E402


class SyntheticTraces(dict):
    """The minimum a scorer needs: per-role, per-trial event times.

    Bypassing the substrate is the point. These behaviours are reference
    YARDSTICKS, not evolved circuits, so they must not be able to fail for
    substrate reasons (bad I/O placement, a net that never settles).
    """

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.overflow = False


def _windows_with_inputs(trial, data_inputs=None):
    """(start, end, data_input_bits) per presented case window.

    A case-valid strobe occupies the LAST lane (targets.periodic_combinational_
    target writes ``streams[tick][data_inputs] = 1``), so the raw stimulus row is
    wider than the truth table for tables that need one. Slice it back down to
    the data bits, or every lookup silently misses and even the target's OWN
    truth table appears to score 0.5.

    NOTE a silent truth-table row (all inputs 0, no strobe) has no onset and
    therefore NO window — it is not presented at all. Reported explicitly.
    """
    out = []
    for start, end in _combinational_windows(trial):
        row = tuple(trial.streams[int(start)])
        if data_inputs is not None:
            row = row[:data_inputs]
        out.append((start, end, row))
    return out


def _emit(target, decide, latency=1.0):
    """Traces for a readout that fires once per window where ``decide`` is true."""
    data_inputs = getattr(target, 'combinational_data_inputs', None)
    events = {}
    for terminal in target.outputs:
        role = terminal.role
        per_trial = []
        for trial in target.trials:
            times = [start + latency
                     for start, _end, bits in _windows_with_inputs(
                         trial, data_inputs)
                     if decide(role, bits)]
            per_trial.append(times)
        events[role] = per_trial
    return SyntheticTraces(events)


def _score(target, decide):
    score, _cases, _alignment = score_contract(_emit(target, decide), target)
    return score


def _truth_lookup(target):
    """role -> {input_bits: expected_bit} from the target's own table."""
    roles = [terminal.role for terminal in target.outputs]
    table = {role: {} for role in roles}
    for input_bits, output_bits in target.combinational_cases:
        for role, bit in zip(roles, output_bits):
            table[role][tuple(input_bits)] = int(bit)
    return table


def probe(name, target, max_inputs):
    effective = periodic_combinational_target(target)
    data_inputs = getattr(effective, 'combinational_data_inputs', None)
    strobe = getattr(effective, 'combinational_strobe', False)
    presented = _windows_with_inputs(effective.trials[0], data_inputs)
    seen = sorted({bits for _s, _e, bits in presented})
    declared = sorted({tuple(b) for b, _o in effective.combinational_cases})
    missing = [row for row in declared if row not in seen]

    n = len(seen[0]) if seen else 0
    roles = [terminal.role for terminal in effective.outputs]
    table = _truth_lookup(effective)

    print('=' * 72)
    print('%s   inputs=%d outputs=%d  rows declared=%d presented=%d%s'
          % (name, n, len(roles), len(declared), len(seen),
             '  [case-valid strobe]' if strobe else ''))
    if missing:
        print('  !! NOT PRESENTED (silent, no input onset -> no window): %s'
              % ', '.join(str(r) for r in missing))

    # A role every presented window expects to be the SAME level has no negative
    # evidence left, so level-balancing has only one group to average and a
    # constant output scores a perfect 1.0. This is how OR came to "solve": its
    # only 0-row is the all-zero row, which the encoding drops as silent.
    one_level = []
    for terminal in effective.outputs:
        expected = set(effective.trials[0].expected_events[terminal.role])
        levels = {any(start <= e < end for e in expected)
                  for start, end, _bits in presented}
        if len(levels) < 2:
            one_level.append('%s=always%d'
                             % (terminal.role, 1 if True in levels else 0))

    correct = _score(effective, lambda role, bits: table[role].get(bits, 0) == 1)
    trivial = {}
    print('  correct truth table      %.4f' % correct)
    for label, decide in (
            ('silence', lambda role, bits: False),
            ('blanket (every window)', lambda role, bits: True)):
        trivial[label] = _score(effective, decide)
        print('  %-24s %.4f' % (label, trivial[label]))
    for index in range(n):
        label = 'echo input %d' % index
        trivial[label] = _score(
            effective, lambda role, bits, i=index: bits[i] == 1)
        print('  %-24s %.4f' % (label, trivial[label]))

    worst = max(trivial.items(), key=lambda kv: kv[1])
    if one_level:
        print('  ** DEGENERATE: no negative evidence (%s) — a constant output '
              'scores %.4f' % (', '.join(one_level), worst[1]))
    elif worst[1] >= correct:
        print('  ** DEGENERATE: trivial behaviour %r scores %.4f >= correct %.4f'
              % (worst[0], worst[1], correct))

    if n > max_inputs:
        print('  (skipping full enumeration: %d inputs > --max-inputs %d)'
              % (n, max_inputs))
        return

    # Best readout that is WRONG SOMEWHERE. Vary ONE role at a time and hold the
    # rest at truth: the aggregate is a mean over roles, so the strongest wrong
    # readout always perturbs a single role. Forcing every output to share one
    # function instead would understate multi-output targets badly (it scored a
    # Half adder ceiling of 0.50, below even silence).
    best_wrong, best_wrong_at = -1.0, None
    for role in roles:
        truth = {r: table[role].get(r, 0) for r in seen}
        for assignment in itertools.product((0, 1), repeat=len(seen)):
            candidate = dict(zip(seen, assignment))
            if candidate == truth:
                continue

            def decide(query_role, bits, role=role, candidate=candidate):
                if query_role == role:
                    return candidate.get(bits, 0) == 1
                return table[query_role].get(bits, 0) == 1

            score = _score(effective, decide)
            if score > best_wrong:
                best_wrong, best_wrong_at = score, (role, dict(candidate), truth)

    print('  best WRONG readout       %.4f' % best_wrong)
    if best_wrong_at is not None:
        role, candidate, truth = best_wrong_at
        wrong_rows = [r for r in seen if candidate[r] != truth[r]]
        print('    %s differs on: %s'
              % (role, ', '.join('%s(want %d got %d)'
                                 % (r, truth[r], candidate[r])
                                 for r in wrong_rows)))
    print('  discrimination gap       %.4f' % (correct - best_wrong))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', action='append',
                        help='probe only this target (repeatable)')
    parser.add_argument('--max-inputs', type=int, default=3,
                        help='skip full 2**(2**n) enumeration above this')
    args = parser.parse_args(argv)

    for name, target in TARGETS.items():
        if args.target and name not in set(args.target):
            continue
        if getattr(target, 'temporal', False) or not getattr(target, 'cases', ()):
            continue
        probe(name, target, args.max_inputs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
