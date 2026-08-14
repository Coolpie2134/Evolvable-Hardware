#!/usr/bin/env python3
"""
tools/fnv_solvable.py - which FNV targets are PHYSICALLY reachable, and why.

The FNV substrate is quiescent: every catalogue entry powers on low, there is no
free-running oscillator, and nothing produces an edge that is not caused by an
input. A target that demands an output while every input is silent therefore has
no implementation on this substrate - not a hard one, none. Evolution cannot do
better than the score that row costs, so benchmarking it measures the ceiling of
an impossible task and reports it as an evolution failure.

The check is the zero-input row of a static truth table:

    all inputs 0  ->  any output 1     =>  IMPOSSIBLE on FNV

The temporal twins are exempt. `coincident_temporal_target` adds a case-valid
STROBE lane exactly when the all-zero row carries evidence, so the circuit is
given a physical event on that row and the function becomes representable. That
is why "NAND" is impossible while "NAND (temporal)" is not.

    py tools/fnv_solvable.py                # human-readable table
    py tools/fnv_solvable.py --names        # solvable names, one per line
    py tools/fnv_solvable.py --csv          # comma-joined, for --targets
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.benchmark import targets_for_backend                    # noqa: E402


def zero_row_verdict(target):
    """(solvable, reason) from the target's own all-zero input row."""
    if getattr(target, 'temporal', False):
        # A temporal target is driven by explicit events. The wrapper adds a
        # strobe wherever silence could not carry the evidence, so there is
        # always something physical to respond to.
        return True, ''
    cases = getattr(target, 'cases', ()) or ()
    if not cases:
        return True, ''
    zero = [outputs for inputs, outputs in cases if not any(inputs)]
    if not zero:
        return True, 'no all-zero row is presented'
    if any(zero[0]):
        return False, ('the all-zero input row demands output %s, but a '
                       'quiescent substrate has nothing to drive it'
                       % (tuple(int(bit) for bit in zero[0]),))
    return True, ''


def classify(node_model='paper_analog'):
    """{name: (solvable, reason)} over the FNV catalogue."""
    catalogue = targets_for_backend('fnv', node_model)
    return {name: zero_row_verdict(target)
            for name, target in sorted(catalogue.items())}


def main():
    parser = argparse.ArgumentParser(
        description='List FNV targets that are physically reachable.')
    parser.add_argument('--names', action='store_true',
                        help='print solvable names one per line')
    parser.add_argument('--csv', action='store_true',
                        help='print solvable names comma-joined for --targets')
    parser.add_argument('--impossible', action='store_true',
                        help='print the excluded names instead')
    args = parser.parse_args()

    verdicts = classify()
    solvable = [name for name, (ok, _why) in verdicts.items() if ok]
    impossible = [name for name, (ok, _why) in verdicts.items() if not ok]
    chosen = impossible if args.impossible else solvable

    if args.csv:
        print(','.join(chosen))
        return
    if args.names:
        print('\n'.join(chosen))
        return

    print('FNV targets: %d total, %d solvable, %d physically impossible\n'
          % (len(verdicts), len(solvable), len(impossible)))
    print('EXCLUDED - no implementation exists on a quiescent substrate:')
    for name in impossible:
        print('  %-22s %s' % (name, verdicts[name][1]))
    print('\nINCLUDED (%d):' % len(solvable))
    for index in range(0, len(solvable), 3):
        print('  ' + '  '.join('%-24s' % n for n in solvable[index:index + 3]))


if __name__ == '__main__':
    main()
