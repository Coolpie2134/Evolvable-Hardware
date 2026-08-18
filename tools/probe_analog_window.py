"""
tools/probe_analog_window.py - characterise the paper analog node's timing.

Answers three questions about substrates/nervous/analog.py, deterministically and
without evolving anything, by driving hand-built circuits:

  1. COINCIDENCE WINDOW. How far apart may two excitatory edges arrive and still
     sum past threshold? Compare that with the node propagation delay, which is
     the smallest timing change adding or removing one cell can make. If the
     window is NARROWER than one hop, a two-input gate's input paths must have
     exactly equal hop counts and evolution cannot approach correct timing in
     steps - every gate is a cliff rather than a slope.
  2. MONOSTABLE WIDTH. How long does the output stay high after one firing?
     This is what a held-level contract reads, and it is set by the same
     constant.
  3. HELD-LEVEL SCORE. What a perfect hand-built AND gate actually scores on a
     held-level truth table, which is the encoding question separate from (1).

Usage:
    py tools/probe_analog_window.py                 # default tau sweep
    py tools/probe_analog_window.py --taus 1.1,2.2  # explicit
    py tools/probe_analog_window.py --json out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.config import nv_run_config
from substrates.nervous.branched import materialise_pads
from substrates.nervous.nervous import interpret_nervous
from substrates.nervous.targets import periodic_combinational_target
from substrates.nervous.temporal import (place_outputs_by_trace,
                                         run_nervous_events, score_contract)
from substrates.nervous.tritile import pack_channels
from substrates.snn.targets import TARGETS

#: One tri-tile whose three channels all coincidence-AND their L and R
#: neighbours: a perfect two-input AND gate, wired to two input pads.
AND_LR = 5
CELL = (1, 1)
PADS = [(0, 1), (2, 1)]
DEFAULT_TAUS = (1.10, 1.60, 2.20, 3.00, 4.50)


def _rig(config):
    """(body, routing, target, role) for the hand-built AND gate."""
    target = periodic_combinational_target(TARGETS['AND'])
    setattr(target, 'io_placement', 'fixed')
    setattr(target, 'pulse_config', config)
    body = materialise_pads({CELL: pack_channels(AND_LR, AND_LR, AND_LR)}, PADS)
    routing, _in_pos, _out = interpret_nervous(body, target, arch='tri3')
    return body, routing, target, target.outputs[0].role


def coincidence_window(config, limit=8.0, step=0.1):
    """Largest skew between two input edges that still fires the gate.

    Swept rather than solved analytically so it measures the ENGINE, not a
    formula that might disagree with it.
    """
    body, routing, target, role = _rig(config)
    best, skew, saturated = 0.0, 0.0, True
    while skew <= limit:
        _states, _traces, rise, _overflow = run_nervous_events(
            body, routing, list(PADS), {role: CELL}, [[0, 0]] * 20, 20,
            arch='tri3', config=config,
            input_events=[[(0.0, 1.0)], [(skew, 1.0)]])
        if not rise.get(CELL):
            saturated = False           # found the real edge of the window
            break
        best = skew
        skew = round(skew + step, 6)
    # Saturated means the gate still fired at the sweep limit, so the window is
    # ">= best", not "= best". Reporting the limit as a measurement would
    # understate a wide window and invent a plateau that is not there.
    return best, saturated


def monostable_width(config):
    """Predicted output pulse width: tau * ln((rest - v0) / (rest - release)).

    v0 is the node after a buffer's two coincident steps; release is the
    hysteretic re-arm level. Reported alongside the measured interval so a
    mismatch between model and engine is visible rather than assumed away.
    """
    rest, threshold = 1.0, config.analog_threshold
    step, hysteresis = config.analog_step, config.analog_hysteresis
    v0 = rest - 2 * step
    release = threshold + hysteresis
    return config.analog_tau_leak * math.log(
        (rest - v0) / (rest - release))


def held_level_score(config):
    """(score, first output interval) for the gate on a HELD-LEVEL AND table."""
    body, routing, target, role = _rig(config)
    _out, traces = place_outputs_by_trace(
        body, routing, list(PADS), target, arch='tri3',
        source_nodes={tuple(pad) for pad in PADS})
    if traces is None:
        return 0.0, None
    score, _cases, _ = score_contract(traces, target)
    intervals = traces.intervals[role][0]
    return score, (intervals[0] if intervals else None)


def sweep(taus=DEFAULT_TAUS):
    base = nv_run_config().pulse
    rows = []
    for tau in taus:
        config = dataclasses.replace(base, analog_tau_leak=float(tau))
        window, saturated = coincidence_window(config)
        score, interval = held_level_score(config)
        rows.append({
            'tau_leak': float(tau),
            'coincidence_window': window,
            'window_saturated': saturated,
            'node_delay': float(config.delay),
            # The headline: a window below 1.0 means one extra cell on either
            # input path silences the gate, and one cell is the smallest edit
            # available - so timing cannot be approached, only hit exactly.
            'hops_of_slack': window / float(config.delay),
            'monostable_width': monostable_width(config),
            'held_level_and_score': score,
            'held_level_interval': list(interval) if interval else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taus', default=None,
                        help='comma-separated tau_leak values')
    parser.add_argument('--json', default=None, help='write results here')
    args = parser.parse_args()
    taus = (tuple(float(v) for v in args.taus.split(','))
            if args.taus else DEFAULT_TAUS)
    rows = sweep(taus)

    print('paper analog node - timing characterisation')
    print('node propagation delay = one hop = %.2f ticks\n' % rows[0]['node_delay'])
    print('%8s  %10s  %10s  %10s  %12s' % (
        'tau_leak', 'coinc win', 'hops slack', 'pulse w', 'held-AND'))
    for row in rows:
        mark = '>=' if row['window_saturated'] else '  '
        print('%8.2f  %s%8.2f  %s%8.2f  %10.2f  %12.4f' % (
            row['tau_leak'], mark, row['coincidence_window'],
            mark, row['hops_of_slack'],
            row['monostable_width'], row['held_level_and_score']))
    print('\nhops slack < 1.00 means a two-input gate needs EXACTLY equal path')
    print('lengths: the smallest edit evolution can make (one cell, one hop)')
    print('is larger than the whole window, so timing is a cliff, not a slope.')
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({'rows': rows}, handle, indent=2)
        print('\nwrote %s' % args.json)


if __name__ == '__main__':
    main()
