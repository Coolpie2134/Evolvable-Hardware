"""
tools/probe_gradient_jitter.py - is the fitness gradient rough because the
ORGANISM changed, or merely because the output PROBE moved?

Whole-organism fitted probes answered one half of the old OUT_RADIUS objection
(lucky far-off cells inflating fitness - refuted by held-out certification). The
other half was never measured: that the selected probe JITTERS between similar
genomes and roughens the gradient.

Those are different defects with different fixes, and they are separable. For
each parent genome this probe scores every one-step mutant TWICE:

    frozen   - the mutant is read at the PARENT's probe cells;
    refitted - the mutant chooses its own probes over the whole organism.

Then:

    |frozen - parent|    is developmental ruggedness: the organism really did
                         change, read at a fixed place.
    |refitted - frozen|  is readout jitter: the same organism, scored at a
                         different place because refitting moved the probe.

If ruggedness dominates, the landscape is simply hard and smoothing the readout
would only hide it. If jitter dominates, the readout is adding noise selection
has to fight, and THAT is the case where proximity bias or hysteresis would be
worth considering. Nothing here changes behaviour; it measures.

    py tools/probe_gradient_jitter.py [--parents 6] [--mutants 12] [--verbose]

Determinism is checked first and treated as a hard precondition: if repeated
evaluation of one genome disagrees, every number below is meaningless.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.evaluation import fit_readout                # noqa: E402
from substrates.nervous.ga import evaluate_nv_full, mutate_nv        # noqa: E402
from substrates.nervous.ga import mutate_input_layout                # noqa: E402
from substrates.nervous.genome import random_hex_genome              # noqa: E402
from substrates.nervous.nervous import grow_nervous, node_delays     # noqa: E402
from substrates.nervous.io_placement import growth_seeds             # noqa: E402
from substrates.nervous.pulse import PulseConfig                     # noqa: E402
from substrates.nervous.scoring import score_contract                # noqa: E402
from substrates.nervous.targets import TEMPORAL_TARGETS              # noqa: E402
from substrates.nervous.temporal import (interpret_nervous,          # noqa: E402
                                         prepare_net, trace_fixed_outputs)


#: (target, architecture) pairs spanning combinational and loop-dependent work.
DEFAULT_BANK = (
    ('Coincidence (2-in)', 'tri3'),
    ('Coincidence (2-in)', 'single'),
    ('Veto gate', 'tri3'),
    ('Toggle flip-flop', 'tri3'),
    ('Toggle flip-flop', 'single'),
    ('SR latch', 'tri3'),
)


def _target(name, arch):
    import dataclasses
    target = dataclasses.replace(TEMPORAL_TARGETS[name])
    model = 'paper_analog' if arch == 'tri3' else 'pulse_delay'
    setattr(target, 'pulse_config', PulseConfig(model=model))
    return target


def _layout_genome(target, arch, chroms=2):
    genome = random_hex_genome(
        chroms, arch=arch, n_inputs=target.n_inputs,
        n_outputs=len(target.outputs), input_layout=True)
    genome.arch = arch
    return genome


def frozen_score(genome, target, probes):
    """Score ``genome`` read at someone else's probe cells."""
    arch = getattr(genome, 'arch', 'single')
    grid = grow_nervous(genome, seeds=growth_seeds(target, 'fixed', genome))
    pads = getattr(genome, 'input_layout', None)
    if not pads or any(pad not in grid for pad in pads):
        return None
    if any(cell not in grid for cell in probes.values()):
        return None
    routing, _in_pos, _ = interpret_nervous(grid, target, arch=arch)
    config = getattr(target, 'pulse_config', None)
    delays = None if arch == 'tri3' else node_delays(genome, grid, config)
    traces = trace_fixed_outputs(
        grid, routing, list(pads), dict(probes), target, delays=delays,
        arch=arch, source_nodes={tuple(cell) for cell in pads})
    if traces is None or getattr(traces, 'overflow', False):
        return None
    return score_contract(traces, target)[0]


def _one_step_mutants(genome, count, layout_share=0.5):
    """One allele edit, or one input-pad edge move."""
    out = []
    for index in range(count):
        child = mutate_nv(genome, 1.0, chromosome_count=len(genome.chromosomes))
        if index < int(count * layout_share):
            moved = mutate_nv(genome, 0.0,
                              chromosome_count=len(genome.chromosomes))
            if mutate_input_layout(moved):
                child = moved
        out.append(child)
    return out


def audit(parents=6, mutants=12, verbose=False, seed=20260728):
    print('determinism precondition: identical genomes must score identically')
    random.seed(seed)
    for name, arch in DEFAULT_BANK[:3]:
        target = _target(name, arch)
        genome = _layout_genome(target, arch)
        scores = {evaluate_nv_full(genome, target)[0] for _ in range(4)}
        assert len(scores) == 1, (name, arch, scores)
    print('  ok - repeated evaluation is exact\n')

    header = ('%-22s %-7s %7s %9s %9s %8s %8s'
              % ('target', 'arch', 'n', 'rugged', 'jitter', 'churn', 'invalid'))
    print(header)
    print('-' * len(header))
    totals = {'rugged': [], 'jitter': [], 'signed': [],
              'churn': 0, 'pairs': 0, 'invalid': 0}
    worst = []
    for name, arch in DEFAULT_BANK:
        target = _target(name, arch)
        rugged, jitter, churn, pairs, invalid = [], [], 0, 0, 0
        signed = []
        random.seed(seed + hash((name, arch)) % 9999)
        made = 0
        while made < parents:
            parent = _layout_genome(target, arch)
            prep = prepare_net(parent, target)
            if prep is None:
                continue
            made += 1
            parent_score = evaluate_nv_full(parent, target)[0]
            probes = {role: cell for role, cell in prep[3].items()
                      if cell is not None}
            for child in _one_step_mutants(parent, mutants):
                refit_prep = prepare_net(child, target)
                if refit_prep is None:
                    invalid += 1
                    continue
                refit = evaluate_nv_full(child, target)[0]
                held = frozen_score(child, target, probes)
                if held is None:
                    invalid += 1
                    continue
                pairs += 1
                rugged.append(abs(held - parent_score))
                jitter.append(abs(refit - held))
                signed.append(refit - held)
                child_probes = {role: cell
                                for role, cell in refit_prep[3].items()
                                if cell is not None}
                if child_probes != probes:
                    churn += 1
                worst.append((abs(refit - held), name, arch,
                              parent_score, held, refit))
        totals['signed'] += signed
        totals['rugged'] += rugged
        totals['jitter'] += jitter
        totals['churn'] += churn
        totals['pairs'] += pairs
        totals['invalid'] += invalid
        print('%-22s %-7s %7d %9.4f %9.4f %7.0f%% %7.0f%%'
              % (name, arch, pairs,
                 statistics.mean(rugged) if rugged else 0.0,
                 statistics.mean(jitter) if jitter else 0.0,
                 100.0 * churn / max(1, pairs),
                 100.0 * invalid / max(1, pairs + invalid)))
    print('-' * len(header))
    mean_rugged = statistics.mean(totals['rugged']) if totals['rugged'] else 0.0
    mean_jitter = statistics.mean(totals['jitter']) if totals['jitter'] else 0.0
    print('%-22s %-7s %7d %9.4f %9.4f %7.0f%% %7.0f%%'
          % ('ALL', '', totals['pairs'], mean_rugged, mean_jitter,
             100.0 * totals['churn'] / max(1, totals['pairs']),
             100.0 * totals['invalid']
             / max(1, totals['pairs'] + totals['invalid'])))
    print()
    print('developmental ruggedness  mean %.4f' % mean_rugged)
    print('readout jitter            mean %.4f' % mean_jitter)
    ratio = mean_jitter / mean_rugged if mean_rugged else float('inf')
    print('jitter / ruggedness       %.2f' % ratio)
    signed = totals['signed']
    rescued = sum(1 for value in signed if value > 1e-9)
    damaged = sum(1 for value in signed if value < -1e-9)
    print('refit ABOVE frozen        %d/%d (%.0f%%)   mean %+.4f'
          % (rescued, len(signed), 100.0 * rescued / max(1, len(signed)),
             statistics.mean(signed) if signed else 0.0))
    print('refit BELOW frozen        %d/%d (%.0f%%)'
          % (damaged, len(signed), 100.0 * damaged / max(1, len(signed))))
    print()
    # MAGNITUDE alone cannot separate the two things that move a probe:
    #   refitting RESCUES a mutant whose output genuinely moved   (refit > frozen)
    #   refitting INFLATES a mutant by finding a lucky cell       (refit > frozen)
    # Both look identical here. The sign does separate a third case: refitting
    # that ACTIVELY LOSES score is incoherent and would be a real defect.
    # Whether the rescues are honest was settled elsewhere, by held-out
    # certification - no OVERFIT verdicts, held-out within 0.02 of train.
    if damaged > 0.05 * max(1, len(signed)):
        print('VERDICT: refitting sometimes SCORES WORSE than the frozen probe.')
        print('  That is incoherent - the global assignment should dominate one')
        print('  fixed choice. Investigate before anything else.')
    elif ratio < 0.5:
        print('VERDICT: developmental ruggedness dominates. The landscape is')
        print('  hard; refitting is not what makes it hard.')
    else:
        print('VERDICT: the probe moves often, and refitting almost always')
        print('  RAISES the score - it is rescuing mutants whose output moved,')
        print('  not randomising them. Combined with the certification result')
        print('  (no OVERFIT, held-out within 0.02 of train) these are honest')
        print('  rescues, so this is NOT evidence for proximity bias or')
        print('  smoothing. A frozen probe would simply score those mutants 0.')
    if verbose and worst:
        worst.sort(reverse=True)
        print('\nworst refit-vs-frozen disagreements:')
        for delta, name, arch, parent, held, refit in worst[:8]:
            print('  %-22s %-7s parent %.3f  frozen %.3f  refit %.3f  (%+.3f)'
                  % (name, arch, parent, held, refit, refit - held))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=int, default=6)
    parser.add_argument('--mutants', type=int, default=12)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    return audit(args.parents, args.mutants, args.verbose)


if __name__ == '__main__':
    raise SystemExit(main())
