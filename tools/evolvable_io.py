"""
tools/evolvable_io.py — A/B experiment: does EVOLVABLE I/O binding help?

By default the substrate wires I/O geometrically (inputs = seed pads in declared
order; outputs = trace-fitted near their terminals). This experiment compares
that 'fixed' baseline against the two evolvable strategies the request asked for:

  * tag_rank       — attach ports to distinct highest-priority cells in order.
  * wiring_chromosome — chromosome three maps each port to a node type and one
                     ordinal instance in a stable shuffled list of matches.

  * spatial_chromosome: chromosome three maps each port to a normalised x/y
                     anchor. Inputs are developmental seeds; outputs claim the
                     nearest available living cell.

Everything else is held constant (same target, grid, contract, seeds, gens, pop),
so the only variable is HOW ports attach to cells. For each strategy we run a
seeded evolution and report train fitness plus HELD-OUT certification when the
target has a reference oracle (fresh stimulus the circuit never trained on) —
the honest test of whether an evolved binding generalises or just memorised.

    py tools/evolvable_io.py                     # default: a few quick targets
    py tools/evolvable_io.py "Echo (delay 3)"    # quote names with spaces
    py tools/evolvable_io.py --gens 40 --pop 80 --seeds 1,2,3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.ga import evolve_nervous                         # noqa: E402
from substrates.nervous.targets import (TEMPORAL_TARGETS, with_io_placement,  # noqa: E402
                            ORACLE_KEY_TO_SPEC)
from substrates.nervous.io_placement import bind_io, cell_tags, io_strategy  # noqa: E402
from substrates.nervous.nervous import grow_nervous                      # noqa: E402
from substrates.nervous.evaluation import fit_readout                    # noqa: E402
from substrates.nervous.oracle import holdout_score, ORACLE_SPECS        # noqa: E402

STRATEGIES = (
    'fixed', 'terminal_nodes', 'tag_rank', 'wiring_chromosome',
    'spatial_chromosome')
DEFAULT_TARGETS = ['Echo (delay 3)', 'Coincidence (2-in)', 'Toggle flip-flop']


def _binding_summary(genome, target):
    """One compact line describing where the evolved ports landed."""
    from substrates.nervous.io_placement import binding_report, growth_seeds
    if io_strategy(target) == 'fixed':
        return 'inputs=seed pads, outputs=trace-fitted'
    grid = grow_nervous(
        genome, seeds=growth_seeds(target, io_strategy(target), genome))
    report = binding_report(genome, grid, target)
    if report is None:
        return 'unbindable (organism has no live cells)'
    parts = []
    for entry in report:
        cells = entry['cells']
        cell_str = (str(cells[0]) if len(cells) == 1
                    else '%d sites' % len(cells))
        if io_strategy(target) == 'tag_rank':
            allele = 'priority%d' % entry['tag']
        elif io_strategy(target) == 'spatial_chromosome':
            allele = 'anchor(%.3f,%.3f)' % entry['anchor']
        else:
            allele = 'type%d/selector%d' % (
                entry['type'], entry['selector'])
        parts.append('%s=%s@%s' % (entry['port'], allele, cell_str))
    return '  '.join(parts)


def _held_out(genome, target_name, target):
    """Mean held-out fitness over fresh oracle seeds, or None if not applicable.

    The evolved binding — output cells, alignment, AND the tag-chosen input cells
    — is fitted ONCE on the strategy target, then frozen while score_frozen scores
    fresh validation stimulus. This is what proves an evolved binding generalises
    rather than memorising one schedule."""
    spec_name = ORACLE_KEY_TO_SPEC.get(target_name)
    if spec_name is None:
        return None
    spec = ORACLE_SPECS.get(spec_name)
    if spec is None:
        return None
    fitted = fit_readout(genome, target, backend='nervous')
    if fitted is None:
        return 0.0
    scores = [holdout_score(genome, spec, backend='nervous', seed=s,
                            fitted=fitted, physics_from=target)
              for s in (101, 202, 303)]
    return sum(scores) / len(scores)


def run(target_name, gens, pop, n_chroms, seeds, strategies=STRATEGIES):
    base = TEMPORAL_TARGETS.get(target_name)
    if base is None:
        raise SystemExit('unknown target %r (see TEMPORAL_TARGETS)' % target_name)
    print('\n=== %s   (gens=%d pop=%d n_chroms=%d seeds=%s) ==='
          % (target_name, gens, pop, n_chroms, ','.join(map(str, seeds))))
    print('%-16s  %-7s  %-9s  %s' % ('strategy', 'train', 'held-out', 'binding'))
    print('-' * 78)
    for strategy in strategies:
        target = with_io_placement(base, strategy)
        trains, helds = [], []
        best_genome = None
        for seed in seeds:
            g, fit = evolve_nervous(target, generations=gens, pop=pop,
                                    n_chroms=n_chroms, seed=seed, verbose=False)
            trains.append(fit)
            ho = _held_out(g, target_name, target)
            if ho is not None:
                helds.append(ho)
            if best_genome is None or fit >= max(trains):
                best_genome = g
        train_mean = sum(trains) / len(trains)
        held_str = ('%.3f' % (sum(helds) / len(helds))) if helds else '   -  '
        print('%-16s  %-7.3f  %-9s  %s' % (
            strategy, train_mean, held_str,
            _binding_summary(best_genome, target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('targets', nargs='*', default=None,
                    help='target names (default: a few quick ones)')
    ap.add_argument('--gens', type=int, default=20)
    ap.add_argument('--pop', type=int, default=50)
    ap.add_argument('--n-chroms', type=int, default=3)
    ap.add_argument('--seeds', default='1,2',
                    help='comma-separated RNG seeds (one evolution each)')
    ap.add_argument(
        '--strategies', default=','.join(STRATEGIES),
        help='comma-separated binding strategies to compare')
    args = ap.parse_args()
    if args.n_chroms < 3:
        ap.error('--n-chroms must be at least 3 because the comparison includes '
                 'the wiring-chromosome strategy')
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    strategies = [
        strategy.strip() for strategy in args.strategies.split(',')
        if strategy.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        ap.error('unknown strategies: %s' % ', '.join(sorted(unknown)))
    targets = args.targets or DEFAULT_TARGETS
    for name in targets:
        run(name, args.gens, args.pop, args.n_chroms, seeds, strategies)


if __name__ == '__main__':
    main()
