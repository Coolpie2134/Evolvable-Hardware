"""
tools/diversity_report.py - characterise an evaluated population.

Fitness spread is zero once everyone solves, so this reports structure instead:
the four-level collapse funnel (substrates/nervous/diversity.py) and, optionally, the
mutational-robustness panel.

    py tools/diversity_report.py results/latest_population.json
    py tools/diversity_report.py results/solver_generation.json --robustness
    py tools/diversity_report.py results/latest_population.json --samples 16

The input is a population checkpoint written by the controller. Measure the
GA's own ``latest_population.json`` and the post-solve diversify() output
SEPARATELY: diversify enforces rule-signature uniqueness by construction, so
its genotype counts describe the tool, not the substrate.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.checkpoint import load_checkpoint          # noqa: E402
from substrates.nervous import diversity as dv                          # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Diversity funnel for an evaluated population.')
    parser.add_argument('path', help='population checkpoint JSON')
    parser.add_argument('--robustness', action='store_true',
                        help='also sample the mutational neighbourhood '
                             '(slow: samples x population growths)')
    parser.add_argument('--samples', type=int, default=8,
                        help='mutants sampled per genome (default 8)')
    parser.add_argument('--valid', type=float, default=None,
                        help="validity threshold (default: the file's)")
    parser.add_argument('--seed', type=int, default=4242,
                        help='robustness sampling seed (default 4242)')
    parser.add_argument('--probe-seed', type=int, default=dv.PROBE_SEED,
                        help='off-spec probe bank seed (default %d); a '
                             'different seed is a different measurement'
                             % dv.PROBE_SEED)
    parser.add_argument('--limit', type=int, default=None,
                        help='characterise only the first N genomes')
    args = parser.parse_args(argv)

    state = load_checkpoint(args.path)
    if 'genomes' not in state:
        parser.error('%s is a single-genome checkpoint; this tool wants a '
                     'population checkpoint' % args.path)
    genomes = state['genomes']
    if args.limit:
        genomes = genomes[:args.limit]
    if not genomes:
        parser.error('%s contains 0 genomes; use the latest fully evaluated '
                     'population (results/latest_population.json)' % args.path)
    target = state['target']
    backend = state['backend']
    config = state.get('run_config')
    valid = args.valid if args.valid is not None else state.get('valid', 0.999)

    probe = dv.make_probe_bank(target, seed=args.probe_seed)
    report = dv.diversity_funnel(genomes, backend, target, config,
                                 probe_target=probe)
    report.probe_seed = args.probe_seed
    fitnesses = state.get('fitnesses')
    if fitnesses is not None and len(fitnesses) >= len(genomes):
        fitnesses = fitnesses[:len(genomes)]
        metadata = state.get('metadata') or {}
        print('Snapshot: %s, try %s, generation %s' % (
            metadata.get('status', 'unknown'), metadata.get('try', '?'),
            metadata.get('generation', '?')))
        print('Fitness: min %.4f   mean %.4f   max %.4f   valid %d/%d '
              '(>= %.3f)\n' % (
                  min(fitnesses), sum(fitnesses) / len(fitnesses),
                  max(fitnesses), sum(value >= valid for value in fitnesses),
                  len(fitnesses), valid))
    print(dv.format_report(report, population=len(genomes),
                           target_name=target.name, valid=valid))

    if args.robustness:
        print()
        rob = dv.robustness(genomes, backend, target, config,
                            samples=args.samples, valid=valid, seed=args.seed)
        print(dv.format_robustness(rob))


if __name__ == '__main__':
    main()
