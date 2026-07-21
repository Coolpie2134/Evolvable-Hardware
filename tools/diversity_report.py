"""
tools/diversity_report.py — characterise a SOLVED population.

Fitness spread is zero once everyone solves, so this reports structure instead:
the four-level collapse funnel (nv_evo/diversity.py) and, optionally, the
mutational-robustness panel.

    py tools/diversity_report.py results/solver_generation.json
    py tools/diversity_report.py results/solver_generation.json --robustness
    py tools/diversity_report.py results/solver_generation.json --samples 16

The input is a population checkpoint written by the controller's diversify
phase (``save_population``). Measure the GA's own post-solve population and the
diversify() output SEPARATELY: diversify enforces rule-signature uniqueness by
construction, so its genotype counts describe the tool, not the substrate.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo_runtime.checkpoint import load_checkpoint          # noqa: E402
from nv_evo import diversity as dv                          # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Diversity funnel for a solved population.')
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
                     'population (results/solver_generation.json)' % args.path)
    genomes = state['genomes']
    if args.limit:
        genomes = genomes[:args.limit]
    target = state['target']
    backend = state['backend']
    config = state.get('run_config')
    valid = args.valid if args.valid is not None else state.get('valid', 0.999)

    probe = dv.make_probe_bank(target, seed=args.probe_seed)
    report = dv.diversity_funnel(genomes, backend, target, config,
                                 probe_target=probe)
    report.probe_seed = args.probe_seed
    print(dv.format_report(report, population=len(genomes),
                           target_name=target.name, valid=valid))

    if args.robustness:
        print()
        rob = dv.robustness(genomes, backend, target, config,
                            samples=args.samples, valid=valid, seed=args.seed)
        print(dv.format_robustness(rob))


if __name__ == '__main__':
    main()
