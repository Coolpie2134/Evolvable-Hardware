"""
tools/divergence_screen.py — which targets can tell the node models apart?

The three nervous node-timing models differ in ONE thing: where a node's output
pulse width comes from (a global constant / its genes / the incoming pulse).
Delay is identical in all three. So on a target whose stimulus is all
base-width pulses and whose circuits never overlap pulses on a multi-input
node, the three models can produce *literally the same behaviour* — evolving on
it compares nothing, at full compute cost.

This screen finds out empirically, with NO evolution: score the same random
genomes under each model and count how often the score differs. Cheap (a few
hundred evaluations), and the output is one table, not a graph.

Read it as a filter for the real experiment:
  * "cannot discriminate" -> drop from the model comparison (keep as a general
    temporal benchmark if you like, but it is not evidence about width).
  * differs under WP  -> the target's stimulus/circuits carry width information.
  * differs under EW  -> intrinsic node width changes the outcome here.

    py tools/divergence_screen.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import random_hex_genome                      # noqa: E402
from nv_evo.ga import evaluate_nv_full                    # noqa: E402
from nv_evo.targets import TEMPORAL_TARGETS               # noqa: E402
from nv_evo.pulse import PulseConfig                      # noqa: E402
from nv_evo.genome import default_state_widths            # noqa: E402

N_GENOMES = 10
GENOME_SEED = 4
WIDTH_SEED = 9


def _fixed_width_vector():
    """One reproducible non-neutral width vector (what an evolved_width genome
    might look like mid-run). Deliberately mixes narrow and wide."""
    widths = default_state_widths()
    rng = random.Random(WIDTH_SEED)
    for state in range(1, len(widths)):
        widths[state] = rng.choice((0.5, 1.0, 1.0, 2.0, 3.0))
    return widths


def _score(genome, target, model, widths=None):
    genome.state_widths = widths
    setattr(target, 'pulse_config', PulseConfig(model=model))
    try:
        return evaluate_nv_full(genome, target)[0]
    finally:
        genome.state_widths = None
        if hasattr(target, 'pulse_config'):
            delattr(target, 'pulse_config')


def screen(target, genomes, wide):
    """(n_differ_width_preserving, n_differ_evolved_width) out of len(genomes)."""
    diff_wp = diff_ew = 0
    for genome in genomes:
        base = _score(genome, target, 'uniform')
        # 'pulse_delay' is the API id of the width-PRESERVING model
        wp = _score(genome, target, 'pulse_delay')
        ew = _score(genome, target, 'evolved_width', widths=list(wide))
        if abs(base - wp) > 1e-9:
            diff_wp += 1
        if abs(base - ew) > 1e-9:
            diff_ew += 1
    return diff_wp, diff_ew


def main():
    wide = _fixed_width_vector()
    rows = []
    for name, target in TEMPORAL_TARGETS.items():
        random.seed(GENOME_SEED)
        genomes = [random_hex_genome(2) for _ in range(N_GENOMES)]
        rows.append((name,) + screen(target, genomes, wide))

    print("%-40s %9s %9s" % ("target", "WP!=unif", "EW!=unif"))
    print("-" * 64)
    for name, diff_wp, diff_ew in rows:
        blind = '' if (diff_wp or diff_ew) else '  <- width-blind'
        print("%-40s %6d/%d %6d/%d%s"
              % (name, diff_wp, N_GENOMES, diff_ew, N_GENOMES, blind))

    blind = [name for name, dw, de in rows if not dw and not de]
    print("\n%d/%d targets are width-blind: every model scored every genome "
          "identically." % (len(blind), len(rows)))
    if blind:
        print("These cannot discriminate the models — drop them from the "
              "comparison:")
        for name in blind:
            print("   %s" % name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
