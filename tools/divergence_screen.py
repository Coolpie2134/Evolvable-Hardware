"""
tools/divergence_screen.py - which targets can tell the node models apart?

The nervous node models differ in ONE thing: where a node's output pulse width
comes from (a global constant, or the incoming pulse). Delay is identical in
both. So on a target whose stimulus is all base-width pulses and whose circuits
never overlap pulses on a multi-input node, the models can produce *literally
the same behaviour* - evolving on it compares nothing, at full compute cost.

This screen finds out empirically, with NO evolution: score the same random
genomes under each model and count how often the score differs. Cheap (a few
hundred evaluations), and the output is one table, not a graph.

Read it as a filter for the real experiment:
  * "width-blind" -> drop from the model comparison (keep as a general temporal
    benchmark if you like, but it is not evidence about width).
  * differs under WP -> the target's stimulus/circuits carry width information.

(A third model made the emitted width a per-node-type genome vector. Width
evolution has been retired, so this screen now compares the two survivors.)

    py tools/divergence_screen.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# stdout is read on the Windows console, whose cp1252 code page cannot encode
# target names like "Temporal sum (dA + dB)"; never let that kill the report
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

from substrates.nervous import random_hex_genome                      # noqa: E402
from substrates.nervous.ga import evaluate_nv_full                    # noqa: E402
from substrates.nervous.targets import TEMPORAL_TARGETS               # noqa: E402
from substrates.nervous.pulse import PulseConfig                      # noqa: E402

N_GENOMES = 10
GENOME_SEED = 4


def _score(genome, target, model):
    """Score one genome under one node model, restoring the target afterwards.

    The target's previous pulse_config is put back rather than deleted: these
    are shared registry objects, and stripping an attribute the registry set
    would leak into every later run in the process.
    """
    had = hasattr(target, 'pulse_config')
    previous = getattr(target, 'pulse_config', None)
    setattr(target, 'pulse_config', PulseConfig(model=model))
    try:
        return evaluate_nv_full(genome, target)[0]
    finally:
        if had:
            setattr(target, 'pulse_config', previous)
        else:
            delattr(target, 'pulse_config')


def screen(target, genomes):
    """How many of `genomes` score differently under width-preserving transport
    than under the paper's fixed-width node."""
    differ = 0
    for genome in genomes:
        base = _score(genome, target, 'uniform')
        # 'pulse_delay' is the API id of the width-PRESERVING model
        wp = _score(genome, target, 'pulse_delay')
        if abs(base - wp) > 1e-9:
            differ += 1
    return differ


def main():
    rows = []
    for name, target in TEMPORAL_TARGETS.items():
        random.seed(GENOME_SEED)
        genomes = [random_hex_genome(2) for _ in range(N_GENOMES)]
        rows.append((name, screen(target, genomes)))

    print('%-44s %12s' % ('target', 'WP != uniform'))
    print('-' * 60)
    for name, differ in rows:
        blind = '' if differ else '   <- width-blind'
        print('%-44s %8d/%d%s' % (name, differ, N_GENOMES, blind))

    blind = [name for name, differ in rows if not differ]
    print('\n%d/%d targets are width-blind: both models scored every genome '
          'identically.' % (len(blind), len(rows)))
    if blind:
        print('These cannot discriminate the models -- drop them from the '
              'comparison:')
        for name in blind:
            print('   %s' % name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
