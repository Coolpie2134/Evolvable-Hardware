"""
snn_evo — Evolvable SNN package using Edwards indirect encoding.

Quick start
-----------
from snn_evo.ga import evolve
best_genome, fitness = evolve(generations=30, pop=50)
"""
from .genome  import Genome, Gene, Chromosome, random_genome
from .growth  import grow_snn, grow_snn_snapshots
from .snn     import (interpret_grid, circuit_summary, Neuron, Synapse,
                      Arch, DEFAULT_ARCH)
from .lif_sim import simulate
from .fitness import evaluate, score, LOGIC_CASES, CURRENT_HIGH, N_OUTPUTS, SEED_A, SEED_B
from .targets import (Target, OutputTerminal, TARGETS, DEFAULT_TARGET,
                      get_target, gate_target, adder_target, truth_table_target)
from .ga      import evolve
