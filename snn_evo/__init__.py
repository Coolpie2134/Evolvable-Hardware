"""
snn_evo — Evolvable SNN package using Edwards indirect encoding.

Quick start
-----------
from snn_evo.ga import evolve
best_genome, fitness = evolve(generations=30, pop=50)
"""
from .genome  import Genome, Gene, Chromosome, random_genome
from .growth  import grow_snn
from .snn     import interpret_grid, circuit_summary, Neuron, Synapse
from .lif_sim import simulate
from .fitness import evaluate, LOGIC_CASES, CURRENT_HIGH, N_OUTPUTS, SEED_A, SEED_B
from .ga      import evolve
