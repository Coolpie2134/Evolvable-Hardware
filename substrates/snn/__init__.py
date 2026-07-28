"""
substrates/snn — Evolvable SNN package using Edwards indirect encoding.

Quick start
-----------
from substrates.snn.ga import evolve
best_genome, fitness = evolve(generations=30, pop=50)
"""
from .genome  import Genome, Gene, Chromosome, random_genome
from .growth  import grow_snn, grow_snn_snapshots
from .snn     import (interpret_grid, circuit_summary, Neuron, Synapse,
                      Arch, DEFAULT_ARCH)
from .lif_sim import (simulate, simulate_events, simulate_trace,
                      DT, SIM_TIME, N_STEPS, SYN_DELAY)
from .visualize import draw_snn_net
from .fitness import evaluate, score, LOGIC_CASES, CURRENT_HIGH, N_OUTPUTS, SEED_A, SEED_B
from .temporal import (temporal_arch, prepare_snn_temporal,
                       score_snn_temporal, snn_temporal_report)
from .targets import (Target, OutputTerminal, TARGETS, DEFAULT_TARGET,
                      get_target, gate_target, adder_target, truth_table_target)
from .ga      import evolve
