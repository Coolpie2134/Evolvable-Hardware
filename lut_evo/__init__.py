"""
lut_evo — boolean-logic lookup (LUT) backend: the paper's Architecture 2
(Edwards EH'02 §5-7).

A square cellular array where each cell is a 16-bit lookup table over its
four neighbours' binary outputs; the system is latched and synchronous.
Growth is the associative-memory ontogeny with 16-bit gene fields (Fig. 10),
run to a stable attractor. Shares the project's problem definitions
(temporal + combinational targets) and trace-scoring maths; the substrate
and genome are its own.

Quick start
-----------
from lut_evo import evolve_lut
from nv_evo import TEMPORAL_TARGETS
best, fit = evolve_lut(TEMPORAL_TARGETS['SR latch'], generations=60)
"""
from .genome import (LutGene, Chromosome, Genome, LUT_STATES,
                     random_lut_gene, random_lut_chromosome, random_lut_genome)
from .lut import SEED_STATE, grow_lut, grow_lut_snapshots, LutSim
from .ga import (evaluate_lut, eval_batch_lut, mutate_lut, crossover_lut,
                 tournament_lut, next_population, evolve_lut, lut_report,
                 prepare_lut, place_outputs_by_trace, trace_fixed_outputs,
                 score_lut_temporal,
                 score_lut_combinational, lut_case_outputs, lut_truth_table,
                 genome_signature, diversify, compact_genome)
from .viz import draw_lut_net, draw_lut_table
from .boolfn import (lut_sop, lut_summary, truth_string, karnaugh, minterms,
                     popcount, lut_class_char, INPUT_NAMES)
