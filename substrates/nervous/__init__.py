"""
substrates/nervous — "nervous network" backend (Edwards, *Circuit Morphologies and
Ontogenies*, EH'02, Architecture 1).

Fully independent of substrates/snn: its own hex genome + growth, its own targets and
its own GA. The grown grid is interpreted as a nervous-net array, not an SNN:

  * each grown cell is a nervous-net node whose state routes its neighbours to
    two excitatory inputs (E1, E2) and one inhibitory input (I1);
  * a node fires when  (E1 op E2) AND NOT I1  — coincidence detection with an
    optional inhibitory veto. States 0-15 are the paper's Fig. 3 table (op = AND;
    each a buffer or an AND, the paper has no disjunction); states 16-31 are OR
    twins (op = OR, fire on either excitatory input) — a non-paper extension;
  * dynamics are asynchronous edge-triggered pulses (pulse.py): the paper model
    emits fixed-width pulses after fixed delay, while optional node models evolve
    width or preserve the incoming waveform with an evolved delay; inputs inject
    onto the perimeter nets (wired-OR). Nothing happens without an input, and
    memory / oscillation are a pulse circulating around a loop of buffers
    "until stopped by application of an inhibitory input" (delay-line memory);
  * temporal scoring retains raw continuous edge timestamps for point-event
    relations, uses cadence invariants for autonomous rhythms, and keeps sampled
    active/quiet windows only where they express persistence. The GA's fitness
    shaping favours nets whose signal graph can actually hold state.

Quick start
-----------
from substrates.nervous import evolve_nervous, TEMPORAL_TARGETS
best, fit = evolve_nervous(TEMPORAL_TARGETS['SR latch'], generations=60)
"""
from .hexgrid import hex_dirs, hex_pixel, ROUTING_HEX, routing_kind, node_fires
from .genome import (HexGene, Chromosome, Genome, random_hex_gene,
                     random_hex_chromosome, random_hex_genome)
from .pulse import PulseSim, DELAY, WIDTH, COINC, TICK
from .nervous import (ROUTING, SEED_STATE, interpret_nervous, evaluate_nervous,
                      score_nervous, nervous_truth_table, nervous_case_outputs,
                      circuit_summary_nervous, grow_nervous, grow_nervous_snapshots)
from .targets import (OutputTerminal, Trial, TemporalTarget, TEMPORAL_TARGETS,
                      periodic_combinational_target, with_io_placement,
                      spike_target, sr_latch, toggle_ff, oscillator, echo,
                      pattern_generator, coincidence_detector, one_shot,
                      pair_detector, temporal_xor, ordered_sequence, veto_gate,
                      burst_generator, divide_by_3)
from .io_placement import (IO_STRATEGIES, io_strategy, cell_tags, bind_io,
                           wiring_chromosome, seed_spatial_from_phenotype,
                           describe_binding)
from .oracle import (oracle_target, holdout_score, ORACLE_TARGETS, ORACLE_SPECS,
                     sample_streams, label_trace)
from .evaluation import FittedReadout, fit_readout, score_frozen
from .contracts import (BehaviorContract, Constraint, logic_contract,
                        event_contract, state_contract, interval_contract,
                        cadence_contract, cadence_step_contract,
                        bounded_state_contract, toggle_contract)
from .scoring import score_contract
from .temporal import (run_nervous, run_nervous_events, score_temporal, temporal_report,
                       prepare_net, windowed_score, exact_tick_accuracy,
                       place_outputs_by_trace, trace_fixed_outputs,
                       signal_graph, cycle_nodes,
                       loop_profile, TemporalTraces, event_score, cadence_score)
from .ga import (evaluate_nv, evaluate_nv_full, eval_batch_nv, eval_batch_cases,
                 mutate_nv, mutate_hex, crossover_nv, tournament_nv,
                 select_parent, next_population, evolve_nervous,
                 genome_signature, diversify)
