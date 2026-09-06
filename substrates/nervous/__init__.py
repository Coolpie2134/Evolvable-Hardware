"""Developmental nervous nets with analog tri-circuit physics.

The current output-rooted encoding grows three independently routed analog
circuits per tile. Shared low-level growth, observation and scoring utilities
remain available for reference tests; retired digital engines are not shipped.
"""
from .hexgrid import hex_dirs, hex_pixel, ROUTING_HEX, routing_kind, node_fires
from .genome import (HexGene, Chromosome, Genome, random_hex_gene,
                     random_hex_chromosome, random_hex_genome)
from .pulse import DELAY, WIDTH, COINC, TICK
from .nervous import (ROUTING, SEED_STATE, interpret_nervous, evaluate_nervous,
                      score_nervous, nervous_truth_table, nervous_case_outputs,
                      circuit_summary_nervous, grow_nervous, grow_nervous_snapshots)
from .targets import (OutputTerminal, Trial, TemporalTarget, TEMPORAL_TARGETS,
                      periodic_combinational_target, with_io_placement,
                      spike_target, sr_latch, toggle_ff, oscillator, echo,
                      pattern_generator, coincidence_detector, one_shot,
                      temporal_xor, ordered_sequence, veto_gate,
                      burst_generator)
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
