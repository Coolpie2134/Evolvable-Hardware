"""
nv_evo — "nervous network" backend (Edwards, *Circuit Morphologies and
Ontogenies*, EH'02, Architecture 1).

Fully independent of snn_evo: its own hex genome + growth, its own targets and
its own GA. The grown grid is interpreted as a nervous-net array, not an SNN:

  * each grown tile contains three independently configured nervous circuits,
    one driving each local L/R/D edge;
  * every circuit selects one of the paper's sixteen Figure-3 configurations
    and fires on (E1 AND E2) AND NOT I1. The tile state is three 4-bit selectors
    (12 bits total), with no shared output wire or OR extension;
  * dynamics are asynchronous edge-triggered pulses (pulse.py): a triggered
    node emits a fixed-width pulse after a fixed delay; inputs are injected
    onto the perimeter nets (wired-OR). Nothing happens without an input, and
    memory / oscillation are a pulse circulating around a loop of buffers
    "until stopped by application of an inhibitory input" (delay-line memory);
  * temporal scoring retains raw continuous edge timestamps for point-event
    relations, uses cadence invariants for autonomous rhythms, and keeps sampled
    active/quiet windows only where they express persistence. The GA's fitness
    shaping favours nets whose signal graph can actually hold state.

Quick start
-----------
from nv_evo import evolve_nervous, TEMPORAL_TARGETS
best, fit = evolve_nervous(TEMPORAL_TARGETS['SR latch'], generations=60)
"""
from .hexgrid import (hex_dirs, hex_pixel, ROUTING_HEX, PAPER_ROUTING,
                      routing_kind, node_fires, DIRECTIONS, TILE_ENCODING,
                      GENOME_ENCODING, CIRCUIT_STATE_COUNT, TILE_STATE_MASK,
                      pack_tile_state, unpack_tile_state, decode_tile_routing,
                      format_tile_state, parse_tile_state, tile_channel,
                      tile_channels, facing_channel, circuit_selector,
                      developmental_context, directional_connections)
from .genome import (HexGene, Chromosome, Genome, random_hex_gene,
                     random_hex_chromosome, random_hex_genome)
from .pulse import PulseSim, DirectionalPulseSim, DELAY, WIDTH, COINC, TICK
from .nervous import (ROUTING, SEED_STATE, interpret_nervous, evaluate_nervous,
                      score_nervous, nervous_truth_table, nervous_case_outputs,
                      circuit_summary_nervous, grow_nervous, grow_nervous_snapshots)
from .targets import (OutputTerminal, Trial, TemporalTarget, TEMPORAL_TARGETS,
                      spike_target, sr_latch, toggle_ff, oscillator, echo,
                      pattern_generator, coincidence_detector, one_shot,
                      pair_detector, temporal_xor, ordered_sequence, veto_gate,
                      burst_generator, divide_by_3)
from .oracle import (oracle_target, holdout_score, ORACLE_TARGETS, ORACLE_SPECS,
                     sample_streams, label_trace)
from .evaluation import FittedReadout, fit_readout, score_frozen
from .persistence import (sr_set_hold_oracle, sr_reset_oracle, sr_full_oracle,
                          evolve_sr_curriculum)
from .temporal import (run_nervous, run_nervous_events, score_temporal, temporal_report,
                       prepare_net, windowed_score, exact_tick_accuracy,
                       place_outputs_by_trace, trace_fixed_outputs,
                       signal_graph, cycle_nodes,
                       loop_profile, TemporalTraces, event_score, cadence_score,
                       score_temporal_bundle)
from .ga import (evaluate_nv, evaluate_nv_full, eval_batch_nv, eval_batch_cases,
                 mutate_nv, mutate_hex, crossover_nv, tournament_nv,
                 select_parent, next_population, evolve_nervous,
                 genome_signature, diversify)
