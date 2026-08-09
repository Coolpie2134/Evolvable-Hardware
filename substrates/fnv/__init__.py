"""Functional NV Net: fixed physical functions on a directed honeycomb."""

from .catalogue import (
    BY_ID, BY_NAME, CATALOGUE_HASH, COMPONENTS, DEFAULT_FAMILIES, DIRECTIONS,
    FAMILIES, NODE_TYPE_DICTIONARY, ComponentType, component,
    enabled_component_ids,
    local_component_ids, normalise_families, state_distance,
    verify_catalogue_hash,
)
from .simulation import (
    FunctionalSim, effective_wiring_edges, facing_direction, source_for_input,
)
from .genome import (
    Chromosome, ContextGene, Genome, InputGene, OutputGene, OUT_STATE,
    PAD_STATE, functional_input_positions, functional_output_positions,
    input_layout_domain, input_layout_radius, input_seed_grid,
    input_seed_state, is_branched, random_functional_genome,
    random_input_layout, resolve_output_layout, sync_output_layout,
)
from .construction import (
    branch_growth_order, develop_constructive, grow_functional,
    grow_functional_snapshots)
from .evaluation import (
    FunctionalTopology, evaluate_functional_full, functional_case_outputs,
    functional_logic_horizon, functional_report, functional_topology,
    native_component_baseline,
    prepare_functional, run_functional_events, score_functional,
)
from .playback import (
    FunctionalPlayer, functional_case_pulses, prepare_functional_playback,
)
from .ga import (
    clone_genome, crossover_functional, mutate_functional,
)

__all__ = [
    "BY_ID", "BY_NAME", "CATALOGUE_HASH", "COMPONENTS", "DEFAULT_FAMILIES",
    "DIRECTIONS", "FAMILIES", "NODE_TYPE_DICTIONARY",
    "Chromosome", "ComponentType", "ContextGene", "InputGene", "OutputGene",
    "FunctionalPlayer", "FunctionalSim", "FunctionalTopology", "Genome",
    "clone_genome", "component",
    "crossover_functional",
    "effective_wiring_edges", "enabled_component_ids", "facing_direction",
    "functional_topology", "local_component_ids",
    "evaluate_functional_full", "functional_case_outputs",
    "functional_case_pulses", "functional_logic_horizon", "functional_report",
    "native_component_baseline", "prepare_functional",
    "branch_growth_order", "develop_constructive", "grow_functional",
    "grow_functional_snapshots",
    "normalise_families",
    "OUT_STATE", "PAD_STATE", "functional_input_positions",
    "functional_output_positions", "input_layout_domain",
    "input_layout_radius", "input_seed_grid", "input_seed_state",
    "is_branched",
    "mutate_functional", "prepare_functional_playback",
    "random_functional_genome", "resolve_output_layout",
    "run_functional_events", "sync_output_layout",
    "random_input_layout", "score_functional",
    "source_for_input", "state_distance",
    "verify_catalogue_hash",
]
