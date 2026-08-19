"""
substrates/lut - boolean-logic lookup (LUT) backend: the paper's Architecture 2
(Edwards EH'02 sections 5-7).

A square cellular array where each cell holds four directional 16-bit lookup
tables over its four neighbours' binary outputs. The dynamics are ASYNCHRONOUS level logic
(substrates.lut.pulse.AsyncLutSim): each cell is a logic element with a fixed
propagation delay, simulated event-driven in continuous time - the same
physical footing as the nervous net. The synchronous latched engine
(lut.LutSim) is retained as the reference its lattice behaviour quantises to.
Growth is the associative-memory ontogeny with 16-bit gene fields (Fig. 10),
run to a fixed point, 2-cycle, or the target's safety cap. It shares the project's problem definitions
(temporal + combinational targets) and trace-scoring maths; the substrate
and genome are its own.

Quick start
-----------
from substrates.lut import evolve_lut
from substrates.nervous import TEMPORAL_TARGETS
best, fit = evolve_lut(TEMPORAL_TARGETS['SR latch'], generations=60)
"""
from .genome import (LutGene, Chromosome, Genome, LUT_STATES,
                     input_layout_domain, input_layout_radius,
                     random_input_layout, random_edge_input_layout,
                     lut_input_positions, lut_exterior_edges,
                     lut_exterior_inputs, lut_growth_seeds, lut_io_mode,
                     random_lut_gene, random_lut_chromosome, random_lut_genome)
from .lut import SEED_STATE, grow_lut, grow_lut_snapshots, LutSim
from .pulse import LutConfig, AsyncLutSim
from .functions import (
    AND, MUX, OR, ROUTING, THRESHOLD, UNRESTRICTED, VETO, XOR,
    DEFAULT_FUNCTION_FAMILIES, FAMILY_TABLES, FUNCTION_FAMILIES,
    INPUT_TABLES, allowed_function_table, enabled_named_tables,
    mutate_function_table, normalise_function_families,
    project_function_table, random_function_table,
)
from .ga import (mutate_lut, crossover_lut,
                 tournament_lut, next_population, evolve_lut, lut_report,
                 prepare_lut, place_outputs_by_trace, trace_fixed_outputs,
                 score_lut_temporal,
                 score_lut_combinational, lut_case_outputs, lut_truth_table,
                 genome_signature, diversify, compact_genome,
                 constrain_genome_functions)
# The array and GA are usable headlessly.  Keep optional plotting from making a
# benchmark or worker process fail merely because the desktop-only Matplotlib
# dependency is absent from its runtime.
try:
    from .viz import draw_lut_net, draw_lut_table
except ModuleNotFoundError as _plot_error:
    if _plot_error.name != 'matplotlib':
        raise

    def _plotting_unavailable(*_args, _error=_plot_error, **_kwargs):
        raise RuntimeError(
            'LUT visualization requires the optional matplotlib dependency') \
            from _error

    draw_lut_net = draw_lut_table = _plotting_unavailable
from .boolfn import lut_sop, minterms, popcount, INPUT_NAMES
