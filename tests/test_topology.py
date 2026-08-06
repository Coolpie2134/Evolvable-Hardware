"""
tests/test_topology.py - structural selection and the gradient-jitter audit.

Topology is the FINAL ranking tier: it prefers organisms with more usable
hardware, and it must be blind to the task. The safety argument is entirely
positional - `viability > fitness > robustness > juvenile > topology` - so a
structurally rich wrong answer can never outrank a correct one.

Also pinned here: the audit invariants from the gradient-jitter probe, and the
drift guard keeping the FNV and Nervous aggregation formulas identical.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import substrates.nervous.ga as nv_ga                              # noqa: E402
from substrates.nervous.genome import random_hex_genome            # noqa: E402
from substrates.nervous.hexgrid import hex_dirs                    # noqa: E402
from substrates.nervous.pulse import PulseConfig                   # noqa: E402
from substrates.nervous.targets import TEMPORAL_TARGETS            # noqa: E402
from substrates.topology import EMPTY, Topology, measure           # noqa: E402


# -- the aggregation -----------------------------------------------------------

def test_every_count_earns_log1p_credit_with_diminishing_returns():
    assert Topology().score == 0.0
    one = Topology(reachable_nodes=1).score
    ten = Topology(reachable_nodes=10).score
    hundred = Topology(reachable_nodes=100).score
    assert one < ten < hundred                       # more always helps
    # Diminishing returns means each ADDITIONAL node is worth less than the
    # one before it - not that a ten-fold jump is worth less than the previous
    # ten-fold jump, which is false for any increasing function.
    first = Topology(reachable_nodes=2).score - one
    later = hundred - Topology(reachable_nodes=99).score
    assert later < first
    assert math.isclose(one, math.log1p(1))


def test_connectivity_and_feedback_both_contribute():
    connected = Topology(reachable_nodes=5, reachable_edges=6)
    looped = Topology(reachable_nodes=5, reachable_edges=6, cyclic_nodes=4,
                      loop_rank=1, loop_regions=1)
    assert looped.score > connected.score


def test_the_fnv_and_nervous_aggregation_formulas_cannot_drift():
    """One arithmetic, two physical graph extractors.

    The extractors MUST differ (a tri channel is not an FNV component port),
    but if the aggregation drifts the substrates stop being comparable.
    """
    from substrates.fnv.evaluation import FunctionalTopology
    values = dict(reachable_nodes=7, reachable_edges=11, integrating_nodes=2,
                  cyclic_nodes=5, loop_rank=3, loop_regions=2)
    shared, functional = Topology(**values), FunctionalTopology(**values)
    assert shared.connectivity == functional.connectivity
    assert shared.feedback == functional.feedback
    assert shared.score == functional.score


# -- what counts, and what does not --------------------------------------------

def test_unreachable_structure_earns_nothing():
    """A perfect ring in a corner no input can write to is not hardware."""
    ring = [((10, 0), (11, 0)), ((11, 0), (12, 0)), ((12, 0), (10, 0))]
    orphan = measure(ring, sources=[(0, 0)])
    assert orphan == EMPTY
    assert orphan.score == 0.0
    # the same ring, now driven by a pad, does count
    reached = measure(ring + [((0, 0), (10, 0))], sources=[(0, 0)])
    assert reached.cyclic_nodes == 3 and reached.loop_regions == 1


def test_a_source_pad_has_no_incoming_edge_and_cannot_join_a_cycle():
    """Pads are source-only, so an arrow pointing back at one is not a wire."""
    feedback = [((0, 0), (1, 0)), ((1, 0), (0, 0))]
    topology = measure(feedback, sources=[(0, 0)])
    assert topology.cyclic_nodes == 0
    assert topology.loop_regions == 0
    assert topology.reachable_nodes == 1          # the body cell only


def test_integrating_nodes_need_two_logical_inputs():
    shared_edges = [((0, 0), (5, 5)), ((1, 1), (5, 5))]
    both = measure(shared_edges, sources=[(0, 0), (1, 1)])
    assert both.integrating_nodes == 1
    single = measure([((0, 0), (5, 5))], sources=[(0, 0)])
    assert single.integrating_nodes == 0


def test_loop_rank_counts_independent_cycles():
    ring = [((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (1, 0))]
    driven = [((0, 0), (1, 0))] + ring
    assert measure(driven, sources=[(0, 0)]).loop_rank == 1
    # a chord across the ring adds one independent cycle
    assert measure(driven + [((1, 0), (3, 0))],
                   sources=[(0, 0)]).loop_rank == 2


# -- tri3 must be measured at channel level ------------------------------------

def test_tri3_channels_in_one_tile_do_not_invent_a_loop():
    """Three electrically separate circuits share a tile coordinate.

    Measured at TILE level they would collapse into one node and manufacture a
    self-loop; measured at channel level they stay distinct.
    """
    tile = (4, 4)
    channel_edges = [((0, 0), (tile[0], tile[1], 'L')),
                     ((tile[0], tile[1], 'R'), (tile[0], tile[1], 'D'))]
    channels = measure(channel_edges, sources=[(0, 0)])
    assert channels.cyclic_nodes == 0
    assert channels.loop_regions == 0
    # the same wiring flattened onto the tile WOULD look like a self-loop
    flattened = measure([((0, 0), tile), (tile, tile)], sources=[(0, 0)])
    assert flattened.cyclic_nodes == 1


def test_nervous_tri3_topology_uses_subnodes():
    from substrates.nervous.nervous import grow_nervous, interpret_nervous
    from substrates.nervous.io_placement import growth_seeds
    from substrates.nervous.temporal import nervous_topology
    target = dataclasses.replace(TEMPORAL_TARGETS['Toggle flip-flop'])
    setattr(target, 'pulse_config', PulseConfig(model='paper_analog'))
    random.seed(5)
    for _ in range(30):
        genome = random_hex_genome(2, arch='tri3', n_inputs=target.n_inputs,
                                   input_layout=True)
        genome.arch = 'tri3'
        grid = grow_nervous(genome, seeds=growth_seeds(target, 'fixed', genome))
        if len(grid) < 6:
            continue
        routing, _in_pos, _ = interpret_nervous(grid, target, arch='tri3')
        topology = nervous_topology(grid, routing, list(genome.input_layout),
                                    arch='tri3')
        # Channel expansion gives up to three sub-nodes per non-input tile, so
        # a tile-level measurement could never exceed the tile count.
        assert topology.reachable_nodes > len(grid) - len(genome.input_layout)
        return
    raise AssertionError('no tri3 organism grew large enough')


# -- ranking -------------------------------------------------------------------

def test_topology_is_the_last_tier_and_never_outranks_fitness():
    rich, poor = random_hex_genome(2), random_hex_genome(2)
    rich._topology_score, poor._topology_score = 25.0, 0.0
    assert nv_ga.rank_key(rich, 0.5) > nv_ga.rank_key(poor, 0.5)
    assert nv_ga.rank_key(rich, 0.5) < nv_ga.rank_key(poor, 0.5 + 1e-9)
    for better, worse in (('_robustness', 0.9), ('_juvenile_score', 0.9)):
        setattr(poor, better, worse)
        assert nv_ga.rank_key(rich, 0.5) < nv_ga.rank_key(poor, 0.5)
        setattr(poor, better, 0.0)


def test_no_nervous_rank_term_prefers_fewer_genes_or_shorter_telomeres():
    """Parsimony is gone: telomeres still bound growth, but not selection."""
    from substrates.nervous.genome import Chromosome
    small = random_hex_genome(2)
    large = nv_ga.clone_genome(small)
    for chromosome in large.chromosomes:
        chromosome.genes = chromosome.genes * 3
        chromosome.telomere = min(20, chromosome.telomere + 5)
    assert nv_ga.rank_key(small, 1.0) == nv_ga.rank_key(large, 1.0)
    assert nv_ga.rank_key(small, 0.5) == nv_ga.rank_key(large, 0.5)
    assert len(nv_ga.rank_key(small, 1.0)) == 5
    del Chromosome


def test_topology_survives_a_cache_hit():
    """A cached genome must rank exactly as a freshly evaluated one."""
    record = (0.5, None, (2, 2), 0.0, None, Topology(reachable_nodes=4), 7.25)
    genome = random_hex_genome(2)
    nv_ga.record_escape_objectives(genome, record)
    assert genome._topology_score == 7.25
    assert genome._topology.reachable_nodes == 4
    fresh = random_hex_genome(2)
    nv_ga.record_escape_objectives(fresh, record)
    assert nv_ga.rank_key(genome, 0.5) == nv_ga.rank_key(fresh, 0.5)


def test_topology_is_blind_to_the_target():
    """Same organism, different target: identical structural measurement."""
    from substrates.nervous.objectives import structural_topology
    random.seed(9)
    genome = random_hex_genome(2, arch='tri3', n_inputs=2, input_layout=True)
    genome.arch = 'tri3'
    scores = set()
    for name in ('Coincidence (2-in)', 'Veto gate'):
        target = dataclasses.replace(TEMPORAL_TARGETS[name])
        setattr(target, 'pulse_config', PulseConfig(model='paper_analog'))
        scores.add(round(structural_topology(genome, target).score, 9))
    assert len(scores) == 1


# -- the gradient-jitter audit -------------------------------------------------

def test_evaluation_is_exactly_deterministic():
    """The audit's precondition: without this every number it prints is noise."""
    target = dataclasses.replace(TEMPORAL_TARGETS['Toggle flip-flop'])
    setattr(target, 'pulse_config', PulseConfig(model='paper_analog'))
    random.seed(17)
    for _ in range(5):
        genome = random_hex_genome(2, arch='tri3', n_inputs=target.n_inputs,
                                   input_layout=True)
        genome.arch = 'tri3'
        scores = {nv_ga.evaluate_nv_full(genome, target)[0] for _ in range(4)}
        assert len(scores) == 1


def test_refitting_a_probe_does_not_lose_score():
    """Refitting picks the globally best assignment, so it cannot score BELOW a
    single frozen choice by more than the placement/contract objective gap.

    A systematic loss here would mean the assignment is choosing badly - the one
    outcome that would make refitting incoherent rather than merely noisy.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    from probe_gradient_jitter import _layout_genome, _target, frozen_score
    from substrates.nervous.temporal import prepare_net
    target = _target('Veto gate', 'tri3')
    random.seed(23)
    losses = comparisons = 0
    for _ in range(40):
        parent = _layout_genome(target, 'tri3')
        prep = prepare_net(parent, target)
        if prep is None:
            continue
        probes = {role: cell for role, cell in prep[3].items()
                  if cell is not None}
        child = nv_ga.mutate_nv(parent, 1.0, chromosome_count=2)
        if prepare_net(child, target) is None:
            continue
        held = frozen_score(child, target, probes)
        if held is None:
            continue
        comparisons += 1
        if nv_ga.evaluate_nv_full(child, target)[0] < held - 1e-9:
            losses += 1
    assert comparisons, 'no comparable pairs'
    assert losses <= 0.2 * comparisons, (
        'refitting lost score on %d/%d pairs' % (losses, comparisons))
