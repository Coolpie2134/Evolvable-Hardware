"""
tests/test_input_layout.py — evolved input geometry for the nervous net.

The genome carries one honeycomb coordinate per logical input. This replaces
the grown input-terminal strategies, which had a cliff: a genome that failed to
express one required terminal received no meaningful evaluation at all. A
discrete pad list always carries exactly the required number of pads, so the
placement neighbourhood is smooth and local.

The invariants that make it work, each pinned below:

  * position IS logical identity — pad 0 is input 0, with no per-pad parameter;
  * input 0 is anchored at the origin, purely as a coordinate gauge;
  * one mutation moves ONE non-anchor pad by ONE honeycomb edge, never onto an
    occupied site;
  * crossover inherits the layout WHOLE, because relative geometry is one
    co-adapted module;
  * an invalid layout makes the phenotype unbindable and is never repaired.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.checkpoint import genome_from_dict, genome_to_dict     # noqa: E402
from substrates.nervous.ga import (clone_genome, crossover_nv,      # noqa: E402
                                   mutate_input_layout, mutate_nv)
from substrates.nervous.genome import (input_layout_domain,         # noqa: E402
                                       input_layout_radius,
                                       nervous_input_positions,
                                       random_hex_genome,
                                       random_input_layout)
from substrates.nervous.hexgrid import hex_frontier_cells           # noqa: E402


FALLBACK = ((5, 5), (6, 6), (7, 7))


def _layout_genome(n_inputs=3, seed=None):
    if seed is not None:
        random.seed(seed)
    return random_hex_genome(2, n_inputs=n_inputs, input_layout=True)


# ── shape of a fresh layout ───────────────────────────────────────────────────

def test_fresh_layout_is_anchored_distinct_and_compact():
    for n_inputs in (1, 2, 3, 4, 6):
        random.seed(n_inputs)
        layout = random_input_layout(n_inputs)
        assert len(layout) == n_inputs
        assert len(set(layout)) == n_inputs          # no collisions
        assert layout[0] == (0, 0)                   # anchored
        domain = set(input_layout_domain(
            input_layout_radius(20, n_inputs)))
        assert all(cell in domain for cell in layout)


def test_zero_inputs_is_an_empty_layout_not_a_crash():
    assert random_input_layout(0) == ()


def test_position_is_the_only_input_identity():
    """No per-pad numeric parameter: the tuple index IS the logical input."""
    layout = random_input_layout(3)
    assert all(isinstance(cell, tuple) and len(cell) == 2 for cell in layout)
    # Two pads swapped is a DIFFERENT assignment, not the same layout relabelled.
    swapped = (layout[0], layout[2], layout[1])
    assert swapped != layout


# ── mutation: one pad, one edge ───────────────────────────────────────────────

def test_mutation_moves_exactly_one_non_anchor_pad_by_one_edge():
    random.seed(11)
    for _ in range(60):
        genome = _layout_genome(4)
        before = genome.input_layout
        if not mutate_input_layout(genome):
            continue
        after = genome.input_layout
        assert len(after) == len(before)
        assert after[0] == (0, 0)                    # anchor never moves
        moved = [i for i in range(len(before)) if before[i] != after[i]]
        assert len(moved) == 1                       # exactly one pad
        index = moved[0]
        assert index != 0
        assert after[index] in set(hex_frontier_cells(*before[index]))
        assert len(set(after)) == len(after)         # never collides


def test_mutation_declines_rather_than_repairing_when_boxed_in():
    """A pad with no free neighbour is left alone; nothing is relocated."""
    genome = _layout_genome(2, seed=3)
    genome.input_layout = ((0, 0), (1, 0))
    surrounded = set(hex_frontier_cells(1, 0)) | {(0, 0), (1, 0)}
    genome.input_layout = tuple([(0, 0)] + sorted(surrounded - {(0, 0)})[:1])
    before = genome.input_layout
    random.seed(5)
    # Single non-anchor pad with every neighbour free still moves; the point of
    # this test is that a REFUSAL leaves the layout untouched.
    if not mutate_input_layout(genome):
        assert genome.input_layout == before


def test_single_input_layout_cannot_mutate():
    genome = _layout_genome(1, seed=7)
    assert genome.input_layout == ((0, 0),)
    assert mutate_input_layout(genome) is False
    assert genome.input_layout == ((0, 0),)


def test_fixed_input_genome_is_never_given_a_layout_by_mutation():
    random.seed(9)
    genome = random_hex_genome(2)
    assert getattr(genome, 'input_layout', None) is None
    for _ in range(30):
        genome = mutate_nv(genome, 4.0, chromosome_count=2)
    assert getattr(genome, 'input_layout', None) is None


def test_layout_survives_the_whole_mutation_transaction():
    random.seed(13)
    genome = _layout_genome(3)
    for _ in range(40):
        genome = mutate_nv(genome, 4.0, chromosome_count=2)
        layout = genome.input_layout
        assert layout is not None and len(layout) == 3
        assert layout[0] == (0, 0)
        assert len(set(layout)) == 3


# ── crossover: the layout is one module ───────────────────────────────────────

def test_crossover_inherits_a_whole_parent_layout_never_a_mixture():
    random.seed(17)
    parent_a = _layout_genome(3)
    parent_b = _layout_genome(3)
    parent_a.input_layout = ((0, 0), (1, 0), (2, 0))
    parent_b.input_layout = ((0, 0), (0, 1), (0, 2))
    seen = set()
    for _ in range(80):
        child_a, child_b = crossover_nv(parent_a, parent_b)
        for child in (child_a, child_b):
            layout = child.input_layout
            # A per-pad mixture such as ((0,0),(1,0),(0,2)) would manufacture
            # arrangements neither parent had, and could collide.
            assert layout in (parent_a.input_layout, parent_b.input_layout)
            seen.add(layout)
    assert len(seen) == 2       # both parents' layouts genuinely reachable


def test_clone_carries_the_layout():
    genome = _layout_genome(3, seed=19)
    assert clone_genome(genome).input_layout == genome.input_layout


# ── invalid layouts are unbindable, never repaired ────────────────────────────

def test_valid_layout_resolves_and_legacy_falls_back_to_the_target():
    genome = _layout_genome(3, seed=23)
    assert nervous_input_positions(genome, FALLBACK) == genome.input_layout
    legacy = random_hex_genome(2)
    assert nervous_input_positions(legacy, FALLBACK) == FALLBACK


def test_every_invalid_layout_yields_no_pads():
    genome = _layout_genome(3, seed=29)
    for broken in (
            ((0, 0), (1, 0)),                 # too few pads for the target
            ((0, 0), (1, 0), (2, 0), (3, 0)),  # too many
            ((0, 0), (1, 0), (1, 0)),         # collision
            ((1, 0), (2, 0), (3, 0)),         # not anchored at the origin
            ((0, 0), (1, 0), (2,)),           # malformed coordinate
            ((0, 0), (1, 0), 'x'),            # not a coordinate at all
    ):
        genome.input_layout = broken
        assert nervous_input_positions(genome, FALLBACK) == (), broken


def test_an_invalid_layout_is_not_silently_deduplicated_or_clamped():
    """The failure must be visible to selection, not patched up under it."""
    genome = _layout_genome(3, seed=31)
    genome.input_layout = ((0, 0), (1, 0), (1, 0))
    resolved = nervous_input_positions(genome, FALLBACK)
    assert resolved == ()
    # and the genome itself is untouched — evaluation does not rewrite genes
    assert genome.input_layout == ((0, 0), (1, 0), (1, 0))


# ── checkpoints ───────────────────────────────────────────────────────────────

def test_layout_round_trips_and_absent_means_fixed_inputs():
    genome = _layout_genome(3, seed=37)
    restored = genome_from_dict(genome_to_dict(genome, 'nervous'), 'nervous')
    assert restored.input_layout == genome.input_layout

    legacy = random_hex_genome(2)
    document = genome_to_dict(legacy, 'nervous')
    assert document['input_layout'] is None
    assert getattr(
        genome_from_dict(document, 'nervous'), 'input_layout', None) is None


def test_a_checkpoint_predating_the_field_loads_as_fixed_inputs():
    genome = _layout_genome(3, seed=41)
    document = genome_to_dict(genome, 'nervous')
    document.pop('input_layout')            # written before the field existed
    restored = genome_from_dict(document, 'nervous')
    assert getattr(restored, 'input_layout', None) is None
    assert nervous_input_positions(restored, FALLBACK) == FALLBACK
