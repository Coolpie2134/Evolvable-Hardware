"""
tests/test_source_pads.py - evolved input pads as external source terminals.

An input pad is an EXTERNAL SOURCE TERMINAL: internal activity can never raise
it, retrigger it, or alter its level; only its assigned external injection may
drive it. Its outgoing connections are untouched, so the pulse it receives still
propagates into the organism.

The rule is MEMBERSHIP, not state id. A cell is a source because it is in the
resolved input-pad set, never because it happens to express a particular state -
otherwise an ordinary evolved body cell expressing that state would silently
become an externally driven terminal in the middle of the organism.
"""
from __future__ import annotations

import dataclasses
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.genome import random_hex_genome           # noqa: E402
from substrates.nervous.hexgrid import (ROUTING_HEX,              # noqa: E402
                                        IO_STATE_INPUT, hex_dirs)
from substrates.nervous.io_placement import terminal_node_sets    # noqa: E402
from substrates.nervous.pulse import PulseConfig, TICK            # noqa: E402
from substrates.nervous.simulation import create_simulator        # noqa: E402
from substrates.nervous.targets import TEMPORAL_TARGETS           # noqa: E402
from substrates.nervous.temporal import prepare_net               # noqa: E402
from substrates.nervous.tritile import TriSim, pack_channels      # noqa: E402


def _buffer_towards(direction):
    """The routing state that buffers the named neighbour direction."""
    for state, (e1, e2, i1, _op) in enumerate(ROUTING_HEX):
        if e1 == direction and e2 == direction and i1 is None and state < 16:
            return state
    raise AssertionError('no buffer state for %r' % direction)


# -- single tile: the feedback-path proof --------------------------------------

def test_internal_feedback_cannot_raise_a_source_pad_single_tile():
    """A neighbour wired straight back INTO the pad must never raise it.

    Without source-only membership the pad is an ordinary node: the body cell
    it drives would drive it right back, and the pair would latch or ring.
    """
    pad = (0, 0)
    neighbours = hex_dirs(*pad)
    body = neighbours['R']
    back = next(d for d, cell in hex_dirs(*body).items() if cell == pad)
    grid = {pad: _buffer_towards('R'), body: _buffer_towards(back)}
    routing = {cell: ROUTING_HEX[state] for cell, state in grid.items()}

    # WITHOUT membership: the feedback path is live, so the pad is re-raised.
    loose = create_simulator(grid, routing, config=PulseConfig())
    loose.inject_pulse(pad, 0.0, 1.0)
    loose.advance_to(40 * TICK)
    assert len(loose.pulse_intervals[body]) >= 1, 'body never fired at all'
    loose_pad_pulses = len(loose.pulse_intervals[pad])

    # WITH membership: exactly the one external injection, forever.
    sourced = create_simulator(grid, routing, config=PulseConfig(),
                               input_nodes={pad})
    sourced.inject_pulse(pad, 0.0, 1.0)
    sourced.advance_to(40 * TICK)
    assert len(sourced.pulse_intervals[pad]) == 1
    assert loose_pad_pulses > 1, (
        'the feedback path must genuinely re-raise an unprotected pad, '
        'or this test proves nothing')


def test_external_injection_still_propagates_outward_from_a_source_pad():
    """Source-only silences INCOMING activation only; the pad still drives."""
    pad = (0, 0)
    body = hex_dirs(*pad)['R']
    back = next(d for d, cell in hex_dirs(*body).items() if cell == pad)
    grid = {pad: 0, body: _buffer_towards(back)}
    routing = {cell: ROUTING_HEX[state] for cell, state in grid.items()}
    sim = create_simulator(grid, routing, config=PulseConfig(),
                           input_nodes={pad})
    sim.inject_pulse(pad, 0.0, 1.0)
    sim.advance_to(20 * TICK)
    assert sim.pulse_intervals[pad], 'external injection must raise the pad'
    assert sim.pulse_intervals[body], 'the pad must still drive its neighbour'


def test_repeated_external_pulses_each_raise_the_pad():
    pad = (0, 0)
    grid = {pad: 0}
    routing = {pad: ROUTING_HEX[0]}
    sim = create_simulator(grid, routing, config=PulseConfig(),
                           input_nodes={pad})
    for start in (0.0, 5.0, 10.0):
        sim.inject_pulse(pad, start, 1.0)
    sim.advance_to(30 * TICK)
    assert len(sim.pulse_intervals[pad]) == 3


# -- membership, not state id --------------------------------------------------

def test_an_ordinary_cell_expressing_the_terminal_state_is_not_a_source():
    """The bug this guards: state 16 must not confer terminal semantics."""
    pad, other = (0, 0), (5, 5)
    body = hex_dirs(*other)['R']
    back = next(d for d, cell in hex_dirs(*body).items() if cell == other)
    grid = {pad: 0,
            other: IO_STATE_INPUT,          # same state as a dedicated terminal
            body: _buffer_towards(back)}
    routing = {cell: ROUTING_HEX[state & 0x1F] for cell, state in grid.items()}
    sim = create_simulator(grid, routing, config=PulseConfig(),
                           input_nodes={pad})
    # `other` is NOT in the source set, so it keeps ordinary node semantics: its
    # sources are built and it can be raised by its neighbour.
    assert other in sim.src
    assert pad not in sim.src


def test_terminal_sets_come_from_the_pad_set_and_are_empty_for_fixed_inputs():
    target = dataclasses.replace(TEMPORAL_TARGETS['Coincidence (2-in)'])
    random.seed(3)
    evolved = random_hex_genome(2, n_inputs=target.n_inputs, input_layout=True)
    probes = {'Q': (2, 2)}
    sources, sinks = terminal_node_sets(
        target, list(evolved.input_layout), probes, genome=evolved)
    assert sources == {tuple(cell) for cell in evolved.input_layout}
    assert sinks == set(), 'fitted outputs are probes, not sink terminals'

    legacy = random_hex_genome(2)
    assert terminal_node_sets(
        target, list(target.inputs), {}, genome=legacy) == (set(), set())
    # and with no genome at all - the historical call shape
    assert terminal_node_sets(target, list(target.inputs), {}) == (set(), set())


def test_tri3_source_tile_has_no_incoming_edges_and_still_drives():
    """A tri input tile collapses to one source-only IN sub-node.

    Its own three channels are not built at all, so no incoming activation can
    reach it, while neighbours read from the IN node so it still drives.
    """
    pad = (0, 0)
    body = hex_dirs(*pad)['R']
    back = next(d for d, cell in hex_dirs(*body).items() if cell == pad)
    buffer_state = _buffer_towards(back)
    grid = {pad: pack_channels(1, 1, 1),
            body: pack_channels(buffer_state, buffer_state, buffer_state)}
    sim = TriSim(grid, [pad], config=PulseConfig(model='paper_analog'))
    sim.inject_pulse(pad, 0.0, 1.0)
    sim.advance_to(40 * TICK)
    # exactly one raise of the pad tile: the external injection
    assert len(sim.pulse_intervals[pad]) == 1
    assert sim.pulse_intervals[body], 'the source tile must still drive out'


def test_tri3_only_the_named_tiles_become_sources():
    from substrates.nervous.tritile import interpret_tri
    pad, elsewhere = (0, 0), (3, 0)
    grid = {pad: pack_channels(1, 1, 1),
            elsewhere: pack_channels(1, 1, 1),
            hex_dirs(*pad)['R']: pack_channels(1, 1, 1)}
    info = interpret_tri(grid, [pad])
    assert set(info['in_nodes']) == {pad}
    # every other tile keeps its three ordinary channel sub-nodes
    assert len(info['tile_nodes'][elsewhere]) == 3
    assert len(info['tile_nodes'][pad]) == 1


# -- growth, survival and translation ------------------------------------------

def _analog_target(name='Coincidence (2-in)'):
    target = dataclasses.replace(TEMPORAL_TARGETS[name])
    setattr(target, 'pulse_config', PulseConfig(model='paper_analog'))
    return target


def test_growth_starts_from_the_exact_evolved_coordinates():
    from substrates.nervous.io_placement import growth_seeds
    target = _analog_target()
    random.seed(5)
    genome = random_hex_genome(
        2, arch='tri3', n_inputs=target.n_inputs, input_layout=True)
    genome.arch = 'tri3'
    assert growth_seeds(target, 'fixed', genome) == genome.input_layout


def test_an_invalid_layout_makes_the_phenotype_unbindable():
    target = _analog_target()
    random.seed(13)
    genome = random_hex_genome(
        2, arch='tri3', n_inputs=target.n_inputs, input_layout=True)
    genome.arch = 'tri3'
    genome.input_layout = ((1, 0), (2, 0))      # not anchored at the origin
    assert prepare_net(genome, target) is None


def test_translation_equivalent_layouts_develop_equivalently():
    """The anchor is a gauge - but only for PARITY-PRESERVING translations.

    ``hex_dirs`` reads L/R/D in the node's own orientation frame, and that frame
    flips with ``(x + y) % 2`` (the paper's context rotation). A shift with an
    ODD coordinate sum therefore mirrors every node's left/right and is a
    genuinely different organism, not the same one moved. Even-sum shifts are
    the real symmetry, and those must develop identically up to the shift.
    """
    from substrates.nervous.nervous import grow_nervous
    random.seed(17)
    genome = random_hex_genome(2, arch='tri3', n_inputs=2, input_layout=True)
    genome.arch = 'tri3'
    pads = genome.input_layout
    base = grow_nervous(genome, seeds=pads)
    for shift in ((2, 0), (0, -2), (-4, 6), (3, 1)):
        assert (shift[0] + shift[1]) % 2 == 0
        moved = tuple((x + shift[0], y + shift[1]) for x, y in pads)
        grown = grow_nervous(genome, seeds=moved)
        expected = {(x + shift[0], y + shift[1]): state
                    for (x, y), state in base.items()}
        assert grown == expected, shift


def test_an_odd_translation_is_a_different_organism_not_the_same_one_moved():
    """Pins the reason the anchor cannot simply be normalised away."""
    from substrates.nervous.nervous import grow_nervous
    random.seed(17)
    genome = random_hex_genome(2, arch='tri3', n_inputs=2, input_layout=True)
    genome.arch = 'tri3'
    pads = genome.input_layout
    base = grow_nervous(genome, seeds=pads)
    differs = False
    for shift in ((1, 0), (0, 1), (3, 0)):
        assert (shift[0] + shift[1]) % 2 == 1
        moved = tuple((x + shift[0], y + shift[1]) for x, y in pads)
        grown = grow_nervous(genome, seeds=moved)
        expected = {(x + shift[0], y + shift[1]): state
                    for (x, y), state in base.items()}
        differs = differs or grown != expected
    assert differs, 'odd shifts must be able to change the organism'


# -- training and held-out use identical pads ---------------------------------

def test_immigrants_match_the_population_layout_length():
    """An immigrant with the wrong pad count would be born unbindable."""
    from substrates.nervous.ga import next_population
    random.seed(31)
    population = [random_hex_genome(2, arch='tri3', n_inputs=3,
                                    input_layout=True) for _ in range(8)]
    for genome in population:
        genome.arch = 'tri3'
    bred = next_population(population, [0.3] * 8, chromosome_count=2)
    assert all(getattr(g, 'input_layout', None) is not None for g in bred)
    assert {len(g.input_layout) for g in bred} == {3}
