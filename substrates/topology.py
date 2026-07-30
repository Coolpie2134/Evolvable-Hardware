"""
substrates/topology.py — target-agnostic computational structure.

Selection needs a final tie-break that prefers organisms with more usable
hardware, and it must be blind to the task: no truth table, expected trace,
target name, fitted output, component family, gene count or telomere may reach
it. Otherwise it stops being a structural preference and becomes a second,
weaker copy of the fitness function.

This module owns the Nervous AGGREGATION and graph measurement. FNV keeps a
parallel result type and physical graph extractor because its component-port
wiring is different; ``tests/test_topology.py`` pins the two aggregation
formulas to identical arithmetic. LUT and SNN do not use this final topology
tier.

The measurements, all counted only over hardware REACHABLE FROM THE SOURCE PADS:

    reachable_nodes     non-input nodes an input can actually drive
    reachable_edges     directed wires between reachable nodes
    integrating_nodes   nodes reachable from two or more logical inputs —
                        where information can combine at all
    cyclic_nodes        nodes inside a directed cycle
    loop_rank           independent cycles (cyclomatic, per component)
    loop_regions        distinct strongly connected feedback regions

Unreachable structure scores nothing. An organism can grow a beautiful ring in
a corner no input can write to; it is not computational hardware, and crediting
it would reward bulk over connection.

Each count is aggregated with ``log1p`` so more connectivity and more feedback
always help, with diminishing returns — the first loop is worth far more than
the twentieth, and no organism can win on size alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    """Structural measurements of one organism's reachable hardware."""

    reachable_nodes: int = 0
    reachable_edges: int = 0
    integrating_nodes: int = 0
    cyclic_nodes: int = 0
    loop_rank: int = 0
    loop_regions: int = 0

    @property
    def connectivity(self):
        return (math.log1p(self.reachable_nodes)
                + math.log1p(self.reachable_edges)
                + math.log1p(self.integrating_nodes))

    @property
    def feedback(self):
        return (math.log1p(self.cyclic_nodes)
                + math.log1p(self.loop_rank)
                + math.log1p(self.loop_regions))

    @property
    def score(self):
        return self.connectivity + self.feedback

    def summary(self):
        return ('nodes %d, edges %d, integrating %d, cyclic %d, '
                'loop-rank %d, regions %d -> %.3f'
                % (self.reachable_nodes, self.reachable_edges,
                   self.integrating_nodes, self.cyclic_nodes,
                   self.loop_rank, self.loop_regions, self.score))


EMPTY = Topology()


def reachable_from(start, adjacency):
    reached, pending = {start}, [start]
    while pending:
        source = pending.pop()
        for destination in adjacency.get(source, ()):
            if destination not in reached:
                reached.add(destination)
                pending.append(destination)
    return reached


def strong_components(nodes, adjacency):
    """Iterative Kosaraju decomposition.

    Iterative rather than recursive on purpose: a large telomere can grow an
    organism deep enough to blow the interpreter's recursion limit, and an
    evaluation must not fail because a genome grew well.
    """
    nodes = set(nodes)
    ordered = {node: tuple(sorted(destination
                                  for destination in adjacency.get(node, ())
                                  if destination in nodes))
               for node in nodes}
    visited, finish_order = set(), []
    for start in sorted(nodes):
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for destination in reversed(ordered[node]):
                if destination not in visited:
                    stack.append((destination, False))
    reverse = {node: set() for node in nodes}
    for source, destinations in ordered.items():
        for destination in destinations:
            reverse[destination].add(source)
    assigned, components = set(), []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component, pending = set(), [start]
        assigned.add(start)
        while pending:
            node = pending.pop()
            component.add(node)
            for source in reverse[node]:
                if source not in assigned:
                    assigned.add(source)
                    pending.append(source)
        components.append(component)
    return components


def measure(edges, sources, nodes=None):
    """Aggregate one substrate's EFFECTIVE wiring into a :class:`Topology`.

    ``edges`` is an iterable of ``(source, destination)`` physical connections
    already resolved by the substrate — for tri3 that means channel/sub-node
    keys, so three electrically separate circuits inside one tile cannot create
    a path or a loop through each other. ``sources`` are the resolved input
    pads; they have outgoing edges but no incoming effective edges, so a pad can
    never appear inside a cycle merely because something points at it.
    """
    source_set = {tuple(cell) if isinstance(cell, (list, tuple)) else cell
                  for cell in sources}
    adjacency = {}
    for source, destination in edges:
        # A pad has no incoming effective edge: external injection is the only
        # thing that drives it, so an arrow pointing back at it is not a wire.
        if destination in source_set:
            continue
        adjacency.setdefault(source, set()).add(destination)
        adjacency.setdefault(destination, set())
    for node in (nodes or ()):
        adjacency.setdefault(node, set())

    per_input = [reachable_from(cell, adjacency)
                 for cell in source_set if cell in adjacency]
    reachable = set().union(*per_input) if per_input else set()
    reachable_nodes = reachable - source_set
    reachable_edges = sum(
        1 for source, destinations in adjacency.items()
        if source in reachable
        for destination in destinations
        if destination in reachable_nodes)
    integrating = sum(1 for cell in reachable_nodes
                      if sum(cell in reached for reached in per_input) >= 2)

    cyclic_nodes = loop_rank = loop_regions = 0
    for component in strong_components(reachable_nodes, adjacency):
        internal = sum(destination in component
                       for source in component
                       for destination in adjacency.get(source, ()))
        cyclic = (len(component) > 1
                  or any(node in adjacency.get(node, ())
                         for node in component))
        if not cyclic:
            continue
        loop_regions += 1
        cyclic_nodes += len(component)
        loop_rank += max(1, internal - len(component) + 1)

    return Topology(reachable_nodes=len(reachable_nodes),
                    reachable_edges=reachable_edges,
                    integrating_nodes=integrating,
                    cyclic_nodes=cyclic_nodes,
                    loop_rank=loop_rank,
                    loop_regions=loop_regions)
