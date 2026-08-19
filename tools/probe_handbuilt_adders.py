"""Construct exact Full Adder circuits without evolution.

This probe separates three questions that a GA benchmark otherwise mixes:

* can the LUT physics implement the truth table?
* can the paper-analog nervous-node physics implement the edge-timed table?
* how much structure must mutation discover before either output is useful?

The LUT circuit is the repository's reverse-compiled, grown spatial genome.
The nervous circuit is an explicit acyclic network of the same Fig. 1 nodes
used by a live ``analog_tri`` run.  It is deliberately described as a source
graph first: that tests the node physics and behavioural contract without
claiming that the current developmental encoding can easily lay it out.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

from runtime.config import nv_run_config
from substrates.lut.ga import evaluate_lut_full
from substrates.lut.synthesis import synthesize_combinational_genome
from substrates.nervous.hexgrid import ROUTING_HEX
from substrates.nervous.scoring import TemporalTraces, _obs_len, score_contract
from substrates.nervous.simulation import create_simulator
from substrates.nervous.targets import coincident_temporal_target
from substrates.snn.targets import get_target


def handbuilt_lut_full_adder():
    """Return the exact grown LUT genome and its contract score."""
    from substrates.nervous.targets import periodic_combinational_target

    target = periodic_combinational_target(get_target('Full adder'))
    target.io_placement = 'spatial_chromosome'
    result = synthesize_combinational_genome(
        target, chromosome_count=3, max_telomere=8)
    score, cases = evaluate_lut_full(result.genome, target)
    return {
        'score': score,
        'worst_case': min(cases),
        'cells': len(result.grid),
        'genes': sum(len(chromosome.genes)
                     for chromosome in result.genome.chromosomes),
        'inverse_exact': bool(result.inverse_report['exact']),
        'genome': result.genome,
        'target': target,
    }


def _compile_branched_lut_body(genome, wanted, wanted_owners):
    """Reverse-grow one exact, already arm-partitioned LUT body.

    This is intentionally a tiny feasibility compiler, not a rescue operator.
    It observes the exact context at each currently reachable missing cell and
    emits a tolerance-zero, depth-specific rule.  A rule is retained only when
    replaying development from scratch adds desired cells and no others.
    """
    from substrates.lut.branched import (
        DEPTH_BANDS, DIRECTIONS, _cell_of, arm_reach,
        develop_branched_lut, growth_candidates, neighbours,
        output_root_sites)
    from substrates.lut.branched_ga import input_pads
    from substrates.lut.branched import LutContextGene

    pads = input_pads(genome)
    roots = output_root_sites(genome, pads)
    output_sites = set(roots.values()) - set(pads)

    def is_prefix(trace):
        return all(cell in wanted and wanted[cell] == state
                   for cell, state in trace.grid.items())

    for _round in range(len(wanted) + 4):
        trace = develop_branched_lut(genome, pads)
        if not is_prefix(trace):
            raise ValueError('branched compiler produced an unintended cell')
        if trace.grid == wanted:
            return trace
        candidates = set(growth_candidates(
            trace.grid, trace.owners, set(pads), output_sites))
        accepted = False
        for cell in sorted(set(wanted) - set(trace.grid)):
            if cell not in candidates:
                continue
            label = wanted_owners[cell]
            depth = arm_reach(cell, label, trace.owners, trace.depths)
            if depth is None:
                continue
            around = neighbours(cell)
            context = tuple(
                _cell_of(around[direction], trace.grid, set(pads), output_sites)
                for direction in DIRECTIONS) + (
                    _cell_of(cell, trace.grid, set(pads), output_sites),)
            chromosome = genome.chromosomes[(label - 1) // 2]
            gene = LutContextGene(
                genome.next_gene_id,
                context[0], context[1], context[2], context[3], context[4],
                wanted[cell], label, min(int(depth), DEPTH_BANDS - 1))
            chromosome.genes.append(gene)
            trial = develop_branched_lut(genome, pads)
            if (is_prefix(trial) and len(trial.grid) > len(trace.grid)):
                genome.next_gene_id += 1
                accepted = True
                break
            chromosome.genes.pop()
        if not accepted:
            raise ValueError('no exact context can extend the branched body')
    raise ValueError('branched compiler did not converge')


def handbuilt_branched_lut_full_adder():
    """A perfect Full Adder in the live output-rooted LUT genome.

    Sum's root touches all three pads directly.  Carry uses a separate mirrored
    arm and four relay cells, satisfying the encoding's exclusive-territory
    rule instead of relying on the shared hub used by the native compiler.
    """
    from substrates.lut.branched import (
        OUT_CELL, BranchedLutChromosome, BranchedLutGenome, LutContextGene,
        LutControlGene, LutInputGene, LutIoChromosome, LutOutputGene,
        table_support)
    from substrates.lut.functions import FAMILY_TABLES, INPUT_TABLES
    from substrates.lut.branched_ga import input_pads
    from substrates.nervous.targets import periodic_combinational_target

    xor3 = next(table for table in FAMILY_TABLES['XOR']
                if table_support(table) == {'N', 'S', 'W'})
    majority3 = next(table for table in FAMILY_TABLES['THRESHOLD']
                     if table_support(table) == {'N', 'S', 'E'})
    relay_e = INPUT_TABLES['E']
    sum_state = (xor3, 0, 0, 0)
    carry_state = (majority3, 0, 0, 0)
    relay_w_from_e = (0, 0, 0, relay_e)
    relay_s_from_e = (0, relay_e, 0, 0)
    relay_n_from_e = (relay_e, 0, 0, 0)

    genome = BranchedLutGenome(
        chromosomes=[BranchedLutChromosome(
            genes=[
                LutContextGene(
                    1, self_in=OUT_CELL, self_out=sum_state, branch_id=1),
                LutContextGene(
                    2, self_in=OUT_CELL, self_out=carry_state, branch_id=2),
            ],
            controls=[LutControlGene(tolerance=0, telomere=8),
                      LutControlGene(tolerance=0, telomere=8)])],
        io_chromosome=LutIoChromosome(
            # square_ring(2)[1] == (1, 1), [7] == (1, -1).
            inputs=[LutInputGene(bearing=1, distance=2),
                    LutInputGene(bearing=7, distance=2)],
            # square_ring(1)[0] == (1, 0), [2] == (-1, 0).
            outputs=[LutOutputGene('Sum', 0, 1, 1),
                     LutOutputGene('Carry', 2, 1, 2)]),
        families=('ROUTING', 'XOR', 'THRESHOLD'), next_gene_id=3)

    assert input_pads(genome) == ((0, 0), (1, 1), (1, -1))
    wanted = {
        (1, 0): sum_state,
        (-1, 0): carry_state,
        (-1, 1): relay_s_from_e,
        (0, 1): relay_w_from_e,
        (-1, -1): relay_n_from_e,
        (0, -1): relay_w_from_e,
    }
    owners = {
        (1, 0): 1,
        **{cell: 2 for cell in wanted if cell != (1, 0)},
    }
    trace = _compile_branched_lut_body(genome, wanted, owners)
    target = periodic_combinational_target(get_target('Full adder'))
    score, cases = evaluate_lut_full(genome, target)
    return {
        'score': score,
        'worst_case': min(cases),
        'cells': len(trace.grid),
        'genes': sum(len(chromosome.genes)
                     for chromosome in genome.chromosomes),
        'genome': genome,
        'target': target,
    }


def _add_node(nodes, sources, name, left, right=None, inhibit=None, op='and'):
    """Add one paper node; a repeated source is its one-input buffer wiring."""
    right = left if right is None else right
    nodes.add(name)
    sources[name] = (left, right, inhibit)
    return name


def nervous_full_adder_graph():
    """An explicit delay-matched Full Adder made only of paper nodes.

    XOR2 is ``(A OR B) AND NOT (A AND B)``.  The direct OR inputs are buffered
    once so their edges reach the veto node at the same instant as ``A AND B``.
    Cascading that construction with C produces Sum in four node delays.
    Carry uses ``(A AND B) OR (C AND (A OR B))`` and is buffered to the same
    four-delay depth.  ``Valid`` is the target's case strobe; the Boolean
    function does not need it and therefore correctly stays quiet on 000.
    """
    inputs = ('A', 'B', 'Cin', 'Valid')
    nodes = set(inputs)
    sources = {}

    ab_and = _add_node(nodes, sources, 'ab_and', 'A', 'B')
    a1 = _add_node(nodes, sources, 'a1', 'A')
    b1 = _add_node(nodes, sources, 'b1', 'B')
    xor_ab = _add_node(
        nodes, sources, 'xor_ab', a1, b1, inhibit=ab_and, op='or')

    c1 = _add_node(nodes, sources, 'c1', 'Cin')
    c2 = _add_node(nodes, sources, 'c2', c1)
    xor_c_and = _add_node(nodes, sources, 'xor_c_and', xor_ab, c2)
    xor_ab_1 = _add_node(nodes, sources, 'xor_ab_1', xor_ab)
    c3 = _add_node(nodes, sources, 'c3', c2)
    _add_node(
        nodes, sources, 'Sum', xor_ab_1, c3,
        inhibit=xor_c_and, op='or')

    ab_or = _add_node(nodes, sources, 'ab_or', 'A', 'B', op='or')
    c_ab = _add_node(nodes, sources, 'c_ab', ab_or, c1)
    ab_and_1 = _add_node(nodes, sources, 'ab_and_1', ab_and)
    carry_or = _add_node(
        nodes, sources, 'carry_or', c_ab, ab_and_1, op='or')
    _add_node(nodes, sources, 'Carry', carry_or)

    routing = {
        node: (None, None, None, 'and') for node in nodes}
    for node in sources:
        routing[node] = (None, None, None,
                         'or' if node in {
                             'xor_ab', 'Sum', 'ab_or', 'carry_or'} else 'and')
    return nodes, routing, sources, inputs, ('Sum', 'Carry')


def handbuilt_nervous_full_adder(target=None, alignment=None):
    """Run the explicit paper-node circuit on every current train schedule."""
    target = target or coincident_temporal_target(get_target('Full adder'))
    config = nv_run_config().pulse
    target.pulse_config = config
    nodes, routing, sources, inputs, outputs = nervous_full_adder_graph()
    role_events = {role: [] for role in outputs}
    role_intervals = {role: [] for role in outputs}
    overflow = False

    for trial in target.trials:
        simulator = create_simulator(
            {node: 1 for node in nodes}, routing,
            max_events=target.max_events, config=config, sources=sources,
            input_nodes=set(inputs), output_nodes=set(outputs))
        schedule = trial.input_events
        if schedule is None:
            # ``spike_target`` stores integer-lattice edge schedules in its
            # compatibility streams.  Each asserted sample is one pulse, not a
            # held level; the rows are separated by the target's large gap.
            schedule = [[] for _lane in inputs]
            for tick, row in enumerate(trial.streams):
                for lane, asserted in enumerate(row):
                    if asserted:
                        schedule[lane].append((float(tick), config.width))
        for lane, events in enumerate(schedule):
            for start, width in events:
                simulator.inject_pulse(inputs[lane], start, width)
        simulator.advance_to(float(_obs_len(target)))
        overflow = overflow or simulator.overflow
        for role in outputs:
            role_events[role].append(list(simulator.rise_times.get(role, ())))
            role_intervals[role].append([
                tuple(interval)
                for interval in simulator.pulse_intervals.get(role, ())])

    traces = TemporalTraces(
        {role: [[] for _trial in target.trials] for role in outputs},
        events=role_events, intervals=role_intervals, overflow=overflow)
    if alignment is None:
        score, cases, used_alignment = score_contract(traces, target)
    else:
        score, cases, used_alignment = score_contract(
            traces, target, alignment=alignment)
    return {
        'score': score,
        'worst_case': min(cases),
        'alignment': used_alignment,
        'nodes': len(nodes) - len(inputs),
        'schedules': len(target.trials),
        'overflow': overflow,
        'traces': traces,
        'target': target,
    }


def _channel_mask(config):
    """Eight-row Boolean mask of one tri-tile output channel."""
    excite_a, excite_b, inhibit, operation = ROUTING_HEX[config]
    bit_of = {'L': 0, 'R': 1, 'D': 2}
    mask = 0
    for row in range(8):
        def value(direction):
            return 0 if direction is None else (
                (row >> bit_of[direction]) & 1)
        excite = (value(excite_a) | value(excite_b)
                  if operation == 'or'
                  else value(excite_a) & value(excite_b))
        mask |= (excite & (1 - value(inhibit))) << row
    return mask


def direct_tri_tile_capability():
    """Whether a single three-channel tile can express each adder output."""
    masks = tuple(_channel_mask(config) for config in range(len(ROUTING_HEX)))
    wanted = {
        'Sum/XOR3': sum((row.bit_count() & 1) << row for row in range(8)),
        'Carry/majority3': sum((row.bit_count() >= 2) << row for row in range(8)),
    }
    result = {}
    for role, truth_mask in wanted.items():
        solution = None
        for width in (1, 2, 3):
            for configs in combinations_with_replacement(range(32), width):
                observed = 0
                for config in configs:
                    observed |= masks[config]
                if observed == truth_mask:
                    solution = configs
                    break
            if solution is not None:
                break
        result[role] = solution
    return result


def main():
    from substrates.nervous.certification import (
        DEFAULT_HOLDOUT_SEEDS, _temporal_logic_holdout_target)

    lut = handbuilt_lut_full_adder()
    branched_lut = handbuilt_branched_lut_full_adder()
    nervous = handbuilt_nervous_full_adder()
    holdouts = [
        handbuilt_nervous_full_adder(
            _temporal_logic_holdout_target(nervous['target'], seed),
            alignment=nervous['alignment'])['score']
        for seed in DEFAULT_HOLDOUT_SEEDS]
    direct = direct_tri_tile_capability()
    print('hand-built Full Adder capability')
    print('LUT grown genome: score=%.6f worst=%.6f cells=%d genes=%d exact=%s'
          % (lut['score'], lut['worst_case'], lut['cells'], lut['genes'],
             lut['inverse_exact']))
    print('LUT live branched genome: score=%.6f worst=%.6f cells=%d genes=%d'
          % (branched_lut['score'], branched_lut['worst_case'],
             branched_lut['cells'], branched_lut['genes']))
    print('Nervous paper-node graph: score=%.6f worst=%.6f nodes=%d '
          'schedules=%d shift=%s overflow=%s holdouts=%s'
          % (nervous['score'], nervous['worst_case'], nervous['nodes'],
             nervous['schedules'], nervous['alignment'], nervous['overflow'],
             ','.join('%.6f' % score for score in holdouts)))
    print('single tri-tile Sum configs:', direct['Sum/XOR3'])
    print('single tri-tile Carry configs:', direct['Carry/majority3'])


if __name__ == '__main__':
    main()
