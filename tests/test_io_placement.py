"""Evolvable input/output placement.

``tag_rank`` makes body-gene tags evolvable expression priorities. Logical
ports claim distinct mature cells in descending priority order.

``wiring_chromosome`` reserves chromosome three as a heritable but
non-developmental port map. Gene i evolves port i's node type and spatial
selector. Every port owns exactly one physical cell, and cells are exclusive so
an input and output cannot collapse onto the same node.

``spatial_chromosome`` uses the same chromosome, but gene i evolves a
normalised (x,y) anchor. Input anchors seed development and remain their exact
logical cells; output anchors attach to the nearest free living cell.

The legacy fixed path and its random stream remain unchanged.
"""
import copy
import random

from substrates.nervous.genome import (random_hex_genome, random_hex_gene, MAX_STATE,
                           HexGene, Chromosome, Genome, RoutingPatch)
from substrates.nervous.nervous import apply_routing_patches, grow_nervous
from substrates.nervous import io_placement as iop
from substrates.nervous.ga import (mutate_nv, clone_genome, crossover_nv,
                       genome_signature, rank_key)
from substrates.nervous.targets import sr_latch, coincidence_detector, with_io_placement
from substrates.nervous.temporal import prepare_net, score_temporal
from substrates.nervous.evaluation import fit_readout, score_frozen
from substrates.snn.targets import get_target
from substrates.lut.genome import (
    LutGene, Chromosome as LutChromosome, Genome as LutGenome)
from substrates.lut.ga import make_seed_genome
from substrates.lut.lut import grow_lut, cell_io_tags
from runtime.checkpoint import genome_to_dict, genome_from_dict


def _tagged_genome(seed, n_chroms=3, wiring=False, spatial=False):
    """A genome seeded for the requested evolvable I/O method."""
    random.seed(seed)
    return random_hex_genome(
        n_chroms=n_chroms,
        wiring_chromosome=wiring,
        spatial_chromosome=spatial,
        tag_rank=not (wiring or spatial),
    )


def _grown(genome, target):
    """Grow from pads, spatial germlines, or the neutral centre as configured."""
    return grow_nervous(
        genome, seeds=iop.growth_seeds(
            target, iop.io_strategy(target), genome))


# ── default-path invariance ─────────────────────────────────────────────────────

def test_default_random_gene_is_byte_identical():
    """A random gene must NOT consume extra RNG for I/O metadata — otherwise
    every seeded run and golden reproduction would diverge. Bodies carry no I/O
    numbers at all now; the tag field defaults to 0 without a draw."""
    random.seed(12345)
    gene = random_hex_gene()
    assert gene.tag == 0
    assert gene.io_kind == iop.IO_KIND_BODY
    random.seed(12345)
    ctx_l = random.randrange(32); ctx_r = random.randrange(32)
    ctx_d = random.randrange(32)
    self_in = 0 if random.random() < 0.25 else random.randrange(32)
    self_out = random.randrange(32)
    assert (gene.ctx_l, gene.ctx_r, gene.ctx_d, gene.self_in, gene.self_out) == (
        ctx_l, ctx_r, ctx_d, self_in, self_out)


def test_default_genome_has_no_tags_or_wiring():
    g = random_hex_genome(n_chroms=3)        # defaults: no tags, no wiring
    assert all(gene.tag == 0 for c in g.chromosomes for gene in c.genes)
    assert all(not getattr(c, 'wiring', False) for c in g.chromosomes)


def test_wiring_genome_leaves_body_genes_untagged():
    """Only the WIRING chromosome's genes carry port-map numbers; body genes
    stay at tag 0 — the phenotype's cell types are the binding alphabet."""
    g = _tagged_genome(7, wiring=True)
    body = [gene for c in g.chromosomes if not c.wiring for gene in c.genes]
    wiring = [gene for c in g.chromosomes if c.wiring for gene in c.genes]
    assert all(gene.tag == 0 for gene in body)
    assert any(gene.tag for gene in wiring)     # desires seeded from body types


def test_fixed_strategy_binding_is_a_noop():
    """bind_io does nothing for 'fixed'; the caller keeps the legacy path."""
    tgt = sr_latch()
    g = random_hex_genome(n_chroms=3)
    grid = _grown(g, tgt)
    assert iop.io_strategy(tgt) == 'fixed'
    assert iop.bind_io(g, grid, tgt) is None


def test_terminal_nodes_bind_one_source_per_input_and_one_sink_per_output():
    """Only genome-expressed terminal cells bind; target geometry is ignored."""
    target = copy.copy(get_target('Half adder'))
    target.io_placement = 'terminal_nodes'
    grid = {
        (-4, 2): 1, (3, -5): 1,
        (8, 9): 1, (-2, -7): 1,
        (4, 4): 1, (0, 3): 1,
    }
    genome = random_hex_genome(n_chroms=2)
    genome._terminal_kinds = {
        (-4, 2): iop.IO_KIND_INPUT,
        (3, -5): iop.IO_KIND_INPUT,
        (8, 9): iop.IO_KIND_OUTPUT,
        (-2, -7): iop.IO_KIND_OUTPUT,
        # An ordinary cell at a declared target input must not bind.
        (0, 3): iop.IO_KIND_BODY,
    }
    binding = iop.bind_io(
        genome, grid, target, tags={cell: 1 for cell in grid})
    assert binding is not None
    inputs, outputs = binding
    assert set(iop.flat_inputs(inputs)) == {(-4, 2), (3, -5)}
    assert set(iop.flat_outputs(outputs)) == {(8, 9), (-2, -7)}
    assert list(outputs) == ['sum', 'carry']
    assert not (set(iop.flat_inputs(inputs)) & set(iop.flat_outputs(outputs)))
    centre = target.grid_size // 2
    assert iop.growth_seeds(target, 'terminal_nodes', genome) == (
        (centre, centre),)


def test_terminal_nodes_single_output_contract_has_exactly_one_sink():
    target = with_io_placement(coincidence_detector(), 'terminal_nodes')
    grid = {(0, 0): 1, (1, 0): 1, (2, 1): 1, (2, 3): 1}
    genome = random_hex_genome(n_chroms=2)
    genome._terminal_kinds = {
        (0, 0): iop.IO_KIND_INPUT,
        (1, 0): iop.IO_KIND_INPUT,
        (2, 1): iop.IO_KIND_OUTPUT,
        (2, 3): iop.IO_KIND_OUTPUT,
    }
    inputs, outputs = iop.bind_io(
        genome, grid, target, tags={cell: 1 for cell in grid})
    assert len(inputs) == target.n_inputs
    assert list(outputs) == ['Q']
    assert len(outputs['Q']) == 1


def test_terminal_states_are_seeded_and_mutate_as_a_body_gene_allele():
    # Nervous terminal identity is the GROWN STATE (16 = input, 17 = output), so
    # seeding and mutation express it through ``self_out`` — not the io_kind tag.
    random.seed(321)
    genome = random_hex_genome(
        n_chroms=2, terminal_nodes=True, n_inputs=2, n_outputs=1)
    states = [
        int(gene.self_out) & 0x1F
        for chromosome in genome.chromosomes for gene in chromosome.genes]
    assert iop.IO_STATE_INPUT in states
    assert iop.IO_STATE_OUTPUT in states
    before = tuple(
        gene.self_out for chromosome in genome.chromosomes
        for gene in chromosome.genes)
    assert iop.mutate_terminal_state(genome)
    after = tuple(
        gene.self_out for chromosome in genome.chromosomes
        for gene in chromosome.genes)
    assert after != before


def test_nv_growth_marks_grown_io_state_cells_as_terminals():
    # A gene that grows the input node-type state (16) makes every cell it grows
    # a dedicated input terminal; identity comes from the grown state, not a tag.
    gene = HexGene(self_in=0, self_out=iop.IO_STATE_INPUT)
    genome = Genome(chromosomes=[
        Chromosome(genes=[gene], split=0, telomere=1)])
    seed = (4, 4)
    grid = grow_nervous(genome, seeds=(seed,))
    assert grid
    assert seed not in genome._terminal_kinds
    assert genome._terminal_kinds
    assert set(genome._terminal_kinds.values()) == {iop.IO_KIND_INPUT}


def test_dedicated_io_states_are_one_way_in_the_pulse_engine():
    # A dedicated INPUT (state 16) cannot fire from a neighbour (source-only); a
    # dedicated OUTPUT (state 17) cannot drive a neighbour (sink-only). The
    # one-way physics applies only when the cell is passed as an I/O terminal.
    from substrates.nervous.pulse import PulseSim, PulseConfig
    cfg = PulseConfig(model='pulse_delay')
    src = {'X': (None, None, None), 'IN': ('X', 'X', None),
           'DR': (None, None, None), 'OUT': ('DR', 'DR', None),
           'Y': ('OUT', 'OUT', None)}
    grid = {c: 0 for c in src}
    routing = {c: (src[c][0], src[c][1], src[c][2], 'and') for c in src}

    def fires(input_nodes, output_nodes):
        sim = PulseSim(grid, routing, config=cfg, sources=src,
                       input_nodes=input_nodes, output_nodes=output_nodes)
        sim.inject_pulse('X', 2.0, 1.0)
        sim.inject_pulse('DR', 2.0, 1.0)
        sim.advance_to(12.0)
        return bool(sim.rise_times.get('IN')), bool(sim.rise_times.get('Y'))

    assert fires(set(), set()) == (True, True)          # ordinary cells relay
    assert fires({'IN'}, {'OUT'}) == (False, False)     # terminals are one-way


def test_lut_growth_expresses_unambiguous_terminal_kind():
    growth_gene = LutGene(
        self_in=0, self_out=0xFFFE, io_kind=iop.IO_KIND_OUTPUT)
    maintenance_gene = LutGene(
        self_in=0xFFFE, self_out=0xFFFE,
        io_kind=iop.IO_KIND_OUTPUT)
    genome = LutGenome(chromosomes=[
        LutChromosome(
            genes=[growth_gene, maintenance_gene], split=1, telomere=1)])
    seed = (3, 3)
    grid = grow_lut(genome, seeds=(seed,), grid_size=7, iters=4)
    assert grid
    assert seed not in genome._terminal_kinds
    assert genome._terminal_kinds
    assert set(genome._terminal_kinds.values()) == {iop.IO_KIND_OUTPUT}


def test_terminal_kind_checkpoint_round_trip_for_nv_and_lut():
    nv = Genome(chromosomes=[Chromosome(
        genes=[HexGene(self_out=1, io_kind=iop.IO_KIND_INPUT)],
        split=0)])
    lut = LutGenome(chromosomes=[LutChromosome(
        genes=[LutGene(self_out=1, io_kind=iop.IO_KIND_OUTPUT)],
        split=0)])
    nv_loaded = genome_from_dict(genome_to_dict(nv, 'nervous'), 'nervous')
    lut_loaded = genome_from_dict(genome_to_dict(lut, 'lut'), 'lut')
    assert nv_loaded.chromosomes[0].genes[0].io_kind == iop.IO_KIND_INPUT
    assert lut_loaded.chromosomes[0].genes[0].io_kind == iop.IO_KIND_OUTPUT


def test_terminal_bindings_are_not_fixed_across_genome_seeds():
    target = copy.copy(get_target('Half adder'))
    target.io_placement = 'terminal_nodes'

    nv_bindings = []
    for seed in (0, 2):
        random.seed(seed)
        genome = random_hex_genome(
            2, terminal_nodes=True,
            n_inputs=target.n_inputs, n_outputs=len(target.outputs))
        grid = grow_nervous(
            genome, iop.growth_seeds(target, 'terminal_nodes', genome))
        nv_bindings.append(iop.bind_io(
            genome, grid, target, 'terminal_nodes'))

    lut_bindings = []
    for seed in (0, 4):
        random.seed(seed)
        genome = make_seed_genome(2)
        iop.seed_terminal_kinds(
            genome, target.n_inputs, len(target.outputs))
        grid = grow_lut(
            genome, iop.growth_seeds(target, 'terminal_nodes', genome),
            target.grid_size, target.iters)
        lut_bindings.append(iop.bind_io(
            genome, grid, target, 'terminal_nodes',
            tags=cell_io_tags(genome, grid)))

    assert all(binding is not None for binding in nv_bindings + lut_bindings)
    assert nv_bindings[0] != nv_bindings[1]
    assert lut_bindings[0] != lut_bindings[1]


def test_default_mutation_stream_is_unchanged_by_evolve_io_flag():
    """With evolve_io=False (the default) the mutation of a plain genome is
    identical to the pre-feature behaviour: no port-map mutation can fire, so a
    seeded mutation reproduces exactly."""
    base = random_hex_genome(n_chroms=3)
    random.seed(999)
    a = mutate_nv(base)
    random.seed(999)
    b = mutate_nv(base, evolve_io=False)
    assert genome_signature(a) == genome_signature(b)


# ── per-cell types (the binding alphabet IS the phenotype) ──────────────────────

def test_cell_tags_are_the_grid_states():
    g = _tagged_genome(7)
    tgt = sr_latch()
    grid = _grown(g, tgt)
    tags = iop.cell_tags(g, grid)
    assert set(tags) == set(grid)
    # a cell's type is literally its settled state — the Designer node type
    assert all(tags[pos] == int(state) for pos, state in grid.items())


def test_a_grown_organism_expresses_several_distinct_cell_types():
    g = _tagged_genome(7)
    grid = _grown(g, sr_latch())
    assert len(set(iop.cell_tags(g, grid).values())) >= 3


# ── Method A: tag_rank (node-type rank) ─────────────────────────────────────────

def _genome_expressing_values(target, min_values, wiring=False, seeds=range(1, 80)):
    """Scan RNG seeds for a genome whose center-grown body expresses at least
    ``min_values`` distinct cell types (no-wrap binding needs one type per
    port). Returns (genome, grid, tags)."""
    for seed in seeds:
        g = _tagged_genome(seed, wiring=wiring)
        grid = _grown(g, target)
        tags = iop.cell_tags(g, grid)
        if len(set(tags.values())) >= min_values:
            return g, grid, tags
    raise AssertionError('no seed grew a body with %d cell types' % min_values)


def test_tag_rank_binds_distinct_cells_by_descending_gene_priority():
    """The highest body-gene tag wins first; ports never share a cell."""
    tgt = with_io_placement(coincidence_detector(), 'tag_rank')  # 2 in, 1 out
    g = Genome(chromosomes=[Chromosome(genes=[
        HexGene(self_out=3, tag=10),
        HexGene(self_out=5, tag=90),
        HexGene(self_out=7, tag=40),
    ])], tag=123)
    grid = {(0, 0): 3, (1, 0): 5, (2, 0): 7}
    in_pos, out_pos = iop.bind_io(g, grid, tgt)
    groups = iop.input_groups(in_pos) + list(iop.output_groups(out_pos).values())
    assert groups == [[(1, 0)], [(2, 0)], [(0, 0)]]
    assert len(set(iop._flat(groups))) == 3
    report = iop.binding_report(g, grid, tgt)
    assert [entry['tag'] for entry in report] == [90, 40, 10]


def test_tag_rank_is_deterministic():
    g = _tagged_genome(7)
    tgt = with_io_placement(sr_latch(), 'tag_rank')
    grid = _grown(g, tgt)
    assert iop.bind_io(g, grid, tgt) == iop.bind_io(g, grid, tgt)


# ── Method B: wiring_chromosome ─────────────────────────────────────────────────────

def _force_wiring_desires(genome, values):
    """Overwrite the wiring chromosome's gene tags (the per-port desired CELL
    TYPES), in order, with fresh gene copies. Harmless to the grown body — the
    tag field has no effect on growth, only on I/O binding."""
    wiring = iop.wiring_chromosome(genome)
    genes = []
    for i, gene in enumerate(wiring.genes):
        gene = copy.copy(gene)
        gene.tag = values[i % len(values)]
        gene.io_limit = 1
        gene.io_selector = i + 1
        genes.append(gene)
    wiring.genes = genes


def test_wiring_chromosome_gene_i_maps_port_i_exact_match_only():
    """Gene i maps port i to exact node types and its selected instances."""
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    n_ports = tgt.n_inputs + tgt.n_outputs
    g, grid, tags = _genome_expressing_values(tgt, n_ports, wiring=True)
    expressed = sorted(set(tags.values()))
    wants = expressed[:n_ports]
    _force_wiring_desires(g, wants)
    genes = iop._wiring_port_genes(g, n_ports)
    assert [iop._gene_tag(gene) for gene in genes] == wants
    in_pos, out_pos = iop.bind_io(g, grid, tgt)
    groups = iop.input_groups(in_pos) + list(iop.output_groups(out_pos).values())
    for want, cells in zip(wants, groups):
        assert len(cells) == 1
        assert all(tags[cell] == want for cell in cells)
    assert len(iop._flat(groups)) == n_ports


def test_wiring_chromosome_unexpressed_desire_is_unbindable():
    """LOCK AND KEY: if any port desires a cell TYPE the grown body does not
    contain, the organism has no I/O at all and scores 0 — random genomes
    cannot wire themselves up by accident. Force the port map to desire type 7
    while the body expresses other types."""
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    g, grid, tags = _genome_expressing_values(tgt, 2, wiring=True)
    absent = next(v for v in range(MAX_STATE) if v not in set(tags.values()))
    _force_wiring_desires(g, [absent])             # a type not on the body
    assert iop.bind_io(g, grid, tgt) is None
    assert score_temporal(g, tgt) == 0.0


def test_wiring_chromosome_shared_type_claims_distinct_instances():
    """Ports may request one type, but they must win different physical cells."""
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    g = Genome(chromosomes=[
        Chromosome(genes=[HexGene(self_out=3)]),
        Chromosome(genes=[HexGene(self_out=4)]),
        Chromosome(genes=[
            HexGene(tag=3, io_limit=1, io_selector=11),
            HexGene(tag=3, io_limit=1, io_selector=22),
            HexGene(tag=3, io_limit=1, io_selector=33),
        ], wiring=True),
    ])
    grid = {(0, 0): 3, (1, 0): 3, (2, 0): 3}
    in_pos, out_pos = iop.bind_io(g, grid, tgt)
    groups = iop.input_groups(in_pos) + list(iop.output_groups(out_pos).values())
    assert all(len(cells) == 1 for cells in groups)
    assert set(iop._flat(groups)) == set(grid)
    # With only two instances the third port cannot alias either one.
    assert iop.bind_io(g, dict(list(grid.items())[:2]), tgt) is None


def test_wiring_always_selects_one_type_instance_and_selector_moves_it():
    """Historical limit values are ignored; the selector chooses one site."""
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    g = Genome(chromosomes=[
        Chromosome(genes=[
            HexGene(self_out=3), HexGene(self_out=4), HexGene(self_out=5),
        ]),
        Chromosome(genes=[HexGene(self_out=6)]),
        Chromosome(genes=[
            HexGene(tag=3, io_limit=0, io_selector=11),
            HexGene(tag=4, io_limit=1, io_selector=22),
            HexGene(tag=5, io_limit=1, io_selector=33),
        ], wiring=True),
    ])
    grid = {
        (0, 0): 3, (1, 0): 3, (2, 0): 3,
        (3, 0): 4, (4, 0): 5,
    }
    first = set(iop.bind_io(g, grid, tgt)[0][0])
    assert len(first) == 1

    # A retired multi-site value cannot change the binding or reported limit.
    legacy = clone_genome(g)
    gene = copy.copy(legacy.chromosomes[2].genes[0])
    gene.io_limit = 8
    legacy.chromosomes[2].genes[0] = gene
    assert set(iop.bind_io(legacy, grid, tgt)[0][0]) == first
    assert iop.binding_report(legacy, grid, tgt)[0]['limit'] == 1

    # The heritable selector can choose a different site without coordinates
    # being hard-coded in the mapping.
    alternatives = []
    for selector in range(1, 100):
        candidate = clone_genome(legacy)
        gene = copy.copy(candidate.chromosomes[2].genes[0])
        gene.io_selector = selector
        candidate.chromosomes[2].genes[0] = gene
        alternatives.append(set(iop.bind_io(candidate, grid, tgt)[0][0]))
    assert any(cells != first for cells in alternatives)

    # One selector step rotates to the next candidate.
    adjacent = clone_genome(legacy)
    gene = copy.copy(adjacent.chromosomes[2].genes[0])
    gene.io_selector += 1
    adjacent.chromosomes[2].genes[0] = gene
    second = set(iop.bind_io(adjacent, grid, tgt)[0][0])
    assert len(second) == 1
    assert first.isdisjoint(second)


def test_phenotype_seed_builds_a_viable_exclusive_mapping():
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    g = Genome(chromosomes=[
        Chromosome(genes=[
            HexGene(self_out=3), HexGene(self_out=4), HexGene(self_out=5),
        ]),
        Chromosome(genes=[HexGene(self_out=6)]),
        Chromosome(genes=[
            HexGene(), HexGene(), HexGene(),
        ], wiring=True),
    ], tag=9123)
    grid = {
        (0, 0): 3, (1, 0): 3, (2, 0): 4,
        (3, 0): 5, (4, 0): 6,
    }
    random.seed(22)
    assert iop.seed_wiring_from_phenotype(g, grid, tgt) == 3
    assert iop.binding_progress(g, grid, tgt) == (3, 3)
    in_pos, out_pos = iop.bind_io(g, grid, tgt)
    groups = iop.input_groups(in_pos) + list(iop.output_groups(out_pos).values())
    assert all(len(group) == 1 for group in groups)
    assert len(iop._flat(groups)) == 3
    genes = iop._wiring_port_genes(g, 3)
    assert all(gene.io_limit == 1 for gene in genes)


def test_binding_progress_grades_incomplete_maps_without_faking_fitness():
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    g = Genome(chromosomes=[
        Chromosome(genes=[HexGene(self_out=3)]),
        Chromosome(genes=[HexGene(self_out=4)]),
        Chromosome(genes=[
            HexGene(tag=3, io_limit=1),
            HexGene(tag=3, io_limit=1),
            HexGene(tag=3, io_limit=1),
        ], wiring=True),
    ])
    grid = {(0, 0): 3, (1, 0): 3}
    assert iop.bind_io(g, grid, tgt) is None
    assert iop.binding_progress(g, grid, tgt) == (2, 3)

    less = clone_genome(g)
    more = clone_genome(g)
    iop.record_binding_progress(less, (1, 3))
    iop.record_binding_progress(more, (2, 3))
    assert rank_key(more, 0.0) > rank_key(less, 0.0)


def test_wiring_chromosome_too_short_is_unbindable():
    """A port map with fewer genes than ports leaves ports unmapped — no
    wrapping, the organism is unbindable."""
    g = _tagged_genome(7, wiring=True)
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    grid = _grown(g, tgt)
    wiring = iop.wiring_chromosome(g)
    wiring.genes = wiring.genes[:1]                # 1 gene for 3 ports
    assert iop._wiring_port_genes(g, 3) is None
    assert iop.bind_io(g, grid, tgt) is None


def test_wiring_chromosome_is_fixed_third_and_non_developmental():
    g = _tagged_genome(7, wiring=True)
    assert [c.wiring for c in g.chromosomes] == [False, False, True]
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    before = _grown(g, tgt)
    h = clone_genome(g)
    for index, gene in enumerate(h.chromosomes[2].genes):
        edited = copy.copy(gene)
        edited.self_out = (gene.self_out + index + 1) % MAX_STATE
        edited.tag = (gene.tag + index + 2) % MAX_STATE
        h.chromosomes[2].genes[index] = edited
    assert _grown(h, tgt) == before


def test_wiring_chromosome_has_no_implicit_body_fallback():
    g = _tagged_genome(7, wiring=False)
    assert all(not c.wiring for c in g.chromosomes)
    assert iop.wiring_chromosome(g) is None
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    assert iop.bind_io(g, _grown(g, tgt), tgt) is None


# ── Method C: spatial_chromosome ───────────────────────────────────────────────

def _spatial_genome(n_ports=3):
    return random_hex_genome(
        3, spatial_chromosome=True, n_ports=n_ports)


def test_spatial_chromosome_maps_anchors_to_nearest_distinct_live_cells():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    genome = _spatial_genome()
    wiring = iop.wiring_chromosome(genome)
    anchors = (
        (0, 0),
        (iop.SPATIAL_COORD_MAX, 0),
        (iop.SPATIAL_COORD_MAX, iop.SPATIAL_COORD_MAX),
    )
    for index, (x_code, y_code) in enumerate(anchors):
        gene = copy.copy(wiring.genes[index])
        gene.tag = x_code
        gene.io_selector = y_code
        wiring.genes[index] = gene
    # Node values are deliberately unrelated to the genes: spatial binding
    # depends only on living positions, never on a mutable node type.
    grid = {
        (0, 0): 31,
        (4, 0): 7,
        (0, 4): 19,
        (4, 4): 2,
    }
    inputs, outputs = iop.bind_io(genome, grid, tgt)
    assert inputs == [[(0, 0)], [(4, 0)]]
    assert outputs == {'Q': [(4, 4)]}
    assert iop.binding_progress(genome, grid, tgt) == (3, 3)


def test_spatial_input_anchors_are_distinct_developmental_germlines():
    """Input placement changes ontogeny instead of being bolted on afterward."""
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    random.seed(0)
    genome = _spatial_genome()
    assert iop.seed_spatial_from_geometry(genome, None, tgt) == 3

    seeds = iop.growth_seeds(tgt, 'spatial_chromosome', genome)
    assert seeds == tuple(tgt.inputs)
    assert len(set(seeds)) == tgt.n_inputs

    grid = grow_nervous(genome, seeds=seeds)
    inputs, _outputs = iop.bind_io(genome, grid, tgt)
    assert inputs == [[cell] for cell in seeds]


def test_spatial_input_mutation_moves_germline_and_regrows_body():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    random.seed(0)
    genome = _spatial_genome()
    iop.seed_spatial_from_geometry(genome, None, tgt)
    before_seeds = iop.growth_seeds(tgt, 'spatial_chromosome', genome)
    before_grid = grow_nervous(genome, seeds=before_seeds)

    wiring = iop.wiring_chromosome(genome)
    gene = copy.copy(wiring.genes[0])
    gene.tag = iop.SPATIAL_COORD_MAX
    gene.io_selector = iop.SPATIAL_COORD_MAX
    wiring.genes[0] = gene

    after_seeds = iop.growth_seeds(tgt, 'spatial_chromosome', genome)
    after_grid = grow_nervous(genome, seeds=after_seeds)
    assert after_seeds != before_seeds
    assert all(seed in after_grid for seed in after_seeds)
    assert after_grid != before_grid
    inputs, _outputs = iop.bind_io(genome, after_grid, tgt)
    assert inputs == [[cell] for cell in after_seeds]


def test_spatial_colliding_input_anchors_resolve_to_distinct_germlines():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    genome = _spatial_genome()
    wiring = iop.wiring_chromosome(genome)
    for index in range(tgt.n_inputs):
        gene = copy.copy(wiring.genes[index])
        gene.tag = iop.SPATIAL_COORD_MAX // 2
        gene.io_selector = iop.SPATIAL_COORD_MAX // 2
        wiring.genes[index] = gene
    seeds = iop.growth_seeds(tgt, 'spatial_chromosome', genome)
    assert len(seeds) == tgt.n_inputs
    assert len(set(seeds)) == tgt.n_inputs


def test_spatial_factory_diversifies_outputs_without_randomising_germlines():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    random.seed(17)
    genome = _spatial_genome()
    assert iop.seed_spatial(
        genome, None, tgt, geometry_prob=0.0) == 3
    assert iop.spatial_input_sites(genome, tgt) == tuple(tgt.inputs)
    sites = iop.spatial_port_sites(genome, tgt)
    assert sites[tgt.n_inputs:] != tuple(
        terminal.pos for terminal in tgt.outputs)


def test_spatial_output_rescue_keeps_body_and_input_alleles_fixed():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    random.seed(19)
    genome = _spatial_genome()
    iop.seed_spatial_from_geometry(genome, None, tgt)
    body_before = [
        tuple((gene.ctx_l, gene.ctx_r, gene.ctx_d,
               gene.self_in, gene.self_out)
              for gene in chromosome.genes)
        for chromosome in iop.body_chromosomes(genome)]
    input_before = [
        (gene.tag, gene.io_selector)
        for gene in iop.wiring_chromosome(genome).genes[:tgt.n_inputs]]

    variants = iop.spatial_output_variants(genome, tgt, limit=8)
    assert len(variants) == 8
    assert len({
        (iop.wiring_chromosome(candidate).genes[tgt.n_inputs].tag,
         iop.wiring_chromosome(candidate).genes[tgt.n_inputs].io_selector)
        for candidate in variants}) == 8
    for candidate in variants:
        assert [
            tuple((gene.ctx_l, gene.ctx_r, gene.ctx_d,
                   gene.self_in, gene.self_out)
                  for gene in chromosome.genes)
            for chromosome in iop.body_chromosomes(candidate)] == body_before
        assert [
            (gene.tag, gene.io_selector)
            for gene in iop.wiring_chromosome(
                candidate).genes[:tgt.n_inputs]] == input_before


def test_mature_routing_patch_changes_one_live_cell_only():
    genome = Genome(routing_patches=[RoutingPatch(1, 0, 7)])
    original = {(0, 0): 1, (1, 0): 3, (2, 0): 5}
    patched = apply_routing_patches(genome, original)
    assert original == {(0, 0): 1, (1, 0): 3, (2, 0): 5}
    assert patched == {(0, 0): 1, (1, 0): 7, (2, 0): 5}


def test_spatial_routing_rescue_is_one_cell_local_and_heritable():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    random.seed(1901)
    genome = _spatial_genome()
    iop.seed_spatial_from_geometry(genome, None, tgt)
    baseline = _grown(genome, tgt)
    body_before = [
        tuple((gene.ctx_l, gene.ctx_r, gene.ctx_d,
               gene.self_in, gene.self_out)
              for gene in chromosome.genes)
        for chromosome in iop.body_chromosomes(genome)]
    ports_before = [
        (gene.tag, gene.io_selector)
        for gene in iop.wiring_chromosome(genome).genes]

    variants = iop.spatial_routing_variants(genome, tgt, limit=12)
    assert variants
    for candidate in variants:
        assert [
            tuple((gene.ctx_l, gene.ctx_r, gene.ctx_d,
                   gene.self_in, gene.self_out)
                  for gene in chromosome.genes)
            for chromosome in iop.body_chromosomes(candidate)] == body_before
        assert [
            (gene.tag, gene.io_selector)
            for gene in iop.wiring_chromosome(candidate).genes] == ports_before
        assert len(candidate.routing_patches) == 1
        patch = candidate.routing_patches[0]
        site = (patch.x, patch.y)
        assert site in baseline
        delta = int(patch.state) ^ int(baseline[site])
        assert delta and delta & (delta - 1) == 0


def test_spatial_seed_is_viable_and_mutation_changes_one_coordinate():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    genome = _spatial_genome()
    grid = {
        (-2, -1): 1,
        (0, 0): 2,
        (3, 1): 3,
        (1, 4): 4,
    }
    random.seed(90210)
    assert iop.seed_spatial_from_phenotype(genome, grid, tgt) == 3
    bound = iop.bind_io(genome, grid, tgt)
    assert bound is not None
    assert len(set(iop.flat_inputs(bound[0])
                   + iop.flat_outputs(bound[1]))) == 3

    before = [
        (gene.tag, gene.io_selector)
        for gene in iop.wiring_chromosome(genome).genes
    ]
    iop.mutate_io_allele(
        genome, MAX_STATE, strategy='spatial_chromosome')
    after = [
        (gene.tag, gene.io_selector)
        for gene in iop.wiring_chromosome(genome).genes
    ]
    changed_fields = sum(
        old_value != new_value
        for old, new in zip(before, after)
        for old_value, new_value in zip(old, new))
    assert changed_fields == 1
    assert all(
        0 <= value <= iop.SPATIAL_COORD_MAX
        for pair in after for value in pair)


def test_spatial_chromosome_only_fails_when_there_are_too_few_cells():
    tgt = with_io_placement(coincidence_detector(), 'spatial_chromosome')
    genome = _spatial_genome()
    tiny = {(0, 0): 1, (1, 0): 1}
    assert iop.binding_progress(genome, tiny, tgt) == (2, 3)
    assert iop.bind_io(genome, tiny, tgt) is None


def test_spatial_genome_factory_reserves_three_random_anchor_genes_all_backends():
    from substrates.lut.genome import random_lut_genome
    from substrates.snn.genome import random_genome
    factories = (
        lambda: random_hex_genome(
            3, spatial_chromosome=True, n_ports=4),
        lambda: random_lut_genome(
            3, spatial_chromosome=True, n_ports=4),
        lambda: random_genome(
            3, spatial_chromosome=True, n_ports=4),
    )
    for make in factories:
        genome = make()
        assert [chromosome.wiring
                for chromosome in genome.chromosomes] == [False, False, True]
        genes = iop.wiring_chromosome(genome).genes
        assert len(genes) == 4
        assert all(
            0 <= value <= iop.SPATIAL_COORD_MAX
            for gene in genes
            for value in (gene.tag, gene.io_selector))


# ── organisms without enough numbers are unbindable (no I/O by accident) ────────

def test_too_few_physical_cells_is_unbindable_under_tag_rank():
    # Node types may repeat; tag_rank needs one exclusive physical site per port.
    g = Genome(chromosomes=[Chromosome(genes=[HexGene(self_out=0)], split=0,
                                       telomere=1)])
    tgt = with_io_placement(coincidence_detector(), 'tag_rank')
    assert iop.bind_io(g, {(0, 0): 1}, tgt) is None
    assert prepare_net(g, tgt) is None
    assert score_temporal(g, tgt) == 0.0


# ── leak-free validation (fit/freeze reuse the evolved binding) ─────────────────

def _bindable_genome(strat, base, seeds=range(1, 120)):
    """Scan seeds for a genome that prepare_net can actually bind under the
    exact-match gate (most random genomes legitimately fail it)."""
    tgt = with_io_placement(base, strat)
    for seed in seeds:
        g = _tagged_genome(
            seed,
            wiring=(strat == 'wiring_chromosome'),
            spatial=(strat == 'spatial_chromosome'))
        if prepare_net(g, tgt) is not None:
            return g, tgt
    raise AssertionError('no bindable genome found for %s' % strat)


def test_fit_readout_captures_input_binding_and_score_frozen_reuses_it():
    for strat in ('tag_rank', 'wiring_chromosome', 'spatial_chromosome'):
        g, tgt = _bindable_genome(strat, sr_latch())
        prep = prepare_net(g, tgt)
        assert prep is not None
        groups = iop.input_groups(prep[2])
        fitted = fit_readout(g, tgt)
        assert fitted is not None
        # inputs are fitted (attachment groups, shape-normalised)
        assert iop.input_groups(fitted.inputs) == groups
        assert iop.input_groups(fitted.input_positions(tgt)) == groups
        # And produce a finite, reproducible score.
        s1 = score_frozen(g, tgt, fitted)
        s2 = score_frozen(g, tgt, fitted)
        assert s1 == s2 and 0.0 <= s1 <= 1.0


def test_fixed_fit_readout_leaves_inputs_empty():
    g = random_hex_genome(n_chroms=3)
    tgt = sr_latch()                          # fixed strategy
    fitted = fit_readout(g, tgt)
    if fitted is not None:
        assert fitted.inputs == ()
        assert fitted.input_positions(tgt) == list(tgt.inputs)


# ── evolvability ─────────────────────────────────────────────────────────────────

def test_evolve_io_mutates_mapping_alleles_but_not_designation():
    g = _tagged_genome(7, wiring=True)
    random.seed(0)
    original = [
        (gene.tag, gene.io_limit, gene.io_selector)
        for gene in iop.wiring_chromosome(g).genes
    ]
    changed = False
    for _ in range(300):
        m = mutate_nv(g, evolve_io=True)
        assert [c.wiring for c in m.chromosomes] == [False, False, True]
        alleles = [
            (gene.tag, gene.io_limit, gene.io_selector)
            for gene in iop.wiring_chromosome(m).genes
        ]
        if alleles != original:
            changed = True
            break
    assert changed, "evolve_io never mutated type or selector"


def test_wiring_crossover_inherits_each_port_map_as_a_whole():
    """A/B/Q form one interface; crossover must not splice its port suffix."""
    a = _tagged_genome(7, wiring=True)
    b = _tagged_genome(19, wiring=True)
    for base, genome in ((100, a), (200, b)):
        wiring = iop.wiring_chromosome(genome)
        for index, old in enumerate(wiring.genes):
            gene = copy.copy(old)
            gene.tag = base + index
            gene.io_selector = base * 10 + index
            wiring.genes[index] = gene

    def port_map(genome):
        return [(gene.tag, gene.io_selector)
                for gene in iop.wiring_chromosome(genome).genes]

    map_a, map_b = port_map(a), port_map(b)
    child_a, child_b = crossover_nv(a, b)
    assert port_map(child_a) == map_a
    assert port_map(child_b) == map_b


def test_tag_rank_gene_priority_is_evolvable_without_changing_growth():
    """Body tags reorder attachment candidates but do not alter development."""
    tgt = with_io_placement(coincidence_detector(), 'tag_rank')
    g = _tagged_genome(7)
    assert _grown(clone_genome(g), tgt) == _grown(g, tgt)
    h = clone_genome(g)
    for ci, c in enumerate(h.chromosomes):
        for gi, gene in enumerate(c.genes):
            ng = copy.copy(gene)
            ng.tag = (ci + 1) * 1000 + gi
            c.genes[gi] = ng
    assert _grown(h, tgt) == _grown(g, tgt)

    # On a controlled mature body, changing only priorities changes the winner.
    a = Genome(chromosomes=[Chromosome(genes=[
        HexGene(self_out=3, tag=100),
        HexGene(self_out=5, tag=10),
        HexGene(self_out=7, tag=1),
    ])])
    b = clone_genome(a)
    b.chromosomes[0].genes[0] = copy.copy(b.chromosomes[0].genes[0])
    b.chromosomes[0].genes[1] = copy.copy(b.chromosomes[0].genes[1])
    b.chromosomes[0].genes[0].tag = 10
    b.chromosomes[0].genes[1].tag = 100
    grid = {(0, 0): 3, (1, 0): 5, (2, 0): 7}
    assert iop.bind_io(a, grid, tgt)[0][0] == [(0, 0)]
    assert iop.bind_io(b, grid, tgt)[0][0] == [(1, 0)]


def test_evolve_io_false_never_touches_the_port_map():
    """With evolve_io=False the port-map operators are unavailable, so the
    wiring chromosome's numbers and the wiring flag are never introduced by
    mutation. Structural ops still change gene/chromosome COUNTS."""
    g = _tagged_genome(7, wiring=True)
    parent_alleles = [
        (gene.tag, gene.io_limit, gene.io_selector)
        for gene in iop.wiring_chromosome(g).genes
    ]
    parent_flags = [c.wiring for c in g.chromosomes]
    random.seed(0)
    for _ in range(300):
        m = mutate_nv(g, evolve_io=False)
        assert [c.wiring for c in m.chromosomes] == parent_flags
        assert [
            (gene.tag, gene.io_limit, gene.io_selector)
            for gene in iop.wiring_chromosome(m).genes
        ] == parent_alleles


def test_signature_separates_genomes_that_differ_only_in_port_map():
    g = _tagged_genome(7, wiring=True)
    h = clone_genome(g)
    wiring = next(c for c in h.chromosomes if c.wiring)
    gene = copy.copy(wiring.genes[0])
    gene.tag = (gene.tag + 1) % MAX_STATE
    wiring.genes[0] = gene
    assert genome_signature(g) != genome_signature(h)
    k = clone_genome(g)
    wiring = iop.wiring_chromosome(k)
    gene = copy.copy(wiring.genes[0])
    gene.io_selector ^= 1
    wiring.genes[0] = gene
    assert genome_signature(g) != genome_signature(k)
    legacy = clone_genome(g)
    wiring = iop.wiring_chromosome(legacy)
    gene = copy.copy(wiring.genes[0])
    gene.io_limit = 8
    wiring.genes[0] = gene
    assert genome_signature(g) == genome_signature(legacy)


# ── LUT backend ──────────────────────────────────────────────────────────────────

def _lut_genome(seed, wiring=False):
    from substrates.lut.genome import random_lut_genome
    random.seed(seed)
    return random_lut_genome(3, wiring_chromosome=wiring)


def test_lut_default_random_gene_is_byte_identical():
    from substrates.lut.genome import random_lut_gene
    random.seed(4242)
    gene = random_lut_gene()
    assert gene.tag == 0
    random.seed(4242)
    draws = [random.randrange(1 << 16) for _ in range(4)]
    self_in = 0 if random.random() < 0.25 else random.randrange(1 << 16)
    self_out = random.randrange(1 << 16)
    assert (gene.ctx_n, gene.ctx_e, gene.ctx_s, gene.ctx_w,
            gene.self_in, gene.self_out) == (*draws, self_in, self_out)


def test_lut_cell_io_tags_are_the_direction_luts():
    """A LUT cell's TYPES are its nonzero directional tables — a cell may be
    several types at once, and a port binds if its desired type matches ANY."""
    from substrates.lut.lut import grow_lut, cell_io_tags
    g = _lut_genome(5)
    base = sr_latch()
    grid = grow_lut(g, seeds=iop.growth_seeds(with_io_placement(base, 'tag_rank')),
                    grid_size=base.grid_size, iters=base.iters)
    tags = cell_io_tags(g, grid)
    assert set(tags) == set(grid)
    for pos, state in grid.items():
        assert set(tags[pos]) == {int(v) for v in state if v}


def test_lut_prepare_honors_strategy_and_freezes_inputs():
    from substrates.lut.ga import prepare_lut
    from substrates.lut.lut import grow_lut, cell_io_tags
    base = sr_latch()
    seeds = iop.growth_seeds(with_io_placement(base, 'tag_rank'))
    for strat in ('tag_rank', 'wiring_chromosome'):
        tgt = with_io_placement(base, strat)
        # scan for a genome that binds under the exact-match gate
        prep = g = grid = None
        for seed in range(1, 120):
            g = _lut_genome(seed, wiring=(strat == 'wiring_chromosome'))
            prep = prepare_lut(g, tgt)
            if prep is not None:
                grid = grow_lut(g, seeds=seeds, grid_size=base.grid_size,
                                iters=base.iters)
                break
        if prep is None:
            continue
        _, out_pos, _, in_pos = prep
        tags = cell_io_tags(g, grid)
        expected = iop.bind_io(g, grid, tgt, tags=tags)
        assert (iop.input_groups(in_pos), out_pos) == (
            iop.input_groups(expected[0]), expected[1])
        fitted = fit_readout(g, tgt, backend='lut')
        if fitted is not None:
            assert iop.input_groups(fitted.inputs) == iop.input_groups(in_pos)
            s1 = score_frozen(g, tgt, fitted)
            assert s1 == score_frozen(g, tgt, fitted) and 0.0 <= s1 <= 1.0


def test_lut_fixed_prep_shape_and_seed_inputs():
    from substrates.lut.ga import prepare_lut
    g = _lut_genome(5)
    tgt = sr_latch()
    prep = prepare_lut(g, tgt)
    if prep is not None:
        assert len(prep) == 4
        assert list(prep[3]) == list(tgt.inputs)  # fixed = seed pads


def test_lut_evolve_io_gates_port_map_mutations():
    from substrates.lut.ga import mutate_lut
    g = _lut_genome(5, wiring=True)
    parent = [(gene.tag, gene.io_limit, gene.io_selector)
              for gene in iop.wiring_chromosome(g).genes]
    random.seed(0)
    for _ in range(200):
        m = mutate_lut(g, evolve_io=False)
        child = [(gene.tag, gene.io_limit, gene.io_selector)
                 for gene in iop.wiring_chromosome(m).genes]
        assert child == parent
    random.seed(0)
    changed = False
    for _ in range(200):
        m = mutate_lut(g, evolve_io=True)
        child = [(gene.tag, gene.io_limit, gene.io_selector)
                 for gene in iop.wiring_chromosome(m).genes]
        if child != parent:
            changed = True
            break
    assert changed, "LUT evolve_io never changed a mapping allele"


def test_seed_io_metadata_seeds_the_port_map_from_body_types():
    from substrates.lut.ga import make_seed_genome
    random.seed(11)
    g = make_seed_genome(3)
    assert all(gene.tag == 0 for c in g.chromosomes for gene in c.genes)
    iop.seed_io_metadata(g, wiring_chromosome=True, n_ports=4)
    assert sum(c.wiring for c in g.chromosomes) == 1
    assert g.chromosomes[2].wiring
    wiring = next(c for c in g.chromosomes if c.wiring)
    assert len(wiring.genes) == 4
    body_types = {g2.self_out for c in g.chromosomes if not c.wiring
                  for g2 in c.genes if g2.self_out}
    # every seeded desire is a body cell type (a self_out a body gene installs)
    assert all(gene.tag in body_types for gene in wiring.genes)
    assert all(gene.io_limit == 1 for gene in wiring.genes)
    assert all(gene.io_selector == 0 for gene in wiring.genes)


# ── SNN backend ──────────────────────────────────────────────────────────────────

def _snn_genome(seed, wiring=False):
    from substrates.snn.genome import random_genome
    random.seed(seed)
    return random_genome(3, wiring_chromosome=wiring,
                         tag_rank=not wiring)


def _snn_target(strategy='fixed'):
    import dataclasses
    from substrates.snn.targets import get_target, DEFAULT_TARGET
    t = get_target(DEFAULT_TARGET)
    # snn targets predate the io_placement field; the controller stamps it
    setattr(t, 'io_placement', strategy)
    return t


def test_snn_default_random_gene_is_byte_identical():
    from substrates.snn.genome import random_gene, MAX_ITER
    random.seed(777)
    gene = random_gene()
    assert gene.tag == 0
    random.seed(777)
    draws = [random.randint(0, 15) for _ in range(5)]
    self_out = random.randint(1, 15)
    limit = MAX_ITER - random.randint(0, MAX_ITER // 3)
    assert (gene.state_n, gene.state_s, gene.state_e, gene.state_w,
            gene.self_in, gene.self_out, gene.limit) == (*draws, self_out, limit)


def test_snn_cell_io_tags_are_the_cell_states():
    from substrates.snn.growth import grow_snn, cell_io_tags
    g = _snn_genome(5)
    t = _snn_target('tag_rank')
    grid = grow_snn(
        g, seeds=iop.growth_seeds(t, iop.io_strategy(t), g),
        grid_size=t.grid_size)
    tags = cell_io_tags(g, grid)
    assert tags == {pos: int(state) for pos, state in grid.items()}


def test_snn_binding_marks_bound_neurons_and_scores():
    from substrates.snn.growth import grow_snn, cell_io_tags
    from substrates.snn.snn import interpret_grid
    from substrates.snn.ga import evaluate_genome
    g = _snn_genome(5, wiring=True)
    t = _snn_target()
    for strat in ('tag_rank', 'wiring_chromosome'):
        setattr(t, 'io_placement', strat)
        grid = grow_snn(
            g, seeds=iop.growth_seeds(t, iop.io_strategy(t), g),
            grid_size=t.grid_size)
        tags = cell_io_tags(g, grid)
        bound = iop.bind_io(g, grid, t, tags=tags)
        if bound is None:
            continue
        in_pos, out_pos = bound
        flat = iop.flat_inputs(in_pos)
        neurons, _ = interpret_grid(grid, target=t,
                                    input_pos=flat, output_pos=out_pos)
        marked_in = [(n.x, n.y) for n in neurons if n.is_input]
        marked_out = {(n.x, n.y) for n in neurons if n.is_output}
        assert sorted(marked_in) == sorted(flat)
        # every bound output cell is marked as an output (two roles sharing one
        # SNN neuron can only carry one role attribute, so check cells not roles)
        assert set(iop.flat_outputs(out_pos)) <= marked_out
        s = evaluate_genome(g, t)
        assert 0.0 <= s <= 1.0
    setattr(t, 'io_placement', 'fixed')


def test_snn_evolve_io_gates_port_map_mutations():
    from substrates.snn.ga import mutate
    g = _snn_genome(5, wiring=True)
    parent = [(gene.tag, gene.io_limit, gene.io_selector)
              for gene in iop.wiring_chromosome(g).genes]
    random.seed(0)
    for _ in range(200):
        m = mutate(g, evolve_io=False)
        child = [(gene.tag, gene.io_limit, gene.io_selector)
                 for gene in iop.wiring_chromosome(m).genes]
        assert child == parent
    random.seed(0)
    changed = False
    for _ in range(200):
        m = mutate(g, evolve_io=True)
        child = [(gene.tag, gene.io_limit, gene.io_selector)
                 for gene in iop.wiring_chromosome(m).genes]
        if child != parent:
            changed = True
            break
    assert changed, "SNN evolve_io never changed a mapping allele"


# ── nervous combinational path ───────────────────────────────────────────────────

def test_nervous_combinational_honors_strategy():
    from substrates.nervous.nervous import (interpret_nervous, _resolve_io_binding,
                                score_nervous)
    g = _tagged_genome(7)
    t = _snn_target('tag_rank')                  # Half adder truth table
    grid = _grown(g, t)                          # center-seeded under strategy
    routing, ip, op = interpret_nervous(grid, t)
    resolved = _resolve_io_binding(g, grid, t, ip, op)
    if resolved is not None:
        rin, rout = resolved
        assert rin != list(t.inputs) or rout != op   # binding actually moved
        assert set(rout) == {term.role for term in t.outputs}
    s = score_nervous(g, t)
    assert 0.0 <= s <= 1.0
    setattr(t, 'io_placement', 'fixed')


# ── checkpoint round-trip ────────────────────────────────────────────────────────

def test_checkpoint_roundtrips_tags_and_wiring_all_backends():
    from runtime.checkpoint import genome_to_dict, genome_from_dict
    import substrates.nervous.ga as nga
    import substrates.lut.ga as lga
    import substrates.snn.ga as sga
    cases = [
        ('nervous', _tagged_genome(9, wiring=True), nga.genome_signature),
        ('nervous', _spatial_genome(), nga.genome_signature),
        ('lut', _lut_genome(9, wiring=True), lga.genome_signature),
        ('snn', _snn_genome(9, wiring=True), sga.genome_signature),
    ]
    for backend, genome, signature in cases:
        restored = genome_from_dict(genome_to_dict(genome, backend), backend)
        assert signature(restored) == signature(genome), backend
        assert any(c.wiring for c in restored.chromosomes), backend


# ── center-seeded growth under evolvable strategies ─────────────────────────────

def test_checkpoint_migrates_retired_multisite_limits_to_one():
    from runtime.checkpoint import genome_to_dict, genome_from_dict
    genome = _tagged_genome(9, wiring=True)
    document = genome_to_dict(genome, 'nervous')
    limit_index = document['gene_fields'].index('io_limit')
    # Simulate files written while zero meant "all" and larger values meant
    # multi-site fan-out.
    document['chromosomes'][2]['genes'][0][limit_index] = 0
    document['chromosomes'][2]['genes'][1][limit_index] = 8
    restored = genome_from_dict(document, 'nervous')
    assert all(gene.io_limit == 1
               for gene in iop.wiring_chromosome(restored).genes)
    saved_again = genome_to_dict(restored, 'nervous')
    assert all(row[limit_index] == 1
               for row in saved_again['chromosomes'][2]['genes'])


def test_growth_seeds_pads_for_fixed_single_center_otherwise():
    tgt = coincidence_detector()
    assert iop.growth_seeds(tgt) == tuple(tgt.inputs)          # fixed = pads
    for strat in ('tag_rank', 'wiring_chromosome'):
        seeds = iop.growth_seeds(with_io_placement(tgt, strat))
        assert len(seeds) == 1                                  # ONE seed
        center = (tgt.grid_size // 2, tgt.grid_size // 2)
        assert seeds == (center,)
        assert center not in tgt.inputs                         # non-input cell


def test_strategy_growth_is_not_anchored_to_the_pads():
    """Under an evolvable strategy the organism nucleates from the center; the
    declared pads have no developmental role (they may or may not end up
    covered by the grown body)."""
    g = _tagged_genome(7)
    tgt = with_io_placement(coincidence_detector(), 'tag_rank')
    grid = _grown(g, tgt)
    center = (tgt.grid_size // 2, tgt.grid_size // 2)
    assert center in grid                                      # nucleus alive
    # the same genome grows a DIFFERENT body than the pad-seeded one
    pad_grid = grow_nervous(g, seeds=tuple(tgt.inputs))
    assert grid != pad_grid
    # and scoring/binding operate on the center-grown body end to end
    prep = prepare_net(g, tgt)
    if prep is not None:
        assert prep[0] == grid


def test_carry_physics_carries_io_placement():
    from substrates.nervous.certification import carry_physics
    src = with_io_placement(sr_latch(), 'tag_rank')
    dst = sr_latch()
    carry_physics(src, dst)
    assert getattr(dst, 'io_placement', 'fixed') == 'tag_rank'
    # fixed source leaves the destination untouched
    dst2 = sr_latch()
    carry_physics(sr_latch(), dst2)
    assert getattr(dst2, 'io_placement', 'fixed') == 'fixed'


def test_multi_site_input_injection_drives_every_attachment_cell():
    """A nervous input bound to several cells injects at ALL of them: the
    driven cone from the grouped binding equals the union of the single-site
    cones."""
    from substrates.nervous.temporal import input_cone
    from substrates.nervous.nervous import interpret_nervous
    tgt = with_io_placement(coincidence_detector(), 'tag_rank')
    g, grid, _ = _genome_expressing_values(
        tgt, tgt.n_inputs + tgt.n_outputs)
    routing, _, _ = interpret_nervous(grid, tgt)
    bound = iop.bind_io(g, grid, tgt)
    assert bound is not None
    in_pos, _ = bound
    cone = input_cone(grid, routing, in_pos)
    union = set()
    for grp in in_pos:
        for cell in grp:
            union |= input_cone(grid, routing, [cell])
    assert cone == union


def test_random_genomes_rarely_score_under_the_wiring_lock():
    """The regression behind the exact-match gate: with near-total immigration
    the per-generation best is essentially the best of a RANDOM sample, and it
    stayed constantly high because binding never actually failed. Under the
    cell-type lock, most random genomes fail to bind at all — their port map
    desires cell TYPES the grown body doesn't contain — so best-of-random
    collapses toward 0."""
    tgt = with_io_placement(coincidence_detector(), 'wiring_chromosome')
    random.seed(2024)
    scores = []
    for _ in range(24):
        g = random_hex_genome(3, wiring_chromosome=True)
        scores.append(score_temporal(g, tgt))
    zeros = sum(1 for s in scores if s == 0.0)
    # a large share of random genomes must have NO I/O at all
    assert zeros >= len(scores) * 0.4, scores
    assert max(scores) < 0.99, scores


def test_genome_text_shows_wiring_marker_and_binding():
    import ui.app as app_module
    g = _tagged_genome(7, wiring=True)
    binding = 'io_placement=wiring_chromosome\n  in[0] <- type 5 @ (0, 0)'
    text = app_module.build_genome_text(g, 0.5, binding=binding)
    assert '[WIRING chromosome - I/O port map]' in text
    assert 'iotag' in text                      # port-map column present
    assert 'I/O binding (evolved):' in text
    assert 'in[0] <- type 5' in text
    # a default (no-wiring) genome renders WITHOUT the column or footer
    plain = random_hex_genome(2)
    plain_text = app_module.build_genome_text(plain)
    assert 'iotag' not in plain_text and '[WIRING' not in plain_text


def test_old_checkpoint_without_tags_still_loads():
    from runtime.checkpoint import genome_to_dict, genome_from_dict
    g = _tagged_genome(9, wiring=True)
    d = genome_to_dict(g, 'nervous')
    # simulate a pre-tag checkpoint: strip the tag column and the wiring flag
    d['gene_fields'] = ['ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out']
    for c in d['chromosomes']:
        c.pop('wiring', None)
        c['genes'] = [row[:5] for row in c['genes']]
    old = genome_from_dict(d, 'nervous')
    assert all(gene.tag == 0 for c in old.chromosomes for gene in c.genes)
    assert all(not c.wiring for c in old.chromosomes)
