"""
tests/test_tritile.py — the paper's THREE-circuit tile (substrates/nervous/tritile.py).

Edwards EH'02 Fig. 2 puts THREE independent nervous circuits in every tile, one
per output direction (L/R/D). The legacy engine collapses that to one circuit /
one broadcast output net. These tests defend the directional tile topology:

  * geometry — the honeycomb back-direction used to wire cross-tile signals is
    unique and mutual;
  * expansion — a grown tri grid becomes three single-circuit sub-nodes per
    live tile, simulated on the unchanged PulseSim via its pre-resolved sources;
  * CAPABILITY — a single tri tile routes two independent signals to two
    different outputs WITHOUT merging them, which one broadcast state cannot do;
  * determinism, and that the whole GA path (grow → interpret → mutate →
    reproduce) stays homogeneously tri3 and explores the full 12-bit alphabet;
  * isolation — the 'single' tile path is completely unaffected (arch defaults
    to 'single' everywhere).

Run under pytest, or standalone:  py tests/test_tritile.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrates.nervous.hexgrid import hex_dirs, _ROUTING_BASE               # noqa: E402
from substrates.nervous.pulse import PulseConfig                             # noqa: E402
from substrates.nervous.tritile import (back_dir, channel_configs, interpret_tri,
                            widen_legacy_state,  # noqa: E402
                            pack_channels, TriSim, TRI_SEED_STATE, TRI_DIRS,
                            _MergedView)
from substrates.nervous.genome import (Genome, Chromosome, HexGene,          # noqa: E402
                           random_hex_genome, TRI_STATE_MAX)
from substrates.nervous.nervous import grow_nervous                          # noqa: E402
from substrates.nervous.nervous import interpret_nervous                     # noqa: E402
from substrates.nervous.simulation import run_schedule                        # noqa: E402
from substrates.nervous.playback import NervousPlayer                         # noqa: E402
from substrates.nervous.ga import (eval_batch_cases, next_population,        # noqa: E402
                       genome_signature, mutate_nv, clone_genome)
from substrates.nervous.targets import coincidence_detector                  # noqa: E402
from runtime.config import GAConfig                          # noqa: E402


def _pack(cL, cR, cD):
    # Independent of tritile.pack_channels: three 5-bit channel fields.
    return cL | (cR << 5) | (cD << 10)


# ── geometry ────────────────────────────────────────────────────────────────────

def test_back_direction_is_unique_and_mutual():
    for x in range(-3, 4):
        for y in range(-3, 4):
            for d, (nx, ny) in hex_dirs(x, y).items():
                bd = back_dir((x, y), d)
                assert hex_dirs(nx, ny)[bd] == (x, y)


def test_channel_unpacking_roundtrips():
    for _ in range(200):
        cL, cR, cD = (random.randrange(32) for _ in range(3))
        assert channel_configs(_pack(cL, cR, cD)) == (cL, cR, cD)
        assert pack_channels(cL, cR, cD) == _pack(cL, cR, cD)


def test_tile_readout_is_a_wired_or_not_three_event_counters():
    tile = (1, 1)
    nodes = {tile: [('n', 'L'), ('n', 'R'), ('n', 'D')]}
    intervals = {
        ('n', 'L'): [[1.0, 3.0]],
        ('n', 'R'): [[1.0, 2.0], [4.0, 5.0]],
        ('n', 'D'): [[2.5, 4.5]],
    }
    assert _MergedView(intervals, nodes, 'intervals').get(tile) == [(1.0, 5.0)]
    assert _MergedView(intervals, nodes, 'rises').get(tile) == [1.0]


# ── expansion ────────────────────────────────────────────────────────────────────

def test_expansion_makes_three_subnodes_per_noninput_tile():
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): TRI_SEED_STATE,
            (1, 0): _pack(1, 4, 2)}
    inputs = [(0, 0)]
    info = interpret_tri(grid, inputs)
    # one IN node for the input tile, three circuit sub-nodes for each other tile
    assert info['in_nodes'] == {(0, 0): (0, 0, 'IN')}
    for tile in ((0, 1), (1, 0)):
        subs = [n for n in info['nodes'] if n[:2] == tile]
        assert sorted(s[2] for s in subs) == ['D', 'L', 'R']


def test_offgrid_and_off_channels_have_no_live_source():
    # A channel set to config 0 (off) selects no excitatory input.
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): _pack(0, 0, 1)}
    info = interpret_tri(grid, [(0, 0)])
    # channels L,R are off -> their sub-nodes have no excitatory sources
    for d in ('L', 'R'):
        s1, s2, si = info['sources'][(0, 1, d)]
        assert s1 is None and s2 is None


# ── capability: independent routing a single broadcast state cannot express ──────

def test_tile_routes_two_signals_to_two_outputs_independently():
    """The load-bearing fidelity claim. Build ONE relay tile P fed by two
    distinct input terminals A and B on two of its input directions. Configure
    P's channels so output-1 buffers ONLY A's direction and output-2 buffers
    ONLY B's direction. A pulse on A must reach output-1 and NOT output-2, and
    vice-versa — three circuits acting independently on the same tile. A single
    broadcast-state tile has one output for all neighbours and cannot do this.
    """
    P = (1, 1)
    dirs = hex_dirs(*P)                       # P's three neighbour tiles
    A, B = dirs['L'], dirs['R']               # two input terminals
    outL_tile = dirs['D']                     # where P's 'L'-frame buffer of L points...
    # Configure: channel D buffers input direction L (reads A); channel R
    # buffers input direction R (reads B). _ROUTING_BASE index: buffer L = 3,
    # buffer R = 2. Channel L off.
    cfg_L, cfg_R, cfg_D = 0, 2, 3             # chanR=buffer R(reads B), chanD=buffer L(reads A)
    grid = {A: TRI_SEED_STATE, B: TRI_SEED_STATE, P: _pack(cfg_L, cfg_R, cfg_D)}
    inputs = [A, B]
    info = interpret_tri(grid, inputs)
    # sub-node (P,'D') should read A's IN; (P,'R') should read B's IN
    assert info['sources'][(P[0], P[1], 'D')][0] == (A[0], A[1], 'IN')
    assert info['sources'][(P[0], P[1], 'R')][0] == (B[0], B[1], 'IN')

    # Pulse A only -> (P,'D') fires, (P,'R') silent.
    simA = TriSim(grid, inputs, config=PulseConfig())
    simA.inject_pulse(A, 0.0, 1.0)
    simA.advance_to(10.0)
    innerA = simA._sim
    assert innerA.rise_times[(P[0], P[1], 'D')], 'output fed by A did not fire'
    assert not innerA.rise_times[(P[0], P[1], 'R')], 'output fed by B fired on A'

    # Pulse B only -> (P,'R') fires, (P,'D') silent — the SAME tile, other channel.
    simB = TriSim(grid, inputs, config=PulseConfig())
    simB.inject_pulse(B, 0.0, 1.0)
    simB.advance_to(10.0)
    innerB = simB._sim
    assert innerB.rise_times[(P[0], P[1], 'R')], 'output fed by B did not fire'
    assert not innerB.rise_times[(P[0], P[1], 'D')], 'output fed by A fired on B'


def test_trisim_determinism():
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): TRI_SEED_STATE,
            (0, 2): TRI_SEED_STATE}
    inputs = [(0, 0)]
    runs = []
    for _ in range(3):
        sim = TriSim(grid, inputs, config=PulseConfig())
        sim.inject_pulse((0, 0), 0.0, 1.0)
        sim.advance_to(20.0)
        runs.append({t: tuple(sim.rise_times.get(t, ())) for t in grid})
    assert runs[0] == runs[1] == runs[2]


def test_shared_schedule_and_playback_paths_honor_tri_architecture():
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): TRI_SEED_STATE}
    inputs = [(0, 0)]
    routing, _, _ = interpret_nervous(grid, arch='tri3')
    assert routing == {}                       # never decode a fake 5-bit arrow
    schedule = [[(0.0, 1.0)]]
    rises, overflow = run_schedule(
        grid, routing, inputs, schedule, 10.0, [(0, 1)],
        config=PulseConfig(), arch='tri3')
    assert not overflow and rises[(0, 1)]

    player = NervousPlayer(
        grid, routing, horizon=10.0, config=PulseConfig(),
        arch='tri3', inputs=inputs)
    player.set_schedule({inputs[0]: schedule[0]})
    player.sim.advance_to(10.0)
    assert player.events_upto((0, 1))


def test_three_output_tile_composes_with_analog_node_physics():
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): TRI_SEED_STATE}
    inputs = [(0, 0)]
    sim = TriSim(grid, inputs, config=PulseConfig(model='paper_analog'))
    sim.inject_pulse(inputs[0], 0.0, 1.0)
    sim.advance_to(20.0)
    assert sim.rise_times.get((0, 1))


# ── GA path ──────────────────────────────────────────────────────────────────────

def test_tri_growth_uses_fifteen_bit_states():
    # Seeded: an unseeded random genome inherits whatever RNG state the previous
    # test left behind, and some of those grow nothing beyond the seed tile.
    random.seed(4242)
    g = random_hex_genome(2, arch='tri3')
    grid = grow_nervous(g, seeds=((0, 0),))
    assert len(grid) > 1
    # seed tile carries the tri seed state (all-buffer), a 15-bit value
    assert grid[(0, 0)] == TRI_SEED_STATE


def test_clone_and_mutation_preserve_arch_and_explore_full_alphabet():
    random.seed(11)
    g = random_hex_genome(2, arch='tri3')
    assert clone_genome(g).arch == 'tri3'
    seen_high = False
    child = g
    for _ in range(60):
        child = mutate_nv(child, mean_mutations=4.0)
        assert child.arch == 'tri3'
        if any(gg.self_out > 31 for c in child.chromosomes for gg in c.genes):
            seen_high = True
    assert seen_high, 'mutation never reached a state above the 5-bit range'


def test_genome_signature_separates_arch():
    gs = random_hex_genome(2, arch='single')
    gt = clone_genome(gs)
    gt.arch = 'tri3'
    assert genome_signature(gs) != genome_signature(gt)


def test_crossover_rejects_mixed_architectures():
    from substrates.nervous.ga import crossover_nv
    gs = random_hex_genome(2, arch='single')
    gt = random_hex_genome(2, arch='tri3')
    try:
        crossover_nv(gs, gt)
    except ValueError:
        return
    raise AssertionError('mixed-architecture crossover was accepted')


def test_generation_stays_homogeneously_tri3():
    random.seed(5)
    tgt = coincidence_detector()
    pop = [random_hex_genome(2, arch='tri3') for _ in range(16)]
    ga = GAConfig(node_model='uniform', tile_arch='tri3', chromosome_count=2)
    cache = {}
    for _ in range(4):
        fits, cases = eval_batch_cases(pop, tgt, cache)
        pop = next_population(pop, fits, ga_config=ga, chromosome_count=2,
                              mean_mutations=4.0, case_vecs=cases)
        assert all(getattr(gm, 'arch', 'single') == 'tri3' for gm in pop)


# ── isolation: the single-tile path is untouched ────────────────────────────────

def test_single_arch_is_default_and_unchanged():
    g = random_hex_genome(2)
    assert getattr(g, 'arch', 'single') == 'single'
    # a single genome grows with 5-bit states only
    grid = grow_nervous(g, seeds=((0, 0),))
    assert all(0 <= s < 32 for s in grid.values())


# ── standalone runner ────────────────────────────────────────────────────────────

def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except AssertionError as e:
            print("FAIL  %s\n      %s" % (fn.__name__, e))
        except Exception as e:                     # noqa: BLE001
            print("ERROR %s\n      %r" % (fn.__name__, e))
    print("\n%d/%d tri-tile tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())


# ── OR twins + the 4-bit -> 5-bit channel migration ─────────────────────────────

def test_tri_channels_reach_the_or_twins():
    """A tri channel indexes the full 32-value alphabet, not just the AND half.

    The tile was AND-only until measurement showed it could not hold a
    circulating pulse: under pure coincidence a lone pulse survives a node only
    where that channel happens to be a buffer, so Oscillator and Pattern were
    unreachable while input-driven targets scored at or above the single-tile
    arch. The OR twins restore the cheap re-fire path.
    """
    and_tile = pack_channels(1, 1, 1)            # buffer-D on every channel
    or_tile = pack_channels(1 + 16, 1, 1)        # L channel promoted to its twin
    grid = {(0, 0): TRI_SEED_STATE, (0, 1): and_tile, (0, 2): or_tile}
    ops = {key: node[3]
           for key, node in interpret_tri(grid, [(0, 0)])['nodes'].items()}
    assert ops[(0, 1, 'L')] == 'and'
    assert ops[(0, 2, 'L')] == 'or'
    assert ops[(0, 2, 'R')] == 'and'             # untouched channels stay AND


def test_every_channel_config_is_a_valid_routing_index():
    for config in range(32):
        tile = pack_channels(config, config, config)
        assert channel_configs(tile) == (config, config, config)
    for bad in (-1, 32):
        try:
            pack_channels(bad, 0, 0)
        except ValueError:
            continue
        raise AssertionError('accepted out-of-range channel %d' % bad)


def test_widening_a_legacy_state_preserves_its_three_channels():
    # Legacy configs are all 0-15 (the AND half), so a widened genome routes
    # identically — the migration only re-lays the bit fields.
    for state in range(4096):
        legacy = (state & 0xF, (state >> 4) & 0xF, (state >> 8) & 0xF)
        assert channel_configs(widen_legacy_state(state)) == legacy


def test_legacy_tri_checkpoint_is_widened_exactly_once():
    from runtime.checkpoint import genome_from_dict, genome_to_dict
    legacy_gene = [0x111, 0x222, 0x333, 0x000, 0x123]     # 12-bit states
    saved = {'tag': 0, 'arch': 'tri3',
             'gene_fields': ['ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'],
             'chromosomes': [{'tag': 0, 'split': 0, 'telomere': 4,
                              'genes': [legacy_gene]}],
             'state_delays': None}                        # no tri_channel_bits
    genome = genome_from_dict(saved, 'nervous')
    gene = genome.chromosomes[0].genes[0]
    assert channel_configs(gene.ctx_l) == (1, 1, 1)
    assert channel_configs(gene.self_out) == (3, 2, 1)

    # A re-saved genome carries the new width and must NOT widen again.
    again = genome_from_dict(genome_to_dict(genome, 'nervous'), 'nervous')
    once = again.chromosomes[0].genes[0]
    assert (once.ctx_l, once.self_out) == (gene.ctx_l, gene.self_out)
