"""Tests for the network -> genome reconstruction (best-effort inverse of Grow).

The reconstruction cannot invert every organism (growth is many-to-one), but it
guarantees two things the designer relies on:

  * a clean, context-consistent hand-built grid regrows EXACTLY (cell-for-cell,
    no extras); and
  * whatever it does reproduce, it reproduces faithfully — the report's `matched`
    count never overstates what regrowing the genome actually yields.

Both backends are covered, plus the guard rails (no seeds).
"""
import random
import unittest

from nv_evo.genome import (HexGene, Genome as NvGenome,
                           Chromosome as NvChromosome, random_hex_genome)
from nv_evo.nervous import grow_nervous, SEED_STATE as NV_SEED, _grow_step
from nv_evo.reverse import (grid_to_genome_nervous, repair_genome_nervous,
                            _ctx)

from lut_evo.genome import Genome, Chromosome, LutGene
from lut_evo.lut import grow_lut, SEED_STATE as LUT_SEED
from lut_evo.reverse import grid_to_genome_lut, repair_genome_lut


class NervousReverseTests(unittest.TestCase):
    def test_delta_repair_absorbs_local_edit_without_damaging_bulk(self):
        seeds = [(0, 0)]
        original = {(0, 0): 1, (1, 0): 2, (2, 0): 3, (0, 1): 4}
        source, _ = grid_to_genome_nervous(original, seeds)
        edited = dict(original)
        edited[(1, 0)] = 3
        repaired, rep = repair_genome_nervous(source, edited, seeds)
        self.assertEqual(grow_nervous(repaired, seeds=tuple(seeds)), edited)
        self.assertTrue(rep['exact'])
        self.assertEqual(rep['unchanged_preserved'], rep['unchanged'])
        self.assertEqual(rep['edits_reproduced'], rep['edits'])
        self.assertGreater(rep['added_genes'], 0)

    def test_delta_repair_refuses_edit_that_would_damage_unchanged_cells(self):
        random.seed(0)
        seeds = [(0, 0), (2, 0)]
        source = random_hex_genome(2)
        original = grow_nervous(source, seeds=tuple(seeds))
        pos = next(p for p in original if p not in seeds)
        edited = dict(original)
        edited[pos] = (edited[pos] % 31) + 1
        repaired, rep = repair_genome_nervous(source, edited, seeds)
        regrown = grow_nervous(repaired, seeds=tuple(seeds))
        self.assertEqual(rep['unchanged_preserved'], rep['unchanged'])
        self.assertEqual(regrown, original)
        self.assertEqual(rep['added_genes'], 0)
        self.assertFalse(rep['exact'])

    def test_any_static_grid_with_valid_seeds_can_be_made_a_fixed_point(self):
        # Reachability is the inverse problem, not stability. Even a snapshot
        # taken from a transient source trajectory can be held by exact local
        # maintenance rules once it has been reached.
        seeds = [(0, 0)]
        grid = {(0, 0): NV_SEED, (1, 0): 7, (2, 0): 19, (0, 1): 4}
        genes = []
        for (x, y), state in grid.items():
            cl, cr, cd = _ctx(grid, x, y)
            genes.append(HexGene(cl, cr, cd, state, state))
        genome = NvGenome([NvChromosome(genes=genes, telomere=1)])
        tel = {p: 0 for p in grid}
        tel[seeds[0]] = 1
        nxt, _ = _grow_step(genome, grid, tel, seeds, 1, {})
        self.assertEqual(nxt, grid)

    def test_handbuilt_chain_is_exact(self):
        grid = {(0, 0): 1, (1, 0): 2, (2, 0): 3, (0, 1): 4}
        seeds = [(0, 0)]
        genome, rep = grid_to_genome_nervous(grid, seeds)
        regrown = grow_nervous(genome, seeds=tuple(seeds))
        self.assertEqual(regrown, grid)
        self.assertTrue(rep['exact'])
        self.assertEqual(rep['matched'], rep['target'])
        self.assertEqual(rep['telomere'], rep['radius'])
        self.assertEqual(rep['extra'], 0)

    def test_report_matched_never_overstates(self):
        # across many evolved grids the reconstruction is only best-effort, but
        # the reported `matched` must equal what regrowing actually reproduces
        for i in range(25):
            random.seed(i)
            g = random_hex_genome(2)
            seeds = [(0, 0), (2, 0)]
            grid = grow_nervous(g, seeds=tuple(seeds))
            if len(grid) <= len(seeds):
                continue
            genome, rep = grid_to_genome_nervous(grid, seeds)
            regrown = grow_nervous(genome, seeds=tuple(seeds))
            true_match = sum(1 for p, s in grid.items() if regrown.get(p) == s)
            self.assertEqual(rep['matched'], true_match, 'seed %d' % i)
            self.assertLessEqual(rep['matched'], rep['target'])

    def test_multistep_trajectory_beats_direct_replay_on_collision_case(self):
        random.seed(0)
        seeds = [(0, 0), (2, 0)]
        grid = grow_nervous(random_hex_genome(2), seeds=tuple(seeds))
        _, direct = grid_to_genome_nervous(grid, seeds, use_multistep=False)
        _, multi = grid_to_genome_nervous(grid, seeds)
        self.assertEqual(multi['strategy'], 'multi-step')
        self.assertTrue(multi['intermediate_states'])
        self.assertGreater(multi['matched'], direct['matched'])

    def test_no_seeds_is_reported_not_crash(self):
        genome, rep = grid_to_genome_nervous({(0, 0): 1, (1, 0): 2}, seeds=[])
        self.assertIn('note', rep)
        self.assertFalse(rep['exact'])
        self.assertIsNotNone(genome)

    def test_true_fixed_point_reproduces_well(self):
        # a grid that its own genome holds STILL (a real fixed point) should
        # reconstruct with full or near-full cover — the closed-loop repair lands
        # on the attractor. Build one by growing until stepping does not change it.
        from nv_evo.nervous import _grow_step
        from nv_evo.genome import germline_telomere
        seeds = [(0, 0), (2, 0)]
        found = 0
        for i in range(60):
            random.seed(i)
            g = random_hex_genome(2)
            grid = grow_nervous(g, seeds=tuple(seeds))
            if len(grid) <= 4:
                continue
            L = germline_telomere(g)
            tel = {p: 0 for p in grid}
            for s in seeds:
                tel[s] = L
            nxt, _ = _grow_step(g, grid, tel, seeds, L, {})
            if nxt != grid:                       # not a fixed point — skip
                continue
            found += 1
            _, rep = grid_to_genome_nervous(grid, seeds)
            self.assertGreaterEqual(rep['matched'] / rep['target'], 0.75,
                                    'fixed-point seed %d only %d/%d'
                                    % (i, rep['matched'], rep['target']))
        self.assertGreater(found, 0)              # the search actually found some

    def test_benchmark_no_regression(self):
        # guards the closed-loop improvement: mean cell-match over the standard
        # random-grid suite must stay well above the old monotone-only baseline
        # (~0.69 direct replay, ~0.77 direct + repair). Multi-step development
        # should remain materially above both without overstating the phenotype.
        seeds = [(0, 0), (2, 0)]
        fracs, overstate = [], 0
        for i in range(60):
            random.seed(i)
            g = random_hex_genome(2)
            grid = grow_nervous(g, seeds=tuple(seeds))
            if len(grid) <= len(seeds):
                continue
            genome, rep = grid_to_genome_nervous(grid, seeds)
            regrown = grow_nervous(genome, seeds=tuple(seeds))
            true_match = sum(1 for p, s in grid.items() if regrown.get(p) == s)
            if rep['matched'] != true_match:
                overstate += 1
            fracs.append(rep['matched'] / rep['target'])
        self.assertEqual(overstate, 0)
        self.assertGreaterEqual(sum(fracs) / len(fracs), 0.88)

    def test_seed_state_mismatch_reported(self):
        # a seed held at a non-seed state cannot be reproduced (seeds are pinned)
        grid = {(0, 0): 7}                    # seed wants state 7, growth pins 1
        genome, rep = grid_to_genome_nervous(grid, seeds=[(0, 0)])
        self.assertIn((0, 0), rep['seed_mismatch'])


class LutReverseTests(unittest.TestCase):
    def test_delta_repair_no_edit_preserves_original_genome(self):
        relay = (0xFFFE, 0xFFFE, 0xFFFE, 0xFFFE)
        seeds = [(0, 0)]
        grid = {(0, 0): LUT_SEED, (1, 0): relay,
                (0, 1): relay, (1, 1): relay}
        source, _ = grid_to_genome_lut(grid, seeds, grid_size=5, iters=20)
        baseline = grow_lut(source, seeds=tuple(seeds), grid_size=5, iters=20)
        repaired, rep = repair_genome_lut(
            source, baseline, seeds, grid_size=5, iters=20)
        self.assertIs(repaired, source)
        self.assertTrue(rep['exact'])
        self.assertEqual(rep['edits'], 0)
        self.assertEqual(rep['added_genes'], 0)

    def test_handbuilt_block_full_cover(self):
        # a 2x2 block of relay cells: every target cell reproduces (extras — the
        # isotropic seed also grows outward — are allowed by the contract)
        relay = (0xFFFE, 0xFFFE, 0xFFFE, 0xFFFE)
        grid = {(0, 0): LUT_SEED, (1, 0): relay, (0, 1): relay, (1, 1): relay}
        seeds = [(0, 0)]
        genome, rep = grid_to_genome_lut(grid, seeds, grid_size=5, iters=20)
        regrown = grow_lut(genome, seeds=tuple(seeds), grid_size=5, iters=20)
        for p, st in grid.items():
            self.assertEqual(regrown.get(p), st, 'cell %s' % (p,))
        self.assertEqual(rep['matched'], rep['target'])
        self.assertEqual(rep['telomere'], rep['radius'])

    def test_report_matched_never_overstates(self):
        for i in range(15):
            random.seed(200 + i)
            genes = [LutGene(ctx_n=random.randrange(1 << 16),
                             ctx_e=random.randrange(1 << 16),
                             ctx_s=random.randrange(1 << 16),
                             ctx_w=random.randrange(1 << 16),
                             self_in=0 if random.random() < 0.4 else random.randrange(1 << 16),
                             self_out=random.randrange(1 << 16)) for _ in range(6)]
            gen = Genome(chromosomes=[Chromosome(genes=genes, split=0, tag=1,
                                                telomere=random.randint(2, 4))], tag=1)
            seeds = [(0, 0)]
            grid = grow_lut(gen, seeds=tuple(seeds), grid_size=7, iters=20)
            if len(grid) <= 1:
                continue
            genome, rep = grid_to_genome_lut(grid, seeds, grid_size=7, iters=20)
            regrown = grow_lut(genome, seeds=tuple(seeds), grid_size=7, iters=20)
            true_match = sum(1 for p, s in grid.items() if regrown.get(p) == s)
            self.assertEqual(rep['matched'], true_match, 'seed %d' % i)

    def test_no_seeds_is_reported_not_crash(self):
        genome, rep = grid_to_genome_lut({(0, 0): LUT_SEED}, seeds=[])
        self.assertIn('note', rep)
        self.assertIsNotNone(genome)


if __name__ == '__main__':
    unittest.main()
