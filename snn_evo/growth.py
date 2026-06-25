from __future__ import annotations
from typing import Dict, Tuple
from .genome import Genome, MAX_ITER, GRID_SIZE

SEED_A     = (0, 3)
SEED_B     = (0, 5)
SEED_STATE = 1

_PC4 = [bin(i).count("1") for i in range(16)]

def _h4(a: int, b: int) -> int:
    return _PC4[(a ^ b) & 0xF]

def _lookup(genome: Genome, sn: int, ss: int, se: int, sw: int,
            si: int, iteration: int) -> int:
    if sn == 0 and ss == 0 and se == 0 and sw == 0 and si == 0:
        return 0
    best_out, best_dist = 0, 999999
    for chrom in genome.chromosomes:
        for gene in chrom.genes:
            if iteration > gene.limit:
                continue
            d = (_h4(gene.state_n, sn) + _h4(gene.state_s, ss) +
                 _h4(gene.state_e, se) + _h4(gene.state_w, sw) +
                 _h4(gene.self_in,  si))
            if d < best_dist:
                best_dist, best_out = d, gene.self_out
    return best_out if best_out else 1

def grow_snn(genome: Genome) -> Dict[Tuple[int, int], int]:
    grid: Dict[Tuple[int, int], int] = {SEED_A: SEED_STATE, SEED_B: SEED_STATE}
    for iteration in range(MAX_ITER):
        frontier: Dict[Tuple[int, int], int] = {}
        for (x, y) in list(grid):
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x+dx, y+dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
                        and (nx,ny) not in grid and (nx,ny) not in frontier):
                    frontier[(nx, ny)] = 0
        working = {**grid, **frontier}
        next_grid: Dict[Tuple[int, int], int] = {}
        for (x, y), state in working.items():
            sn = working.get((x, y+1), 0); ss = working.get((x, y-1), 0)
            se = working.get((x+1, y),  0); sw = working.get((x-1, y), 0)
            ns = _lookup(genome, sn, ss, se, sw, state, iteration)
            if ns:
                next_grid[(x, y)] = ns
        next_grid[SEED_A] = SEED_STATE
        next_grid[SEED_B] = SEED_STATE
        grid = next_grid
    return grid
