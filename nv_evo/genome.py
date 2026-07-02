"""
nv_evo/genome.py — genome native to the hexagonal nervous net.

Fully self-contained: the nervous net shares *no* code with snn_evo. The SNN's
Gene has four neighbour fields (N/S/E/W) and forbids self_out==0; the hex node
has only three directions (L/R/D) and *wants* to express "off" (a dead /
unrouted cell), so it gets its own gene and its own container types:

    HexGene(ctx_l, ctx_r, ctx_d, self_in, self_out)
        ctx_*     expected neighbour state in each hex direction (context)
        self_in   expected own state (context)
        self_out  state to emit if this gene wins   (0 = off / cell dies)

This mirrors the paper's associative-memory gene exactly: a context (the states
of the neighbouring cells + the cell's own state) mapped to a new state, chosen
by minimum Hamming distance. There is no per-gene time field — the paper's genes
are timeless; growth is bounded only by the fixed iteration count (one of the
paper's acknowledged "artificial methods" of limiting growth, §7).

The genetic operators (mutation, crossover, selection) live in nv_evo/ga.py.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

MAX_STATE  = 16
MAX_GENES  = 24
MAX_CHROMS = 6


@dataclass
class HexGene:
    ctx_l:    int = 0
    ctx_r:    int = 0
    ctx_d:    int = 0
    self_in:  int = 0
    self_out: int = 0        # 0 = off / dead (native — no shift needed)


@dataclass
class Chromosome:
    genes: List[HexGene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0


def random_hex_gene() -> HexGene:
    return HexGene(
        ctx_l    = random.randrange(MAX_STATE),
        ctx_r    = random.randrange(MAX_STATE),
        ctx_d    = random.randrange(MAX_STATE),
        self_in  = random.randrange(MAX_STATE),
        self_out = random.randrange(MAX_STATE),      # 0..15, 0 = death
    )


def random_hex_chromosome(n_genes=None) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_hex_gene() for _ in range(n_genes)],
        split = random.randint(1, max(1, n_genes - 1)),
        tag   = random.randint(0, 999),
    )


def random_hex_genome(n_chroms=2) -> Genome:
    return Genome(
        chromosomes = [random_hex_chromosome() for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
