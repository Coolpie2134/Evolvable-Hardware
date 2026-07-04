"""
lut_evo/genome.py — genome for the boolean-logic lookup (LUT) array.

Paper Architecture 2 (Edwards EH'02 §5-6; sim6 reference): each cell of a
square, 4-neighbour grid holds four 16-bit lookup tables (one per output
direction). The associative-memory gene is Fig. 10 verbatim: a context of five
16-bit LUT states (the four neighbours' facing LUTs + the cell's own LUT for
that direction) mapped to a new 16-bit LUT, chosen by minimum Hamming distance
over the full 80-bit context. During growth each direction is looked up with
the context rotated to that direction (see lut.py). self_out == 0 (the all-zero
LUT) means that direction is dead; a cell with all four LUTs zero is removed.
Genes are timeless.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

LUT_BITS     = 16
LUT_STATES   = 1 << LUT_BITS     # 65536 possible cell states
MAX_GENES    = 24
MAX_CHROMS   = 6
MAX_TELOMERE = 20                # longest evolvable growth phase (iterations)


@dataclass
class LutGene:
    ctx_n:    int = 0
    ctx_e:    int = 0
    ctx_s:    int = 0
    ctx_w:    int = 0
    self_in:  int = 0
    self_out: int = 0            # 0 = off / dead


@dataclass
class Chromosome:
    """`telomere`: growth clock — the chromosome's growth rules (self_in == 0)
    only apply while iteration < telomere (sim6's set_limit caps exactly those
    genes); maintenance rules never expire. Evolvable, so the genome bounds its
    own size on the unbounded field."""
    genes: List[LutGene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0
    telomere: int = MAX_TELOMERE


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0


def random_lut_gene() -> LutGene:
    # self_in == 0 makes a GROWTH rule — the only kind that can bring a dead
    # direction to life under the sim6 empty-cell guard, and the kind expired
    # by telomeres. With uniform 16-bit self_in the chance of one is ~1/65536,
    # so random genomes must be seeded with them explicitly (sim6's
    # table_create manufactures them for every empty context it meets).
    return LutGene(
        ctx_n    = random.randrange(LUT_STATES),
        ctx_e    = random.randrange(LUT_STATES),
        ctx_s    = random.randrange(LUT_STATES),
        ctx_w    = random.randrange(LUT_STATES),
        self_in  = 0 if random.random() < 0.25 else random.randrange(LUT_STATES),
        self_out = random.randrange(LUT_STATES),
    )


def random_lut_chromosome(n_genes=None) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_lut_gene() for _ in range(n_genes)],
        split = random.randint(1, max(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, 5),
    )


def random_lut_genome(n_chroms=2) -> Genome:
    return Genome(
        chromosomes = [random_lut_chromosome() for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
