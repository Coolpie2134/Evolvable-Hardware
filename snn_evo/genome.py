from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

MAX_ITER   = 12
GRID_SIZE  = 9
MAX_STATE  = 16
MAX_GENES  = 24
MAX_CHROMS = 6

@dataclass
class Gene:
    state_n:  int = 0
    state_s:  int = 0
    state_e:  int = 0
    state_w:  int = 0
    self_in:  int = 0
    self_out: int = 1
    limit:    int = MAX_ITER

@dataclass
class Chromosome:
    genes: List[Gene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0

@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0

def random_gene() -> Gene:
    return Gene(
        state_n  = random.randint(0, MAX_STATE - 1),
        state_s  = random.randint(0, MAX_STATE - 1),
        state_e  = random.randint(0, MAX_STATE - 1),
        state_w  = random.randint(0, MAX_STATE - 1),
        self_in  = random.randint(0, MAX_STATE - 1),
        self_out = random.randint(1, MAX_STATE - 1),
        limit    = MAX_ITER - random.randint(0, MAX_ITER // 3),
    )

def random_chromosome(n_genes=None) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_gene() for _ in range(n_genes)],
        split = random.randint(1, max(1, n_genes - 1)),
        tag   = random.randint(0, 999),
    )

def random_genome(n_chroms=1) -> Genome:
    return Genome(
        chromosomes = [random_chromosome() for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
