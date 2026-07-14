from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

from evo_runtime.limits import MAX_CHROMOSOME_COUNT

MAX_ITER     = 12        # legacy per-gene time cap (kept only for pickle compat)
GRID_SIZE    = 9         # field size — a generous OUTER bound; the telomere is
                         # now the real limiter and normally halts growth first
MAX_STATE    = 16
MAX_GENES    = 24
MAX_CHROMS   = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 16        # longest evolvable growth radius (Hayflick divisions)

@dataclass
class Gene:
    state_n:  int = 0
    state_s:  int = 0
    state_e:  int = 0
    state_w:  int = 0
    self_in:  int = 0
    self_out: int = 1
    limit:    int = MAX_ITER      # unused by growth now; retained for old pickles

@dataclass
class Chromosome:
    """`telomere` is a HAYFLICK division limit, mirroring nv_evo: seed cells start
    with the germline length L, and each division hands the daughter one less. A
    cell at telomere 0 is senescent — still alive and functional, but it can no
    longer birth a neighbour, so growth self-limits at radius L from the seeds
    (replacing the old fixed iteration cap). Telomere length is evolvable, so the
    genome decides how large it grows; the GRID_SIZE walls are just an outer
    safety bound the telomere normally stays inside."""
    genes: List[Gene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0
    telomere: int = MAX_TELOMERE

@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0


def germline_telomere(genome) -> int:
    """The organism's germline telomere L: the longest division program across
    its chromosomes (seed cells start here). No chromosomes → 1 (seeds only)."""
    return max((getattr(c, 'telomere', 1) for c in genome.chromosomes), default=1)

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
        split = (0 if n_genes < 2 else random.randint(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        # Start generous so the organism spans the I/O layout and stays solvable;
        # parsimony then shrinks the telomere toward the smallest body that still
        # solves (it is the telomere, not the grid, that limits a mature circuit).
        telomere = random.randint(9, min(14, MAX_TELOMERE)),
    )

def random_genome(n_chroms=1) -> Genome:
    return Genome(
        chromosomes = [random_chromosome() for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
