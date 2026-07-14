"""
nv_evo/genome.py — genome native to the hexagonal nervous net.

Fully self-contained: the nervous net shares *no* code with snn_evo. A physical
tile contains three core circuits, but a gene acts on ONE circuit at a time.
Every state field is therefore exactly the paper's 4-bit Figure-3 selector:

    HexGene(ctx_l, ctx_r, ctx_d, self_in, self_out)
        ctx_*     expected facing-circuit state in each rotated direction
        self_in   expected own circuit state (context)
        self_out  circuit state to emit if this gene wins (0 = off)

This mirrors the paper's associative-memory gene exactly: a context (the states
of the neighbouring cells + the cell's own state) mapped to a new state, chosen
by minimum Hamming distance. There is no per-gene time field — the paper's genes
are timeless. Growth is bounded biologically instead of by the paper's "artificial
methods" (§7): each chromosome carries a telomere (a Hayflick division limit) so
the organism halts its own growth (see Chromosome + nv_evo/nervous.py).

The genetic operators (mutation, crossover, selection) live in nv_evo/ga.py.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

MAX_STATE    = 16       # one 4-bit Figure-3 core-circuit configuration
MAX_GENES    = 24
MAX_CHROMS   = 6
MAX_TELOMERE = 20        # longest evolvable growth phase (iterations)


@dataclass
class HexGene:
    ctx_l:    int = 0
    ctx_r:    int = 0
    ctx_d:    int = 0
    self_in:  int = 0
    self_out: int = 0        # 0 = off / dead (native — no shift needed)

    def __post_init__(self):
        """Reject packed tile words at the genotype boundary.

        A phenotype tile contains three selectors, but each gene allele is one
        physical 4-bit core-circuit configuration.  Failing here makes that
        distinction structural instead of relying on whichever editor happens
        to display the genome.
        """
        for name in ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'):
            value = int(getattr(self, name))
            if not 0 <= value < MAX_STATE:
                raise ValueError('%s must be a 4-bit circuit state (0..15)' % name)
            setattr(self, name, value)


@dataclass
class Chromosome:
    """`telomere` is a HAYFLICK LIMIT — the number of times a lineage may still
    divide, not a global clock. The germline / seed cells start with telomere
    length L (see `germline_telomere`); every time a cell divides (births a live
    frontier cell) the daughter inherits its parent's telomere MINUS ONE. When a
    cell's telomere reaches 0 it is senescent: it stays alive and keeps its
    function (maintenance rules always apply), but it can no longer divide, so no
    new cell is born beyond it. Growth therefore halts on its own at radius L
    from the seeds — the telomere alone bounds the organism's SIZE (replacing the
    old grid_size clip) and its growth DURATION (≤ L rings, replacing the old
    fixed iteration cap). Telomere length is evolvable, so the genome itself
    decides how large it grows on the unbounded field."""
    genes: List[HexGene] = field(default_factory=list)
    split: int = 0
    tag:   int = 0
    telomere: int = MAX_TELOMERE


def germline_telomere(genome) -> int:
    """The organism's germline telomere length L: the longest division program
    across its chromosomes. Seed cells start here; growth reaches radius L. A
    genome with no chromosomes cannot divide (L falls back to 1 — seeds only)."""
    return max((getattr(c, 'telomere', 1) for c in genome.chromosomes), default=1)


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0


def random_hex_gene() -> HexGene:
    # self_in == 0 makes a GROWTH rule: it matches empty cells and is the only
    # kind that can bring an empty cell to life under the sim6 empty-cell guard
    # (division is further gated by the parent's Hayflick telomere). Random
    # genomes need a healthy share of them or nothing ever grows — sim6 gets this
    # for free because table_create manufactures a gene for every empty-cell
    # context it meets.
    # Retain the historical 1/32 off-output prior explicitly.  The important
    # distinction is that every sampled value is now a single 4-bit circuit
    # configuration; no gene contains a packed 12-bit tile word.
    self_out = (0 if random.randrange(32) == 0
                else random.randrange(1, MAX_STATE))
    return HexGene(
        ctx_l    = random.randrange(MAX_STATE),
        ctx_r    = random.randrange(MAX_STATE),
        ctx_d    = random.randrange(MAX_STATE),
        self_in  = 0 if random.random() < 0.25 else random.randrange(MAX_STATE),
        self_out = self_out,                         # 0 = this circuit is off
    )


def random_hex_chromosome(n_genes=None, max_telomere=MAX_TELOMERE) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_hex_gene() for _ in range(n_genes)],
        split = (0 if n_genes < 2 else random.randint(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, min(5, max_telomere)),
    )


def random_hex_genome(n_chroms=2, max_telomere=MAX_TELOMERE) -> Genome:
    return Genome(
        chromosomes = [random_hex_chromosome(max_telomere=max_telomere)
                       for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
