"""
substrates/lut/genome.py — genome for the boolean-logic lookup (LUT) array.

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

from runtime.limits import MAX_CHROMOSOME_COUNT

LUT_BITS     = 16
LUT_STATES   = 1 << LUT_BITS     # 65536 possible cell states
MAX_GENES    = 24
MAX_CHROMS   = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20                # longest evolvable growth phase (iterations)


@dataclass
class LutGene:
    ctx_n:    int = 0
    ctx_e:    int = 0
    ctx_s:    int = 0
    ctx_w:    int = 0
    self_in:  int = 0
    self_out: int = 0            # 0 = off / dead
    # Heritable developmental node identity. 0 = ordinary body cell,
    # 1 = source-only input terminal, 2 = sink-only output terminal.
    io_kind:  int = 0
    # Body-gene expression priority under tag_rank; desired direction-LUT type
    # or normalised x coordinate on a chromosome-3 port gene.
    tag:      int = 0
    # Deprecated checkpoint field, pinned to 1: every port owns one cell.
    io_limit: int = 1
    # Type-instance selector, or normalised y for spatial wiring.
    io_selector: int = 0


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
    # Non-developmental port-chromosome marker (see substrates/nervous/genome.py). Its genes
    # hold either type/selector or x/y I/O mappings. Default False leaves every
    # existing genome unchanged.
    wiring:   bool = False


@dataclass
class Genome:
    chromosomes: List[Chromosome] = field(default_factory=list)
    tag: int = 0
    # Optional heritable polarisation of the developmental seed cell.  ``None``
    # preserves the historical isotropic SEED_STATE exactly.  A four-LUT state
    # lets designed/evolved organisms break the otherwise unavoidable fourfold
    # symmetry of growth from one neutral centre cell.
    seed_state: tuple[int, int, int, int] | None = None
    # Non-behavioural audit label. Compiler rescues retain their origin through
    # mutation/checkpointing so they cannot be mistaken for unaided discoveries.
    provenance: str = ''


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


def random_lut_chromosome(n_genes=None, wiring=False) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_lut_gene() for _ in range(n_genes)],
        split = (0 if n_genes < 2 else random.randint(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, min(5, MAX_TELOMERE)),
        wiring   = wiring,
    )


def random_lut_genome(n_chroms=2, wiring_chromosome=False, n_ports=None,
                      tag_rank=False, spatial_chromosome=False,
                      terminal_nodes=False, n_inputs=0,
                      n_outputs=0) -> Genome:
    """Build a fixed genome or seed one of the evolvable I/O strategies."""
    chroms = [random_lut_chromosome() for _ in range(n_chroms)]
    genome = Genome(
        chromosomes = chroms,
        tag = random.randint(0, 9999),
    )
    if wiring_chromosome or spatial_chromosome or tag_rank:
        from substrates.nervous.io_placement import seed_io_metadata
        seed_io_metadata(genome, wiring_chromosome=wiring_chromosome,
                         n_ports=n_ports, tag_rank=tag_rank,
                         spatial_chromosome=spatial_chromosome)
    if terminal_nodes:
        from substrates.nervous.io_placement import seed_terminal_kinds
        seed_terminal_kinds(genome, n_inputs, n_outputs)
    return genome
