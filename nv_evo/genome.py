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
are timeless. Growth is bounded biologically instead of by the paper's "artificial
methods" (§7): each chromosome carries a telomere (a Hayflick division limit) so
the organism halts its own growth (see Chromosome + nv_evo/nervous.py).

The genetic operators (mutation, crossover, selection) live in nv_evo/ga.py.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List

from evo_runtime.limits import MAX_CHROMOSOME_COUNT

MAX_STATE    = 32       # 5-bit cell state: 0-15 = paper AND routing, 16-31 = OR
MAX_GENES    = 24
MAX_CHROMS   = MAX_CHROMOSOME_COUNT
MAX_TELOMERE = 20        # longest evolvable growth phase (iterations)


@dataclass
class HexGene:
    ctx_l:    int = 0
    ctx_r:    int = 0
    ctx_d:    int = 0
    self_in:  int = 0
    self_out: int = 0        # 0 = off / dead (native — no shift needed)


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
    # 'evolved_width' node-timing model ONLY (see nv_evo/pulse.py NODE_MODELS):
    # a per-node-type pulse-width MULTIPLIER indexed by the cell's 5-bit routing
    # state (0..31). Each entry scales that node type's emitted pulse width
    # relative to PulseConfig.width; 1.0 = the paper's uniform width. ``None``
    # (the default, and every non-'evolved_width' genome) means "all 1.0",
    # so a fresh 'evolved_width' genome behaves exactly like the paper until
    # mutation differentiates a type's width. Not decoded during growth — width
    # is a physical property of the node, orthogonal to its routing.
    state_widths: List[float] = None
    # 'pulse_delay' / width-preserving model ONLY: per-routing-state delay
    # multipliers. Both transported edges use the same selected delay, so width
    # is preserved while propagation speed becomes heritable. ``None`` is the
    # neutral all-1.0 vector and therefore retains the configured fixed delay.
    state_delays: List[float] = None


# Evolvable pulse-width multiplier bounds (of PulseConfig.width). The floor is
# 0.5 because trace/persistence scoring samples each wire ONCE at mid-tick
# (t + 0.5*TICK, half-open intervals): a narrower pulse rising on the tick
# lattice would be genuinely firing yet invisible to the sampler, letting the
# same net score differently across score modes for non-behavioural reasons.
WIDTH_MULT_MIN = 0.5
WIDTH_MULT_MAX = 4.0
DELAY_MULT_MIN = 0.25    # evolvable propagation-delay bounds (of PulseConfig.delay)
DELAY_MULT_MAX = 4.0


def default_state_widths():
    """A neutral width-multiplier vector (all 1.0): identical to 'uniform'."""
    return [1.0] * MAX_STATE


def default_state_delays():
    """A neutral delay-multiplier vector (all 1.0): fixed-delay transport."""
    return [1.0] * MAX_STATE


def random_hex_gene() -> HexGene:
    # self_in == 0 makes a GROWTH rule: it matches empty cells and is the only
    # kind that can bring an empty cell to life under the sim6 empty-cell guard
    # (division is further gated by the parent's Hayflick telomere). Random
    # genomes need a healthy share of them or nothing ever grows — sim6 gets this
    # for free because table_create manufactures a gene for every empty-cell
    # context it meets.
    return HexGene(
        ctx_l    = random.randrange(MAX_STATE),
        ctx_r    = random.randrange(MAX_STATE),
        ctx_d    = random.randrange(MAX_STATE),
        self_in  = 0 if random.random() < 0.25 else random.randrange(MAX_STATE),
        self_out = random.randrange(MAX_STATE),      # 0..31, 0 = death
    )


def random_hex_chromosome(n_genes=None, max_telomere=MAX_TELOMERE) -> Chromosome:
    if n_genes is None:
        n_genes = random.randint(3, MAX_GENES // 2)
    return Chromosome(
        genes = [random_hex_gene() for _ in range(n_genes)],
        split = random.randint(1, max(1, n_genes - 1)),
        tag   = random.randint(0, 999),
        telomere = random.randint(2, min(5, max_telomere)),
    )


def random_hex_genome(n_chroms=2, max_telomere=MAX_TELOMERE) -> Genome:
    return Genome(
        chromosomes = [random_hex_chromosome(max_telomere=max_telomere)
                       for _ in range(n_chroms)],
        tag = random.randint(0, 9999),
    )
