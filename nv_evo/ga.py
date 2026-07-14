"""
nv_evo/ga.py — genetic algorithm native to the nervous net, tuned for
evolving loops and memory.

Temporal fitness landscapes are deceptive: an SR latch scores nothing until a
feedback loop *and* its input/output wiring all appear together, so a plain GA
stalls on flat plateaus and converges on loop-free delay chains. This GA
differs from snn_evo's in seven ways:

  1. Trace-matched outputs + spike-event scoring (temporal targets):
     each output role is read at the live cell whose trace best matches the
     expected trace across ALL trials (terminal distance only breaks ties), so
     evolution only has to build the mechanism, not also route its answer to
     one prescribed cell. The trace is scored by SPIKE-EVENT precision/recall
     (F1, see nv_evo/temporal.METRIC): reward each correctly-timed output
     spike, penalise expected spikes that never occur, penalise extra spikes
     that weren't expected. A do-nothing output recalls nothing and so scores
     0 — silence is not rewarded (measured: under plain per-tick accuracy the
     GA converged onto the constant-0 baseline on every preset). 1.0 iff every
     expected spike is produced and nothing fires where it shouldn't.

  2. Loop-aware shaping: a small bonus, scaled by (1 - score) so a perfect
     score is still exactly 1.0, rewards nets whose signal graph contains
     directed cycles — especially "relevant" cycles that inputs can write and
     outputs can read (loop_profile). Among equally scoring nets, the ones
     structurally *capable* of memory win the tie.

  3. Gene duplication as a mutation operator: loops are built from repeated
     local routing motifs (two cells buffering each other), and duplicating a
     working gene then tweaking one field reaches those far more often than
     fresh random genes.

  4. Random immigrants: a few fresh genomes replace the worst each generation,
     keeping exploration alive on plateaus instead of inbreeding to a stall.

  5. Fitness caching: temporal evaluation (trials x T ticks) is expensive and
     converged populations re-submit the same genomes; a signature cache skips
     re-evaluating elites and duplicates.

  6. Senescence / metabolic cost (rank_key): among equally-fit genomes the one
     that grows a cheaper, smaller body wins — fewer genes, then a shorter
     telomere (a smaller organism, since telomeres are now a Hayflick growth
     bound). This never distorts the fitness value, so a solved run still reads
     exactly 1.0 while both genome and body shrink when the task allows.

  7. Stress-induced mutagenesis (adaptive_mutation_rate): the bacterial SOS response —
     hold the mutation rate at baseline until the run stalls, then ramp it up the
     longer it stays stuck and relax the instant progress resumes. Aimed squarely
     at the deceptive temporal plateaus where only a burst of variation reaches
     the next rung.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from evo_runtime.cache import LRUCache
from evo_runtime.parallel import map_ordered

from .genome import (MAX_STATE, MAX_GENES, MAX_CHROMS, MAX_TELOMERE,
                     Genome, Chromosome, germline_telomere,
                     random_hex_gene, random_hex_chromosome, random_hex_genome)
from .nervous import score_nervous


def clone_genome(genome):
    """Fast structural copy: new Genome/Chromosome objects with fresh gene LISTS
    but SHARED gene objects. Safe because genes are never mutated in place —
    mutation always builds a new gene via _tweak_gene — so sharing the (immutable)
    gene objects is equivalent to deep-copying them. copy.deepcopy dominated
    reproduction (~90% of next_population's time); this replaces it on the hot
    path with an identical-behaviour, ~10x cheaper copy."""
    return Genome(
        chromosomes=[Chromosome(genes=c.genes[:], split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE))
                     for c in genome.chromosomes],
        tag=genome.tag)
from .temporal import prepare_net, score_temporal_bundle, loop_profile

POPSIZE        = 120
ELITE_FRAC     = 0.10        # elites = this fraction of pop, UNLESS ELITE_COUNT set
ELITE_COUNT    = None        # exact elite count (GUI override); None = use ELITE_FRAC
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
# Preserve specialist lineages: small elite pools otherwise produce nearly
# exclusive champion × champion reproduction.
EXPLORATION_PARENT_FRAC = 0.30
MEAN_MUTATIONS = 4.0         # HOT-START mutation rate for simulated annealing:
                            # broad early exploration, cooled each generation by
                            # MUT_DECAY toward ~0 as the genes self-organise.
                            # (Was 1.2; annealing wants a high start — see below.)
MUT_DECAY      = 0.997       # slow cooldown: hard recurrent tasks need late
                             # variation — 0.997 cools 4.0 -> ~0.89 by gen 500,
                             # where the old 0.99 crashed it to ~0.03. α is
                             # PER-GENERATION, so tie it to run length — for very
                             # long runs use α close to 1 (e.g. 0.9999); α = 1.0
                             # disables annealing.
LOOP_WEIGHT    = 0.05          # max shaping bonus, as a fraction of (1 - score)
# A counter-like mechanism needs extra structure.  Do not prune that structure
# merely because it ties a smaller shortcut before the behavioral task is solved.
PARSIMONY_START_FITNESS = 0.999
# Population evaluation is embarrassingly parallel; the old min(cpu, 8) left
# most cores idle on many-core machines (a 20-core box used 8). Scale with the
# machine, leave 2 cores for the GUI/OS, and cap at 16 where returns flatten
# (measured: 16->20 workers gained <1.1x while adding IPC overhead).
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))

# Very long runs (10k–100k generations) accumulate one fitness-cache entry per
# distinct genome ever seen. That is the only structure that grows without bound
# over a run, so cap it: when it exceeds this, drop it and let it refill (elites
# re-cache within a generation). Everything else the loop keeps is O(1) per gen.
FITNESS_CACHE_MAX = 200_000


# ── evaluation ─────────────────────────────────────────────────────────────────

def _loop_bonus(grid, routing, in_pos, out_pos):
    """[0,1] structural credit for memory capability. Any cycle earns a little;
    cycles that inputs can write and outputs can read (a latch's skeleton)
    earn the rest, saturating at 4 relevant cycle nodes."""
    prof = loop_profile(grid, routing, in_pos, out_pos)
    if not prof['n_cycle']:
        return 0.0
    return 0.3 + 0.7 * min(1.0, prof['n_relevant'] / 4.0)


def evaluate_nv_full(genome, target):
    """(scalar fitness, per-case score vector). Cases are the individual
    (trial, role) traces — the units ε-lexicase selection streams over. For
    combinational targets there is no trial structure: cases is None and
    selection falls back to tournament."""
    if not getattr(target, 'temporal', False):
        return score_nervous(genome, target), None
    retention_mode = getattr(target, 'score_mode', '')
    if retention_mode == 'retention':
        from .persistence import evaluate_retention    # event-native memory fitness
        return evaluate_retention(genome, target)
    if retention_mode == 'sr_retention':
        from .persistence import evaluate_sr_retention
        return evaluate_sr_retention(genome, target)
    n_cases = sum(len(tr.expected) for tr in target.trials)
    prep = prepare_net(genome, target)
    if prep is None:
        return 0.0, (0.0,) * n_cases
    grid, routing, in_pos, out_pos, traces = prep
    if getattr(traces, 'overflow', False):
        return 0.0, (0.0,) * n_cases
    s, cases, _ = score_temporal_bundle(traces, target)
    if s < 1.0:
        s = s + (1.0 - s) * LOOP_WEIGHT * _loop_bonus(grid, routing, in_pos, out_pos)
    return s, cases


def evaluate_nv(genome, target):
    """Scalar fitness (back-compat wrapper around evaluate_nv_full)."""
    return evaluate_nv_full(genome, target)[0]


def genome_signature(genome):
    """Hashable identity of a genome's evolvable content (for the fitness cache)."""
    return tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0),
         tuple((g.ctx_l, g.ctx_r, g.ctx_d, g.self_in, g.self_out)
               for g in c.genes))
        for c in genome.chromosomes)


def eval_batch_cases(genomes, target, cache=None, executor=None,
                     should_stop=None, on_progress=None):
    """Evaluate a population in parallel -> (fitnesses, case_vectors). `cache`
    ({signature: (fit, cases)}, owned by the caller) skips seen genomes. If a
    persistent `executor` (a ProcessPoolExecutor) is passed, it is reused instead
    of spawning a fresh worker pool every call — on Windows the per-generation
    spawn+re-import dominated runtime, so reuse is a large speed-up. Omitting it
    keeps the original one-shot-pool behaviour. `should_stop`/`on_progress` are
    threaded to map_ordered so a run stays cancellable without a chunk barrier."""
    out  = [None] * len(genomes)
    todo = list(range(len(genomes)))
    if cache is not None and len(cache) > FITNESS_CACHE_MAX:
        cache.clear()                      # bound memory on very long runs
    if cache is not None:
        sigs = [genome_signature(g) for g in genomes]
        todo = [i for i in todo if sigs[i] not in cache]
        for i in range(len(genomes)):
            if sigs[i] in cache:
                out[i] = cache[sigs[i]]
    if todo:
        fn = partial(evaluate_nv_full, target=target)
        # A generation can contain the same genome more than once (especially
        # around a strong parent).  The old code submitted every occurrence
        # before any result reached the cache, wasting whole worker slots on
        # identical evaluations.  Submit each signature once and fan its result
        # back out to every matching population index.
        if cache is not None:
            unique = {}
            for i in todo:
                unique.setdefault(sigs[i], []).append(i)
            representatives = [(sig, indices[0]) for sig, indices in unique.items()]
        else:
            representatives = [(i, i) for i in todo]
            unique = {i: [i] for i in todo}
        subset = [genomes[i] for _, i in representatives]
        if executor is not None:
            results = map_ordered(executor, fn, subset, should_stop, on_progress)
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                results = map_ordered(ex, fn, subset, should_stop, on_progress)
        for (sig, _), r in zip(representatives, results):
            for i in unique[sig]:
                out[i] = r
            if cache is not None:
                cache[sig] = r
    return [r[0] for r in out], [r[1] for r in out]


def eval_batch_nv(genomes, target, cache=None):
    """Back-compat scalar batch (shares the (fit, cases) cache format)."""
    return eval_batch_cases(genomes, target, cache)[0]


# ── genetic operators ────────────────────────────────────────────────────────────

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"]
_STATE_BITS = (MAX_STATE - 1).bit_length()


def _normalize_split(chromosome):
    """Keep a chromosome split on a real between-gene boundary.

    A one-gene chromosome has no such boundary; crossover handles that case
    inside the gene instead.  Structural mutations used to leave terminal or
    out-of-range split values behind, silently turning later crossover into a
    clone operation.
    """
    count = len(chromosome.genes)
    chromosome.split = (0 if count < 2 else
                        max(1, min(int(chromosome.split), count - 1)))


def _recombine_gene_fields(gene_a, gene_b):
    """Recombine a single rule when no between-gene cut exists.

    The fields are the gene's actual alleles.  If at least two differ, exchange
    a non-empty proper subset so both children inherit from both parents and
    neither is a parental clone.  With fewer than two differing alleles a novel
    recombinant is mathematically impossible; the mandatory mutation that
    follows crossover still supplies new variation.
    """
    differing = [field for field in _GENE_FIELDS
                  if getattr(gene_a, field) != getattr(gene_b, field)]
    if len(differing) < 2:
        return gene_a, gene_b
    exchanged = set(random.sample(
        differing, random.randint(1, len(differing) - 1)))
    child_a, child_b = copy.copy(gene_a), copy.copy(gene_b)
    for field in exchanged:
        value_a, value_b = getattr(gene_a, field), getattr(gene_b, field)
        setattr(child_a, field, value_b)
        setattr(child_b, field, value_a)
    return child_a, child_b


def _recombination_signature(genome):
    """Alleles crossover can actually exchange (not slot/object identity)."""
    return tuple(
        tuple(tuple(getattr(gene, field) for field in _GENE_FIELDS)
              for gene in chromosome.genes)
        for chromosome in genome.chromosomes)


def _other_state(value):
    """Flip one physical SRAM bit in a 4-bit core-circuit configuration."""
    return int(value) ^ (1 << random.randrange(_STATE_BITS))


def _state_excluding(*values):
    """Choose a one-bit neighbour excluding the supplied values when possible."""
    excluded = sorted({int(value) for value in values
                       if value is not None and 0 <= int(value) < MAX_STATE})
    base = int(values[0]) if values and values[0] is not None else 0
    nearby = [base ^ (1 << bit) for bit in range(_STATE_BITS)
              if (base ^ (1 << bit)) not in excluded]
    if nearby:
        return random.choice(nearby)
    # Defensive fallback when exclusions cover every one-bit neighbour.
    pick = random.randrange(MAX_STATE - len(excluded))
    for value in excluded:
        if pick >= value:
            pick += 1
    return pick


def _force_nonparent_tweak(genome, parent):
    """Finish a multi-event mutation transaction with protected novelty.

    Earlier edits may cancel one another.  This last allele is chosen different
    from both its current value and the value at the same parental locus, so the
    transaction cannot net back to a clone.  It is constructive; no offspring is
    generated and rejected/retried by signature.
    """
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(random_hex_chromosome())
        else:
            random.choice(genome.chromosomes).genes.append(random_hex_gene())
        with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                      if chromosome.genes]
    ci = random.choice(with_genes)
    gi = random.randrange(len(genome.chromosomes[ci].genes))
    field = random.choice(_GENE_FIELDS)
    gene = copy.copy(genome.chromosomes[ci].genes[gi])
    parent_value = None
    if (ci < len(parent.chromosomes)
            and gi < len(parent.chromosomes[ci].genes)):
        parent_value = getattr(parent.chromosomes[ci].genes[gi], field)
    setattr(gene, field, _state_excluding(getattr(gene, field), parent_value))
    genome.chromosomes[ci].genes[gi] = gene


def _tweak_gene(gene):
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and g.self_in != 0 and random.random() < 0.2:
        g.self_in = 0                     # keep growth rules reachable
    else:
        setattr(g, fld, _other_state(getattr(g, fld)))
    return g


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere"]
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08]


def _mutate_once_nv(genome, max_telomere=MAX_TELOMERE,
                    chromosome_count=None):
    """Apply one feasible, state-changing mutation to a nervous-net genome."""
    if not genome.chromosomes:
        genome.chromosomes.append(random_hex_chromosome(
            max_telomere=max_telomere))
        return

    chroms = genome.chromosomes
    with_genes = [c for c in chroms if c.genes]
    options = []
    if with_genes:
        options.append('tweak')
    if any(c.genes and len(c.genes) < MAX_GENES for c in chroms):
        options.append('duplicate')
    if any(len(c.genes) < MAX_GENES for c in chroms):
        options.append('add_gene')
    if any(len(c.genes) > 1 for c in chroms):
        options.append('del_gene')
    # A configured chromosome count is a structural constraint. Direct callers
    # may leave it as None to retain the older, evolvable-count experiment.
    if chromosome_count is None and len(chroms) < MAX_CHROMS:
        options.append('add_chrom')
    if chromosome_count is None and len(chroms) > 1:
        options.append('del_chrom')
    if any(len(c.genes) > 2 for c in chroms):
        options.append('split')
    telomeres = []
    for c in chroms:
        base = getattr(c, 'telomere', 10)
        values = [base + d for d in (-3, -2, -1, 1, 2, 3)
                  if 1 <= base + d <= max_telomere]
        if values:
            telomeres.append((c, values))
    if telomeres:
        options.append('telomere')

    weights = [_MUT_WEIGHTS[_MUT_OPS.index(op)] for op in options]
    op = random.choices(options, weights=weights)[0]
    if op == 'tweak':
        chrom = random.choice(with_genes)
        idx = random.randrange(len(chrom.genes))
        chrom.genes[idx] = _tweak_gene(chrom.genes[idx])
    elif op == 'duplicate':
        chrom = random.choice([c for c in chroms
                               if c.genes and len(c.genes) < MAX_GENES])
        chrom.genes.insert(random.randrange(len(chrom.genes) + 1),
                           _tweak_gene(random.choice(chrom.genes)))
    elif op == 'add_gene':
        random.choice([c for c in chroms if len(c.genes) < MAX_GENES]).genes.append(
            random_hex_gene())
    elif op == 'del_gene':
        chrom = random.choice([c for c in chroms if len(c.genes) > 1])
        chrom.genes.pop(random.randrange(len(chrom.genes)))
    elif op == 'add_chrom':
        chroms.append(random_hex_chromosome(max_telomere=max_telomere))
    elif op == 'del_chrom':
        chroms.remove(min(chroms, key=lambda c: len(c.genes)))
    elif op == 'split':
        chrom = random.choice([c for c in chroms if len(c.genes) > 2])
        values = [s for s in range(1, len(chrom.genes)) if s != chrom.split]
        chrom.split = random.choice(values)
    else:  # telomere
        chrom, values = random.choice(telomeres)
        chrom.telomere = random.choice(values)


def mutate_nv(genome, mean_mutations=None, max_telomere=MAX_TELOMERE,
              chromosome_count=None):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    g = clone_genome(genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    events = max(1, _poisson(lam))
    for _ in range(events - 1):
        _mutate_once_nv(g, max_telomere=max_telomere,
                        chromosome_count=chromosome_count)
    if events == 1:
        _mutate_once_nv(g, max_telomere=max_telomere,
                        chromosome_count=chromosome_count)
    else:
        _force_nonparent_tweak(g, genome)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    return g


# back-compat name (the old mutation operator lived in nv_evo/genome.py)
mutate_hex = mutate_nv


def crossover_nv(pa, pb):
    """Tag-matched hierarchical crossover.

    Multi-gene homologs cross at a genuine interior gene boundary.  When the
    common homolog has only one gene, recombine that rule's fields instead, so a
    minimal chromosome still has useful sexual recombination.
    """
    ca, cb = clone_genome(pa), clone_genome(pb)
    used_b = set()
    for i, chrom_a in enumerate(ca.chromosomes):
        best_j, best_dist = None, float("inf")
        for j, chrom_b in enumerate(cb.chromosomes):
            if j in used_b:
                continue
            d = abs(chrom_a.tag - chrom_b.tag)
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is None:
            continue
        used_b.add(best_j)
        chrom_b = cb.chromosomes[best_j]
        # Snapshot BOTH parents before assigning either child.  Previously
        # chrom_a aliased ca.chromosomes[i]; assigning child A first overwrote the
        # suffix then used to build child B, making child B a parent-B clone.
        genes_a, genes_b = chrom_a.genes[:], chrom_b.genes[:]
        common = min(len(genes_a), len(genes_b))
        if common >= 2:
            sp = max(1, min(int(chrom_a.split), common - 1))
            ca.chromosomes[i].genes = genes_a[:sp] + genes_b[sp:]
            cb.chromosomes[best_j].genes = genes_b[:sp] + genes_a[sp:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = sp
        elif common == 1:
            gene_a, gene_b = _recombine_gene_fields(genes_a[0], genes_b[0])
            # Preserve variable-length suffix exchange while hybridising the
            # one shared rule.  For 1x1 homologs these are simply the hybrids.
            ca.chromosomes[i].genes = [gene_a] + genes_b[1:]
            cb.chromosomes[best_j].genes = [gene_b] + genes_a[1:]
            ca.chromosomes[i].split = cb.chromosomes[best_j].split = 0
        else:
            # A genuinely empty homolog has no allele-level cut. Preserve the
            # old whole-list transfer, but make it reciprocal rather than the
            # alias-bug behavior that copied parent B into both children.
            ca.chromosomes[i].genes = genes_b
            cb.chromosomes[best_j].genes = genes_a
        _normalize_split(ca.chromosomes[i])
        _normalize_split(cb.chromosomes[best_j])
    for chromosome in ca.chromosomes + cb.chromosomes:
        _normalize_split(chromosome)
    return ca, cb


def n_genes(genome):
    return sum(len(c.genes) for c in genome.chromosomes)


def total_telomere(genome):
    """Sum of the chromosome telomeres — a proxy for how large a body the genome
    is licensed to grow (radius scales with the germline telomere)."""
    return sum(getattr(c, 'telomere', 0) for c in genome.chromosomes)


def rank_key(genome, fitness):
    """Selection / ranking key (maximise). Fitness dominates absolutely; ties
    break toward a cheaper organism — SENESCENCE / METABOLIC COST: fewer genes
    first (the paper's parsimony pressure), then a shorter GERMLINE telomere (the
    organism's growth RADIUS = its real body-size cost). Using the germline (max)
    telomere, not the SUM over chromosomes, is deliberate: the sum penalises
    chromosome COUNT — it collapsed multi-chromosome genomes to 1 even when each
    chromosome was small — whereas the germline reflects actual organism size and
    leaves chromosome count to gene-parsimony alone. Neither tie-break distorts
    the fitness value, so a solved run still reads exactly 1.0."""
    if fitness < PARSIMONY_START_FITNESS:
        return (fitness, 0, 0)
    return (fitness, -n_genes(genome), -germline_telomere(genome))


def tournament_nv(population, fitnesses):
    """Tournament selection with a metabolic-cost tie-break: at EQUAL fitness the
    cheaper organism wins (fewer genes, then a shorter telomere — see rank_key),
    applied without distorting the fitness value itself, so a solved run still
    reads exactly 1.0 while complexity keeps shrinking when it isn't needed."""
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: rank_key(population[i], fitnesses[i]))]


def _lexicase_parent(population, case_vecs):
    """ε-lexicase selection (La Cava et al.): stream the cases in random order;
    at each case keep only candidates within ε (median absolute deviation) of
    that case's best. Averages hide a single failing trial (~1/12 of a mean);
    lexicase makes every case a hard filter some of the time, so specialists on
    the currently-failing cases are selected and recombined — the mechanism that
    drives populations to ALL-cases-perfect rather than high-average."""
    n_cases = len(case_vecs[0])
    cand = list(range(len(population)))
    for c in random.sample(range(n_cases), n_cases):
        vals = [case_vecs[i][c] for i in cand]
        best = max(vals)
        srt  = sorted(vals)
        med  = srt[len(srt) // 2]
        eps  = sorted(abs(v - med) for v in vals)[len(vals) // 2]
        cand = [i for i, v in zip(cand, vals) if v >= best - eps]
        if len(cand) == 1:
            break
    return population[random.choice(cand)]


# Parent-selection scheme. Measured head-to-head at equal budget (see memory):
# lexicase helped only the pair detector (+0.11), regressed coincidence (-0.13)
# and pattern (-0.13), and left every plateau (latch/toggle/stepper) at the
# IDENTICAL value tournament reaches — the plateaus are representational, not a
# selection-pressure problem. Tournament is the better default; lexicase stays
# available for experiments.
SELECTION = 'tournament'          # 'tournament' | 'lexicase'


def select_parent(population, fitnesses, case_vecs=None):
    if (SELECTION == 'lexicase' and case_vecs is not None
            and case_vecs[0] is not None):
        return _lexicase_parent(population, case_vecs)
    return tournament_nv(population, fitnesses)


# Stress-induced mutagenesis (the bacterial SOS response): baseline mutation
# until the population has stalled for STRESS_PATIENCE generations, then the rate
# ramps toward STRESS_MAX_MULT the longer it stays stuck — raising variation to
# climb off a plateau, then relaxing the instant progress resumes. The temporal
# plateaus (latch/toggle/stepper) are exactly the deceptive landscapes this is
# meant for: flat regions where only a burst of variation reaches the next rung.
STRESS_PATIENCE = 12
STRESS_MAX_MULT = 8.0


def adaptive_mutation_rate(annealed_rate, stagnation, solved=False):
    """Actual mutation rate with a plateau reheat independent of annealing.

    Once the target is SOLVED (best fitness == 1.0), stagnation is permanent —
    nothing can beat 1.0 — so the SOS reheat would fire forever and pointlessly
    scatter the population (there is no plateau left to escape). When `solved`,
    skip the reheat and stay at the annealed base so the population can settle."""
    base = max(1.0, annealed_rate)
    if solved or stagnation < STRESS_PATIENCE:
        return base
    ramp = 1.0 + ((stagnation - STRESS_PATIENCE)
                  / max(1.0, STRESS_PATIENCE / 2.0))
    return max(base, min(STRESS_MAX_MULT, ramp))


def consolidate_population(parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
    """Terminal ``(mu + lambda)`` survivor selection.

    The exploratory GA still creates a complete, genuinely mutated offspring
    generation. Once a terminal solution exists, choose the next live population
    from parents + offspring so successful circuits accumulate instead of being
    discarded before the charted mean can converge. Exact-rank ties are shuffled
    first, preserving neutral turnover rather than preferring the parent half by
    stable-sort accident. Case vectors follow their genome/fitness entry.
    """
    pop = len(parents)
    if len(parent_fitnesses) != pop or len(offspring) != pop:
        raise ValueError('parent/offspring population sizes must match')
    if len(offspring_fitnesses) != pop:
        raise ValueError('offspring fitness count must match population size')
    case_vectors = parent_cases is not None or offspring_cases is not None
    if case_vectors and (parent_cases is None or offspring_cases is None):
        raise ValueError('parent and offspring case vectors must both be present')
    if case_vectors and (len(parent_cases) != pop or len(offspring_cases) != pop):
        raise ValueError('case-vector count must match population size')

    genomes = list(parents) + list(offspring)
    fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
    cases = ((list(parent_cases) + list(offspring_cases))
             if case_vectors else None)
    order = list(range(len(genomes)))
    random.shuffle(order)
    order.sort(key=lambda i: rank_key(genomes[i], fitnesses[i]), reverse=True)
    keep = order[:pop]
    return ([genomes[i] for i in keep],
            [fitnesses[i] for i in keep],
            ([cases[i] for i in keep] if cases is not None else None))


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None, selection=None, ga_config=None,
                    chromosome_count=None):
    """One generation of a steady, exploratory GA — elitism preserves the best,
    the rest are tournament/ε-lexicase parents recombined and mutated, plus a
    few random immigrants that keep exploration alive.

    While unsolved this remains full replacement: no parent is smuggled through
    reproduction as a clone. The caller applies ``consolidate_population`` only
    after a terminal solution exists, using evaluated survivor selection rather
    than weakening these exploration operators."""
    pop = len(population)
    elite_count = (ELITE_COUNT if ga_config is None else ga_config.elite_count)
    immigrant_fraction = (IMMIGRANT_FRAC if ga_config is None
                          else ga_config.immigrant_fraction)
    tournament_size = (TOURNAMENT_K if ga_config is None
                       else ga_config.tournament_size)
    if ga_config is not None and ga_config.chromosome_count is not None:
        chromosome_count = ga_config.chromosome_count
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    if make_genome is None:
        make_genome = lambda: random_hex_genome(chromosome_count or 2)
    n_elite = elite_count if elite_count is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = min(int(round(pop * immigrant_fraction)), pop)
    order   = sorted(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]),
                     reverse=True)
    # Elites are the RECOMBINATION PARENT POOL (truncation selection): the top
    # n_elite genomes breed the offspring but are NOT copied by this operator.
    # Before the task is solved, the population's best can therefore regress.
    # After solving, the caller performs evaluated parent+offspring survival.
    # Parents are drawn by
    # tournament WITHIN that pool (TOURNAMENT_K=1 → uniform among elites). n_elite==0
    # falls back to normal selection over the whole population.
    # Elites are breeding material here. Keeping terminal winners is a selection
    # decision made after their children have been evaluated, not a clone hidden
    # inside reproduction.
    new_pop = [make_genome() for _ in range(n_imm)]                  # immigrants
    # ε-lexicase (when selected) must stream over the WHOLE population; the elite-only
    # breeding pool would otherwise mask it whenever elites>0, so bypass the pool then.
    selection = ((SELECTION if ga_config is None else ga_config.selection)
                 if selection is None else selection)
    use_lexicase = (selection == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    if n_elite > 0 and not use_lexicase:
        elite = order[:n_elite]
        k = min(tournament_size, len(elite))
        elite_parent = lambda: population[max(
            random.sample(elite, k),
            key=lambda i: rank_key(population[i], fitnesses[i]))]
    else:
        elite_parent = lambda: (_lexicase_parent(population, case_vecs)
                                 if use_lexicase else population[max(
                                     random.sample(range(pop), min(tournament_size, pop)),
                                     key=lambda i: rank_key(population[i], fitnesses[i]))])
    residual = order[n_elite:] if n_elite < pop else order
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]

    def parent():
        # A low-scoring specialist can carry the missing later-period behavior;
        # let it occasionally recombine with an elite instead of disappearing.
        if residual and random.random() < EXPLORATION_PARENT_FRAC:
            return population[random.choice(residual)]
        return elite_parent()

    elite_pool = order[:n_elite] if n_elite else order

    def pick_index(candidates):
        if use_lexicase:
            local_pop = [population[i] for i in candidates]
            local_cases = [case_vecs[i] for i in candidates]
            chosen = _lexicase_parent(local_pop, local_cases)
            return candidates[next(i for i, genome in enumerate(local_pop)
                                   if genome is chosen)]
        k = min(tournament_size, len(candidates))
        return max(random.sample(candidates, k),
                   key=lambda i: rank_key(population[i], fitnesses[i]))

    def parent_pair():
        if pop == 1:
            return population[0], population[0]

        def choose(exclude=None):
            if use_lexicase:
                # Lexicase must see every specialist. Restricting it to the
                # scalar elite pool recreates tournament selection and loses
                # exactly the partial behaviors a curriculum is meant to join.
                candidates = [i for i in range(pop) if i != exclude]
                return pick_index(candidates)
            candidates = [i for i in elite_pool if i != exclude]
            exploratory = [i for i in residual if i != exclude]
            if exploratory and random.random() < EXPLORATION_PARENT_FRAC:
                candidates = exploratory
            if not candidates:
                candidates = [i for i in range(pop) if i != exclude]
            return pick_index(candidates)

        first = choose()
        second = choose(first)
        if recombination_signatures[second] == recombination_signatures[first]:
            # Different population slots can still be the same genotype.  If
            # any genuinely different breeding material exists, use it; this is
            # parent selection, not an after-the-fact child rejection loop.
            distinct = [
                i for i in range(pop)
                if i != first
                and recombination_signatures[i] != recombination_signatures[first]
            ]
            if distinct:
                second = pick_index(distinct)
        return population[first], population[second]

    while len(new_pop) < pop:
        ca, cb = crossover_nv(*parent_pair())
        new_pop.append(mutate_nv(
            ca, mean_mutations,
            max_telomere=(MAX_TELOMERE if ga_config is None
                          else ga_config.max_telomere),
            chromosome_count=chromosome_count))
        if len(new_pop) < pop:
            new_pop.append(mutate_nv(
                cb, mean_mutations,
                max_telomere=(MAX_TELOMERE if ga_config is None
                              else ga_config.max_telomere),
                chromosome_count=chromosome_count))
    if (chromosome_count is not None
            and any(len(genome.chromosomes) != chromosome_count
                    for genome in new_pop)):
        raise ValueError('genome factory violated configured chromosome count')
    return new_pop[:pop]


# ── main loop (headless; the GUI runs its own equivalent in app.py) ──────────────

def evolve_nervous(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True,
                   seed=None, seed_genomes=None, selection=None,
                   return_population=False):
    # Reproducibility: fitness evaluation is deterministic (grow + score, no RNG),
    # so seeding the main-process RNG that drives the genetic operators pins the
    # whole evolutionary trajectory. Pass a seed for any result meant to be re-run.
    if seed is not None:
        random.seed(seed)
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    make_genome = lambda: random_hex_genome(n_chroms)
    cache       = LRUCache(FITNESS_CACHE_MAX)
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:                                                       # (avoids per-gen respawn)
        # optional WARM START: begin from provided genomes (e.g. a circuit that
        # already has the easy behaviour) so evolution climbs toward the harder
        # objective instead of rediscovering the basics from scratch.
        seeds = list(seed_genomes or [])[:pop]
        if any(len(g.chromosomes) != n_chroms for g in seeds):
            raise ValueError('warm-start genomes must match n_chroms')
        population  = [clone_genome(g) for g in seeds]
        population += [make_genome() for _ in range(pop - len(population))]
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        bi           = max(range(pop),
                           key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        best_rank    = rank_key(best_genome, best_fitness)

        solved_at, stagnation = None, 0
        mut_rate = MEAN_MUTATIONS           # annealing schedule (see MUT_DECAY)
        if verbose:
            print("%5s  %6s  %6s  %5s" % ("Gen", "Best", "Mean", "Mut"))
            print("-" * 30)
        # NO early stop at best == 1.0: the run continues so terminal survivor
        # selection can consolidate robust, increasingly parsimonious solutions.
        for gen in range(generations):
            mut_rate *= MUT_DECAY                                  # anneal: cool down
            mm = adaptive_mutation_rate(mut_rate, stagnation,
                                        solved=best_fitness >= 1.0)
            parents, parent_fitnesses, parent_cases = population, fitnesses, cases
            offspring = next_population(
                parents, parent_fitnesses, make_genome, parent_cases, mm,
                selection=selection, chromosome_count=n_chroms)
            offspring_fitnesses, offspring_cases = eval_batch_cases(
                offspring, target, cache, ex)
            if max(best_fitness, max(offspring_fitnesses)) >= 1.0:
                population, fitnesses, cases = consolidate_population(
                    parents, parent_fitnesses, parent_cases,
                    offspring, offspring_fitnesses, offspring_cases)
            else:
                population, fitnesses, cases = (
                    offspring, offspring_fitnesses, offspring_cases)
            gi = max(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]))
            gen_rank = rank_key(population[gi], fitnesses[gi])
            # SOS tracks FITNESS plateaus only: a parsimony-only win (same fitness,
            # leaner body) still updates the champion but must NOT reset the stress
            # counter, or the ever-shrinking tie-break would keep it pinned at 1.0.
            if fitnesses[gi] > best_fitness + 1e-12:
                stagnation = 0
            else:
                stagnation += 1
            if gen_rank > best_rank:
                best_rank    = gen_rank
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
            if solved_at is None and best_fitness >= 1.0:
                solved_at = gen
                if verbose:
                    print("Solved at generation %d." % gen)
            if verbose and gen % 10 == 0:
                print("%5d  %6.4f  %6.4f  %5.3f" % (gen, best_fitness,
                                                    sum(fitnesses) / pop, mm))
        if return_population:
            return (best_genome, best_fitness, population, fitnesses, cases)
        return best_genome, best_fitness
    finally:
        ex.shutdown()


# ── diversification: a whole generation of DISTINCT valid solutions ──────────────

def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None, executor=None, should_stop=None, on_progress=None,
              max_telomere=MAX_TELOMERE, chromosome_count=None):
    """Fill a population with evaluated, genetically distinct valid offspring.

    Distinctness is based on the rule alleles crossover can exchange, not tags,
    split metadata, or a neutral telomere edit. Once two valid rule programs are
    available they breed through the normal crossover + mutation operators;
    mutation alone bootstraps the pool when only one solver exists.

    Returns the list of unique valid genomes found (up to pop_size). Where the
    target has a broad neutral network it fills the population; where solutions
    are isolated spikes it returns however few exist — an honest ceiling, not a
    monoculture faked with copies."""
    if cache is None:
        cache = LRUCache(FITNESS_CACHE_MAX)
    if batch is None:
        batch = max(48, pop_size)
    should_stop = should_stop or (lambda: False)
    seeds = list(seeds)
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count for genome in seeds):
            raise ValueError('seed violates configured chromosome count')
    _eval = lambda gs: eval_batch_cases(gs, target, cache, executor)[0]   # reuse pool
    pool, pool_signatures, seen = [], [], set()
    for g, f in zip(seeds, _eval(seeds)):
        s = _recombination_signature(g)
        if f >= valid and s not in seen:
            pool.append(g); pool_signatures.append(s); seen.add(s)
    if not pool:
        return pool
    for round_i in range(rounds):
        if len(pool) >= pop_size or should_stop():
            break
        if on_progress is not None:
            on_progress(round_i + 1, rounds, len(pool))
        cands, candidate_signatures = [], []
        for _ in range(batch):
            parent_index = random.randrange(len(pool))
            parent_a = pool[parent_index]
            signature_a = pool_signatures[parent_index]
            mates = [index for index, signature in enumerate(pool_signatures)
                     if signature != signature_a]
            if mates:
                base = random.choice(crossover_nv(
                    parent_a, pool[random.choice(mates)]))
            else:
                base = parent_a
            c = mutate_nv(base, max_telomere=max_telomere,
                          chromosome_count=chromosome_count)
            s = _recombination_signature(c)
            if s not in seen:
                seen.add(s)       # invalid programs stay seen: do not re-evaluate
                cands.append(c); candidate_signatures.append(s)
        if should_stop():
            break
        for c, s, f in zip(cands, candidate_signatures, _eval(cands)):
            if f >= valid and len(pool) < pop_size:
                pool.append(c); pool_signatures.append(s)
    return pool[:pop_size]
