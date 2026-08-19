"""
substrates/nervous/ga.py - genetic algorithm native to the nervous net, tuned for
evolving loops and memory.

Temporal fitness landscapes are deceptive: an SR latch scores nothing until a
feedback loop *and* its input/output wiring all appear together, so a plain GA
stalls on flat plateaus and converges on loop-free delay chains. This GA
differs from substrates/snn's in seven ways:

  1. Contract-matched outputs (temporal and periodic-combinational targets):
     every live non-input cell is a candidate read-only probe. Output roles are
     assigned globally to distinct cells using their complete Behavior Contract
     scores across all trials, so evolution need not route every answer to a
     prescribed coordinate and an early role cannot greedily consume the only
     strong probe for a later one. Point-event contracts use precision/recall;
     state, cadence, interval, and logic contracts use their own declared
     semantics through the shared ``score_contract`` evaluator.

  2. Contract-honest selection: reported fitness is exactly the declared
     behavior-contract score. Among equally scoring nets, a target-blind
     topology tie-break still prefers structures capable of carrying signal,
     integrating inputs, or closing feedback loops.

  3. Gene duplication as a mutation operator: loops are built from repeated
     local routing motifs (two cells buffering each other), and duplicating a
     working gene then tweaking one field reaches those far more often than
     fresh random genes.

  4. Random immigrants: a few fresh genomes replace the worst each generation,
     keeping exploration alive on plateaus instead of inbreeding to a stall.

  5. Fitness caching: temporal evaluation (trials x T ticks) is expensive and
     converged populations re-submit the same genomes; a signature cache skips
     re-evaluating elites and duplicates.

  6. Target-blind topology (rank_key): after viability, fitness, robustness,
     and juvenile score, ties prefer more source-reachable wiring,
     multi-input integration, and feedback. Disconnected bulk earns nothing,
     and gene count/telomere never enter this rank.

  7. Stress-induced mutagenesis (adaptive_mutation_rate): the bacterial SOS response -
     hold the mutation rate at baseline until the run stalls, then ramp it up the
     longer it stays stuck and relax the instant progress resumes. Aimed squarely
     at the deceptive temporal plateaus where only a burst of variation reaches
     the next rung.
"""
from __future__ import annotations
import copy, math, os, random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from runtime.cache import LRUCache
from runtime.mutation import adaptive_mutation_rate, STRESS_PATIENCE
from runtime.parallel import map_ordered

from .genome import (MAX_STATE, TRI_STATE_MAX, MAX_GENES, MAX_CHROMS, MAX_TELOMERE,
                     MAX_ROUTING_PATCHES, ARCH_STATE_MAX,
                     DELAY_MULT_MIN, DELAY_MULT_MAX,
                     DELAY_LOG_STEP,
                     default_state_delays,
                     Genome, Chromosome, RoutingPatch, germline_telomere,
                     random_hex_gene, random_hex_chromosome, random_hex_genome,
                     random_input_layout)
from .hexgrid import hex_frontier_cells

# Process workers keep the current target in their process.  Passing a wide
# target (decoder/multiplier/comparator) through a ProcessPool partial for every
# genome repeatedly pickles all trials and expected traces.
_WORKER_TARGET = None


def init_eval_worker(target):
    global _WORKER_TARGET
    _WORKER_TARGET = target


def _evaluate_worker(genome):
    return _evaluate_nv_selection_record(genome, _WORKER_TARGET)


def is_branched(genome):
    """True for the branched, output-rooted encoding (substrates/nervous/branched).

    The GA's population machinery - elitism, lexicase, immigrants, the escape
    mechanisms - is encoding-agnostic; only five primitives are not (copy,
    signature, mutate, cross, rank). Those five dispatch on this, which is why
    the branched encoding needed no second copy of next_population.
    """
    from .branched import BranchedHexGenome
    return isinstance(genome, BranchedHexGenome)


def clone_genome(genome):
    """Fast structural copy: new Genome/Chromosome objects with fresh gene LISTS
    but SHARED gene objects. Safe because genes are never mutated in place -
    mutation always builds a new gene via _tweak_gene - so sharing the (immutable)
    gene objects is equivalent to deep-copying them. copy.deepcopy dominated
    reproduction (~90% of next_population's time); this replaces it on the hot
    path with an identical-behaviour, ~10x cheaper copy."""
    if is_branched(genome):
        # Branched genes ARE mutated in place (mutate_branched_hex edits the
        # fields of a chosen rule), so the shared-gene trick above would corrupt
        # the parent. This encoding pays for a real copy.
        from .branched_ga import clone_branched_hex
        clone = clone_branched_hex(genome)
        for attribute in ('_io_binding_progress', '_mut_rate'):
            if hasattr(genome, attribute):
                setattr(clone, attribute, getattr(genome, attribute))
        return clone
    sd = getattr(genome, 'state_delays', None)
    clone = Genome(
        chromosomes=[Chromosome(genes=c.genes[:], split=c.split, tag=c.tag,
                                telomere=getattr(c, 'telomere', MAX_TELOMERE),
                                wiring=getattr(c, 'wiring', False))
                     for c in genome.chromosomes],
        tag=genome.tag,
        state_delays=(sd[:] if sd else None),
        arch=getattr(genome, 'arch', 'single'),
        routing_patches=[
            RoutingPatch(patch.x, patch.y, patch.state)
            for patch in (getattr(genome, 'routing_patches', None) or ())])
    layout = getattr(genome, 'input_layout', None)
    if layout is not None:
        clone.input_layout = tuple(tuple(cell) for cell in layout)
    if hasattr(genome, '_io_binding_progress'):
        clone._io_binding_progress = genome._io_binding_progress
    if hasattr(genome, '_mut_rate'):
        # Self-adaptive mutation rate (runtime/escape.py) is a heritable trait
        # of the LINEAGE, so it must survive cloning. It is deliberately absent
        # from genome_signature: it changes no phenotype, and letting it into
        # the key would give one circuit several fitness-cache entries.
        clone._mut_rate = genome._mut_rate
    return clone
from .temporal import prepare_net, score_contract
from .scoring import contract_case_count
from .objectives import total_case_count

POPSIZE        = 120
ELITE_FRAC     = 0.10        # elites = this fraction of pop, UNLESS ELITE_COUNT set
ELITE_COUNT    = None        # exact elite count (GUI override); None = use ELITE_FRAC
IMMIGRANT_FRAC = 0.08
TOURNAMENT_K   = 4
# Preserve specialist lineages: small elite pools otherwise overuse one
# champion lineage.
EXPLORATION_PARENT_FRAC = 0.30
# Recombination has no effect if every hybrid is immediately subjected to a
# separate multi-edit mutation transaction.  Reserve a small, fixed cohort for
# evaluating the crossover itself.  This is not elitism (the cohort still has
# to earn a survivor slot next generation), nor a target-specific seed: it is
# the only chance for selection to see whether two inherited modules work
# together before a third operation changes them.
RECOMBINATION_EVALUATION_FRACTION = 0.10
MEAN_MUTATIONS = 4.0         # HOT-START mutation rate for simulated annealing:
                            # broad early exploration, cooled each generation by
                            # MUT_DECAY toward ~0 as the genes self-organise.
                            # (Was 1.2; annealing wants a high start - see below.)
MUT_DECAY      = 0.997       # slow cooldown: hard recurrent tasks need late
                             # variation - 0.997 cools 4.0 -> ~0.89 by gen 500,
                             # where the old 0.99 crashed it to ~0.03. alpha is
                             # PER-GENERATION, so tie it to run length - for very
                             # long runs use alpha close to 1 (e.g. 0.9999); alpha = 1.0
                             # disables annealing.
# Population evaluation is embarrassingly parallel; the old min(cpu, 8) left
# most cores idle on many-core machines (a 20-core box used 8). Scale with the
# machine, leave 2 cores for the GUI/OS, and cap at 16 where returns flatten
# (measured: 16->20 workers gained <1.1x while adding IPC overhead).
N_WORKERS      = max(1, min((os.cpu_count() or 2) - 2, 16))

# Very long runs (10k-100k generations) accumulate one fitness-cache entry per
# distinct genome ever seen. That is the only structure that grows without bound
# over a run, so cap it: when it exceeds this, drop it and let it refill (elites
# re-cache within a generation). Everything else the loop keeps is O(1) per gen.
FITNESS_CACHE_MAX = 200_000


# -- evaluation -----------------------------------------------------------------

def evaluate_nv_full(genome, target, *, _developed=None):
    """(scalar fitness, per-case score vector). Cases are the individual
    (trial, role) traces - the units epsilon-lexicase selection streams over. For
    Static combinational cases are the individual truth-table row/output
    checks.  They are deliberately retained rather than compressed into the
    scalar score, so selection can act on actual correctness.

    Under LIFESPAN SCORING (runtime/escape.py) the case vector is extended by
    one entry per developmental checkpoint. The scalar fitness is unaffected:
    it remains the ADULT organism's score, so a solved run still reads 1.0 for
    the circuit that actually gets grown."""
    if not getattr(target, 'temporal', False):
        from .nervous import score_nervous_full
        return score_nervous_full(
            genome, target, _developed=_developed)
    escape = getattr(target, '_escape', None)
    lifespan = escape is not None and escape.lifespan_scoring
    n_cases = total_case_count(target)
    # Optional fine timing: locally hill-climb inherited delays while holding the
    # grown topology and output readout fixed. Off by default (0), so the ordinary
    # path below is unchanged.
    samples = int(getattr(target, '_lifetime_samples', 0) or 0)
    if samples > 0:
        from .temporal import score_temporal_plastic
        seed = int(getattr(target, '_lifetime_seed', 20260727))
        step = float(getattr(target, '_lifetime_step', DELAY_LOG_STEP))
        tuned = score_temporal_plastic(
            genome, target, samples=samples, seed=seed, step=step)
        if tuned is not None:
            s, cases = tuned
            # Lifetime timing plasticity re-tunes delays inside its own session,
            # so it owns the whole evaluation. Lifespan checkpoints would have to
            # re-run that session per stage; instead the juvenile slots inherit
            # the adult score, leaving the case vector the right LENGTH (which is
            # all epsilon-lexicase requires) without inventing a juvenile measurement.
            if lifespan:
                cases = tuple(cases or ()) + (
                    (float(s),) * escape.lifespan_checkpoints)
            return s, cases
    snapshots, strategy = None, None
    if lifespan:
        # Grow ONCE and reuse the trajectory: the final snapshot is bit-identical
        # to grow_nervous's result, so the adult evaluation below is unchanged.
        from .io_placement import io_strategy
        from .objectives import grown_snapshots, prepare_grid
        strategy = io_strategy(target)
        snapshots = grown_snapshots(genome, target, 'nervous', strategy)
        prep = prepare_grid(genome, target, 'nervous', snapshots[-1], strategy)
    else:
        if _developed is None:
            prep = prepare_net(genome, target)
        else:
            from .temporal import prepare_net_grid
            grid, strategy = _developed
            prep = prepare_net_grid(
                genome, target, grid, strategy=strategy)
    if prep is None:
        return 0.0, (0.0,) * n_cases
    grid, routing, in_pos, out_pos, traces = prep
    if getattr(traces, 'overflow', False):
        return 0.0, (0.0,) * n_cases
    s, cases, _ = score_contract(traces, target)
    if lifespan:
        from .objectives import juvenile_scores
        cases = tuple(cases or ()) + juvenile_scores(
            genome, target, 'nervous', snapshots, strategy,
            escape.lifespan_checkpoints, s)
    return s, cases


def evaluate_nv(genome, target):
    """Scalar fitness (back-compat wrapper around evaluate_nv_full)."""
    return evaluate_nv_full(genome, target)[0]


def _output_module_scores(cases, target):
    """Per-output behavior scores used only to retain interchangeable arms.

    The executable contract still owns the reported score.  This extracts the
    already-evaluated cells belonging to each output so a good Sum arm is not
    discarded merely because Carry is still incomplete.  Static tables retain
    their level-balanced view; event-only temporal targets have one scored cell
    per ``(trial, role)`` in the order emitted by ``score_contract``.
    """
    base = tuple(float(value) for value in (cases or ()))
    outputs = tuple(getattr(target, 'outputs', ()) or ())
    roles = tuple(str(output.role) for output in outputs)
    if len(roles) < 2:
        return ()
    if not getattr(target, 'temporal', False):
        rows = tuple(getattr(target, 'cases', ()) or ())
        if len(base) != len(rows) * len(roles):
            return ()
        scores = []
        for output_index in range(len(roles)):
            levels = {0: [], 1: []}
            for row_index, (_inputs, expected) in enumerate(rows):
                if len(expected) <= output_index:
                    return ()
                levels[int(bool(expected[output_index]))].append(
                    base[row_index * len(roles) + output_index])
            means = [sum(values) / len(values)
                     for values in levels.values() if values]
            scores.append(sum(means) / len(means) if means else 0.0)
        return tuple(scores)

    constraints = tuple(getattr(
        getattr(target, 'contract', None), 'constraints', ()) or ())
    if (len(constraints) != 1
            or getattr(constraints[0], 'relation', '') != 'event_correspondence'):
        return ()
    by_role = {role: [] for role in roles}
    case_index = 0
    for trial in getattr(target, 'trials', ()):
        for role in getattr(trial, 'expected', {}):
            if case_index >= len(base):
                return ()
            if role in by_role:
                by_role[role].append(base[case_index])
            case_index += 1
    if case_index == 0:
        return ()
    return tuple(
        sum(by_role[role]) / len(by_role[role]) if by_role[role] else 0.0
        for role in roles)


def _evaluate_nv_selection_record(genome, target):
    """Fitness record plus selection-only I/O viability and escape objectives.

    Behavioral fitness remains untouched. The extra progress tuple only
    distinguishes otherwise-equal zero-fitness wiring genomes; the juvenile and
    robustness entries are the escape objectives, which rank_key applies
    strictly BELOW fitness (see runtime/escape.py).
    """
    from .io_placement import growth_seeds, io_strategy
    from .objectives import escape_objectives, structural_topology
    escape = getattr(target, '_escape', None)
    lifespan = escape is not None and escape.lifespan_scoring
    lifetime_samples = int(
        getattr(target, '_lifetime_samples', 0) or 0)
    developed = None
    if is_branched(genome):
        # Deliberately NOT pre-grown. The grow-once optimisation hands the body
        # to prepare_net_grid, which fits output probes to the whole organism -
        # and a branched genome's outputs are its arm ROOTS. Routing it through
        # the shared path would silently turn an output-rooted encoding into a
        # branched one with fitted probes, which is a different experiment.
        pass
    elif not lifespan and lifetime_samples <= 0:
        # Behavioral evaluation and the final topology tie-break inspect the
        # same mature phenotype. Grow once and pass that body to both.
        from .nervous import grow_nervous
        strategy = io_strategy(target)
        grid = grow_nervous(
            genome, seeds=growth_seeds(target, strategy, genome),
            grid_size=target.grid_size, iters=target.iters)
        developed = (grid, strategy)
    fitness, cases = evaluate_nv_full(
        genome, target, _developed=developed)
    output_scores = _output_module_scores(cases, target)
    total = target.n_inputs + len(target.outputs)
    juvenile, robust = escape_objectives(genome, target, 'nervous', cases)
    topology = structural_topology(
        genome, target, _developed=developed)
    if io_strategy(target) not in (
            'terminal_nodes', 'wiring_chromosome', 'spatial_chromosome'):
        return (fitness, cases, (total, total), juvenile, robust,
                topology, topology.score, output_scores)
    progress = getattr(genome, '_io_binding_progress', (total, total))
    return (fitness, cases, progress, juvenile, robust,
            topology, topology.score, output_scores)


def branched_signature(genome):
    """Cache identity of a branched genome.

    Gene IDs are part of it, not incidental bookkeeping: development breaks a
    contested cell by (distance, arm label, gene id), so two rule sets that
    differ only in their IDs really can grow different bodies.
    """
    chroms = tuple(
        (tuple((gene.gene_id, gene.ctx_l, gene.ctx_r, gene.ctx_d,
                gene.self_in, gene.self_out, gene.branch_id, gene.depth,
                # The tau allele is PHYSICAL identity: two rule sets differing
                # only here grow the same shape with different timing and score
                # differently, so omitting it would let one cache entry answer
                # for both.
                getattr(gene, 'tau_index', 0))
               for gene in chromosome.genes),
         tuple((control.tolerance, control.telomere)
               for control in chromosome.controls))
        for chromosome in genome.chromosomes)
    inputs = tuple((gene.bearing, gene.distance) for gene in genome.inputs)
    outputs = tuple((gene.role, gene.bearing, gene.distance, gene.branch_id)
                    for gene in genome.outputs)
    return ('branched', chroms, inputs, outputs,
            getattr(genome, 'arch', 'tri3'))


def genome_signature(genome):
    """Hashable identity of a genome's evolvable content (for the fitness cache).
    Includes native input geometry, timing, architecture, and routing patches
    so physically different genomes are never cache-aliased."""
    if is_branched(genome):
        return branched_signature(genome)
    chroms = tuple(
        (c.tag, c.split, getattr(c, 'telomere', 0), getattr(c, 'wiring', False),
         tuple((g.ctx_l, g.ctx_r, g.ctx_d, g.self_in, g.self_out,
                getattr(g, 'tag', 0), getattr(g, 'io_selector', 0),
                getattr(g, 'io_kind', 0))
               for g in c.genes))
        for c in genome.chromosomes)
    sd = getattr(genome, 'state_delays', None)
    # arch is part of identity: the same integer fields decode to different
    # hardware under 'single' vs 'tri3', so they must never share a cache slot.
    patches = tuple(
        (int(patch.x), int(patch.y), int(patch.state))
        for patch in (getattr(genome, 'routing_patches', None) or ()))
    layout = (
        None if getattr(genome, 'input_layout', None) is None
        else tuple(tuple(cell) for cell in genome.input_layout))
    return (layout, chroms, tuple(sd) if sd else None,
            getattr(genome, 'arch', 'single'), patches)


def eval_batch_cases(genomes, target, cache=None, executor=None,
                     should_stop=None, on_progress=None):
    """Evaluate a population in parallel -> (fitnesses, case_vectors). `cache`
    ({signature: (fit, cases, binding_progress)}, owned by the caller) skips
    seen genomes. If a
    persistent `executor` (a ProcessPoolExecutor) is passed, it is reused instead
    of spawning a fresh worker pool every call - on Windows the per-generation
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
        # The controller-installed worker target avoids serializing the same
        # wide target once per genome.  Keep the explicit-target fallback for
        # direct API callers and tests that supply an uninitialised executor.
        fn = (_evaluate_worker if getattr(target, '_worker_target_installed', False)
              else
              partial(_evaluate_nv_selection_record, target=target))
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
    from .io_placement import record_binding_progress
    for genome, record in zip(genomes, out):
        progress = record[2] if len(record) > 2 else (
            target.n_inputs + len(target.outputs),) * 2
        record_binding_progress(genome, progress)
        record_escape_objectives(genome, record)
        genome._output_scores = (
            tuple(record[7]) if len(record) > 7 else ())
    return [r[0] for r in out], [r[1] for r in out]


def record_escape_objectives(genome, record):
    """Copy an evaluation record's escape objectives onto the genome.

    Evaluation happens in worker processes, so the objectives travel back in
    the record and are attached here, in the parent, exactly where the existing
    I/O binding progress is attached. rank_key reads them from the genome; the
    robustness SCALAR is derived later by EscapeState.apply_robustness_blend,
    because its mean-to-worst-case aggregation depends on the run's current
    best fitness, which a worker cannot know.
    """
    genome._juvenile_score = float(record[3]) if len(record) > 3 else 0.0
    genome._robust_cases = record[4] if len(record) > 4 else None
    if getattr(genome, '_robust_cases', None) is None:
        genome._robustness = 0.0
    # Topology travels in the record like every other worker-computed value, so
    # a CACHE HIT restores it exactly as a fresh evaluation would. Without this
    # a cached genome would rank with topology 0 and lose ties it should win.
    genome._topology = record[5] if len(record) > 5 else None
    genome._topology_score = (
        float(record[6]) if len(record) > 6 else 0.0)


def eval_batch_nv(genomes, target, cache=None):
    """Back-compat scalar batch (shares the (fit, cases) cache format)."""
    return eval_batch_cases(genomes, target, cache)[0]


# -- genetic operators ------------------------------------------------------------

def _poisson(lam):
    L = math.exp(-lam); k, p = 0, 1.0
    while p > L:
        k += 1; p *= random.random()
    return k - 1


_GENE_FIELDS = ["ctx_l", "ctx_r", "ctx_d", "self_in", "self_out"]
_STATE_BITS = (MAX_STATE - 1).bit_length()
# Tri-tile states are 15-bit (three packed 5-bit channels). Single-bit flips on
# those disjoint fields act on ONE channel at a time, giving per-channel
# mutability for free.
_TRI_STATE_BITS = (TRI_STATE_MAX - 1).bit_length()


def _arch_bits(genome):
    return _TRI_STATE_BITS if getattr(genome, 'arch', 'single') == 'tri3' else _STATE_BITS


def mutate_input_layout(genome, max_telomere=MAX_TELOMERE):
    """Move exactly ONE non-anchor input pad by ONE valid honeycomb edge.

    This is the whole input-placement neighbourhood, and it is deliberately
    tiny. A grown input terminal has a cliff - a genome that fails to express
    one required terminal receives no meaningful evaluation at all - whereas a
    discrete pad list always carries exactly the required number of pads, so
    every mutation lands on a valid, evaluable layout one step away from its
    parent.

    Input 0 stays at the origin as a coordinate gauge: translating the whole
    organism is behaviourally meaningless, so letting the anchor drift would
    only add neutral wandering. Every real RELATIVE arrangement is still
    reachable by moving the others. Occupied sites are excluded rather than
    repaired, so a move never manufactures a collision.
    """
    from .genome import input_layout_domain, input_layout_radius
    layout = getattr(genome, 'input_layout', None)
    if layout is None or len(layout) < 2:
        return False
    sites = [tuple(map(int, cell)) for cell in layout]
    domain = set(input_layout_domain(
        input_layout_radius(max_telomere, len(sites))))
    occupied = set(sites)
    indices = list(range(1, len(sites)))
    random.shuffle(indices)
    for index in indices:
        options = [neighbour
                   for neighbour in hex_frontier_cells(*sites[index])
                   if neighbour in domain and neighbour not in occupied]
        if not options:
            continue
        sites[index] = random.choice(options)
        genome.input_layout = tuple(sites)
        return True
    return False


def _arch_random_gene(genome, terminals=False):
    return random_hex_gene('tri3' if getattr(genome, 'arch', 'single') == 'tri3'
                           else 'single', terminals=terminals)


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


def _recombine_gene_fields(gene_a, gene_b, fields=None):
    """Recombine a single rule when no between-gene cut exists.

    The fields are the gene's actual alleles.  If at least two differ, exchange
    a non-empty proper subset so both children inherit from both parents and
    neither is a parental clone.  With fewer than two differing alleles a novel
    recombinant is mathematically impossible; the mandatory mutation that
    follows crossover still supplies new variation.
    """
    fields = (
        tuple(_GENE_FIELDS) + ('tag', 'io_kind')
        if fields is None else fields)
    differing = [field for field in fields
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
    layout = (
        None if getattr(genome, 'input_layout', None) is None
        else tuple(tuple(cell) for cell in genome.input_layout))
    return (
        getattr(genome, 'arch', 'single'),
        layout,
        tuple(
            (getattr(chromosome, 'wiring', False),
             tuple((*(getattr(gene, field) for field in _GENE_FIELDS),
                    getattr(gene, 'tag', 0), getattr(gene, 'io_selector', 0),
                    getattr(gene, 'io_kind', 0))
                   for gene in chromosome.genes))
            for chromosome in genome.chromosomes),
        tuple(
            (int(patch.x), int(patch.y), int(patch.state))
            for patch in (getattr(genome, 'routing_patches', None) or ())),
    )


def _recombination_environment(genome):
    """Physical context that branched arms must share before they can mix."""
    if not is_branched(genome):
        return None
    from .branched_ga import input_pads
    return tuple(input_pads(genome))


def _canonicalise(value, bits=_STATE_BITS, terminals=False):
    """Normalise a configuration onto its canonical encoding.

    The register is physically 5 bits and mutation really is a single bit flip,
    but only 22 of the 32 settings are distinct circuits (see
    hexgrid.CANONICAL_STATES). Normalising after the flip keeps the hardware
    model while stopping a genome from drifting into alias encodings - where a
    provably-inert bit would consume mutation events and split one circuit
    across two apparent node types.
    """
    from .hexgrid import canonical_state
    value = int(value)
    if bits == _STATE_BITS:
        return canonical_state(value, terminals)
    from .tritile import channel_configs, pack_channels
    # Tri tiles canonicalise per CHANNEL: the three 5-bit fields are three
    # independent Fig. 3 circuits and each carries the same redundancy.
    return pack_channels(*(canonical_state(channel)
                           for channel in channel_configs(value)))


def _other_state(value, bits=_STATE_BITS, terminals=False):
    """Flip one physical SRAM bit in a core-circuit configuration (single tile:
    5 bits; tri tile: 15 bits, one flip landing within a single channel), and
    land on a DIFFERENT circuit.

    Canonicalising a raw flip is not enough: flipping the AND/OR select bit of a
    buffer produces that buffer's own alias, which normalises straight back to
    where it started. That silently turned a fifth of all state mutations into
    no-ops - and a no-op mutation lets a multi-event transaction cancel back to
    an exact copy of its parent, which reproduction relies on never happening.
    So the flip is drawn from the one-bit neighbours that are a different
    circuit, which is the same rule ``_state_excluding`` already applies.
    """
    return _state_excluding(value, bits=bits, terminals=terminals)


def _state_excluding(*values, bits=_STATE_BITS, terminals=False):
    """Choose a one-bit neighbour excluding the supplied values when possible.

    Exclusion is judged on the CANONICAL encoding: a neighbour that merely
    aliases an excluded state is the same circuit and would not have excluded
    anything."""
    state_max = 1 << bits
    excluded = {_canonicalise(value, bits, terminals) for value in values
                if value is not None and 0 <= int(value) < state_max}
    base = int(values[0]) if values and values[0] is not None else 0
    nearby = [candidate for candidate in
              (_canonicalise(base ^ (1 << bit), bits, terminals)
               for bit in range(bits))
              if candidate not in excluded]
    if nearby:
        return random.choice(nearby)
    # Defensive fallback when exclusions cover every one-bit neighbour.
    for _attempt in range(32):
        pick = _canonicalise(random.randrange(state_max), bits, terminals)
        if pick not in excluded:
            return pick
    return _canonicalise(random.randrange(state_max), bits, terminals)


def _force_nonparent_tweak(genome, parent, terminals=False):
    """Finish a multi-event mutation transaction with protected novelty.

    Earlier edits may cancel one another.  This last allele is chosen different
    from both its current value and the value at the same parental locus, so the
    transaction cannot net back to a clone.  It is constructive; no offspring is
    generated and rejected/retried by signature.
    """
    bits = _arch_bits(genome)
    with_genes = [ci for ci, chromosome in enumerate(genome.chromosomes)
                  if chromosome.genes and not getattr(chromosome, 'wiring', False)]
    if not with_genes:
        if not genome.chromosomes:
            genome.chromosomes.append(_arch_random_chromosome(genome))
        else:
            random.choice(genome.chromosomes).genes.append(
                _arch_random_gene(genome, terminals))
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
    setattr(gene, field,
            _state_excluding(getattr(gene, field), parent_value,
                             bits=bits, terminals=terminals))
    genome.chromosomes[ci].genes[gi] = gene


def _arch_random_chromosome(genome, max_telomere=MAX_TELOMERE):
    return random_hex_chromosome(
        max_telomere=max_telomere,
        arch='tri3' if getattr(genome, 'arch', 'single') == 'tri3' else 'single')


def _tweak_gene(gene, bits=_STATE_BITS, terminals=False):
    g   = copy.copy(gene)
    fld = random.choice(_GENE_FIELDS)
    if fld == 'self_in' and g.self_in != 0 and random.random() < 0.2:
        g.self_in = 0                     # keep growth rules reachable
    else:
        setattr(g, fld,
                _other_state(getattr(g, fld), bits=bits,
                             terminals=terminals))
    return g


def _mutate_state_delay(genome):
    """Fine-mutate one routing state's width-preserving propagation delay."""
    base = genome.state_delays
    delays = list(base) if base else default_state_delays()
    s = random.randrange(1, MAX_STATE)
    factor = math.exp(random.choice((-DELAY_LOG_STEP, DELAY_LOG_STEP)))
    delays[s] = min(DELAY_MULT_MAX,
                    max(DELAY_MULT_MIN, delays[s] * factor))
    genome.state_delays = delays


def _mutate_routing_patch(genome):
    """Make one strictly local edit to the mature routing overlay."""
    patches = list(getattr(genome, 'routing_patches', None) or ())
    if not patches:
        return False
    index = random.randrange(len(patches))
    patch = copy.copy(patches[index])
    choice = random.random()
    if choice < 0.65:
        maximum = ARCH_STATE_MAX[getattr(genome, 'arch', 'single')]
        bits = (maximum - 1).bit_length()
        candidate = _canonicalise(
            int(patch.state) ^ (1 << random.randrange(bits)), bits)
        if not 0 < candidate < maximum:
            candidate = _canonicalise(
                1 + (int(patch.state) % (maximum - 1)), bits)
        patch.state = candidate
        patches[index] = patch
    elif choice < 0.90:
        axis = random.choice(('x', 'y'))
        setattr(patch, axis, int(getattr(patch, axis))
                + random.choice((-1, 1)))
        patches[index] = patch
    else:
        patches.pop(index)
    # One coordinate has one final routing allele.
    unique = {}
    for item in patches:
        unique[(int(item.x), int(item.y))] = item
    genome.routing_patches = list(unique.values())[-MAX_ROUTING_PATCHES:]
    return True


_MUT_OPS     = ["tweak", "duplicate", "add_gene", "del_gene",
                "add_chrom", "del_chrom", "split", "telomere", "delay"]
# Delay appears only under the width-preserving model. Its weight keeps timing
# tuning frequent enough to evolve alongside routing. I/O mutation is scheduled
# when an evolvable io_placement strategy is active (evolve_io) - off by default,
# separately below, so structural reheating cannot repeatedly scramble it.
_MUT_WEIGHTS = [0.32, 0.14, 0.14, 0.11, 0.05, 0.05, 0.11, 0.08, 0.30]

# I/O placement is a co-adapted interface, not another body edit. Ordinary
# children retain one independent low-frequency edit. Plateau archive
# descendants may instead receive an explicit coordinated bundle: this crosses
# a port co-adaptation valley without letting every reheated structural event
# scramble the interface.
IO_MUTATION_PROB = 0.20


def _mutate_once_nv(genome, max_telomere=MAX_TELOMERE,
                    chromosome_count=None, evolve_delay=False,
                    evolve_io=False, terminals=False):
    """Apply one feasible, state-changing mutation to a nervous-net genome.

    ``evolve_delay`` (the width-preserving 'pulse_delay' model) adds a mutation
    that tunes a node type's propagation delay. It is off for the paper's
    'uniform' node. I/O placement is handled once per child by ``mutate_nv``
    rather than once per structural mutation event."""
    bits = _arch_bits(genome)
    if not genome.chromosomes:
        genome.chromosomes.append(_arch_random_chromosome(genome, max_telomere))
        return

    chroms = genome.chromosomes
    has_wiring = any(getattr(c, 'wiring', False) for c in chroms)
    body = [c for c in chroms if not getattr(c, 'wiring', False)]
    with_genes = [c for c in body if c.genes]
    options = []
    if evolve_delay:
        options.append('delay')
    if with_genes:
        options.append('tweak')
    if any(c.genes and len(c.genes) < MAX_GENES for c in body):
        options.append('duplicate')
    if any(len(c.genes) < MAX_GENES for c in body):
        options.append('add_gene')
    if any(len(c.genes) > 1 for c in body):
        options.append('del_gene')
    # A configured chromosome count is a structural constraint. Direct callers
    # may leave it as None to retain the older, evolvable-count experiment.
    # A dedicated I/O map is chromosome three by definition.  Do not let
    # chromosome-count experiments insert/delete body chromosomes around it.
    if chromosome_count is None and not has_wiring and len(chroms) < MAX_CHROMS:
        options.append('add_chrom')
    if chromosome_count is None and not has_wiring and len(body) > 1:
        options.append('del_chrom')
    if any(len(c.genes) > 2 for c in body):
        options.append('split')
    telomeres = []
    for c in body:
        base = getattr(c, 'telomere', 10)
        values = [base + d for d in (-3, -2, -1, 1, 2, 3)
                  if 1 <= base + d <= max_telomere]
        if values:
            telomeres.append((c, values))
    if telomeres:
        options.append('telomere')

    weights = [_MUT_WEIGHTS[_MUT_OPS.index(op)] for op in options]
    op = random.choices(options, weights=weights)[0]
    if op == 'delay':
        _mutate_state_delay(genome)
    elif op == 'tweak':
        chrom = random.choice(with_genes)
        idx = random.randrange(len(chrom.genes))
        chrom.genes[idx] = _tweak_gene(chrom.genes[idx], bits=bits,
                                       terminals=terminals)
    elif op == 'duplicate':
        chrom = random.choice([c for c in body
                               if c.genes and len(c.genes) < MAX_GENES])
        chrom.genes.insert(random.randrange(len(chrom.genes) + 1),
                           _tweak_gene(random.choice(chrom.genes),
                                       bits=bits, terminals=terminals))
    elif op == 'add_gene':
        random.choice([c for c in body if len(c.genes) < MAX_GENES]).genes.append(
            _arch_random_gene(genome, terminals))
    elif op == 'del_gene':
        chrom = random.choice([c for c in body if len(c.genes) > 1])
        chrom.genes.pop(random.randrange(len(chrom.genes)))
    elif op == 'add_chrom':
        chroms.append(_arch_random_chromosome(genome, max_telomere))
    elif op == 'del_chrom':
        chroms.remove(min(body, key=lambda c: len(c.genes)))
    elif op == 'split':
        chrom = random.choice([c for c in body if len(c.genes) > 2])
        values = [s for s in range(1, len(chrom.genes)) if s != chrom.split]
        chrom.split = random.choice(values)
    else:  # telomere
        chrom, values = random.choice(telomeres)
        chrom.telomere = random.choice(values)


def timing_mutation_flags(model, evolve_delay=None):
    """Resolve the delay-mutation toggle for a node-timing model.

    ``None`` keeps the model's pairing ('pulse_delay' <-> delay mutation) -
    today's behaviour. An explicit False disables it, giving width-preserving
    transport at the FIXED base delay: the ablation that isolates width
    preservation from delay evolvability when comparing models."""
    if evolve_delay is None:
        evolve_delay = model == 'pulse_delay'
    return evolve_delay


def mutate_nv(genome, mean_mutations=None, max_telomere=MAX_TELOMERE,
              chromosome_count=None, evolve_delay=False, evolve_io=False,
              io_placement=None, io_mutations=1, coordinated_io=False,
              local_only=False):
    if (chromosome_count is not None
            and len(genome.chromosomes) != chromosome_count):
        raise ValueError('expected %d chromosomes, got %d' %
                         (chromosome_count, len(genome.chromosomes)))
    if is_branched(genome):
        # The branched operator owns its own I/O alleles and arm controls, so
        # the native layout / delay / patch arguments have nothing to act on
        # here; they are accepted and ignored rather than raising, because the
        # breeding loop passes them uniformly for every encoding.
        from .branched_ga import MAX_TELOMERE as BRANCHED_MAX_TELOMERE
        from .branched_ga import mutate_branched_hex
        child = mutate_branched_hex(
            genome, MEAN_MUTATIONS if mean_mutations is None else mean_mutations,
            max_telomere=min(int(max_telomere), BRANCHED_MAX_TELOMERE))
        if hasattr(genome, '_mut_rate'):
            child._mut_rate = genome._mut_rate
        return child
    g = clone_genome(genome)
    # Under terminal_nodes binding, states 16/17 are the dedicated input and
    # output NODE TYPES rather than aliases of dead/buffer-D, so canonical
    # normalisation must leave them alone or mutation would erase every
    # terminal the organism has grown.
    terminals = (io_placement == 'terminal_nodes')
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    if local_only and _mutate_routing_patch(g):
        return g
    lam = MEAN_MUTATIONS if mean_mutations is None else mean_mutations
    events = max(1, _poisson(lam))
    for _ in range(events - 1):
        _mutate_once_nv(g, max_telomere=max_telomere,
                        chromosome_count=chromosome_count,
                        evolve_delay=evolve_delay, evolve_io=evolve_io,
                        terminals=terminals)
    if events == 1:
        _mutate_once_nv(g, max_telomere=max_telomere,
                        chromosome_count=chromosome_count,
                        evolve_delay=evolve_delay, evolve_io=evolve_io,
                        terminals=terminals)
    else:
        # A single mandatory routing tweak guarantees non-clone novelty. When
        # timing is evolving on top of settled routing, also give the transaction
        # a chance to land on a timing edit so runs do not stall on routing churn.
        if evolve_delay and random.random() < 0.5:
            _mutate_state_delay(g)
        else:
            _force_nonparent_tweak(g, genome, terminals)
    # Evolved input geometry: one pad, one edge. Gated on the genome actually
    # carrying a layout, so fixed-input genomes are untouched.
    if (getattr(g, 'input_layout', None) is not None
            and len(g.input_layout) > 1
            and random.random() < IO_MUTATION_PROB):
        mutate_input_layout(g, max_telomere)
    for chromosome in g.chromosomes:
        _normalize_split(chromosome)
    return g


# back-compat name (the old mutation operator lived in substrates/nervous/genome.py)
mutate_hex = mutate_nv


def crossover_nv(pa, pb, io_placement=None):
    """Tag-matched hierarchical crossover.

    Multi-gene homologs cross at a genuine interior gene boundary.  When the
    common homolog has only one gene, recombine that rule's fields instead, so a
    minimal body chromosome still has useful field-level recombination. A
    wiring chromosome is inherited whole because its port assignments form one
    co-adapted interface.
    """
    arch_a = getattr(pa, 'arch', 'single')
    arch_b = getattr(pb, 'arch', 'single')
    if arch_a != arch_b:
        raise ValueError('cannot crossover different tile architectures')
    if is_branched(pa) or is_branched(pb):
        if not (is_branched(pa) and is_branched(pb)):
            # Native chromosomes and output-rooted arms have no homologous
            # unit.  Treat this as mutation-only reproduction rather than
            # passing a native genome to the branched arm operator.
            return clone_genome(pa), clone_genome(pb)
        # Branched recombination trades whole ARMS, and its operator returns one
        # child, so the reciprocal cross supplies the second. Tag matching does
        # not apply: an arm's homolog is the arm at the same chromosome and half.
        from .branched_ga import crossover_branched_hex
        return (crossover_branched_hex(pa, pb),
                crossover_branched_hex(pb, pa))
    ca, cb = clone_genome(pa), clone_genome(pb)
    used_b = set()
    for i, chrom_a in enumerate(ca.chromosomes):
        best_j, best_dist = None, float("inf")
        for j, chrom_b in enumerate(cb.chromosomes):
            if j in used_b:
                continue
            if (getattr(chrom_a, 'wiring', False)
                    != getattr(chrom_b, 'wiring', False)):
                continue
            d = abs(chrom_a.tag - chrom_b.tag)
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is None:
            continue
        used_b.add(best_j)
        chrom_b = cb.chromosomes[best_j]
        if (getattr(chrom_a, 'wiring', False)
                and io_placement != 'spatial_chromosome'):
            # Splicing A/B/Q assignments from different parents destroys both
            # interfaces even when each parental map is valid. The cloned
            # children already carry one complete map apiece; variation remains
            # available through the independent I/O mutation above.
            continue
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
            fields = (('tag', 'io_selector')
                      if getattr(chrom_a, 'wiring', False) else None)
            gene_a, gene_b = _recombine_gene_fields(
                genes_a[0], genes_b[0], fields=fields)
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
    patches_a = list(getattr(pa, 'routing_patches', None) or ())
    patches_b = list(getattr(pb, 'routing_patches', None) or ())
    # Input geometry is a single co-adapted physical module. Recombining
    # individual pad coordinates would manufacture collisions and tear apart
    # relative arrangements that only work together, so a child takes one
    # parent's LAYOUT ENTIRE or the other's - never a mixture.
    layout_b = getattr(pb, 'input_layout', None)
    layout_a = getattr(pa, 'input_layout', None)
    for child, mine, theirs in ((ca, layout_a, layout_b),
                                (cb, layout_b, layout_a)):
        chosen = mine
        if (theirs is not None and mine is not None
                and len(theirs) == len(mine) and random.random() < 0.5):
            chosen = theirs
        if chosen is not None:
            child.input_layout = tuple(tuple(cell) for cell in chosen)
    if patches_a or patches_b:
        cut_a = random.randrange(len(patches_a) + 1)
        cut_b = random.randrange(len(patches_b) + 1)

        def patch_child(items):
            # Later patches at the same coordinate win during evaluation; keep
            # only that final allele so the heritable representation is honest.
            unique = {}
            for patch in items:
                unique[(int(patch.x), int(patch.y))] = RoutingPatch(
                    int(patch.x), int(patch.y), int(patch.state))
            return list(unique.values())[-MAX_ROUTING_PATCHES:]

        ca.routing_patches = patch_child(
            patches_a[:cut_a] + patches_b[cut_b:])
        cb.routing_patches = patch_child(
            patches_b[:cut_b] + patches_a[cut_a:])
    return ca, cb


def n_genes(genome):
    return (
        sum(len(c.genes) for c in genome.chromosomes)
        + len(getattr(genome, 'routing_patches', None) or ()))


def rank_key(genome, fitness):
    """Selection / ranking key (maximise).

    Fully wired organisms rank above incomplete wiring maps; within the same
    viability tier honest behavioral fitness dominates. The final tier is
    target-blind source-reachable topology: connected wiring, multi-input
    integration, and feedback receive diminishing-return credit, while
    disconnected bulk earns nothing. Gene count and telomere are deliberately
    absent. These selection tiers never alter the reported behavioral fitness.

    Between fitness and topology sit the two ESCAPE objectives, in this order
    and no other:

        viability > fitness > robustness > juvenile > topology

    Both are LEXICOGRAPHICALLY BELOW fitness, which is the whole safety
    argument for them. Robustness can only ever separate two circuits that are
    already equally correct, so a robust-but-wrong circuit can never outrank a
    correct one; juvenile (lifespan) credit likewise only breaks ties, most
    usefully across the flat zero-fitness region where nothing else does. Both
    are 0.0 for every genome when their mechanism is off, so the ordering
    collapses to exactly the pre-escape one."""
    from .io_placement import binding_viability
    viability = binding_viability(genome)
    robustness = getattr(genome, '_robustness', 0.0) or 0.0
    juvenile = getattr(genome, '_juvenile_score', 0.0) or 0.0
    topology = getattr(genome, '_topology_score', 0.0) or 0.0
    return (viability, fitness, robustness, juvenile, topology)


def tournament_nv(population, fitnesses):
    """Tournament selection using :func:`rank_key`.

    Behavior dominates; source-reachable topology breaks final ties without
    changing the reported score or preferring smaller genomes."""
    idx = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    return population[max(idx, key=lambda i: rank_key(population[i], fitnesses[i]))]


def _lexicase_parent(population, case_vecs, case_subset=None):
    """epsilon-lexicase selection (La Cava et al.): stream the cases in random order;
    at each case keep only candidates within epsilon (median absolute deviation) of
    that case's best. Averages hide a single failing trial (~1/12 of a mean);
    lexicase makes every case a hard filter some of the time, so specialists on
    the currently-failing cases are selected and recombined - the mechanism that
    drives populations to ALL-cases-perfect rather than high-average.

    The epsilon is what makes this usable on CONTINUOUS scores. Plain lexicase filters
    on exact ties, which essentially never occur between floats, so the first
    case drawn would decide every selection on its own and the rest would be
    dead weight - indistinguishable from single-case selection while still
    looking like it maintains diversity.

    ``case_subset`` restricts the stream to a sample of case indices for this
    generation (downsampled lexicase). Same selection quality for a fraction of
    the cases touched, and because the sample is redrawn every generation it is
    also what "rotate the stimulus set" amounts to here."""
    from .io_placement import binding_viability
    n_cases = len(case_vecs[0])
    viability = [binding_viability(genome) for genome in population]
    best_viability = max(viability)
    cand = [index for index, value in enumerate(viability)
            if value >= best_viability - 1e-12]
    pool = (list(case_subset) if case_subset
            else list(range(n_cases)))
    for c in random.sample(pool, len(pool)):
        vals = [case_vecs[i][c] for i in cand]
        best = max(vals)
        # A truth-table check is an exact 0/1 fact, not a noisy measurement.
        # MAD epsilon can become 1.0 on an even 50/50 split and retain BOTH the right
        # and wrong candidates, making that case exert no selection at all.
        # Keep epsilon for genuinely continuous temporal/duty scores, but use exact
        # lexicase whenever every value is discrete.
        if all(abs(v - round(v)) <= 1e-12 for v in vals):
            eps = 0.0
        else:
            srt = sorted(vals)
            med = srt[len(srt) // 2]
            eps = sorted(abs(v - med) for v in vals)[len(vals) // 2]
        cand = [i for i, v in zip(cand, vals) if v >= best - eps]
        if len(cand) == 1:
            break
    return population[random.choice(cand)]


# Parent-selection scheme. Measured head-to-head at equal budget (see memory):
# lexicase helped only the pair detector (+0.11), regressed coincidence (-0.13)
# and pattern (-0.13), and left every plateau (latch/toggle/stepper) at the
# IDENTICAL value tournament reaches - the plateaus are representational, not a
# selection-pressure problem. Tournament is the better default; lexicase stays
# available for experiments.
SELECTION = 'tournament'          # 'tournament' | 'lexicase'


def _escape_off():
    """The all-mechanisms-off escape config (imported lazily: runtime.config
    imports the substrate's PulseConfig, so a module-level import here would
    close an import cycle)."""
    from runtime.escape import OFF
    return OFF


def select_parent(population, fitnesses, case_vecs=None, case_subset=None):
    if (SELECTION == 'lexicase' and case_vecs is not None
            and case_vecs[0] is not None):
        return _lexicase_parent(population, case_vecs, case_subset)
    return tournament_nv(population, fitnesses)


# Stress-induced mutagenesis (the bacterial SOS response): baseline mutation
# until the population has stalled for STRESS_PATIENCE generations, then the rate
# ramps toward STRESS_MAX_MULT the longer it stays stuck - raising variation to
# climb off a plateau, then relaxing the instant progress resumes. The temporal
# plateaus (latch/toggle/stepper) are exactly the deceptive landscapes this is
# meant for: flat regions where only a burst of variation reaches the next rung.
def consolidate_population(parents, parent_fitnesses, parent_cases,
                           offspring, offspring_fitnesses, offspring_cases):
    """Terminal ``(mu + lambda)`` selection after a perfect solution exists.

    Ordinary evolution is mostly generational, with a bounded rotating reserve
    of distinct best-on-case behaviors. Once fitness 1.0 has been reached, this terminal phase selects
    the best population-sized set from evaluated parents and offspring so
    perfect circuits can accumulate and the population mean can converge to 1.
    Exact-rank ties are shuffled before sorting to preserve neutral turnover.
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


def _append_role_module_candidates(children, population, fitnesses,
                                   recombination):
    """Keep and join measured output specialists in compatible pad cohorts."""
    if (not population or not is_branched(population[0])
            or len(getattr(population[0], 'outputs', ())) < 2):
        return
    from .branched_ga import assemble_role_modules, input_pads
    roles = tuple(getattr(population[0], 'outputs', ()))
    groups = {}
    for index, genome in enumerate(population):
        groups.setdefault(tuple(input_pads(genome)), []).append(index)

    assemblies = []
    if recombination:
        for indices in groups.values():
            donors, donor_scores = {}, []
            for role_index, role in enumerate(roles):
                eligible = [index for index in indices
                            if len(getattr(population[index], '_output_scores', ()))
                            > role_index]
                if not eligible:
                    break
                best = max(eligible, key=lambda index: (
                    population[index]._output_scores[role_index],
                    rank_key(population[index], fitnesses[index])))
                donors[int(role.branch_id)] = population[best]
                donor_scores.append(population[best]._output_scores[role_index])
            else:
                if len(set(id(donor) for donor in donors.values())) > 1:
                    base = max(indices, key=lambda index: rank_key(
                        population[index], fitnesses[index]))
                    assemblies.append(((min(donor_scores), sum(donor_scores)),
                                       assemble_role_modules(
                                           population[base], donors)))
    if assemblies and len(children) < len(population):
        children.append(max(assemblies, key=lambda item: item[0])[1])

    # These retain already evaluated behavior; they neither change fitness nor
    # prescribe a circuit. A specialist must survive until a later crossover
    # has a chance to join it with a different role specialist.
    seen = {genome_signature(genome) for genome in children}
    for role_index, _role in enumerate(roles):
        if len(children) >= len(population):
            break
        eligible = [index for index, genome in enumerate(population)
                    if len(getattr(genome, '_output_scores', ())) > role_index]
        if not eligible:
            continue
        best = max(eligible, key=lambda index: (
            population[index]._output_scores[role_index],
            rank_key(population[index], fitnesses[index])))
        signature = genome_signature(population[best])
        if signature not in seen:
            children.append(clone_genome(population[best]))
            seen.add(signature)


def next_population(population, fitnesses, make_genome=None, case_vecs=None,
                    mean_mutations=None, selection=None, ga_config=None,
                    chromosome_count=None, recombination=True,
                    evolve_delay=None, evolve_io=False, io_placement=None,
                    archive_parent=None, stagnation=0,
                    rescue_candidates=None, escape=None, mutation_limit=None):
    """Breed one exploratory offspring generation.

    Elites are normally a breeding pool only. A stalled spatial run preserves
    one cloned archive champion while it explores phenotype-local routing
    patches; every other returned entry is a rescue proposal, immigrant, or
    recombined/mutated child."""
    pop = len(population)
    elite_count = (ELITE_COUNT if ga_config is None else ga_config.elite_count)
    immigrant_fraction = (IMMIGRANT_FRAC if ga_config is None
                          else ga_config.immigrant_fraction)
    tournament_size = (TOURNAMENT_K if ga_config is None
                       else ga_config.tournament_size)
    # Local-minimum escape mechanisms. An explicit argument wins; otherwise the
    # run configuration supplies them, so the desktop controller and the
    # headless driver cannot end up breeding under different rules.
    if escape is None:
        escape = (getattr(ga_config, 'escape', None) if ga_config is not None
                  else None) or _escape_off()
    if mutation_limit is None:
        mutation_limit = (getattr(ga_config, 'mutation_limit', 8.0)
                          if ga_config is not None else 8.0)
    recombination = (
        recombination and
        (getattr(ga_config, 'recombination_enabled', True)
         if ga_config is not None else True))
    population_archs = {getattr(genome, 'arch', 'single')
                        for genome in population}
    if len(population_archs) > 1:
        raise ValueError('population mixes tile architectures')
    pop_arch = next(iter(population_archs), 'single')
    if (ga_config is not None
            and getattr(ga_config, 'tile_arch', pop_arch) != pop_arch):
        raise ValueError('population violates configured tile architecture')
    if ga_config is not None and ga_config.chromosome_count is not None:
        chromosome_count = ga_config.chromosome_count
    if chromosome_count is not None:
        if not 1 <= chromosome_count <= MAX_CHROMS:
            raise ValueError('chromosome_count must be between 1 and %d' %
                             MAX_CHROMS)
        if any(len(genome.chromosomes) != chromosome_count
               for genome in population):
            raise ValueError('population violates configured chromosome count')
    # Evolvable strategies carry body priorities, type mappings, or x/y anchors.
    strategy = (io_placement or (
        getattr(ga_config, 'io_placement', 'fixed')
        if ga_config is not None else 'fixed'))
    if not evolve_io and strategy in (
            'terminal_nodes', 'tag_rank', 'wiring_chromosome',
            'spatial_chromosome'):
        evolve_io = True
    if make_genome is None:
        # Immigrants must match the population's tile architecture, or a 'single'
        # immigrant would pollute a tri3 run (and vice-versa) - different hardware
        # under the same integer genome. Method B immigrants also need a flagged
        # wiring chromosome with a seeded port map.
        want_wiring = strategy in (
            'wiring_chromosome', 'spatial_chromosome')
        inferred_ports = 0
        inferred_inputs = inferred_outputs = 0
        if want_wiring and population:
            from .io_placement import wiring_chromosome
            port_map = wiring_chromosome(population[0])
            inferred_ports = len(port_map.genes) if port_map is not None else 0
        if strategy == 'terminal_nodes' and population:
            kinds = [
                int(getattr(gene, 'io_kind', 0))
                for chromosome in population[0].chromosomes
                for gene in chromosome.genes]
            inferred_inputs = max(1, kinds.count(1) // 2)
            inferred_outputs = max(1, kinds.count(2) // 2)
        make_genome = lambda: random_hex_genome(chromosome_count or 2,
                                                arch=pop_arch,
                                                wiring_chromosome=(
                                                    strategy
                                                    == 'wiring_chromosome'),
                                                spatial_chromosome=(
                                                    strategy
                                                    == 'spatial_chromosome'),
                                                n_ports=inferred_ports,
                                                tag_rank=(strategy == 'tag_rank'),
                                                terminal_nodes=(
                                                    strategy == 'terminal_nodes'),
                                                n_inputs=inferred_inputs,
                                                n_outputs=inferred_outputs)
        # An immigrant must carry a layout of the SAME length as the population
        # it joins, or it is born unbindable and wastes an evaluation slot.
        # Read the shape from the population rather than guessing it.
        reference_layout = (getattr(population[0], 'input_layout', None)
                            if population else None)
        if reference_layout is not None:
            pad_count = len(reference_layout)
            plain_genome = make_genome

            def make_genome():
                genome = plain_genome()
                genome.input_layout = random_input_layout(
                    pad_count,
                    MAX_TELOMERE if ga_config is None
                    else ga_config.max_telomere)
                return genome
    # Enable the timing mutation belonging to the selected node model. An
    # explicit argument wins; otherwise read the run configuration - its
    # evolve_delay toggle overrides the model pairing (None = paired).
    if evolve_delay is None:
        model = (getattr(ga_config, 'node_model', 'uniform')
                 if ga_config is not None else 'uniform')
        evolve_delay = timing_mutation_flags(
            model, getattr(ga_config, 'evolve_delay', None))
    n_elite = elite_count if elite_count is not None else int(pop * ELITE_FRAC)
    n_elite = max(0, min(n_elite, pop))
    n_imm   = min(int(round(pop * immigrant_fraction)), pop)
    order   = sorted(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]),
                     reverse=True)
    # Elites are the RECOMBINATION PARENT POOL (truncation selection): the top
    # n_elite genomes breed the offspring. The sole exception is one cloned
    # archive champion during a stalled spatial local-search phase.
    # Parents are drawn by
    # tournament WITHIN that pool (TOURNAMENT_K=1 -> uniform among elites). n_elite==0
    # falls back to normal selection over the whole population.
    spatial_plateau = (
        strategy == 'spatial_chromosome'
        and archive_parent is not None
        and stagnation >= STRESS_PATIENCE)
    new_pop = (
        [clone_genome(archive_parent)] if spatial_plateau and pop else [])
    room = pop - len(new_pop)
    new_pop += [
        clone_genome(candidate)
        for candidate in list(rescue_candidates or ())[:room]]
    remaining = pop - len(new_pop)
    new_pop += [make_genome() for _ in range(min(n_imm, remaining))]
    if (archive_parent is not None and stagnation >= STRESS_PATIENCE
            and len(new_pop) < pop):
        archive_count = min(
            max(1, int(round(pop * 0.10))), pop - len(new_pop))
        for index in range(archive_count):
            new_pop.append(mutate_nv(
                archive_parent, mean_mutations,
                max_telomere=(MAX_TELOMERE if ga_config is None
                              else ga_config.max_telomere),
                chromosome_count=chromosome_count,
                evolve_delay=evolve_delay, evolve_io=evolve_io,
                io_placement=strategy, io_mutations=2,
                coordinated_io=(evolve_io and index % 2 == 0),
                local_only=(
                    spatial_plateau
                    and bool(getattr(
                        archive_parent, 'routing_patches', None))
                    and index < max(1, archive_count // 2))))
    _append_role_module_candidates(new_pop, population, fitnesses,
                                   recombination)
    # epsilon-lexicase (when selected) must stream over the WHOLE population; the elite-only
    # breeding pool would otherwise mask it whenever elites>0, so bypass the pool then.
    selection = ((SELECTION if ga_config is None else ga_config.selection)
                 if selection is None else selection)
    use_lexicase = (selection == 'lexicase' and case_vecs is not None
                    and case_vecs[0] is not None)
    # One case sample per GENERATION, shared by every selection event in it -
    # resampling per parent would average the downsampling away and lose the
    # selection pressure it is supposed to concentrate.
    case_subset = None
    if use_lexicase:
        from runtime.escape import lexicase_case_subset
        case_subset = lexicase_case_subset(len(case_vecs[0]), escape)
    if n_elite > 0 and not use_lexicase:
        elite = order[:n_elite]
        k = min(tournament_size, len(elite))
        elite_parent = lambda: population[max(
            random.sample(elite, k),
            key=lambda i: rank_key(population[i], fitnesses[i]))]
    else:
        elite_parent = lambda: (
            _lexicase_parent(population, case_vecs, case_subset)
            if use_lexicase else population[max(
                random.sample(range(pop), min(tournament_size, pop)),
                key=lambda i: rank_key(population[i], fitnesses[i]))])
    residual = order[n_elite:] if n_elite < pop else order
    recombination_signatures = [
        _recombination_signature(genome) for genome in population]
    recombination_environments = [
        _recombination_environment(genome) for genome in population]

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
            chosen = _lexicase_parent(local_pop, local_cases, case_subset)
            return candidates[next(i for i, genome in enumerate(local_pop)
                                   if genome is chosen)]
        k = min(tournament_size, len(candidates))
        return max(random.sample(candidates, k),
                   key=lambda i: rank_key(population[i], fitnesses[i]))

    def parent_pair():
        if pop == 1:
            return population[0], population[0]

        def mate_pool(first, candidates):
            distinct = [
                index for index in candidates
                if recombination_signatures[index]
                != recombination_signatures[first]]
            pool = distinct or candidates
            environment = recombination_environments[first]
            if environment is not None:
                compatible = [
                    index for index in pool
                    if recombination_environments[index] == environment]
                if compatible:
                    pool = compatible
            return pool

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
        candidates = [i for i in range(pop) if i != first]
        parent_pool = mate_pool(first, candidates)
        if use_lexicase:
            from runtime.escape import complementary_parent_index
            second = complementary_parent_index(
                first, parent_pool, case_vecs, fitnesses,
                case_subset)
        else:
            second = choose(first)
            if second not in parent_pool:
                second = pick_index(parent_pool)
        return population[first], population[second]

    # ``mean_mutations`` is None on the direct-API path, where mutate_nv falls
    # back to its own default. Self-adaptation needs a real number to perturb,
    # so resolve it here rather than propagating the None.
    adaptive_base = MEAN_MUTATIONS if mean_mutations is None else mean_mutations

    def child_rate(child):
        # Under self-adaptive mutation each individual carries its own rate, so
        # a stuck lineage can heat up while a lineage that is still improving
        # stays cool - the per-lineage counterpart to the population-wide SOS
        # reheat in runtime/mutation.py.
        if not escape.self_adaptive_mutation:
            return mean_mutations
        from runtime.escape import mutation_rate_of
        return mutation_rate_of(child, adaptive_base)

    def mutate_child(child):
        # Plateau pressure is a portfolio, not eight destructive edits on every
        # descendant: most patched lineages receive one or two cell-local
        # changes, while the remainder retain the ordinary macro operator and
        # its reheated rate.
        local = (
            spatial_plateau
            and bool(getattr(child, 'routing_patches', None))
            and random.random() < 0.60)
        if not local:
            return mutate_nv(
                child, child_rate(child),
                max_telomere=(MAX_TELOMERE if ga_config is None
                              else ga_config.max_telomere),
                chromosome_count=chromosome_count,
                evolve_delay=evolve_delay, evolve_io=evolve_io,
                io_placement=strategy)
        result = mutate_nv(
            child, 1.0,
            max_telomere=(MAX_TELOMERE if ga_config is None
                          else ga_config.max_telomere),
            chromosome_count=chromosome_count,
            evolve_delay=evolve_delay, evolve_io=False,
            io_placement=strategy, local_only=True)
        if (getattr(result, 'routing_patches', None)
                and random.random() < 0.25):
            result = mutate_nv(
                result, 1.0,
                max_telomere=(MAX_TELOMERE if ga_config is None
                              else ga_config.max_telomere),
                chromosome_count=chromosome_count,
                evolve_delay=evolve_delay, evolve_io=False,
                io_placement=strategy, local_only=True)
        return result

    if escape.self_adaptive_mutation:
        # Immigrants have no lineage to inherit a rate from, so they start on a
        # randomised spread around the run rate; selection sorts out which end
        # of that spread this landscape rewards.
        from runtime.escape import seed_mutation_rate
        for genome in new_pop:
            if not hasattr(genome, '_mut_rate'):
                seed_mutation_rate(genome, adaptive_base, mutation_limit)

    # A child whose crossover is always followed by several random edits cannot
    # reveal whether the two parental modules were complementary.  Evaluate a
    # small crossover-only cohort first; it remains ordinary offspring, so the
    # next environmental selection can reject it just like any other child.
    # Keeping this bounded preserves the mutation-led exploratory majority.
    crossover_slots = min(
        pop - len(new_pop),
        max(1, int(round(pop * RECOMBINATION_EVALUATION_FRACTION))))
    while recombination and pop > 1 and crossover_slots > 0:
        pa, pb = parent_pair()
        ca, cb = crossover_nv(pa, pb, io_placement=strategy)
        if escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate
            inherit_mutation_rate(ca, pa, pb, escape, adaptive_base,
                                  mutation_limit)
            inherit_mutation_rate(cb, pb, pa, escape, adaptive_base,
                                  mutation_limit)
        new_pop.append(ca)
        crossover_slots -= 1
        if crossover_slots > 0 and len(new_pop) < pop:
            new_pop.append(cb)
            crossover_slots -= 1
    while len(new_pop) < pop:
        pa, pb = parent_pair()
        ca, cb = (crossover_nv(pa, pb, io_placement=strategy)
                  if recombination else
                  (clone_genome(pa), clone_genome(pb)))
        if escape.self_adaptive_mutation:
            from runtime.escape import inherit_mutation_rate
            inherit_mutation_rate(ca, pa, pb, escape, adaptive_base,
                                  mutation_limit)
            inherit_mutation_rate(cb, pb, pa, escape, adaptive_base,
                                  mutation_limit)
        new_pop.append(mutate_child(ca))
        if len(new_pop) < pop:
            new_pop.append(mutate_child(cb))
    if (chromosome_count is not None
            and any(len(genome.chromosomes) != chromosome_count
                    for genome in new_pop)):
        raise ValueError('genome factory violated configured chromosome count')
    if any(getattr(genome, 'arch', 'single') != pop_arch for genome in new_pop):
        raise ValueError('genome factory violated configured tile architecture')
    return new_pop[:pop]


# -- main loop (headless; the GUI runs its own equivalent in app.py) --------------

def _assimilate_timing_parents(population, fitnesses, target, count,
                               samples, seed, step):
    """Return a breeding list with locally learned delays made heritable.

    Evaluation already assigned each parent the score/case vector achieved by
    the deterministic local tuning session. Replaying that session here recovers
    its winning vector; the selected parent is cloned before write-back so the
    evaluated population and any cache entries are never mutated in place. With
    the tuner readout now fixed, the stored fitness and cases describe exactly
    the delay phenotype written into the clone.
    """
    from .temporal import score_temporal_plastic

    parents = list(population)
    order = sorted(
        range(len(parents)),
        key=lambda i: rank_key(parents[i], fitnesses[i]),
        reverse=True)
    changed = set()
    for i in order[:max(0, min(int(count), len(parents)))]:
        tuned = score_temporal_plastic(
            parents[i], target, samples=samples, seed=seed, step=step,
            return_settings=True)
        won = tuned[2].get('state_delays') if tuned else None
        if won is None:
            continue
        assimilated = clone_genome(parents[i])
        assimilated.state_delays = list(won)
        parents[i] = assimilated
        changed.add(i)
    return parents, changed


def evolve_nervous(target, generations=100, pop=POPSIZE, n_chroms=2, verbose=True,
                   seed=None, seed_genomes=None, selection=None,
                   return_population=False, tile_arch='tri3',
                   escape=None, ga_config=None):
    # Reproducibility: fitness evaluation is deterministic (grow + score, no RNG),
    # so seeding the main-process RNG that drives the genetic operators pins the
    # whole evolutionary trajectory. Pass a seed for any result meant to be re-run.
    if seed is not None:
        random.seed(seed)
    if not 1 <= n_chroms <= MAX_CHROMS:
        raise ValueError('n_chroms must be between 1 and %d' % MAX_CHROMS)
    if tile_arch not in ('single', 'tri3'):
        raise ValueError("tile_arch must be 'single' or 'tri3'")
    # This is also a fresh-run entry point, so apply the same two-profile
    # contract as the desktop controller. With no explicit target physics,
    # choose the current profile belonging to the requested architecture.
    import dataclasses as _dc
    from runtime.config import GAConfig, validate_new_nv_profile
    from runtime.escape import build_escape_state, OFF
    from .pulse import PulseConfig
    pulse_config = getattr(target, 'pulse_config', None)
    if pulse_config is None:
        pulse_config = PulseConfig(
            model=('paper_analog' if tile_arch == 'tri3' else 'pulse_delay'))
        setattr(target, 'pulse_config', pulse_config)
    validate_new_nv_profile(GAConfig(
        tile_arch=tile_arch, node_model=pulse_config.model))
    # Escape mechanisms. ``ga_config`` carries a whole run configuration;
    # ``escape`` overrides just the mechanism set. Neither is required, and with
    # both absent this driver behaves exactly as it did before the module
    # existed. The resolved config is attached to the target so it also reaches
    # the evaluation workers, exactly as the desktop controller does it.
    if ga_config is None:
        ga_config = GAConfig(tile_arch=tile_arch,
                             node_model=pulse_config.model,
                             chromosome_count=n_chroms)
    if escape is not None:
        ga_config = _dc.replace(ga_config, escape=escape)
    escape_cfg = ga_config.escape or OFF
    setattr(target, '_escape', escape_cfg)
    from .io_placement import (
        growth_seeds, io_strategy, seed_spatial, seed_wiring_from_phenotype,
        uses_port_chromosome)
    strategy = io_strategy(target)
    if (selection is None
            and (not getattr(target, 'temporal', False)
                 or getattr(target, 'combinational_cases', ())
                 or getattr(target, 'temporal_logic_cases', ()))):
        selection = 'lexicase'
    if uses_port_chromosome(strategy) and n_chroms < 3:
        raise ValueError('chromosome-based I/O requires n_chroms >= 3')
    evolve_io = strategy in (
        'terminal_nodes', 'tag_rank', 'wiring_chromosome',
        'spatial_chromosome')
    n_ports = target.n_inputs + len(target.outputs)
    def make_genome(input_genes=None):
        if strategy == 'fixed':
            # The live encoding: branched and output-rooted, exactly what the
            # desktop controller breeds. The two drive paths have drifted apart
            # before (see the two-GA-drive-paths note); this keeps them on one
            # encoding. The retired compatibility placements below still build
            # native genomes because they bind I/O through gene tags, which the
            # branched genome does not have.
            from .branched_ga import (random_branched_hex_genome,
                                       select_developmental_seed)
            return select_developmental_seed(
                lambda: random_branched_hex_genome(
                    n_chroms, max_telomere=ga_config.max_telomere,
                    n_inputs=target.n_inputs,
                    output_roles=tuple(terminal.role
                                       for terminal in target.outputs),
                    input_genes=input_genes),
                attempts=make_genome.developmental_seed_candidates)
        genome = random_hex_genome(
            n_chroms, arch=tile_arch,
            wiring_chromosome=(strategy == 'wiring_chromosome'),
            spatial_chromosome=(strategy == 'spatial_chromosome'),
            n_ports=n_ports, tag_rank=(strategy == 'tag_rank'),
            terminal_nodes=(strategy == 'terminal_nodes'),
            n_inputs=target.n_inputs, n_outputs=len(target.outputs),
            input_layout=True)
        if strategy == 'spatial_chromosome':
            seed_spatial(genome, None, target)
        elif uses_port_chromosome(strategy):
            from .nervous import grow_nervous
            grid = grow_nervous(
                genome, seeds=growth_seeds(target, strategy, genome),
                grid_size=target.grid_size, iters=target.iters)
            seed_wiring_from_phenotype(genome, grid, target)
        return genome
    if strategy == 'fixed':
        make_genome.developmental_seed_candidates = 6
    cache       = LRUCache(FITNESS_CACHE_MAX)
    ex          = ProcessPoolExecutor(max_workers=N_WORKERS)   # reuse one pool
    try:                                                       # (avoids per-gen respawn)
        # optional WARM START: begin from provided genomes (e.g. a circuit that
        # already has the easy behaviour) so evolution climbs toward the harder
        # objective instead of rediscovering the basics from scratch.
        seeds = list(seed_genomes or [])[:pop]
        if any(len(g.chromosomes) != n_chroms for g in seeds):
            raise ValueError('warm-start genomes must match n_chroms')
        if any(getattr(g, 'arch', 'single') != tile_arch for g in seeds):
            raise ValueError('warm-start genomes must match tile_arch')
        population = [clone_genome(g) for g in seeds]
        if not population and strategy == 'fixed':
            cohort_inputs = []
            cohort_count = min(4, max(1, pop // 8))
            for index in range(pop):
                if index < cohort_count:
                    genome = make_genome()
                    cohort_inputs.append(list(genome.inputs))
                else:
                    genome = make_genome(
                        cohort_inputs[(index - cohort_count) % cohort_count])
                population.append(genome)
            make_genome.developmental_seed_candidates = 2
        else:
            population += [make_genome() for _ in range(pop - len(population))]
            if strategy == 'fixed':
                make_genome.developmental_seed_candidates = 2
        fitnesses, cases = eval_batch_cases(population, target, cache, ex)
        escape_state = build_escape_state(
            'nervous', ga_config, chromosome_count=n_chroms,
            io_placement=strategy, evolve_io=evolve_io,
            evolve_delay=ga_config.timing_mutations())
        # Robustness scalars must exist before anything is ranked.
        escape_state.apply_robustness_blend(population, max(fitnesses))
        bi           = max(range(pop),
                           key=lambda i: rank_key(population[i], fitnesses[i]))
        best_genome  = clone_genome(population[bi])
        best_fitness = fitnesses[bi]
        best_rank    = rank_key(best_genome, best_fitness)
        escape_state.record_champion(0, best_genome, best_fitness)
        escape_state.note_contract_progress(cases, fitnesses)

        # The target's timing model enables its matching timing mutation.
        _pc = getattr(target, 'pulse_config', None)
        evolve_delay = timing_mutation_flags(
            getattr(_pc, 'model', 'uniform') if _pc is not None else 'uniform')
        # Lifetime tuning + genetic assimilation config (both opt-in via target
        # attrs; defaults make the loop byte-identical to before).
        _lifetime_samples = int(getattr(target, '_lifetime_samples', 0) or 0)
        _lifetime_seed = int(getattr(target, '_lifetime_seed', 20260727))
        _lifetime_step = float(
            getattr(target, '_lifetime_step', DELAY_LOG_STEP))
        _assimilate_n = (max(1, int(round(pop * 0.10)))
                         if (_lifetime_samples > 0
                             and getattr(target, '_lifetime_assimilate', False))
                         else 0)
        solved_at, stagnation = None, 0
        mut_rate = MEAN_MUTATIONS           # annealing schedule (see MUT_DECAY)
        spatial_rescue_queues = {}
        if verbose:
            print("%5s  %6s  %6s  %5s" % ("Gen", "Best", "Mean", "Mut"))
            print("-" * 30)
        # NO early stop at best == 1.0: the requested generation budget remains
        # comparable, while the all-time champion is archived separately.
        for gen in range(generations):
            mut_rate *= MUT_DECAY                                  # anneal: cool down
            mm = adaptive_mutation_rate(mut_rate, stagnation,
                                        solved=best_fitness >= 1.0)
            parents = list(population)
            parent_fitnesses, parent_cases = fitnesses, cases
            rescue = ()
            if (strategy == 'fixed'
                    and best_fitness < 1.0
                    and stagnation >= STRESS_PATIENCE):
                from .branched_ga import plateau_rescue_candidates
                rescue = plateau_rescue_candidates(
                    best_genome, target, limit=min(48, max(1, pop // 2)),
                    max_telomere=ga_config.max_telomere)
            elif (strategy == 'spatial_chromosome'
                    and best_fitness < 1.0
                    and stagnation >= STRESS_PATIENCE):
                from .io_placement import (
                    spatial_input_sites, spatial_output_variants,
                    spatial_routing_variants)
                from .nervous import grow_nervous
                limit = min(48, max(1, pop // 2))
                grid = grow_nervous(
                    best_genome,
                    seeds=growth_seeds(
                        target, 'spatial_chromosome', best_genome),
                    grid_size=target.grid_size, iters=target.iters)
                body_key = (
                    tuple(sorted(grid.items())),
                    tuple(spatial_input_sites(best_genome, target)))
                queues = spatial_rescue_queues.get(body_key)
                if queues is None:
                    spatial_rescue_queues.clear()
                    queues = {
                        'outputs': spatial_output_variants(
                            best_genome, target, limit=10_000),
                        'routing': spatial_routing_variants(
                            best_genome, target, limit=10_000),
                    }
                    spatial_rescue_queues[body_key] = queues
                output_queue = queues['outputs']
                output_count = min(
                    len(output_queue), max(1, limit // 3))
                rescue = output_queue[:output_count]
                del output_queue[:output_count]
                routing_queue = queues['routing']
                routing_count = min(
                    len(routing_queue), max(0, limit - len(rescue)))
                rescue += routing_queue[:routing_count]
                del routing_queue[:routing_count]
            # Memetic/Lamarckian write-back: replay the deterministic local
            # timing session for top breeders and copy accepted fine adjustments
            # into cloned parents before reproduction. Off by default.
            if _assimilate_n and strategy == 'fixed':
                parents, assimilated = _assimilate_timing_parents(
                    parents, parent_fitnesses, target, _assimilate_n,
                    _lifetime_samples, _lifetime_seed, _lifetime_step)
                if assimilated:
                    pi = max(
                        range(pop),
                        key=lambda i: rank_key(parents[i], parent_fitnesses[i]))
                    parent_rank = rank_key(
                        parents[pi], parent_fitnesses[pi])
                    # Preserve an assimilated version of an equal champion. Its
                    # score is unchanged, but its accepted timing is now present
                    # in the genome returned from the run and used for rescue.
                    if pi in assimilated and parent_rank >= best_rank:
                        best_rank = parent_rank
                        best_fitness = parent_fitnesses[pi]
                        best_genome = clone_genome(parents[pi])
            # One pool, or separate demes at their own mutation rates when
            # islands are on. Shared with the desktop controller.
            offspring = escape_state.breed(
                gen, parents, parent_fitnesses, parent_cases, mm,
                lambda deme, deme_fitnesses, deme_cases, deme_rate:
                    next_population(
                        deme, deme_fitnesses, make_genome, deme_cases,
                        deme_rate,
                        selection=selection, chromosome_count=n_chroms,
                        evolve_delay=evolve_delay, evolve_io=evolve_io,
                        io_placement=strategy, archive_parent=best_genome,
                        stagnation=stagnation, rescue_candidates=rescue,
                        escape=escape_cfg,
                        mutation_limit=ga_config.mutation_limit))
            offspring_fitnesses, offspring_cases = eval_batch_cases(
                offspring, target, cache, ex)
            # Collapse robust case vectors under the current anneal before
            # anything is ranked (see runtime/escape.py).
            escape_state.apply_robustness_blend(
                list(parents) + list(offspring),
                max(best_fitness, max(offspring_fitnesses)))
            # Survivor selection, shared with the desktop controller: terminal
            # consolidation once solved, otherwise optional crowding plus the
            # baseline rotating contract-elite reserve.
            population, fitnesses, cases = escape_state.merge_generation(
                parents, parent_fitnesses, parent_cases,
                offspring, offspring_fitnesses, offspring_cases,
                consolidate=consolidate_population,
                solved=max(best_fitness, max(offspring_fitnesses)) >= 1.0)
            gi = max(range(pop),
                     key=lambda i: rank_key(population[i], fitnesses[i]))
            gen_rank = rank_key(population[gi], fitnesses[gi])
            # Topology-only wins do not reset stress. Honest scalar progress or
            # improvement in one organism's weakest declared cases does.
            case_progress = escape_state.note_contract_progress(
                cases, fitnesses)
            if (fitnesses[gi] > best_fitness + 1e-12
                    or case_progress):
                stagnation = 0
            else:
                stagnation += 1
            if escape_state.accepts(gen_rank, best_rank):
                best_rank    = gen_rank
                best_fitness = fitnesses[gi]
                best_genome  = clone_genome(population[gi])
            escape_state.record_champion(gen, best_genome, best_fitness)
            population, fitnesses, cases, rebirth_info = \
                escape_state.maybe_rebirth(
                    gen, population, fitnesses, cases, mm, stagnation,
                    best_fitness,
                    lambda genomes: eval_batch_cases(
                        genomes, target, cache, ex))
            if rebirth_info is not None:
                # A rebirth answers the stall that triggered it; leaving the
                # counter set would re-fire it on the very next generation.
                stagnation = 0
                escape_state.note_contract_progress(cases, fitnesses)
                if verbose:
                    print('Rebirth at generation %d: %d genomes from ancestors '
                          '%s at rate %.2f'
                          % (gen, rebirth_info['reborn'],
                             rebirth_info['ancestors'], rebirth_info['rate']))
            escape_state.tick()
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


# -- diversification: a whole generation of DISTINCT valid solutions --------------

def diversify(seeds, target, pop_size, valid=0.999, rounds=25, batch=None,
              cache=None, executor=None, should_stop=None, on_progress=None,
              max_telomere=MAX_TELOMERE, chromosome_count=None,
              evolve_delay=None, evolve_io=False, io_placement=None):
    """Fill a population with evaluated, genetically distinct valid offspring.

    Distinctness is based on the rule alleles crossover can exchange, not tags,
    split metadata, or a neutral telomere edit. Once two valid rule programs are
    available they breed through the normal crossover + mutation operators;
    mutation alone bootstraps the pool when only one solver exists.

    Returns the list of unique valid genomes found (up to pop_size). Where the
    target has a broad neutral network it fills the population; where solutions
    are isolated spikes it returns however few exist - an honest ceiling, not a
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
    model = getattr(getattr(target, 'pulse_config', None), 'model', 'uniform')
    evolve_delay = timing_mutation_flags(model, evolve_delay)
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
                    parent_a, pool[random.choice(mates)],
                    io_placement=io_placement))
            else:
                base = parent_a
            c = mutate_nv(base, max_telomere=max_telomere,
                          chromosome_count=chromosome_count,
                          evolve_delay=evolve_delay, evolve_io=evolve_io,
                          io_placement=io_placement)
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
