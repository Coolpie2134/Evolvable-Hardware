"""
substrates/nervous/nervous.py — hexagonal nervous-network growth, interpretation, scoring.

The array is a honeycomb (each node has 3 neighbours: L, R, D — see hexgrid.py).
A grown cell's 5-bit state is decoded (ROUTING_HEX) into a routing config
(e1, e2, i1, op): which of the 3 directions feed the two excitatory and the one
inhibitory input of a nervous-net node, and how the two excitatory inputs combine
(op = 'and' for states 0-15, the paper's Fig. 3; 'or' for the 16-31 OR twins).
Output:

        out = (val(e1) op val(e2)) AND NOT val(i1)        # coincidence/OR + veto

Inputs are held at the seed cells and the array is relaxed; outputs are read at
the target's terminals. No LIF — pure pulse logic, and the degree-3 topology is
what forms the excitatory / inhibitory loops.
"""
from __future__ import annotations

from .hexgrid import (hex_dirs, hex_frontier_cells, ROUTING_HEX, routing_kind,
                      IO_STATE_INPUT, IO_STATE_OUTPUT)

# Grown-state -> terminal kind (1 = input, 2 = output), matching io_placement's
# IO_KIND_INPUT / IO_KIND_OUTPUT. A cell's I/O identity is now its GROWN STATE
# (16 = input, 17 = output), not a separate gene tag, so a gene's self_out can
# grow a dedicated terminal. Consulted only when terminal tracking is on (the
# terminal_nodes strategy); every other path ignores it.
_STATE_IO_KIND = {IO_STATE_INPUT: 1, IO_STATE_OUTPUT: 2}
from .genome import germline_telomere, MAX_STATE, TRI_STATE_MAX
from .tritile import TRI_SEED_STATE

ROUTING    = ROUTING_HEX       # back-compat name (used by the GUI colourer)
SEED_STATE = 1


def _seed_state(genome):
    """The seed/germline state for this genome's tile architecture."""
    return (TRI_SEED_STATE if getattr(genome, 'arch', 'single') == 'tri3'
            else SEED_STATE)

# The four context fields occupy disjoint bit ranges, so their summed Hamming
# distance is exactly the popcount of one packed XOR. Compile gene contexts once
# per growth run: one XOR + bit_count replaces four fieldwise popcounts per gene
# while preserving chromosome-major, first-wins lookup order.
#
# The widths are DERIVED from the state alphabets, never hard-coded: the packed
# value is also the growth cache key, so a field narrower than its alphabet
# would silently truncate and alias two distinct contexts onto one cache entry.
# Single-tile is 5 bits — 4 would lose the OR twins: states 0-15 are the
# paper's AND routing and 16-31 their OR counterparts, so the growth match must
# see the 5th bit to tell them apart. Tri-tile is 15 bits (three packed 5-bit
# channels, each the same 32-value alphabet); because the channels are disjoint
# fields, a 15-bit Hamming is exactly the sum of the three per-channel Hammings.
_SINGLE_BITS = (MAX_STATE - 1).bit_length()        # 5
_TRI_BITS = (TRI_STATE_MAX - 1).bit_length()       # 15


def _pack_context(sL, sR, sD, si, bits):
    mask = (1 << bits) - 1
    return ((sL & mask) | ((sR & mask) << bits)
            | ((sD & mask) << (2 * bits))
            | ((si & mask) << (3 * bits)))


def _compile_lookup(genome, bits=None):
    """Return packed gene contexts in the original lookup/tie-break order."""
    if bits is None:
        bits = (_TRI_BITS if getattr(genome, 'arch', 'single') == 'tri3'
                else _SINGLE_BITS)
    entries = []
    for chrom in genome.chromosomes:
        if getattr(chrom, 'wiring', False):
            continue
        for gene in chrom.genes:
            context = _pack_context(gene.ctx_l, gene.ctx_r, gene.ctx_d,
                                    gene.self_in, bits)
            entries.append((context, gene))
    return bits, tuple(entries)


# ── hex growth (native hex genome: self_out == 0 means the cell dies) ──────────
# Context reads the 3 hex neighbours (L/R/D) + self, matched against each gene's
# ctx_l/ctx_r/ctx_d/self_in by minimum Hamming distance. Directions are the
# node's own orientation-relative L/R/D (see hex_dirs).
#
# The field is UNBOUNDED; the genome bounds its own size BIOLOGICALLY, with a
# telomere acting as a Hayflick division limit (see genome.py):
#   * per-cell telomere — a cell divides (births a live frontier cell) only if a
#     live neighbour still has telomere > 0; the daughter inherits parent - 1.
#     A cell at telomere 0 is senescent: alive and functional, but it cannot
#     divide, so growth provably halts at radius L (the germline length) from the
#     seeds. This alone bounds SIZE (was grid_size) and DURATION (was iters);
#   * the empty-cell guard — even where a lineage still has telomere to spend, an
#     empty cell only comes alive via a GROWTH rule (self_in == 0; sim6
#     table_lookup: "if self is zero and not an exact match, return zero").
# Maintenance rules (self_in != 0) act on LIVE cells every step regardless of
# telomere — telomeres limit a cell's REPLICATION, not its function, exactly as
# in biology.

def _lookup_compiled(program, sL, sR, sD, si, packed=None,
                     return_kind=False):
    """Associative next-state lookup (min Hamming). No time/telomere term — the
    telomere now gates DIVISION per cell in _grow_step, not which genes exist.

    The single-tile alphabet is 5-bit; the tri-tile alphabet (three packed 5-bit
    channels) is 15-bit. Both are compared by Hamming distance over the whole
    state — and because the tri channels occupy DISJOINT bit fields, a 15-bit
    Hamming is exactly the sum of the three per-channel Hammings, so context
    matching is per-channel for free."""
    if sL == 0 and sR == 0 and sD == 0 and si == 0:
        return (0, 0) if return_kind else 0
    bits, entries = program
    context = (_pack_context(sL, sR, sD, si, bits)
               if packed is None else packed)
    best_gene, best_dist = None, 1 << 30
    for gene_context, gene in entries:
        distance = (gene_context ^ context).bit_count()
        if distance < best_dist:             # strict: first gene wins every tie
            best_dist, best_gene = distance, gene
    if best_gene is None:
        return (0, 0) if return_kind else 0
    if si == 0 and best_gene.self_in != 0:         # empty cells grow only via
        return (0, 0) if return_kind else 0         # growth rules (sim6 guard)
    state = best_gene.self_out                     # 0 = off / death (native)
    if return_kind:
        return state, int(getattr(best_gene, 'io_kind', 0))
    return state


def _lookup_nv(genome, sL, sR, sD, si):
    """Back-compatible one-shot lookup; growth compiles once and reuses it."""
    return _lookup_compiled(_compile_lookup(genome), sL, sR, sD, si)


def _next_state(program, sL, sR, sD, si, cache, return_kind=False):
    """Cached packed lookup keyed on context alone (telomere does not affect the
    lookup, so the cache is valid for the whole run — sim6 table_lookup_cached)."""
    key = _pack_context(sL, sR, sD, si, program[0])
    ns = cache.get(key)
    if ns is None:
        ns = _lookup_compiled(
            program, sL, sR, sD, si, packed=key,
            return_kind=return_kind)
        cache[key] = ns
    return ns


def _grow_step(program, grid, tel, seeds, L, cache, seed_state=SEED_STATE,
               terminal_kinds=None):
    """One development step. `grid` = {pos: state}, `tel` = {pos: remaining
    telomere}. Returns (next_grid, next_tel).

      * surviving cells: every live cell runs its maintenance lookup and keeps
        its telomere (function, not replication);
      * division: an empty frontier cell is born only if some live neighbour
        still has telomere > 0 (the Hayflick gate) AND a growth rule fires; the
        daughter inherits (max live-neighbour telomere) - 1;
      * seeds: germline / stem cells, always present at full telomere L."""
    nxt, nxt_tel = {}, {}
    nxt_kinds = {} if terminal_kinds is not None else None
    # maintenance / survival of the existing organism
    for (x, y), state in grid.items():
        nb = hex_dirs(x, y)
        result = _next_state(
            program, grid.get(nb['L'], 0), grid.get(nb['R'], 0),
            grid.get(nb['D'], 0), state, cache,
            return_kind=(nxt_kinds is not None))
        ns = result[0] if nxt_kinds is not None else result
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = tel.get((x, y), 0)
            if nxt_kinds is not None:
                io = _STATE_IO_KIND.get(ns & 0x1F)
                if io:
                    nxt_kinds[(x, y)] = io
    # division into empty frontier cells (Hayflick-gated)
    frontier = set()
    for (x, y) in grid:
        for cell in hex_frontier_cells(x, y):
            if cell not in grid:
                frontier.add(cell)
    for (x, y) in frontier:
        nb = hex_dirs(x, y)
        parent_tel = max((tel.get(nb[d], 0) for d in ('L', 'R', 'D')
                          if nb[d] in grid), default=0)
        if parent_tel <= 0:                       # every neighbour senescent
            continue                              # -> no division (Hayflick)
        result = _next_state(
            program, grid.get(nb['L'], 0), grid.get(nb['R'], 0),
            grid.get(nb['D'], 0), 0, cache,
            return_kind=(nxt_kinds is not None))
        ns = result[0] if nxt_kinds is not None else result
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = parent_tel - 1
            if nxt_kinds is not None:
                io = _STATE_IO_KIND.get(ns & 0x1F)
                if io:
                    nxt_kinds[(x, y)] = io
    for pos in seeds:
        nxt[pos] = seed_state
        nxt_tel[pos] = L
        if nxt_kinds is not None:
            nxt_kinds.pop(pos, None)
    if nxt_kinds is not None:
        return nxt, nxt_tel, nxt_kinds
    return nxt, nxt_tel


# Growth is bounded SOLELY by the telomere — there is no external iteration cap
# and no grid-size clip. Spatially the organism can never pass radius L from the
# seeds (the telomere runs out), so cell addition provably stops after ~L steps.
# What remains is the paper's state settling — "the cycles of table lookups may
# continue but the growth stops due to the fact that all queries return the same
# value out as was passed in" (§6-7) — which ends at a fixed point or 2-cycle,
# detected for early exit. The iteration budget is derived ENTIRELY from L: a
# state change propagates one cell per step, so settling the whole organism
# (diameter ~2L) after the ~L growth steps needs O(L) more — hence 3L + margin.
# Nothing outside the genome's own telomere governs how far or how long it grows.

def _grow_budget(L):
    return 3 * L + 6


def apply_routing_patches(genome, grid):
    """Apply heritable mature-cell routing edits without changing development."""
    patches = getattr(genome, 'routing_patches', None) or ()
    if not patches:
        return grid
    maximum = (
        TRI_STATE_MAX if getattr(genome, 'arch', 'single') == 'tri3'
        else MAX_STATE)
    result = dict(grid)
    for patch in patches:
        pos = (int(patch.x), int(patch.y))
        state = int(patch.state)
        # Patches override existing routing only. Inert/out-of-range alleles
        # remain heritable and may later mutate back onto the living body.
        if pos in result and 0 < state < maximum:
            result[pos] = state
    return result


# `grid_size` and `iters` are accepted but IGNORED — vestigial, kept only so
# existing callers and pickles keep working. Growth is governed entirely by the
# genome's telomere (Hayflick limit); pass nothing and it still self-limits.

def grow_nervous(genome, seeds, grid_size=None, iters=None):
    """Grow on the unbounded field until the attractor (fixed point or 2-cycle).
    Size and duration are bounded by the genome's telomere ALONE — `grid_size`
    and `iters` are ignored (see note above)."""
    L = germline_telomere(genome)
    seed_state = _seed_state(genome)
    program = _compile_lookup(genome)
    # Terminal identity is now the GROWN STATE (16 = input, 17 = output), so track
    # terminals whenever a gene can express one — not the retired io_kind tag.
    track_terminals = any(
        (int(gene.self_out) & 0x1F) in (IO_STATE_INPUT, IO_STATE_OUTPUT)
        for _, gene in program[1])
    grid = {pos: seed_state for pos in seeds}
    tel  = {pos: L for pos in seeds}
    prev, cache = None, {}
    for _ in range(_grow_budget(L)):
        result = _grow_step(
            program, grid, tel, seeds, L, cache, seed_state,
            terminal_kinds=({} if track_terminals else None))
        if track_terminals:
            nxt, nxt_tel, nxt_kinds = result
        else:
            nxt, nxt_tel = result
            nxt_kinds = {}
        if nxt == grid or nxt == prev:      # fixed point (mature) or 2-cycle
            genome._terminal_kinds = nxt_kinds
            return apply_routing_patches(genome, nxt)
        prev, grid, tel = grid, nxt, nxt_tel
    genome._terminal_kinds = nxt_kinds
    return apply_routing_patches(genome, grid)


def grow_nervous_snapshots(genome, seeds, grid_size=None, iters=None):
    L = germline_telomere(genome)
    seed_state = _seed_state(genome)
    program = _compile_lookup(genome)
    # Terminal identity is now the GROWN STATE (16 = input, 17 = output), so track
    # terminals whenever a gene can express one — not the retired io_kind tag.
    track_terminals = any(
        (int(gene.self_out) & 0x1F) in (IO_STATE_INPUT, IO_STATE_OUTPUT)
        for _, gene in program[1])
    snaps = [{pos: seed_state for pos in seeds}]
    grid = dict(snaps[0])
    tel  = {pos: L for pos in seeds}
    prev, cache = None, {}
    for _ in range(_grow_budget(L)):
        result = _grow_step(
            program, grid, tel, seeds, L, cache, seed_state,
            terminal_kinds=({} if track_terminals else None))
        if track_terminals:
            nxt, nxt_tel, nxt_kinds = result
        else:
            nxt, nxt_tel = result
            nxt_kinds = {}
        snaps.append(dict(nxt))
        if nxt == grid or nxt == prev:
            genome._terminal_kinds = nxt_kinds
            snaps[-1] = apply_routing_patches(genome, snaps[-1])
            break
        prev, grid, tel = grid, nxt, nxt_tel
    else:
        genome._terminal_kinds = nxt_kinds
        snaps[-1] = apply_routing_patches(genome, snaps[-1])
    return snaps


# ── interpret / evaluate ────────────────────────────────────────────────────────

def _place_outputs(grid, target):
    """Assign each output role a live cell: nearest free non-input cell to its
    terminal ("terminals"), or nearest-to-mid in the rightmost columns (legacy
    "heuristic"). Candidates scan in sorted-cell order so ties are stable.
    Returns {role: (x,y) | None}."""
    input_set = set(target.inputs)
    non_input = [p for p in sorted(grid) if p not in input_set]
    out_pos   = {term.role: None for term in target.outputs}
    if getattr(target, 'output_strategy', 'terminals') == 'terminals':
        used = set()
        for term in target.outputs:
            tx, ty = term.pos
            cands  = [p for p in non_input if p not in used]
            if not cands:
                break
            best = min(cands, key=lambda p: abs(p[0] - tx) + abs(p[1] - ty))
            used.add(best)
            out_pos[term.role] = best
    else:
        mid_y  = target.grid_size // 2
        x_cols = sorted({x for (x, _) in non_input}, reverse=True)
        for i, term in enumerate(target.outputs):
            if i >= len(x_cols):
                break
            col = [p for p in non_input if p[0] == x_cols[i]]
            out_pos[term.role] = min(col, key=lambda p: abs(p[1] - mid_y))
    return out_pos


def node_delays(genome, grid, config=None):
    """Per-cell delays for evolved-delay width-preserving transport, or None.

    Delay is indexed by the same 5-bit routing state used by the retired
    single-circuit timing profile. The helper owns the model gate so other paths cannot
    accidentally consume a dormant delay vector from a checkpoint.
    """
    if config is None or getattr(config, 'model', 'uniform') != 'pulse_delay':
        return None
    mult = getattr(genome, 'state_delays', None)
    if not mult:
        return None
    base = config.delay
    n = len(mult)
    return {pos: base * mult[state & 0x1F]
            for pos, state in grid.items()
            if (state & 0x1F) < n}


def interpret_nervous(grid, target=None, arch='single'):
    """Return (routing {pos:(e1,e2,i1,op)}, input_pos, output_pos {role:(x,y)|None})."""
    if arch == 'single':
        routing = {pos: ROUTING_HEX[state & 0x1F]
                   for pos, state in grid.items()}
    elif arch == 'tri3':
        # TriSim expands each tile into three directional circuit nodes itself.
        # Returning no legacy routing prevents UIs/analyses from drawing a
        # fictitious single broadcast arrow decoded from the low five bits.
        routing = {}
    else:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    if target is not None:
        input_pos  = list(target.inputs)
        output_pos = _place_outputs(grid, target)
    else:
        input_pos, output_pos = [], {}
    return routing, input_pos, output_pos


def evaluate_nervous(grid, routing, input_vals, grid_size, steps=None,
                     config=None, arch='single', terminal_inputs=None,
                     terminal_outputs=None):
    """Evaluate one combinational case on the asynchronous pulse engine.
    Input levels are held for the whole horizon (one long pulse on each driven
    input net — a single edge). Returns {pos: 0/1} where 1 means the cell's
    wire PULSED at some point — the natural read-out of a pulse-based array
    (an AND gate's output is a pulse, not a held level)."""
    from .simulation import create_simulator
    if steps is None:
        steps = 2 * grid_size + 4
    if arch == 'tri3':
        from .tritile import TriSim
        sim = TriSim(grid, input_vals.keys(), config=config,
                     outputs=terminal_outputs)
    elif arch == 'single':
        sim = create_simulator(
            grid, routing, config=config, input_nodes=terminal_inputs,
            output_nodes=terminal_outputs)
    else:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    held = {c: int(b) for c, b in input_vals.items()}
    for _ in range(steps):
        sim.step(held)
    return dict(sim.ever)


def _resolve_io_binding(genome, grid, target, in_pos, out_pos):
    """Swap the geometric port binding for the genome's evolvable one when the
    target opts into an io_placement strategy (substrates/nervous/io_placement.py). Returns
    (in_pos, out_pos) — under a strategy each in_pos entry is a LIST of
    attachment cells (an input may fan out; ports may share sites) — or None
    when the strategy is active but the organism cannot bind every port.
    'fixed' passes the geometric binding through."""
    from .io_placement import io_strategy, bind_io
    if io_strategy(target) == 'fixed':
        return in_pos, out_pos
    return bind_io(genome, grid, target)


def _input_levels(in_pos, in_bits):
    """{cell: 0/1} drive map for one combinational case. Inputs with several
    attachment cells drive them all; a cell shared by several inputs is the
    wired-OR of their bits."""
    from .io_placement import input_groups
    levels = {}
    for index, cells in enumerate(input_groups(in_pos)):
        bit = int(in_bits[index]) if index < len(in_bits) else 0
        for cell in cells:
            levels[cell] = levels.get(cell, 0) | bit
    return levels


def score_nervous_full(genome, target, *, _developed=None):
    """Return behavioral fitness and the exact truth-table case vector.

    Static Nervous evaluation used to discard ``score_contract``'s second
    return value.  That forced selection to collapse every row/output result
    into one average even though the evaluator had already measured the
    individual checks.  Keep those checks so combinational evolution can use
    case-aware selection without another growth or simulation pass.
    """
    from .scoring import contract_case_count, score_contract
    from .io_placement import (growth_seeds, io_strategy, binding_progress,
                               record_binding_progress)
    failed = (0.0, (0.0,) * contract_case_count(target))
    if _developed is None:
        strategy = io_strategy(target)
        grid = grow_nervous(genome, seeds=growth_seeds(
                                target, strategy, genome),
                            grid_size=target.grid_size, iters=target.iters)
    else:
        grid, strategy = _developed
    if strategy in (
            'terminal_nodes', 'wiring_chromosome', 'spatial_chromosome'):
        record_binding_progress(
            genome, binding_progress(genome, grid, target))
    if len(grid) <= target.n_inputs:
        return failed
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, out_pos = interpret_nervous(grid, target, arch=arch)
    resolved = _resolve_io_binding(genome, grid, target, in_pos, out_pos)
    if resolved is None:
        return failed
    in_pos, out_pos = resolved
    if any(out_pos[t.role] is None for t in target.outputs):
        return failed
    from .io_placement import (flat_inputs, output_groups,
                               terminal_node_sets)
    live = set(grid)
    if any(p not in live for p in flat_inputs(in_pos)):
        return failed
    n_checks = len(target.cases) * len(target.outputs)
    if n_checks == 0:
        return failed
    observations = []
    terminal_inputs, terminal_outputs = terminal_node_sets(
        target, in_pos, out_pos)
    for in_bits, out_bits in target.cases:
        invals = _input_levels(in_pos, in_bits)
        outs   = evaluate_nervous(
            grid, routing, invals, target.grid_size,
            config=getattr(target, 'pulse_config', None), arch=arch,
            terminal_inputs=terminal_inputs,
            terminal_outputs=terminal_outputs)
        groups = output_groups(out_pos)
        observations.append([
            float(any(outs.get(cell, 0) for cell in groups[term.role]))
            for term in target.outputs])
    score, cases, _ = score_contract(observations, target)
    return score, cases


def score_nervous(genome, target, *, _developed=None):
    """Scalar compatibility wrapper around :func:`score_nervous_full`."""
    return score_nervous_full(
        genome, target, _developed=_developed)[0]


def nervous_case_outputs(genome, target):
    from .io_placement import growth_seeds, io_strategy
    grid = grow_nervous(genome, seeds=growth_seeds(
                            target, io_strategy(target), genome),
                        grid_size=target.grid_size, iters=target.iters)
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, out_pos = interpret_nervous(grid, target, arch=arch)
    cases = []
    resolved = _resolve_io_binding(genome, grid, target, in_pos, out_pos)
    if resolved is None:
        return grid, in_pos, {t.role: None for t in target.outputs}, cases
    in_pos, out_pos = resolved
    if len(grid) <= target.n_inputs or any(out_pos[t.role] is None for t in target.outputs):
        return grid, in_pos, out_pos, cases
    from .io_placement import terminal_node_sets
    terminal_inputs, terminal_outputs = terminal_node_sets(
        target, in_pos, out_pos)
    for in_bits, out_bits in target.cases:
        invals = _input_levels(in_pos, in_bits)
        outs   = evaluate_nervous(
            grid, routing, invals, target.grid_size,
            config=getattr(target, 'pulse_config', None), arch=arch,
            terminal_inputs=terminal_inputs,
            terminal_outputs=terminal_outputs)
        from .io_placement import output_groups
        groups = output_groups(out_pos)
        acts = {t.role: int(any(outs.get(cell, 0)
                               for cell in groups[t.role]))
                for t in target.outputs}
        cases.append({'in_bits': in_bits, 'out_bits': out_bits,
                      'node_outputs': outs, 'acts': acts})
    return grid, in_pos, out_pos, cases


def circuit_summary_nervous(grid, arch='single'):
    kinds = {'off': 0, 'buffer': 0, 'coincidence': 0, 'or': 0, 'inhibited': 0}
    if arch == 'tri3':
        from .tritile import channel_configs
        for state in grid.values():
            for config in channel_configs(state):
                kinds[routing_kind(ROUTING_HEX[config])] += 1
        return ('%d tiles / %d directional circuits  '
                '(%d buffer, %d coincidence, %d inhibited, %d off)'
                % (len(grid), 3 * len(grid), kinds['buffer'],
                   kinds['coincidence'], kinds['inhibited'], kinds['off']))
    if arch != 'single':
        raise ValueError('unknown tile architecture: %r' % (arch,))
    for state in grid.values():
        kinds[routing_kind(ROUTING_HEX[state & 0x1F])] += 1
    return ('%d nodes  (%d buffer, %d coincidence, %d OR, %d inhibited, %d off)'
            % (len(grid), kinds['buffer'], kinds['coincidence'], kinds['or'],
               kinds['inhibited'], kinds['off']))


def nervous_truth_table(genome, target):
    grid, in_pos, out_pos, cases = nervous_case_outputs(genome, target)
    arch = getattr(genome, 'arch', 'single')
    from .contracts import behavior_contract_lines
    lines = (['Target: ' + target.name + '   [hex nervous net]', ''] +
             behavior_contract_lines(target) +
             ['', 'Circuit: ' + circuit_summary_nervous(grid, arch=arch)])
    from .io_placement import output_groups
    groups = output_groups(out_pos)
    for term in target.outputs:
        cells = groups.get(term.role, [])
        lines.append("  out '%s': %s" % (
            term.role,
            ('wired-OR at %s' % ', '.join(map(str, cells)))
            if cells else '(not found)'))
    if not cases:
        lines += ['', '(circuit incomplete — inputs/outputs missing)']
        return '\n'.join(lines)
    in_hdr  = ' '.join('i%d' % i for i in range(len(target.inputs)))
    out_hdr = ' '.join('%s:e/a' % t.role for t in target.outputs)
    lines += ['', '  %s | %s | result' % (in_hdr, out_hdr),
              '  ' + '-' * (len(in_hdr) + len(out_hdr) + 14)]
    correct = total = 0
    for case in cases:
        cells, row_ok = [], True
        for i, term in enumerate(target.outputs):
            act = case['acts'][term.role]; exp = case['out_bits'][i]
            ok = act == exp; row_ok = row_ok and ok
            total += 1; correct += 1 if ok else 0
            cells.append('%d/%d' % (exp, act))
        in_str  = ' '.join(str(b) for b in case['in_bits']).ljust(len(in_hdr))
        out_str = ' '.join(c.ljust(len('%s:e/a' % t.role))
                           for c, t in zip(cells, target.outputs))
        lines.append('  %s | %s | %s' % (in_str, out_str, 'PASS' if row_ok else 'FAIL'))
    fit = correct / total if total else 0.0
    lines += ['', '  => %d/%d checks  (fitness = %.4f)%s'
              % (correct, total, fit, '   ALL PASS' if correct == total else '')]
    return '\n'.join(lines)
