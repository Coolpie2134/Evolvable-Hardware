"""
nv_evo/nervous.py — hexagonal nervous-network growth, interpretation, scoring.

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

from .hexgrid import hex_dirs, hex_frontier_cells, ROUTING_HEX, routing_kind
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
# see the 5th bit to tell them apart. Tri-tile is 12 bits (three packed 4-bit
# channels); because the channels are disjoint fields, a 12-bit Hamming is
# exactly the sum of the three per-channel Hammings.
_SINGLE_BITS = (MAX_STATE - 1).bit_length()        # 5
_TRI_BITS = (TRI_STATE_MAX - 1).bit_length()       # 12


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

def _lookup_compiled(program, sL, sR, sD, si, packed=None):
    """Associative next-state lookup (min Hamming). No time/telomere term — the
    telomere now gates DIVISION per cell in _grow_step, not which genes exist.

    The single-tile alphabet is 5-bit; the tri-tile alphabet (three packed 4-bit
    channels) is 12-bit. Both are compared by Hamming distance over the whole
    state — and because the tri channels occupy DISJOINT bit fields, a 12-bit
    Hamming is exactly the sum of the three per-channel Hammings, so context
    matching is per-channel for free."""
    if sL == 0 and sR == 0 and sD == 0 and si == 0:
        return 0
    bits, entries = program
    context = (_pack_context(sL, sR, sD, si, bits)
               if packed is None else packed)
    best_gene, best_dist = None, 1 << 30
    for gene_context, gene in entries:
        distance = (gene_context ^ context).bit_count()
        if distance < best_dist:             # strict: first gene wins every tie
            best_dist, best_gene = distance, gene
    if best_gene is None:
        return 0
    if si == 0 and best_gene.self_in != 0:         # empty cells grow only via
        return 0                                   # growth rules (sim6 guard)
    return best_gene.self_out                      # 0 = off / death (native)


def _lookup_nv(genome, sL, sR, sD, si):
    """Back-compatible one-shot lookup; growth compiles once and reuses it."""
    return _lookup_compiled(_compile_lookup(genome), sL, sR, sD, si)


def _lookup_nv_tri(genome, sL, sR, sD, si):
    """Back-compatible explicit tri3 lookup (normally dispatched by _lookup_nv)."""
    return _lookup_compiled(_compile_lookup(genome, bits=_TRI_BITS),
                            sL, sR, sD, si)


def _next_state(program, sL, sR, sD, si, cache):
    """Cached packed lookup keyed on context alone (telomere does not affect the
    lookup, so the cache is valid for the whole run — sim6 table_lookup_cached)."""
    key = _pack_context(sL, sR, sD, si, program[0])
    ns = cache.get(key)
    if ns is None:
        ns = _lookup_compiled(program, sL, sR, sD, si, packed=key)
        cache[key] = ns
    return ns


def _grow_step(program, grid, tel, seeds, L, cache, seed_state=SEED_STATE):
    """One development step. `grid` = {pos: state}, `tel` = {pos: remaining
    telomere}. Returns (next_grid, next_tel).

      * surviving cells: every live cell runs its maintenance lookup and keeps
        its telomere (function, not replication);
      * division: an empty frontier cell is born only if some live neighbour
        still has telomere > 0 (the Hayflick gate) AND a growth rule fires; the
        daughter inherits (max live-neighbour telomere) - 1;
      * seeds: germline / stem cells, always present at full telomere L."""
    nxt, nxt_tel = {}, {}
    # maintenance / survival of the existing organism
    for (x, y), state in grid.items():
        nb = hex_dirs(x, y)
        ns = _next_state(program, grid.get(nb['L'], 0), grid.get(nb['R'], 0),
                         grid.get(nb['D'], 0), state, cache)
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = tel.get((x, y), 0)
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
        ns = _next_state(program, grid.get(nb['L'], 0), grid.get(nb['R'], 0),
                         grid.get(nb['D'], 0), 0, cache)
        if ns:
            nxt[(x, y)] = ns
            nxt_tel[(x, y)] = parent_tel - 1
    for pos in seeds:
        nxt[pos] = seed_state
        nxt_tel[pos] = L
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
    grid = {pos: seed_state for pos in seeds}
    tel  = {pos: L for pos in seeds}
    prev, cache = None, {}
    for _ in range(_grow_budget(L)):
        nxt, nxt_tel = _grow_step(program, grid, tel, seeds, L, cache, seed_state)
        if nxt == grid or nxt == prev:      # fixed point (mature) or 2-cycle
            return nxt
        prev, grid, tel = grid, nxt, nxt_tel
    return grid


def grow_nervous_snapshots(genome, seeds, grid_size=None, iters=None):
    L = germline_telomere(genome)
    seed_state = _seed_state(genome)
    program = _compile_lookup(genome)
    snaps = [{pos: seed_state for pos in seeds}]
    grid = dict(snaps[0])
    tel  = {pos: L for pos in seeds}
    prev, cache = None, {}
    for _ in range(_grow_budget(L)):
        nxt, nxt_tel = _grow_step(program, grid, tel, seeds, L, cache, seed_state)
        snaps.append(dict(nxt))
        if nxt == grid or nxt == prev:
            break
        prev, grid, tel = grid, nxt, nxt_tel
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

    Delay is indexed by the same 5-bit routing state as width evolution. The
    helper owns the model gate so uniform/evolved-width paths cannot
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
                     config=None, arch='single'):
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
        sim = TriSim(grid, input_vals.keys(), config=config)
    elif arch == 'single':
        sim = create_simulator(grid, routing, config=config)
    else:
        raise ValueError('unknown tile architecture: %r' % (arch,))
    held = {c: int(b) for c, b in input_vals.items()}
    for _ in range(steps):
        sim.step(held)
    return dict(sim.ever)


def score_nervous(genome, target):
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, out_pos = interpret_nervous(grid, target, arch=arch)
    if any(out_pos[t.role] is None for t in target.outputs):
        return 0.0
    live = set(grid)
    if any(p not in live for p in in_pos):
        return 0.0
    n_checks = len(target.cases) * len(target.outputs)
    if n_checks == 0:
        return 0.0
    correct = 0
    for in_bits, out_bits in target.cases:
        invals = {in_pos[i]: in_bits[i] for i in range(len(in_pos))}
        outs   = evaluate_nervous(
            grid, routing, invals, target.grid_size,
            config=getattr(target, 'pulse_config', None), arch=arch)
        for i, term in enumerate(target.outputs):
            if outs.get(out_pos[term.role], 0) == out_bits[i]:
                correct += 1
    return correct / n_checks


def nervous_case_outputs(genome, target):
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    arch = getattr(genome, 'arch', 'single')
    routing, in_pos, out_pos = interpret_nervous(grid, target, arch=arch)
    cases = []
    if len(grid) <= target.n_inputs or any(out_pos[t.role] is None for t in target.outputs):
        return grid, in_pos, out_pos, cases
    for in_bits, out_bits in target.cases:
        invals = {in_pos[i]: in_bits[i] for i in range(len(in_pos))}
        outs   = evaluate_nervous(
            grid, routing, invals, target.grid_size,
            config=getattr(target, 'pulse_config', None), arch=arch)
        acts   = {t.role: outs.get(out_pos[t.role], 0) for t in target.outputs}
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
    lines = ['Target: ' + target.name + '   [hex nervous net]',
             'Circuit: ' + circuit_summary_nervous(grid, arch=arch)]
    for term in target.outputs:
        p = out_pos.get(term.role)
        lines.append("  out '%s': %s" % (term.role, ('pos=%s' % (p,)) if p else '(not found)'))
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
