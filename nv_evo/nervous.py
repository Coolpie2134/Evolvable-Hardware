"""
nv_evo/nervous.py — hexagonal nervous-network growth, interpretation, scoring.

The array is a honeycomb (each node has 3 neighbours: L, R, D — see hexgrid.py).
A grown cell's 4-bit state is decoded (ROUTING_HEX, the paper's Fig. 3) into a
routing config (e1, e2, i1): which of the 3 directions feed the two excitatory
and the one inhibitory input of a nervous-net node. Output:

        out = (val(e1) AND val(e2)) AND NOT val(i1)       # coincidence + veto

Inputs are held at the seed cells and the array is relaxed; outputs are read at
the target's terminals. No LIF — pure pulse logic, and the degree-3 topology is
what forms the excitatory / inhibitory loops.
"""
from __future__ import annotations

from .hexgrid import hex_dirs, hex_frontier_cells, ROUTING_HEX, routing_kind

ROUTING    = ROUTING_HEX       # back-compat name (used by the GUI colourer)
SEED_STATE = 1

# 4-bit popcount for the Hamming-distance gene lookup.
_PC4 = [bin(i).count("1") for i in range(16)]


def _h4(a, b):
    return _PC4[(a ^ b) & 0xF]


# ── hex growth (native hex genome: self_out == 0 means the cell dies) ──────────
# Context reads the 3 hex neighbours (L/R/D) + self, matched against each gene's
# ctx_l/ctx_r/ctx_d/self_in by minimum Hamming distance — exactly the paper's
# associative-memory lookup (all genes apply at every iteration; there is no
# per-gene time gate). Directions are the node's own orientation-relative L/R/D
# (see hex_dirs), so a single gene can express symmetric structure.

def _lookup_nv(genome, sL, sR, sD, si):
    if sL == 0 and sR == 0 and sD == 0 and si == 0:
        return 0
    best_out, best_dist = 0, 1 << 30
    for chrom in genome.chromosomes:
        for gene in chrom.genes:
            d = (_h4(gene.ctx_l, sL) + _h4(gene.ctx_r, sR) +
                 _h4(gene.ctx_d, sD) + _h4(gene.self_in, si))
            if d < best_dist:
                best_dist, best_out = d, gene.self_out
    return best_out          # 0 = off / death (native)


def _grow_step(genome, grid, seeds, grid_size):
    frontier = {}
    for (x, y) in list(grid):
        for (nx, ny) in hex_frontier_cells(x, y, grid_size):
            if (nx, ny) not in grid and (nx, ny) not in frontier:
                frontier[(nx, ny)] = 0
    working = {**grid, **frontier}
    nxt = {}
    for (x, y), state in working.items():
        nb = hex_dirs(x, y)
        ns = _lookup_nv(genome, working.get(nb['L'], 0), working.get(nb['R'], 0),
                        working.get(nb['D'], 0), state)
        if ns:
            nxt[(x, y)] = ns
    for pos in seeds:
        nxt[pos] = SEED_STATE
    return nxt


def grow_nervous(genome, seeds, grid_size, iters):
    grid = {pos: SEED_STATE for pos in seeds}
    for _ in range(iters):
        grid = _grow_step(genome, grid, seeds, grid_size)
    return grid


def grow_nervous_snapshots(genome, seeds, grid_size, iters):
    snaps = [{pos: SEED_STATE for pos in seeds}]
    grid = dict(snaps[0])
    for _ in range(iters):
        grid = _grow_step(genome, grid, seeds, grid_size)
        snaps.append(dict(grid))
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


def interpret_nervous(grid, target=None):
    """Return (routing {pos:(e1,e2,i1)}, input_pos, output_pos {role:(x,y)|None})."""
    routing = {pos: ROUTING_HEX[state & 0xF] for pos, state in grid.items()}
    if target is not None:
        input_pos  = list(target.inputs)
        output_pos = _place_outputs(grid, target)
    else:
        input_pos, output_pos = [], {}
    return routing, input_pos, output_pos


def evaluate_nervous(grid, routing, input_vals, grid_size, steps=None):
    """Evaluate one combinational case on the asynchronous pulse engine.
    Input levels are held for the whole horizon (one long pulse on each driven
    input net — a single edge). Returns {pos: 0/1} where 1 means the cell's
    wire PULSED at some point — the natural read-out of a pulse-based array
    (an AND gate's output is a pulse, not a held level)."""
    from .pulse import PulseSim
    if steps is None:
        steps = 2 * grid_size + 4
    sim = PulseSim(grid, routing)
    held = {c: int(b) for c, b in input_vals.items()}
    for _ in range(steps):
        sim.step(held)
    return dict(sim.ever)


def score_nervous(genome, target):
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    if len(grid) <= target.n_inputs:
        return 0.0
    routing, in_pos, out_pos = interpret_nervous(grid, target)
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
        outs   = evaluate_nervous(grid, routing, invals, target.grid_size)
        for i, term in enumerate(target.outputs):
            if outs.get(out_pos[term.role], 0) == out_bits[i]:
                correct += 1
    return correct / n_checks


def nervous_case_outputs(genome, target):
    grid = grow_nervous(genome, seeds=tuple(target.inputs),
                        grid_size=target.grid_size, iters=target.iters)
    routing, in_pos, out_pos = interpret_nervous(grid, target)
    cases = []
    if len(grid) <= target.n_inputs or any(out_pos[t.role] is None for t in target.outputs):
        return grid, in_pos, out_pos, cases
    for in_bits, out_bits in target.cases:
        invals = {in_pos[i]: in_bits[i] for i in range(len(in_pos))}
        outs   = evaluate_nervous(grid, routing, invals, target.grid_size)
        acts   = {t.role: outs.get(out_pos[t.role], 0) for t in target.outputs}
        cases.append({'in_bits': in_bits, 'out_bits': out_bits,
                      'node_outputs': outs, 'acts': acts})
    return grid, in_pos, out_pos, cases


def circuit_summary_nervous(grid):
    kinds = {'off': 0, 'buffer': 0, 'coincidence': 0, 'inhibited': 0}
    for state in grid.values():
        kinds[routing_kind(ROUTING_HEX[state & 0xF])] += 1
    return ('%d nodes  (%d buffer, %d coincidence, %d inhibited, %d off)'
            % (len(grid), kinds['buffer'], kinds['coincidence'],
               kinds['inhibited'], kinds['off']))


def nervous_truth_table(genome, target):
    grid, in_pos, out_pos, cases = nervous_case_outputs(genome, target)
    lines = ['Target: ' + target.name + '   [hex nervous net]',
             'Circuit: ' + circuit_summary_nervous(grid)]
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
