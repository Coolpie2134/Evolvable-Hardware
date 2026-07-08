"""
nv_evo/temporal.py — temporal scoring and loop analysis over the pulse engine.

The nervous net is a temporal system: a pulse injected by an input circulates
around a loop of buffers (a stored bit = a circulating pulse) until inhibition
stops it. The dynamics themselves are the asynchronous edge-triggered pulse
simulation in nv_evo/pulse.py (PulseSim); this module drives it tick-by-tick
for scoring and gives the interactive "pulse playback" its engine.

Core (shared by interactive playback and temporal scoring):
    run_nervous(grid, routing, in_pos, out_pos, streams, T) -> (states, traces)
    score_temporal(genome, ttarget)                 -> behavioural fitness [0,1]
    loop_profile(grid, routing, in_pos, out_pos)    -> feedback-loop stats

Loop analysis feeds the GA's fitness shaping (nv_evo/ga.py): memory needs a
directed cycle in the signal graph, and a cycle only matters if the inputs can
write into it and it can drive an output.

The target types (Trial, TemporalTarget, presets) live in nv_evo/targets.py;
they are re-exported here for back-compat with older imports and pickles.
"""
from __future__ import annotations

from .nervous import grow_nervous, interpret_nervous
from .hexgrid import hex_dirs
from .pulse import PulseSim
from .targets import (OutputTerminal, Trial, TemporalTarget, TEMPORAL_TARGETS,
                      sr_latch, toggle_ff, oscillator, echo)


# ── dynamics (asynchronous pulse engine, sampled per tick) ──────────────────────
# The actual dynamics are event-driven edge-triggered pulses (see pulse.py);
# scoring and playback observe them by sampling every wire once per tick.

def input_cone(grid, routing, in_pos):
    """The cells whose value can be influenced by an input — everything forward-
    reachable from the input seeds in the signal graph. On the nervous net there
    is no spontaneous activity, so every cell OUTSIDE this cone is silent for all
    time; simulating only the cone is exact and, on the unbounded field where
    growth may leave a large off-cone blob, far cheaper."""
    edges = signal_graph(grid, routing)          # u -> cells that read u
    seen  = set(p for p in in_pos if p in grid)
    stack = list(seen)
    while stack:
        for v in edges.get(stack.pop(), ()):
            if v not in seen:
                seen.add(v); stack.append(v)
    return seen


def run_nervous(grid, routing, in_pos, out_pos, streams, T, prune=True):
    """
    Run T ticks of the asynchronous pulse simulation. streams[t] = tuple of
    input bits (one per in_pos); a 0->1 transition injects a pulse edge onto
    that input cell's net (held 1s are one long pulse). out_pos is
    {role: (x,y)}. Returns (states, traces):
        states : list[T] of sampled {(x,y):0/1} activity maps
        traces : {role: [0/1 over T]}
    `prune` restricts the simulation to the input cone (exact for the nervous
    net); off-cone cells are absent from `states` and read as 0 by callers.
    """
    sub = grid
    if prune:
        cone = input_cone(grid, routing, in_pos)
        if len(cone) < len(grid):
            sub = {c: grid[c] for c in cone}
    sim    = PulseSim(sub, routing)
    states = []
    traces = {role: [] for role in out_pos}
    for t in range(T):
        state = sim.step({in_pos[i]: streams[t][i] for i in range(len(in_pos))})
        states.append(state)
        for role, p in out_pos.items():
            traces[role].append(state.get(p, 0) if p else 0)
    return states, traces


# ── temporal scoring ───────────────────────────────────────────────────────────
# The expected trace decomposes into WINDOWS: maximal runs of a constant
# expected level (settle gaps of None separate them). Scoring is per-window,
# then balanced across the two levels, so every behavioural phase (off before
# set, hold after set, off after reset, ...) carries equal weight — a long hold
# can't drown out a missed reset, and a constant output caps at 0.5.
#
# A "store 1" window is scored PHASE-TOLERANTLY: nervous-net memory holds a bit
# as a pulse *circulating* in a loop, which reads as a ripple (e.g. 1010…) at
# any single cell — the honeycomb has no triangles, so you can't OR the phases
# back into a steady DC level. What matters is that the cell is *actively
# ringing* with no long silent gap, so a store-1 window scores by activity
# coverage (each tick counts if it or an immediate neighbour fires): a ripple
# scores 1.0, a single blip scores low, silence scores 0. A "store 0" window
# still demands true silence (exact), so a net that keeps ringing after reset
# is penalised. Isolated single-tick expectations (echo, oscillator) are
# length-1 windows, for which coverage reduces to exact per-tick matching.

def _expected_windows(exp):
    """[(level, [ticks])] — maximal constant-level runs of scored ticks."""
    wins, cur, lvl = [], [], None
    for t, e in enumerate(exp):
        if e is None:
            if cur:
                wins.append((lvl, cur))
            cur, lvl = [], None
        elif lvl is None or e == lvl:
            lvl = e
            cur.append(t)
        else:
            wins.append((lvl, cur))
            lvl, cur = e, [t]
    if cur:
        wins.append((lvl, cur))
    return wins


def _window_score(trace, lvl, ticks):
    """Score one constant-level window. lvl==0: fraction of silent ticks.
    lvl==1: activity coverage — fraction of ticks that fire or sit next to a
    firing tick OF THE SAME WINDOW (rewards sustained ringing, not a lone blip;
    echoes in the unscored settle gaps outside the window don't count)."""
    if lvl == 0:
        return sum(1 for t in ticks if not trace[t]) / len(ticks)
    tick_set = set(ticks)
    covered = sum(1 for t in ticks
                  if trace[t]
                  or (t - 1 in tick_set and trace[t - 1])
                  or (t + 1 in tick_set and trace[t + 1]))
    return covered / len(ticks)


def _role_trace_score(trace, exp):
    """Windowed, level-balanced score of one trace against one expected trace:
    mean per-window score within each level, averaged across levels. A constant
    output scores 0.5; a correctly ringing/quiet memory scores 1.0.
    (Retained for diagnostics; selection uses the precision/recall score below.)"""
    per_level = {0: [], 1: []}
    for lvl, ticks in _expected_windows(exp):
        ticks = [t for t in ticks if t < len(trace)]
        if ticks:
            per_level[lvl].append(_window_score(trace, lvl, ticks))
    parts = [sum(v) / len(v) for v in per_level.values() if v]
    return sum(parts) / len(parts) if parts else 0.0


# ── precision / recall scoring (the selection metric) ────────────────────────────
# Rather than scoring the huge, easy mass of expected-0 ticks (which lets a
# do-nothing output look good on rare-positive targets), score ONLY the highs:
#   recall    = expected-1 ticks that the output actually hits — punishes silence
#   precision = of the output's own pulses, the fraction that belong — punishes
#               always-high / spurious firing
# combined as their harmonic mean (F1). Two asymmetries make it fit this
# substrate: recall allows ±1 COVERAGE (memory is a circulating pulse that rings
# every other tick, so an expected-1 counts as hit if the cell fires on it or an
# adjacent tick), while precision is EXACT (a pulse landing on an expected-0 tick
# is a false positive) — without that, a dense oscillator target would let an
# always-high output through. Unscored (settle) ticks count for neither.

def _pr_counts(trace, exp, tol=1):
    n = min(len(exp), len(trace))
    fires = set(t for t in range(n) if trace[t])
    exp_high = [t for t in range(len(exp)) if exp[t] == 1]
    tp_rec = sum(1 for t in exp_high
                 if any((t + d) in fires for d in range(-tol, tol + 1)))
    scored_fires = [t for t in fires if exp[t] is not None]
    tp_prec = sum(1 for t in scored_fires if exp[t] == 1)
    return tp_rec, len(exp_high), tp_prec, len(scored_fires)


def _f1(tp_rec, n_exp, tp_prec, n_act):
    rec  = tp_rec / n_exp  if n_exp else 1.0     # nothing to recall -> vacuously 1
    prec = tp_prec / n_act if n_act else 1.0     # fired nothing    -> vacuously 1
    return (2 * rec * prec / (rec + prec)) if (rec + prec) else 0.0


def _pr_score(trace, exp):
    """Per-trace F1 of the highs (see _pr_counts). Silent -> 0 when anything is
    expected; always-high -> low; a correct ringing/quiet output -> 1."""
    return _f1(*_pr_counts(trace, exp))


# Selection metric. The fitness question is "did the network produce the correct
# SPIKE EVENTS?", not "was the output level right at every tick?" — so the metric
# is the precision/recall (F1) of the output spikes above:
#   recall    — expected spikes the output actually produced (missing ones cost)
#   precision — of the spikes it produced, the fraction that were expected
#               (extra, unexpected spikes cost)
# A do-nothing output recalls nothing, so it scores 0 the moment any spike is
# expected — silence is no longer rewarded. A fire-constantly output has terrible
# precision. Only the correct spikes at the correct ticks (and nowhere else) reach
# 1.0. The other two metrics are kept for diagnostics / experiments only:
#   'balanced' — mean of pooled expected-0 / expected-1 window scores (this is the
#                one that rewarded silence: a constant output scores 0.5)
#   'blend'    — average of f1 and balanced (the former default; diluted f1 with
#                balanced's partial credit for doing nothing)
METRIC = 'f1'


def _trace_metric(trace, exp, metric=None):
    m = metric or METRIC
    if m == 'f1':
        return _pr_score(trace, exp)
    if m == 'balanced':
        return _role_trace_score(trace, exp)
    return 0.5 * (_role_trace_score(trace, exp) + _pr_score(trace, exp))


# The output is read at a cell NEAR the target's terminal, not anywhere in the
# organism. On the unbounded field a net can be hundreds of cells, and scoring
# every one both costs O(cells) per genome and lets a lucky far-off cell inflate
# the fitness without a real mechanism (the output location then jitters between
# genomes, wrecking the gradient). Restricting to the OUT_RADIUS-nearest cells
# to the terminal keeps placement cheap and the fitness signal about a fixed,
# local output — the terminal is a designated read-out point, after all.
OUT_RADIUS = 12


def _output_candidates(grid, in_set, term):
    tx, ty = term.pos
    cands = [c for c in grid if c not in in_set]
    cands.sort(key=lambda c: (abs(c[0] - tx) + abs(c[1] - ty), c))
    return cands[:OUT_RADIUS]


def place_outputs_by_trace(grid, routing, in_pos, ttarget):
    """Assign each output role the live non-input cell — among those nearest its
    terminal (see OUT_RADIUS) — whose activity trace best matches the expected
    trace across ALL trials (ties broken by distance to the terminal, then cell
    order). Evolution builds the computation and lands it near the read-out.

    Returns (out_pos {role: (x,y)|None}, traces {role: [trace per trial]}).
    """
    in_set = set(in_pos)
    out_pos = {term.role: None for term in ttarget.outputs}
    traces  = {}
    if len(grid) <= len(in_set):
        return out_pos, traces
    # one dynamics run per trial; every cell's trace falls out of the states
    trial_states = [run_nervous(grid, routing, in_pos, {}, tr.streams, ttarget.T)[0]
                    for tr in ttarget.trials]
    used = set()
    for term in ttarget.outputs:
        best, best_key = None, None
        for c in _output_candidates(grid, in_set, term):
            if c in used:
                continue
            scores = []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                tr = [trial_states[ti][t].get(c, 0) for t in range(ttarget.T)]
                scores.append(_trace_metric(tr, exp))
            s   = sum(scores) / len(scores) if scores else 0.0
            key = (-s, abs(c[0] - term.pos[0]) + abs(c[1] - term.pos[1]), c)
            if best_key is None or key < best_key:
                best_key, best = key, c
        if best is None:
            break
        used.add(best)
        out_pos[term.role] = best
        traces[term.role]  = [[trial_states[ti][t].get(best, 0)
                               for t in range(ttarget.T)]
                              for ti in range(len(ttarget.trials))]
    return out_pos, traces


def prepare_net(genome, ttarget):
    """Grow + interpret a genome for a temporal target, placing outputs by
    trace match. Returns (grid, routing, in_pos, out_pos, traces) — where
    traces[role][i] is the chosen cell's trace in trial i — or None if the net
    is unusable (too small, no candidate output cells, or an input seed dead)."""
    grid = grow_nervous(genome, seeds=tuple(ttarget.inputs),
                        grid_size=ttarget.grid_size, iters=ttarget.iters)
    if len(grid) <= ttarget.n_inputs:
        return None
    routing, in_pos, _ = interpret_nervous(grid, ttarget)
    if any(p not in grid for p in in_pos):
        return None
    out_pos, traces = place_outputs_by_trace(grid, routing, in_pos, ttarget)
    if any(out_pos[t.role] is None for t in ttarget.outputs):
        return None
    return grid, routing, in_pos, out_pos, traces


def windowed_score(traces, ttarget, metric=None):
    """Selection fitness core, pooled globally over every (trial, role) under
    METRIC (or an explicit `metric`):
      f1       — spike-event precision/recall (default; silent = 0, always-high low)
      balanced — pooled expected-0 vs expected-1 window scores, averaged (diagnostic)
      blend    — mean of the two (diagnostic)
    1.0 iff every expected spike is produced (ringing allowed) and nothing fires
    where it shouldn't."""
    m = metric or METRIC
    tp_rec = n_exp = tp_prec = n_act = 0
    per_level = {0: [], 1: []}
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            if ti >= len(tr):
                continue
            trace = tr[ti]
            if m in ('f1', 'blend'):
                a, b, c, d = _pr_counts(trace, exp)
                tp_rec += a; n_exp += b; tp_prec += c; n_act += d
            if m in ('balanced', 'blend'):
                for lvl, ticks in _expected_windows(exp):
                    ticks = [t for t in ticks if t < len(trace)]
                    if ticks:
                        per_level[lvl].append(_window_score(trace, lvl, ticks))
    f1 = _f1(tp_rec, n_exp, tp_prec, n_act)
    parts = [sum(v) / len(v) for v in per_level.values() if v]
    bal = sum(parts) / len(parts) if parts else 0.0
    if m == 'f1':
        return f1
    if m == 'balanced':
        return bal
    return 0.5 * (bal + f1)


def exact_tick_accuracy(traces, ttarget):
    """Plain fraction of scored ticks matched exactly (no phase tolerance).
    Diagnostic only — a working circulating-pulse latch ripples, so this reads
    below 1.0 even when the memory is behaviourally correct."""
    correct = total = 0
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            tr_i = tr[ti] if ti < len(tr) else []
            for t in range(min(ttarget.T, len(exp), len(tr_i))):
                if exp[t] is None:
                    continue
                total += 1
                if tr_i[t] == exp[t]:
                    correct += 1
    return correct / total if total else 0.0


def score_temporal(genome, ttarget):
    """Behavioural fitness in [0,1]: the spike-event F1 (see METRIC) at the
    trace-matched output cells (no loop shaping — see nv_evo/ga.py). This is the
    ground-truth 'does it produce the right spikes' metric; 1.0 == solved."""
    prep = prepare_net(genome, ttarget)
    if prep is None:
        return 0.0
    _, _, _, _, traces = prep
    return windowed_score(traces, ttarget)


def temporal_report(ttarget, genome=None):
    """Human-readable explanation of a temporal target for the GUI: what the
    circuit must do, the trial bank (input pulses vs expected trace), and —
    when a genome is given — the evolved net's actual traces with per-trial
    scores. '.' marks unscored ticks (latency / settle windows)."""
    lines = ['Target: %s   [temporal nervous net]' % ttarget.name]
    desc = getattr(ttarget, 'description', '')
    if desc:
        lines += [''] + desc.splitlines()
    lines += ['',
              '%d input%s, output %s read wherever the behaviour appears '
              '(trace-matched),' % (ttarget.n_inputs,
                                    '' if ttarget.n_inputs == 1 else 's',
                                    '/'.join(t.role for t in ttarget.outputs)),
              "%d ticks per trial, %d trials. '.' = unscored tick."
              % (ttarget.T, len(ttarget.trials)),
              '',
              'WHY a 1111 hold is satisfied by 1010: a node is refractory for',
              'DELAY+WIDTH after each firing, so NO wire can stay high — max duty',
              'is 50%. A stored 1 is a pulse circulating a loop, read as ringing',
              '(1010...) at any one cell. The F1 metric below (the SAME one the GA',
              'optimises) therefore counts an expected 1 as hit if the cell fires',
              'on it or an adjacent tick (±1 ring tolerance), while every pulse',
              'landing on an expected-0 tick costs precision exactly:',
              '    highs hit  — expected 1s the output reaches (misses cost)',
              '    pulses ok  — its own pulses that belong (extras cost)',
              '(The "exact per-tick" number at the bottom ignores ring tolerance;',
              'a perfect circulating-pulse latch reads ~0.75 there by physics.)']

    prep = traces = None
    if genome is not None:
        prep = prepare_net(genome, ttarget)
        if prep is None:
            lines += ['', '(circuit incomplete — grew too little or inputs dead)']
        else:
            _, _, _, out_pos, traces = prep
            for t in ttarget.outputs:
                lines.append("out '%s' read at %s" % (t.role, out_pos[t.role]))

    names = [chr(65 + i) for i in range(ttarget.n_inputs)]
    for ti, trial in enumerate(ttarget.trials):
        pulses = {n: [t for t in range(len(trial.streams)) if trial.streams[t][i]]
                  for i, n in enumerate(names)}
        lines += ['', 'Trial %d:  %s' % (ti + 1, '   '.join(
            '%s pulses@%s' % (n, p if p else '(none)') for n, p in pulses.items()))]
        for role, exp in trial.expected.items():
            lines.append('  expect %s %s' % (
                ''.join('.' if e is None else str(e) for e in exp),
                role if len(trial.expected) > 1 else ''))
            if traces is not None:
                tr = traces.get(role, [])
                tr_i = tr[ti] if ti < len(tr) else []
                # show the SELECTION metric (F1) with its components, not the old
                # 'balanced' diagnostic — that one under-counted spurious pulses
                # and made failing trials read deceptively high.
                tp_rec, n_exp, tp_prec, n_act = _pr_counts(tr_i, exp)
                s = _f1(tp_rec, n_exp, tp_prec, n_act)
                lines.append('  actual %s (F1 %.3f %s  highs hit %d/%d, pulses ok %d/%d)'
                             % (''.join(str(v) for v in tr_i), s,
                                'PASS' if s >= 0.999 else 'FAIL',
                                tp_rec, n_exp, tp_prec, n_act))
    if traces is not None:
        total = windowed_score(traces, ttarget)
        lines += ['', '=> behavioural score %.4f%s   (exact per-tick %.4f)'
                  % (total, '   SOLVED' if total >= 0.999 else '',
                     exact_tick_accuracy(traces, ttarget))]
    else:
        lines += ['', '(run the GA or Load Saved to see the evolved traces here;',
                  ' drive it live in the Interactive tab with Step / Run)']
    return '\n'.join(lines)


# ── feedback-loop analysis (for the GA's loop-aware shaping) ────────────────────

def signal_graph(grid, routing):
    """Directed signal graph as {node: set(readers)}: u -> v iff node v's
    routing reads neighbour u (excitatory or inhibitory)."""
    edges = {c: set() for c in grid}
    for v, (e1, e2, i1) in routing.items():
        nb = hex_dirs(*v)
        for d in (e1, e2, i1):
            if d is None:
                continue
            u = nb[d]
            if u in edges:
                edges[u].add(v)
    return edges


def _reachable(edges, sources):
    seen, stack = set(), [s for s in sources if s in edges]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(edges[u])
    return seen


def cycle_nodes(edges):
    """Nodes that lie on a directed cycle (can reach themselves). Grids are
    tiny (<= ~50 nodes), so per-node reachability is plenty fast."""
    on_cycle = set()
    for s in edges:
        seen, stack = set(), list(edges[s])
        while stack:
            u = stack.pop()
            if u == s:
                on_cycle.add(s)
                break
            if u in seen or u not in edges:
                continue
            seen.add(u)
            stack.extend(edges[u])
    return on_cycle


def loop_profile(grid, routing, in_pos, out_pos):
    """Feedback-loop stats of a grown net:
        n_cycle    — nodes on any directed cycle
        n_relevant — cycle nodes both writable from an input and driving an output
    A latch is exactly a relevant cycle: inputs can set/clear the circulating
    value and the output can read it."""
    edges = signal_graph(grid, routing)
    cyc   = cycle_nodes(edges)
    if not cyc:
        return {'n_cycle': 0, 'n_relevant': 0}
    from_in = _reachable(edges, in_pos)
    rev = {c: set() for c in edges}
    for u, vs in edges.items():
        for v in vs:
            rev[v].add(u)
    to_out = _reachable(rev, [p for p in out_pos.values() if p])
    return {'n_cycle': len(cyc), 'n_relevant': len(cyc & from_in & to_out)}
