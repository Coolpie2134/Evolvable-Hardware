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

def run_nervous(grid, routing, in_pos, out_pos, streams, T):
    """
    Run T ticks of the asynchronous pulse simulation. streams[t] = tuple of
    input bits (one per in_pos); a 0->1 transition injects a pulse edge onto
    that input cell's net (held 1s are one long pulse). out_pos is
    {role: (x,y)}. Returns (states, traces):
        states : list[T] of full {(x,y):0/1} sampled activity maps (for the GUI)
        traces : {role: [0/1 over T]}
    """
    sim    = PulseSim(grid, routing)
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
    output scores 0.5; a correctly ringing/quiet memory scores 1.0."""
    per_level = {0: [], 1: []}
    for lvl, ticks in _expected_windows(exp):
        ticks = [t for t in ticks if t < len(trace)]
        if ticks:
            per_level[lvl].append(_window_score(trace, lvl, ticks))
    parts = [sum(v) / len(v) for v in per_level.values() if v]
    return sum(parts) / len(parts) if parts else 0.0


def place_outputs_by_trace(grid, routing, in_pos, ttarget):
    """Assign each output role the live non-input cell whose activity trace
    best matches the role's expected trace across ALL trials (ties broken by
    distance to the role's terminal, then cell order). The mechanism decides
    where the answer lives; the terminal only anchors ties — so evolution has
    to build the computation, not also route it to one prescribed cell.

    Returns (out_pos {role: (x,y)|None}, traces {role: [trace per trial]}).
    """
    in_set = set(in_pos)
    cands  = [c for c in sorted(grid) if c not in in_set]
    out_pos = {term.role: None for term in ttarget.outputs}
    traces  = {}
    if not cands:
        return out_pos, traces
    # one dynamics run per trial; every cell's trace falls out of the states
    trial_states = [run_nervous(grid, routing, in_pos, {}, tr.streams, ttarget.T)[0]
                    for tr in ttarget.trials]
    used = set()
    for term in ttarget.outputs:
        best, best_key = None, None
        for c in cands:
            if c in used:
                continue
            scores = []
            for ti, trial in enumerate(ttarget.trials):
                exp = trial.expected.get(term.role)
                if exp is None:
                    continue
                tr = [trial_states[ti][t].get(c, 0) for t in range(ttarget.T)]
                scores.append(_role_trace_score(tr, exp))
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


def windowed_score(traces, ttarget):
    """Selection fitness core: mean windowed, level-balanced score over every
    (trial, role). 1.0 iff every store-1 window rings throughout and every
    store-0 window is silent (the phase-tolerant notion of a correct memory)."""
    vals = []
    for ti, trial in enumerate(ttarget.trials):
        for role, exp in trial.expected.items():
            tr = traces.get(role, [])
            if ti < len(tr):
                vals.append(_role_trace_score(tr[ti], exp))
    return sum(vals) / len(vals) if vals else 0.0


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
    """Behavioural fitness in [0,1]: the phase-tolerant windowed score at the
    trace-matched output cells (no loop shaping — see nv_evo/ga.py). This is the
    ground-truth 'does it compute the goal' metric; 1.0 == solved."""
    prep = prepare_net(genome, ttarget)
    if prep is None:
        return 0.0
    _, _, _, _, traces = prep
    return windowed_score(traces, ttarget)


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
