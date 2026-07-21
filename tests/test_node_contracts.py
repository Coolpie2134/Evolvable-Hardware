"""
tests/test_node_contracts.py — primitive conformance tests for the node
contracts (NODE_CONTRACTS.md).

Each node-timing model must obey its OWN stated physics before any cross-model
comparison means anything. tests/test_pulse_models.py pins the basics and the
V3 aggregation/determinism suite; this file fills the remaining contract
clauses: coincidence window sweeps, refractory arithmetic, the four inhibitor
timing cases, loop regimes, buffer-chain width drift, OR union case tables,
the coincidence width-selection rule, translation invariance for the variant
models, and the complete-interval (`pulse_intervals`) exposure.

Run under pytest, or standalone:  py tests/test_node_contracts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo.pulse import PulseSim, PulseConfig                # noqa: E402
from nv_evo.hexgrid import hex_dirs                           # noqa: E402

TOL = 1e-9
INF = float('inf')


# ── fixtures ─────────────────────────────────────────────────────────────────────

def _back(cell, prev):
    return [d for d in ('L', 'R', 'D') if hex_dirs(*cell)[d] == prev][0]


def _buffer_pair():
    a = (0, 0)
    b = hex_dirs(*a)['R']
    grid = {a: 1, b: 1}
    routing = {a: (None, None, None, 'and'),
               b: (_back(b, a), _back(b, a), None, 'and')}
    return grid, routing, a, b


def _buffer_chain(n_buffers):
    """Input cell + n_buffers, each buffering its predecessor (a lattice walk)."""
    cells = [(0, 0)]
    while len(cells) < n_buffers + 1:
        cur = cells[-1]
        nxt = next(hex_dirs(*cur)[d] for d in ('R', 'D', 'L')
                   if hex_dirs(*cur)[d] not in cells)
        cells.append(nxt)
    assert len(set(cells)) == n_buffers + 1, "chain walk revisited a cell"
    grid = {c: 1 for c in cells}
    routing = {cells[0]: (None, None, None, 'and')}
    for prev, cur in zip(cells, cells[1:]):
        d = _back(cur, prev)
        routing[cur] = (d, d, None, 'and')
    return grid, routing, cells


def _two_input(op='and'):
    v = (0, 0)
    nb = hex_dirs(*v)
    a, b = nb['L'], nb['R']
    grid = {a: 1, b: 1, v: 1}
    routing = {a: (None, None, None, 'and'),
               b: (None, None, None, 'and'),
               v: ('L', 'R', None, op)}
    return grid, routing, a, b, v


def _inhibited_buffer():
    v = (0, 0)
    nb = hex_dirs(*v)
    src, inh = nb['L'], nb['D']
    grid = {src: 1, inh: 1, v: 1}
    routing = {src: (None, None, None, 'and'),
               inh: (None, None, None, 'and'),
               v: ('L', 'L', 'D', 'and')}
    return grid, routing, src, inh, v


def _loop_pair():
    u = (0, 0)
    v = hex_dirs(*u)['R']
    grid = {u: 1, v: 1}
    routing = {u: (_back(u, v), _back(u, v), None, 'and'),
               v: (_back(v, u), _back(v, u), None, 'and')}
    return grid, routing, u, v


def _closed(intervals):
    """[(start, end)] with any still-open aggregate dropped."""
    return [(s, e) for s, e in intervals if e != INF]


UNIFORM = PulseConfig()
WP = PulseConfig(model='pulse_delay')          # width-preserving transport


# ── uniform: window, refractory, loop ────────────────────────────────────────────

def test_uniform_coincidence_window_sweep():
    """Edge separation == coincidence triggers (inclusive); beyond it does not.
    Overlap is irrelevant: the rule is the leading-edge window."""
    grid, routing, a, b, v = _two_input('and')
    for offset, fires in ((0.49, True), (0.5, True), (0.51, False)):
        sim = PulseSim(grid, routing, config=UNIFORM)
        sim.inject_pulse(a, 1.0, 1.0)
        sim.inject_pulse(b, 1.0 + offset, 1.0)
        sim.advance_to(10.0)
        assert bool(sim.rise_times[v]) == fires, (offset, sim.rise_times[v])
        if fires:                                # triggered by the LATER edge
            assert abs(sim.rise_times[v][0] - (2.0 + offset)) <= TOL


def test_uniform_refractory_double_arrival():
    """Dead time is [t, t+delay+width), half-open: a second excitatory edge
    inside it is discarded; at exactly t+delay+width it is accepted."""
    grid, routing, a, b = _buffer_pair()
    inside = PulseSim(grid, routing, config=UNIFORM)
    inside.inject_pulse(a, 0.0, 1.0)
    inside.inject_pulse(a, 1.5, 1.0)             # refractory until 2.0
    inside.advance_to(10.0)
    assert inside.rise_times[b] == [1.0]
    boundary = PulseSim(grid, routing, config=UNIFORM)
    boundary.inject_pulse(a, 0.0, 1.0)
    boundary.inject_pulse(a, 2.0, 1.0)           # exactly at re-arm
    boundary.advance_to(10.0)
    assert boundary.rise_times[b] == [1.0, 3.0]


def test_uniform_or_refractory_single_pulse():
    """Two nearby edges into an OR node regenerate ONE pulse; edges separated
    past the dead time give two."""
    grid, routing, a, b, v = _two_input('or')
    near = PulseSim(grid, routing, config=UNIFORM)
    near.inject_pulse(a, 0.0, 1.0)
    near.inject_pulse(b, 0.5, 1.0)
    near.advance_to(10.0)
    assert near.rise_times[v] == [1.0]
    far = PulseSim(grid, routing, config=UNIFORM)
    far.inject_pulse(a, 0.0, 1.0)
    far.inject_pulse(b, 2.0, 1.0)
    far.advance_to(10.0)
    assert far.rise_times[v] == [1.0, 3.0]


def test_uniform_loop_period_and_survival():
    """A circulating pulse in a 2-loop survives with period 2·delay."""
    grid, routing, u, v = _loop_pair()
    sim = PulseSim(grid, routing, config=UNIFORM, max_events=64)
    sim.inject_pulse(u, 0.0, 1.0)
    sim.advance_to(12.0)
    assert not sim.overflow
    assert sim.rise_times[u][:5] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert sim.rise_times[v][:5] == [1.0, 3.0, 5.0, 7.0, 9.0]


# ── the four inhibitor timing cases (shared contract, every model) ───────────────

def _model_sims():
    for name, config in (('uniform', UNIFORM),
                         ('width_preserving', WP)):
        yield name, config


def test_inhibitor_timing_cases():
    """before: an expired veto does not block. simultaneous: veto wins (batch
    applies rises before notifications). during-delay: no cancellation of a
    committed trigger. during-output: no truncation of an active pulse."""
    for name, config in _model_sims():
        grid, routing, src, inh, v = _inhibited_buffer()

        before = PulseSim(grid, routing, config=config)
        before.inject_pulse(inh, 0.0, 1.0)
        before.inject_pulse(src, 1.0, 1.0)       # veto low at half-open end
        before.advance_to(10.0)
        assert before.rise_times[v] == [2.0], (name, before.rise_times[v])

        simultaneous = PulseSim(grid, routing, config=config)
        simultaneous.inject_pulse(inh, 1.0, 1.0)
        simultaneous.inject_pulse(src, 1.0, 1.0)
        simultaneous.advance_to(10.0)
        assert simultaneous.rise_times[v] == [], name

        during_delay = PulseSim(grid, routing, config=config)
        during_delay.inject_pulse(src, 0.0, 1.0)
        during_delay.inject_pulse(inh, 0.5, 1.0)  # after trigger, before rise
        during_delay.advance_to(10.0)
        assert during_delay.rise_times[v] == [1.0], (
            '%s: a committed trigger must not be cancelled' % name)

        during_output = PulseSim(grid, routing, config=config)
        during_output.inject_pulse(src, 0.0, 1.0)
        during_output.inject_pulse(inh, 1.2, 0.5)  # while output is high
        during_output.advance_to(10.0)
        expected_w = 1.0
        (start, end), = _closed(during_output.pulse_intervals[v])
        assert abs((end - start) - expected_w) <= TOL, (
            '%s: an active output must not be truncated' % name)






# ── width-preserving: chains, OR unions, coincidence widths, loops ───────────────

def test_wp_chain_width_drift_zero():
    """Width is preserved exactly through 1, 2, 4 and 8 buffer hops."""
    for hops in (1, 2, 4, 8):
        grid, routing, cells = _buffer_chain(hops)
        for w in (0.4, 1.7):
            sim = PulseSim(grid, routing, config=WP)
            sim.inject_pulse(cells[0], 0.25, w)
            sim.advance_to(40.0)
            for k, cell in enumerate(cells):
                (start, end), = _closed(sim.pulse_intervals[cell])
                assert abs(start - (0.25 + k)) <= TOL, (hops, w, k)
                assert abs((end - start) - w) <= TOL, (
                    'width drifted at hop %d (chain %d, w %.1f): %r'
                    % (k, hops, w, (start, end)))


def test_wp_or_union_cases():
    """OR output is the delayed union of the connected carrier set."""
    grid, routing, a, b, v = _two_input('or')
    cases = (
        ('separated',   (0.0, 1.0), (3.0, 1.0), [(1.0, 2.0), (4.0, 5.0)]),
        ('overlapping', (0.0, 2.0), (1.0, 2.0), [(1.0, 4.0)]),
        ('nested',      (0.0, 3.0), (1.0, 1.0), [(1.0, 4.0)]),
        ('touching',    (0.0, 1.0), (1.0, 1.0), [(1.0, 3.0)]),
    )
    for label, (sa, wa), (sb, wb), expected in cases:
        sim = PulseSim(grid, routing, config=WP)
        sim.inject_pulse(a, sa, wa)
        sim.inject_pulse(b, sb, wb)
        sim.advance_to(20.0)
        got = _closed(sim.pulse_intervals[v])
        assert len(got) == len(expected), (label, got)
        for (gs, ge), (es, ee) in zip(got, expected):
            assert abs(gs - es) <= TOL and abs(ge - ee) <= TOL, (label, got)


def test_wp_or_busy_discard():
    """One waveform per node: a DISJOINT pulse arriving while the output is
    still high is lost; at exactly the output fall it is accepted (the batch
    frees the node before notifications)."""
    grid, routing, a, b, v = _two_input('or')
    lost = PulseSim(grid, routing, config=WP)
    lost.inject_pulse(a, 0.0, 2.0)               # out [1, 3)
    lost.inject_pulse(b, 2.5, 0.5)               # rises while out is high
    lost.advance_to(20.0)
    assert _closed(lost.pulse_intervals[v]) == [(1.0, 3.0)]
    boundary = PulseSim(grid, routing, config=WP)
    boundary.inject_pulse(a, 0.0, 2.0)
    boundary.inject_pulse(b, 3.0, 0.5)           # exactly at the output fall
    boundary.advance_to(20.0)
    assert [s for s, _ in _closed(boundary.pulse_intervals[v])] == [1.0, 4.0]


def test_wp_coincidence_later_edge_width():
    """Non-tie coincidence transports the LATER edge's waveform — its width,
    bound to its fall — whichever input that is (the decided rule; see the
    recorded discontinuity note in NODE_CONTRACTS.md)."""
    grid, routing, a, b, v = _two_input('and')
    narrow_later = PulseSim(grid, routing, config=WP)
    narrow_later.inject_pulse(a, 0.0, 3.0)
    narrow_later.inject_pulse(b, 0.3, 0.5)
    narrow_later.advance_to(20.0)
    assert _closed(narrow_later.pulse_intervals[v]) == [(1.3, 1.8)]
    wide_later = PulseSim(grid, routing, config=WP)
    wide_later.inject_pulse(a, 0.0, 0.5)
    wide_later.inject_pulse(b, 0.3, 3.0)
    wide_later.advance_to(20.0)
    assert _closed(wide_later.pulse_intervals[v]) == [(1.3, 4.3)]


def test_wp_coincidence_window_sweep():
    """The trigger rule stays the uniform edge window (inclusive)."""
    grid, routing, a, b, v = _two_input('and')
    for offset, fires in ((0.5, True), (0.51, False)):
        sim = PulseSim(grid, routing, config=WP)
        sim.inject_pulse(a, 0.0, 1.0)
        sim.inject_pulse(b, offset, 1.0)
        sim.advance_to(20.0)
        assert bool(sim.rise_times[v]) == fires, (offset, sim.rise_times[v])


def test_wp_loop_token_circulates():
    """Loop regime (a): a token no wider than the per-hop delay circulates
    losslessly, width preserved every lap (incl. the w == delay boundary)."""
    grid, routing, u, v = _loop_pair()
    for w in (0.5, 1.0):
        sim = PulseSim(grid, routing, config=WP, max_events=64)
        sim.inject_pulse(u, 0.0, w)
        sim.advance_to(12.0)
        assert not sim.overflow
        assert sim.rise_times[u][:5] == [0.0, 2.0, 4.0, 6.0, 8.0], w
        for start, end in _closed(sim.pulse_intervals[v]):
            assert abs((end - start) - w) <= TOL, (
                'circulating width not preserved: %r' % ((start, end),))


def test_wp_loop_intermediate_width_dies_out():
    """Loop regime (b): delay < w < 2·delay — the returning rise finds the
    node busy and the token dies out after about one lap."""
    grid, routing, u, v = _loop_pair()
    sim = PulseSim(grid, routing, config=WP, max_events=64)
    sim.inject_pulse(u, 0.0, 1.5)
    sim.advance_to(14.0)
    assert len(sim.rise_times[u]) <= 2 and len(sim.rise_times[v]) <= 2
    assert sum(sim.activity_at(13.0).values()) == 0, "loop should be dead"


def test_wp_loop_wide_pulse_latches():
    """Loop regime (c): w >= 2·delay — union swallows the returning edge and
    the mutual fall-binding latches both wires permanently high (no edges)."""
    grid, routing, u, v = _loop_pair()
    for w in (2.0, 3.0):
        sim = PulseSim(grid, routing, config=WP, max_events=64)
        sim.inject_pulse(u, 0.0, w)
        sim.advance_to(14.0)
        assert len(sim.rise_times[u]) == 1 and len(sim.rise_times[v]) == 1, w
        assert sum(sim.activity_at(13.0).values()) == 2, (
            'w=%.1f: loop should latch permanently high' % w)


def test_wp_veto_no_truncation():
    """An inhibitor rising mid-transport neither cancels the committed rise
    nor truncates the transported waveform."""
    grid, routing, src, inh, v = _inhibited_buffer()
    sim = PulseSim(grid, routing, config=WP)
    sim.inject_pulse(src, 0.0, 3.0)              # out [1, 4)
    sim.inject_pulse(inh, 1.5, 1.0)              # during the output
    sim.advance_to(20.0)
    assert _closed(sim.pulse_intervals[v]) == [(1.0, 4.0)]


def test_wp_translation_invariance():
    _assert_translation(WP, widths_key=False)


def _assert_translation(config, widths_key):
    """All output edges shift by exactly the stimulus shift (sub-tick)."""
    grid, routing, a, b, v = _two_input('and')
    delta = 0.37

    def run(shift):
        sim = PulseSim(grid, routing, config=config)
        sim.inject_pulse(a, 1.0 + shift, 2.0)
        sim.inject_pulse(b, 1.3 + shift, 0.5)
        sim.advance_to(30.0 + shift)
        return sim

    base, moved = run(0.0), run(delta)
    for cell in grid:
        xs, ys = base.rise_times[cell], moved.rise_times[cell]
        assert len(xs) == len(ys), cell
        for x, y in zip(xs, ys):
            assert abs((x + delta) - y) <= 1e-6, (cell, x, y)


# ── pulse_intervals: the complete-waveform log ───────────────────────────────────

def test_intervals_legacy_log_matches_waveform():
    """Legacy models: one [start, end] entry per pulse; wired-OR merges extend
    the open entry instead of adding a spurious one."""
    grid, routing, a, b = _buffer_pair()
    sim = PulseSim(grid, routing, config=UNIFORM)
    sim.inject_pulse(a, 0.0, 1.0)
    sim.inject_pulse(a, 0.5, 2.0)                # merges into [0, 2.5)
    sim.advance_to(10.0)
    assert sim.pulse_intervals[a] == [[0.0, 2.5]]
    assert sim.pulse_intervals[b] == [[1.0, 2.0]]


def test_intervals_step_held_input():
    """A held stream input logs as ONE growing interval, per the one-edge rule."""
    grid, routing, a, b = _buffer_pair()
    sim = PulseSim(grid, routing, config=UNIFORM)
    for bit in (1, 1, 1, 0, 0):
        sim.step({a: bit})
    sim.advance_to(6.0)
    assert sim.pulse_intervals[a] == [[0.0, 3.0]]


def test_intervals_open_until_fall():
    """Width-preserving aggregates log [t, inf] while high, closed at the fall
    — so interval-aware scoring can distinguish 'still high' from 'fell'."""
    grid, routing, a, b = _buffer_pair()
    sim = PulseSim(grid, routing, config=WP)
    sim.inject_pulse(a, 0.0, 4.0)
    sim.advance_to(2.0)                          # mid-pulse
    assert sim.pulse_intervals[a] == [[0.0, INF]]
    assert sim.pulse_intervals[b] == [[1.0, INF]]
    sim.advance_to(20.0)
    assert sim.pulse_intervals[a] == [[0.0, 4.0]]
    assert sim.pulse_intervals[b] == [[1.0, 5.0]]


# ── standalone runner ────────────────────────────────────────────────────────────

def _main():
    tests = [f for name, f in sorted(globals().items())
             if name.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except AssertionError as e:
            print("FAIL  %s\n      %s" % (fn.__name__, e))
        except Exception as e:                     # noqa: BLE001
            print("ERROR %s\n      %r" % (fn.__name__, e))
    print("\n%d/%d node-contract tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
