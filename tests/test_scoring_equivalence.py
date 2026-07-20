"""
tests/test_scoring_equivalence.py — the refactor equivalence gate.

tests/fixtures/scoring_golden.json was captured by replaying a synthetic
bundle battery (every registered target x {perfect, shift2, half, silence,
always} x hold_tol variants) through the PRE-refactor scorers, plus a battery
of float-time retention/coverage scenarios. This test replays the exact same
bundles through the consolidated nv_evo/scoring.py path and demands identical
scores, per-case vectors, and fitted alignments — so the consolidation is
provably a relocation, not a semantics change. It stays in the suite as a
permanent scorer-regression net: any future scoring edit that shifts numbers
must consciously regenerate the goldens (py tools/make_scoring_golden.py)
and say why in the commit message.

Run under the suite runner:  py tests/run_tests.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo.targets import TEMPORAL_TARGETS                     # noqa: E402
from nv_evo.scoring import (TemporalTraces, score_temporal_bundle,  # noqa: E402
                            windowed_score, relation_spec, RELATIONS,
                            needs_samples, score_retention,
                            score_retention_graded, score_state_intervals,
                            score_interval_graded, score_reset_influence)

_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'scoring_golden.json')
EPS = 1e-12


def _load():
    with open(_GOLDEN, encoding='utf-8') as fh:
        return json.load(fh)


def _rebuild(bundle, hold_tol):
    traces = TemporalTraces(
        {role: [list(t) for t in ts] for role, ts in bundle['samples'].items()},
        events={role: [[float(v) for v in t] for t in ts]
                for role, ts in bundle['events'].items()},
        intervals={role: [[tuple(p) for p in t] for t in ts]
                   for role, ts in bundle['intervals'].items()})
    if hold_tol is not None:
        traces.hold_tol = hold_tol
    return traces


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= EPS


def test_registry_covers_every_historical_mode():
    assert set(RELATIONS) == {'events', 'trace', 'waveform', 'cadence',
                              'period_stepper', 'retention', 'sr_retention'}
    families = {spec.family for spec in RELATIONS.values()}
    assert families == {'match', 'rhythm'}
    # observable drives what the harness collects; 'samples' only where the
    # scorer genuinely consumes ticks
    assert needs_samples(TEMPORAL_TARGETS[next(
        name for name, t in TEMPORAL_TARGETS.items()
        if getattr(t, 'score_mode', 'trace') == 'trace')])


def test_bundle_scores_match_legacy_goldens():
    data = _load()
    assert data['bundles'], 'golden battery is empty'
    missing, checked = [], 0
    for rec in data['bundles']:
        target = TEMPORAL_TARGETS.get(rec['target'])
        if target is None:            # target renamed/retired since capture
            missing.append(rec['target'])
            continue
        traces = _rebuild(rec['bundle'], rec['hold_tol'])
        score, cases, alignment = score_temporal_bundle(traces, target)
        ctx = '%s/%s/tol=%s' % (rec['target'], rec['variant'], rec['hold_tol'])
        assert _close(score, rec['score']), (ctx, score, rec['score'])
        assert len(cases) == len(rec['cases']), ctx
        assert all(_close(a, b) for a, b in zip(cases, rec['cases'])), ctx
        assert _close(alignment, rec['alignment']), (
            ctx, alignment, rec['alignment'])
        if rec['frozen_score'] is not None:
            traces2 = _rebuild(rec['bundle'], rec['hold_tol'])
            fscore, fcases, fused = score_temporal_bundle(
                traces2, target, alignment=rec['alignment'])
            assert _close(fscore, rec['frozen_score']), ctx
            assert all(_close(a, b)
                       for a, b in zip(fcases, rec['frozen_cases'])), ctx
            assert _close(fused, rec['frozen_alignment']), ctx
        traces3 = _rebuild(rec['bundle'], rec['hold_tol'])
        assert _close(windowed_score(traces3, target), rec['windowed']), ctx
        checked += 1
    # the battery must genuinely cover the suite: a mass-rename would silently
    # skip everything and this assert is the tripwire
    assert checked >= max(1, len(data['bundles']) - 10 * max(1, len(missing)))
    assert not missing, 'golden targets missing from registry: %s' % missing


def test_retention_coverage_scores_match_legacy_goldens():
    data = _load()
    assert data['coverage'], 'coverage battery is empty'
    for rec in data['coverage']:
        rise = [float(v) for v in rec['rise']]
        offset = float(rec['offset'])
        par = [tuple(iv) for iv in rec['parity_64']]
        par2 = [tuple(iv) for iv in rec['parity_96']]
        sri = [tuple(iv) for iv in rec['sr_96']]
        ctx = '%s/offset=%s' % (rec['label'], rec['offset'])
        assert _close(score_retention(rise, par, offset),
                      rec['score_retention_p64']), ctx
        assert _close(score_retention_graded(rise, par, offset),
                      rec['score_retention_graded_p64']), ctx
        assert _close(score_retention(rise, par2, offset),
                      rec['score_retention_p96']), ctx
        assert _close(score_retention_graded(rise, par2, offset),
                      rec['score_retention_graded_p96']), ctx
        assert _close(score_state_intervals(rise, sri, offset),
                      rec['score_state_intervals_sr']), ctx
        got = [score_interval_graded(rise, state, a, b, offset)
               for (state, a, b) in sri]
        assert all(_close(a, b)
                   for a, b in zip(got, rec['score_interval_graded'])), ctx
        assert _close(score_reset_influence(rise, 37.0, offset),
                      rec['score_reset_influence']), ctx


def test_evaluator_dispatch_is_registry_driven():
    """The retention pipelines are selected by the registry, not by string
    comparison scattered through ga/evaluation."""
    from nv_evo.persistence import retention_oracle, sr_full_oracle
    assert relation_spec(retention_oracle()).evaluator == 'retention'
    assert relation_spec(sr_full_oracle()).evaluator == 'sr_retention'
    for name, target in TEMPORAL_TARGETS.items():
        assert relation_spec(target).evaluator == 'bundle', name
