"""
tools/make_scoring_golden.py — regenerate tests/fixtures/scoring_golden.json.

The golden file freezes the scoring contract: a synthetic bundle battery
(every registered target x {perfect, shift2, half, silence, always} x hold_tol
variants) plus float-time retention/coverage scenarios, scored through
nv_evo/scoring.py. tests/test_scoring_equivalence.py replays the stored
bundles and demands identical scores/cases/alignments, so ANY change to
scoring semantics fails the suite until this script is deliberately re-run.

Only regenerate after an INTENTIONAL scoring change, and say why in the
commit message — the whole point of the gate is that scores never drift
silently.

    py tools/make_scoring_golden.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nv_evo.targets import TEMPORAL_TARGETS                       # noqa: E402
from nv_evo.scoring import (TemporalTraces, score_temporal_bundle,  # noqa: E402
                            windowed_score, _waveform_expected,
                            _expected_events, _obs_len)
from nv_evo import scoring as sc                                  # noqa: E402

BUNDLE_VARIANTS = ('perfect', 'shift2', 'half', 'silence', 'always')


def exp_to_samples(exp, obs):
    return [1 if v == 1 else 0 for v in exp] + [0] * max(0, obs - len(exp))


def thin_samples(samples):
    out, seen = list(samples), 0
    for t, v in enumerate(samples):
        if v:
            if seen % 2 == 1:
                out[t] = 0
            seen += 1
    return out


def build_bundle(target, variant):
    obs = _obs_len(target)
    mode = getattr(target, 'score_mode', 'trace')
    samples, events, intervals = {}, {}, {}
    roles = sorted({role for tr in target.trials for role in tr.expected})
    for role in roles:
        s_list, e_list, i_list = [], [], []
        for tr in target.trials:
            exp = tr.expected.get(role, [])
            ev = [float(t) for t in _expected_events(tr, role)] if exp else []
            iv = ([(float(a), float(b)) for a, b in
                   _waveform_expected(target, tr, role)]
                  if mode == 'waveform' else [])
            sm = exp_to_samples(exp, obs)
            if variant == 'shift2':
                ev = [t + 2.0 for t in ev]
                iv = [(a + 2.0, b + 2.0) for a, b in iv]
                sm = [0, 0] + sm[:-2]
            elif variant == 'half':
                ev = ev[::2]
                iv = iv[::2]
                sm = thin_samples(sm)
            elif variant == 'silence':
                ev, iv, sm = [], [], [0] * obs
            elif variant == 'always':
                ev = [float(t) for t in range(obs)]
                iv = [(0.0, float(obs))]
                sm = [1] * obs
            s_list.append(sm)
            e_list.append(ev)
            i_list.append(iv)
        samples[role] = s_list
        events[role] = e_list
        intervals[role] = i_list
    return samples, events, intervals


def bundle_from_parts(samples, events, intervals, hold_tol=None):
    b = TemporalTraces({r: [list(t) for t in ts] for r, ts in samples.items()},
                       events={r: [list(t) for t in ts]
                               for r, ts in events.items()},
                       intervals={r: [[tuple(p) for p in t] for t in ts]
                                  for r, ts in intervals.items()})
    if hold_tol is not None:
        b.hold_tol = hold_tol
    return b


def main():
    records = []
    skipped = []
    for name, target in sorted(TEMPORAL_TARGETS.items()):
        mode = getattr(target, 'score_mode', 'trace')
        if mode in ('retention', 'sr_retention'):
            skipped.append(name)
            continue
        tols = (None, 0) if mode == 'trace' else (None,)
        for variant in BUNDLE_VARIANTS:
            samples, events, intervals = build_bundle(target, variant)
            for tol in tols:
                b = bundle_from_parts(samples, events, intervals, tol)
                score, cases, alignment = score_temporal_bundle(b, target)
                b2 = bundle_from_parts(samples, events, intervals, tol)
                if alignment is not None:
                    fscore, fcases, fused = score_temporal_bundle(
                        b2, target, alignment=alignment)
                else:
                    fscore, fcases, fused = None, None, None
                b3 = bundle_from_parts(samples, events, intervals, tol)
                wscore = windowed_score(b3, target)
                records.append({
                    'target': name, 'mode': mode, 'variant': variant,
                    'hold_tol': tol,
                    'bundle': {'samples': samples, 'events': events,
                               'intervals': intervals},
                    'score': score, 'cases': list(cases),
                    'alignment': alignment,
                    'frozen_score': fscore,
                    'frozen_cases': None if fcases is None else list(fcases),
                    'frozen_alignment': fused,
                    'windowed': wscore,
                })
        print('done %-42s %s' % (name, mode))

    # ── float-time coverage/retention scorer goldens ─────────────────────────
    ring = [7.0 + 2.0 * k for k in range(29)]              # sustained ring to 63
    decayed = [7.0 + 2.0 * k for k in range(7)]            # dies at ~19
    dense = [float(t) for t in range(65)]
    sr_ring = [7.0 + 2.0 * k for k in range(15)]           # active 7..35, then quiet
    scenarios = []
    par = sc.parity_intervals([5.0], 64)
    par2 = sc.parity_intervals([5.0, 37.0], 96)
    sri = sc.sr_intervals([5.0], [37.0], 96)
    for label, rise in (('ring', ring), ('decayed', decayed),
                        ('silence', []), ('dense', dense),
                        ('sr_ring', sr_ring)):
        for offset in (2.0, 3.0):
            rec = {'label': label, 'offset': offset, 'rise': rise,
                   'parity_64': par, 'parity_96': par2, 'sr_96': sri}
            rec['score_retention_p64'] = sc.score_retention(rise, par, offset)
            rec['score_retention_graded_p64'] = sc.score_retention_graded(
                rise, par, offset)
            rec['score_retention_p96'] = sc.score_retention(rise, par2, offset)
            rec['score_retention_graded_p96'] = sc.score_retention_graded(
                rise, par2, offset)
            rec['score_state_intervals_sr'] = sc.score_state_intervals(
                rise, sri, offset)
            rec['score_interval_graded'] = [
                sc.score_interval_graded(rise, state, a, b, offset)
                for (state, a, b) in sri]
            rec['score_reset_influence'] = sc.score_reset_influence(
                rise, 37.0, offset)
            scenarios.append(rec)

    out = {'bundles': records, 'coverage': scenarios,
           'skipped_modes': skipped}
    path = os.path.join(ROOT, 'tests', 'fixtures')
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'scoring_golden.json'), 'w') as fh:
        json.dump(out, fh)
    print('wrote %d bundle records, %d coverage records'
          % (len(records), len(scenarios)))


if __name__ == '__main__':
    main()
