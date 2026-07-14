"""
reproduce.py - reproduce this project's headline claims from scratch, seeded.

The north-star is a *defensible, reproducible* claim that the asynchronous
nervous-net substrate can EVOLVE useful async circuits - not one lucky run. This
script operationalises that: each claim is a seeded evolution plus a certification
gate (fresh held-out stimulus for input-driven targets), reported with an honest
verdict - including the KNOWN substrate limits, which are reported as such rather
than hidden.

    py reproduce.py                 # list claims
    py reproduce.py all             # run every claim
    py reproduce.py c_element       # run one claim
    py reproduce.py async_substrate # run the "no hidden clock" synchrony audit

Each claim prints: train fitness, held-out fitness on 3 fresh seeds (for oracle
targets), and a verdict:
    SOLVED     train >= 0.999
    CERTIFIED  train and mean held-out both >= the claim's threshold
    OVERFIT    train >= threshold but held-out below it (memorised timing)
    PLATEAU    a documented substrate limit; reported, not a pass/fail
"""
from __future__ import annotations

import sys


# claim -> how to reproduce it. `oracle` (when set) enables held-out
# certification via a fresh-seed re-sample; `kind` selects the verdict rule.
CLAIMS = {
    'c_element': dict(
        target='C-element (2-in join)', oracle='C-element (oracle)',
        gens=60, pop=80, seed=1, kind='certify', threshold=0.99,
        note='OPEN FRONTIER: the corrected balanced rendezvous bank rejects '
             'single-input echoes and fixed oscillators; seed 1 currently '
             'generalises around 0.72 but does not solve'),
    'echo': dict(
        target='Echo (delay 3)', oracle='Echo (oracle)',
        gens=40, pop=80, seed=1, kind='certify', threshold=0.90,
        note='delay line - reproduce the input-edge train'),
    'toggle': dict(
        target='Toggle flip-flop', oracle='Toggle (oracle)',
        gens=80, pop=100, seed=1, kind='certify', threshold=0.90,
        note='T flip-flop - one input flips a stored bit (HARD: a single input '
             'must do a context-dependent flip; lands ~0.92 and generalises but '
             'sits near a known attractor, so it does not cleanly solve here)'),
    'period_doubler': dict(
        target='Period doubler (2x)', oracle='Period doubler (oracle)',
        gens=80, pop=100, seed=1, kind='certify', threshold=0.95,
        note='halve the edge rate: a period-p input train -> period-2p output '
             '(emit every 2nd edge); mixed rates + silent guard forbid fixed '
             'cadences and free-running oscillators'),
    'oscillator': dict(
        target='Oscillator', oracle=None,
        gens=40, pop=80, seed=1, kind='solve', threshold=0.999,
        note='autonomous period-2 rhythm (no input relation to hold out)'),
    'sr_latch': dict(
        target='SR latch', oracle='SR latch (oracle)',
        gens=80, pop=100, seed=1, kind='plateau', threshold=0.95,
        note='KNOWN LIMIT: clearing a circulating pulse on the degree-3 '
             'honeycomb is geometrically hard; caps ~0.97 (see memory).'),
    'gated_oscillator': dict(
        target='Gated oscillator', oracle='Gated oscillator (oracle)',
        gens=60, pop=80, seed=1, kind='certify', threshold=0.90,
        note='OPEN FRONTIER (controllable memory): A starts a period-2 run, B '
             'stops it. Lands ~0.78 and generalises but does not solve - '
             'stopping a circulating pulse is the same hard clearing problem.'),
    'resettable_toggle': dict(
        target='Resettable toggle', oracle='Resettable toggle (oracle)',
        gens=60, pop=80, seed=1, kind='certify', threshold=0.90,
        note='OPEN FRONTIER (clear+reload): A flips a stored bit, B clears it. '
             'Trains ~0.80 but held-out drops (weak generalisation) - the '
             'flagship robust-clear-and-reload gap is still open.'),
}


def _run_claim(name, spec):
    from nv_evo import TEMPORAL_TARGETS
    from nv_evo.ga import evolve_nervous
    from nv_evo.oracle import ORACLE_SPECS, holdout_score

    target = TEMPORAL_TARGETS[spec['target']]
    print("\n" + "=" * 62)
    print("CLAIM  %s" % name)
    print("  %s" % spec['note'])
    print("  target=%r  seed=%d  pop=%d  gens=%d"
          % (spec['target'], spec['seed'], spec['pop'], spec['gens']))
    best, train = evolve_nervous(target, generations=spec['gens'],
                                 pop=spec['pop'], seed=spec['seed'], verbose=False)
    print("  train fitness      : %.4f" % train)

    holdouts = []
    if spec['oracle'] is not None:
        oracle_fn = ORACLE_SPECS[spec['oracle']]
        for hs in (4242, 777, 31415):
            holdouts.append(holdout_score(best, oracle_fn, seed=hs))
        print("  held-out (3 fresh) : %s  (mean %.4f)"
              % (", ".join("%.3f" % h for h in holdouts),
                 sum(holdouts) / len(holdouts)))

    from nv_evo.certification import classify
    thr = spec['threshold']
    mean_ho = sum(holdouts) / len(holdouts) if holdouts else None
    verdict = classify(train, mean_ho, thr, kind=spec['kind'])
    print("  VERDICT            : %s" % verdict)
    return name, train, mean_ho, verdict


def _run_synchrony_audit():
    """The 'substrate is genuinely asynchronous' claim = the metamorphic
    no-hidden-clock test suite."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'tests'))
    import importlib
    runner = importlib.import_module('run_tests')
    print("\n" + "=" * 62)
    print("CLAIM  async_substrate - metamorphic 'no hidden clock' audit")
    return runner.main()


def main(argv):
    what = argv[1] if len(argv) > 1 else None
    if what is None:
        print(__doc__)
        print("Available claims:")
        for name, spec in CLAIMS.items():
            print("  %-14s %s" % (name, spec['note'].split(';')[0]))
        print("  %-14s %s" % ('async_substrate',
                              'metamorphic no-hidden-clock synchrony audit'))
        print("  %-14s %s" % ('all', 'run every evolution claim'))
        return 0

    if what == 'async_substrate':
        return _run_synchrony_audit()

    names = list(CLAIMS) if what == 'all' else [what]
    if any(n not in CLAIMS for n in names):
        print("unknown claim %r; run with no args to list claims" % what)
        return 2
    results = [_run_claim(n, CLAIMS[n]) for n in names]
    print("\n" + "=" * 62 + "\nSUMMARY")
    for name, train, mean_ho, verdict in results:
        ho = "%.3f" % mean_ho if mean_ho is not None else "  -  "
        print("  %-12s train %.3f  held-out %s  ->  %s"
              % (name, train, ho, verdict))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
