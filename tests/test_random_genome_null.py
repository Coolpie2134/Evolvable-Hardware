"""tests/test_random_genome_null.py - the RANDOM GENOME null model.

`tests/test_null_models.py` sweeps hand-picked cheat behaviours through
`score_contract`. That validates the SCORER, which was already the careful part
of this project. It builds its traces synthetically and so never grows a body,
never exercises a substrate, and never touches the fitted output readout.

Every degeneracy found on this branch lived in exactly that unguarded gap:

    Echo (delay 3)        10% of RANDOM FNV genomes scored a perfect 1.000
    AND (temporal)         5%
    Veto gate              2.5%
    XOR (temporal)         2.5%
    Temporal XOR (2-in)    1.2%

None were reachable by the synthetic-trace gauntlet, because the thing that
produced the free 1.000 was the substrate plus best-of-N probe selection, not
the contract maths. A target an unevolved random genome solves is not measuring
search, and a solve-rate table including it is reporting a free win.

So this module asserts the property that matters end to end: a genome drawn at
random, grown, and scored through the REAL evaluation path must not solve the
target and must not reach the cheat ceiling.

It also asserts the opposite failure the same sweep exposed - targets that
return exactly 0.000 for every random genome. A landscape flat at zero gives a
GA nothing to climb, so such a target is not "hard", it is unreachable, and it
should be labelled rather than silently benchmarked.

Kept cheap so it runs in the ordinary suite; raise SAMPLES locally when adding
a target.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.config import FNVConfig                                # noqa: E402
from substrates.fnv.evaluation import score_functional              # noqa: E402
from substrates.fnv.genome import random_functional_genome          # noqa: E402
from substrates.nervous.targets import TEMPORAL_TARGETS             # noqa: E402

#: Shared with tests/test_null_models.py. No trivial behaviour may reach it.
CHEAT_CEILING = 0.85
SAMPLES = 40
SEED = 20260817

#: Targets a random genome is currently known to solve outright. Recorded
#: rather than skipped so the list can only ever SHRINK: if a target stops
#: being freely solvable that is good news, and this fails until someone
#: deletes the entry deliberately.
KNOWN_DEGENERATE = {
    'AND (temporal)',
    'Veto gate',
    'XOR (temporal)',
    'Temporal XOR (2-in)',
    # Added when the constructive draw was opened to every enabled family.
    # That change was necessary - without the stateful families the substrate
    # could not express memory at all - but it makes this target trivially
    # winnable, because a single TOGGLE component IS the answer. The general
    # rule it implies: a target whose answer is one catalogue part measures
    # placement, not evolution.
    'Toggle flip-flop',
    # 'Echo (delay 3)' was here and is NOT any more: it needed a 3-hop chain of
    # pass-through parts, near-certain while DELAY held 64.6% of every grown
    # body and rare now that the draw is family-balanced.
}

#: Targets with no gradient at all under random sampling: every draw scores
#: exactly 0.0, so there is nothing for selection to climb from.
KNOWN_ZERO_GRADIENT = {
    'Oscillator',
    'Pattern (1000)',
}


def _sample_scores(target, samples=SAMPLES, seed=SEED):
    families = tuple(FNVConfig().families)
    random.seed(seed)
    scores = []
    for _ in range(samples):
        try:
            genome = random_functional_genome(
                n_chroms=2, max_telomere=8, families=families,
                n_inputs=int(target.n_inputs),
                output_roles=tuple(t.role for t in target.outputs))
            score = score_functional(genome, target)
        except Exception:
            score = None
        if score is not None:
            scores.append(float(score))
    return scores


def _interesting_targets():
    """A stable, cheap cross-section: the degenerate and zero-gradient sets
    plus a few genuinely hard controls."""
    names = sorted(KNOWN_DEGENERATE | KNOWN_ZERO_GRADIENT | {
        'Divide-by-3', 'Gated oscillator', 'C-element (2-in join)',
        'SR latch', 'Toggle flip-flop'})
    return [(name, TEMPORAL_TARGETS[name])
            for name in names if name in TEMPORAL_TARGETS]


def test_random_genomes_do_not_solve_non_degenerate_targets():
    """No random genome may score a perfect 1.000 on a target not already
    admitted to be degenerate."""
    offenders = []
    for name, target in _interesting_targets():
        if name in KNOWN_DEGENERATE:
            continue
        scores = _sample_scores(target)
        perfect = [s for s in scores if s >= 0.999]
        if perfect:
            offenders.append((name, len(perfect), len(scores)))
    assert not offenders, (
        'random genomes solved targets outright (substrate/readout '
        'degeneracy, not search): %r' % (offenders,))


def test_random_genomes_stay_under_the_cheat_ceiling():
    """The bar tests/test_null_models.py applies to synthetic cheats, applied
    to whole random circuits."""
    offenders = []
    for name, target in _interesting_targets():
        if name in KNOWN_DEGENERATE:
            continue
        scores = _sample_scores(target)
        if scores and max(scores) >= CHEAT_CEILING:
            offenders.append((name, round(max(scores), 4)))
    assert not offenders, (
        'random genomes reached the cheat ceiling %.2f: %r'
        % (CHEAT_CEILING, offenders))


def test_known_degenerate_targets_are_still_degenerate():
    """The admitted list may only shrink."""
    still = set()
    for name in sorted(KNOWN_DEGENERATE):
        target = TEMPORAL_TARGETS.get(name)
        if target is None:
            continue
        if any(s >= 0.999 for s in _sample_scores(target, samples=80)):
            still.add(name)
    fixed = KNOWN_DEGENERATE - still
    assert not fixed, (
        'these targets no longer admit a free random solve - delete them from '
        'KNOWN_DEGENERATE: %r' % (sorted(fixed),))


def test_zero_gradient_targets_are_declared():
    """A target scoring 0.0 for every random genome offers no gradient.

    Legitimate to keep, but it must be DECLARED: a run reporting 0.000 on one
    is not evidence of a hard problem, it is evidence search never had a
    foothold.
    """
    undeclared = []
    for name, target in _interesting_targets():
        scores = _sample_scores(target)
        if scores and max(scores) <= 0.0 and name not in KNOWN_ZERO_GRADIENT:
            undeclared.append(name)
    assert not undeclared, (
        'targets with no random-genome gradient must be declared in '
        'KNOWN_ZERO_GRADIENT: %r' % (undeclared,))
