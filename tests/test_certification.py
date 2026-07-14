"""
tests/test_certification.py — the held-out verdict rule that guards every
"solved" claim (nv_evo.certification). Fast/pure: no growth, no evolution, no
multiprocessing. The end-to-end certify() on a real evolved winner is exercised
by reproduce.py.

Run under pytest, or standalone:  py tests/test_certification.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv_evo import TEMPORAL_TARGETS                        # noqa: E402
from nv_evo.certification import classify, oracle_spec_for, certify  # noqa: E402


def test_classify_certified_when_holdout_clears_bar():
    assert classify(1.00, 1.00, 0.90) == 'CERTIFIED'
    assert classify(0.93, 0.91, 0.90) == 'CERTIFIED'   # held-out alone decides


def test_classify_overfit_only_on_a_large_gap():
    # trains well but generalisation collapses -> memorised timing
    assert classify(0.98, 0.60, 0.90).startswith('OVERFIT')


def test_classify_below_threshold_is_not_overfit():
    # a small train->held-out drop below the bar is honest under-solving, not
    # an accusation of memorisation
    v = classify(0.921, 0.893, 0.90)
    assert v.startswith('BELOW') and 'generalises' in v


def test_classify_below_with_large_gap_reports_weak_generalisation():
    # below the bar AND a big train->held-out drop must NOT be called
    # "generalises" (resettable-toggle 0.80/0.64 case)
    v = classify(0.802, 0.644, 0.90)
    assert v.startswith('BELOW') and 'weak generalisation' in v


def test_classify_solve_and_plateau_and_uncertified():
    assert classify(1.0, None, 0.999, kind='solve') == 'SOLVED'
    assert classify(0.5, None, 0.999, kind='solve').startswith('NOT SOLVED')
    assert classify(0.89, None, 0.95, kind='plateau').startswith('PLATEAU')
    assert classify(1.0, None, 0.90).startswith('UNCERTIFIED')   # no held-out


def test_oracle_mapping_present_only_for_oracle_targets():
    assert oracle_spec_for(TEMPORAL_TARGETS['C-element (2-in join)']) is not None
    assert oracle_spec_for(TEMPORAL_TARGETS['SR latch']) is not None
    assert oracle_spec_for(TEMPORAL_TARGETS['Oscillator']) is None   # autonomous


def test_certify_non_oracle_target_is_uncertified_without_scoring():
    """A target with no reference oracle must be reported UNCERTIFIED and must
    NOT attempt held-out scoring (no growth/sim needed)."""
    from nv_evo import random_hex_genome
    res = certify(random_hex_genome(1), TEMPORAL_TARGETS['Oscillator'],
                  train=1.0, backend='nervous')
    assert res['verdict'].startswith('UNCERTIFIED')
    assert res['holdouts'] is None and res['holdout'] is None


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
            passed += 1
        except Exception as e:                     # noqa: BLE001
            print("%s  %s: %s" % ("FAIL" if isinstance(e, AssertionError)
                                  else "ERROR", fn.__name__, e))
    print("\n%d/%d certification tests passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(_main())
