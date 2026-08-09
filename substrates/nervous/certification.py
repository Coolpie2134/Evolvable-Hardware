"""
substrates/nervous/certification.py - turn a "solved" fitness into a defensible verdict.

A training fitness near 1.0 is not a claim: temporal targets can be gamed by a
circuit that memorised the training schedules' exact timing, or by a leaky ring
that a short training window can't tell from real storage. The north-star's
credibility gate is generalisation to FRESH stimulus. This module re-samples new
schedules from a target's reference oracle, scores the winner with its TRAINING
readout and alignment frozen (no re-fit on validation), and classifies the gap:

    CERTIFIED  held-out >= threshold          (generalises - a real circuit)
    OVERFIT    trains >= threshold but the train->held-out drop is large
               (memorised timing)
    BELOW      trains and generalises but under the bar (not solved at this budget)
    SOLVED / PLATEAU / UNCERTIFIED             (autonomous / documented limit /
                                                no replayable reference)

One verdict rule, shared by reproduce.py, the evolution controller, and tests.
LUT exterior-edge I/O is not certified yet: its frozen readout adapter cannot
currently replay outside-to-facing-edge links, so certification returns an
explicit UNCERTIFIED verdict instead of presenting an unaudited score.
"""
from __future__ import annotations
import random

# The train-vs-held-out drop that distinguishes memorised timing (a big drop)
# from a target that simply isn't fully solved yet (a small drop, still general).
OVERFIT_GAP = 0.10
DEFAULT_HOLDOUT_SEEDS = (4242, 777, 31415)


def classify(train, holdout, threshold, kind='certify', overfit_gap=OVERFIT_GAP):
    """The single verdict rule. `holdout` may be None (no oracle reference)."""
    if kind == 'solve':
        return 'SOLVED' if train >= 0.999 else 'NOT SOLVED (train %.3f)' % train
    if kind == 'plateau':
        return 'PLATEAU (documented limit; train %.3f)' % train
    if holdout is None:
        return 'UNCERTIFIED (no oracle reference for this target)'
    if holdout >= threshold:
        return 'CERTIFIED'
    if train >= threshold and (train - holdout) > overfit_gap:
        return ('OVERFIT (memorised timing: train %.3f vs held-out %.3f)'
                % (train, holdout))
    # Below the bar: only claim it GENERALISES when the train->held-out drop is
    # small; a large drop is weak generalisation (partial memorisation), which
    # must not be dressed up as "generalises".
    quality = ('generalises' if (train - holdout) <= overfit_gap
               else 'weak generalisation, held-out drops %.3f' % (train - holdout))
    return ('BELOW THRESHOLD %.2f (train %.3f, held-out %.3f - %s, '
            'not fully solved at this budget)'
            % (threshold, train, holdout, quality))


def oracle_spec_for(target):
    """The reference-oracle spec function for a registry target, or None if the
    target has no oracle (autonomous rhythms, hand-built combinational, etc.)."""
    from .targets import ORACLE_KEY_TO_SPEC, LEGACY_ORACLE_KEY_TO_SPEC
    from .oracle import ORACLE_SPECS
    name = getattr(target, 'name', None)
    spec_name = (ORACLE_KEY_TO_SPEC.get(name)
                 or LEGACY_ORACLE_KEY_TO_SPEC.get(name))
    return ORACLE_SPECS.get(spec_name) if spec_name else None


# The run's physics config lives on the (mutable) training target the controller
# built (setattr'd pulse_config / lut_config). Fresh spec-built holdout targets
# start WITHOUT it, so without carrying it forward a champion evolved under a
# non-default node-timing model (pulse_delay / paper_analog) would be certified
# under the default uniform physics - every such run would read as OVERFIT. Copy
# it onto each spec target so fit + holdout run under the SAME physics as training.
_PHYSICS_ATTRS = (
    'pulse_config', 'lut_config', '_fnv_families',
    '_fnv_readout_mode', '_lut_function_families', 'lut_io_mode', 'max_events')


def carry_physics(src_target, dst_target):
    """Copy the run's physics config from the training target onto a freshly
    built spec target (returns dst).

    Also carries ``io_placement`` because it can change both binding and
    developmental origin. In particular, spatial input alleles are germline
    positions; a validation target left at ``fixed`` could grow a different
    organism from the fitted one."""
    for attr in _PHYSICS_ATTRS:
        cfg = getattr(src_target, attr, None)
        if cfg is not None:
            setattr(dst_target, attr, cfg)
    strategy = getattr(src_target, 'io_placement', 'fixed')
    if strategy != 'fixed':
        setattr(dst_target, 'io_placement', strategy)
    return dst_target


def _combinational_holdout_target(target, seed):
    """Fresh exhaustive truth-table schedule with a shuffled row order.

    The training wrapper already repeats every row under two fixed orders and
    phases. Certification freezes its fitted input/output sites, then presents
    the same complete Boolean contract in a seed-specific order. This catches
    a recurrent circuit that memorised schedule position instead of computing
    the row currently applied.
    """
    from substrates.snn.targets import OutputTerminal, Target
    from .targets import periodic_combinational_target

    cases = [
        (tuple(input_bits), tuple(output_bits))
        for input_bits, output_bits in target.combinational_cases]
    random.Random(int(seed)).shuffle(cases)
    n_inputs = int(getattr(target, 'combinational_data_inputs', 0))
    base = Target(
        getattr(target, 'name', 'truth table'),
        [(0, index) for index in range(n_inputs)],
        [OutputTerminal(terminal.role, tuple(terminal.pos))
         for terminal in target.outputs],
        cases)
    return carry_physics(
        target,
        periodic_combinational_target(
            base, repeats=2,
            latency=int(getattr(target, 'latency', 1))))


def _static_combinational_holdout_target(target, seed):
    """The same exhaustive table in a fresh order for settled-level FNV."""
    import dataclasses
    cases = list(target.cases)
    random.Random(int(seed)).shuffle(cases)
    return carry_physics(target, dataclasses.replace(target, cases=cases))


def certify(genome, target, train=None, backend='nervous',
            seeds=DEFAULT_HOLDOUT_SEEDS, threshold=0.90, kind='certify'):
    """Certify a winning genome against fresh held-out stimulus.

    Returns a dict: {target, train, holdouts, holdout (mean), verdict}. For a
    target with no oracle reference the verdict is UNCERTIFIED and no held-out
    scoring is attempted. `train` is the training fitness (passed in - this does
    not re-evolve); when omitted it is left None and only held-out is reported.
    """
    from .oracle import holdout_score
    from .evaluation import fit_readout, score_frozen

    name = getattr(target, 'name', '?')
    result = {'target': name, 'train': train, 'holdouts': None,
              'holdout': None, 'verdict': None}
    if (backend == 'lut'
            and getattr(target, 'lut_io_mode', 'source_pads')
            == 'exterior_edges'):
        result['verdict'] = (
            'UNCERTIFIED (LUT exterior-edge frozen validation is not '
            'implemented)')
        return result
    periodic_logic = bool(getattr(target, 'combinational_cases', ()))
    static_logic = (
        not getattr(target, 'temporal', False)
        and bool(getattr(target, 'cases', ())))
    if periodic_logic or static_logic:
        fitted = fit_readout(genome, target, backend=backend)
        if fitted is None:
            holdouts = [0.0] * len(seeds)
        else:
            holdouts = [
                score_frozen(
                    genome, (
                        _combinational_holdout_target(target, seed)
                        if periodic_logic else
                        _static_combinational_holdout_target(target, seed)),
                    fitted)
                for seed in seeds]
        mean_ho = sum(holdouts) / len(holdouts) if holdouts else 0.0
        # A complete truth table has no statistical 90% notion of correctness:
        # every row/output must remain correct under the fresh schedule.
        logic_threshold = 0.999
        result['holdouts'] = holdouts
        result['holdout'] = mean_ho
        result['verdict'] = classify(
            train if train is not None else mean_ho,
            mean_ho, logic_threshold, kind=kind)
        return result
    spec = oracle_spec_for(target)
    if spec is None:
        result['verdict'] = classify(train or 0.0, None, threshold, kind=kind)
        return result

    # fit once (under the run's physics), reuse for every held-out seed
    fitted = fit_readout(genome, carry_physics(target, spec()), backend=backend)
    if fitted is None:
        result['holdouts'] = [0.0] * len(seeds)
        result['holdout'] = 0.0
        result['verdict'] = classify(train or 0.0, 0.0, threshold, kind=kind)
        return result
    holdouts = [holdout_score(genome, spec, backend=backend, seed=s, fitted=fitted,
                              physics_from=target)
                for s in seeds]
    mean_ho = sum(holdouts) / len(holdouts)
    result['holdouts'] = holdouts
    result['holdout'] = mean_ho
    result['verdict'] = classify(train if train is not None else mean_ho,
                                 mean_ho, threshold, kind=kind)
    return result
