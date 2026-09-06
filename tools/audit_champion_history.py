"""Offline unseen-schedule evaluation with frozen training readouts.

No evolutionary search is run here. Every distinct archived champion is tested;
the resulting values are never supplied to selection or a stopping rule.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.checkpoint import _atomic_json, load_checkpoint, _target_to_dict
from runtime.observation import atomic_observation
from substrates.nervous.certification import (
    carry_physics, oracle_spec_for, _temporal_logic_holdout_target,
    _combinational_holdout_target, _static_combinational_holdout_target)
from substrates.nervous.evaluation import fit_readout, score_frozen
from substrates.nervous.scoring import score_contract

AUDIT_SEEDS = (202609041, 202609042, 202609043, 202609044, 202609045)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


def audit_targets(target, seeds):
    if getattr(target, 'temporal_logic_cases', ()):
        builder = lambda s: _temporal_logic_holdout_target(target, s)
    elif getattr(target, 'combinational_cases', ()):
        builder = lambda s: _combinational_holdout_target(target, s)
    elif not getattr(target, 'temporal', False) and getattr(target, 'cases', ()):
        builder = lambda s: _static_combinational_holdout_target(target, s)
    else:
        spec = oracle_spec_for(target)
        if spec is None:
            raise ValueError('No independent schedule generator for ' + target.name)
        builder = lambda s: carry_physics(target, spec(seed=s))
    return [builder(s) for s in seeds]


def evaluate_checkpoint(path, seeds=AUDIT_SEEDS):
    checkpoint = load_checkpoint(str(path))
    genome, target = checkpoint['best_genome'], checkpoint['target']
    config, backend = checkpoint['run_config'], checkpoint['backend']
    target.io_placement = config.ga.io_placement
    fitted = fit_readout(genome, target, backend=backend)
    unseen = audit_targets(target, seeds)
    suites = []
    for seed, probe in zip(seeds, unseen):
        captured = []
        score = (score_frozen(genome, probe, fitted,
                              trace_observer=captured.append)
                 if fitted is not None else 0.0)
        traces = captured[0] if captured else None
        valid = traces is not None and not getattr(traces, 'overflow', False)
        cases = (list(score_contract(traces, probe,
                                    alignment=fitted.alignment)[1])
                 if valid else None)
        suites.append({
            'seed': seed, 'score': float(score), 'pass': score >= 0.999,
            'case_scores': cases,
            'status': ('observed' if valid else
                       'no_training_readout' if fitted is None else
                       'overflow' if traces is not None else 'invalid_circuit'),
            'target': _target_to_dict(probe),
            'observations': None if traces is None else {
                'samples': dict(traces),
                'events': getattr(traces, 'events', {}),
                'intervals': getattr(traces, 'intervals', {}),
                'overflow': bool(getattr(traces, 'overflow', False))}})
    return json_safe({
        'checkpoint': str(path.resolve()),
        'fitted_readout': None if fitted is None else dataclasses.asdict(fitted),
        'replayed_training_score': None if fitted is None else fitted.training_score,
        'suites': suites, 'all_suites_pass': all(s['pass'] for s in suites),
        'mean_unseen_score': sum(s['score'] for s in suites) / len(suites),
        'suite_pass_fraction': sum(s['pass'] for s in suites) / len(suites)})


def audit_index(path, seeds=AUDIT_SEEDS):
    index = json.loads(path.read_text(encoding='utf-8'))
    out = path.parent / 'unseen_audit.json'
    existing = json.loads(out.read_text(encoding='utf-8')) if out.exists() else None
    if existing and existing['audit_seeds'] != list(seeds):
        raise ValueError('Existing audit uses different test seeds')
    document = existing or {
        'format': 'unseen-champion-audit-v1', 'backend': index['backend'],
        'target': index['target'], 'seed': index['seed'],
        'audit_seeds': list(seeds), 'pass_threshold': 0.999,
        'protocol': 'offline; frozen training readout; all suites must pass',
        'champions': {}}
    for record in index['records']:
        key = record['genome_sha256']
        if key not in document['champions']:
            document['champions'][key] = evaluate_checkpoint(
                path.parent / record['checkpoint'], seeds)
            document['records'] = index['records']
            atomic_observation(out, document)
    document['records'] = index['records']
    atomic_observation(out, document)
    print('%s: %d champions, %d generations' %
          (out.parent.name, len(document['champions']), len(index['records'])),
          flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    args = parser.parse_args()
    document = json.loads(args.benchmark.read_text(encoding='utf-8'))
    for cell in document['cells']:
        for seed in cell['seeds']:
            if seed.get('error'):
                raise ValueError('Cannot audit an errored run')
            audit_index(Path(seed['champion_index']))


if __name__ == '__main__':
    main()
