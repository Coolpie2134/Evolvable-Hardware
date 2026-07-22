"""Reproducible all-target/all-substrate contract-v1 evolution benchmark.

The result file is checkpointed after every row, so an interrupted exhaustive
run resumes without repeating completed 100-generation searches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from snn_evo.targets import TARGETS                                  # noqa: E402
from snn_evo.ga import evolve as evolve_snn                          # noqa: E402
from nv_evo.targets import TEMPORAL_TARGETS                          # noqa: E402
from nv_evo.ga import evolve_nervous                                 # noqa: E402
from nv_evo.pulse import PulseConfig                                 # noqa: E402
from lut_evo.ga import evolve_lut                                    # noqa: E402


SUBSTRATES = (
    ('snn', None, None),
    ('nervous_legacy', 'single', 'pulse_delay'),
    ('nervous_digital_tri', 'tri3', 'uniform'),
    ('nervous_analog_tri', 'tri3', 'paper_analog'),
    ('lut', None, None),
)


def _seed(base, target_name):
    digest = hashlib.sha256(target_name.encode('utf-8')).digest()
    return int(base) + int.from_bytes(digest[:4], 'big') % 1_000_000


def _atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=directory, delete=False,
        prefix='.contract-benchmark-', suffix='.tmp')
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def _target_rows():
    rows = [('logic', name, target) for name, target in TARGETS.items()]
    rows.extend(('temporal', name, target)
                for name, target in TEMPORAL_TARGETS.items())
    return rows


def _support_reason(substrate, kind, target, model):
    if substrate == 'snn' and kind != 'logic':
        return 'SNN has no continuous-time temporal simulator'
    if substrate.startswith('nervous'):
        allowed = getattr(target, 'supported_backends', ())
        if allowed and 'nervous' not in allowed:
            return 'target excludes nervous backend'
        models = getattr(target, 'supported_models', ())
        if models and model not in models:
            return 'target requires nervous model %s' % ', '.join(models)
        return None
    if substrate == 'lut':
        allowed = getattr(target, 'supported_backends', ())
        if allowed and 'lut' not in allowed:
            return 'target excludes LUT backend'
        return None
    return None


def _run(substrate, arch, model, target, generations, pop, chromosomes, seed):
    target = copy.deepcopy(target)
    if substrate == 'snn':
        return evolve_snn(
            generations=generations, verbose=False, n_chroms=chromosomes,
            pop=pop, target=target, seed=seed)[1]
    if substrate == 'lut':
        return evolve_lut(
            target, generations=generations, pop=pop,
            n_chroms=chromosomes, verbose=False, seed=seed)[1]
    target.pulse_config = PulseConfig(model=model)
    return evolve_nervous(
        target, generations=generations, pop=pop, n_chroms=chromosomes,
        verbose=False, seed=seed, tile_arch=arch)[1]


def _markdown(document):
    cfg = document['config']
    rows = document['results']
    lines = [
        '# Contract v1: 100-generation fitness matrix', '',
        'Generated: %s' % document.get('finished_at', document['started_at']), '',
        ('Settings: generations=%d, population=%d, chromosomes=%d, base seed=%d. '
         'Fitness is the maximum observed at or before the final generation.'
         % (cfg['generations'], cfg['population'], cfg['chromosomes'],
            cfg['base_seed'])), '',
        '| Substrate | Target | Kind | Max fitness | Status | Seconds |',
        '|---|---|---:|---:|---|---:|',
    ]
    for row in rows:
        fitness = ('%.6f' % row['max_fitness']
                   if row.get('max_fitness') is not None else '-')
        status = row['status']
        if row.get('reason'):
            status += ': ' + row['reason']
        lines.append('| %s | %s | %s | %s | %s | %.2f |' % (
            row['substrate'], row['target'].replace('|', '\\|'), row['kind'],
            fitness, status.replace('|', '\\|'), row.get('seconds', 0.0)))

    lines += ['', '## Summary', '']
    for substrate, _, _ in SUBSTRATES:
        measured = [r for r in rows
                    if r['substrate'] == substrate and r['status'] == 'ok']
        unsupported = [r for r in rows
                       if r['substrate'] == substrate
                       and r['status'] == 'unsupported']
        errors = [r for r in rows
                  if r['substrate'] == substrate and r['status'] == 'error']
        solved = sum(r['max_fitness'] >= 0.999 for r in measured)
        mean = (sum(r['max_fitness'] for r in measured) / len(measured)
                if measured else 0.0)
        lines.append('- %s: %d measured, %d solved, mean %.6f, '
                     '%d unsupported, %d errors.' % (
                         substrate, len(measured), solved, mean,
                         len(unsupported), len(errors)))
    return '\n'.join(lines) + '\n'


def _write_markdown(path, document):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=directory, delete=False,
        prefix='.contract-benchmark-', suffix='.md.tmp')
    try:
        with handle:
            handle.write(_markdown(document))
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--pop', type=int, default=12)
    parser.add_argument('--chromosomes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260721)
    parser.add_argument('--json', default=os.path.join(
        ROOT, 'results', 'contract_v1_100gen.json'))
    parser.add_argument('--markdown', default=os.path.join(
        ROOT, 'results', 'contract_v1_100gen.md'))
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)

    config = {
        'generations': args.generations, 'population': args.pop,
        'chromosomes': args.chromosomes, 'base_seed': args.seed,
        'substrates': [name for name, _, _ in SUBSTRATES],
    }
    document = {'schema': 1, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'config': config, 'results': []}
    if not args.no_resume and os.path.exists(args.json):
        with open(args.json, encoding='utf-8') as handle:
            previous = json.load(handle)
        if previous.get('config') != config:
            raise SystemExit('existing result config differs; use --no-resume '
                             'or a different output path')
        document = previous
    done = {(row['substrate'], row['kind'], row['target'])
            for row in document['results']}

    targets = _target_rows()
    total = len(SUBSTRATES) * len(targets)
    index = 0
    for substrate, arch, model in SUBSTRATES:
        for kind, name, target in targets:
            index += 1
            key = substrate, kind, name
            if key in done:
                continue
            reason = _support_reason(substrate, kind, target, model)
            row = {'substrate': substrate, 'kind': kind, 'target': name,
                   'generations': args.generations, 'population': args.pop,
                   'chromosomes': args.chromosomes,
                   'seed': _seed(args.seed, name),
                   'status': 'unsupported' if reason else 'running',
                   'reason': reason, 'max_fitness': None, 'seconds': 0.0}
            print('[%d/%d] %s / %s ...' %
                  (index, total, substrate, name), flush=True)
            started = time.perf_counter()
            if reason is None:
                try:
                    row['max_fitness'] = float(_run(
                        substrate, arch, model, target, args.generations,
                        args.pop, args.chromosomes, row['seed']))
                    row['status'] = 'ok'
                except Exception as exc:  # record the full matrix; fail at end
                    row['status'] = 'error'
                    row['reason'] = '%s: %s' % (type(exc).__name__, exc)
            row['seconds'] = round(time.perf_counter() - started, 3)
            document['results'].append(row)
            _atomic_json(args.json, document)
            _write_markdown(args.markdown, document)
            print('  %s%s (%.2fs)' % (
                row['status'],
                '' if row['max_fitness'] is None else
                ' max=%.6f' % row['max_fitness'], row['seconds']), flush=True)

    document['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    _atomic_json(args.json, document)
    _write_markdown(args.markdown, document)
    errors = [row for row in document['results'] if row['status'] == 'error']
    print('complete: %d rows, %d errors' %
          (len(document['results']), len(errors)), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
