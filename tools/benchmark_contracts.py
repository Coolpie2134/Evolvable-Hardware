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

from substrates.snn.targets import TARGETS                                  # noqa: E402
from substrates.snn.ga import evolve as evolve_snn                          # noqa: E402
from substrates.nervous.targets import (TEMPORAL_TARGETS,                        # noqa: E402
                            periodic_combinational_target,
                            with_io_placement)
from substrates.nervous.io_placement import IO_STRATEGIES, uses_port_chromosome  # noqa: E402
from substrates.nervous.ga import evolve_nervous                                 # noqa: E402
from substrates.nervous.pulse import PulseConfig                                 # noqa: E402
from substrates.lut.ga import evolve_lut                                    # noqa: E402


SUBSTRATES = (
    ('snn', None, None),
    ('nervous_legacy', 'single', 'pulse_delay'),
    ('nervous_digital_tri', 'tri3', 'uniform'),
    ('nervous_analog_tri', 'tri3', 'paper_analog'),
    ('lut', None, None),
)


def _replace_atomic(source, destination):
    """Replace a result despite brief OneDrive/antivirus sharing locks."""
    destination = os.path.abspath(destination)
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


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
        _replace_atomic(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def _target_rows(kind=None):
    rows = [('logic', name, target) for name, target in TARGETS.items()]
    rows.extend(('temporal', name, target)
                for name, target in TEMPORAL_TARGETS.items())
    return rows if kind is None else [row for row in rows if row[0] == kind]


def _effective_target(substrate, target):
    """The target object the substrate actually evolves against.

    The asynchronous backends never see a static truth table: the app wraps
    every combinational target in the periodic asynchronous encoding before
    handing it to the GA (see AsyncApp._targets_for_backend). Benchmarking the
    raw table instead measured a path the product does not use, and understated
    those rows badly — under an identical budget AND and OR reach 1.0 through
    the wrapper and stall at 0.83 without it.
    """
    if substrate == 'snn' or getattr(target, 'temporal', False):
        return target
    return periodic_combinational_target(target)


def _support_reason(substrate, kind, target, model):
    if substrate == 'snn':
        allowed = getattr(target, 'supported_backends', ())
        if allowed and 'snn' not in allowed:
            return 'target excludes SNN backend'
        return None
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


def _run(substrate, arch, model, target, generations, pop, chromosomes, seed,
         io_placement='fixed'):
    target = with_io_placement(copy.deepcopy(target), io_placement)
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
        '# Contract v1: bounded fitness matrix', '',
        'Generated: %s' % document.get('finished_at', document['started_at']), '',
        ('Settings: generations=%d, population=%d, chromosomes=%d, base seed=%d, '
         'I/O=%s. Fitness is the maximum observed at or before the final '
         'generation.'
         % (cfg['generations'], cfg['population'], cfg['chromosomes'],
            cfg['base_seed'], cfg.get('io_placement', 'fixed'))), '',
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
        _replace_atomic(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--pop', type=int, default=12)
    parser.add_argument('--chromosomes', type=int, default=2)
    parser.add_argument(
        '--io-placement', choices=IO_STRATEGIES, default='fixed',
        help='I/O architecture used by every measured backend')
    parser.add_argument('--seed', type=int, default=20260721)
    parser.add_argument(
        '--substrate', action='append', choices=[row[0] for row in SUBSTRATES],
        help='run only this substrate (repeatable)')
    parser.add_argument('--kind', choices=('logic', 'temporal'),
                        help='run only one target kind')
    parser.add_argument(
        '--target', action='append',
        help='run only this target by exact name (repeatable). Handy for '
             're-measuring a single row after a target/scoring fix instead of '
             'repeating a multi-hour sweep.')
    parser.add_argument('--json', default=os.path.join(
        ROOT, 'results', 'contract_v1_100gen.json'))
    parser.add_argument('--markdown', default=os.path.join(
        ROOT, 'results', 'contract_v1_100gen.md'))
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args(argv)
    if uses_port_chromosome(args.io_placement) and args.chromosomes < 3:
        parser.error('--chromosomes must be at least 3 for %s'
                     % args.io_placement)

    selected_substrates = tuple(
        row for row in SUBSTRATES
        if not args.substrate or row[0] in set(args.substrate))
    config = {
        'generations': args.generations, 'population': args.pop,
        'chromosomes': args.chromosomes, 'base_seed': args.seed,
        'io_placement': args.io_placement,
        'substrates': [name for name, _, _ in selected_substrates],
        'kind': args.kind,
        # Part of the identity of the run: a filtered sweep must never resume
        # into (or be mistaken for) a full one.
        'targets': sorted(args.target) if args.target else None,
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

    targets = _target_rows(args.kind)
    if args.target:
        wanted = set(args.target)
        targets = [row for row in targets if row[1] in wanted]
        missing = wanted - {row[1] for row in targets}
        if missing:
            raise SystemExit('unknown target name(s): %s'
                             % ', '.join(sorted(missing)))
    total = len(selected_substrates) * len(targets)
    index = 0
    for substrate, arch, model in selected_substrates:
        for kind, name, target in targets:
            index += 1
            key = substrate, kind, name
            if key in done:
                continue
            effective = _effective_target(substrate, target)
            reason = _support_reason(substrate, kind, effective, model)
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
                        substrate, arch, model, effective, args.generations,
                        args.pop, args.chromosomes, row['seed'],
                        args.io_placement))
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
