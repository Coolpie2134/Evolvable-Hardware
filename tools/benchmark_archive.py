"""Verify or extract the published final benchmark without running evolution.

    python tools/benchmark_archive.py verify
    python tools/benchmark_archive.py extract
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLICATION = ROOT / 'benchmarks/final'


def archive_path(name):
    """Reject paths that could escape the requested extraction directory."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts
            or '\\' in name or ':' in name):
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path


def read_json(archive, name):
    return json.loads(archive.read(name))


def verify(directory):
    import numpy as np
    from tools.multi_target_campaign import plot_horizon, slug, trajectory

    expected_files = set()
    for line in (directory / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = directory.joinpath(*archive_path(name).parts)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        expected_files.add(name)
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                    if p.is_file() and p.name != 'checksums.sha256'}
    if actual_files != expected_files:
        raise ValueError('Publication files differ from the checksum manifest')

    plan = json.loads((directory / 'plan.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    counts = Counter()
    seen = set()
    with ZipFile(directory / 'runs.zip') as archive:
        for name in archive.namelist():
            archive_path(name)
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive members')
        names = sorted(n for n in archive.namelist() if n.endswith('/result.json'))
        for target in plan['targets']:
            cells = {c['id']: c for c in plan['cells'] if c['target'] == target}
            cohort = []
            for name in names:
                if PurePosixPath(name).parts[1] not in cells:
                    continue
                run = read_json(archive, name)
                cell, index = run['cell']['id'], run['index']
                if run['cell'] != cells[cell] or (cell, index) in seen:
                    raise ValueError(f'Duplicate or mismatched run: {name}')
                seen.add((cell, index))
                metadata = read_json(archive, str(PurePosixPath(name).parent / 'benchmark_metadata.json'))
                if metadata['config']['targets'] != [target]:
                    raise ValueError(f'Mismatched configuration: {name}')
                cohort.append(run)
                counts['recorded_runs'] += 1
                counts['history_rows'] += len(run['history'])
                for key in ('population_solved', 'trained', 'certified', 'timed_out', 'hard_timeout'):
                    counts[key] += bool(run.get(key))
            display_end = plot_horizon(cohort, plan['seeds'], plan['generations'])
            curves = json.loads((directory / 'figures' / f'{slug(target)}.json').read_text())
            for curve in curves:
                selected = [r for r in cohort if r['cell']['backend'] == curve['backend']]
                paths = [trajectory(r) for r in selected]
                horizon = min(min(map(len, paths)), display_end + 1)
                mean = np.asarray([p[:horizon] for p in paths]).mean(axis=0)
                np.testing.assert_array_equal(mean, curve['mean'])
                assumed = min((r['last_generation'] + 1 for r in selected
                               if r.get('population_solved')), default=horizon)
                if (curve['n'] != len(paths) or curve['display_end'] != display_end
                        or curve['assumed_tail_from'] != assumed):
                    raise ValueError(f'Curve coverage mismatch: {target}')
                counts['verified_curves'] += 1
        if len(names) != counts['recorded_runs']:
            raise ValueError('Unexpected records outside the planned targets')
        for name in ('plan.json', 'status.json'):
            if archive.read(name) != (directory / name).read_bytes():
                raise ValueError(f'Archive metadata mismatch: {name}')
    for skip in status['skipped']:
        key = (skip['cell']['id'], skip['index'])
        if key in seen:
            raise ValueError(f'Run both recorded and skipped: {key}')
        seen.add(key)
    expected = {(c['id'], i) for c in plan['cells'] for i in range(plan['seeds'])}
    if seen != expected:
        raise ValueError('Recorded runs and skips do not cover the campaign plan')
    for key, count in counts.items():
        if summary[key] != count:
            raise ValueError(f'Summary mismatch: {key}')
    with (directory / 'runs.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise ValueError('CSV does not contain one row per planned run')
    with ZipFile(directory / 'source.zip') as source:
        for name, expected_hash in read_json(source, 'final_source_manifest.json').items():
            if hashlib.sha256(source.read('source/' + name)).hexdigest() != expected_hash:
                raise ValueError(f'Frozen source mismatch: {name}')
    print(f'Verified {len(expected_files)} files, {counts["recorded_runs"]} runs, '
          f'{counts["history_rows"]:,} measured generations, and '
          f'{counts["verified_curves"]} curves across {len(plan["targets"])} targets.')


def extract(directory, destination):
    # A dedicated new directory protects both published data and local runs.
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}; choose a new --output directory')
    verify(directory)
    with ZipFile(directory / 'runs.zip') as archive:
        for name in archive.namelist():
            archive_path(name)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    print(f'Extracted raw records and configurations to {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'extract'))
    parser.add_argument('--archive', type=Path, default=PUBLICATION,
                        help='Publication directory (default: benchmarks/final)')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/multi_target_600gen_20260915',
                        help='New extraction directory; existing paths are never overwritten')
    args = parser.parse_args()
    try:
        if args.command == 'verify':
            verify(args.archive)
        else:
            extract(args.archive, args.output)
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        parser.exit(1, f'Benchmark archive check failed: {exc}\n')


if __name__ == '__main__':
    main()
