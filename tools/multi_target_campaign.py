"""Resumable, capped multi-target campaign. Run, --plot, or --smoke.

Create results/multi_target_600gen_20260915/STOP to finish active jobs and pause.
Remove STOP and rerun to resume. Interrupted attempts are retained and restarted.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark
from runtime.observation import atomic_observation

OUT = ROOT / 'results/multi_target_600gen_20260915'
BACKENDS = ('nervous', 'fnv', 'lut')
BASE = 202609150
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def slug(target):
    return re.sub(r'[^a-z0-9]+', '-', target.lower()).strip('-')


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        return json.loads((OUT / 'plan.json').read_text())
    catalogues = {b: list(benchmark.targets_for_backend(b, 'paper_analog')) for b in BACKENDS}
    preferred = ['AND (temporal)', 'XOR (temporal)', 'Half adder (temporal)',
                 'Toggle flip-flop', 'Echo (delay 3)', 'Oscillator']
    targets = list(dict.fromkeys(preferred + sorted(set(sum(catalogues.values(), [])))))
    cells = [{'target': t, 'backend': b, 'id': slug(t) + '_' + b}
             for t in targets for b in BACKENDS if t in catalogues[b]]
    source = OUT / 'source'
    hashes = {}
    for package in ('runtime', 'substrates', 'tools', 'ui'):
        for path in (ROOT / package).rglob('*.py'):
            rel = path.relative_to(ROOT)
            dest = source / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_observation(OUT / 'source_manifest.json', hashes)
    plan = dict(schema=1, targets=targets, cells=cells, seeds=10, generations=600,
                population=60, workers=4, concurrent_jobs=2, time_cap_s=600,
                hard_cap_s=900, seed_base=BASE, stop_on_population_solve=True,
                slow_policy='Skip remaining seeds when first two attempts both time out below generation 50.',
                plotting='Carry raw population-perfect terminal observations to 600; truncate each fixed cohort at its shortest other history. Never discard failed runs from a cohort.')
    atomic_observation(OUT / 'plan.json', plan)
    return plan


def arguments(cell, index, folder, smoke=False):
    return ['--architectures', cell['backend'], '--targets', cell['target'],
            '--seeds', '1', '--seed-base', str(BASE + index * 1_000_003),
            '--gens', '1' if smoke else '600', '--pop', '60', '--workers', '4',
            '--tries', '1', '--time-cap', '600', '--stop-on-population-solve',
            '--live-history', str(folder / 'live_history.json'),
            '--champion-dir', str(folder / 'champions'),
            '--snapshot-dir', str(folder / 'populations'),
            '--out', str(folder / 'benchmark.json'), '--progress-every', '50']


def job(cell, index, smoke=False):
    parent = OUT / ('smoke' if smoke else 'jobs') / cell['id'] / f'{index:02d}'
    parent.mkdir(parents=True, exist_ok=True)
    if (parent / 'result.json').exists():
        previous = json.loads((parent / 'result.json').read_text())
        shutil.copy2(parent / 'result.json', OUT / previous['folder'] / 'result.json')
    folder = parent / f'attempt_{len(list(parent.glob("attempt_*"))) + 1:02d}'
    folder.mkdir()
    started = time.time()
    command = [sys.executable, str(OUT / 'source/tools/benchmark.py')] + arguments(cell, index, folder, smoke)
    atomic_observation(parent / 'active.json', dict(command=command, started=started, folder=str(folder)))
    timed_out = False
    with (folder / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=OUT / 'source', stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=HIDDEN)
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, creationflags=HIDDEN)
            process.wait(timeout=30)
    result = dict(cell=cell, index=index, folder=str(folder.relative_to(OUT)),
                  elapsed_s=time.time()-started, returncode=process.returncode,
                  hard_timeout=timed_out, history=[], error=None)
    try:
        if (folder / 'benchmark.json').exists():
            document = json.loads((folder / 'benchmark.json').read_text())
            c = document['cells'][0]
            assert c['backend'] == cell['backend'] and c['target'] == cell['target']
            seed = c['seeds'][0]
            assert seed['seed'] == benchmark.cell_seed(BASE, cell['backend'], cell['target'], index)
            result.update(seed)
        elif (folder / 'live_history.json').exists():
            result['history'] = json.loads((folder / 'live_history.json').read_text())['history']
            result['error'] = 'Hard timeout' if timed_out else 'Process exited without final result'
        else:
            result['error'] = 'Process exited without history'
    except Exception as exc:
        result['error'] = repr(exc)
    result['population_solved'] = (not result['error'] and bool(result['history'])
        and result['history'][-1].get('population_mean_raw', -1) >= 1.0)
    result['last_generation'] = max((h['absolute_generation'] for h in result['history']), default=-1)
    result['timed_out'] = timed_out or 'time cap' in str(result.get('stopped_early'))
    atomic_observation(parent / 'result.json', result)
    return result


def records():
    return [json.loads(p.read_text()) for p in sorted((OUT / 'jobs').glob('*/*/result.json'))]


def trajectory(result):
    """Keep missing generations missing; only an explicit raw-mean solve fills tails."""
    values = {h['absolute_generation']: h['population_mean'] for h in result['history']}
    if result.get('population_solved') and values:
        last = max(values)
        assert values[last] == 1.0
        values.update({g: 1.0 for g in range(last + 1, 601)})
    end = 0
    while end in values and end <= 600:
        end += 1
    return [values[g] for g in range(end)]


def plot_horizon(runs, seeds=10, generations=600):
    """Crop only complete, population-perfect cohorts for all three substrates."""
    solves = []
    for backend in BACKENDS:
        cohort = [r for r in runs if r['cell']['backend'] == backend]
        if len(cohort) != seeds or {r['index'] for r in cohort} != set(range(seeds)):
            return generations
        for run in cohort:
            if run.get('error') or not run.get('population_solved') or not run['history']:
                return generations
            # Use the start of the final observed perfect plateau, not a
            # rounded score or a champion-only solve.
            solved = None
            for row in reversed(run['history']):
                if row.get('population_mean_raw', -1) < 1.0:
                    break
                solved = row['absolute_generation']
            if solved is None:
                return generations
            solves.append(solved)
    return min(generations, max(solves))


def plot(plan, only=None):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib-config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    all_runs = records()
    status_path = OUT / 'status.json'
    complete = status_path.exists() and json.loads(status_path.read_text()).get('complete', False)
    figs = OUT / 'figures'
    figs.mkdir(exist_ok=True)
    for number, target in enumerate(plan['targets'], 1):
        if only is not None and target != only:
            continue
        runs = [r for r in all_runs if r['cell']['target'] == target]
        if not runs:
            continue
        display_end = plot_horizon(runs, plan['seeds'], plan['generations'])
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(left=.09, right=.97, top=.96, bottom=.25)
        notes = []
        curves = []
        for b, label, color in zip(BACKENDS, ('Nervous net', 'FNV', 'LUT'), ('#3078a8', '#bf6b19', '#198274')):
            cohort = [r for r in runs if r['cell']['backend'] == b]
            if not cohort:
                continue
            paths = [trajectory(r) for r in cohort]
            horizon = min(min(map(len, paths)), display_end + 1)
            notes.append(f'{label}: {len(cohort)}/10 runs, through g{horizon-1}' if horizon else f'{label}: failed run without data')
            if not horizon:
                continue
            a = np.array([v[:horizon] for v in paths])
            mean = a.mean(axis=0)
            g = np.arange(horizon)
            if len(a) > 1:
                rng = np.random.default_rng(BASE)
                boot = a[rng.integers(0, len(a), (2000, len(a)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975], axis=0)
                ax.fill_between(g, lo, hi, color=color, alpha=.15)
            assumed = min((r['last_generation'] + 1 for r in cohort if r.get('population_solved')), default=horizon)
            ax.plot(g[:assumed], mean[:assumed], color=color, lw=2, label=f'{label} ({len(cohort)} runs)')
            if assumed < horizon:
                ax.plot(g[max(0, assumed-1):], mean[max(0, assumed-1):], color=color, lw=2, ls='--')
            curves.append(dict(backend=b, n=len(a), mean=mean.tolist(), assumed_tail_from=assumed, display_end=display_end))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
        ax.set(xlim=(0, max(1, display_end)), ylim=(0, 1.025), xlabel='Evolutionary generation', ylabel='Population mean fitness, averaged across independent runs')
        if display_end == 600:
            ax.set_xticks([0, 50, 100, 200, 300, 400, 500, 600])
        else:
            from matplotlib.ticker import MaxNLocator
            ticks = MaxNLocator(nbins=6, integer=True).tick_values(0, max(1, display_end))
            ax.set_xticks(sorted(set([int(t) for t in ticks if 0 <= t < display_end * .9] + [display_end])))
        ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
        if curves:
            ax.legend(frameon=False, loc='lower right')
        fig.text(.09, .17, f'Fig. {number}. {target}: developmental evolution through generation {display_end}.', fontsize=12)
        fig.text(.09, .13, 'Population 60 · 10 planned seeds per substrate · one attempt · 10-minute soft / 15-minute hard run cap', fontsize=10)
        fig.text(.09, .095, 'Mean and pointwise 95% bootstrap intervals. Dashed portions include assumed population-perfect tails.', fontsize=10)
        detail = ('All 30 runs population-perfect; axis ends at the last solve.' if display_end < 600 else
                  'Incomplete cohorts stop at their shortest history; failed runs are retained.')
        fig.text(.09, .06, detail + ('' if complete else ' Provisional campaign results.'), fontsize=10)
        fig.text(.09, .025, ' | '.join(notes), fontsize=9)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(figs / f'{slug(target)}.{ext}', dpi=170, facecolor='white')
        plt.close(fig)
        atomic_observation(figs / f'{slug(target)}.json', curves)
    links = ['# Multi-target developmental evolution campaign', '',
             ('Campaign finished.' if complete else 'Provisional figures update after each completed run.') + ' Targets with all 30 runs population-perfect end at the last solve generation. Dashed portions include assumed solved tails, not measured evolution. Training fitness is distinct from held-out certification.', '',
             'Create a STOP file here to pause after active jobs. Delete it and rerun tools/multi_target_campaign.py to resume.', '',
             f'{len(all_runs)} run attempts finished; {sum(r.get("population_solved", False) for r in all_runs)} population-perfect stops.', '']
    links += [f'- [{t}](figures/{slug(t)}.png)' for t in plan['targets'] if (figs / f'{slug(t)}.png').exists()]
    (OUT / 'README.md').write_text('\n'.join(links), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--retry-errors', action='store_true',
                        help='Retry failed runs once this invocation, preserving their prior attempts.')
    args = parser.parse_args()
    plan = initialize()
    if args.plot:
        plot(plan)
        return
    if args.smoke:
        for b in BACKENDS:
            cell = next(c for c in plan['cells'] if c['backend'] == b)
            r = job(cell, 0, smoke=True)
            assert not r['error'] and r['last_generation'] == 1, r
            print('Smoke passed:', b, flush=True)
        return
    # Windows releases this lock even if the coordinator crashes.
    import msvcrt
    lock = (OUT / 'runner.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    known = {(r['cell']['id'], r['index']): r for r in records()}
    if args.retry_errors:
        known = {key: result for key, result in known.items() if not result.get('error')}
    queue = [(c, i) for i in range(10) for c in plan['cells'] if (c['id'], i) not in known]
    active = {}
    skipped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while queue or active:
            while queue and len(active) < 2 and not (OUT / 'STOP').exists():
                cell, index = queue.pop(0)
                initial = [known.get((cell['id'], i)) for i in (0, 1)]
                if index >= 2 and all(r and r['timed_out'] and r['last_generation'] < 50 for r in initial):
                    skipped.append(dict(cell=cell, index=index, reason=plan['slow_policy']))
                    continue
                active[executor.submit(job, cell, index)] = (cell, index)
            atomic_observation(OUT / 'status.json', dict(updated=time.time(), pid=os.getpid(), finished=len(known),
                planned=len(plan['cells'])*10, queued=len(queue), active=list(active.values()), skipped=skipped,
                paused=(OUT / 'STOP').exists(), complete=not queue and not active))
            if not active:
                break
            done, _ = concurrent.futures.wait(active, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                cell, index = active.pop(future)
                r = future.result()
                known[(cell['id'], index)] = r
                print(cell['target'], cell['backend'], index, 'g', r['last_generation'], r.get('stopped_early'), r['error'], flush=True)
                try:
                    plot(plan, only=cell['target'])
                except Exception as exc:
                    print('Plot error:', repr(exc), flush=True)
    atomic_observation(OUT / 'status.json', dict(updated=time.time(), pid=os.getpid(), finished=len(known),
        planned=len(plan['cells'])*10, queued=len(queue), active=[], skipped=skipped,
        paused=(OUT / 'STOP').exists(), complete=not queue))
    if not queue:
        plot(plan)


if __name__ == '__main__':
    main()
