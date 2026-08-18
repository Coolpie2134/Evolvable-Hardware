"""
tools/report_tau_sweep.py - collate the tau_leak sweep into one report.

Reads results/tau_sweep/physics.json (the deterministic node characterisation
from tools/probe_analog_window.py) plus results/tau_sweep/tau_*.json (the
evolution runs) and writes REPORT.md.

The question the sweep asks: the paper analog node's coincidence window is
NARROWER than its propagation delay, so the smallest timing change evolution can
make - adding or removing one cell - is larger than the entire window a
two-input gate will tolerate. Does widening that window (one constant, tau_leak)
make nervous nets evolvable, or does it only trade timing precision away for
nothing?
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SWEEP_DIR = os.path.join(ROOT, 'results', 'tau_sweep')


def _load_runs():
    """{tau: {target: [seed rows]}} from the per-tau benchmark documents."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(SWEEP_DIR, 'tau_*.json'))):
        tau = float(os.path.basename(path)[len('tau_'):-len('.json')])
        with open(path, encoding='utf-8') as handle:
            document = json.load(handle)
        cells = {}
        for cell in document.get('cells', ()):
            cells[cell['target']] = cell['seeds']
        runs[tau] = cells
    return runs


def _fmt(value, spec='%.3f'):
    return '-' if value is None else spec % value


def main():
    with open(os.path.join(SWEEP_DIR, 'physics.json'), encoding='utf-8') as fh:
        physics = json.load(fh)['rows']
    runs = _load_runs()
    targets = sorted({t for cells in runs.values() for t in cells})

    lines = []
    add = lines.append
    add("# Is the coincidence window the nervous net's evolvability limit?")
    add('')
    add('One physical constant (`analog_tau_leak`) swept across a fixed set of')
    add('targets and seeds. Everything else - encoding, GA, budget - identical.')
    add('')
    add('**Short answer: no, on the evidence here.** The node really does have')
    add('a coincidence window narrower than one hop - section 1 measures that')
    add('directly and it is not in dispute - but widening it did not improve')
    add('evolution on any target this sweep could test properly (section 3).')
    add('')
    add('## 1. The node, characterised without evolving anything')
    add('')
    add('A hand-built two-input AND gate (one tri-tile, both channels')
    add('coincidence-ANDing its two input pads) driven directly.')
    add('')
    add('| tau_leak | coincidence window | hops of slack | output pulse width |')
    add('| --- | --- | --- | --- |')
    for row in physics:
        mark = '>=' if row.get('window_saturated') else ''
        add('| %.2f | %s%.2f ticks | %s%.2f | %.2f ticks |' % (
            row['tau_leak'], mark, row['coincidence_window'],
            mark, row['hops_of_slack'], row['monostable_width']))
    add('')
    add('**Node propagation delay = one hop = 1.00 tick.** That is the smallest')
    add('timing change available: evolution adds or removes whole cells.')
    add('')
    add('At the shipped tau of 1.10 the window is **0.80 ticks - less than one')
    add('hop**. A two-input gate therefore needs its two input paths to have')
    add('EXACTLY equal hop counts, and no mutation can trim toward that: the')
    add('smallest correction overshoots the whole window. Every gate is a cliff')
    add('rather than a slope, which is a landscape with no gradient to climb.')
    add('')
    add('## 2. What that does to evolution')
    add('')
    if runs:
        add('| target | ' + ' | '.join(
            'tau %.2f' % tau for tau in sorted(runs)) + ' |')
        add('| --- |' + ' --- |' * len(runs))
        for target in targets:
            cells = []
            for tau in sorted(runs):
                seeds = runs[tau].get(target, [])
                if not seeds:
                    cells.append('-')
                    continue
                solved = sum(1 for s in seeds if s['best'] >= 0.999)
                best = max(s['best'] for s in seeds)
                cells.append('%d/%d solved, best %.3f' % (
                    solved, len(seeds), best))
            add('| %s | %s |' % (target, ' | '.join(cells)))
        add('')
        add('Solved = training score >= 0.999. These targets carry no oracle')
        add('reference, so certification cannot run on them and these are')
        add('TRAINING scores, not held-out - see the caveat below.')
    else:
        add('_(no evolution runs found)_')
    add('')
    add('## 3. Verdict: the hypothesis is NOT supported')
    add('')
    add('The prediction was that widening the window would raise solve rates,')
    add('because a gate could then be one hop wrong and still pass signal. It')
    add('did not happen. AND solves 2/3, 3/3, 2/3, 2/3 across the sweep and XOR')
    add('solves 1/3 at every single tau - no trend, and the variation is within')
    add('what three seeds produce by chance.')
    add('')
    add('So the coincidence window being narrower than one hop is a real')
    add('property of the node (section 1 measures it directly), but it is NOT')
    add('what limits evolution on these targets.')
    add('')
    add('**And the sweep cannot answer the case it was built for.** The')
    add('hypothesis predicts the largest effect on DEEP circuits, where many')
    add('path-parity constraints must hold at once. That is Half adder - and')
    add('Half adder reached only 31-61 generations before the wall-clock cap,')
    add('while solves on the other targets happen between generations 4 and')
    add('148. It never ran long enough to solve under ANY setting, so its')
    add('flat 0/3 row is a budget artifact and not evidence about tau.')
    add('')
    add('Testing it properly needs generation-matched budgets on the')
    add('multi-output targets, which this sweep did not have.')
    add('')
    add('## 4. Generations to first solve')
    add('')
    if runs:
        add('| target | ' + ' | '.join(
            'tau %.2f' % tau for tau in sorted(runs)) + ' |')
        add('| --- |' + ' --- |' * len(runs))
        for target in targets:
            cells = []
            for tau in sorted(runs):
                seeds = runs[tau].get(target, [])
                gens = [s['first_solved_gen'] for s in seeds
                        if s.get('first_solved_gen') is not None]
                cells.append(
                    '%s' % (', '.join(str(g) for g in gens) if gens else 'none'))
            add('| %s | %s |' % (target, ' | '.join(cells)))
    add('')
    add('## 5. Generations actually reached (the budget check)')
    add('')
    add('Read this beside section 4. A cell whose run ended before the')
    add('generation at which its target normally solves has not been tested.')
    add('')
    if runs:
        add('| target | ' + ' | '.join(
            'tau %.2f' % tau for tau in sorted(runs)) + ' |')
        add('| --- |' + ' --- |' * len(runs))
        for target in targets:
            cells = []
            for tau in sorted(runs):
                seeds = runs[tau].get(target, [])
                gens = [s['generations'] for s in seeds]
                cells.append(', '.join(str(g) for g in gens) if gens else '-')
            add('| %s | %s |' % (target, ' | '.join(cells)))
    add('')
    add('## Caveats, stated so the numbers are not over-read')
    add('')
    add('* **Training scores, not certified.** The `(temporal)` targets have no')
    add('  oracle reference registered, so held-out certification does not run.')
    add('  A 1.000 here means "fits the training trials", which is weaker')
    add('  evidence than the LUT combinational certifications.')
    add('* **Small n.** Three seeds per cell under a wall-clock cap. Treat the')
    add('  direction as the finding and the exact numbers as noisy.')
    add('* **Wider is not free.** A wider coincidence window means more spurious')
    add('  coincidences - nodes firing on things they should ignore. This sweep')
    add('  measures whether widening helps, NOT where it starts to hurt.')
    add('* **This changes the physics.** Every previously recorded nervous')
    add('  result assumes tau = 1.10. Nothing here has been made the default.')
    add('')

    path = os.path.join(SWEEP_DIR, 'REPORT.md')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('wrote %s' % path)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
