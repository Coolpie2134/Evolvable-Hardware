"""One fixed-target graph for the complete fresh 600-generation campaign."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib-config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=ROOT/'results/half_adder_600gen_10seed_20260905'


def main():
    d=json.loads((OUT/'benchmark.json').read_text());assert d['complete']
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':11})
    fig,ax=plt.subplots(figsize=(14,8));fig.subplots_adjust(left=.09,right=.97,top=.96,bottom=.23)
    rows=[];curves=[];rng=np.random.default_rng(202609080)
    for backend,name,color in zip(('nervous','fnv','lut'),('Nervous net','FNV','LUT'),('#3078a8','#bf6b19','#198274')):
        c=next(c for c in d['cells'] if c['backend']==backend)
        assert len(c['seeds'])==10
        values=[]
        for s in c['seeds']:
            assert not s['error'] and not s['stopped_early']
            assert [h['absolute_generation'] for h in s['history']]==list(range(601))
            v=[h['population_mean'] for h in s['history']];values.append(v)
            rows.append({'backend':backend,'seed':s['seed'],'trained':s['trained'],'certified':s['certified'],
                         **{f'population_mean_g{g}':v[g] for g in (0,50,100,200,400,600)}})
        a=np.array(values);mean=a.mean(axis=0)
        resamples=a[rng.integers(0,10,size=(2000,10))].mean(axis=1)
        lo,hi=np.quantile(resamples,[.025,.975],axis=0)
        ax.fill_between(range(601),lo,hi,color=color,alpha=.15)
        ax.plot(range(601),mean,color=color,lw=2,label=f'{name} (10 runs)')
        curves.extend({'backend':backend,'generation':g,'mean':float(mean[g]),'bootstrap_low':float(lo[g]),'bootstrap_high':float(hi[g])} for g in range(601))
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16);ax.set_axisbelow(True)
    ax.set_xlim(0,600);ax.set_ylim(0,1.025);ax.set_xticks([0,50,100,200,300,400,500,600])
    ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
    ax.set_xlabel('Evolutionary generation');ax.set_ylabel('Population mean fitness, averaged across independent runs')
    ax.legend(frameon=False,loc='lower right')
    fig.text(.09,.115,'Fig. 1. Temporal half adder: 600 generations of developmental evolution.',fontsize=13)
    fig.text(.09,.077,'10 independent runs per substrate · population 60 · 600 generations · no restarts or early stopping',fontsize=10)
    fig.text(.09,.043,'Lines: mean population fitness across runs. Shading: pointwise 95% bootstrap intervals (2,000 whole-run resamples).',fontsize=10)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'population_mean_600gen.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    for name,data in [('run_summary.csv',rows),('curve_data.csv',curves)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Temporal half adder: fresh 600-generation confirmation','',
        'Thirty complete fresh runs, ten per developmental substrate, population 60, 600 generations, one try, no time cap and no stopping after a solve. Every run includes all 601 population observations. This is a new cohort, not an extension or pooling of the old selected two-seed pilot.','',
        '| Substrate | Gen 0 | Gen 200 | Gen 400 | Gen 600 | Training-perfect runs | Standard certified |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for b in ('nervous','fnv','lut'):
        rs=[r for r in rows if r['backend']==b]
        vals=[np.mean([r[f'population_mean_g{g}'] for r in rs]) for g in (0,200,400,600)]
        lines.append('| '+b+' | '+' | '.join(f'{x:.1%}' for x in vals)+f" | {sum(r['trained'] for r in rs)}/10 | {sum(r['certified'] for r in rs)}/10 |")
    lines+=['','The primary outcome is the arithmetic mean of run-level survivor-population means. Bootstrap intervals resample ten complete run trajectories with replacement (2,000 resamples, fixed random seed); they are pointwise intervals, not a simultaneous confidence band. Neither smoothing nor per-curve normalization is used.','',
        'FNV/LUT retain top parents and offspring after finding a training-perfect individual. A perfect survivor-population mean does not imply that every new mutant solves. Training perfection and standard held-out certification are separate outcomes. Longer search is not assumed to guarantee a solution.','',
        'The source configurations were checked against unseen_behavior_pilot_v2_2seed_200gen.json. Only the generation budget and seed cohort change scientifically; independent jobs are scheduled two at a time, eight evaluation workers each. Each per-seed benchmark JSON records its actual seed base and complete configuration. Source runtime files are frozen and checked unchanged.','',
        'Artifacts: benchmark.json combines all verified jobs; jobs/ contains raw benchmarks, histories, champion indices and populations; plan.json and source_snapshot.zip preserve protocol and code. run_summary.csv and curve_data.csv contain the plotted numbers.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'figure_manifest.json').write_text(json.dumps({'benchmark_sha256':hashlib.sha256((OUT/'benchmark.json').read_bytes()).hexdigest(),'runs':30,'generation_records':18030},indent=2))
    print('Final graph and report written.',flush=True)


if __name__=='__main__':main()
