"""Durable 30-run, 600-generation temporal-half-adder confirmation campaign.

Two independent benchmark processes run concurrently (eight evaluation workers
each). Jobs interleave substrates. Re-running resumes only complete valid jobs.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import benchmark

OUT=ROOT/'results/half_adder_600gen_10seed_20260905'
BACKENDS=('nervous','fnv','lut')
TARGET='Half adder (temporal)'
BASE=202609080


def arguments(backend,base,folder,gens=600):
    return ['--architectures',backend,'--targets',TARGET,'--seeds','1','--seed-base',str(base),
            '--gens',str(gens),'--pop','60','--workers','8','--tries','1',
            '--champion-dir',str(folder/'champions'),'--snapshot-dir',str(folder/'populations'),
            '--out',str(folder/'benchmark.json'),'--progress-every','100']


def verify_config(actual,backend):
    original=json.loads((ROOT/'results/unseen_behavior_pilot_v2_2seed_200gen.json').read_text())['config']
    for field in ('substrate','pulse','io','fnv_families','fnv_readout','escape','certification'):
        assert actual[field]==original[field],field
    for field,value in original['ga'].items():
        if field in ('resolved_tuning','resolved_max_telomere'):
            assert actual['ga'][field][backend]==value[backend],field
        else:assert actual['ga'][field]==value,field
    for field in ('population','restarts','chromosomes','workers','input_high','time_cap','stop_on_solve','record_champions'):
        assert actual['run'][field]==original['run'][field],field


def checked(path,backend,index,gens=600):
    d=json.loads(path.read_text())
    verify_config(d['config'],backend)
    assert len(d['cells'])==1 and len(d['cells'][0]['seeds'])==1
    c=d['cells'][0];s=c['seeds'][0]
    assert c['backend']==backend and c['target']==TARGET
    assert s['seed']==benchmark.cell_seed(BASE,backend,TARGET,index)
    assert not s['error'] and s['stopped_early'] in (None,'')
    assert [h['absolute_generation'] for h in s['history']]==list(range(gens+1))
    assert s['champion_index']
    return d


def aggregate():
    parser=benchmark.build_parser()
    args=parser.parse_args(['--architectures',','.join(BACKENDS),'--targets',TARGET,
        '--seeds','10','--seed-base',str(BASE),'--gens','600','--pop','60','--workers','8',
        '--champion-dir',str(OUT/'jobs')])
    args.targets=[TARGET];args.exclude=[];args.behavior_checkpoints=[]
    config=benchmark.config_record(args,list(BACKENDS))
    cells=[];files=[]
    for backend in BACKENDS:
        runs=[]
        for i in range(10):
            path=OUT/'jobs'/f'{i:02d}_{backend}'/'benchmark.json'
            if not path.exists():continue
            try:d=checked(path,backend,i)
            except (AssertionError,ValueError,KeyError):continue
            runs.append(d['cells'][0]['seeds'][0]);files.append(str(path.relative_to(ROOT)))
        cells.append(benchmark.summarise_cell({'backend':backend,'target':TARGET,'seeds':runs}))
    result={'schema':'long-half-adder-confirmation/v1','config':config,'cells':cells,'skipped':[],
        'complete':sum(c['n'] for c in cells)==30,'source_jobs':files,
        'note':'Independent per-seed benchmark jobs; algorithm/configuration identical to the selected pilot except fresh seeds and 600-generation budget.'}
    benchmark.write_json(OUT/'benchmark.json',result)
    return result


def job(backend,index):
    folder=OUT/'jobs'/f'{index:02d}_{backend}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'benchmark.json'
    if path.exists():
        try:checked(path,backend,index);return backend,index,'already_complete'
        except (AssertionError,ValueError,KeyError):pass
    command=[sys.executable,str(ROOT/'tools/benchmark.py')]+arguments(backend,BASE+index*1_000_003,folder)
    with (folder/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'{backend} seed {index}: see {folder}/run.log')
    checked(path,backend,index)
    return backend,index,'completed'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'source_manifest.json'
    if manifest.exists():
        hashes=json.loads(manifest.read_text())
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items()),'Frozen sources changed'
    else:
        # Verify the exact requested runtime configuration with a one-generation
        # smoke of each backend before committing to the long campaign.
        for backend in BACKENDS:
            folder=OUT/'smoke'/backend;folder.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(ROOT/'tools/benchmark.py')]+arguments(backend,BASE-1,folder,gens=1)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            d=json.loads((folder/'benchmark.json').read_text());verify_config(d['config'],backend)
            s=d['cells'][0]['seeds'][0];assert not s['error'] and len(s['history'])==2
        print('Smoke passed; configurations match the selected archived pilot.',flush=True)
        files=sorted([p for f in ('runtime','substrates') for p in (ROOT/f).rglob('*.py')]+
                     [ROOT/'tools/benchmark.py',Path(__file__),ROOT/'tools/plot_long_half_adder.py'])
        hashes={}
        with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in files:
                rel=p.relative_to(ROOT).as_posix();raw=p.read_bytes();hashes[rel]=hashlib.sha256(raw).hexdigest();z.writestr(rel,raw)
        manifest.write_text(json.dumps(hashes,indent=2))
        (OUT/'plan.json').write_text(json.dumps({'target':TARGET,'backends':BACKENDS,'seeds_per_backend':10,
            'seed_base':BASE,'generations':600,'population':60,'restarts':1,'stop_on_solve':False,
            'time_cap':0,'concurrent_jobs':2,'workers_per_job':8,'reference':'unseen_behavior_pilot_v2_2seed_200gen.json',
            'primary_metric':'Arithmetic mean of selected population fitness across independent runs; raw values, no smoothing or normalization.',
            'uncertainty':'Pointwise 95% bootstrap interval, resampling whole seed trajectories.',
            'secondary':'Training-perfect runs and independent standard certification at completion; no promise of eventual solving.',
            'retention':'All histories, per-generation champions, population checkpoints; all planned seeds retained, including failures.'},indent=2))
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        pending=[executor.submit(job,b,i) for i in range(10) for b in BACKENDS]
        for future in concurrent.futures.as_completed(pending):
            backend,index,status=future.result()
            d=aggregate()
            print(backend,index,status,'total',sum(c['n'] for c in d['cells']),'/30',flush=True)
    d=aggregate();assert d['complete']
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    subprocess.run([sys.executable,str(ROOT/'tools/plot_long_half_adder.py')],cwd=ROOT,check=True)
    (OUT/'completion.json').write_text(json.dumps({'complete':True,'elapsed_this_invocation_s':time.time()-start,'sources_unchanged':True},indent=2))
    print('Campaign and figures complete.',flush=True)


if __name__=='__main__':main()
