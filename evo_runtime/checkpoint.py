from __future__ import annotations

import dataclasses
import json
import os
import pickle
import tempfile

FORMAT = 'evohw-checkpoint-v2'

_FIELDS = {
    'snn': ('state_n', 'state_s', 'state_e', 'state_w', 'self_in', 'self_out', 'limit'),
    'nervous': ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'),
    'lut': ('ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in', 'self_out'),
}


def _genome_types(backend):
    if backend == 'snn':
        from snn_evo.genome import Gene, Chromosome, Genome
    elif backend == 'nervous':
        from nv_evo.genome import HexGene as Gene, Chromosome, Genome
    elif backend == 'lut':
        from lut_evo.genome import LutGene as Gene, Chromosome, Genome
    else:
        raise ValueError('unknown backend: %s' % backend)
    return Gene, Chromosome, Genome


def genome_to_dict(genome, backend):
    fields = _FIELDS[backend]
    return {
        'tag': int(genome.tag), 'gene_fields': list(fields),
        'chromosomes': [
            {'tag': int(c.tag), 'split': int(c.split),
             'telomere': int(getattr(c, 'telomere', 1)),
             'genes': [[int(getattr(g, f)) for f in fields] for g in c.genes]}
            for c in genome.chromosomes],
    }


def genome_from_dict(data, backend):
    Gene, Chromosome, Genome = _genome_types(backend)
    fields = tuple(data.get('gene_fields') or _FIELDS[backend])
    chroms = []
    for item in data['chromosomes']:
        genes = [Gene(**dict(zip(fields, map(int, row)))) for row in item['genes']]
        chroms.append(Chromosome(
            genes=genes, split=int(item.get('split', 0)),
            tag=int(item.get('tag', 0)), telomere=int(item.get('telomere', 1))))
    return Genome(chromosomes=chroms, tag=int(data.get('tag', 0)))


def _target_to_dict(target):
    kind = 'temporal' if getattr(target, 'temporal', False) else 'logic'
    extras = {
        name: value for name, value in vars(target).items()
        if name.startswith('_sr_')
    }
    return {'kind': kind, 'data': dataclasses.asdict(target), 'extras': extras}


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _target_from_dict(item):
    data = dict(item['data'])
    if item['kind'] == 'temporal':
        from nv_evo.targets import TemporalTarget, Trial, OutputTerminal
        data['inputs'] = [tuple(p) for p in data['inputs']]
        data['outputs'] = [OutputTerminal(role=o['role'], pos=tuple(o['pos']))
                           for o in data['outputs']]
        data['trials'] = [Trial(
            streams=[tuple(row) for row in t['streams']], expected=t['expected'],
            expected_events=t.get('expected_events', {})) for t in data['trials']]
        target = TemporalTarget(**data)
        for name, value in item.get('extras', {}).items():
            setattr(target, name, _tuples(value))
        return target
    from snn_evo.targets import Target, OutputTerminal
    data['inputs'] = [tuple(p) for p in data['inputs']]
    data['outputs'] = [OutputTerminal(
        role=o['role'], pos=tuple(o['pos']),
        complement_inputs=bool(o.get('complement_inputs', False)),
        invert_spike=bool(o.get('invert_spike', False))) for o in data['outputs']]
    data['cases'] = [(tuple(a), tuple(b)) for a, b in data['cases']]
    return Target(**data)


def _atomic_json(path, document):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=directory, delete=False,
        prefix='.checkpoint-', suffix='.tmp')
    try:
        with handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def save_checkpoint(path, genome, fitness, target, arch, seed, backend, run_config=None):
    document = {
        'format': FORMAT, 'backend': backend, 'fitness': float(fitness),
        'seed': seed, 'genome': genome_to_dict(genome, backend),
        'target': _target_to_dict(target),
        'arch': None if arch is None else dataclasses.asdict(arch),
        'run_config': None if run_config is None else dataclasses.asdict(run_config),
    }
    _atomic_json(path, document)


def save_population(path, genomes, target, backend, valid, run_config=None):
    _atomic_json(path, {
        'format': FORMAT + '-population', 'backend': backend,
        'valid': float(valid), 'target': _target_to_dict(target),
        'run_config': None if run_config is None else dataclasses.asdict(run_config),
        'genomes': [genome_to_dict(g, backend) for g in genomes],
    })


def load_checkpoint(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            doc = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Read-only migration path for existing trusted local checkpoints.
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    if not str(doc.get('format', '')).startswith(FORMAT):
        raise ValueError('unsupported checkpoint format')
    backend = doc['backend']
    if 'genomes' in doc:
        from .config import RunConfig
        run_config = RunConfig.from_dict(doc.get('run_config'))
        target = _target_from_dict(doc['target'])
        setattr(target, 'pulse_config', run_config.pulse)
        return {'genomes': [genome_from_dict(g, backend) for g in doc['genomes']],
                'target': target, 'backend': backend,
                'valid': doc.get('valid', 0.999), 'run_config': run_config}
    from snn_evo.snn import Arch
    arch_data = dict(doc['arch']) if doc.get('arch') else None
    if arch_data:
        arch_data['vth_levels'] = tuple(arch_data['vth_levels'])
        arch_data['tau_levels'] = tuple(arch_data['tau_levels'])
    arch = Arch(**arch_data) if arch_data else None
    target = _target_from_dict(doc['target'])
    from .config import RunConfig
    run_config = RunConfig.from_dict(doc.get('run_config'))
    setattr(target, 'pulse_config', run_config.pulse)
    return {'best_genome': genome_from_dict(doc['genome'], backend),
            'best_fitness': float(doc['fitness']), 'target': target,
            'target_name': target.name, 'arch': arch, 'seed': doc.get('seed'),
            'backend': backend, 'run_config': run_config}
