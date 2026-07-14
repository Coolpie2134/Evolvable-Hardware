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


def _nervous_gene_rows(values, encoding):
    """Migrate one serialized NV rule to 4-bit core-circuit rule row(s)."""
    from nv_evo.hexgrid import (GENOME_ENCODING, TILE_ENCODING,
                                unpack_tile_state)
    values = [int(value) for value in values]
    if len(values) != len(_FIELDS['nervous']):
        raise ValueError('nervous gene must contain exactly five state fields')
    if encoding == GENOME_ENCODING:
        if any(not 0 <= value < 16 for value in values):
            raise ValueError('nervous genome state outside 4-bit range')
        return [values]
    if encoding == TILE_ENCODING:
        if any(not 0 <= value <= 0xFFF for value in values):
            raise ValueError('packed nervous phenotype state outside 12-bit range')
        components = [unpack_tile_state(value) for value in values]
        return [[field[circuit] for field in components]
                for circuit in range(3)]
    if encoding is not None:
        raise ValueError('unsupported nervous genome state encoding: %s' %
                         encoding)
    # Pre-directional genomes had one 0..31 selector.  The upper bit selected
    # the removed OR extension; retain its Figure-3 wiring nibble.
    if any(not 0 <= value < 32 for value in values):
        raise ValueError('legacy nervous genome state outside 5-bit range')
    return [[value & 0xF for value in values]]


def _migrate_nervous_genome_object(genome, encoding):
    """In-place migration for trusted legacy pickle genome objects."""
    from nv_evo.hexgrid import GENOME_ENCODING, TILE_ENCODING
    fields = _FIELDS['nervous']
    if encoding == GENOME_ENCODING:
        # Pickle bypasses dataclass construction, so validate the supposedly
        # current representation explicitly before accepting it.
        for chromosome in genome.chromosomes:
            for gene in chromosome.genes:
                _nervous_gene_rows(
                    [getattr(gene, field) for field in fields], encoding)
        return
    for chromosome in genome.chromosomes:
        old_genes = list(chromosome.genes)
        new_genes = []
        for gene in old_genes:
            values = [getattr(gene, field) for field in fields]
            for row in _nervous_gene_rows(values, encoding):
                new_genes.append(type(gene)(**dict(zip(fields, row))))
        chromosome.genes = new_genes
        raw_split = int(getattr(chromosome, 'split', 0))
        if encoding == TILE_ENCODING:
            raw_split *= 3
        chromosome.split = (0 if len(new_genes) < 2 else
                            max(1, min(raw_split, len(new_genes) - 1)))


def _migrate_legacy_nervous_pickle(state):
    """Migrate trusted nervous pickles to 4-bit genes + 12-bit phenotype tiles."""
    if not isinstance(state, dict) or state.get('backend') != 'nervous':
        return state
    from nv_evo.hexgrid import (GENOME_ENCODING, TILE_ENCODING,
                                promote_legacy_state, unpack_tile_state,
                                normalize_output_channel)

    grid_encoding = state.get('state_encoding')
    genome_encoding = state.get('genome_state_encoding')
    if genome_encoding is None:
        genome_encoding = (TILE_ENCODING if grid_encoding == TILE_ENCODING
                           else None)

    genomes = []
    if state.get('best_genome') is not None:
        genomes.append(state['best_genome'])
    if state.get('genome') is not None:
        genomes.append(state['genome'])
    genomes.extend(state.get('genomes') or [])
    seen = set()
    for genome in genomes:
        if id(genome) in seen:
            continue
        seen.add(id(genome))
        _migrate_nervous_genome_object(genome, genome_encoding)
    grid = state.get('grid')
    if isinstance(grid, dict):
        if grid_encoding != TILE_ENCODING:
            state['grid'] = {pos: promote_legacy_state(value)
                             for pos, value in grid.items()}
        else:
            for value in grid.values():
                unpack_tile_state(value)  # reject corrupt packed phenotype words
    if isinstance(state.get('grid'), dict):
        if state.get('out_pos'):
            state['out_pos'] = {
                role: normalize_output_channel(state['grid'], pos)
                for role, pos in state['out_pos'].items()}
    state['state_encoding'] = TILE_ENCODING
    state['genome_state_encoding'] = GENOME_ENCODING
    return state


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
    if backend == 'nervous':
        from nv_evo.hexgrid import GENOME_ENCODING
        for chromosome in genome.chromosomes:
            for gene in chromosome.genes:
                _nervous_gene_rows(
                    [getattr(gene, field) for field in fields],
                    GENOME_ENCODING)
    def split_for(chromosome):
        count = len(chromosome.genes)
        return (0 if count < 2 else
                max(1, min(int(chromosome.split), count - 1)))
    document = {
        'tag': int(genome.tag), 'gene_fields': list(fields),
        'chromosomes': [
            {'tag': int(c.tag), 'split': split_for(c),
             'telomere': int(getattr(c, 'telomere', 1)),
             'genes': [[int(getattr(g, f)) for f in fields] for g in c.genes]}
            for c in genome.chromosomes],
    }
    if backend == 'nervous':
        document['state_encoding'] = GENOME_ENCODING
    return document


def genome_from_dict(data, backend):
    Gene, Chromosome, Genome = _genome_types(backend)
    fields = tuple(data.get('gene_fields') or _FIELDS[backend])
    encoding = data.get('state_encoding') if backend == 'nervous' else None
    if backend == 'nervous':
        from nv_evo.hexgrid import TILE_ENCODING
    chroms = []
    for item in data['chromosomes']:
        genes = []
        for row in item['genes']:
            values = list(map(int, row))
            rows = (_nervous_gene_rows(values, encoding)
                    if backend == 'nervous' else [values])
            genes.extend(Gene(**dict(zip(fields, migrated)))
                         for migrated in rows)
        raw_split = int(item.get('split', 0))
        if backend == 'nervous' and encoding == TILE_ENCODING:
            raw_split *= 3
        split = (0 if len(genes) < 2 else
                 max(1, min(raw_split, len(genes) - 1)))
        chroms.append(Chromosome(
            genes=genes, split=split,
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


def save_checkpoint(path, genome, fitness, target, arch, seed, backend,
                    run_config=None, certification=None):
    configured_count = getattr(
        getattr(run_config, 'ga', None), 'chromosome_count', None)
    if (configured_count is not None
            and len(genome.chromosomes) != configured_count):
        raise ValueError('checkpoint genome violates configured chromosome count')
    document = {
        'format': FORMAT, 'backend': backend, 'fitness': float(fitness),
        'seed': seed, 'genome': genome_to_dict(genome, backend),
        'target': _target_to_dict(target),
        'arch': None if arch is None else dataclasses.asdict(arch),
        'run_config': None if run_config is None else dataclasses.asdict(run_config),
        # Held-out verdict provenance (advisory): CERTIFIED / OVERFIT / BELOW / etc.
        'certification': certification,
    }
    _atomic_json(path, document)


def save_population(path, genomes, target, backend, valid, run_config=None,
                   certification=None):
    configured_count = getattr(
        getattr(run_config, 'ga', None), 'chromosome_count', None)
    if (configured_count is not None
            and any(len(genome.chromosomes) != configured_count
                    for genome in genomes)):
        raise ValueError('checkpoint population violates configured chromosome count')
    _atomic_json(path, {
        'format': FORMAT + '-population', 'backend': backend,
        'valid': float(valid), 'target': _target_to_dict(target),
        'run_config': None if run_config is None else dataclasses.asdict(run_config),
        # Held-out certification verdict for the winning genome (advisory
        # provenance: was the solved fitness a real, generalising circuit?).
        'certification': certification,
        'genomes': [genome_to_dict(g, backend) for g in genomes],
    })


def load_checkpoint(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            doc = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Read-only migration path for existing trusted local checkpoints.
        with open(path, 'rb') as handle:
            return _migrate_legacy_nervous_pickle(pickle.load(handle))
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
