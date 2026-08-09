from __future__ import annotations

import dataclasses
import json
import os
import pickle
import tempfile

FORMAT = 'evohw-checkpoint-v2'

_FIELDS = {
    # Evolvable I/O alleles. ``io_limit`` is retained in the wire format only
    # for old-checkpoint compatibility; every save/load pins it to one.
    'snn': ('state_n', 'state_s', 'state_e', 'state_w', 'self_in', 'self_out',
            'limit', 'tag', 'io_limit', 'io_selector'),
    # Nervous no longer writes the retired per-gene I/O alleles (tag, io_limit,
    # io_selector, io_kind): its I/O is an evolved input LAYOUT on the genome
    # plus fitted output probes, so those fields encode nothing. Old documents
    # carry their own 'gene_fields' list and still load - the fields are read
    # only to identify and cleanly reject an incompatible retired-placement run.
    'nervous': ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'),
    'lut': ('ctx_n', 'ctx_e', 'ctx_s', 'ctx_w', 'self_in', 'self_out', 'tag',
            'io_limit', 'io_selector', 'io_kind'),
    'fnv': ('ctx_l', 'ctx_r', 'ctx_d', 'self_in', 'self_out'),
}


def _genome_types(backend):
    if backend == 'snn':
        from substrates.snn.genome import Gene, Chromosome, Genome
    elif backend == 'nervous':
        from substrates.nervous.genome import HexGene as Gene, Chromosome, Genome
    elif backend == 'lut':
        from substrates.lut.genome import LutGene as Gene, Chromosome, Genome
    elif backend == 'fnv':
        from substrates.fnv.genome import ContextGene as Gene, Chromosome, Genome
    else:
        raise ValueError('unknown backend: %s' % backend)
    return Gene, Chromosome, Genome


def _fnv_gene_document(gene):
    from substrates.fnv.genome import ControlGene, InputGene, OutputGene
    if isinstance(gene, InputGene):
        return {'id': int(gene.gene_id), 'branch': int(gene.branch_id),
                'pad': [int(gene.distance), int(gene.bearing)]}
    if isinstance(gene, OutputGene):
        return {'id': int(gene.gene_id), 'branch': int(gene.branch_id),
                'output': [str(gene.role), int(gene.distance),
                           int(gene.bearing)]}
    if isinstance(gene, ControlGene):
        return {'id': int(gene.gene_id), 'branch': int(gene.branch_id),
                'control': [int(gene.tolerance), int(gene.telomere)]}
    return {'id': int(gene.gene_id), 'branch': int(gene.branch_id),
            'context': [int(gene.ctx_l), int(gene.ctx_r), int(gene.ctx_d),
                        int(gene.self_in)],
            'out': int(gene.self_out), 'depth': int(gene.depth)}


def genome_to_dict(genome, backend):
    from substrates.nervous.tritile import _CHAN_BITS
    # FNV has exactly one encoding, so its growth genes are always context rules.
    constructive_fnv = backend == 'fnv'
    fields = (() if constructive_fnv else _FIELDS[backend])
    def split_for(chromosome):
        count = len(chromosome.genes)
        if constructive_fnv:
            # The branched centromere may sit at either end: an arm is allowed
            # to be empty, and clamping it inward would silently move rules into
            # the other branch and change what they mean.
            return max(0, min(int(chromosome.split), count))
        return (0 if count < 2 else
                max(1, min(int(chromosome.split), count - 1)))
    sd = getattr(genome, 'state_delays', None)     # nervous width-preserving model
    if backend == 'fnv':
        from substrates.fnv.catalogue import CATALOGUE_HASH
        from substrates.fnv.genome import genome_development_version
        catalogue_hash = CATALOGUE_HASH
        development_version = genome_development_version(genome)
    else:
        catalogue_hash = None
        development_version = None
    if constructive_fnv:
        chromosome_documents = [
            {
                'tag': int(c.tag),
                'split': split_for(c),
                'wiring': False,
                # Two gene kinds: context rules, and one control gene per arm
                # carrying that arm's reach and lifespan.
                'genes': [_fnv_gene_document(g) for g in c.genes],
            }
            for c in genome.chromosomes]
    else:
        chromosome_documents = [
            {'tag': int(c.tag), 'split': split_for(c),
             'telomere': int(getattr(c, 'telomere', 1)),
             # wiring-chromosome marker for evolvable I/O binding (Method B)
             'wiring': bool(getattr(c, 'wiring', False)),
             'genes': [[1 if f == 'io_limit' else int(getattr(g, f))
                        for f in fields] for g in c.genes]}
            for c in genome.chromosomes]
    return {
        'tag': int(genome.tag), 'gene_fields': list(fields),
        'catalogue_hash': catalogue_hash,
        'development_version': development_version,
        'encoding': getattr(genome, 'encoding', None) if backend == 'fnv' else None,
        'next_gene_id': (
            int(getattr(genome, 'next_gene_id', 1))
            if constructive_fnv else None),
        'seed_state': (
            [int(value) for value in genome.seed_state]
            if backend == 'lut' and getattr(genome, 'seed_state', None)
            is not None else None),
        'provenance': (
            str(getattr(genome, 'provenance', ''))
            if backend == 'lut' else ''),
        'routing_patches': (
            [[int(patch.x), int(patch.y), int(patch.state)]
             for patch in (getattr(genome, 'routing_patches', None) or ())]
            if backend == 'nervous' else []),
        # Evolved input pads. Absent/null marks a FIXED-input genome, which
        # keeps reading its pads from the target - that is the documented
        # legacy load path, not a missing field to be repaired.
        'input_layout': (
            [[int(cell[0]), int(cell[1])] for cell in genome.input_layout]
            if getattr(genome, 'input_layout', None) is not None else None),
        'edge_input_layout': (
            [int(value) for value in genome.edge_input_layout]
            if backend == 'lut'
            and getattr(genome, 'edge_input_layout', None) is not None
            else None),
        'chromosomes': chromosome_documents,
        'input_chromosome': (
            {'tag': int(genome.input_chromosome.tag),
             'genes': [_fnv_gene_document(g)
                       for g in genome.input_chromosome.genes]}
            if backend == 'fnv'
            and getattr(genome, 'input_chromosome', None) is not None
            else None),
        'output_layout': (
            [[str(role), [int(cell[0]), int(cell[1])]]
             for role, cell in (getattr(genome, 'output_layout', ()) or ())]
            if backend == 'fnv' else None),
        'output_chromosome': (
            {'tag': int(genome.output_chromosome.tag),
             'genes': [_fnv_gene_document(g)
                       for g in genome.output_chromosome.genes]}
            if backend == 'fnv'
            and getattr(genome, 'output_chromosome', None) is not None
            else None),
        'state_delays': ([float(x) for x in sd] if sd else None),
        'arch': getattr(genome, 'arch', 'single'),      # nervous tile architecture
        # Tri channel field width. Absent (or 4) marks the pre-OR layout, which
        # genome_from_dict widens on load; stamping 5 stops a re-widen.
        'tri_channel_bits': _CHAN_BITS,
    }


def genome_from_dict(data, backend):
    if backend == 'fnv':
        from substrates.fnv.catalogue import verify_catalogue_hash
        from substrates.fnv.genome import (
            BRANCHED_ENCODING, DEVELOPMENT_VERSION, Chromosome, ContextGene,
            ControlGene, Genome, InputGene, OutputGene, sync_input_layout,
            sync_output_layout,
        )
        # The retired associative and constructive encodings stored genes that
        # this one would silently misread, so the version stamp rejects them
        # rather than loading a genome that means something else.
        verify_catalogue_hash(data.get('catalogue_hash'))
        if (data.get('encoding') != BRANCHED_ENCODING
                or int(data.get('development_version', -1))
                != DEVELOPMENT_VERSION):
            raise ValueError(
                "FNV checkpoint is not a %s genome (development version %d): "
                "found encoding %r version %r" % (
                    BRANCHED_ENCODING, DEVELOPMENT_VERSION,
                    data.get('encoding'), data.get('development_version')))
        encoding = BRANCHED_ENCODING
        chromosomes = []
        for item in data.get('chromosomes', ()):
            genes = []
            for row in item.get('genes', ()):
                if 'control' in row:
                    tolerance, telomere = row['control']
                    genes.append(ControlGene(
                        gene_id=int(row['id']), tolerance=int(tolerance),
                        telomere=int(telomere),
                        branch_id=int(row.get('branch', row['id']))))
                    continue
                ctx_l, ctx_r, ctx_d, self_in = row['context']
                genes.append(ContextGene(
                    gene_id=int(row['id']),
                    ctx_l=int(ctx_l), ctx_r=int(ctx_r), ctx_d=int(ctx_d),
                    self_in=int(self_in), self_out=int(row['out']),
                    branch_id=int(row.get('branch', row['id'])),
                    depth=int(row.get('depth', -1))))
            split = max(0, min(int(item.get('split', 0)), len(genes)))
            chromosomes.append(Chromosome(
                genes=genes, split=split, tag=int(item.get('tag', 0))))
        layout = data.get('input_layout')
        max_id = max((gene.gene_id for chromosome in chromosomes
                      for gene in chromosome.genes), default=0)
        max_id = max(max_id, max(
            (int(row['id'])
             for row in ((data.get('input_chromosome') or {}).get('genes')
                         or ())),
            default=0))
        max_id = max(max_id, max(
            (int(row['id'])
             for row in ((data.get('output_chromosome') or {}).get('genes')
                         or ())),
            default=0))
        roots = data.get('output_chromosome')
        if roots is None:
            raise ValueError('FNV v6 checkpoint has no output chromosome')
        output_genes = []
        for row in roots.get('genes', ()):
            role, distance, bearing = row['output']
            output_genes.append(OutputGene(
                gene_id=int(row['id']), role=str(role),
                distance=int(distance), bearing=int(bearing),
                branch_id=int(row['branch'])))
        genome = Genome(
            chromosomes=chromosomes,
            tag=int(data.get('tag', 0)),
            input_layout=(
                tuple((int(cell[0]), int(cell[1])) for cell in layout)
                if layout is not None else None),
            output_chromosome=Chromosome(
                genes=output_genes, split=0, tag=int(roots.get('tag', 0))),
            encoding=encoding,
            next_gene_id=max(
                max_id + 1, int(data.get('next_gene_id') or 1)),
        )
        pads = data.get('input_chromosome')
        if pads is not None:
            genome.input_chromosome = Chromosome(
                genes=[
                    InputGene(gene_id=int(row['id']),
                              distance=int(row['pad'][0]),
                              bearing=int(row['pad'][1]),
                              branch_id=int(row.get('branch', row['id'])))
                    for row in pads['genes']],
                split=0, tag=int(pads.get('tag', 0)))
            sync_input_layout(genome)
        sync_output_layout(genome)
        return genome
    Gene, Chromosome, Genome = _genome_types(backend)
    fields = tuple(data.get('gene_fields') or _FIELDS[backend])
    chroms = []
    for item in data['chromosomes']:
        genes = []
        for row in item['genes']:
            values = dict(zip(fields, map(int, row)))
            # Historical 0/all and multi-site limits are retired. Keeping the
            # field readable avoids breaking old files while guaranteeing that
            # every loaded port has single-cell semantics.
            if 'io_limit' in fields:
                values['io_limit'] = 1
            genes.append(Gene(**values))
        split = (0 if len(genes) < 2 else
                 max(1, min(int(item.get('split', 0)), len(genes) - 1)))
        chroms.append(Chromosome(
            genes=genes, split=split,
            tag=int(item.get('tag', 0)), telomere=int(item.get('telomere', 1)),
            # 'sex' is the flag's retired spelling - readable, never written
            **({'wiring': bool(item.get('wiring', item.get('sex', False)))}
               if backend != 'fnv' else {})))
    seed_state = data.get('seed_state') if backend == 'lut' else None
    backend_fields = {}
    if backend == 'lut':
        layout = data.get('input_layout')
        backend_fields.update({
            'seed_state': (
                tuple(int(value) & 0xFFFF for value in seed_state)
                if seed_state is not None else None),
            'provenance': str(data.get('provenance', '')),
            'input_layout': (
                tuple((int(cell[0]), int(cell[1])) for cell in layout)
                if layout is not None else None),
            'edge_input_layout': (
                tuple(int(value) for value in data['edge_input_layout'])
                if data.get('edge_input_layout') is not None else None),
        })
    genome = Genome(
        chromosomes=chroms, tag=int(data.get('tag', 0)),
        **backend_fields)
    # 'state_widths' appears in checkpoints written before width evolution was
    # retired. It is deliberately ignored: the vector no longer exists on the
    # genome and no engine reads it (RunConfig.from_dict moves such runs onto
    # the paper's fixed-width 'uniform' node).
    sd = data.get('state_delays')                  # evolved-delay preservation
    if sd and backend == 'nervous':
        genome.state_delays = [float(x) for x in sd]
    if backend == 'nervous':                       # tri3 vs single tile decode
        from substrates.nervous.genome import RoutingPatch
        genome.routing_patches = [
            RoutingPatch(int(row[0]), int(row[1]), int(row[2]))
            for row in data.get('routing_patches', ())
            if len(row) >= 3]
        layout = data.get('input_layout')
        if layout is not None:
            genome.input_layout = tuple(
                (int(cell[0]), int(cell[1])) for cell in layout)
        genome.arch = data.get('arch', 'single')
        # Tri channels were 4-bit AND-only before the OR twins were added. A
        # legacy checkpoint's packed bits would be re-cut at the new 5-bit field
        # boundaries and decode as different channels, so widen them. Legacy
        # configs are all 0-15 (the AND half), hence behaviour is preserved.
        if (genome.arch == 'tri3'
                and int(data.get('tri_channel_bits', 4)) == 4):
            from substrates.nervous.tritile import widen_legacy_state
            for chrom in genome.chromosomes:
                for gene in chrom.genes:
                    for field_name in ('ctx_l', 'ctx_r', 'ctx_d',
                                       'self_in', 'self_out'):
                        setattr(gene, field_name,
                                widen_legacy_state(getattr(gene, field_name)))
    return genome


def _target_to_dict(target):
    kind = 'temporal' if getattr(target, 'temporal', False) else 'logic'
    extras = {
        name: value for name, value in vars(target).items()
        if name.startswith('_sr_') or name.startswith('_retention_')
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
        from substrates.nervous.targets import TemporalTarget, Trial, OutputTerminal
        from substrates.nervous.contracts import contract_from_dict, legacy_contract
        legacy_mode = data.pop('score_mode', None)
        data['contract'] = (legacy_contract(legacy_mode, data)
                            if legacy_mode is not None and 'contract' not in data
                            else contract_from_dict(data.get('contract')))
        for obsolete in ('event_tolerance', 'waveform_tolerance',
                         'event_max_shift', 'fit_latency', 'cadence_period',
                         'cadence_tolerance', 'cadence_settle',
                         'cadence_min_events', 'stepper_min_period',
                         'stepper_max_period', 'stepper_settle',
                         'stepper_min_events', 'stepper_max_delay'):
            data.pop(obsolete, None)
        data['inputs'] = [tuple(p) for p in data['inputs']]
        data['outputs'] = [OutputTerminal(role=o['role'], pos=tuple(o['pos']))
                           for o in data['outputs']]
        data['trials'] = [Trial(
            streams=[tuple(row) for row in t['streams']], expected=t['expected'],
            expected_events=t.get('expected_events', {}),
            input_events=(None if t.get('input_events') is None else
                          [[tuple(event) for event in events]
                           for events in t['input_events']]),
            expected_intervals={
                role: [tuple(interval) for interval in intervals]
                for role, intervals in t.get('expected_intervals', {}).items()},
            case_windows=[
                (float(window[0]), float(window[1]),
                 tuple(int(bit) for bit in window[2]))
                for window in t.get('case_windows', ())])
                          for t in data['trials']]
        data['combinational_cases'] = [
            (tuple(input_bits), tuple(output_bits))
            for input_bits, output_bits in data.get(
                'combinational_cases', ())]
        target = TemporalTarget(**data)
        for name, value in item.get('extras', {}).items():
            setattr(target, name, _tuples(value))
        return target
    from substrates.snn.targets import Target, OutputTerminal
    from substrates.nervous.contracts import contract_from_dict, logic_contract
    data['contract'] = (contract_from_dict(data['contract'])
                        if data.get('contract') else logic_contract())
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
    configured_arch = getattr(
        getattr(run_config, 'ga', None), 'tile_arch', None)
    if (backend == 'nervous' and configured_arch is not None
            and getattr(genome, 'arch', 'single') != configured_arch):
        raise ValueError('checkpoint genome violates configured tile architecture')
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
                   certification=None, fitnesses=None, metadata=None):
    genomes = list(genomes)
    if fitnesses is not None:
        fitnesses = [float(fitness) for fitness in fitnesses]
        if len(fitnesses) != len(genomes):
            raise ValueError('population fitness count does not match genome count')
    if (backend == 'nervous'
            and len({getattr(genome, 'arch', 'single')
                     for genome in genomes}) > 1):
        raise ValueError('checkpoint population mixes tile architectures')
    configured_count = getattr(
        getattr(run_config, 'ga', None), 'chromosome_count', None)
    if (configured_count is not None
            and any(len(genome.chromosomes) != configured_count
                    for genome in genomes)):
        raise ValueError('checkpoint population violates configured chromosome count')
    configured_arch = getattr(
        getattr(run_config, 'ga', None), 'tile_arch', None)
    if (backend == 'nervous' and configured_arch is not None
            and any(getattr(genome, 'arch', 'single') != configured_arch
                    for genome in genomes)):
        raise ValueError('checkpoint population violates configured tile architecture')
    _atomic_json(path, {
        'format': FORMAT + '-population', 'backend': backend,
        'valid': float(valid), 'target': _target_to_dict(target),
        'run_config': None if run_config is None else dataclasses.asdict(run_config),
        # Held-out certification verdict for the winning genome (advisory
        # provenance: was the solved fitness a real, generalising circuit?).
        'certification': certification,
        # Optional snapshot provenance.  In particular, stopped runs record
        # which fully evaluated generation supplied this fixed-file snapshot.
        'fitnesses': fitnesses,
        'metadata': metadata,
        'genomes': [genome_to_dict(g, backend) for g in genomes],
    })


RETIRED_NERVOUS_PLACEMENTS = (
    'terminal_nodes', 'tag_rank', 'wiring_chromosome', 'spatial_chromosome')


def _reject_retired_nervous_placement(backend, run_config):
    """Fail loudly on a checkpoint saved under a retired Nervous I/O strategy.

    Those bindings live in per-gene tags, a reserved wiring chromosome or
    normalised x/y anchors. None of them carry the information the coordinate
    layout needs, so any automatic conversion would be an invention rather than
    a migration - the run would silently become a DIFFERENT organism wearing the
    old fitness. Refusing is the honest outcome.
    """
    if backend != 'nervous' or run_config is None:
        return
    placement = getattr(getattr(run_config, 'ga', None), 'io_placement',
                        'fixed')
    if placement in RETIRED_NERVOUS_PLACEMENTS:
        raise ValueError(
            'retired Nervous I/O placement %r in this checkpoint. The nervous '
            'substrate now uses an evolved input layout plus fitted output '
            'probes; tag / wiring / spatial / terminal-node bindings cannot be '
            'converted into pad coordinates without inventing them, so this '
            'run cannot be loaded. (Fixed-input nervous checkpoints still load '
            'normally and keep their original wired-OR input physics.)'
            % (placement,))


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
        _reject_retired_nervous_placement(backend, run_config)
        target = _target_from_dict(doc['target'])
        setattr(target, 'pulse_config', run_config.pulse)
        if backend == 'lut':
            setattr(target, 'lut_io_mode', run_config.ga.lut_io_mode)
            setattr(
                target, '_lut_function_families',
                run_config.ga.lut_function_families)
        return {'genomes': [genome_from_dict(g, backend) for g in doc['genomes']],
                'target': target, 'backend': backend,
                'valid': doc.get('valid', 0.999), 'run_config': run_config,
                'fitnesses': doc.get('fitnesses'),
                'metadata': doc.get('metadata'),
                'certification': doc.get('certification')}
    from substrates.snn.snn import Arch
    arch_data = dict(doc['arch']) if doc.get('arch') else None
    if arch_data:
        arch_data['vth_levels'] = tuple(arch_data['vth_levels'])
        arch_data['tau_levels'] = tuple(arch_data['tau_levels'])
    arch = Arch(**arch_data) if arch_data else None
    target = _target_from_dict(doc['target'])
    from .config import RunConfig
    run_config = RunConfig.from_dict(doc.get('run_config'))
    _reject_retired_nervous_placement(backend, run_config)
    setattr(target, 'pulse_config', run_config.pulse)
    if backend == 'lut':
        setattr(target, 'lut_io_mode', run_config.ga.lut_io_mode)
        setattr(
            target, '_lut_function_families',
            run_config.ga.lut_function_families)
    elif backend == 'fnv':
        setattr(target, '_fnv_families', run_config.fnv.families)
        setattr(target, '_fnv_readout_mode', run_config.fnv.readout_mode)
    return {'best_genome': genome_from_dict(doc['genome'], backend),
            'best_fitness': float(doc['fitness']), 'target': target,
            'target_name': target.name, 'arch': arch, 'seed': doc.get('seed'),
            'backend': backend, 'run_config': run_config}
