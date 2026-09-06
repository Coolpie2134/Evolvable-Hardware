"""Passive, replayable champion history. Never evaluates or changes a genome."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .checkpoint import _atomic_json, genome_to_dict, save_checkpoint


def atomic_observation(path, document):
    """Retry transient Windows sharing locks without losing a completed run."""
    for attempt in range(8):
        try:
            _atomic_json(str(path), document)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * 2 ** min(attempt, 4))


class ChampionRecorder:
    def __init__(self, directory, backend, target, arch, seed, run_config):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backend, self.target, self.arch = backend, target, arch
        self.seed, self.run_config = seed, run_config
        self.index_path = self.directory / 'index.json'
        if self.index_path.exists():
            raise FileExistsError('Refusing to overwrite champion history: %s'
                                  % self.index_path)
        self.document = {'format': 'champion-history-v1', 'backend': backend,
                         'target': target.name, 'seed': seed, 'records': []}

    def record(self, genome, fitness, try_index, generation, telemetry):
        encoded = json.dumps(genome_to_dict(genome, self.backend),
                             sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        checkpoint = self.directory / (digest + '.json')
        if not checkpoint.exists():
            save_checkpoint(str(checkpoint), genome, fitness, self.target,
                            self.arch, self.seed, self.backend, self.run_config)
        self.document['records'].append({
            'try': int(try_index), 'generation': int(generation),
            'absolute_generation': telemetry['absolute_generation'],
            'evaluations': telemetry['evaluations'],
            'nominal_evaluations': telemetry['nominal_evaluations'],
            'training_score': float(fitness), 'genome_sha256': digest,
            'checkpoint': checkpoint.name})
        atomic_observation(self.index_path, self.document)
