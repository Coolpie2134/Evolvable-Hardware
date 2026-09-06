"""Passive recording and confidence-interval checks; no expensive evolution."""
import json
import random
import tempfile
from pathlib import Path
from unittest import mock

from runtime.observation import ChampionRecorder
from tools.audit_champion_history import audit_targets, json_safe


def test_recorder_deduplicates_without_randomness_or_mutation():
    from types import SimpleNamespace
    genome = {'a': [1, 2]}
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch('runtime.observation.genome_to_dict',
                       side_effect=lambda g, b: g), \
            mock.patch('runtime.observation.save_checkpoint') as save:
        recorder = ChampionRecorder(tmp, 'fnv', SimpleNamespace(name='T'),
                                    None, 1, None)
        telemetry = dict(absolute_generation=0, evaluations=8,
                         nominal_evaluations=8)
        state = random.getstate()
        recorder.record(genome, .5, 1, 0, telemetry)
        assert random.getstate() == state
        record = recorder.document['records'][0]
        (Path(tmp) / record['checkpoint']).touch()
        recorder.record(genome, .5, 1, 1, telemetry)
        assert save.call_count == 1
        assert genome == {'a': [1, 2]}
        assert len(json.loads(recorder.index_path.read_text())['records']) == 2
        try:
            ChampionRecorder(tmp, 'fnv', SimpleNamespace(name='T'), None, 1, None)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing observations must not be overwritten')


def test_audit_inputs_are_fixed_and_do_not_consume_evolution_rng():
    from substrates.nervous.targets import TEMPORAL_TARGETS
    from runtime.checkpoint import _target_to_dict
    target = TEMPORAL_TARGETS['Toggle flip-flop']
    state = random.getstate()
    probes = audit_targets(target, (202609041, 202609042))
    assert random.getstate() == state
    assert _target_to_dict(probes[0]) != _target_to_dict(target)
    assert _target_to_dict(probes[0]) != _target_to_dict(probes[1])
    assert _target_to_dict(probes[0]) == _target_to_dict(
        audit_targets(target, (202609041,))[0])


def test_open_ended_intervals_have_explicit_json_representation():
    assert json_safe({'intervals': [(2.0, float('inf'))]}) == {
        'intervals': [[2.0, 'Infinity']]}


def test_frozen_trace_observer_does_not_change_score():
    from types import SimpleNamespace
    from substrates.nervous.evaluation import FittedReadout, score_frozen
    target = SimpleNamespace(outputs=[SimpleNamespace(role='Q')], inputs=[(0, 0)],
                             n_inputs=1, grid_size=7, iters=30, temporal=True)
    fitted = FittedReadout('fnv', (('Q', (0, 1)),), 0, .5)
    traces = {'Q': [[0, 1]]}
    with mock.patch('substrates.fnv.construction.grow_functional',
                    return_value={(0, 0): 1, (0, 1): 1}), \
            mock.patch('substrates.fnv.evaluation.trace_fixed_outputs',
                       return_value=traces), \
            mock.patch('substrates.nervous.scoring.score_contract',
                       return_value=(.5, [.5], 0)):
        observed = []
        assert score_frozen(None, target, fitted) == score_frozen(
            None, target, fitted, trace_observer=observed.append)
        assert observed == [traces]




def test_observation_writer_retries_transient_file_lock():
    from runtime.observation import atomic_observation
    with mock.patch('runtime.observation._atomic_json',
                    side_effect=[PermissionError('locked'), None]) as writer, \
            mock.patch('runtime.observation.time.sleep') as sleep:
        atomic_observation('unused.json', {'records': []})
        assert writer.call_count == 2
        sleep.assert_called_once_with(.05)
