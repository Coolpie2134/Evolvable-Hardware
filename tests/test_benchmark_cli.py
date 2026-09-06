"""The local benchmark CLI plans matrices without remote side effects."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from tools import benchmark, benchmark_contracts
from substrates.snn.targets import TARGETS


def test_run_one_persists_generation_history():
    def fake_run(_gens, _pop, _chroms, _tries, _target, _arch, messages,
                 _stop_event, **_kwargs):
        messages.put(('gen', 1, 0, 0.25, 0.10, 0.25, 4.0, 0.05))
        messages.put(('gen', 1, 1, 0.50, 0.20, 0.50, 3.9, 0.10))
        messages.put(('done', None, 0.50))

    args = benchmark.build_parser().parse_args([
        '--gens', '1', '--pop', '4', '--tries', '1', '--quiet'])
    config = SimpleNamespace(
        pulse=None,
        ga=SimpleNamespace(io_placement='fixed',
                           lut_io_mode='source_pads'),
        fnv=SimpleNamespace(families=()))
    with tempfile.TemporaryDirectory() as tmp_path, \
            mock.patch.object(benchmark, 'run_evolution', fake_run), \
            mock.patch.object(benchmark, 'build_run_config',
                              return_value=config), \
            mock.patch.object(benchmark, 'build_arch',
                              return_value=(None, None)), \
            mock.patch.object(benchmark, 'effective_target',
                              side_effect=lambda target, _high: target):
        result = benchmark.run_one(
            'nervous', 'AND', SimpleNamespace(), args, 7, tmp_path, quiet=True)

    assert [row['generation'] for row in result['history']] == [0, 1]
    assert [row['evaluations'] for row in result['history']] == [4, 8]
    assert result['history'][1]['best_so_far'] == 0.5
    assert result['history'][1]['population_std'] == 0.1
    assert result['history'][1]['nominal_evaluations'] == 8


def test_run_one_records_actual_evaluations_and_effective_selection():
    def fake_run(_gens, _pop, _chroms, _tries, _target, _arch, messages,
                 _stop_event, **_kwargs):
        messages.put(('gen', 1, 0, 0.25, 0.10, 0.25, 4.0, 0.05, {
            'phenotype_evaluations': 3,
            'cache_hits': 1,
            'cache_misses': 3,
            'cache_evictions': 0,
            'effective_selection': 'lexicase',
        }))
        messages.put(('done', None, 0.25))

    args = benchmark.build_parser().parse_args([
        '--gens', '0', '--pop', '4', '--tries', '1', '--quiet'])
    config = SimpleNamespace(
        pulse=None,
        ga=SimpleNamespace(io_placement='fixed',
                           lut_io_mode='source_pads'),
        fnv=SimpleNamespace(families=()))
    with tempfile.TemporaryDirectory() as tmp_path, \
            mock.patch.object(benchmark, 'run_evolution', fake_run), \
            mock.patch.object(benchmark, 'build_run_config',
                              return_value=config), \
            mock.patch.object(benchmark, 'build_arch',
                              return_value=(None, None)), \
            mock.patch.object(benchmark, 'effective_target',
                              side_effect=lambda target, _high: target):
        result = benchmark.run_one(
            'nervous', 'AND', SimpleNamespace(), args, 7, tmp_path, quiet=True)

    assert result['history'][0]['evaluations'] == 3
    assert result['history'][0]['nominal_evaluations'] == 4
    assert result['history'][0]['cache_hits'] == 1
    assert result['effective_selection'] == 'lexicase'


def test_architecture_names_and_sets_expand_in_requested_order():
    assert benchmark.resolve_architectures('paper,fnv') == [
        'nervous', 'lut', 'fnv']
    assert benchmark.resolve_architectures('nv,lut,nervous') == [
        'nervous', 'fnv', 'lut']
    assert benchmark.resolve_architectures('all') == list(
        benchmark.ARCHITECTURES)


def test_contract_benchmark_dispatches_native_snn_targets_without_async_io():
    with mock.patch.object(
            benchmark_contracts, 'evolve_snn', return_value=(None, 0.75)) as evolve:
        result = benchmark_contracts._run(
            'snn', None, None, TARGETS['XOR'], 1, 2, 2, 123)
    assert result == 0.75
    dispatched = evolve.call_args.kwargs['target']
    assert dispatched.name == 'XOR'
    assert not hasattr(dispatched, 'io_placement')


def test_list_architectures_is_terminal_only():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = benchmark.main(['--list-architectures'])
    text = output.getvalue()
    assert status == 0
    assert 'Architectures' in text
    assert 'paper     nervous, lut' in text
    assert 'cellular  nervous, fnv, lut' in text


def test_report_names_numeric_lexicase_downsampling_as_an_active_escape():
    parser = benchmark.build_parser()
    args = parser.parse_args(['--lexicase-sample', '0.5'])
    args.targets = ['Full adder']
    args.exclude = []
    args.fnv_families = benchmark.parse_list(args.fnv_families)
    args.lut_function_families = benchmark.parse_list(
        args.lut_function_families)
    document = {
        'generated_utc': '2026-08-01T00:00:00+00:00',
        'config': benchmark.config_record(args, ['nervous']),
        'cells': [],
    }
    report = benchmark.render_markdown(document)
    assert '`lexicase_downsample=0.50`' in report


def test_benchmarks_skip_solver_bank_work_and_record_the_worker_limit():
    args = benchmark.build_parser().parse_args([])
    args.targets = ['Veto gate']
    args.exclude = []
    args.fnv_families = benchmark.parse_list(args.fnv_families)
    args.lut_function_families = benchmark.parse_list(
        args.lut_function_families)

    config = benchmark.build_run_config(args, 'nervous', args.chroms)
    record = benchmark.config_record(args, ['nervous'])

    assert config.ga.diversify_solvers is False
    assert config.ga.evaluation_workers == args.workers
    assert record['run']['workers'] == args.workers
    assert record['ga']['diversify_solvers'] is False


def test_fresh_ga_defaults_are_resolved_per_substrate_and_overridable():
    args = benchmark.build_parser().parse_args([])
    args.fnv_families = benchmark.parse_list(args.fnv_families)
    args.lut_function_families = benchmark.parse_list(
        args.lut_function_families)

    nervous = benchmark.build_run_config(args, 'nervous', 2).ga
    fnv = benchmark.build_run_config(args, 'fnv', 2).ga
    assert (nervous.mean_mutations, nervous.immigrant_fraction,
            nervous.tournament_size, nervous.elite_count,
            nervous.mutation_decay, nervous.stagnation_beta) == (
                4.0, 0.08, 4, 5, 0.997, 1.0)
    assert (fnv.mean_mutations, fnv.immigrant_fraction,
            fnv.tournament_size, fnv.elite_count,
            fnv.mutation_decay, fnv.stagnation_beta) == (
                2.0, 0.12, 3, 3, 1.0, 2.0)

    explicit = benchmark.build_parser().parse_args([
        '--mutations', '7', '--immigrants', '0.2', '--tournament', '6',
        '--elites', '2', '--anneal', '0.99', '--plateau-beta', '0.5'])
    explicit.fnv_families = benchmark.parse_list(explicit.fnv_families)
    explicit.lut_function_families = benchmark.parse_list(
        explicit.lut_function_families)
    configured = benchmark.build_run_config(explicit, 'fnv', 2).ga
    assert (configured.mean_mutations, configured.immigrant_fraction,
            configured.tournament_size, configured.elite_count,
            configured.mutation_decay, configured.stagnation_beta) == (
                7.0, 0.2, 6, 2, 0.99, 0.5)


def _seed(gen):
    """One seed record shaped like the ones run_one() emits."""
    solved = gen is not None
    return {
        'seed': 1, 'first_solved_gen': gen,
        'best': 1.0 if solved else 0.9062,
        'train': 1.0 if solved else 0.9062,
        'holdout': 1.0 if solved else 0.9062,
        'holdouts': [1.0] if solved else [0.9062],
        'verdict': 'CERTIFIED' if solved else 'BELOW THRESHOLD 1.00',
        'category': 'certified' if solved else 'below',
        'certified': solved, 'trained': solved,
        'generations': 150, 'elapsed_s': 1.0, 'error': None,
    }


def _cell(solve_gens, n, target='Full adder'):
    """A cell carrying what summarise_cell and the caveat both read."""
    seeds = [_seed(gen) for gen in solve_gens]
    seeds += [_seed(None)] * (n - len(solve_gens))
    return {'backend': 'fnv', 'target': target, 'n': n, 'seeds': seeds,
            'solve_gens': sorted(solve_gens)}


def test_budget_caveat_flags_a_rate_still_rising_at_the_cutoff():
    # Seeds solving for the first time at generation 140 of 150, while two
    # never solved, means --gens truncated the measurement.
    cell = _cell([18, 26, 32, 94, 110, 140], 8)
    assert benchmark.budget_caveat(cell, 150) == 'truncated'


def test_budget_caveat_is_silent_when_every_seed_solved():
    # No unsolved seed means nothing was cut off, however late the last solve.
    cell = _cell([18, 140], 2)
    assert benchmark.budget_caveat(cell, 150) is None


def test_budget_caveat_is_silent_when_solves_finish_early():
    # Unsolved seeds plus only early solves is a real plateau, not truncation.
    cell = _cell([12, 20], 8)
    assert benchmark.budget_caveat(cell, 150) is None


def test_budget_caveat_reports_a_zero_solve_cell_as_bounding_nothing():
    # The case a "latest solve" rule cannot see, and the one most often
    # misread as a structural limit.
    assert benchmark.budget_caveat(_cell([], 1), 20) == 'no-solves'


def test_render_markdown_shows_solve_generations_and_the_caveat():
    args = benchmark.build_parser().parse_args(['--gens', '150'])
    args.fnv_families = benchmark.parse_list(args.fnv_families)
    args.lut_function_families = benchmark.parse_list(
        args.lut_function_families)
    cell = _cell([18, 26, 32, 94, 110, 140], 8)
    document = {
        'generated_utc': '2026-08-03T00:00:00+00:00',
        'config': benchmark.config_record(args, ['fnv']),
        'cells': [cell],
    }
    report = benchmark.render_markdown(document)
    assert '18/94/140' in report
    assert 'Budget caveats' in report
    assert 'lower bound' in report
