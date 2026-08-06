"""The local benchmark CLI plans matrices without remote side effects."""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import tempfile

from tools import benchmark


def test_architecture_names_and_sets_expand_in_requested_order():
    assert benchmark.resolve_architectures('paper,fnv') == [
        'nervous', 'lut', 'fnv']
    assert benchmark.resolve_architectures('nv,lut,nervous') == [
        'nervous', 'fnv', 'lut']
    assert benchmark.resolve_architectures('all') == list(
        benchmark.ARCHITECTURES)


def test_cli_contains_no_version_control_or_remote_publication_hooks():
    source = inspect.getsource(benchmark)
    assert 'GITHUB_' not in source
    assert 'subprocess' not in source
    help_text = benchmark.build_parser().format_help()
    assert '--architectures' in help_text
    assert '--list-architectures' in help_text
    assert '--fail-under' not in help_text
    assert '--baseline' not in help_text


def test_list_architectures_is_terminal_only():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = benchmark.main(['--list-architectures'])
    text = output.getvalue()
    assert status == 0
    assert 'Architectures' in text
    assert 'paper     nervous, lut' in text
    assert 'cellular  nervous, fnv, lut' in text


def test_dry_run_expands_architecture_target_matrix_without_evolving_or_writing():
    with tempfile.TemporaryDirectory() as directory:
        output_path = os.path.join(directory, 'must-not-exist.json')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = benchmark.main([
                '--architectures', 'paper',
                '--targets', 'AND,Full adder',
                '--seeds', '2',
                '--gens', '3',
                '--pop', '4',
                '--dry-run',
                '--out', output_path,
            ])
        text = output.getvalue()
        assert status == 0
        assert '4 cell(s) x 2 seed(s) = 8 run(s)' in text
        assert text.count('nervous') >= 2
        assert text.count('lut') >= 2
        assert 'Full adder' in text
        assert not os.path.exists(output_path)


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


def test_logic_temporal_selector_picks_only_the_derived_twins():
    names, _unsupported = benchmark.resolve_target_names(
        ['logic-temporal'], 'nervous', None)
    assert names, 'logic-temporal selected nothing'
    assert all(name.endswith('(temporal)') for name in names)
    # They are real temporal targets, so the broad keyword still covers them.
    broad, _ = benchmark.resolve_target_names(['temporal'], 'nervous', None)
    assert set(names) <= set(broad)
    combinational, _ = benchmark.resolve_target_names(
        ['combinational'], 'nervous', None)
    assert len(names) == len(combinational)
