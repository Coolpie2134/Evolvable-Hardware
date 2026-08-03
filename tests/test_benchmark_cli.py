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
