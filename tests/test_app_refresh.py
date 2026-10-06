"""UI integration contracts; no evolution or scoring logic is mocked into tests."""
import queue
from types import SimpleNamespace
from unittest.mock import Mock

from ui.app import App


def test_worker_import_of_root_launcher_does_not_import_the_gui():
    import subprocess
    import sys
    from pathlib import Path
    subprocess.run([sys.executable, '-c',
                    'import app,sys; assert "ui.app" not in sys.modules; '
                    'assert "matplotlib" not in sys.modules'],
                   cwd=Path(__file__).resolve().parents[1], check=True)


def test_generation_telemetry_does_not_stop_ui_polling():
    for telemetry in ((), ({'absolute_generation': 0, 'evaluations': 4},)):
        app = App.__new__(App)
        app.q = queue.Queue()
        app.q.put(('gen', 1, 0, .8, .4, .6, 2.0, .1) + telemetry)
        app._abs_gen = 0
        app._gen_history = []
        app._progress = Mock()
        app._redraw_fit_chart = Mock()
        app._status = Mock()
        app.target = SimpleNamespace(name='Target')
        app._worker = None
        app.root = Mock()
        app.root.winfo_exists.return_value = True
        app._poll()
        assert app._gen_history == [(1, .8, .4, .6, 2.0, .1)]
        app._redraw_fit_chart.assert_called_once_with()
        app.root.after.assert_called_once_with(60, app._poll)


def test_hidden_views_are_deferred_and_refresh_once_on_selection():
    app = App.__new__(App)
    app.root = Mock()
    app._nb = Mock()
    app._interactive = Mock()
    app._panel_refresh_job = None
    app._status = Mock()
    app._evolve_frame, app._growth_frame = 'evolution', 'growth'
    app._voltage_frame, app._genome_frame = 'voltage', 'genome'
    app._interactive_frame = 'interactive'
    app._update_truth_table = Mock()
    app._draw_growth = Mock()
    app._draw_voltages = Mock()
    app._draw_genome = Mock()
    genome = object()
    app._nb.select.return_value = 'evolution'
    app._update_all(genome, .5)
    app._draw_growth.assert_not_called()
    app._refresh_visible_panel()
    app._update_truth_table.assert_called_once_with(genome)
    app._nb.select.return_value = 'growth'
    app._refresh_visible_panel()
    app._refresh_visible_panel()
    app._draw_growth.assert_called_once_with(genome, .5)
    app._draw_voltages.assert_not_called()
    app._nb.select.return_value = 'interactive'
    app._refresh_visible_panel()
    app._interactive.sync.assert_called_once_with()


def test_export_renders_hidden_views_before_saving():
    # Real export is covered by the GUI smoke; pin the render order without I/O.
    app = App.__new__(App)
    app.best_genome = object()
    app._dirty_panels = {'growth', 'voltage', 'genome'}
    app._status = Mock()
    app._render_panel = Mock(side_effect=[None, None, ValueError('broken view')])
    app._save_pngs()
    assert [c.args[0] for c in app._render_panel.call_args_list] == ['growth', 'voltage', 'genome']
    assert 'broken view' in app._status.set.call_args.args[0]
