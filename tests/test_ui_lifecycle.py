"""UI timer and shutdown regressions; no display or simulation is required."""
import queue
import threading
from unittest import mock

from ui.diversity_ui import DiversityTab
from ui.interactive import InteractiveTab
from ui.designer import DesignerTab


class _Timers:
    def __init__(self):
        self.jobs = {}
        self.next_id = 0

    def after(self, delay, callback):
        self.next_id += 1
        self.jobs[self.next_id] = (delay, callback)
        return self.next_id

    def after_cancel(self, job):
        self.jobs.pop(job, None)

    def fire(self, job):
        _delay, callback = self.jobs.pop(job)
        callback()


class _Button:
    def config(self, **kwargs):
        self.text = kwargs.get('text')


def _interactive(backend='nervous'):
    tab = InteractiveTab.__new__(InteractiveTab)
    tab.parent = _Timers()
    tab._running = False
    tab._after_id = None
    tab._backend = backend
    tab._run_btn = _Button()
    tab.steps = []
    tab._step = lambda: tab.steps.append(len(tab.steps))
    return tab


def test_quick_pause_resume_leaves_only_one_playback_timer():
    tab = _interactive()
    tab._toggle_run()
    pending = tab._after_id
    assert len(tab.steps) == 1 and pending in tab.parent.jobs
    tab._toggle_run()
    assert not tab.parent.jobs and tab._run_btn.text == 'Run'
    tab._toggle_run()
    assert pending not in tab.parent.jobs
    assert len(tab.parent.jobs) == 1 and len(tab.steps) == 2
    tab.parent.fire(tab._after_id)
    assert len(tab.steps) == 3 and len(tab.parent.jobs) == 1


def test_interactive_close_cancels_the_timer_and_disconnects_editor_once():
    tab = _interactive('snn')
    disconnected = []

    class Editor:
        def disconnect(self):
            disconnected.append(True)

    tab._editor = Editor()
    tab._toggle_run()
    assert tab.parent.jobs[tab._after_id][0] == 90
    tab.close()
    tab.close()
    assert not tab._running and not tab.parent.jobs
    assert disconnected == [True]


def test_stop_during_a_playback_step_does_not_reschedule():
    tab = _interactive()
    tab._step = tab._stop
    tab._toggle_run()
    assert not tab._running and not tab.parent.jobs


def test_diversity_close_cancels_polling_and_ignores_late_worker_messages():
    tab = DiversityTab.__new__(DiversityTab)
    tab.parent = _Timers()
    tab._stop = threading.Event()
    tab._queue = queue.Queue()
    tab._after_id = tab.parent.after(120, tab._poll)
    tab._closed = False
    tab.close()
    tab.close()
    assert tab._stop.is_set() and not tab.parent.jobs
    tab._queue.put(('done', None))
    tab._poll()  # Widgets are already gone; this must never touch them.
    tab.analyse()
    assert not tab.parent.jobs and tab._queue.qsize() == 1


def test_designer_pause_reset_and_close_cancel_the_pending_playback_timer():
    tab = DesignerTab.__new__(DesignerTab)
    tab.parent = _Timers()
    tab._running = False
    tab._after_id = None
    tab._run_btn = _Button()
    tab.grid = {(0, 0): 1}
    tab.out_pos = {}
    tab._player = type('Player', (), {'at_end': lambda self: False})()
    tab._step = lambda: True
    tab._toggle_run()
    tab._toggle_run()
    assert not tab.parent.jobs and not tab._running
    tab._toggle_run()
    assert len(tab.parent.jobs) == 1
    tab._reset_sim()
    assert not tab.parent.jobs and not tab._running
    tab._player = type('Player', (), {'at_end': lambda self: False})()
    tab._toggle_run()
    tab.close()
    assert not tab.parent.jobs and not tab._running


def test_tab_close_releases_only_its_own_figures():
    with mock.patch('matplotlib.pyplot.close') as close:
        interactive = _interactive()
        interactive.fig = object()
        interactive.close()
        close.assert_called_once_with(interactive.fig)
        close.reset_mock()

        designer = DesignerTab.__new__(DesignerTab)
        designer.fig, designer._lut_fig, designer._tl_fig = object(), object(), object()
        designer.close()
        assert close.call_args_list == [mock.call(designer.fig),
                                        mock.call(designer._lut_fig),
                                        mock.call(designer._tl_fig)]
        close.reset_mock()

        diversity = DiversityTab.__new__(DiversityTab)
        diversity._stop = threading.Event()
        diversity._after_id = None
        diversity._fig = object()
        diversity.close()
        close.assert_called_once_with(diversity._fig)
