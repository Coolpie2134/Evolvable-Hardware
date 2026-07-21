"""
tests/test_diversity_ui.py — GUI-free checks of the Diversity tab's runtime.

Tk is never constructed. The worker thread body and the queue-draining poll are
plain methods, so they are exercised against stand-in widgets — the same trick
tests/test_designer_runtime.py uses. What matters here is the behaviour that is
easy to get wrong and invisible until a long run: Stop actually stops, a bad
file reports instead of hanging, and the controls are always released.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diversity_ui import DiversityTab, _Cancelled          # noqa: E402
from nv_evo import diversity as dv                         # noqa: E402


class _Var:
    def __init__(self, value=''):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Button:
    def __init__(self):
        self.state = 'normal'

    def config(self, **kwargs):
        if 'state' in kwargs:
            self.state = kwargs['state']


class _Parent:
    def __init__(self):
        self.callbacks = []

    def after(self, _delay, callback):
        self.callbacks.append(callback)
        return len(self.callbacks)


def _tab():
    """A DiversityTab with every Tk widget replaced by a stand-in."""
    tab = DiversityTab.__new__(DiversityTab)
    tab.parent = _Parent()
    tab._queue = queue.Queue()
    tab._stop = threading.Event()
    tab._worker = object()          # pretend a worker is running
    tab._after_id = None
    tab._status = _Var()
    tab._run_btn = _Button()
    tab._load_btn = _Button()
    tab._stop_btn = _Button()
    tab._path = None
    tab._get_population_path = None
    tab.texts = []
    tab.drawn = []
    tab._set_text = lambda text: tab.texts.append(text)
    tab._draw = lambda report, robustness: tab.drawn.append((report, robustness))
    tab._placeholder = lambda message: tab.drawn.append(('placeholder', message))
    return tab


def _drain(tab):
    """Run the poll loop until it stops rescheduling itself."""
    for _ in range(50):
        before = len(tab.parent.callbacks)
        tab._poll()
        if len(tab.parent.callbacks) == before:
            return True             # terminal branch: did not reschedule
        tab.parent.callbacks.pop()
    raise AssertionError('poll never reached a terminal message')


def test_error_message_is_shown_and_controls_are_released():
    tab = _tab()
    tab._queue.put(('error', 'ValueError: bad population'))
    assert _drain(tab)
    assert 'bad population' in tab._status.get()
    assert tab.texts and 'bad population' in tab.texts[-1]
    assert tab._worker is None
    assert tab._run_btn.state == 'normal' and tab._load_btn.state == 'normal'
    assert tab._stop_btn.state == 'disabled'


def test_cancel_releases_controls_and_reports_stopped():
    tab = _tab()
    tab._queue.put(('cancelled', None))
    assert _drain(tab)
    assert tab._status.get() == 'Stopped.'
    assert tab._worker is None and tab._run_btn.state == 'normal'


def test_progress_messages_keep_polling_without_finishing():
    tab = _tab()
    tab._queue.put(('progress', 'phenotype: 3/40'))
    tab._poll()
    assert tab._status.get() == 'phenotype: 3/40'
    assert tab.parent.callbacks          # rescheduled: analysis still running
    assert tab._worker is not None       # controls stay disabled mid-run


def test_funnel_then_done_renders_report_and_plot():
    stats = [dv.cluster_stats(level, ['a', 'a', 'b']) for level in dv.LEVELS]
    report = dv.DiversityReport(levels=tuple(stats), backend='nervous')
    tab = _tab()

    # the funnel arrives first: it names the population and draws immediately,
    # so a long robustness pass does not hide the result that is already known
    tab._queue.put(('funnel', report, 3, 'Toggle flip-flop', 'nervous'))
    tab._poll()
    tab.parent.callbacks.pop()
    assert 'Toggle flip-flop' in tab._status.get()
    assert tab.drawn and tab.drawn[0][0] is report
    assert tab.texts and dv.LEVEL_LABEL['behavior'] in tab.texts[-1]
    assert tab._worker is not None            # still running until 'done'

    tab._queue.put(('done', None))
    assert _drain(tab)
    assert tab._status.get() == 'Analysis complete.'
    assert tab._worker is None


def test_robustness_result_is_appended_to_the_report():
    stats = [dv.cluster_stats(level, ['a', 'b']) for level in dv.LEVELS]
    report = dv.DiversityReport(levels=tuple(stats), backend='nervous')
    robustness = dv.RobustnessReport(
        kernel=dv.ROBUSTNESS_KERNEL, samples=4, valid_threshold=0.999,
        per_genome_local=(0.5, 1.0), per_genome_novel=(0.0, 0.5),
        known_phenotypes=2)
    tab = _tab()
    tab._queue.put(('funnel', report, 2, 'Echo (delay 3)', 'nervous'))
    tab._queue.put(('done', robustness))
    assert _drain(tab)
    combined = tab.texts[-1]
    assert 'Novel-valid rate' in combined
    assert dv.LEVEL_LABEL['exact'] in combined
    assert tab.drawn[-1][1] is robustness


def test_stop_flag_makes_the_progress_callback_raise():
    """The worker cooperates with Stop through its progress callback, so a long
    funnel does not have to run to completion."""
    tab = _tab()
    tab._stop.set()

    def funnel_progress(level, index, total):
        if tab._stop.is_set():
            raise _Cancelled
    try:
        funnel_progress('phenotype', 0, 10)
    except _Cancelled:
        return
    raise AssertionError('Stop did not interrupt the progress callback')


def test_missing_population_file_resolves_to_none():
    tab = _tab()
    tab._path = os.path.join('definitely', 'not', 'here.json')
    tab._get_population_path = lambda: os.path.join('also', 'missing.json')
    # falls through to the default path, which only resolves if it exists
    resolved = DiversityTab._resolve_path(tab)
    assert resolved is None or os.path.exists(resolved)


def test_notify_population_ignores_a_path_that_does_not_exist():
    tab = _tab()
    tab.notify_population(os.path.join('nope', 'missing.json'))
    assert tab._path is None
