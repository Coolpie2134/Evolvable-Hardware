"""GUI-free checks for the Designer playback controller."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from designer import DesignerTab
from app import App
from nv_evo.targets import TEMPORAL_TARGETS
from nv_evo.pulse import PulseConfig


class _Player:
    def __init__(self):
        self.cursor = 0.0
        self.horizon = 1.0

    def step(self):
        self.cursor = min(self.horizon, self.cursor + 0.5)
        return self.cursor

    def at_end(self):
        return self.cursor >= self.horizon


class _Button:
    def __init__(self):
        self.text = None

    def config(self, **values):
        self.text = values.get('text', self.text)


class _Parent:
    def __init__(self):
        self.callbacks = []

    def after(self, _delay, callback):
        self.callbacks.append(callback)


def test_designer_run_stops_cleanly_at_the_horizon():
    tab = DesignerTab.__new__(DesignerTab)
    tab._running = True
    tab._player = _Player()
    tab._run_btn = _Button()
    tab.parent = _Parent()
    statuses = []
    tab._status = type('Status', (), {'set': lambda _self, value: statuses.append(value)})()

    def step_once():
        tab._player.step()
        return True

    tab._step = step_once
    tab._tick_loop()
    assert tab._running
    assert len(tab.parent.callbacks) == 1

    tab.parent.callbacks.pop()()
    assert not tab._running
    assert tab._run_btn.text == 'Run'
    assert not tab.parent.callbacks
    assert statuses[-1] == 'Playback complete at 1.0 seconds.'


def test_designer_prefers_loaded_target_physics_over_registry_copy():
    key, registered = next(
        (key, target) for key, target in TEMPORAL_TARGETS.items()
        if key != target.name)
    loaded = type('LoadedTarget', (), {})()
    loaded.name = registered.name
    loaded.pulse_config = PulseConfig(
        model='pulse_delay', delay=0.4, width=1.7)

    tab = DesignerTab.__new__(DesignerTab)
    tab.backend = 'nervous'
    tab._loaded_target = loaded
    tab._target_var = type('Var', (), {'get': lambda _self: key})()

    assert DesignerTab._current_target(tab) is loaded


def test_interactive_case_dropdown_loads_each_trial():
    """The Interactive tab exposes EVERY stored test case, not just trial 0:
    selecting a case loads exactly that trial's physical schedule into the
    timeline, and a subsequent hand edit flips the box to '(custom schedule)'
    instead of silently claiming to still show the case."""
    from interactive import InteractiveTab
    from nv_evo.playback import pulses_from_trial

    target = TEMPORAL_TARGETS['Odd pulse selector']
    n_inputs = len(target.inputs)

    tab = InteractiveTab.__new__(InteractiveTab)
    tab._circuit = {'target': target}
    tab._in_pos = list(target.inputs)
    tab._running = False

    labels = InteractiveTab._case_labels(tab, target)
    assert len(labels) == len(target.trials)
    assert labels[0] == 'Case 1/%d: silent' % len(target.trials)   # empty bank
    assert labels[1].startswith('Case 2/%d: A[' % len(target.trials))

    class _Editor:
        pulses = None

        def set_pulses(self, pulses):
            self.pulses = pulses

        def schedule(self, _cells):
            return {}

    class _Player:
        schedule = None

        def set_schedule(self, schedule):
            self.schedule = schedule

    class _Var:
        value = None

        def set(self, value):
            self.value = value

    tab._editor = _Editor()
    tab._player = _Player()
    tab._case_var = _Var()
    tab._draw_async = lambda: None

    for index in (1, 4, len(target.trials) - 1):
        tab._case_cb = type('Box', (), {'current': lambda _self: index})()
        InteractiveTab._on_case_selected(tab)
        expected = [[(float(s), float(w)) for s, w in lane]
                    for lane in target.trials[index].input_events]
        assert tab._editor.pulses == expected
        assert tab._editor.pulses == pulses_from_trial(target, n_inputs, index)
        # loading a case must not mark the timeline as hand-edited
        assert tab._case_var.value is None

    # a manual timeline edit (editor on_change) flips the label to custom
    InteractiveTab._nv_schedule_changed(tab)
    assert tab._case_var.value == '(custom schedule)'


def test_interactive_width_strip_clips_open_intervals_to_cursor():
    """The width panel reads the engine's waveform log: closed pulses keep
    their real span, a still-high pulse is clipped to the cursor and marked
    open, not-yet-started pulses are hidden, and engines without a waveform
    log degrade to an empty panel instead of crashing."""
    from interactive import InteractiveTab

    tab = InteractiveTab.__new__(InteractiveTab)

    class _Sim:
        pulse_intervals = {(2, 2): [[1.0, 2.5], [4.0, float('inf')],
                                    [9.0, 10.0]]}

    tab._player = type('P', (), {'sim': _Sim(), 'cursor': 5.5})()
    assert InteractiveTab._output_spans(tab, (2, 2)) == [
        (1.0, 2.5, False), (4.0, 5.5, True)]
    assert InteractiveTab._output_spans(tab, None) == []
    assert InteractiveTab._output_spans(tab, (0, 0)) == []   # unlogged wire

    tab._player = type('P', (), {'sim': object(), 'cursor': 5.5})()
    assert InteractiveTab._output_spans(tab, (2, 2)) == []


def test_charge_levels_follow_the_waveform_like_a_capacitor():
    """Playback nodes charge toward 1 while their wire is high and decay
    exponentially after it falls (display-only RC follower of the binary
    waveform); engines without a waveform log return None."""
    import math
    from nv_evo.playback import charge_levels, CHARGE_TAU, DISCHARGE_TAU

    class _Sim:
        pulse_intervals = {'w': [[1.0, 3.0], [6.0, float('inf')]]}

    sim = _Sim()
    assert charge_levels(object(), 5.0) is None          # no waveform log
    assert charge_levels(sim, 0.5)['w'] == 0.0           # before any pulse
    # mid-pulse: charging toward 1
    q2 = charge_levels(sim, 2.0)['w']
    assert abs(q2 - (1.0 - math.exp(-1.0 / CHARGE_TAU))) <= 1e-9
    # after the fall: exponential discharge from the level at the fall
    q_fall = charge_levels(sim, 3.0)['w']
    q4 = charge_levels(sim, 4.0)['w']
    assert abs(q4 - q_fall * math.exp(-1.0 / DISCHARGE_TAU)) <= 1e-9
    assert 0.0 < q4 < q_fall
    # an open-ended pulse keeps charging monotonically
    assert charge_levels(sim, 8.0)['w'] > charge_levels(sim, 6.5)['w']
    # levels always stay inside [0, 1] for the colour interpolation
    for t in (0.0, 1.5, 3.0, 5.0, 7.0, 40.0):
        assert 0.0 <= charge_levels(sim, t)['w'] <= 1.0


class _Line:
    def __init__(self):
        self.x = self.y = None

    def set_data(self, x, y):
        self.x, self.y = list(x), list(y)


class _Axis:
    def __init__(self):
        self.limits = None

    def set_xlim(self, lo, hi):
        self.limits = (lo, hi)

    def set_ylim(self, lo, hi):
        self.limits = (lo, hi)


class _Text:
    def __init__(self):
        self.value = ''

    def set_text(self, value):
        self.value = value


def test_fitness_chart_reports_effective_mutation_rate():
    app = App.__new__(App)
    app._gen_history = [
        (1, 0.40, 0.20, 0.35, 4.0, 0.08),
        (2, 0.45, 0.22, 0.42, 3.5, 0.11),
        (3, 0.45, 0.21, 0.40, 5.0, 0.09),
    ]
    app._best_line = _Line()
    app._mean_line = _Line()
    app._genbest_line = _Line()
    app._mutation_line = _Line()
    app._std_line = _Line()
    app._fit_ax = _Axis()
    app._mut_ax = _Axis()
    app._mutation_text = _Text()
    app._fit_canvas = type('Canvas', (), {'draw_idle': lambda _self: None})()

    App._redraw_fit_chart(app)

    assert app._mutation_line.y == [4.0, 3.5, 5.0]
    assert app._std_line.y == [0.08, 0.11, 0.09]
    assert app._mutation_text.value == 'Mutation: 5.000'
    assert app._mut_ax.limits[1] > 5.0
