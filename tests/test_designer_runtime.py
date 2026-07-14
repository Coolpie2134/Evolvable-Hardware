"""GUI-free checks for the Designer playback controller."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from designer import DesignerTab
from app import App


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
        (1, 0.40, 0.20, 0.35, 4.0),
        (2, 0.45, 0.22, 0.42, 3.5),
        (3, 0.45, 0.21, 0.40, 5.0),
    ]
    app._best_line = _Line()
    app._mean_line = _Line()
    app._genbest_line = _Line()
    app._mutation_line = _Line()
    app._fit_ax = _Axis()
    app._mut_ax = _Axis()
    app._mutation_text = _Text()
    app._fit_canvas = type('Canvas', (), {'draw_idle': lambda _self: None})()

    App._redraw_fit_chart(app)

    assert app._mutation_line.y == [4.0, 3.5, 5.0]
    assert app._mutation_text.value == 'Mutation: 5.000'
    assert app._mut_ax.limits[1] > 5.0
