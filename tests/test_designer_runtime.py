"""GUI-free checks for the Designer playback controller."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from designer import DesignerTab, _genome_to_dict, _genome_from_dict
from app import App
from nv_evo.targets import TEMPORAL_TARGETS
from nv_evo.pulse import PulseConfig
from nv_evo.genome import random_hex_genome
from evo_runtime.config import (NV_NEW_RUN_PROFILES, GAConfig,
                                is_current_nv_profile,
                                validate_new_nv_profile)


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


def test_app_exposes_only_the_current_nv_profiles():
    assert NV_NEW_RUN_PROFILES == {
        'legacy': ('single', 'pulse_delay', None),
        'digital_tri': ('tri3', 'uniform', None),
        'analog_tri': ('tri3', 'paper_analog', None),
    }
    assert is_current_nv_profile(GAConfig(
        tile_arch='single', node_model='pulse_delay'))
    assert is_current_nv_profile(GAConfig(
        tile_arch='tri3', node_model='uniform'))
    assert is_current_nv_profile(GAConfig(
        tile_arch='tri3', node_model='paper_analog'))
    assert not is_current_nv_profile(GAConfig(
        tile_arch='single', node_model='uniform'))

    app = App.__new__(App)
    app._NV_PROFILE_LABELS = {
        'Legacy': NV_NEW_RUN_PROFILES['legacy'],
        'Analog': NV_NEW_RUN_PROFILES['analog_tri'],
    }
    app._nv_profile_var = type(
        'Var', (), {'get': lambda self: self.value})()
    app._nv_profile_var.value = 'Legacy'
    assert App._selected_tile_arch(app) == 'single'
    assert App._selected_node_model(app) == ('pulse_delay', None)
    app._nv_profile_var.value = 'Analog'
    assert App._selected_tile_arch(app) == 'tri3'
    assert App._selected_node_model(app) == ('paper_analog', None)


def test_wiring_io_selection_automatically_provides_chromosome_three():
    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    statuses = []
    app = App.__new__(App)
    app._IO_PLACEMENT_LABELS = {
        'Fixed': 'fixed',
        'Wiring': 'wiring_chromosome',
        'Spatial': 'spatial_chromosome',
    }
    app._io_placement_var = Var('Spatial')
    app._chroms_var = Var('2')
    app._status = type(
        'Status', (), {'set': lambda _self, value: statuses.append(value)})()

    App._on_io_placement_change(app)
    assert app._chroms_var.get() == '3'
    assert 'chromosome 3' in statuses[-1]

    app._io_placement_var.set('Wiring')
    app._chroms_var.set('2')
    App._on_io_placement_change(app)
    assert app._chroms_var.get() == '3'

    app._chroms_var.set('5')
    App._on_io_placement_change(app)
    assert app._chroms_var.get() == '5'


def test_new_nv_run_validation_rejects_every_retired_pairing():
    validate_new_nv_profile(GAConfig(
        tile_arch='single', node_model='pulse_delay'))
    validate_new_nv_profile(GAConfig(
        tile_arch='tri3', node_model='uniform'))
    validate_new_nv_profile(GAConfig(
        tile_arch='tri3', node_model='paper_analog'))
    for config in (
            GAConfig(tile_arch='single', node_model='uniform'),
            GAConfig(tile_arch='single', node_model='paper_analog'),
            GAConfig(tile_arch='single', node_model='pulse_delay',
                     evolve_delay=False)):
        try:
            validate_new_nv_profile(config)
        except ValueError:
            pass
        else:
            raise AssertionError('retired NV pairing was accepted: %r' %
                                 (config,))


def test_analog_profile_relabels_and_locks_irrelevant_controls():
    class Widget:
        def __init__(self):
            self.options = {}

        def configure(self, **options):
            self.options.update(options)

    class RowWidget(Widget):
        def __init__(self):
            super().__init__()
            self.packed = False

        def pack(self, **_kw):
            self.packed = True

        def pack_forget(self):
            self.packed = False

    app = App.__new__(App)
    app._NV_PROFILE_LABELS = {
        'Analog': NV_NEW_RUN_PROFILES['analog_tri'],
    }
    app._nv_profile_var = type('Var', (), {'get': lambda self: 'Analog'})()
    app._pulse_entries = [Widget(), Widget(), Widget()]
    app._pulse_labels = [Widget(), Widget(), Widget()]
    app._nv_profile_cb = Widget()
    app._tune_reset_btn = Widget()
    app._nv_controls_locked = False
    app._analog_row = RowWidget()
    app._analog_entries = [Widget(), Widget(), Widget(), Widget()]
    app._backend_var = type('Var', (), {'get': lambda self: 'Nervous'})()
    app._nb = object()

    App._sync_nv_profile_controls(app)
    assert [w.options['text'] for w in app._pulse_labels] == [
        'Propagation delay:', 'Input width:', 'Coinc (emergent):']
    assert [w.options['state'] for w in app._pulse_entries] == [
        'normal', 'normal', 'disabled']
    assert app._nv_profile_cb.options['state'] == 'readonly'
    assert app._analog_row.packed
    assert all(w.options['state'] == 'normal' for w in app._analog_entries)

    App._set_nv_controls_locked(app, True)
    assert all(w.options['state'] == 'disabled' for w in app._pulse_entries)
    assert app._nv_profile_cb.options['state'] == 'disabled'
    assert app._tune_reset_btn.options['state'] == 'disabled'
    # the analog row stays visible during a run but its constants lock
    assert app._analog_row.packed
    assert all(w.options['state'] == 'disabled' for w in app._analog_entries)


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


def test_designer_json_roundtrip_preserves_tri_tile_architecture():
    genome = random_hex_genome(2, arch='tri3')
    restored = _genome_from_dict(
        _genome_to_dict(genome, 'nervous'), 'nervous')
    assert restored.arch == 'tri3'
    assert restored.chromosomes == genome.chromosomes


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


def test_interactive_combinational_cases_load_fitness_pulses():
    """A COMBINATIONAL target has no temporal trials, only a truth table, so the
    Interactive tab must enumerate its cases and load the SAME input pulses
    fitness scores (aligned-start, schedule widths for LUT) — otherwise 'see what
    fitness scored' shows nothing for gates/adders."""
    from interactive import InteractiveTab
    from nv_evo.playback import pulses_from_case
    from lut_evo.ga import _combinational_schedule
    from snn_evo.targets import get_target

    target = get_target('AND')
    n_inputs = len(target.inputs)

    tab = InteractiveTab.__new__(InteractiveTab)
    tab._circuit = {'target': target}
    tab._in_pos = list(target.inputs)
    tab._backend = 'lut'
    tab._running = False
    # no _case_kind set — the tab must derive 'cases' from the target itself.

    labels = InteractiveTab._case_labels(tab, target)
    assert len(labels) == len(target.cases)
    assert labels[0] == 'Case 1/4: in=00 -> 0'
    assert labels[3] == 'Case 4/4: in=11 -> 1'

    delay, widths = _combinational_schedule(target)[0]

    class _Editor:
        pulses = None

        def set_pulses(self, pulses):
            self.pulses = pulses

        def schedule(self, _cells):
            return {}

    class _Player:
        def set_schedule(self, _schedule):
            pass

    class _Var:
        value = None

        def set(self, value):
            self.value = value

    tab._editor = _Editor()
    tab._player = _Player()
    tab._case_var = _Var()
    tab._draw_async = lambda: None

    for index in range(len(target.cases)):
        tab._case_cb = type('Box', (), {'current': lambda _s, i=index: i})()
        InteractiveTab._on_case_selected(tab)
        assert tab._editor.pulses == pulses_from_case(
            target, n_inputs, index, 'lut')
        in_bits = target.cases[index][0]
        starts = [lane[0][0] for lane, b in zip(tab._editor.pulses, in_bits) if b]
        assert all(s == starts[0] for s in starts), 'start edges not aligned'
        assert tab._case_var.value is None


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
