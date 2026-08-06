"""Reusable categorized, searchable target picker for the Tk GUIs."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


ALL_TARGETS = 'All targets'
COMBINATIONAL = 'Combinational logic'
PULSE_WIDTH = 'Pulse width & duration'
CATEGORY_ORDER = (
    ALL_TARGETS,
    COMBINATIONAL,
    'Timed events',
    'Memory & state',
    'Cadence & patterns',
    PULSE_WIDTH,
)


def target_category(name, target):
    """Return a stable user-facing category from a target's semantics.

    A target may pin itself with an explicit ``category`` attribute - used by
    the periodic combinational wrappers (temporal encodings of truth tables,
    which would otherwise sort under 'Timed events') and the pulse-width
    targets (whose fitness is about physical durations, not just edge times).
    Everything else falls back to score-mode semantics; every non-temporal
    target is a truth table, i.e. combinational."""
    if target is None:
        return ALL_TARGETS
    explicit = getattr(target, 'category', '')
    if explicit:
        return explicit
    if getattr(target, 'temporal', False):
        relations = {c.relation for c in target.contract.constraints}
        if relations & {'sustained_cadence', 'commanded_cadence'}:
            return 'Cadence & patterns'
        if 'event_correspondence' in relations:
            return 'Timed events'
        return 'Memory & state'
    return COMBINATIONAL


class TargetPicker(ttk.Frame):
    """Category filter plus editable combobox with incremental name search."""

    def __init__(self, parent, targets, variable=None, command=None,
                 include_none=False, target_width=22):
        super().__init__(parent)
        self.variable = variable or tk.StringVar(self)
        self.command = command
        self.include_none = include_none
        self._targets = {}

        self.category_var = tk.StringVar(self, value=ALL_TARGETS)
        self.category_cb = ttk.Combobox(
            self, textvariable=self.category_var, state='readonly', width=22)
        self.category_cb.pack(side='left', padx=(0, 3))
        self.category_cb.bind('<<ComboboxSelected>>', self._on_category)

        self.target_cb = ttk.Combobox(
            self, textvariable=self.variable, state='normal', width=target_width)
        self.target_cb.pack(side='left')
        self.target_cb.bind('<<ComboboxSelected>>', self._commit)
        self.target_cb.bind('<Return>', self._commit)
        self.target_cb.bind('<KeyRelease>', self._on_search)
        self.set_targets(targets)

    def _names(self, query=''):
        category = self.category_var.get()
        query = query.strip().lower()
        names = []
        if self.include_none and category == ALL_TARGETS and not query:
            names.append('(none)')
        for name, target in self._targets.items():
            if category != ALL_TARGETS and target_category(name, target) != category:
                continue
            if query and query not in name.lower():
                continue
            names.append(name)
        return names

    def _refresh_values(self, query=''):
        self.target_cb['values'] = self._names(query)

    def _on_category(self, _event=None):
        self._refresh_values()
        current = self.variable.get()
        values = list(self.target_cb['values'])
        if current not in values and values:
            self.variable.set(values[0])
            if self.command:
                self.command()
        self.target_cb.focus_set()

    def _on_search(self, event):
        if event.keysym in {'Return', 'Escape', 'Up', 'Down', 'Left', 'Right',
                           'Home', 'End', 'Tab'}:
            return
        self._refresh_values(self.variable.get())

    def _commit(self, _event=None):
        typed = self.variable.get().strip()
        matches = [name for name in self._targets
                   if name.lower() == typed.lower()]
        if typed == '(none)' and self.include_none:
            chosen = typed
        elif matches:
            chosen = matches[0]
        else:
            values = list(self.target_cb['values'])
            if len(values) != 1:
                return
            chosen = values[0]
        self.variable.set(chosen)
        self._refresh_values()
        if self.command:
            self.command()

    def set_targets(self, targets, preserve=True):
        """Replace available targets while preserving a valid selection."""
        current = self.variable.get()
        self._targets = dict(targets)
        categories = {target_category(name, target)
                      for name, target in self._targets.items()}
        ordered = [c for c in CATEGORY_ORDER
                   if c == ALL_TARGETS or c in categories]
        # explicit categories outside the standard order (e.g. from imported
        # targets) still get a folder rather than becoming unreachable
        ordered += sorted(categories - set(CATEGORY_ORDER))
        self.category_cb['values'] = ordered
        if self.category_var.get() not in self.category_cb['values']:
            self.category_var.set(ALL_TARGETS)
        self._refresh_values()
        valid = current in self._targets or (self.include_none and current == '(none)')
        if not preserve or not valid:
            self.variable.set('(none)' if self.include_none else '')

    def select(self, name, notify=False):
        if name != '(none)' and name not in self._targets:
            return False
        if name != '(none)':
            self.category_var.set(target_category(name, self._targets[name]))
        else:
            self.category_var.set(ALL_TARGETS)
        self.variable.set(name)
        self._refresh_values()
        if notify and self.command:
            self.command()
        return True

    def set_state(self, state):
        """Enable both controls, or lock both while a run is active."""
        disabled = state == 'disabled'
        self.category_cb.configure(state='disabled' if disabled else 'readonly')
        self.target_cb.configure(state='disabled' if disabled else 'normal')
