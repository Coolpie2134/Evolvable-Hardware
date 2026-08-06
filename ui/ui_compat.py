"""
ui_compat.py - small cross-platform helpers so the Tk GUI looks and behaves the
same on Windows and Linux (and macOS).

Two portability gaps this closes:

  * Monospace fonts: the truth-table / genome panels want a fixed-width font, but
    'Courier New' only exists on Windows; Linux typically has 'DejaVu Sans Mono'.
    `mono_family()` probes the actually-installed families and returns the first
    match, falling back to Tk's guaranteed 'TkFixedFont'.
  * ttk theming: the default theme differs per platform ('vista' on Windows,
    'default'/'clam' on X11), which changes widget metrics and colours. `apply_theme`
    selects 'clam' - present everywhere and visually consistent - so layouts don't
    shift between machines.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# Preference order: nice-looking modern monospaces first, then the classic
# platform faces, then the generic fallback. Windows: Consolas/Courier New;
# Linux: DejaVu/Liberation; macOS: Menlo.
_MONO_CANDIDATES = ('Consolas', 'DejaVu Sans Mono', 'Menlo',
                    'Liberation Mono', 'Ubuntu Mono', 'Courier New',
                    'Courier', 'monospace')

_mono_cache = None


def mono_family(root=None):
    """First installed monospace family from the preference list, or the
    always-present 'TkFixedFont' named font. Cached after the first probe."""
    global _mono_cache
    if _mono_cache is not None:
        return _mono_cache
    try:
        available = set(tkfont.families(root))
    except Exception:
        available = set()
    _mono_cache = next((f for f in _MONO_CANDIDATES if f in available),
                       'TkFixedFont')
    return _mono_cache


def apply_theme(root):
    """Pin a consistent ttk theme so metrics/colours match across platforms.
    Returns the ttk.Style (already themed). Safe if 'clam' is unavailable."""
    style = ttk.Style(root)
    try:
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    except tk.TclError:
        pass
    # slightly roomier, readable defaults that render identically on both OSes
    try:
        style.configure('TButton', padding=(8, 3))
        style.configure('TLabelframe.Label', foreground='#334')
    except tk.TclError:
        pass
    return style


def fit_window(root, min_width=900, min_height=640, margin=72):
    """Size a Tk window to its content without exceeding the visible screen.

    Large matplotlib canvases otherwise make Tk request a window taller than a
    typical laptop display.  The returned ``(width, height)`` is useful in GUI
    layout tests and the explicit geometry prevents notebook tab changes from
    resizing the toplevel.
    """
    root.update_idletasks()
    max_width = max(640, root.winfo_screenwidth() - margin)
    max_height = max(480, root.winfo_screenheight() - margin)
    width = min(max(root.winfo_reqwidth(), min_width), max_width)
    height = min(max(root.winfo_reqheight(), min_height), max_height)
    root.minsize(min(min_width, max_width), min(min_height, max_height))
    root.geometry('%dx%d' % (width, height))
    return width, height


def cancel_after_callbacks(root):
    """Cancel pending Tk/Matplotlib idle jobs before destroying a window."""
    try:
        jobs = root.tk.splitlist(root.tk.call('after', 'info'))
    except tk.TclError:
        return
    for job in jobs:
        try:
            root.after_cancel(job)
        except tk.TclError:
            pass
