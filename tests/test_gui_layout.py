"""
tests/test_gui_layout.py - the GUI must not silently hide its own controls.

Tk's packer does not clip a row that is too wide: it UNMAPS the widgets that no
longer fit, with nothing on screen to say anything is missing. That is how the
NV profile selector, the I/O binding control and the Reset tuning button all
disappeared from the nervous net's tuning row - the row needed about 2070px in
a window that is normally 1500px wide. These tests pin the row budget so the
same failure cannot return unnoticed.

The Tk tests degrade to a no-op where no display can be opened, because the
bare runner has no skip mechanism and a headless box is not a regression.

Run under the suite runner:  py tests/run_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402

from substrates.fnv.catalogue import BY_ID, COMPONENTS           # noqa: E402
from substrates.fnv.viz import (component_label,                 # noqa: E402
                                draw_functional_net,
                                FAMILY_COLORS, FUNCTION_LEGEND)

#: Widest window the layout must survive. A 1366x768 laptop is the realistic
#: floor, and 1280 leaves room for the window frame.
LAPTOP_WIDTH = 1280
BACKEND_LABELS = {'nervous': 'Nervous', 'lut': 'LUT', 'fnv': 'FNV',
                  'snn': 'SNN'}


def _app():
    """A built App, or None when no display is available."""
    try:
        import tkinter as tk
    except Exception:                                   # noqa: BLE001
        return None, None
    try:
        root = tk.Tk()
    except Exception:                                   # noqa: BLE001
        return None, None
    from ui.app import App
    root.geometry('%dx900+0+0' % LAPTOP_WIDTH)
    return root, App(root)


def _close(root, app):
    """App.close() cancels the poll timer; a bare destroy() leaves it firing."""
    try:
        app.close()
    except Exception:                                   # noqa: BLE001
        root.destroy()


def _rows(root):
    return [child for child in root.winfo_children()
            if child.winfo_ismapped() and child.winfo_children()]


def test_no_control_row_is_too_wide_for_a_laptop_window():
    root, app = _app()
    if root is None:
        return
    try:
        overflowing = []
        for backend, label in BACKEND_LABELS.items():
            app._backend_var.set(label)
            app._on_backend_change()
            root.update()
            for row in _rows(root):
                if row.winfo_reqwidth() > LAPTOP_WIDTH:
                    overflowing.append(
                        '%s: a row needs %dpx' % (backend, row.winfo_reqwidth()))
        assert not overflowing, (
            'control rows overflow at %dpx, so Tk will unmap their tail '
            'widgets:\n  %s' % (LAPTOP_WIDTH, '\n  '.join(overflowing)))
    finally:
        _close(root, app)


def test_every_substrate_control_is_actually_on_screen():
    """The controls that vanished, pinned by name.

    Each is checked on the backend that owns it - the NV profile selector is
    nervous-only, the family bank is FNV-only - so hiding one for an unrelated
    backend still passes, but losing it to overflow does not.
    """
    root, app = _app()
    if root is None:
        return
    expected = {
        'nervous': ('NV profile:', 'I/O binding:', 'Reset tuning', 'Pulse:'),
        'lut':     ('I/O binding:', 'Reset tuning', 'LUT function banks:'),
        'fnv':     ('I/O binding:', 'Reset tuning', 'FNV component families:',
                    'Output readout:'),
        'snn':     ('I/O binding:', 'Reset tuning', 'Substrate:'),
    }

    def texts(widget, found):
        for child in widget.winfo_children():
            try:
                value = str(child.cget('text'))
            except Exception:                           # noqa: BLE001
                value = ''
            if value and child.winfo_ismapped():
                found.add(value)
            texts(child, found)
        return found

    try:
        missing = []
        for backend, wanted in expected.items():
            app._backend_var.set(BACKEND_LABELS[backend])
            app._on_backend_change()
            root.update()
            visible = texts(root, set())
            missing += ['%s: %r' % (backend, text)
                        for text in wanted if text not in visible]
        assert not missing, (
            'controls built but never visible: %s' % ', '.join(missing))
    finally:
        _close(root, app)


def test_switching_backend_never_leaves_a_stale_row_packed():
    """A row belonging to one substrate must not survive onto another."""
    root, app = _app()
    if root is None:
        return
    owned = {'FNV component families:': 'fnv',
             'LUT function banks:': 'lut',
             'NV profile:': 'nervous'}

    def visible_texts(widget, found):
        for child in widget.winfo_children():
            try:
                value = str(child.cget('text'))
            except Exception:                           # noqa: BLE001
                value = ''
            if value and child.winfo_ismapped():
                found.add(value)
            visible_texts(child, found)
        return found

    try:
        leaked = []
        for backend, label in BACKEND_LABELS.items():
            app._backend_var.set(label)
            app._on_backend_change()
            root.update()
            visible = visible_texts(root, set())
            leaked += ['%s shows %r' % (backend, text)
                       for text, owner in owned.items()
                       if text in visible and owner != backend]
        assert not leaked, 'stale substrate rows: %s' % ', '.join(leaked)
    finally:
        _close(root, app)


# -- FNV body colouring ---------------------------------------------------------

def test_every_component_has_a_short_readable_function_label():
    """Labels are drawn INSIDE a node, so length is a correctness property.

    "VETO" overflowed its marker and rendered as "ETO" - unreadable, and worse,
    ambiguous. Three characters is what fits.
    """
    for entry in COMPONENTS:
        if entry.id == 0:
            continue
        label = component_label(entry)
        assert label, entry.name
        assert len(label) <= 3, (entry.name, label)


def test_function_labels_separate_components_that_behave_differently():
    """Two parts that do different things must not read the same.

    A delay of 1 and a delay of 2 are different hardware; if both said "D" the
    view would be prettier and wrong. Routing IS deliberately collapsed - it is
    already drawn as the wires.
    """
    by_label = {}
    for entry in COMPONENTS:
        if entry.id == 0:
            continue
        signature = (entry.behavior, entry.duration, entry.high_time,
                     entry.low_time)
        by_label.setdefault(component_label(entry), set()).add(signature)
    collisions = {label: sorted(sigs) for label, sigs in by_label.items()
                  if len(sigs) > 1}
    assert not collisions, collisions


def test_every_family_present_in_a_body_can_be_explained_by_the_legend():
    legend = dict(FUNCTION_LEGEND)
    for family in FAMILY_COLORS:
        assert family in legend, family


def _tiny_body():
    grid, cell = {}, (0, 0)
    for entry in COMPONENTS[1:6]:
        grid[cell] = entry.id
        cell = (cell[0] + 1, cell[1])
    return grid


def test_both_colour_modes_render_and_label_differently():
    """Branch mode answers "who built it", function mode "what is it"."""
    grid = _tiny_body()
    branches = {cell: 1 + index for index, cell in enumerate(sorted(grid))}
    labels = {}
    for mode in ('branch', 'function'):
        figure = plt.figure()
        axes = figure.add_subplot(111)
        draw_functional_net(axes, grid, input_positions=[],
                            branches=branches, color_by=mode)
        labels[mode] = sorted(
            artist.get_text() for artist in axes.texts if artist.get_text())
        plt.close(figure)
    assert labels['branch'] != labels['function'], labels
    assert any(text == component_label(BY_ID[state])
               for state in grid.values() for text in labels['function'])


def test_colour_mode_defaults_preserve_every_existing_caller():
    """Callers written before the option keep their exact previous view."""
    grid = _tiny_body()
    branches = {cell: 1 for cell in grid}
    for supplied, expected in ((branches, 'branch'), (None, 'function')):
        figure = plt.figure()
        default_axes = figure.add_subplot(211)
        draw_functional_net(default_axes, grid, input_positions=[],
                            branches=supplied)
        explicit_axes = figure.add_subplot(212)
        draw_functional_net(explicit_axes, grid, input_positions=[],
                            branches=supplied, color_by=expected)
        assert ([a.get_text() for a in default_axes.texts]
                == [a.get_text() for a in explicit_axes.texts])
        plt.close(figure)


def test_an_unknown_colour_mode_is_refused_rather_than_guessed():
    grid = _tiny_body()
    figure = plt.figure()
    axes = figure.add_subplot(111)
    try:
        draw_functional_net(axes, grid, input_positions=[], color_by='family')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown colour mode was silently accepted')
    finally:
        plt.close(figure)


def test_a_panel_series_shares_one_scale():
    """Growth must read as growth, which needs a fixed frame of reference.

    With per-panel limits a two-cell seed was magnified to fill its panel
    while the grown body was crammed into the same space, so nothing held
    still between frames and the sequence could not be read.
    """
    from substrates.fnv.viz import body_extent
    small = {(0, 0): 1, (1, 0): 2}
    large = dict(small)
    large.update({(index, 1): 3 for index in range(6)})
    extent = body_extent([small, large])
    assert extent is not None
    limits = []
    for grid in (small, large):
        figure = plt.figure()
        axes = figure.add_subplot(111)
        draw_functional_net(axes, grid, input_positions=[], extent=extent,
                            legend=False)
        limits.append((axes.get_xlim(), axes.get_ylim()))
        plt.close(figure)
    assert limits[0] == limits[1], limits
    # And the shared extent really does cover the bigger body.
    assert body_extent([large])[1] <= extent[1] + 1e-9


def test_the_key_does_not_sit_on_top_of_the_circuit():
    """A legend that covers the lowest nodes explains nothing.

    Drawing it straight onto the axes put it over the body, so the axes
    reserve a band underneath instead.
    """
    grid = _tiny_body()
    spans = {}
    for legend in (False, True):
        figure = plt.figure()
        axes = figure.add_subplot(111)
        draw_functional_net(axes, grid, input_positions=[], legend=legend)
        spans[legend] = axes.get_ylim()
        plt.close(figure)
    assert spans[True][0] < spans[False][0], spans
    assert spans[True][1] == spans[False][1], spans


def test_the_key_is_a_legend_so_its_entries_cannot_collide():
    """Hand-placed text laid out in axes fractions while matplotlib sized it
    in points; on a narrow panel the entries overlapped into a smear."""
    from substrates.fnv.viz import draw_key
    grid = _tiny_body()
    figure = plt.figure(figsize=(2.2, 2.0))       # deliberately narrow
    axes = figure.add_subplot(111)
    draw_key(axes, grid)
    legend = axes.get_legend()
    assert legend is not None
    figure.canvas.draw()
    boxes = [text.get_window_extent(figure.canvas.get_renderer())
             for text in legend.get_texts()]
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            assert not boxes[first].overlaps(boxes[second]), (
                'legend entries overlap: %r vs %r'
                % (legend.get_texts()[first].get_text(),
                   legend.get_texts()[second].get_text()))
    plt.close(figure)


def test_a_figure_level_key_is_available_for_panel_grids():
    """Nine copies of one key is clutter, and each copy shrinks its panel."""
    from substrates.fnv.viz import draw_key
    grid = _tiny_body()
    figure = plt.figure()
    figure.add_subplot(111)
    draw_key(figure, grid)
    assert figure.legends, 'no figure-level key was produced'
    plt.close(figure)


# -- genome tab -----------------------------------------------------------------

def _fnv_genome(target_genes):
    import random
    from substrates.fnv.genome import random_functional_genome
    random.seed(5)
    best = best_count = None
    for _ in range(4000):
        genome = random_functional_genome(
            2, max_telomere=32, n_inputs=2, output_roles=('sum', 'carry'))
        count = sum(len(c.genes) for c in genome.chromosomes)
        if best is None or abs(count - target_genes) < abs(best_count
                                                           - target_genes):
            best, best_count = genome, count
        if best_count == target_genes:
            break
    return best, best_count


def test_the_genome_tab_shows_every_gene_and_grows_to_fit():
    """It used to cap at 24 cards and hide the rest behind a note.

    The tab scrolls now, so the page grows with the gene count instead of the
    genes being dropped to fit a fixed page.
    """
    root, app = _app()
    if root is None:
        return
    try:
        app._backend_var.set('FNV')
        app._on_backend_change()
        app._disp_backend = 'fnv'
        root.update()
        heights = []
        for wanted in (8, 60):
            genome, count = _fnv_genome(wanted)
            app._draw_genome(genome, 1.0)
            root.update()
            axes = app._genome_fig.axes[0]
            hidden = [text for text in axes.texts
                      if 'more genes not shown' in text.get_text()]
            assert not hidden, (count, hidden[0].get_text())
            heights.append((count, app._genome_fig.get_size_inches()[1]))
        (small, short), (large, tall) = heights
        assert large > small, heights
        assert tall > short, (
            'the page did not grow with the gene count: %s' % (heights,))
    finally:
        _close(root, app)


def test_the_genome_scroll_region_matches_the_rendered_page():
    """A scroll region shorter than the figure cannot reach the last genes."""
    root, app = _app()
    if root is None:
        return
    try:
        app._backend_var.set('FNV')
        app._on_backend_change()
        app._disp_backend = 'fnv'
        root.update()
        genome, _count = _fnv_genome(60)
        app._draw_genome(genome, 1.0)
        root.update()
        width, height = (app._genome_fig.get_size_inches()
                         * app._genome_fig.dpi)
        region = [float(value)
                  for value in str(app._genome_view.cget('scrollregion')).split()]
        assert region, 'no scroll region was set'
        assert abs(region[3] - height) <= 2.0, (region, height)
        assert abs(region[2] - width) <= 2.0, (region, width)
        # The page must actually be taller than the viewport, or there would be
        # nothing to scroll and the test would prove nothing.
        assert height > app._genome_view.winfo_height()
    finally:
        _close(root, app)


def test_genome_rendering_does_not_require_the_scrolling_viewport():
    """PNG export and the headless tests drive the figure with no Tk around."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from ui.app import App

    genome, _count = _fnv_genome(12)
    app = App.__new__(App)
    app._disp_backend = 'fnv'
    app._disp_target = None
    app._genome_fig = Figure(figsize=(8, 5))
    app._genome_canvas = FigureCanvasAgg(app._genome_fig)
    App._draw_genome(app, genome, 0.25)         # must not raise
    assert app._genome_fig.axes


def test_interactive_offers_both_modes_and_keeps_the_choice():
    """The mode is a view setting; reloading a circuit must not reset it."""
    from ui.interactive import InteractiveTab
    modes = set(InteractiveTab._FNV_COLOR_MODES.values())
    assert modes == {'branch', 'function'}, modes
    # The attribute is set in __init__, before any circuit exists, precisely so
    # that rebuilding the control on load does not lose the user's choice.
    import inspect
    source = inspect.getsource(InteractiveTab.__init__)
    assert '_fnv_color_mode' in source
