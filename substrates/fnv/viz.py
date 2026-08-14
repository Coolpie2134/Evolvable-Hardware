"""Matplotlib rendering for Functional NV Net bodies."""
from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from substrates.nervous.hexgrid import hex_pixel

from .catalogue import BY_ID
from .simulation import facing_direction, source_for_input

FAMILY_COLORS = {
    "LOGIC": "#2f6fbd",
    "DELAY": "#6c8ebf",
    "NORMALIZER": "#00a6a6",
    "HOLD": "#8e6bbd",
    "C_ELEMENT": "#d18b00",
    "TOGGLE": "#c04b87",
    "GATED_OSCILLATOR": "#d34b4b",
}

#: One colour per BRANCH, cycled by label. A genome has two arms per chromosome,
#: so these are read in pairs: branches 1/2 are chromosome a's two arms, 3/4
#: chromosome b's, and so on.
BRANCH_COLORS = (
    "#2f6fbd", "#7fb2e8",      # chromosome a: top arm, bottom arm
    "#c04b87", "#e79ec0",      # chromosome b
    "#00a6a6", "#7fd6d6",      # chromosome c
    "#d18b00", "#efc266",      # chromosome d
)
#: Cells no branch owns - nothing changed them after they appeared.
UNOWNED_COLOR = "#9aa0a6"
INPUT_COLOR = "#111111"
OUTPUT_COLOR = "#e8a33d"
ROOT_COLOR = "#7a3db8"


def branch_color(label):
    if not label:
        return UNOWNED_COLOR
    return BRANCH_COLORS[(int(label) - 1) % len(BRANCH_COLORS)]


def component_label(entry):
    """Short in-node text naming the component's FUNCTION, not its catalogue id.

    The id alone ("94") says nothing about what a node does, and the full
    catalogue name ("GOSC_H1_L2_L_TO_RD") does not fit in a node. These carry
    the part of the name that is behaviour plus the timing parameter that
    distinguishes members of the same family; routing is already visible as the
    drawn wires, so it is deliberately dropped.
    """
    behavior = entry.behavior
    if behavior == "VETO":
        # Three characters is what fits inside a node at this scale; the full
        # word overflowed the marker and rendered as "ETO".
        return "V"
    if behavior in ("AND", "OR", "XOR"):
        return behavior
    if behavior == "DELAY":
        return "D%d" % entry.duration
    if behavior == "NORMALIZER":
        return "N%d" % entry.duration
    if behavior == "HOLD":
        return "H%d" % entry.duration
    if behavior == "C_ELEMENT":
        return "C"
    if behavior == "TOGGLE":
        return "T"
    if behavior == "GATED_OSCILLATOR":
        return "O%d%d" % (entry.high_time, entry.low_time)
    return str(entry.id)


#: Expansion of the short node text, for the function-colour legend.
FUNCTION_LEGEND = (
    ("LOGIC", "AND/OR/XOR  V = veto"),
    ("DELAY", "Dn = delay n"),
    ("NORMALIZER", "Nn = 1 pulse of n"),
    ("HOLD", "Hn = extend by n"),
    ("C_ELEMENT", "C = Muller C"),
    ("TOGGLE", "T = toggle"),
    ("GATED_OSCILLATOR", "Ohl = osc high/low"),
)


def draw_functional_net(ax, grid, *, input_positions=(),
                        output_positions=None, activity=None,
                        show_edges=True, title=None, branches=None,
                        root_positions=None, color_by=None, legend=True,
                        extent=None):
    """Draw components and their actual outbound directed wires.

    ``branches`` is the developmental trace's ``owners`` map, cell -> the branch
    that last changed it.

    ``color_by`` picks what the fill and the in-node text mean. The two views
    answer different questions and neither subsumes the other:

    * ``'branch'`` - which developmental arm built each cell, labelled with the
      branch number. This is the ontogeny view: it shows whether the arms are
      doing separate work or duplicating each other.
    * ``'function'`` - what each node physically IS, coloured by catalogue
      family and labelled with :func:`component_label`. This is the hardware
      view: it shows the circuit as a parts list.

    Defaults to ``'branch'`` when a ``branches`` map is supplied and
    ``'function'`` otherwise, which is what every caller predating the option
    already expected. Inputs and outputs are ringed under both.

    ``legend`` draws the colour key. A panel series should switch it off for
    all but one panel: nine copies of the same key is clutter, and on a narrow
    panel the key costs more space than the body it explains. ``extent`` pins
    the axis limits so a series of panels shares one scale (see
    :func:`body_extent`).
    """
    if color_by is None:
        color_by = 'branch' if branches is not None else 'function'
    if color_by not in ('branch', 'function'):
        raise ValueError('color_by must be "branch" or "function", not %r'
                         % (color_by,))
    by_branch = (color_by == 'branch' and branches is not None)
    output_positions = output_positions or {}
    root_positions = root_positions or {}
    output_cells = {
        cell: role for role, cell in output_positions.items()
        if cell is not None
    }
    root_cells = {
        cell: role for role, cell in root_positions.items()
        if cell is not None
    }
    input_cells = {
        tuple(cell): index for index, cell in enumerate(input_positions)
    }
    activity = activity or {}
    # Keep dead genetic roots visible. Silence is a legitimate phenotype, but a
    # missing component must still look like an output niche rather than vanish
    # from the view and masquerade as a fitted readout elsewhere.
    visible_cells = set(grid) | set(output_cells) | set(root_cells)
    pixels = {cell: hex_pixel(*cell) for cell in visible_cells}
    if show_edges:
        # Mirror FunctionalSim's receiver-side wiring test exactly. External
        # source pads can drive every adjacent input-facing component no matter
        # which function happens to be pinned at that developmental seed.
        for destination, state in grid.items():
            entry = BY_ID[state]
            for direction in entry.inputs:
                source = source_for_input(destination, direction)
                if source not in grid:
                    continue
                source_entry = BY_ID[grid[source]]
                facing = facing_direction(source, destination)
                if (source not in input_cells
                        and facing not in source_entry.outputs):
                    continue
                x0, y0 = pixels[source]
                x1, y1 = pixels[destination]
                ax.add_patch(FancyArrowPatch(
                    (x0, y0), (x1, y1), arrowstyle="-|>",
                    mutation_scale=6, linewidth=0.75, color="#777777",
                    alpha=0.65, shrinkA=7, shrinkB=7,
                    connectionstyle="arc3,rad=0.10"))
    for cell, state in grid.items():
        entry = BY_ID[state]
        x, y = pixels[cell]
        active = bool(activity.get(cell, 0))
        fill = (branch_color(branches.get(cell)) if by_branch
                else FAMILY_COLORS.get(entry.family, "#aaaaaa"))
        if cell in root_cells:
            ax.scatter([x], [y], s=220, facecolors="none",
                       edgecolors=ROOT_COLOR, linewidths=2.2, zorder=2.8)
        label = component_label(entry)
        if cell in input_cells:
            label = f"I{input_cells[cell]}"
        elif cell in output_cells:
            label = output_cells[cell][:3]
        elif by_branch:
            label = str(branches.get(cell, "-"))
        # I/O ring first, so a port is identifiable whatever the fill says.
        if cell in input_cells:
            edge, width, size = INPUT_COLOR, 2.4, 165
        elif cell in output_cells:
            edge, width, size = OUTPUT_COLOR, 2.4, 165
        else:
            # A three-character function name ("XOR", "O12") needs a wider node
            # than a branch number does, so the marker follows the label.
            edge, width, size = "#222222", 0.7, (105 if len(label) <= 2 else 140)
        if active:
            edge, width, size = "#ffea00", max(width, 1.6), max(size, 130)
        ax.scatter([x], [y], s=size, c=[fill], edgecolors=edge,
                   linewidths=width, zorder=3)
        # Shrink the longer labels rather than let them spill past the marker,
        # where the white text lands on the light background and vanishes.
        ax.text(x, y, label, ha="center", va="center",
                fontsize=5.5 if len(label) <= 2 else 4.3,
                color="white", fontweight="bold", zorder=4)
    for cell in sorted((set(output_cells) | set(root_cells)) - set(grid)):
        x, y = pixels[cell]
        if cell in root_cells:
            ax.scatter([x], [y], s=220, facecolors="none",
                       edgecolors=ROOT_COLOR, linewidths=2.2, zorder=3)
        if cell in output_cells:
            ax.scatter([x], [y], s=155, facecolors="none",
                       edgecolors=OUTPUT_COLOR, linewidths=2.4, zorder=3.1)
            label = output_cells[cell][:3]
        else:
            label = "R:" + root_cells[cell][:3]
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5,
                color="#333333", fontweight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    if extent is not None:
        x0, x1, y0, y1 = extent
    elif pixels:
        xs, ys = zip(*pixels.values())
        x0, x1 = min(xs) - 0.8, max(xs) + 0.8
        y0, y1 = min(ys) - 0.8, max(ys) + 0.8
    else:
        x0 = x1 = y0 = y1 = None
    if x0 is not None:
        if legend:
            # Reserve a band under the body for the key. Drawn straight onto
            # the axes it landed on top of the lowest nodes, which is how a
            # legend ends up hiding the thing it explains.
            y0 -= 0.30 * max(y1 - y0, 1.0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    if legend:
        draw_key(ax, grid, branches if by_branch else None)
    if title:
        ax.set_title(title, fontsize=7)


def body_extent(grids, *extra_cells):
    """Shared (x0, x1, y0, y1) covering every supplied body.

    Growth panels each used their own limits, so a two-cell seed was magnified
    to fill its panel while a forty-cell body was crammed into the same space -
    the sequence could not be read as growth because nothing held still between
    frames. One extent across the series fixes the scale so a node that does
    not move stays put.
    """
    cells = set()
    for grid in grids:
        cells |= set(grid)
    for group in extra_cells:
        cells |= {tuple(cell) for cell in group if cell is not None}
    if not cells:
        return None
    xs, ys = zip(*(hex_pixel(*cell) for cell in cells))
    return (min(xs) - 0.8, max(xs) + 0.8, min(ys) - 0.8, max(ys) + 0.8)


def draw_key(holder, grid, branches=None):
    """Colour key for one body, as a real legend.

    Hand-placed text was laying entries out in axes fractions while matplotlib
    sized them in points, so on a narrow panel they overlapped into an
    unreadable smear. A legend measures its own entries.

    ``holder`` may be an Axes or a Figure. A panel SERIES wants the figure: one
    key for the whole grid, clear of every panel, instead of a copy shrinking
    each one.
    """
    if branches is not None:
        present = sorted({int(value) for value in branches.values()})
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=branch_color(value),
                          markeredgecolor="#222222", markeredgewidth=0.4,
                          label="branch %d" % value)
                   for value in present]
        empty = "no branch owns a cell"
    else:
        legend_text = dict(FUNCTION_LEGEND)
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=FAMILY_COLORS.get(family, "#aaaaaa"),
                          markeredgecolor="#222222", markeredgewidth=0.4,
                          label=legend_text[family])
                   for family, _text in FUNCTION_LEGEND
                   if any(BY_ID[state].family == family
                          for state in grid.values())]
        empty = "no components"
    on_figure = hasattr(holder, "add_subplot")
    if not handles:
        if not on_figure:
            holder.text(0.01, 0.01, empty, transform=holder.transAxes,
                        fontsize=6, color="#444444")
        return
    # A figure-level key has the whole width, so it lays out in one row; an
    # axes key is narrow and wraps. Either way the legend measures its own
    # entries, so they cannot overlap each other.
    holder.legend(
        handles=handles, loc="lower left", fontsize=5.5, frameon=False,
        ncol=len(handles) if on_figure else (2 if len(handles) > 3 else 1),
        handletextpad=0.4, columnspacing=0.9, labelspacing=0.25,
        borderpad=0.1, borderaxespad=0.1)
