"""Matplotlib rendering for Functional NV Net bodies."""
from __future__ import annotations

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


def draw_functional_net(ax, grid, *, input_positions=(),
                        output_positions=None, activity=None,
                        show_edges=True, title=None, branches=None,
                        root_positions=None):
    """Draw components and their actual outbound directed wires.

    ``branches`` is the developmental trace's ``owners`` map, cell -> the branch
    that last changed it. Supplying it colours the body by WHICH BRANCH BUILT
    WHAT rather than by component family, which is what shows whether the arms
    are doing separate work. Inputs and outputs are ringed either way.
    """
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
        fill = (branch_color(branches.get(cell)) if branches is not None
                else FAMILY_COLORS.get(entry.family, "#aaaaaa"))
        if cell in root_cells:
            ax.scatter([x], [y], s=220, facecolors="none",
                       edgecolors=ROOT_COLOR, linewidths=2.2, zorder=2.8)
        # I/O ring first, so a port is identifiable whatever the fill says.
        if cell in input_cells:
            edge, width, size = INPUT_COLOR, 2.4, 165
        elif cell in output_cells:
            edge, width, size = OUTPUT_COLOR, 2.4, 165
        else:
            edge, width, size = "#222222", 0.7, 105
        if active:
            edge, width, size = "#ffea00", max(width, 1.6), max(size, 130)
        ax.scatter([x], [y], s=size, c=[fill], edgecolors=edge,
                   linewidths=width, zorder=3)
        label = str(entry.id)
        if cell in input_cells:
            label = f"I{input_cells[cell]}"
        elif cell in output_cells:
            label = output_cells[cell][:3]
        elif branches is not None:
            label = str(branches.get(cell, "-"))
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5,
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
    if branches is not None:
        present = sorted({int(v) for v in branches.values()})
        legend = "  ".join("branch %d" % value for value in present)
        ax.text(0.01, 0.01, legend or "no branch owns a cell",
                transform=ax.transAxes, fontsize=6, color="#444444")
    ax.set_aspect("equal")
    ax.axis("off")
    if pixels:
        xs, ys = zip(*pixels.values())
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    if title:
        ax.set_title(title, fontsize=7)
