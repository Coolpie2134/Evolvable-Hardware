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


def draw_functional_net(ax, grid, *, input_positions=(),
                        output_positions=None, activity=None,
                        show_edges=True, title=None):
    """Draw components and their actual outbound directed wires."""
    output_positions = output_positions or {}
    output_cells = {
        cell: role for role, cell in output_positions.items()
        if cell is not None
    }
    input_cells = {
        tuple(cell): index for index, cell in enumerate(input_positions)
    }
    activity = activity or {}
    pixels = {cell: hex_pixel(*cell) for cell in grid}
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
        edge = "#ffea00" if active else "#222222"
        size = 130 if active else 105
        ax.scatter(
            [x], [y], s=size, c=[FAMILY_COLORS.get(entry.family, "#aaaaaa")],
            edgecolors=edge, linewidths=1.6 if active else 0.7, zorder=3)
        label = str(entry.id)
        if cell in input_cells:
            label = f"I{input_cells[cell]}"
        elif cell in output_cells:
            label = output_cells[cell][:3]
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    if pixels:
        xs, ys = zip(*pixels.values())
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    if title:
        ax.set_title(title, fontsize=7)
