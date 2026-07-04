"""
nv_evo/viz.py — honeycomb rendering for the nervous net.

Nodes are drawn as CIRCLES at the *corners* of a hex array ("cells are
positioned at the corners of a hex array" — automaton_arrays), i.e. on
triangular zigzag rows joined by vertical rungs, so it is visually unambiguous
that every node touches exactly THREE others (not six):

  * a faint grey LATTICE between every pair of adjacent cells shows the
    physical degree-3 connectivity;
  * routed signals are ARROWS from the source cell to the reader — green for
    excitatory inputs, red for the inhibitory veto — so loops and their
    direction of circulation can be followed by eye;
  * each cell is coloured by its routing kind and labelled with its 4-bit
    STATE NUMBER, so the colour/type can be mapped back to the genome's
    states.

Used by the GUI's growth / activity / interactive views.
"""
from __future__ import annotations
import math

from matplotlib.patches import Circle, FancyArrowPatch

from .hexgrid import hex_dirs, hex_pixel, routing_kind, ROUTING_HEX

_KIND_FC = {'buffer': '#8fb3e0', 'coincidence': '#2f6fc0',
            'inhibited': '#e0902e', 'off': '#e8e8e8'}
_EXC     = '#2e8b57'      # excitatory wire
_INH     = '#d0332e'      # inhibitory wire
_LATTICE = '#d4d9de'      # physical adjacency (unrouted)


_R = 0.28               # node radius (small vs the unit edge so wires show)


def _arrow(ax, src, dst, col):
    """Directed wire src -> dst: starts at the source cell's edge, its head
    lands on the reader's edge, offset a little to the right of travel so a
    pair of opposite arrows (a 2-loop) doesn't overlap."""
    (qx, qy), (px, py) = src, dst
    dx, dy = px - qx, py - qy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    ox, oy = 0.07 * uy, -0.07 * ux                 # perpendicular offset
    a = (qx + ux * _R + ox, qy + uy * _R + oy)     # source edge
    b = (px - ux * (_R + 0.03) + ox, py - uy * (_R + 0.03) + oy)  # reader edge
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8,
                                 shrinkA=0, shrinkB=0, color=col,
                                 lw=1.7, alpha=0.95, zorder=3))


def draw_hex_net(ax, grid, grid_size, routing=None, in_pos=None, out_pos=None,
                 activity=None, show_edges=True, title=None):
    """
    grid       : {(x,y): state}
    routing    : {(x,y): (e1,e2,i1)} (for arrows / node-type colour)
    in_pos     : list of input positions (labelled A, B, …)
    out_pos    : {role: (x,y)}  outputs (labelled)
    activity   : {(x,y): 0/1}   if given, nodes coloured by activity not type
    """
    in_pos  = in_pos or []
    out_pos = out_pos or {}
    in_set  = set(in_pos)
    ax.clear()
    ax.set_aspect('equal'); ax.axis('off')

    # physical lattice first (bottom layer): every adjacent live pair — makes
    # it obvious each cell connects to exactly 3 neighbours.
    drawn = set()
    for (x, y) in grid:
        px, py = hex_pixel(x, y)
        for (nx, ny) in hex_dirs(x, y).values():
            if (nx, ny) in grid and ((nx, ny), (x, y)) not in drawn:
                drawn.add(((x, y), (nx, ny)))
                qx, qy = hex_pixel(nx, ny)
                ax.plot([px, qx], [py, qy], color=_LATTICE, lw=0.8,
                        zorder=0, solid_capstyle='round')

    # routed signals as direction arrows (source -> reader)
    if show_edges and routing:
        for (x, y), (e1, e2, i1) in routing.items():
            px, py = hex_pixel(x, y); nb = hex_dirs(x, y)
            wires = {(e1, _EXC), (e2, _EXC)}       # set: a buffer's e1==e2 once
            wires.add((i1, _INH))
            for d, col in wires:
                if d is None:
                    continue
                if nb[d] in grid:
                    _arrow(ax, hex_pixel(*nb[d]), (px, py), col)

    # nodes (a bit smaller than the pitch so the wires stay visible)
    for (x, y), state in grid.items():
        px, py = hex_pixel(x, y)
        if activity is not None:
            fc = '#18b34a' if activity.get((x, y), 0) else '#e6e9ee'
        elif routing is not None:
            fc = _KIND_FC[routing_kind(ROUTING_HEX[state & 0xF])]
        else:
            fc = '#8fb3e0'
        inp = (x, y) in in_set
        ax.add_patch(Circle((px, py), radius=_R, facecolor=fc, alpha=0.95,
                            edgecolor='#b02020' if inp else '#5b6b7d',
                            lw=2.0 if inp else 0.7, zorder=2))
        # state number: maps the colour/type back to the genome's 0-15 states
        ax.text(px, py - 0.13, str(state & 0xF), ha='center', va='center',
                fontsize=4.5, color='#3a4450', zorder=4)

    for i, (x, y) in enumerate(in_pos):
        px, py = hex_pixel(x, y)
        ax.text(px, py + 0.08, chr(65 + i) if i < 26 else '*',
                ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=4)
    for role, p in out_pos.items():
        if p and p in grid:
            px, py = hex_pixel(*p)
            ax.text(px, py + 0.08, role[:2], ha='center', va='center',
                    fontsize=6.5, color='#111', fontweight='bold', zorder=4)

    xs = [hex_pixel(x, y)[0] for (x, y) in grid] or [0]
    ys = [hex_pixel(x, y)[1] for (x, y) in grid] or [0]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)
