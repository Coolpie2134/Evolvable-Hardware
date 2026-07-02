"""
nv_evo/viz.py — honeycomb rendering for the nervous net.

Draws the hex array with each node's excitatory (green) and inhibitory (red)
wiring, so the loops the paper cares about are visible. Used by the GUI's
growth / activity / interactive views.
"""
from __future__ import annotations
from matplotlib.patches import RegularPolygon

from .hexgrid import hex_dirs, hex_pixel, routing_kind, ROUTING_HEX

_KIND_FC = {'buffer': '#8fb3e0', 'coincidence': '#2f6fc0',
            'inhibited': '#e0902e', 'off': '#e8e8e8'}
_EXC = '#2e8b57'      # excitatory wire
_INH = '#d0332e'      # inhibitory wire


def draw_hex_net(ax, grid, grid_size, routing=None, in_pos=None, out_pos=None,
                 activity=None, show_edges=True, title=None):
    """
    grid       : {(x,y): state}
    routing    : {(x,y): (e1,e2,i1)} (for edges / node-type colour)
    in_pos     : list of input positions (labelled A, B, …)
    out_pos    : {role: (x,y)}  outputs (labelled)
    activity   : {(x,y): 0/1}   if given, nodes coloured by activity not type
    """
    in_pos  = in_pos or []
    out_pos = out_pos or {}
    in_set  = set(in_pos)
    ax.clear()
    ax.set_aspect('equal'); ax.axis('off')

    # wires first (under the nodes): each node's inputs, coloured by role.
    # Full centre-to-centre lines show through the gaps between hexes -> loops.
    if show_edges and routing:
        for (x, y), (e1, e2, i1) in routing.items():
            px, py = hex_pixel(x, y); nb = hex_dirs(x, y)
            for d, col in ((e1, _EXC), (e2, _EXC), (i1, _INH)):
                if d is None:
                    continue
                nx, ny = nb[d]
                if (nx, ny) in grid:
                    qx, qy = hex_pixel(nx, ny)
                    ax.plot([px, qx], [py, qy], color=col, lw=2.2, alpha=0.9,
                            zorder=1, solid_capstyle='round')

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
        ax.add_patch(RegularPolygon((px, py), numVertices=6, radius=0.46,
                                    orientation=0, facecolor=fc, alpha=0.92,
                                    edgecolor='#b02020' if inp else '#5b6b7d',
                                    lw=2.0 if inp else 0.7, zorder=2))

    for i, (x, y) in enumerate(in_pos):
        px, py = hex_pixel(x, y)
        ax.text(px, py, chr(65 + i) if i < 26 else '*', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=3)
    for role, p in out_pos.items():
        if p and p in grid:
            px, py = hex_pixel(*p)
            ax.text(px, py, role[:2], ha='center', va='center',
                    fontsize=7, color='#111', fontweight='bold', zorder=3)

    xs = [hex_pixel(x, y)[0] for (x, y) in grid] or [0]
    ys = [hex_pixel(x, y)[1] for (x, y) in grid] or [0]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)
