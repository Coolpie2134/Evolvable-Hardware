"""
substrates/nervous/viz.py - honeycomb rendering for the nervous net.

Nodes are drawn as CIRCLES at the *corners* of a hex array ("cells are
positioned at the corners of a hex array" - automaton_arrays), i.e. on
triangular zigzag rows joined by vertical rungs, so it is visually unambiguous
that every node touches exactly THREE others (not six):

  * a faint grey LATTICE between every pair of adjacent cells shows the
    physical degree-3 connectivity;
  * routed signals are ARROWS from the source cell to the reader - green for
    excitatory inputs, red for the inhibitory veto - so loops and their
    direction of circulation can be followed by eye;
  * each cell is coloured by its routing kind and labelled with its 5-bit
    STATE NUMBER (0-31: 0-15 AND, 16-31 OR), so the colour/type can be mapped
    back to the genome's states.

Used by the GUI's growth / activity / interactive views.
"""
from __future__ import annotations
import math

from matplotlib.patches import Circle, FancyArrowPatch

from .hexgrid import hex_dirs, hex_pixel, routing_kind, ROUTING_HEX
from .tritile import channel_configs

_KIND_FC = {'buffer': '#8fb3e0', 'coincidence': '#2f6fc0', 'or': '#7b52c4',
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


_IDLE_RGB   = (0xe6, 0xe9, 0xee)
_ACTIVE_RGB = (0x18, 0xb3, 0x4a)


def _activity_color(level):
    """Idle-grey -> active-green node fill. Accepts the legacy binary 0/1 and
    graded 0..1 charge levels (capacitor-style playback), interpolated
    linearly so a discharging node visibly fades instead of snapping off."""
    level = max(0.0, min(1.0, float(level)))
    return '#%02x%02x%02x' % tuple(
        int(round(idle + (active - idle) * level))
        for idle, active in zip(_IDLE_RGB, _ACTIVE_RGB))


def draw_hex_net(ax, grid, grid_size, routing=None, in_pos=None, out_pos=None,
                 activity=None, show_edges=True, title=None, arch='single'):
    """
    grid       : {(x,y): state}
    routing    : {(x,y): (e1,e2,i1[,op])} (for arrows / node-type colour)
    in_pos     : list of input positions (labelled A, B, ...)
    out_pos    : {role: (x,y)}  outputs (labelled)
    activity   : {(x,y): level} if given, nodes coloured by activity not type;
                 level may be binary 0/1 or a graded 0..1 charge
    arch       : 'single' (5-bit broadcast node) or 'tri3' (three packed
                 directional circuits). Tri tiles are labelled L/R/D in hex
                 and never shown with a fictitious single routing arrow.
    """
    in_pos  = in_pos or []
    out_pos = out_pos or {}
    in_set  = set(in_pos)
    ax.clear()
    ax.set_aspect('equal'); ax.axis('off')

    # physical lattice first (bottom layer): every adjacent live pair - makes
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
    if show_edges and routing and arch == 'single':
        for (x, y), entry in routing.items():
            e1, e2, i1 = entry[0], entry[1], entry[2]
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
            fc = _activity_color(activity.get((x, y), 0))
        elif arch == 'tri3':
            configs = channel_configs(state)
            kinds = [routing_kind(ROUTING_HEX[value]) for value in configs]
            if all(kind == 'off' for kind in kinds):
                fc = _KIND_FC['off']
            elif any(kind == 'inhibited' for kind in kinds):
                fc = _KIND_FC['inhibited']
            elif all(kind == 'buffer' for kind in kinds):
                fc = _KIND_FC['buffer']
            else:
                fc = _KIND_FC['coincidence']
        elif routing is not None:
            fc = _KIND_FC[routing_kind(ROUTING_HEX[state & 0x1F])]
        else:
            fc = '#8fb3e0'
        inp = (x, y) in in_set
        ax.add_patch(Circle((px, py), radius=_R, facecolor=fc, alpha=0.95,
                            edgecolor='#b02020' if inp else '#5b6b7d',
                            lw=2.0 if inp else 0.7, zorder=2))
        # State label maps the colour/type back to the encoded routing. Tri3
        # shows the three channel nibbles in L/R/D order instead of truncating.
        label = (''.join('%X' % value for value in channel_configs(state))
                 if arch == 'tri3' else str(state & 0x1F))
        ax.text(px, py - 0.13, label, ha='center', va='center',
                fontsize=4.5, color='#3a4450', zorder=4)

    for i, (x, y) in enumerate(in_pos):
        px, py = hex_pixel(x, y)
        ax.text(px, py + 0.08, chr(65 + i) if i < 26 else '*',
                ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=4)
    from .io_placement import output_groups
    for role, cells in output_groups(out_pos).items():
        for p in cells:
            if p in grid:
                px, py = hex_pixel(*p)
                ax.text(px, py + 0.08, role[:2], ha='center', va='center',
                        fontsize=6.5, color='#111', fontweight='bold', zorder=4)

    xs = [hex_pixel(x, y)[0] for (x, y) in grid] or [0]
    ys = [hex_pixel(x, y)[1] for (x, y) in grid] or [0]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)
