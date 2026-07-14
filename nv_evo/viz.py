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
  * every physical tile contains three visibly separate L/R/D core circuits.
    Routed arrows terminate on the exact circuit they configure, rather than at
    a shared tile centre where valid fan-out looks like contradictory wiring.

Used by the GUI's growth / activity / interactive views.
"""
from __future__ import annotations
import math

from matplotlib.patches import Circle, FancyArrowPatch

from .hexgrid import (hex_dirs, hex_pixel, routing_kind, ROUTING_HEX,
                      DIRECTIONS, decode_tile_routing, is_directional_routing,
                      tile_channel, channel_tile, is_channel,
                      unpack_tile_state, directional_connections)

_KIND_FC = {'buffer': '#8fb3e0', 'coincidence': '#2f6fc0', 'or': '#7b52c4',
            'inhibited': '#e0902e', 'off': '#e8e8e8'}
_EXC     = '#2e8b57'      # excitatory wire
_INH     = '#d0332e'      # inhibitory wire
_LATTICE = '#d4d9de'      # physical adjacency (unrouted)


_TILE_R = 0.31          # physical three-circuit package
_CORE_R = 0.105         # one Figure-3 nervous-network circuit
_CORE_OFFSET = 0.145


def channel_pixel(channel):
    """Drawing centre of one exact ``(x,y,L/R/D)`` core circuit."""
    x, y, direction = channel
    px, py = hex_pixel(x, y)
    nx, ny = hex_pixel(*hex_dirs(x, y)[direction])
    dx, dy = nx - px, ny - py
    length = math.hypot(dx, dy) or 1.0
    return (px + _CORE_OFFSET * dx / length,
            py + _CORE_OFFSET * dy / length)


def _arrow(ax, src, dst, col, lane=0.0, radius=_CORE_R, inhibitory=False):
    """Draw an excitatory arrow or inhibitory T-bar between core circuits."""
    (qx, qy), (px, py) = src, dst
    dx, dy = px - qx, py - qy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    offset = 0.025 + lane
    ox, oy = offset * uy, -offset * ux             # perpendicular offset
    a = (qx + ux * radius + ox, qy + uy * radius + oy)
    b = (px - ux * (radius + 0.018) + ox,
         py - uy * (radius + 0.018) + oy)
    if inhibitory:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.7,
                alpha=0.95, zorder=3, solid_capstyle='round')
        cap = 0.045
        ax.plot([b[0] - cap * uy, b[0] + cap * uy],
                [b[1] + cap * ux, b[1] - cap * ux],
                color=col, lw=2.0, alpha=0.98, zorder=3,
                solid_capstyle='round')
    else:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8,
                                     shrinkA=0, shrinkB=0, color=col,
                                     lw=1.7, alpha=0.95, zorder=3))


def draw_hex_net(ax, grid, grid_size, routing=None, in_pos=None, out_pos=None,
                 activity=None, show_edges=True, title=None):
    """
    grid       : {(x,y): state}
    routing    : {(x,y): (L_entry,R_entry,D_entry)} for paper tiles
    in_pos     : list of input positions (labelled A, B, …)
    out_pos    : {role: (x,y,direction)} exact output channels (labelled)
    activity   : tile and/or channel activity; core circles show channel values
    """
    in_pos  = in_pos or []
    out_pos = out_pos or {}
    if routing is None and grid:
        routing = {pos: decode_tile_routing(state)
                   for pos, state in grid.items()}
    in_set  = set(in_pos)
    directional = is_directional_routing(routing)
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

    # Routed signals terminate at exact core circuits.  In particular, an
    # E1=E2 buffer is one green nerve fanned into two internal terminals, not
    # two drawn excitatory connections.
    if show_edges and routing and directional:
        for source, destination, role in directional_connections(grid, routing):
            _arrow(ax, channel_pixel(source), channel_pixel(destination),
                   _EXC if role == 'exc' else _INH,
                   inhibitory=(role == 'inh'))
    elif show_edges and routing:
        for (x, y), entry in routing.items():
            px, py = hex_pixel(x, y); nb = hex_dirs(x, y)
            e1, e2, i1 = entry[0], entry[1], entry[2]
            for d in dict.fromkeys((e1, e2)):
                if d is not None and nb[d] in grid:
                    _arrow(ax, hex_pixel(*nb[d]), (px, py), _EXC,
                           radius=_TILE_R)
            if i1 is not None and nb[i1] in grid:
                _arrow(ax, hex_pixel(*nb[i1]), (px, py), _INH,
                       radius=_TILE_R, inhibitory=True)

    # nodes (a bit smaller than the pitch so the wires stay visible)
    for (x, y), state in grid.items():
        px, py = hex_pixel(x, y)
        inp = (x, y) in in_set
        if directional:
            entries = routing.get((x, y), decode_tile_routing(state))
            selectors = unpack_tile_state(state)
            # The outer package is one physical Figure-2 node.  The three inner
            # circles are its independent output circuits and each owns a 0-F
            # Figure-3 configuration.
            ax.add_patch(Circle((px, py), radius=_TILE_R,
                                facecolor='#f7f8fa', alpha=0.92,
                                edgecolor='#b02020' if inp else '#8a97a5',
                                lw=2.0 if inp else 0.8, zorder=2))
            for direction, entry, selector in zip(DIRECTIONS, entries, selectors):
                wire = tile_channel((x, y), direction)
                cx, cy = channel_pixel(wire)
                if activity is not None:
                    on = activity.get(wire, activity.get((x, y), 0))
                    fc = '#18b34a' if on else '#e6e9ee'
                else:
                    fc = _KIND_FC[routing_kind(entry)]
                ax.add_patch(Circle((cx, cy), radius=_CORE_R, facecolor=fc,
                                    edgecolor='#536273', lw=0.55, alpha=0.98,
                                    zorder=3))
                ax.text(cx, cy, '%s%X' % (direction, selector), ha='center',
                        va='center', fontsize=3.4, color='#202b36', zorder=4)
        else:
            if activity is not None:
                fc = '#18b34a' if activity.get((x, y), 0) else '#e6e9ee'
            elif routing is not None:
                fc = _KIND_FC[routing_kind(ROUTING_HEX[state & 0x1F])]
            else:
                fc = '#8fb3e0'
            ax.add_patch(Circle((px, py), radius=_TILE_R, facecolor=fc, alpha=0.95,
                                edgecolor='#b02020' if inp else '#5b6b7d',
                                lw=2.0 if inp else 0.7, zorder=2))
            ax.text(px, py - 0.13, str(state & 0x1F), ha='center', va='center',
                    fontsize=4.5, color='#3a4450', zorder=4)

    for i, (x, y) in enumerate(in_pos):
        px, py = hex_pixel(x, y)
        ax.text(px, py + 0.08, chr(65 + i) if i < 26 else '*',
                ha='center', va='center',
                fontsize=7, color='#8b1e1e', fontweight='bold', zorder=5)
    for role, p in out_pos.items():
        tile = channel_tile(p) if p else None
        if tile and tile in grid:
            if is_channel(p):
                cx, cy = channel_pixel(p)
                tx, ty = hex_pixel(*tile)
                dx, dy = cx - tx, cy - ty
                length = math.hypot(dx, dy) or 1.0
                px, py = (cx + 0.23 * dx / length,
                          cy + 0.23 * dy / length)
            else:
                px, py = hex_pixel(*tile)
            label = role[:2] + (':' + p[2] if is_channel(p) else '')
            ax.text(px, py, label, ha='center', va='center',
                    fontsize=6.2, color='#111', fontweight='bold', zorder=5,
                    bbox={'facecolor': 'white', 'edgecolor': 'none',
                          'alpha': 0.78, 'pad': 0.25})

    xs = [hex_pixel(x, y)[0] for (x, y) in grid] or [0]
    ys = [hex_pixel(x, y)[1] for (x, y) in grid] or [0]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)
