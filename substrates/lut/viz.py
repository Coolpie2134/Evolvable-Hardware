"""
substrates/lut/viz.py — square-array rendering for the LUT model (paper Architecture 2).

A cell is NOT fixed directional wiring — it is FOUR 16-bit lookup tables, one per
output direction (Ln, Ls, Le, Lw). During growth (associative-memory ontogeny)
each cell's four LUT *states* are rewritten by a table lookup on its neighbours'
states, so a mature organism is a field of position-dependent LUT states — the
paper's Fig. 13 ("unique colours are assigned to unique LUT states, one pixel per
LUT"). This renderer shows exactly that:

  * a faint grey LATTICE joins physically-adjacent live cells (degree-4);
  * each cell is drawn as FOUR directional WEDGES (N/E/S/W), each coloured by
    that direction's 16-bit LUT STATE — equal states read the same colour, so
    the internal state structure and its variation with position are visible (a
    dead direction, LUT == 0, is near-white). This is the paper's "one pixel per
    LUT" made directional.
  * given an ACTIVITY map (each cell's 4-bit N/S/E/W output this tick), the
    wedges instead show each directional OUTPUT BIT as green (1) or red (0) —
    the paper's Fig. 14 "instantaneous digital state ... coloured in binary
    values red and green" — so the running dynamics can be watched tick by tick.

Input seeds are ringed red (labelled A, B, …), outputs ringed green/labelled.
"""
from __future__ import annotations
import colorsys

from matplotlib.patches import Polygon, Rectangle

# output-nibble bit layout (matches lut.py LutSim.step): N=8 S=4 E=2 W=1
_BIT  = {'N': 0x8, 'S': 0x4, 'E': 0x2, 'W': 0x1}
# LUT-tuple index per direction: state = (Ln, Ls, Le, Lw)
_LIDX = {'N': 0, 'S': 1, 'E': 2, 'W': 3}

_LATTICE   = '#d4d9de'      # physical adjacency
_DEAD      = (0.95, 0.96, 0.97)   # a direction whose LUT is 0 (dead)
# Fig. 14 "instantaneous digital state ... coloured in binary values red and
# green": each directional output bit is green (1) or red (0) this tick.
_EMIT      = '#1ea64a'      # output bit = 1 (green)
_LOW       = '#d64a3c'      # output bit = 0 (red)
_WEDGE_EC  = '#c3ccd6'
_CELL_EC   = '#5b6b7d'
_SEED_EC   = '#b02020'
_OUT_EC    = '#17902f'

_R = 0.40                   # half-size of a cell square (< 0.5 so lattice shows)


def _lut_color(v):
    """Stable colour for a 16-bit LUT state: equal states -> equal colour,
    distinct states well separated (golden-ratio hue hash). 0 = dead = near-white."""
    if not v:
        return _DEAD
    h = (v * 0.6180339887498949) % 1.0
    return colorsys.hsv_to_rgb(h, 0.52, 0.93)


def _wedges(x, y, r):
    """The four directional triangles of a cell (apex at centre, base on a side)."""
    tl, tr = (x - r, y + r), (x + r, y + r)
    br, bl = (x + r, y - r), (x - r, y - r)
    c = (x, y)
    return {'N': (c, tl, tr), 'E': (c, tr, br),
            'S': (c, br, bl), 'W': (c, bl, tl)}


def draw_lut_net(ax, grid, grid_size=None, activity=None, in_pos=None,
                 out_pos=None, show_edges=True, title=None):
    """
    grid     : {(x,y): (Ln, Ls, Le, Lw)}  grown 4-LUT cell states
    activity : {(x,y): nibble}  4-bit N/S/E/W emission this tick — if given, the
               wedges show firing (green) / silent instead of LUT-state colour
    in_pos   : list of input positions (labelled A, B, …)
    out_pos  : {role: (x,y)} outputs (green-ringed, labelled)
    """
    in_pos  = in_pos or []
    out_pos = out_pos or {}
    in_set  = set(in_pos)
    from substrates.nervous.io_placement import output_groups
    out_cells = {p: role for role, cells in output_groups(out_pos).items()
                 for p in cells}
    ax.clear()
    ax.set_aspect('equal'); ax.axis('off')
    if not grid:
        if title:
            ax.set_title(title, fontsize=9)
        return

    # physical lattice (bottom layer): join every adjacent live pair (degree-4)
    for (x, y) in grid:
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in grid:
                ax.plot([x, x + dx], [y, y + dy], color=_LATTICE, lw=0.8,
                        zorder=0, solid_capstyle='round')

    for (x, y), state in grid.items():
        wed = _wedges(x, y, _R)
        nib = activity.get((x, y), 0) if activity is not None else None
        for d in ('N', 'E', 'S', 'W'):
            if activity is not None:                 # Fig. 14: output bit green/red
                fc = _EMIT if (nib & _BIT[d]) else _LOW
            else:                                    # Fig. 13: colour = LUT state
                fc = _lut_color(state[_LIDX[d]])
            ax.add_patch(Polygon(wed[d], closed=True, facecolor=fc,
                                 edgecolor=_WEDGE_EC, lw=0.3, zorder=2))
        # cell outline / IO ring
        seed = (x, y) in in_set
        outp = (x, y) in out_cells
        ax.add_patch(Rectangle((x - _R, y - _R), 2 * _R, 2 * _R, fill=False,
                     edgecolor=_SEED_EC if seed else _OUT_EC if outp else _CELL_EC,
                     lw=2.2 if (seed or outp) else 0.6, zorder=3))

    for i, (x, y) in enumerate(in_pos):
        if (x, y) in grid:
            ax.text(x, y, chr(65 + i) if i < 26 else '*', ha='center', va='center',
                    fontsize=8, color='#111', fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
    for p, role in out_cells.items():
        ax.text(p[0], p[1], role[:2], ha='center', va='center', fontsize=7,
                color='#0d5b1f', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

    xs = [x for (x, _) in grid]; ys = [y for (_, y) in grid]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)


def draw_lut_table(ax, lut, swatch=None, title=None):
    """Draw the ACTUAL 16-bit lookup table as a 4x4 truth grid: columns vary the
    two bits coming from N,S; rows vary the two from E,W; a cell is green when
    the table outputs 1 for that neighbour input and white when it outputs 0.
    This is the raw table itself (out = table[N + 2S + 4E + 8W]) — the thing the
    wedge colours only hash. `swatch` (an RGB) draws the matching net colour so a
    table can be tied back to the cells; `title` labels it (hex / logic)."""
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    lut &= 0xFFFF
    for r in range(4):                     # r = (E, W) bits -> row (top = 0)
        e, w = r & 1, (r >> 1) & 1
        for c in range(4):                 # c = (N, S) bits -> column
            n, s = c & 1, (c >> 1) & 1
            idx = n | (s << 1) | (e << 2) | (w << 3)
            on = (lut >> idx) & 1
            ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=_EMIT if on else 'white',
                                   edgecolor='#b9c2cc', lw=0.6, zorder=1))
    # axis labels: which neighbour-input each column / row stands for
    for c in range(4):
        ax.text(c + 0.5, 4.08, '%d%d' % (c & 1, (c >> 1) & 1),
                ha='center', va='bottom', fontsize=5, color='#555')
    ax.text(2, 4.62, 'N S', ha='center', va='bottom', fontsize=6, color='#333')
    for r in range(4):
        ax.text(-0.12, 3 - r + 0.5, '%d%d' % (r & 1, (r >> 1) & 1),
                ha='right', va='center', fontsize=5, color='#555')
    ax.text(-0.75, 2, 'E W', ha='center', va='center', fontsize=6,
            color='#333', rotation=90)
    if swatch is not None:
        ax.add_patch(Rectangle((0, -0.85), 0.7, 0.6, facecolor=swatch,
                               edgecolor='#5b6b7d', lw=0.5, zorder=1))
        ax.text(0.85, -0.55, 'net colour', ha='left', va='center', fontsize=5,
                color='#555')
    ax.set_xlim(-1.1, 4.4)
    ax.set_ylim(-1.1, 4.9)
    if title:
        ax.set_title(title, fontsize=6.5, pad=3)
