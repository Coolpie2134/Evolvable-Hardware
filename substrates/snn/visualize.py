from __future__ import annotations
import math
from .genome import GRID_SIZE

SEED_A = (0, 3)
SEED_B = (0, 5)

# -- matplotlib network view (used by the Interactive tab) ---------------------

_EXC     = '#2e8b57'      # excitatory synapse (matches substrates/nervous/viz)
_INH     = '#d0332e'      # inhibitory synapse
_REST    = (0.914, 0.929, 0.949)   # #e9edf2 - membrane at rest
_HOT     = (1.000, 0.353, 0.212)   # #ff5a36 - membrane near threshold
_FLASH   = '#fff17a'      # a neuron spiking THIS frame
_R       = 0.32           # node radius in grid units


def _charge_rgb(frac):
    """Rest->threshold colour ramp: calm light-grey climbing to hot orange."""
    f = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    return tuple(_REST[i] + (_HOT[i] - _REST[i]) * f for i in range(3))


def _syn_arrow(ax, src, dst, col, alpha):
    from matplotlib.patches import FancyArrowPatch
    (qx, qy), (px, py) = src, dst
    dx, dy = px - qx, py - qy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    ox, oy = 0.06 * uy, -0.06 * ux                       # perpendicular offset
    a = (qx + ux * _R + ox, qy + uy * _R + oy)
    b = (px - ux * (_R + 0.02) + ox, py - uy * (_R + 0.02) + oy)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7,
                                 shrinkA=0, shrinkB=0, color=col,
                                 lw=1.3, alpha=alpha, zorder=2))


def draw_snn_net(ax, neurons, synapses, v=None, fired=None,
                 show_edges=True, title=None):
    """Render a grown LIF network for playback.

    neurons  : list[Neuron] (.x .y .vth .is_input .is_output .out_role)
    synapses : list[Synapse] (.pre .post .weight - sign = excit/inhib)
    v        : per-neuron-id membrane potential, or None for the resting view
    fired    : per-neuron-id truthy flag for neurons spiking this frame (a flash)

    Neurons glow from calm grey (rest) to hot orange as their membrane charges
    toward threshold, and flash bright yellow on the step they fire; excitatory
    synapses are green arrows, inhibitory ones red.
    """
    from matplotlib.patches import Circle
    ax.clear()
    ax.set_aspect('equal'); ax.axis('off')
    if not neurons:
        ax.text(0.5, 0.5, '(empty circuit)', ha='center', va='center',
                color='#999', transform=ax.transAxes)
        return
    pos = {n.id: (n.x, n.y) for n in neurons}

    if show_edges:
        for s in synapses:
            if s.pre in pos and s.post in pos:
                lit = (fired is not None and s.pre < len(fired) and fired[s.pre])
                col = _EXC if s.weight >= 0 else _INH
                _syn_arrow(ax, pos[s.pre], pos[s.post], col, 0.95 if lit else 0.22)

    inputs = sorted((n for n in neurons if n.is_input), key=lambda n: (n.x, n.y))
    in_letter = {n.id: (chr(65 + i) if i < 26 else '*') for i, n in enumerate(inputs)}

    for n in neurons:
        frac = 0.0
        if v is not None and n.id < len(v) and n.vth:
            frac = float(v[n.id]) / float(n.vth)
        flash = fired is not None and n.id < len(fired) and fired[n.id]
        fc = _FLASH if flash else _charge_rgb(frac)
        if flash:                                        # soft glow behind a spike
            ax.add_patch(Circle((n.x, n.y), radius=_R * 1.7, facecolor=_FLASH,
                                 alpha=0.35, edgecolor='none', zorder=3))
        ec, lw = '#5b6b7d', 0.8
        if n.is_input:  ec, lw = '#b02020', 2.2
        if n.is_output: ec, lw = '#1a1a1a', 2.2
        ax.add_patch(Circle((n.x, n.y), radius=_R, facecolor=fc, alpha=0.97,
                             edgecolor=ec, lw=lw, zorder=4))
        if n.id in in_letter:
            ax.text(n.x, n.y, in_letter[n.id], ha='center', va='center',
                    fontsize=7, color='#7a1414', fontweight='bold', zorder=6)
        elif n.is_output:
            ax.text(n.x, n.y + _R + 0.18, (n.out_role or 'out')[:3], ha='center',
                    va='bottom', fontsize=6.5, color='#111', fontweight='bold', zorder=6)

    xs = [n.x for n in neurons]; ys = [n.y for n in neurons]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    if title:
        ax.set_title(title, fontsize=9)

