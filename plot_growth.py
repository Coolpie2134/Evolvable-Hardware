#!/usr/bin/env python3
"""
plot_growth.py  —  Visualise how the SNN circuit grows from 2 seeds across 12 iterations.

Usage:
    python plot_growth.py            # uses results/best_genome.pkl
    python plot_growth.py --random   # uses a fresh random genome (seed 42)

Output: results/growth_plot.png
"""
import sys, os, pickle, argparse, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from snn_evo import random_genome, grow_snn
from snn_evo.genome import MAX_ITER, GRID_SIZE
from snn_evo.growth import SEED_A, SEED_B, SEED_STATE, _lookup
from snn_evo.snn import interpret_grid
from snn_evo.fitness import N_OUTPUTS

# ── growth with snapshots ────────────────────────────────────────────────────

def grow_with_snapshots(genome):
    """Run growth and capture grid state after each iteration."""
    grid = {SEED_A: SEED_STATE, SEED_B: SEED_STATE}
    snapshots = [{SEED_A: SEED_STATE, SEED_B: SEED_STATE}]   # iteration 0 = seeds only

    for iteration in range(MAX_ITER):
        frontier = {}
        for (x, y) in list(grid):
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = x+dx, y+dy
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE
                        and (nx,ny) not in grid and (nx,ny) not in frontier):
                    frontier[(nx, ny)] = 0
        working = {**grid, **frontier}
        next_grid = {}
        for (x, y), state in working.items():
            sn = working.get((x, y+1), 0); ss = working.get((x, y-1), 0)
            se = working.get((x+1, y),  0); sw = working.get((x-1, y), 0)
            ns = _lookup(genome, sn, ss, se, sw, state, iteration)
            if ns:
                next_grid[(x, y)] = ns
        next_grid[SEED_A] = SEED_STATE
        next_grid[SEED_B] = SEED_STATE
        grid = next_grid
        snapshots.append(dict(grid))

    return snapshots

# ── colour mapping ───────────────────────────────────────────────────────────

def grid_to_rgba(grid, output_pos):
    """
    Convert a grid dict to an RGBA image array (GRID_SIZE x GRID_SIZE x 4).

    Colour key:
        white           — empty cell
        red             — input seed  (A or B)
        blue gradients  — excitatory neuron (state→alpha)
        orange gradients— inhibitory neuron (state→alpha)
        lime green      — output neuron (sum / carry)
    """
    img = np.ones((GRID_SIZE, GRID_SIZE, 4), dtype=float)   # white bg

    for (x, y), state in grid.items():
        # image row = GRID_SIZE-1-y (y=0 at bottom, plot rows go top-down)
        row, col = GRID_SIZE - 1 - y, x
        if (x, y) in (SEED_A, SEED_B):
            img[row, col] = [0.9, 0.1, 0.1, 1.0]            # red — seed
        elif (x, y) in output_pos:
            img[row, col] = [0.1, 0.8, 0.2, 1.0]            # green — output
        else:
            excit = not bool((state >> 3) & 0x1)
            alpha = 0.35 + 0.55 * (state & 0x7) / 7.0       # brightness ~ state
            if excit:
                img[row, col] = [0.15, 0.35, 0.85, alpha]   # blue — excitatory
            else:
                img[row, col] = [0.90, 0.50, 0.10, alpha]   # orange — inhibitory

    return img

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random', action='store_true', help='use a random genome instead of saved')
    args = ap.parse_args()

    RESULTS_DIR = os.path.join(ROOT, 'results')
    CKPT        = os.path.join(RESULTS_DIR, 'best_genome.pkl')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.random:
        random.seed(42)
        genome  = random_genome(2)
        fitness = None
        print('Using random genome (seed=42)')
    else:
        if not os.path.exists(CKPT):
            print('No saved genome found — run main.py first, or use --random')
            sys.exit(1)
        with open(CKPT, 'rb') as f:
            state = pickle.load(f)
        genome  = state['best_genome']
        fitness = state['best_fitness']
        print('Loaded genome  fitness=%.4f' % fitness)

    # figure out output positions from final grid
    final_grid = grow_snn(genome)
    ns, _      = interpret_grid(final_grid, n_outputs=N_OUTPUTS)
    output_pos = {(n.x, n.y) for n in ns if n.is_output}

    snapshots = grow_with_snapshots(genome)          # list of 13 dicts (0..12)
    n_panels  = len(snapshots)                       # = MAX_ITER + 1  = 13
    ncols, nrows = 4, 4                              # 4×4 = 16 slots, last 3 spare

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.4 + 0.6),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.15})

    title_str = 'SNN Circuit Growth — Edwards Indirect Encoding'
    if fitness is not None:
        title_str += '  (fitness=%.4f)' % fitness
    fig.suptitle(title_str, fontsize=11, fontweight='bold', y=0.98)

    # draw each snapshot
    for idx, snap in enumerate(snapshots):
        ax = axes[idx // ncols][idx % ncols]
        img = grid_to_rgba(snap, output_pos)
        ax.imshow(img, origin='upper', aspect='equal', interpolation='nearest',
                  extent=[-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5])

        # grid lines
        for i in range(GRID_SIZE + 1):
            ax.axhline(i - 0.5, color='#cccccc', lw=0.3)
            ax.axvline(i - 0.5, color='#cccccc', lw=0.3)

        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks([]); ax.set_yticks([])

        n_alive = len(snap)
        label   = 'Iter %d' % idx if idx > 0 else 'Seed'
        ax.set_title('%s  (%d cells)' % (label, n_alive), fontsize=7.5, pad=2)

        # mark seed positions explicitly (always present)
        for sx, sy in (SEED_A, SEED_B):
            ax.text(sx, GRID_SIZE - 1 - sy, 'A' if (sx,sy)==SEED_A else 'B',
                    ha='center', va='center', fontsize=5, color='white', fontweight='bold')

        # mark output positions (show only in last frame where they're labelled)
        for ox, oy in output_pos:
            if (ox, oy) in snap:
                role_n = next((n for n in ns if n.x==ox and n.y==oy), None)
                lbl    = role_n.out_role[0].upper() if role_n else 'O'
                ax.text(ox, GRID_SIZE - 1 - oy, lbl,
                        ha='center', va='center', fontsize=5, color='white', fontweight='bold')

    # hide spare axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # legend
    legend_ax = axes[-1][-1]
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    patches = [
        mpatches.Patch(color=(0.9,0.1,0.1), label='Input seed (A / B)'),
        mpatches.Patch(color=(0.1,0.8,0.2), label='Output neuron'),
        mpatches.Patch(color=(0.15,0.35,0.85), label='Excitatory'),
        mpatches.Patch(color=(0.90,0.50,0.10), label='Inhibitory'),
        mpatches.Patch(color=(1.0,1.0,1.0), label='Empty', edgecolor='#aaa'),
    ]
    legend_ax.legend(handles=patches, loc='center', fontsize=7.5,
                     frameon=False, title='Cell type', title_fontsize=8)

    out_path = os.path.join(RESULTS_DIR, 'growth_plot.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved →', out_path)

if __name__ == '__main__':
    main()
