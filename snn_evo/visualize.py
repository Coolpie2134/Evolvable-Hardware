from __future__ import annotations
from .genome import GRID_SIZE

SEED_A = (0, 3)
SEED_B = (0, 5)

def grid_ascii(grid, neurons):
    n_map = {(n.x, n.y): n for n in neurons}
    lines = []
    for y in range(GRID_SIZE - 1, -1, -1):
        row = []
        for x in range(GRID_SIZE):
            if (x, y) in n_map:
                n = n_map[(x, y)]
                if (x, y) == SEED_A:   row.append("A")
                elif (x, y) == SEED_B: row.append("B")
                elif n.is_output:      row.append("O")
                else:                  row.append(str(n.state % 10))
            else:
                row.append(".")
        lines.append(" ".join(row))
    return chr(10).join(lines)

def spike_raster(spikes, neurons, width=60, sim_time=20.0):
    lines = []
    for n in neurons:
        times = spikes.get(n.id, [])
        row   = ["."] * width
        for t in times:
            col = min(int(t / sim_time * width), width - 1)
            row[col] = "|"
        tag = "(IN)" if n.is_input else ("(OUT)" if n.is_output else "     ")
        lines.append("N%02d%s %s  %d spikes" % (n.id, tag, "".join(row), len(times)))
    return chr(10).join(lines)
