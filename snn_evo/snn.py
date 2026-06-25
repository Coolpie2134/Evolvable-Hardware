from __future__ import annotations
import dataclasses
from .genome import GRID_SIZE

@dataclasses.dataclass
class Neuron:
    id:        int
    x:         int
    y:         int
    state:     int
    vth:       float
    tau:       float
    excit:     bool
    is_input:  bool  = False
    is_output: bool  = False
    out_role:  str   = ""   # "sum", "carry", ""

@dataclasses.dataclass
class Synapse:
    pre:    int
    post:   int
    weight: float

SEED_A     = (0, 3)
SEED_B     = (0, 5)
SYN_WEIGHT = 2.0

def interpret_grid(grid, n_outputs=1):
    neurons    = []
    pos_to_id  = {}
    vth_levels = [0.3, 0.5, 0.7, 0.9]
    tau_levels = [8.0, 15.0]

    for (x, y), state in sorted(grid.items()):
        if state == 0:
            continue
        nid   = len(neurons)
        vth   = vth_levels[state & 0x3]
        tau   = tau_levels[(state >> 2) & 0x1]
        excit = not bool((state >> 3) & 0x1)
        n = Neuron(id=nid, x=x, y=y, state=state, vth=vth, tau=tau,
                   excit=excit, is_input=((x, y) in (SEED_A, SEED_B)))
        neurons.append(n)
        pos_to_id[(x, y)] = nid

    mid_y      = GRID_SIZE // 2
    non_input  = [n for n in neurons if not n.is_input]
    x_cols     = sorted(set(n.x for n in non_input), reverse=True)

    if n_outputs == 1:
        if x_cols:
            best = min([n for n in non_input if n.x == x_cols[0]],
                       key=lambda n: abs(n.y - mid_y))
            best.is_output = True
            best.out_role  = "sum"
    else:
        if len(x_cols) >= 1:
            best = min([n for n in non_input if n.x == x_cols[0]],
                       key=lambda n: abs(n.y - mid_y))
            best.is_output = True
            best.out_role  = "sum"
        if len(x_cols) >= 2:
            best = min([n for n in non_input if n.x == x_cols[1]],
                       key=lambda n: abs(n.y - mid_y))
            best.is_output = True
            best.out_role  = "carry"

    synapses = []
    seen     = set()
    for (x, y), pre_id in pos_to_id.items():
        for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
            nx, ny = x+dx, y+dy
            if (nx, ny) not in pos_to_id:
                continue
            post_id = pos_to_id[(nx, ny)]
            pair    = (pre_id, post_id)
            if pair in seen:
                continue
            seen.add(pair)
            sign = 1.0 if neurons[pre_id].excit else -1.0
            synapses.append(Synapse(pre=pre_id, post=post_id,
                                    weight=sign * SYN_WEIGHT))
    return neurons, synapses

def circuit_summary(neurons, synapses):
    inp  = sum(1 for n in neurons if n.is_input)
    out  = sum(1 for n in neurons if n.is_output)
    hid  = len(neurons) - inp - out
    exc  = sum(1 for s in synapses if s.weight > 0)
    inh  = sum(1 for s in synapses if s.weight < 0)
    return ("%d neurons (%din/%dhid/%dout), %d syn (%dexc/%dinh)"
            % (len(neurons), inp, hid, out, len(synapses), exc, inh))
