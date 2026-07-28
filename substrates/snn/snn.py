from __future__ import annotations
import dataclasses
from typing import Tuple
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


@dataclasses.dataclass(frozen=True)
class Arch:
    """
    Tunable substrate parameters (the electrical model), kept separate from the
    Target (the problem). Threaded through interpretation so it survives the
    process pool. NB: whether a coincidence/AND detector is buildable hinges on
    the one-EPSC voltage excursion ``syn_weight * (1 - exp(-EPSC_DUR / tau))``
    relative to ``vth``.  A usable coincidence cell has one pulse below its
    threshold and two overlapping pulses above it; if every available cell
    crosses on a single EPSC, the network is effectively OR-only.
    """
    syn_weight: float = SYN_WEIGHT
    vth_levels: Tuple[float, float, float, float] = (0.3, 0.5, 0.7, 0.9)
    tau_levels: Tuple[float, float] = (8.0, 15.0)
    # Static truth-table runs preserve the original feed-forward lattice.
    # Temporal SNN runs opt into reciprocal physical adjacencies so evolved
    # bodies can contain feedback loops, oscillators and retained state.
    recurrent: bool = False


DEFAULT_ARCH = Arch()


def interpret_grid(grid, n_outputs=1, target=None, arch=None,
                   input_pos=None, output_pos=None):
    """
    Build (neurons, synapses) from a grown grid.

    If `target` is given, input/output layout follows the target (any number of
    inputs and outputs). Otherwise the legacy 2-seed, sum/carry heuristic is used
    (controlled by n_outputs) so existing callers keep working unchanged.

    `input_pos` / `output_pos` override the port binding — the evolvable
    io_placement path (substrates/nervous/io_placement.py) passes the genome's tag-chosen
    cells here: `input_pos` is the ordered list of driven cells, `output_pos` is
    {role: (x, y)}. None keeps the target/legacy layout unchanged.
    """
    if arch is None:
        arch = DEFAULT_ARCH
    if input_pos is None:
        input_pos = list(target.inputs) if target is not None else [SEED_A, SEED_B]
    else:
        input_pos = list(input_pos)
    grid_size = target.grid_size    if target is not None else GRID_SIZE

    neurons    = []
    pos_to_id  = {}
    vth_levels = arch.vth_levels
    tau_levels = arch.tau_levels

    for (x, y), state in sorted(grid.items()):
        if state == 0:
            continue
        nid   = len(neurons)
        vth   = vth_levels[state & 0x3]
        tau   = tau_levels[(state >> 2) & 0x1]
        excit = not bool((state >> 3) & 0x1)
        n = Neuron(id=nid, x=x, y=y, state=state, vth=vth, tau=tau,
                   excit=excit, is_input=((x, y) in input_pos))
        neurons.append(n)
        pos_to_id[(x, y)] = nid

    non_input = [n for n in neurons if not n.is_input]

    if output_pos is not None:
        # Evolvable binding: every selected site joins the role's wired-OR bus.
        from substrates.nervous.io_placement import output_groups
        by_pos = {(n.x, n.y): n for n in neurons}
        for role, cells in output_groups(output_pos).items():
            for pos in cells:
                nrn = by_pos.get(tuple(pos))
                if nrn is not None:
                    nrn.is_output = True
                    nrn.out_role  = role
    elif target is not None and target.output_strategy == "terminals":
        # Assign each output role to the nearest free grown neuron to its terminal.
        for term in target.outputs:
            tx, ty = term.pos
            cands  = [n for n in non_input if not n.is_output]
            if not cands:
                break
            best = min(cands, key=lambda n: abs(n.x - tx) + abs(n.y - ty))
            best.is_output = True
            best.out_role  = term.role
    else:
        # Legacy heuristic: outputs are nearest-to-mid in the rightmost columns.
        roles  = ([t.role for t in target.outputs] if target is not None
                  else (["sum"] if n_outputs == 1 else ["sum", "carry"]))
        mid_y  = grid_size // 2
        x_cols = sorted(set(n.x for n in non_input), reverse=True)
        for i, role in enumerate(roles):
            if i >= len(x_cols):
                break
            best = min([n for n in non_input if n.x == x_cols[i]],
                       key=lambda n: abs(n.y - mid_y))
            best.is_output = True
            best.out_role  = role

    synapses = []
    seen     = set()
    # The four positive lattice offsets visit every physical adjacency once.
    # The original architecture orients that edge from the lower lexicographic
    # coordinate to the higher one and is therefore a DAG. Temporal SNN runs
    # opt into reciprocal adjacencies through Arch.recurrent; this preserves
    # every combinational checkpoint while making feedback an explicit,
    # serialized hardware choice rather than an accidental target-side trick.
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
                                    weight=sign * arch.syn_weight))
            if arch.recurrent:
                reverse = (post_id, pre_id)
                if reverse not in seen:
                    seen.add(reverse)
                    reverse_sign = 1.0 if neurons[post_id].excit else -1.0
                    synapses.append(Synapse(
                        pre=post_id, post=pre_id,
                        weight=reverse_sign * arch.syn_weight))
    return neurons, synapses

def circuit_summary(neurons, synapses):
    inp  = sum(1 for n in neurons if n.is_input)
    out  = sum(1 for n in neurons if n.is_output)
    hid  = len(neurons) - inp - out
    exc  = sum(1 for s in synapses if s.weight > 0)
    inh  = sum(1 for s in synapses if s.weight < 0)
    return ("%d neurons (%din/%dhid/%dout), %d syn (%dexc/%dinh)"
            % (len(neurons), inp, hid, out, len(synapses), exc, inh))
