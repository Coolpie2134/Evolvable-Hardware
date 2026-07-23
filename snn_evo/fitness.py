from __future__ import annotations
from .lif_sim import simulate
from .targets import (Target, get_target, CURRENT_HIGH, MIN_SPIKES,
                      DEFAULT_TARGET)
from nv_evo.scoring import score_contract

SEED_A    = (0, 3)
SEED_B    = (0, 5)
N_OUTPUTS = 2

# Back-compat: the half-adder's 4 cases in the old ((ia, ib), (sum, carry)) form.
LOGIC_CASES = [
    ((0.0,          0.0         ), (0, 0)),
    ((0.0,          CURRENT_HIGH), (1, 0)),
    ((CURRENT_HIGH, 0.0         ), (1, 0)),
    ((CURRENT_HIGH, CURRENT_HIGH), (0, 1)),
]


def score(neurons, synapses, target: Target, in_pos=None) -> float:
    """
    Generic fitness in [0, 1]: fraction of (case x output) checks the grown
    circuit gets right, for any registered Target.

    ``in_pos`` overrides the driven input positions (the evolvable io_placement
    binding — must match the ``input_pos`` the grid was interpreted with);
    None keeps the target's seed pads. Each entry may itself be a group of
    attachment cells: the logical input drives every member neuron, and a
    neuron shared by several inputs receives the strongest drive (wired-OR).
    """
    from nv_evo.io_placement import input_groups
    # Map input positions and output roles to neuron ids (one id GROUP per
    # logical input).
    by_pos = {(n.x, n.y): n for n in neurons if n.is_input}
    in_ids = []
    for cells in input_groups(target.inputs if in_pos is None else in_pos):
        ids = [by_pos[c].id for c in cells if c in by_pos]
        if len(ids) != len(cells):
            return 0.0
        in_ids.append(ids)

    out_ids = []
    for term in target.outputs:
        ids = [n.id for n in neurons
               if n.is_output and n.out_role == term.role]
        if not ids:
            return 0.0
        out_ids.append(ids)

    n_checks = len(target.cases) * len(target.outputs)
    if n_checks == 0:
        return 0.0

    observations = []
    encodings = {t.complement_inputs for t in target.outputs}
    for in_bits, out_bits in target.cases:
        # One simulation per distinct input encoding (normal / complement).
        sims = {}
        for comp in encodings:
            currents = {}
            for bit, ids in zip(in_bits, in_ids):
                base = target.high if bit else 0.0
                level = (target.high - base) if comp else base
                for iid in ids:
                    # a neuron serving several logical inputs takes the
                    # strongest drive — the wired-OR of the held levels
                    currents[iid] = max(currents.get(iid, 0.0), level)
            sims[comp] = simulate(neurons, synapses, currents)
        row = []
        for i, term in enumerate(target.outputs):
            sp = sims[term.complement_inputs]
            fired = any(len(sp.get(output_id, [])) >= MIN_SPIKES
                        for output_id in out_ids[i])
            row.append(float(not fired) if term.invert_spike else float(fired))
        observations.append(row)

    return score_contract(observations, target)[0]


def evaluate(neurons, synapses):
    """Half-adder fitness (back-compat wrapper around the generic scorer)."""
    return score(neurons, synapses, get_target(DEFAULT_TARGET))
