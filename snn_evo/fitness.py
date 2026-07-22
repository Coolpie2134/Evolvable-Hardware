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


def score(neurons, synapses, target: Target) -> float:
    """
    Generic fitness in [0, 1]: fraction of (case x output) checks the grown
    circuit gets right, for any registered Target.
    """
    # Map input seed positions and output roles to neuron ids.
    in_ids = []
    for pos in target.inputs:
        nrn = next((n for n in neurons if (n.x, n.y) == pos and n.is_input), None)
        if nrn is None:
            return 0.0
        in_ids.append(nrn.id)

    out_ids = []
    for term in target.outputs:
        nrn = next((n for n in neurons if n.is_output and n.out_role == term.role), None)
        if nrn is None:
            return 0.0
        out_ids.append(nrn.id)

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
            for bit, iid in zip(in_bits, in_ids):
                base = target.high if bit else 0.0
                currents[iid] = (target.high - base) if comp else base
            sims[comp] = simulate(neurons, synapses, currents)
        row = []
        for i, term in enumerate(target.outputs):
            sp = sims[term.complement_inputs]
            n  = len(sp.get(out_ids[i], []))
            fired = n >= MIN_SPIKES
            row.append(float(not fired) if term.invert_spike else float(fired))
        observations.append(row)

    return score_contract(observations, target)[0]


def evaluate(neurons, synapses):
    """Half-adder fitness (back-compat wrapper around the generic scorer)."""
    return score(neurons, synapses, get_target(DEFAULT_TARGET))
