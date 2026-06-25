from __future__ import annotations
from .lif_sim import simulate

CURRENT_HIGH = 4.0   # nA  (logical "1" input current)
MIN_SPIKES   = 1     # spikes needed to count as "fired"
SEED_A       = (0, 3)
SEED_B       = (0, 5)
N_OUTPUTS    = 2

# Half adder:  Sum = A XOR B,   Carry = A AND B
# Each entry: ((ia_nA, ib_nA), (expected_sum, expected_carry))
LOGIC_CASES = [
    ((0.0,          0.0         ), (0, 0)),
    ((0.0,          CURRENT_HIGH), (1, 0)),
    ((CURRENT_HIGH, 0.0         ), (1, 0)),
    ((CURRENT_HIGH, CURRENT_HIGH), (0, 1)),
]

def evaluate(neurons, synapses):
    """Return fitness in [0, 1].  1.0 = perfect half adder."""
    sum_n   = next((n for n in neurons if n.is_output and n.out_role == "sum"),   None)
    carry_n = next((n for n in neurons if n.is_output and n.out_role == "carry"), None)
    if sum_n is None or carry_n is None:
        return 0.0
    a_n = next((n for n in neurons if (n.x, n.y) == SEED_A), None)
    b_n = next((n for n in neurons if (n.x, n.y) == SEED_B), None)
    if a_n is None or b_n is None:
        return 0.0

    a_id, b_id   = a_n.id, b_n.id
    sum_correct  = 0
    carry_correct = 0

    for (ia, ib), (exp_sum, exp_carry) in LOGIC_CASES:
        # Sum:   normal encoding,     binary XOR check
        sp_sum  = simulate(neurons, synapses, {a_id: ia, b_id: ib})
        act_sum = 1 if len(sp_sum.get(sum_n.id, [])) >= MIN_SPIKES else 0
        if act_sum == exp_sum:
            sum_correct += 1

        # Carry: complement encoding + complement output interpretation.
        # Flipping inputs turns AND into NAND-of-complements = OR-complement,
        # which any connected circuit trivially computes.
        ia_c = CURRENT_HIGH - ia
        ib_c = CURRENT_HIGH - ib
        sp_c    = simulate(neurons, synapses, {a_id: ia_c, b_id: ib_c})
        n_carry = len(sp_c.get(carry_n.id, []))
        act_carry = 0 if n_carry >= MIN_SPIKES else 1
        if act_carry == exp_carry:
            carry_correct += 1

    return (sum_correct + carry_correct) / (len(LOGIC_CASES) * 2)
