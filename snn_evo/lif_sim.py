from __future__ import annotations
import numpy as np

DT           = 0.1    # ms per step
SIM_TIME     = 20.0   # ms total
V_REST       = 0.0
V_RESET      = 0.0
R_M          = 1.0    # membrane resistance (GOhm)
T_REFRAC     = 2.0    # refractory period ms
EPSC_DUR     = 5.0    # EPSC duration ms
N_STEPS      = int(SIM_TIME / DT)
REFRAC_STEPS = int(T_REFRAC / DT)
EPSC_STEPS   = int(EPSC_DUR / DT)

def simulate(neurons, synapses, input_currents):
    """Return {neuron_id: [spike_times_ms]}."""
    n = len(neurons)
    if n == 0:
        return {}
    TAU = np.array([nu.tau for nu in neurons], dtype=np.float32)
    VTH = np.array([nu.vth for nu in neurons], dtype=np.float32)
    W   = np.zeros((n, n), dtype=np.float32)
    for s in synapses:
        if 0 <= s.pre < n and 0 <= s.post < n:
            W[s.post, s.pre] += s.weight
    I_ext = np.zeros(n, dtype=np.float32)
    for nid, curr in input_currents.items():
        if 0 <= nid < n:
            I_ext[nid] = float(curr)
    has_post   = (W != 0)
    V          = np.zeros(n, dtype=np.float32)
    refractory = np.zeros(n, dtype=np.int32)
    epsc       = np.zeros((n, n), dtype=np.int32)
    spikes     = {i: [] for i in range(n)}
    for step in range(N_STEPS):
        I_syn  = (W * (epsc > 0)).sum(axis=1)
        active = refractory == 0
        dV     = (DT / TAU) * (-(V - V_REST) + R_M * (I_syn + I_ext))
        V      = np.where(active, V + dV, V_RESET)
        refractory = np.maximum(refractory - 1, 0)
        fired  = active & (V >= VTH)
        for i in np.where(fired)[0]:
            spikes[i].append(step * DT)
            V[i]          = V_RESET
            refractory[i] = REFRAC_STEPS
            epsc[has_post[:, i], i] = EPSC_STEPS
        epsc = np.maximum(epsc - 1, 0)
    return spikes
