#!/usr/bin/env python3
"""
plot_ha.py  —  Membrane-voltage plot for the evolved half-adder.

Reads results/best_genome.pkl and saves half_adder_plot.png.
Run:  python plot_ha.py
"""
import sys, os, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from snn_evo import (grow_snn, interpret_grid, simulate,
                     LOGIC_CASES, CURRENT_HIGH, N_OUTPUTS, SEED_A, SEED_B)
from snn_evo.lif_sim import DT, SIM_TIME, N_STEPS, V_RESET, REFRAC_STEPS, EPSC_STEPS

CKPT    = os.path.join(ROOT, 'results', 'best_genome.pkl')
OUT_PNG = os.path.join(ROOT, 'results', 'half_adder_plot.png')


def simulate_vmem(neurons, synapses, input_currents, track_ids):
    """Like simulate() but also records per-step membrane voltage."""
    n   = len(neurons)
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
    vmem       = {i: np.zeros(N_STEPS, dtype=np.float32) for i in track_ids}
    for step in range(N_STEPS):
        for tid in track_ids:
            vmem[tid][step] = V[tid]
        I_syn  = (W * (epsc > 0)).sum(axis=1)
        active = refractory == 0
        dV     = (DT / TAU) * (-(V - V_REST) + R_M * (I_syn + I_ext))
        V      = np.where(active, V + dV, V_RESET)
        refractory = np.maximum(refractory - 1, 0)
        fired  = active & (V >= VTH)
        for i in np.where(fired)[0]:
            spikes[i].append(step * DT)
            if i in vmem: vmem[i][step] = float(VTH[i]) + 0.1
            V[i]          = V_RESET
            refractory[i] = REFRAC_STEPS
            epsc[has_post[:, i], i] = EPSC_STEPS
        epsc = np.maximum(epsc - 1, 0)
    return spikes, vmem

V_REST = 0.0
R_M    = 1.0

def main():
    if not os.path.exists(CKPT):
        print('No saved genome at', CKPT)
        print('Run  python main.py  first to evolve one.')
        return

    with open(CKPT, 'rb') as f:
        state = pickle.load(f)
    genome = state['best_genome']

    grid = grow_snn(genome)
    ns, ss = interpret_grid(grid, n_outputs=N_OUTPUTS)
    sum_n   = next(n for n in ns if n.is_output and n.out_role == 'sum')
    carry_n = next(n for n in ns if n.is_output and n.out_role == 'carry')
    a_n = next(n for n in ns if (n.x, n.y) == SEED_A)
    b_n = next(n for n in ns if (n.x, n.y) == SEED_B)
    track = [sum_n.id, carry_n.id]

    t = np.arange(N_STEPS) * DT
    case_labels = [
        'A=0, B=0\nSum=0, Carry=0',
        'A=0, B=1\nSum=1, Carry=0',
        'A=1, B=0\nSum=1, Carry=0',
        'A=1, B=1\nSum=0, Carry=1',
    ]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        'Evolved Half-Adder — Membrane Voltages\n'
        'Sum = A XOR B  |  Carry = A AND B  (complement encoded)',
        fontsize=12, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.91, bottom=0.07, left=0.09, right=0.97)

    for row, ((ia, ib), (es, ec)) in enumerate(LOGIC_CASES):
        sp_s, vm_s = simulate_vmem(ns, ss, {a_n.id: ia, b_n.id: ib}, track)
        sp_c, vm_c = simulate_vmem(
            ns, ss, {a_n.id: CURRENT_HIGH-ia, b_n.id: CURRENT_HIGH-ib}, track)

        n_sum   = len(sp_s.get(sum_n.id, []))
        n_carry = len(sp_c.get(carry_n.id, []))
        act_s   = 1 if n_sum   >= 1 else 0
        act_c   = 0 if n_carry >= 1 else 1

        for col, (neuron, vmem, n_spikes, act, exp, color, title_base) in enumerate([
            (sum_n,   vm_s, n_sum,   act_s, es, '#2196F3',
             'SUM neuron  (excitatory, pos=%s)' % str((sum_n.x, sum_n.y))),
            (carry_n, vm_c, n_carry, act_c, ec, '#E91E63',
             'CARRY neuron  (inhibitory, pos=%s, complement inputs)' % str((carry_n.x, carry_n.y))),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t, vmem[neuron.id], color=color, lw=1.2)
            ax.axhline(neuron.vth, color='gray', lw=0.8, ls='--', alpha=0.7)
            ax.set_xlim(0, SIM_TIME)
            ax.set_ylim(-0.15, neuron.vth + 0.35)
            ax.set_ylabel('V (V)', fontsize=8)
            if row == 0:
                ax.set_title(title_base, fontsize=9)
            ax.text(0.02, 0.88, case_labels[row],
                    transform=ax.transAxes, fontsize=7.5, va='top')
            result = 'act=%d  exp=%d  %s' % (act, exp, 'OK' if act==exp else 'FAIL')
            ax.text(0.98, 0.88, result, transform=ax.transAxes,
                    fontsize=8, va='top', ha='right',
                    color='green' if act==exp else 'red', fontweight='bold')
            if row == 3:
                ax.set_xlabel('Time (ms)', fontsize=8)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print('Saved:', OUT_PNG)

if __name__ == '__main__':
    main()
