#!/usr/bin/env python3
"""
main.py  —  Evolvable Hardware: half-adder via Edwards indirect encoding.

Usage
-----
  python main.py               # evolve from scratch, save winner
  python main.py --load        # load saved winner and display results
  python main.py --gens 50 --pop 60 --tries 10

Results are saved to  results/best_genome.pkl
"""
import sys, os, random, argparse, pickle, copy

# make snn_evo importable from anywhere
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from snn_evo import (grow_snn, interpret_grid, circuit_summary,
                     simulate, evaluate, LOGIC_CASES, CURRENT_HIGH,
                     N_OUTPUTS, SEED_A, SEED_B, random_genome)
from snn_evo.visualize import grid_ascii
from snn_evo.ga import evolve, _eval_batch, mutate, crossover, tournament, ELITE_FRAC

RESULTS_DIR = os.path.join(ROOT, 'results')
CKPT        = os.path.join(RESULTS_DIR, 'best_genome.pkl')

os.makedirs(RESULTS_DIR, exist_ok=True)


def print_truth_table(genome):
    grid = grow_snn(genome)
    ns, ss = interpret_grid(grid, n_outputs=N_OUTPUTS)
    sum_n   = next((n for n in ns if n.is_output and n.out_role == 'sum'),   None)
    carry_n = next((n for n in ns if n.is_output and n.out_role == 'carry'), None)
    a_n = next((n for n in ns if (n.x, n.y) == SEED_A), None)
    b_n = next((n for n in ns if (n.x, n.y) == SEED_B), None)
    if not all([sum_n, carry_n, a_n, b_n]):
        print('(circuit incomplete)'); return

    print('Circuit:', circuit_summary(ns, ss))
    print('Sum   neuron: pos=(%d,%d)  vth=%.1f  excit=%s' %
          (sum_n.x, sum_n.y, sum_n.vth, sum_n.excit))
    print('Carry neuron: pos=(%d,%d)  vth=%.1f  excit=%s  <-- inhibitory gating' %
          (carry_n.x, carry_n.y, carry_n.vth, carry_n.excit))
    print()
    print('Half-adder truth table  (Sum = A XOR B,  Carry = A AND B):')
    print('  A  B |  Sum  Carry |  #sum  #carry | result')
    print('  ' + '-' * 46)
    all_pass = True
    for name, ((ia, ib), (es, ec)) in zip(['0,0','0,1','1,0','1,1'], LOGIC_CASES):
        sp_s  = simulate(ns, ss, {a_n.id: ia, b_n.id: ib})
        ns_   = len(sp_s.get(sum_n.id, []))
        sp_c  = simulate(ns, ss, {a_n.id: CURRENT_HIGH-ia, b_n.id: CURRENT_HIGH-ib})
        nc    = len(sp_c.get(carry_n.id, []))
        act_s = 1 if ns_ >= 1 else 0
        act_c = 0 if nc  >= 1 else 1
        ok    = act_s == es and act_c == ec
        if not ok: all_pass = False
        print('  %s |  %3d  %5d |  %5d  %6d | %s' %
              (name, act_s, act_c, ns_, nc, 'PASS' if ok else 'FAIL'))
    print()
    print('  =>', 'ALL PASS  (fitness = 1.0)' if all_pass else 'Some failures')
    print()
    print('Grid  (A = input A,  B = input B,  O = outputs,  digit = neuron state):')
    print(grid_ascii(grid, ns))


def run_ga(gens, pop, tries):
    best_fit = 0.0
    best_g   = None
    for seed in range(tries):
        random.seed(seed + 1)
        population = [random_genome(2) for _ in range(pop)]
        fitnesses  = _eval_batch(population)
        bi         = max(range(pop), key=lambda i: fitnesses[i])
        g, f       = copy.deepcopy(population[bi]), fitnesses[bi]
        for gen in range(gens):
            n_e     = max(1, int(pop * ELITE_FRAC))
            ei      = sorted(range(pop), key=lambda i: fitnesses[i], reverse=True)
            new_pop = [copy.deepcopy(population[i]) for i in ei[:n_e]]
            while len(new_pop) < pop:
                ca, cb = crossover(tournament(population, fitnesses),
                                   tournament(population, fitnesses))
                new_pop.append(mutate(ca))
                if len(new_pop) < pop: new_pop.append(mutate(cb))
            population = new_pop[:pop]
            fitnesses  = _eval_batch(population)
            gi         = max(range(pop), key=lambda i: fitnesses[i])
            if fitnesses[gi] > f: f = fitnesses[gi]; g = copy.deepcopy(population[gi])
            if gen % 10 == 0:
                print('  seed=%d  gen=%3d  best=%.4f' % (seed+1, gen, f))
            if f >= 1.0:
                print('  SOLVED  seed=%d  gen=%d' % (seed+1, gen)); break
        print('Seed %d finished: best=%.4f' % (seed+1, f))
        if f > best_fit: best_fit, best_g = f, copy.deepcopy(g)
        if best_fit >= 1.0: break
    return best_g, best_fit


def main():
    ap = argparse.ArgumentParser(description='Evolvable Hardware — Half-Adder SNN')
    ap.add_argument('--gens',  type=int, default=30,  help='generations per seed')
    ap.add_argument('--pop',   type=int, default=50,  help='population size')
    ap.add_argument('--tries', type=int, default=20,  help='independent random restarts')
    ap.add_argument('--load',  action='store_true',   help='load saved winner')
    args = ap.parse_args()

    print('=' * 60)
    print('  Evolvable Hardware — Edwards Indirect Encoding')
    print('  Target: Half Adder (Sum=A XOR B, Carry=A AND B)')
    print('=' * 60)
    print()

    if args.load:
        if not os.path.exists(CKPT):
            print('No saved genome found at', CKPT)
            print('Run without --load to evolve one first.')
            sys.exit(1)
        with open(CKPT, 'rb') as f:
            state = pickle.load(f)
        genome  = state['best_genome']
        fitness = state['best_fitness']
        print('Loaded saved genome  (fitness=%.4f)\n' % fitness)
    else:
        print('Evolving...  pop=%d  gens=%d  restarts=%d\n' %
              (args.pop, args.gens, args.tries))
        genome, fitness = run_ga(args.gens, args.pop, args.tries)
        with open(CKPT, 'wb') as f:
            pickle.dump({'best_genome': genome, 'best_fitness': fitness}, f)
        print('\nSaved to', CKPT)
        print('Final fitness: %.4f\n' % fitness)

    print_truth_table(genome)


if __name__ == '__main__':
    main()
