# Evolvable Hardware — Half-Adder via Spiking Neural Networks

A research project implementing **Edwards indirect encoding** to evolve spiking neural network (SNN) circuits that compute logic functions.

## What it does

A genetic algorithm evolves a genome that encodes growth rules. Starting from two seed cells, a 9×9 cellular automaton grows over 12 iterations into a network of leaky integrate-and-fire (LIF) neurons. The target function is a **half-adder** (Sum = A XOR B, Carry = A AND B).

## How it works

- **Indirect encoding**: the genome is an associative memory of rules mapping a cell's neighbourhood state → output state, looked up by minimum Hamming distance
- **Circuit ontogeny**: 12 growth iterations expand two input seeds into a full circuit
- **LIF simulation**: neurons communicate via synaptic currents; output is read as spike / no-spike
- **Complement encoding**: carry (AND) is evaluated with flipped inputs + inverted output, making it trivially learnable

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — run the GA or load a saved winner |
| `plot_growth.py` | Visualise circuit growth across all 12 iterations |
| `plot_ha.py` | Plot membrane voltage traces for all 4 input cases |
| `snn_evo/` | Core library (genome, growth, LIF sim, fitness, GA) |

## Usage

```bash
pip install -r requirements.txt

python main.py                        # evolve from scratch
python main.py --load                 # load saved winner and show truth table
python main.py --gens 50 --pop 80 --tries 20

python plot_growth.py                 # visualise circuit growth
python plot_ha.py                     # plot membrane voltages
```

## Requirements

- Python 3.8+
- numpy
- matplotlib
