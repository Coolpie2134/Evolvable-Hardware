# Evolvable Hardware — Logic Circuits via Spiking Neural Networks

A research project implementing **Edwards indirect encoding** to evolve spiking neural network (SNN) circuits that compute logic functions.

## What it does

A genetic algorithm evolves a genome that encodes growth rules. Starting from input seed cells, a cellular automaton grows over a number of iterations into a network of leaky integrate-and-fire (LIF) neurons. The **target function** is pluggable: single logic gates (AND/OR/XOR/NAND/NOR/XNOR), a half-adder (Sum = A XOR B, Carry = A AND B), a full adder, multi-bit adders, or any custom truth table you enter in the GUI.

## How it works

- **Indirect encoding**: the genome is an associative memory of rules mapping a cell's neighbourhood state → output state, looked up by minimum Hamming distance
- **Circuit ontogeny**: growth iterations expand the input seeds into a full circuit
- **LIF simulation**: neurons communicate via synaptic currents; output is read as spike / no-spike
- **Per-output encoding**: an output may use complement inputs and/or an inverted spike reading — e.g. the half-adder's carry (AND) is evaluated with flipped inputs + inverted output, making it trivially learnable
- **Pluggable targets**: a `Target` (`snn_evo/targets.py`) is the single source of truth for a problem's input seeds, output terminals and truth table; the growth, interpretation, fitness and GUI all read their I/O layout from it

## Files

| File | Description |
|------|-------------|
| `app.py` | Single-window GUI: evolve any target, view growth snapshots and voltage traces |
| `snn_evo/` | Core library (genome, growth, LIF sim, fitness, targets, GA) |
| `snn_evo/targets.py` | Target-function registry + builders (gates, adders, custom truth tables) |

## Usage

```bash
pip install -r requirements.txt

python app.py        # launch the GUI
```

In the GUI: pick a **Target** from the dropdown (or click **Custom…** to enter your
own truth table), set Pop / Gens / Tries, and click **Run**. Use **Load Saved** to
reload `results/best_genome.pkl` and **Save PNGs** to export the growth and voltage
figures.

Tick **Graded** for harder targets (adders, large custom tables): instead of a binary
pass/fail per output it gives smooth partial credit, which keeps a usable fitness
gradient where binary scoring would otherwise flatline. A perfect circuit still scores
1.0, so it's safe to leave on.

> Note: large targets (full/multi-bit adders, big custom tables) have many truth-table
> rows and evolve slowly — the framework supports them, but small targets are best for
> interactive runs.

## Requirements

- Python 3.8+
- numpy
- matplotlib
