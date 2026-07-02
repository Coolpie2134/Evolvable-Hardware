# Evolvable Hardware — Logic Circuits via Grown Neural Substrates

A research project implementing **Edwards indirect encoding** to evolve grown circuits, with two independent backends: spiking neural networks (SNN) and hexagonal "nervous nets".

## What it does

A genetic algorithm evolves a genome that encodes growth rules. Starting from input seed cells, a cellular automaton grows over a number of iterations into a circuit. Two separate backends interpret and evaluate the grown grid:

- **SNN** (`snn_evo/`): a square grid of leaky integrate-and-fire (LIF) neurons computing combinational logic — single gates (AND/OR/XOR/NAND/NOR/XNOR), half/full adders, multi-bit adders, or any custom truth table you enter in the GUI.
- **Nervous net** (`nv_evo/`): a honeycomb array of nodes that fire when `(E1 AND E2) AND NOT I1` — coincidence detection with an inhibitory veto, exactly the 16 routing states of Edwards EH'02 Fig. 3 (every state is a buffer or an AND, optionally vetoed; there is no disjunction). Directions are read in each node's own orientation frame (the paper's context rotation — down-parity tiles are the 180°-rotated circuit), and inputs are **wired-OR injections** onto the input cell's net, like pulsing the physical perimeter wire — not level clamps. So **with no input the array is inert**, yet a pulse injected into a loop of buffers routed through an input cell circulates as delay-line memory "until stopped by application of an inhibitory input" (§3). Besides combinational targets it evolves **temporal circuits** — SR latches, toggle flip-flops, oscillators, delay lines, plus async-native ones where edge timing *is* the computation: a two-input coincidence detector, a one-shot/monostable (self-terminating burst via delayed self-inhibition), and a double-pulse pair detector (delay line + coincidence). Dynamics are the paper's **asynchronous edge-triggered pulses** (`nv_evo/pulse.py`): an event-driven simulation where a triggered node emits a fixed-width pulse after a fixed delay, coincidence requires both excitatory edges within a window, and inhibition vetoes at trigger time; scoring samples the wires once per tick. Every element (identical tiled nodes, 4-bit routing SRAM, genome as associative memory / CAM, delay / pulse-width / coincidence-window constants) maps directly onto the paper's hardware.

The two backends share no code: each has its own genome, growth automaton, targets and GA.

## How it works

- **Indirect encoding**: the genome is an associative memory of rules mapping a cell's neighbourhood state → output state, looked up by minimum Hamming distance (the nervous-net gene is exactly the paper's `context → new state`, with no per-gene time field)
- **Context rotation** (nervous net): each node reads its L/R/D neighbours in its own orientation frame, so one gene can grow symmetric structure ("right/left/down rotated to match topology", EH'02 Fig. 4)
- **Circuit ontogeny**: growth iterations expand the input seeds into a full circuit
- **LIF simulation** (SNN): neurons communicate via synaptic currents; output is read as spike / no-spike
- **Per-output encoding** (SNN): an output may use complement inputs and/or an inverted spike reading — e.g. the half-adder's carry (AND) is evaluated with flipped inputs + inverted output, making it trivially learnable
- **Pluggable targets**: a `Target` (`snn_evo/targets.py`) is the single source of truth for a problem's input seeds, output terminals and truth table; the growth, interpretation, fitness and GUI all read their I/O layout from it
- **Temporal targets** (`nv_evo/targets.py`): scored on output *traces* over time, across several trials with shifted pulse timings — so only genuine memory passes, not a lucky delay chain
- **Fitness for temporal targets** (`nv_evo/temporal.py`): three ideas make memory learnable —
  - *Trace-matched output placement*: each output role is read at whichever live cell best reproduces the expected trace, so evolution only has to build the mechanism, not also route the answer to one prescribed cell.
  - *Windowed, level-balanced scoring*: the trace is split into constant-level windows (each behavioural phase weighs equally) and the two levels are balanced, so a permanently-silent output caps at 0.5 instead of riding the 0-heavy traces to the top.
  - *Phase-tolerant holds*: because a bit is stored as a pulse *circulating* in a loop, it reads as a ripple (e.g. `1010`) at any one cell — the honeycomb has no triangles, so the phases can't be OR'd back into a steady level. A "store 1" window is therefore scored by activity coverage (is the cell actively ringing, no long gap?), while "store 0" still demands true silence.
- **Loop-aware GA** (`nv_evo/ga.py`): shaping that rewards feedback cycles the inputs can write and the outputs can read, gene-duplication mutations that build repeated loop motifs, random immigrants against premature convergence, and a fitness cache

## Files

| File | Description |
|------|-------------|
| `app.py` | Single-window GUI: pick a model + target, evolve, inspect growth / activity / genome, drive circuits interactively |
| `snn_evo/` | SNN backend (genome, growth, LIF sim, fitness, targets, GA) |
| `snn_evo/targets.py` | Combinational target registry + builders (gates, adders, custom truth tables) |
| `nv_evo/` | Nervous-net backend (hex genome, honeycomb growth, pulse dynamics, targets, GA) |
| `nv_evo/targets.py` | Temporal target registry (SR latch, toggle flip-flop, oscillator, echo) |
| `nv_evo/ga.py` | Loop/memory-tuned GA (shaping, duplication, immigrants, caching) |

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
