# Evolvable Hardware — Logic Circuits via Grown Neural Substrates

A research project implementing **Edwards indirect encoding** to evolve grown circuits, with three backends: spiking neural networks (SNN), hexagonal "nervous nets", and a square **LUT array** (the paper's Architecture 2 / `sim6`).

## What it does

A genetic algorithm evolves a genome that encodes growth rules. Starting from input seed cells, a cellular automaton grows over a number of iterations into a circuit. Two separate backends interpret and evaluate the grown grid:

- **SNN** (`snn_evo/`): a square grid of leaky integrate-and-fire (LIF) neurons computing combinational logic — single gates (AND/OR/XOR/NAND/NOR/XNOR), half/full adders, multi-bit adders, or any custom truth table you enter in the GUI.
- **Nervous net** (`nv_evo/`): a honeycomb array of nodes that fire when `(E1 AND E2) AND NOT I1` — coincidence detection with an inhibitory veto, exactly the 16 routing states of Edwards EH'02 Fig. 3 (every state is a buffer or an AND, optionally vetoed; there is no disjunction). Directions are read in each node's own orientation frame (the paper's context rotation — down-parity tiles are the 180°-rotated circuit), and inputs are **wired-OR injections** onto the input cell's net, like pulsing the physical perimeter wire — not level clamps. So **with no input the array is inert**, yet a pulse injected into a loop of buffers routed through an input cell circulates as delay-line memory "until stopped by application of an inhibitory input" (§3). Besides combinational targets it evolves **temporal circuits** — SR latches, toggle flip-flops, oscillators, delay lines, plus async-native ones where edge timing *is* the computation: a two-input coincidence detector, a one-shot/monostable (self-terminating burst via delayed self-inhibition), and a double-pulse pair detector (delay line + coincidence). Dynamics are the paper's **asynchronous edge-triggered pulses** (`nv_evo/pulse.py`): an event-driven simulation where a triggered node emits a fixed-width pulse after a fixed delay, coincidence requires both excitatory edges within a window, and inhibition vetoes at trigger time; scoring samples the wires once per tick. Every element (identical tiled nodes, 4-bit routing SRAM, genome as associative memory / CAM, delay / pulse-width / coincidence-window constants) maps directly onto the paper's hardware.

- **LUT array** (`lut_evo/`): the paper's **Architecture 2**, a faithful port of the `sim6` reference — a **square** grid where each cell wires to **4 neighbours** (N/S/E/W) and holds **four** directional 16-bit lookup tables (one per output; `automaton_arrays.pdf`: "4 lookup tables per cell, each 16 states"). Growth looks each direction's LUT up with the neighbour context **rotated** so the output direction is "front" (context rotation); dynamics are latched and synchronous — each cell indexes its four LUTs with the 4 bits its neighbours aim back at it. Unlike the nervous net, LUT cells can invert, so spontaneous activity exists (faithful to the paper's Fig. 14). Runs the same temporal and combinational targets.

  (The **hex** nervous array is the degree-3 case — `automaton_arrays.pdf`: 3 neighbours, 3 LUTs of 8 states — drawn as a brick-wall; deliberately *not* the degree-6 triangular reading.)

The SNN and nervous backends share no code; the LUT backend shares only the problem definitions and trace-scoring maths. Growth in the nervous and LUT models happens on an **unbounded field** — "if the field is big enough, the circuit can dictate its own boundary" — and runs until it **converges to a stable attractor** (the paper's "naturally self-limiting growth" / maturity). Size is bounded **genetically** rather than by a wall. In the nervous net the evolvable **telomere** is now a per-cell **Hayflick limit**: seed/germline cells start with the genome's telomere length *L*, a cell may divide (bring an empty neighbour to life via a `self_in == 0` growth rule) only while its telomere is unspent, and each daughter inherits one less — so the organism provably halts its own growth at radius *L* from the seeds. This *alone* bounds both size and duration, so the old grid-size clip and iteration cap have been **removed** (they no longer affect nervous growth — grid size survives only as an I/O layout scale). *Maintenance* rules (`self_in != 0`) act on live cells every step regardless of telomere — telomeres limit replication, not function, as in biology. (The LUT backend still uses the simpler per-chromosome telomere that expires growth rules after a fixed iteration count.) The GA never stops early at fitness 1.0 — runs continue so that, via a **senescence / parsimony tie-break** (at equal fitness the smaller genome wins, then the one that grows a smaller, cheaper body — a shorter telomere), solved genomes keep shrinking. Convergence is **not forced**: reproduction stays a single steady exploratory regime (elitism + tournament/lexicase parents + random immigrants), so the population contracts onto a genome only as far as selection genuinely favours it — it converges naturally when a solution truly out-competes the field, and keeps exploring otherwise (the population *mean* is therefore not herded up to the best).

## How it works

- **Indirect encoding**: the genome is an associative memory of rules mapping a cell's neighbourhood state → output state, looked up by minimum Hamming distance (the nervous-net gene is exactly the paper's `context → new state`, with no per-gene time field)
- **Context rotation** (nervous net): each node reads its L/R/D neighbours in its own orientation frame, so one gene can grow symmetric structure ("right/left/down rotated to match topology", EH'02 Fig. 4)
- **Circuit ontogeny**: growth iterations expand the input seeds into a full circuit
- **LIF simulation** (SNN): neurons communicate via synaptic currents; output is read as spike / no-spike
- **Per-output encoding** (SNN): an output may use complement inputs and/or an inverted spike reading — e.g. the half-adder's carry (AND) is evaluated with flipped inputs + inverted output, making it trivially learnable
- **Pluggable targets**: a `Target` (`snn_evo/targets.py`) is the single source of truth for a problem's input seeds, output terminals and truth table; the growth, interpretation, fitness and GUI all read their I/O layout from it
- **Combinational target library** (`snn_evo/targets.py`, run on **all three** models): logic gates, half/full/2-bit adders, plus the harder **2:1 MUX**, **Majority-3**, **Parity-3 (XOR3)**, **2-to-4 decoder**, **2-bit comparator** (GT/EQ/LT) and **2×2 multiplier** (4-bit product) — data routing, voting, wide parity, relational logic and arithmetic
- **Temporal target library** (`nv_evo/targets.py`, run on **Nervous + LUT**): latch, toggle, echo, oscillator, one-shot, pair/coincidence detector, plus **Temporal XOR** (fire iff exactly one input pulses), **Sequence A→B** (ordered two-input detector), **Veto gate** (B suppresses A), **Burst ×3** (one kick → fixed spike burst) and **Divide-by-3** (fire every 3rd pulse — modulo-3 counter)
- **Temporal targets** (`nv_evo/targets.py`): scored on output *traces* over time, across several trials with shifted pulse timings — so only genuine memory passes, not a lucky delay chain
- **Oracle targets** (`nv_evo/oracle.py`): rather than hand-picking input timings (which can be adversarial to a circuit's internal phase — an evolved SR latch with a period-2 loop resets on even ticks and misses odd ones), a goal is specified as a *reference state machine* `oracle(inputs, state) -> (outputs, state)` plus a *stimulus generator*. Many random schedules are sampled and labelled by the oracle; the circuit is scored on reproducing the input→output *relation*, and `holdout_score` re-samples fresh schedules to certify it generalises rather than memorising timings. Registered as the `… (oracle)` variants in the target dropdown.
- **Fitness for temporal targets** (`nv_evo/temporal.py`): the question is *"did the network produce the correct spike events?"*, not *"was the output level right at every tick?"* — three ideas make memory learnable —
  - *Spike-event precision/recall (F1)* (`METRIC = 'f1'`, the default): reward each correctly-timed output spike, penalise expected spikes that never occur, penalise extra spikes that weren't expected, pooled over every trial. A do-nothing output recalls nothing and so scores **0** — silence is never rewarded — while a fire-constantly output has terrible precision; only the correct spikes at the correct ticks (and nowhere else) reach 1.0. (The older windowed/level-balanced metric, which handed a silent net 0.5, is kept only for diagnostics.)
  - *Trace-matched output placement*: each output role is read at whichever live cell best reproduces the expected trace, so evolution only has to build the mechanism, not also route the answer to one prescribed cell.
  - *Phase-tolerant recall*: because a bit is stored as a pulse *circulating* in a loop, it reads as a ripple (e.g. `1010`) at any one cell — the honeycomb has no triangles, so the phases can't be OR'd back into a steady level. Recall therefore allows ±1 tick of coverage (a stored 1 counts as hit if the cell fires on that tick or an adjacent one), while precision stays exact so spurious firing is still punished.
- **Describe a target as spike events** (`nv_evo.spike_target`): define a new temporal function directly from test cases — `spike_target(name, cases, T, n_inputs=…)` where each case is `(input_spikes, output_spikes)` (input ticks per input, expected output ticks). Pair it with `nv_evo.ga.diversify` to turn one solution into a whole generation of genotypically-unique valid solvers.
- **Loop-aware GA** (`nv_evo/ga.py`): shaping that rewards feedback cycles the inputs can write and the outputs can read, gene-duplication mutations that build repeated loop motifs, random immigrants against premature convergence, and a fitness cache
- **Biologically-inspired search dynamics** (`nv_evo/ga.py`, `lut_evo/ga.py`): a **senescence / metabolic cost** (at equal fitness the genome that grows a smaller, cheaper body wins — parsimony extended to organism size via the telomere); **stress-induced hypermutation** (the bacterial SOS response — the mutation rate ramps up when a run stalls and relaxes on progress); and **simulated-annealing mutation decay** (an *Anneal α* that multiplies the per-child mutation rate each generation, for a hot-start / cool-down schedule). Very long runs stay tractable — the fitness cache is size-bounded and the live fitness chart is decimated, so 10k–100k-generation runs don't gum up. Population evaluation reuses a **single persistent worker pool** across generations (`eval_batch_cases(..., executor=)`); spawning a fresh pool every generation dominated runtime on Windows, so reuse is a large speed-up. Reproduction avoids `copy.deepcopy` — `clone_genome` copies only the genome structure and shares the (never-mutated-in-place) gene objects, ~13× faster than deepcopy which dominated `next_population`; and the min-Hamming growth lookup inlines its 4-bit popcount, so a run is bounded by real work, not copy/call overhead
- **Substrate topology** (both lattices): the hex (degree-3) and square (degree-4) grids are **bipartite** — 2-colourable by `(x+y)` parity, hex girth 6 — so a single pulse can only circulate an **even-length** loop. Odd output periods are *not* impossible, but they cost more: a period-*p* output needs a length-*2p* loop carrying two evenly-spaced pulses. That's why a period-2 target (toggle) solves trivially while an odd-period one (e.g. the period-3 pattern generator) plateaus — the cheap single-pulse loop is topologically unavailable, not the behaviour itself
- **LUT logic view** (`lut_evo/boolfn.py`): every 16-bit lookup table is really a boolean function of the four neighbour input bits, so it is decoded to a minimised sum-of-products expression `out = f(N,S,E,W)` (verified exact over all 65 536 tables) — the genome reads as logic instead of hex, and the Growth tab shows the mature organism's distinct tables as actual 4×4 truth grids
- **sim6 ontogeny** (`lut_evo/ontogeny.py`): a faithful port of the reference `table_create` morphogenesis that *grows* a dense genome on the fly (inventing a gene per unseen context), reproducing sim6's varied biomorphs. Runs as a standalone shape-browser (`py -m lut_evo.ontogeny`) or as an optional GUI **Ontogeny seed** toggle for the LUT backend. Diagnostic finding: the GA's drift to small "diamond" genomes is genome *sparsity*, not the growth engine — dense ontogeny seeds restore the rich shapes but (measured) solve *worse* and cost ~50×, so the toggle is off by default

## Files

| File | Description |
|------|-------------|
| `app.py` | Single-window GUI: pick a model + target, evolve, inspect growth / activity / genome, drive circuits interactively |
| `snn_evo/` | SNN backend (genome, growth, LIF sim, fitness, targets, GA) |
| `snn_evo/targets.py` | Combinational target registry + builders (gates, adders, custom truth tables) |
| `nv_evo/` | Nervous-net backend (hex genome, honeycomb growth, pulse dynamics, targets, GA) |
| `nv_evo/targets.py` | Temporal target registry (SR latch, toggle, oscillator, pattern generator, echo, coincidence, one-shot, pair detector) |
| `lut_evo/` | LUT-array backend (16-bit lookup cells, paper Architecture 2) |
| `lut_evo/boolfn.py` | Decode 16-bit LUTs to boolean logic (minimised SOP over N/S/E/W) |
| `lut_evo/ontogeny.py` | sim6 `table_create` morphogenesis — shape browser + optional GA seed |
| `nv_evo/ga.py` | Loop/memory-tuned GA (shaping, duplication, immigrants, caching, senescence cost, SOS hypermutation, annealing) |

## Usage

```bash
pip install -r requirements.txt

python app.py        # launch the GUI
```

In the GUI: pick a **Model** (SNN / Nervous / LUT) and a **Target** from the dropdown
(or click **Custom…** to enter your own truth table), set Pop / Gens / Tries plus the
GA tuning row — **Mutations/child**, **Anneal α** (α < 1 cools the mutation rate
each generation for a hot-start simulated-annealing schedule; α = 1 is off),
immigrants, tournament size and **Elites** (exact number of top genomes copied
unchanged into the next generation) — and click **Run**. The fitness chart plots
three lines: all-time best (monotonic), the best of the current generation (dips
at each restart when Tries > 1), and the population mean. Defaults are a single long run
(Gens 500, Tries 1) with a hot-start anneal (Mutations 4.0, α 0.99) so slow, steady
progress is visible. Growth is self-limiting via the telomere, so there are no
grid-size or iteration controls; **Max telomere** caps how large organisms may
grow (a chromosome's telomere is its growth *radius*, and eval cost ≈ radius²) —
lower it (e.g. 8–10) to keep runs fast, or raise it to allow bigger nets. Use **Load Saved** to reload `results/best_genome.pkl`
and **Save PNGs** to export the growth and voltage figures.

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
