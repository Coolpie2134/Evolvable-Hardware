# Evolvable Hardware — Logic Circuits via Grown Neural Substrates

A research project implementing **Edwards indirect encoding** to evolve grown circuits, with three backends: spiking neural networks (SNN), hexagonal "nervous nets", and a square **LUT array** (the paper's Architecture 2 / `sim6`).

## What's new

- **Paper-faithful tri-circuit tile** (`nv_evo/tritile.py`): the nervous net can now run Edwards EH'02 Fig. 2 as written — **three independently configured circuits per tile**, one per L/R/D output direction (12-bit tile state), instead of the legacy single broadcast circuit. One tile can route two signals to two outputs independently, which the single-circuit tile physically cannot.
- **Analog node physics** (`nv_evo/analog.py`): an event-driven model of the paper's Fig. 1 circuit — capacitive input steps, Vbp leak toward Vdd, comparator with hysteresis. Coincidence window, output pulse width, dense-input stretch (paralyzability) and refractory recovery all **emerge from the same physical constants** instead of being independent knobs; validated against an independent numerical integrator.
- **NV run profiles**: fresh nervous-net runs choose one of three coherent substrate profiles from a single GUI dropdown — **Legacy** (single tile, width-preserving + evolved delay), **Digital tri-circuit** (paper tile, frozen digital node — the topology-ablation leg), and **Analog tri-circuit** (paper tile + analog physics). Retired engine pairings still load from old checkpoints but cannot start new runs (`evo_runtime/config.py: validate_new_nv_profile`). The analog profile exposes its physical constants (Vth, step, tau, hysteresis) in a dedicated GUI row.
- **One scoring contract** (`nv_evo/scoring.py`): the seven historical score modes collapsed into a single module with a relation registry — a **match** family (edge F1 / interval / coverage relations) and a **rhythm** family (cadence invariants) — consumed by both async backends, the Designer, and the retention pipelines. A golden-fixture equivalence gate (`tests/test_scoring_equivalence.py` + `tools/make_scoring_golden.py`) pins every score to 1e-12, so scoring semantics can never drift silently again.

## Background — grown, not designed

Conventional *evolvable hardware* evolves a circuit by encoding it **directly**: the genome is a blueprint, one gene per wire or gate. That doesn't scale — the genome grows with the circuit, identical sub-structures must be rediscovered independently, and the fitness landscape is jagged because one bit-flip rewires one gate.

This project takes the **indirect** route from Edwards' *Evolvable Hardware* work: the genome encodes **growth rules**, not a circuit. A circuit is *grown* by a developmental process (ontogeny) from a handful of seed cells, the way a body is grown from DNA. Every cell carries the same genome; each consults it as an **associative memory** — "given the state of my neighbourhood, what do I become?" — so position and local context decide a cell's fate. This buys three things the direct encoding lacks:

- **Compact, scale-free genomes** — a fixed rule set grows an arbitrarily large organism, and a rule that builds one motif builds it everywhere.
- **Evolvability** — small genome changes produce coherent, often-repeated structural changes (a rule tweak reshapes every copy of a motif at once), which smooths the search landscape.
- **Self-limiting size** — growth halts on its own, at a boundary the genome dictates (here, an evolvable telomere / Hayflick limit), rather than at an arbitrary wall.

The substrate is a regular lattice of identical, reconfigurable cells. This repo implements three such cell types — a spiking neuron plus the two cellular-automaton architectures from the Edwards papers — and evolves both **combinational** logic (truth tables) and **temporal** circuits, where the answer is carried in the *timing* of events: memory that latches, oscillators that free-run, counters, delay lines and pattern generators.

## The three substrates at a glance

| | **SNN** (`snn_evo/`) | **Nervous net** (`nv_evo/`) | **LUT array** (`lut_evo/`) |
|---|---|---|---|
| Origin | LIF variant | Edwards EH'02 honeycomb | paper Architecture 2 / `sim6` |
| Lattice | square grid | honeycomb, **degree 3** (bipartite) | square grid, **degree 4** (bipartite) |
| Cell function | leaky integrate-and-fire neuron | fires on `(E1 op E2) AND NOT I1` — coincidence/veto over 16 routing states (+16 OR twins) | four directional 16-bit lookup tables |
| Dynamics | **event-driven** continuous LIF, synaptic currents | **asynchronous** edge-triggered pulses | **asynchronous** level logic, event-driven (inertial gate delay) |
| Spontaneous activity? | — | **no** — inert until an input is injected | **yes** — cells can invert |
| Computes | combinational | combinational **+ temporal** | combinational **+ temporal** |
| Shares code with others | none | — | only problem defs + trace-scoring maths |

## What it does

A genetic algorithm evolves a genome that encodes growth rules. Starting from input seed cells, a cellular automaton grows over a number of iterations into a circuit. Two separate backends interpret and evaluate the grown grid:

- **SNN** (`snn_evo/`): a square grid of leaky integrate-and-fire (LIF) neurons computing combinational logic — single gates (AND/OR/XOR/NAND/NOR/XNOR), half/full adders, multi-bit adders, or any custom truth table you enter in the GUI. Growth is **telomere-bounded** like the nervous net: each chromosome carries an evolvable Hayflick division limit, so a lineage stops dividing once its telomere is spent and the body self-limits at radius L from the seeds (replacing the old fixed iteration cap), and the GA's senescence/parsimony tie-break then shrinks that body — fewer genes, shorter telomere — toward the smallest circuit that still solves. (A generous `GRID_SIZE` remains as an outer wall so growth coordinates stay valid for the growth view and fixed I/O terminals; the telomere is the real limiter.) Its interactive view is a live LIF animation — see `interactive.py`.
- **Nervous net** (`nv_evo/`): a honeycomb array of nodes that fire when `(E1 op E2) AND NOT I1` — coincidence/veto detection over the paper's 16 routing states of Edwards EH'02 Fig. 3 (states 0–15: each one a buffer or an AND, optionally vetoed — the paper itself has **no disjunction**), plus 16 **OR twins** (states 16–31: the same wiring but `op = OR`, so the node fires on *either* excitatory input) added as a deliberate extension beyond the paper. The extra state bit is 0 for every legacy genome, so 0–15 grow and behave bit-identically; only the OR of two *different* lines is genuinely new (for a buffer or off state, OR and AND coincide, so those twins are benign aliases). Directions are read in each node's own orientation frame (the paper's context rotation — down-parity tiles are the 180°-rotated circuit), and inputs are **wired-OR injections** onto the input cell's net, like pulsing the physical perimeter wire — not level clamps. So **with no input the array is inert**, yet a pulse injected into a loop of buffers routed through an input cell circulates as delay-line memory "until stopped by application of an inhibitory input" (§3). Besides combinational targets it evolves **temporal circuits** — SR latches, toggle flip-flops, oscillators, delay lines, plus async-native ones where edge timing *is* the computation: a two-input coincidence detector, a one-shot/monostable (self-terminating burst via delayed self-inhibition), and a double-pulse pair detector (delay line + coincidence). Dynamics are the paper's **asynchronous edge-triggered pulses** (`nv_evo/pulse.py`): an event-driven simulation where a triggered node emits a fixed-width pulse after a fixed delay, coincidence requires both excitatory edges within a window, and inhibition vetoes at trigger time. Every wire's raw continuous leading-edge timestamps are retained for behavioral scoring; once-per-tick samples exist only for playback and legacy persistence windows, so sub-tick pulses are no longer discarded. Every element (identical tiled nodes, 5-bit routing SRAM — 4 bits in the paper, the extra bit selecting the AND/OR variant — genome as associative memory / CAM, delay / pulse-width / coincidence-window constants) maps directly onto the paper's hardware.

- **LUT array** (`lut_evo/`): the paper's **Architecture 2**, a faithful port of the `sim6` reference — a **square** grid where each cell wires to **4 neighbours** (N/S/E/W) and holds **four** directional 16-bit lookup tables (one per output; `automaton_arrays.pdf`: "4 lookup tables per cell, each 16 states"). Growth looks each direction's LUT up with the neighbour context **rotated** so the output direction is "front" (context rotation); dynamics are **asynchronous level logic** (`lut_evo/pulse.py`): each cell is a logic element with a fixed propagation delay that re-indexes its four LUTs with the 4 bits its neighbours aim back at it whenever an input wire changes (inertial delay — sub-delay blips are filtered), simulated event-driven in continuous time. With delay = 1 tick and lattice stimuli this reproduces the `sim6` synchronous latched update bit for bit (the retained `lut.LutSim` is the audited quantization reference). Unlike the nervous net, LUT cells can invert, so spontaneous activity exists (faithful to the paper's Fig. 14). Runs the same temporal and combinational targets.

  (The **hex** nervous array is the degree-3 case — `automaton_arrays.pdf`: 3 neighbours, 3 LUTs of 8 states — drawn as a brick-wall; deliberately *not* the degree-6 triangular reading.)

- **Nervous-net substrate profiles** (the **NV profile** dropdown in the pulse-physics row; `evo_runtime/config.py: NV_NEW_RUN_PROFILES`): a fresh nervous-net run picks one of three coherent architecture/physics pairings —
  - **Legacy: width-preserving + evolved delay** (`single` + `pulse_delay`) — the tuned single-circuit tile: every node transports the complete incoming waveform after its routing-state delay `d_node`: `[t, t+w)` becomes `[t+d_node, t+d_node+w)`. Both edges use the same heritable delay, so pulse width survives each hop while propagation speed evolves.
  - **Digital tri-circuit** (`tri3` + `uniform`) — the paper's Fig. 2 tile (three independent L/R/D circuits, 12-bit tile state, per-channel single-bit mutation) under the frozen digital node abstraction (fixed `WIDTH` after fixed `DELAY`, `COINC` window). The topology-ablation leg: same tile as analog, controlled physics.
  - **Analog tri-circuit** (`tri3` + `paper_analog`) — the paper tile with the Fig. 1 analog node (`nv_evo/analog.py`): capacitive down-steps, Vbp leak, comparator + hysteresis. Coincidence, output width, and refractory are **emergent**, so the Coinc knob is disabled and the analog constants (Vth / step / tau / hysteresis) get their own GUI row, defaulting to the audited frozen values.
  Retired pairings (`uniform` on the single tile, and other combinations) remain in the engine solely so old checkpoints load and controlled comparisons replay; `validate_new_nv_profile` rejects them for new runs. A further variant made the emitted pulse width itself a per-node-type genome vector; **width evolution has been removed from the substrate** — the genome no longer carries a width vector, and a checkpoint saved under it loads on the paper's fixed-width node instead. Timing models are audited in `tests/test_pulse_models.py`, the tri tile in `tests/test_tritile.py`, and the analog node against an independent Δt integrator in `tests/test_analog_reference.py`. Width-sensitive temporal targets now exercise useful relations rather than isolated node contracts: **Pulse width sum (A+B)** emits a Q interval whose duration is `width(A)+width(B)`, while **Odd pulse selector** passes the 1st/3rd/5th input pulses and preserves their individual widths. Both score complete output waveforms (`PulseSim.pulse_intervals`, real rises *and* falls), rather than treating pulses with equal leading edges as equivalent. A related mixed-width stateful family counts every physical pulse once regardless of its 0.5-to-2.25-second duration: **A-count parity queried by B** retains cumulative odd/even parity, **A-count multiple-of-3 queried by B** recognizes positive multiples of three, and **Odd A batch closed by B** reports parity since the preceding B and then clears it.

The SNN and nervous backends share no code; the LUT backend shares only the problem definitions and trace-scoring maths. Growth in the nervous and LUT models happens on an **unbounded field** — "if the field is big enough, the circuit can dictate its own boundary" — and runs until it **converges to a stable attractor** (the paper's "naturally self-limiting growth" / maturity). Size is bounded **genetically** rather than by a wall. In the nervous net the evolvable **telomere** is now a per-cell **Hayflick limit**: seed/germline cells start with the genome's telomere length *L*, a cell may divide (bring an empty neighbour to life via a `self_in == 0` growth rule) only while its telomere is unspent, and each daughter inherits one less — so the organism provably halts its own growth at radius *L* from the seeds. This *alone* bounds both size and duration, so the old grid-size clip and iteration cap have been **removed** (they no longer affect nervous growth — grid size survives only as an I/O layout scale). *Maintenance* rules (`self_in != 0`) act on live cells every step regardless of telomere — telomeres limit replication, not function, as in biology. (The LUT backend still uses the simpler per-chromosome telomere that expires growth rules after a fixed iteration count.) The GA never stops early at fitness 1.0 — runs continue so that, via a **senescence / parsimony tie-break** (at equal fitness the smaller genome wins, then the one that grows a smaller, cheaper body — a shorter telomere), solved genomes keep shrinking. Before a solution exists, reproduction remains a full-replacement exploratory regime (tournament/lexicase parents, real mutations and random immigrants). Once a terminal 1.0 circuit appears, evaluated **parent + offspring survivor selection** retains successful circuits while the same exploratory offspring continue to be tested. That makes the live population mean converge toward the best without disguising self-crosses or no-op mutations as new children.

## How it works

- **Indirect encoding**: the genome is an associative memory of rules mapping neighbourhood context → output state, looked up by minimum Hamming distance. A nervous-net rule contains five scalar 4-bit values (`L context`, `R context`, `D context`, `self in`, `self out`), each strictly **0–15**; packed 12-bit values exist only in developed tiles.
- **Context rotation** (nervous net): one scalar gene is evaluated independently for each of a tile's L/R/D core circuits. Its physical neighbour context is rotated so the circuit being updated is "front", letting the same rule grow symmetric structure ("right/left/down rotated to match topology", EH'02 Fig. 4).
- **Hierarchical crossover**: multi-gene chromosomes exchange reciprocal suffixes at a real interior boundary; a one-gene chromosome exchanges a proper subset of that rule's active fields, so minimal genomes can still produce recombinant children. Parent choice prefers different rule content, and multi-edit mutation transactions finish with a protected non-parent allele so inverse edits cannot quietly cancel back to a clone
- **Circuit ontogeny**: growth iterations expand the input seeds into a full circuit
- **LIF simulation** (SNN): neurons communicate via synaptic currents; output is read as spike / no-spike
- **Per-output encoding** (SNN): experimental targets may use complemented inputs and/or an inverted spike reading, but all bundled arithmetic carries are direct: input/output high means logic 1
- **Pluggable targets**: a `Target` (`snn_evo/targets.py`) is the single source of truth for a problem's input seeds, output terminals and truth table; the growth, interpretation, fitness and GUI all read their I/O layout from it
- **Combinational target library** (`snn_evo/targets.py`, run on **all three** models): logic gates, half/full/2-bit adders, plus the harder **2:1 MUX**, **Majority-3**, **Parity-3 (XOR3)**, **2-to-4 decoder**, **2-bit comparator** (GT/EQ/LT) and **2×2 multiplier** (4-bit product) — data routing, voting, wide parity, relational logic and arithmetic
- **Temporal target library** (`nv_evo/targets.py`, run on **Nervous + LUT**): latch, toggle, echo, oscillator, one-shot, pair/coincidence detector, plus **Temporal XOR** (fire iff exactly one input pulses), **Sequence A→B** (ordered two-input detector), **Veto gate** (B suppresses A), **Burst ×3** (one kick → fixed spike burst), **Divide-by-3** (fire every 3rd pulse — modulo-3 counter) the **C-element (2-input join)** — a transition-signalling Muller C-element / rendezvous that emits only once *both* inputs have produced an edge (either order) then rearms, the asynchronous-handshake keystone (it must *remember* the first arrival while it waits for the second). Its corrected bank explicitly tests A-first, B-first, ties, repeated first arrivals, incomplete rounds, and A-only/B-only/silent guards; the fixed reproduction budget currently reaches about 0.78 training and 0.72 held-out, so a fully solved C-element remains an open frontier rather than a certified claim. The **Period doubler (2×)** asks: for a periodic input train of period p, emit every 2nd input edge so the output is periodic at 2p. Edge-native by design — the input information is inter-edge *intervals*, exactly what an asynchronous substrate computes with. The bank mixes periods p ∈ {2,3,4} at varying phases (a fixed free-running cadence fits only one rate) plus a silent guard trial (no input ⇒ no output, killing pure oscillators); period 1 is deliberately excluded because a pulse every tick wired-OR merges into one held level and carries no period. (A pulse-*width* doubler variant — hold x ticks in, hold 2x out, widths mixed for the same anti-cheat reason — remains available programmatically as `Pulse doubler (oracle)` in `ORACLE_SPECS`.)
- **Interval transformation targets**: **Period tripler (3×)** emits every third input edge, **Period halver (½×)** first measures a complete even input period and then inserts midpoint events, and **Temporal sum (ΔA + ΔB)** measures two intervals on each of two inputs and encodes their sum as the interval between two Q events. Their seeded banks mix periods/intervals and include silent, one-lane, and incomplete guards, preventing direct wires, fixed bursts, and free-running oscillators from solving them.
- **Physical pair target** (Nervous + LUT — the asynchronous backends): **Pair detection gap (2x pulse width)** supplies explicit fractional-phase input pulses and emits Q only when consecutive leading edges are separated by exactly twice the supplied input-pulse width; wrong relative gaps, chains, isolated pulses, and silence are included.
- **Temporal time units**: target names, descriptions, reports, and plots use seconds. Nervous events may occur at fractional seconds; one LUT gate delay is defined as one second.
- **Temporal targets** (`nv_evo/targets.py`): scored on raw events, cadence invariants, or persistence traces as the behavior requires, across several shifted trials — so only genuine timing/memory passes, not a lucky delay chain
- **Oracle targets** (`nv_evo/oracle.py`): rather than hand-picking input timings (which can be adversarial to a circuit's internal phase), a goal is specified as a *reference state machine* `oracle(inputs, state) -> (outputs, state)` plus a *stimulus generator*. Many random schedules are sampled and labelled by the oracle; the circuit is scored on reproducing the input→output *relation*, and `holdout_score` re-samples fresh schedules to certify it generalises rather than memorising timings. Memory commands are globally spaced by 10–12 seconds so the output has a real observation interval in which ringing, quiet, clearing, and toggling can be distinguished. Registered as the `… (oracle)` variants in the target dropdown. The oracle-defined set is: **SR latch**, **Toggle flip-flop**, **Echo** (an exact three-second delay: unlike relation-only targets it does not fit a free latency, so a direct wire fails), **One-shot** (monostable), **Pair detector**, plus three that exercise memory *and* a control line — **Gated oscillator** (input A starts a free-running period-2 oscillation, input B stops it — the paper's "circulate a loop until stopped by an inhibitory input" made controllable), **Resettable toggle** (input A flips a stored bit, input B clears it), and the **Period stepper** (a *cadence* controller: the first command starts a period-2 oscillation and each later command steps it slower — 2 → 4 → 6 — scored on sustaining the cadence, not matching an arbitrary transient phase; see the period-stepper scoring below)
- **Fitness for temporal targets** (`nv_evo/scoring.py` — the single scoring contract; `nv_evo/temporal.py` is the harness that runs trials and places outputs): the question is *"did the network implement the timing relation?"*, not *"was the output level right at every tick?"*. Every score mode is an entry in one relation registry (`RELATIONS`), split into a **match** family (compare against per-trial expectations) and a **rhythm** family (measure an invariant with legitimately free phase) —
  - *Raw point-event precision/recall* (`score_mode='events'`): expected and produced leading edges are matched one-to-one within the substrate's time tolerance. Relation-only targets fit one continuous latency offset shared by every trial/output; precision-delay targets can disable that fit and require their declared absolute timing, as Echo delay 3 does. Missing events reduce recall, extras reduce precision, and silence cannot pass a positive target. The LUT feeds the same contract with its wires' raw leading-edge timestamps.
  - *Cadence invariants* (`score_mode='cadence'`): oscillators and repeating one-pulse patterns are scored on regular inter-event intervals, sustained coverage of the post-kick dwell, and pre-kick silence; absolute phase is irrelevant.
  - *Persistence windows* (`score_mode='trace'`): latches, toggles and controlled memory retain phase-tolerant active/quiet regimes because two valid circulating-pulse memories can have different event cadences. The oracle latch adds explicit never-set and Set→Reset→Set contrasts so a one-shot or unconditional oscillator cannot masquerade as storage.
  - A deterministic per-run event cap rejects pathological oscillators before one genome monopolises evaluation. Event/cadence evaluation schedules input edges directly, skips `T × cells` display snapshots, reuses one input cone across every trial, and uses allocation-free sparse matching; sampled states are built only for playback and persistence targets.
  - *Trace-matched output placement*: each output role is read at whichever live cell best reproduces the expected trace, so evolution only has to build the mechanism, not also route the answer to one prescribed cell.
  - *Phase-tolerant recall*: because a bit is stored as a pulse *circulating* in a loop, it reads as a ripple (e.g. `1010`) at any one cell — the honeycomb has no triangles, so the phases can't be OR'd back into a steady level. Recall therefore allows ±1 tick of coverage (a stored 1 counts as hit if the cell fires on that tick or an adjacent one), while precision stays exact so spurious firing is still punished.
- **Semantic period-stepper scoring** (`score_mode='period_stepper'`, `nv_evo/temporal.py`): some behaviours have no single "correct" trace to match. A cadence controller must merely hold a *regular* output pulse rate and make it *slower* each time it is commanded — the absolute phase is irrelevant. Exact-trace F1 would reward a plain period-2 oscillator that happens to overlap the fast segment. Instead the stepper is scored on its *invariant*: each command-delimited dwell is checked for a sustained regular cadence (enough evenly-spaced pulses covering the dwell), the transition tick is left unscored, and every later dwell must have a strictly longer period than the one before. A fixed-rate oscillator scores **0** on the "gets slower" term no matter how many raw pulses it matches. A single bounded, causal latency is searched for the whole genome, keeping the useful latency-invariance without letting a giant shift hide a missed early dwell.
- **Describe a target as spike events** (`nv_evo.spike_target`): define a new temporal function directly from test cases — `spike_target(name, cases, T, n_inputs=…)` where each case is `(input_spikes, output_spikes)` (input ticks per input, expected output ticks). Pair it with `nv_evo.ga.diversify` to turn one solution into a whole generation of genotypically-unique valid solvers.
- **Loop-aware GA** (`nv_evo/ga.py`): shaping that rewards feedback cycles the inputs can write and the outputs can read, gene-duplication mutations that build repeated loop motifs, random immigrants against premature convergence, and a fitness cache
- **Biologically-inspired search dynamics** (`nv_evo/ga.py`, `lut_evo/ga.py`): a **senescence / metabolic cost** (at equal fitness the genome that grows a smaller, cheaper body wins — parsimony extended to organism size via the telomere); **stress-induced hypermutation** (the bacterial SOS response — after 12 flat generations the mutation rate ramps up and relaxes on progress); and **simulated-annealing mutation decay** (an *Anneal α* that multiplies the per-child mutation rate each generation). The editable **Plateau β** controls the reheat slope: 0 disables plateau reheating, 1 preserves the tuned behavior, and larger values raise mutation faster. The chart plots the effective rate on a secondary axis. Very long runs stay tractable — the fitness cache is size-bounded and the live fitness chart is decimated, so 10k–100k-generation runs don't gum up. Population evaluation reuses a **single persistent worker pool** across generations (`eval_batch_cases(..., executor=)`); spawning a fresh pool every generation dominated runtime on Windows, so reuse is a large speed-up. Reproduction avoids `copy.deepcopy` — `clone_genome` copies only the genome structure and shares the (never-mutated-in-place) gene objects, ~13× faster than deepcopy which dominated `next_population`; and the min-Hamming growth lookup inlines its 4-bit popcount, so a run is bounded by real work, not copy/call overhead
- **Substrate topology** (both lattices): the hex (degree-3) and square (degree-4) grids are **bipartite** — 2-colourable by `(x+y)` parity, hex girth 6 — so a single circulating pulse can only traverse an **even-length** loop, and its output period is therefore even. Odd output periods are *not* impossible, but they cost far more: a period-*p* output needs a length-*2p* loop carrying two evenly-spaced pulses (`output_period = loop_length / n_pulses`), a conjunction the GA path dips through lower fitness to reach and empirically never crosses. This is a real design constraint, not a bug: a period-2 target (toggle) solves trivially, and the **pattern generator** is deliberately set to an even period — `Pattern (1000)`, period 4, one pulse in a length-4 loop — which the cheap single-pulse route reaches directly. The earlier odd-period `Pattern (100)` was retired precisely because parity made it topologically out of reach (neither a bigger grid nor hundreds of generations moved it off ~0.76). Autonomous targets that must run on **both** lattices are chosen with this in mind
- **LUT logic view** (`lut_evo/boolfn.py`): every 16-bit lookup table is really a boolean function of the four neighbour input bits, so it is decoded to a minimised sum-of-products expression `out = f(N,S,E,W)` (verified exact over all 65 536 tables) — the genome reads as logic instead of hex, and the Growth tab shows the mature organism's distinct tables as actual 4×4 truth grids
- **sim6 ontogeny** (`lut_evo/ontogeny.py`): a faithful port of the reference `table_create` morphogenesis that *grows* a dense genome on the fly (inventing a gene per unseen context), reproducing sim6's varied biomorphs. Runs as a standalone shape-browser (`py -m lut_evo.ontogeny`) and is **the** LUT seed path: every LUT population/immigrant genome (evolver and Designer alike) comes from `lut_evo.ga.make_seed_genome` — a dense biomorph seed drawn from a cached pool so immigrants stay cheap — and the LUT GA drops the smaller-genome parsimony tie-break, which used to prune the rich shapes back to sparse uniform diamonds. (Diagnostic finding behind this: the "uninteresting" diamond growth was genome *sparsity*, not the growth engine.)

## Files

**Entry points**

| File | Description |
|------|-------------|
| `app.py` | Single-window GUI: pick a model + target, evolve, inspect growth / activity / genome, drive circuits interactively |
| `designer.py` | Manual circuit **designer** — build a circuit by hand at either encoding level: edit the genome (chromosomes/genes) and press Grow, or edit the grown grid directly (place cells, set routing states). Simulate and score it against any target; available as a GUI tab and standalone |
| `diversity_ui.py` | "Diversity" tab — load a solved population and analyse it: the four-level collapse funnel and (opt-in) the mutational-robustness histogram. Runs on a worker thread with progress and Stop; nervous/LUT only |
| `interactive.py` | "Interactive / Test" tab — after evolving/loading a circuit, drive its inputs and watch the response play out. SNN shows live LIF playback; nervous nets and LUT arrays share the same clickable continuous-time pulse timeline, played through their respective asynchronous event engines |
| `nv_evo/playback.py` | Shared continuous-time nervous-net player and pulse-lane editor used by Interactive and Designer; wraps the same paper-faithful `PulseSim` used by Nervous evolution |
| `concept_gui.py` | GUI playground for the proof-of-concept GAs in `concept/`, with live matplotlib fitness / best-organism visuals |

**Backends**

| File | Description |
|------|-------------|
| `snn_evo/` | SNN backend (genome, growth, LIF sim, fitness, targets, GA) |
| `snn_evo/targets.py` | Combinational target registry + builders (gates, adders, MUX, majority, parity, decoder, comparator, multiplier, custom truth tables) |
| `nv_evo/` | Nervous-net backend (hex genome, honeycomb growth, pulse dynamics, targets, GA) |
| `nv_evo/targets.py` | Temporal target registry — oscillator, `Pattern (1000)`, coincidence, temporal XOR, sequence, veto, burst, divide-by-3 (hand-built); plus the oracle variants: SR latch, toggle, echo, one-shot, pair detector, period doubler/tripler/halver, temporal sum, period stepper, gated oscillator, resettable toggle |
| `nv_evo/oracle.py` | Reference-state-machine targets + stimulus generator + held-out generalisation scoring |
| `nv_evo/scoring.py` | **The scoring contract**: relation registry (match/rhythm families), event F1, interval, coverage and retention scorers, alignment discipline, shared score report |
| `nv_evo/temporal.py` | Temporal evaluation harness: trial running, trace-matched output placement, `prepare_net`; re-exports every scorer name for back-compat |
| `nv_evo/tritile.py` | The paper's three-circuit tile (Fig. 2): 12-bit tile states expanded onto the unchanged pulse engine via pre-resolved sources |
| `nv_evo/analog.py` | Analog Fig. 1 node (charge / leak / comparator / hysteresis) with emergent coincidence, width and refractory; event-driven with analytical crossings |
| `nv_evo/persistence.py` | Bounded-retention memory targets + staged event-native retention fitness (hold across long horizons, clear on command) |
| `nv_evo/certification.py` | The held-out verdict rule (CERTIFIED / OVERFIT / SOLVED / PLATEAU) shared by the GUI and `reproduce.py` |
| `nv_evo/diversity.py` | Solved-population diversity: the four-level collapse funnel (exact genotype → functional → phenotype → off-spec behaviour) plus mutational robustness. Fitness spread is zero once everyone solves, so variety has to be read off structure |
| `nv_evo/hexgrid.py` | Honeycomb geometry and facing-channel wiring |
| `nv_evo/ga.py` | Loop/memory-tuned GA (shaping, duplication, immigrants, caching, senescence cost, SOS hypermutation, annealing) |
| `evo_runtime/` | Backend-agnostic run machinery: run/GA config (incl. NV profiles), controller, checkpointing, mutation schedule, worker pool |
| `tools/make_scoring_golden.py` | Regenerates the scoring golden fixtures that pin the scoring contract (see tests below) |
| `tools/diversity_report.py` | Prints the diversity funnel (and optionally the robustness panel) for a saved solved population |
| `lut_evo/` | LUT-array backend (16-bit lookup cells, paper Architecture 2) |
| `lut_evo/boolfn.py` | Decode 16-bit LUTs to boolean logic (minimised SOP over N/S/E/W) |
| `lut_evo/ontogeny.py` | sim6 `table_create` morphogenesis — shape browser + the LUT GA's seed factory |
| `concept/` | Standalone proof-of-concept GA sims (hierarchical / coderack / multi-layer experiments) that predate the main backends |
| `ui_compat.py` | Cross-platform Tk helpers (monospace-font probing, ttk theming) so the GUI behaves the same on Windows / Linux / macOS |

## Usage

```bash
pip install -r requirements.txt

python app.py        # launch the GUI
```

In the GUI: pick a **Model** (SNN / Nervous / LUT) and a **Target** from the dropdown
(use the category filter or type part of a name to search; click **Custom…** to
enter your own truth table), set Population / Generations / Restarts plus the
GA tuning row — **Mutations/child**, **Anneal α** (α < 1 cools the mutation rate
each generation for a hot-start simulated-annealing schedule; α = 1 is off),
**Plateau β** (stagnation reheat strength; β = 0 disables it, β = 1 is default),
**Mutation cap** (the maximum effective mutations per child after the starting
rate, annealing, and plateau reheating are combined; default 8),
immigrants, tournament size and **Elites** (size of the elite *breeding pool* — the
top N genomes are the recombination parents for the next generation but are **not**
copied by the reproduction operator; after a terminal solve, evaluated parents may
survive environmental selection; 0 = breed from the whole population) — and click
**Run**. **Chroms** is the exact chromosome count for the run (1–32), is preserved by
mutation, and is stored with the checkpoint. The fitness chart plots all-time
best, the best newly generated offspring before survivor selection, population
mean, and effective mutation rate (secondary axis). Defaults are a single long run
(Gens 500, Tries 1) with a hot-start anneal (Mutations 4.0, α 0.997) so slow, steady
progress is visible. Growth is self-limiting via the telomere, so there are no
grid-size or iteration controls; **Max telomere** caps how large organisms may
grow (a chromosome's telomere is its growth *radius*, and eval cost ≈ radius²) —
LUT runs default to 8 while nervous-net runs retain 20; raise it when a task
genuinely needs a larger organism. Use **Load Saved** to reload `results/best_genome.json`
and **Save PNGs** to export the growth and voltage figures.

Tick **Graded** for harder targets (adders, large custom tables): instead of a binary
pass/fail per output it gives smooth partial credit, which keeps a usable fitness
gradient where binary scoring would otherwise flatline. A perfect circuit still scores
1.0, so it's safe to leave on.

> Note: large targets (full/multi-bit adders, big custom tables) have many truth-table
> rows and evolve slowly — the framework supports them, but small targets are best for
> interactive runs.

## Tests & reproducing the claims

The project's goal is a *defensible, reproducible* claim that the asynchronous
substrate can **evolve** useful async circuits — so the substrate's clock-freedom
and the headline results are both runnable, not asserted.

```bash
py tests/run_tests.py          # whole suite, no pytest needed (py -m pytest also works)
py reproduce.py                # list reproducible claims
py reproduce.py c_element      # evolve one claim from a fixed seed, certify on held-out
py reproduce.py async_substrate  # the metamorphic "no hidden clock" audit
```

- **`tests/`** — a metamorphic **synchrony audit** (`test_synchrony.py`) that
  drives the pulse engine through the audited float path and asserts the
  relations a *genuinely asynchronous*, continuous-time system must satisfy and a
  tick-clocked one cannot: a sub-tick input shift shifts every output by exactly
  that amount (**translation invariance**), scaling the schedule and the physical
  constants scales every output time (**scale covariance**), plus determinism,
  event-order independence and no-spontaneous-activity. `test_lut_synchrony.py`
  runs the same audit against the LUT array's asynchronous engine (translation /
  scale / determinism / order independence on a quiescent relay organism, plus
  inertial blip filtering and honest spontaneity) and pins the quantization
  contract: on the tick lattice the async engine equals `LutSim` bit for bit,
  fast path and event loop alike. `test_engine_semantics.py`
  pins the wired-OR / held-level / pulse-width semantics; `test_oracle_logic.py`
  pins the reference state machines that define the temporal targets;
  `test_tritile.py` proves the three-circuit tile's independent routing;
  `test_analog.py` + `test_analog_reference.py` validate the analog node's
  emergent behaviours against an independent numerical integrator; and
  `test_scoring_equivalence.py` replays a frozen golden battery (every
  registered target × synthetic output variants, plus retention scenarios)
  through the scoring contract and demands identical numbers to 1e-12 — any
  intentional scoring change must regenerate the goldens via
  `py tools/make_scoring_golden.py` and justify itself in the commit.
- **`reproduce.py`** — each *claim* is a seeded evolution plus a certification
  gate: for input-driven targets it re-samples fresh held-out schedules and
  reports **CERTIFIED** (generalises), **OVERFIT** (memorised timing), **SOLVED**,
  or **PLATEAU** (a documented substrate limit, e.g. the SR-latch clear, reported
  honestly rather than hidden).
- **Automatic certification** — every GUI/controller run of an oracle-backed
  temporal target certifies its winner the same way when it finishes: the verdict
  is shown in the status line and saved into the checkpoint
  (`nv_evo/certification.py`, one shared rule with `reproduce.py`), so a
  high training fitness is never trusted on its own.

## Requirements

- Python 3.10+
- numpy
- matplotlib
- pytest is **optional** — the suite also runs with bare Python via
  `py tests/run_tests.py`

## Concepts & provenance

The architectures and mechanisms trace to Andrew Edwards' *Evolvable Hardware* work on **grown**, indirectly-encoded circuits; the code cites specific figures inline where they apply:

- **Nervous net** — the honeycomb "nervous net" of Edwards **EH'02**: three independently configured directional circuits per tile (Fig. 2), the 16 coincidence/veto states per circuit (Fig. 3), and context rotation (Fig. 4). The temporal/oracle target suite is layered on that substrate.
- **LUT array** — the paper's **Architecture 2**, ported faithfully from the `sim6` reference implementation and the degree-4 case of `automaton_arrays` (4 neighbours × four 16-state lookup tables per cell; `table_create` morphogenesis in `lut_evo/ontogeny.py`).
- **Substrate framing** — identical tiled cells, routing SRAM / lookup tables as the reconfigurable element, and the genome as an associative memory (`context → new state`) all map onto the papers' hardware picture; the code comments name the corresponding hardware element at each point.

Two design threads are this project's own, layered on that base: **temporal / memory circuits** scored on spike-event timing (SR latches, oscillators, cadence steppers, gated memory) rather than only combinational truth tables, and **biologically-inspired search** (telomere/Hayflick-bounded growth, senescence parsimony, stress-induced hypermutation, annealed mutation). See the section comments in `nv_evo/ga.py` and `nv_evo/temporal.py` for the details behind each.
