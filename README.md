# Evolvable Hardware — Logic Circuits via Grown Neural Substrates

A research project implementing **Edwards indirect encoding** to evolve grown circuits, with four backends: spiking neural networks (SNN), hexagonal "nervous nets", the **Functional NV Net (FNV)**, and a square **LUT array** (the paper's Architecture 2 / `sim6`).

## What's new

- **Paper-faithful tri-circuit tile** (`substrates/nervous/tritile.py`): the nervous net can now run Edwards EH'02 Fig. 2 as written — **three independently configured circuits per tile**, one per L/R/D output direction (15-bit tile state: three 5-bit channels), instead of the legacy single broadcast circuit. One tile can route two signals to two outputs independently, which the single-circuit tile physically cannot.
- **Analog node physics** (`substrates/nervous/analog.py`): an event-driven model of the paper's Fig. 1 circuit — capacitive input steps, Vbp leak toward Vdd, comparator with hysteresis. Coincidence window, output pulse width, dense-input stretch (paralyzability) and refractory recovery all **emerge from the same physical constants** instead of being independent knobs; validated against an independent numerical integrator.
- **One current NV profile**: fresh nervous-net runs use **Analog tri-circuit** (the paper's three-output tile plus the paper-inspired analog node). The GUI keeps a single descriptive profile entry and exposes Vth, step, tau, and hysteresis in a dedicated row. The former **Legacy** single-tile and **Digital tri-circuit** configurations remain loadable reference engines for old checkpoints and controlled ablations, but `validate_new_nv_profile` rejects them for a fresh run.
- **Executable behavior contracts** (`substrates/nervous/contracts.py`, `substrates/nervous/scoring.py`): every target declares the idea it must implement as data—truth-table correspondence, one-to-one timed events, phase-invariant logical state, complete pulse intervals, or cadence restrictions. SNN, nervous, FNV, and LUT simulators only adapt their physical output into observations; all fitness enters the single `score_contract(...)` evaluator. The seven target-selected scoring modes and separate retention dispatches are gone. State scoring uses maximum silent gap over continuous intervals, so translating the phase of the same circulating memory ring cannot change its score. The app's Evolution panel now renders the executable Contract v1 data before a run and its scored observations afterwards, while Interactive playback keeps a compact contract badge visible. Semantic anti-cheat tests replace the old equivalence golden.
- **Case-aware truth-table search**: asynchronous combinational runs retain every
  row/output correctness check and automatically use exact lexicase selection;
  they do not ask one averaged percentage to choose parents. Continuous timing
  checks keep MAD ε, while exact Boolean 0/1 checks use ε=0. Multi-output
  aggregation blends the mean with the weakest output, and global output-probe
  fitting optimizes that same objective, so solving an easy output cannot hide
  a chance-level hard output or cause the fitter to install an inferior
  readout. A scalar remains for plotting, stopping at exactly 1.0, and temporal
  ranking, but combinational breeding is driven by the contract cases.
- **Contract specialists survive and recombine**: before a solve, a rotating
  40% environmental reserve carries distinct best-on-case behaviors across
  generations, ordered by the cases the population currently finds hardest.
  The remaining slots keep ordinary generational churn. Under lexicase,
  crossover chooses its second parent by how well the pair's case envelope
  fills the first parent's gaps, instead of making two unrelated draws that
  often specialize on the same rows. Both rules consume only the target's
  declared case vector and apply unchanged to logic, timing, memory, and
  cadence contracts.
- **FNV contract-class initialization**: when LOGIC is present in the selected
  component bank, exhaustive combinational runs seed their first population
  and immigrants from LOGIC plus available DELAY routing/fan-out components.
  Mutation still uses the complete
  selected bank, so delays, holds, normalizers and stateful components remain
  reachable; temporal runs and explicitly logic-free banks are unchanged.
- **FNV coherent multi-output selection**: the exact row/output cells remain
  independently selectable, while FNV also exposes per-output balanced
  accuracy, per-row joint correctness, and weakest-output accuracy to parent
  and survivor selection. Reported fitness and certification are unchanged.
  Crossover transplants complete labelled dependency cones instead of mixing
  the fields of unrelated components; parents with matching input geometry can
  combine whole multi-branch functions while local collisions displace only
  the overlapping dependency subtree.
- **FNV labelled construction genome**: fresh FNV runs no longer use the
  homogeneous-medium nearest-context classifier. Every placement explicitly
  names stable source/output-port labels, a fixed catalogue component, and a
  branch-block label. Dependencies determine placement rather than list order;
  a collision suppresses only the conflicting placement and leaves unrelated
  branches intact. Mutation can extend or join live tips, alter a compatible
  component, build short or long physical bridges, move/duplicate/delete a
  whole branch block, or move an input pad. Crossover remaps stable IDs and
  transplants a complete sink dependency cone when parent layouts match;
  otherwise it re-anchors one compatible branch without moving the recipient's
  co-adapted source pads. Fan-out
  remains physicalâ€”only source pads and real two-output components create
  multiple wires. Associative-v2 checkpoints still load under their original
  interpreter, while new runs use constructive v3.

## Background — grown, not designed

Conventional *evolvable hardware* evolves a circuit by encoding it **directly**: the genome is a blueprint, one gene per wire or gate. That doesn't scale — the genome grows with the circuit, identical sub-structures must be rediscovered independently, and the fitness landscape is jagged because one bit-flip rewires one gate.

This project takes the **indirect** route from Edwards' *Evolvable Hardware* work: the genome encodes **growth rules**, not a circuit. A circuit is *grown* by a developmental process (ontogeny) from a handful of seed cells, the way a body is grown from DNA. Every cell carries the same genome; each consults it as an **associative memory** — "given the state of my neighbourhood, what do I become?" — so position and local context decide a cell's fate. This buys three things the direct encoding lacks:

- **Compact, scale-free genomes** — a fixed rule set grows an arbitrarily large organism, and a rule that builds one motif builds it everywhere.
- **Evolvability** — small genome changes produce coherent, often-repeated structural changes (a rule tweak reshapes every copy of a motif at once), which smooths the search landscape.
- **Self-limiting size** — growth halts on its own, at a boundary the genome dictates (here, an evolvable telomere / Hayflick limit), rather than at an arbitrary wall.

The substrate is a regular lattice of reconfigurable cells. This repo implements four cell architectures — a spiking neuron, the two cellular-automaton architectures from the Edwards papers, and FNV's fixed-function component bank — and evolves both **combinational** logic (truth tables) and **temporal** circuits, where the answer is carried in the *timing* of events: memory that latches, oscillators, counters, delay lines and pattern generators.

## The four substrates at a glance

| | **SNN** (`substrates/snn/`) | **Nervous net** (`substrates/nervous/`) | **FNV** (`substrates/fnv/`) | **LUT array** (`substrates/lut/`) |
|---|---|---|---|---|
| Origin | LIF variant | Edwards EH'02 honeycomb | standalone fixed-function extension | paper Architecture 2 / `sim6` |
| Lattice | square grid | honeycomb, **degree 3** | honeycomb, **degree 3**, antiparallel wires | square grid, **degree 4** |
| Cell function | leaky integrate-and-fire neuron | `(E1 op E2) AND NOT I1` routing node | one of 118 permanent routed component types | four directional 16-bit tables, optionally restricted to fixed gate banks |
| Dynamics | **event-driven** continuous LIF | **asynchronous** edge-triggered pulses | **event-driven** fixed-function logic/state | **asynchronous** level logic |
| Spontaneous activity? | **no** | **no** | **no** — every type is quiescent | arbitrary LUTs: **yes**; named gate banks: **no** |
| Computes | combinational **+ temporal** | combinational **+ temporal** | combinational **+ temporal** | combinational **+ temporal** |

## What it does

A genetic algorithm evolves a physical circuit. Nervous, SNN, and LUT use
backend-specific developmental rules. FNV is the fixed-function control arm:
its labelled dependency genome assembles routed components outward from evolved
source pads. Four separate backends interpret and evaluate the resulting grid:

- **SNN** (`substrates/snn/`): a square grid of leaky integrate-and-fire (LIF) neurons. Static truth-table runs retain the original feed-forward interpretation; temporal runs opt into reciprocal physical adjacencies (`Arch.recurrent=True`) and drive exact continuous-time source edges through the same executable behavior contracts as the other backends. Contract seconds are explicitly converted to the LIF model's millisecond units (4.8 ms per contract second), so requested periods are compared to physically achievable feedback cycles rather than to a mismatched clock. That permits evolved oscillators, latches and counters without changing old combinational checkpoints. A deterministic event cap rejects runaway loops, and a constructive two-neuron feedback witness sustains activity after the input disappears and scores 1.000 on the real period-2 oscillator contract. Growth is **telomere-bounded** like the nervous net: each chromosome carries an evolvable Hayflick division limit, so a lineage stops dividing once its telomere is spent and the body self-limits at radius L from the seeds (replacing the old fixed iteration cap), and the GA's senescence/parsimony tie-break then shrinks that body — fewer genes, shorter telomere — toward the smallest circuit that still solves. (A generous `GRID_SIZE` remains as an outer wall so growth coordinates stay valid for the growth view and fixed I/O terminals; the telomere is the real limiter.) Its interactive view is a live LIF animation — see `ui/interactive.py`.
- **Nervous net** (`substrates/nervous/`): a honeycomb array of nodes that fire when `(E1 op E2) AND NOT I1` — coincidence/veto detection over the paper's 16 routing states of Edwards EH'02 Fig. 3 (states 0–15: each one a buffer or an AND, optionally vetoed — the paper itself has **no disjunction**), plus 16 **OR twins** (states 16–31: the same wiring but `op = OR`, so the node fires on *either* excitatory input) added as a deliberate extension beyond the paper. The extra state bit is 0 for every legacy genome, so 0–15 grow and behave bit-identically; only the OR of two *different* lines is genuinely new (for a buffer or off state, OR and AND coincide, so those twins are benign aliases). Directions are read in each node's own orientation frame (the paper's context rotation — down-parity tiles are the 180°-rotated circuit), and inputs are **wired-OR injections** onto the input cell's net, like pulsing the physical perimeter wire — not level clamps. So **with no input the array is inert**, yet a pulse injected into a loop of buffers routed through an input cell circulates as delay-line memory "until stopped by application of an inhibitory input" (§3). Besides combinational targets it evolves **temporal circuits** — SR latches, toggle flip-flops, oscillators, delay lines, plus async-native ones where edge timing *is* the computation: a two-input coincidence detector, a one-shot/monostable (self-terminating burst via delayed self-inhibition), and a double-pulse pair detector (delay line + coincidence). Dynamics are the paper's **asynchronous edge-triggered pulses** (`substrates/nervous/pulse.py`): an event-driven simulation where a triggered node emits a fixed-width pulse after a fixed delay, coincidence requires both excitatory edges within a window, and inhibition vetoes at trigger time. Every wire's raw continuous leading-edge timestamps are retained for behavioral scoring; once-per-tick samples exist only for playback and legacy persistence windows, so sub-tick pulses are no longer discarded. Every element (identical tiled nodes, 5-bit routing SRAM — 4 bits in the paper, the extra bit selecting the AND/OR variant — genome as associative memory / CAM, delay / pulse-width / coincidence-window constants) maps directly onto the paper's hardware.

  New nervous genomes use one native I/O architecture. They evolve one
  collision-free relative honeycomb coordinate per logical input; input 0 is
  anchored at `(0, 0)` only to remove whole-body translation, and every other
  pad moves by one-edge mutations. These pads seed development and are
  source-only at runtime, so feedback cannot reactivate them. Every mature
  non-input tile is eligible as an output probe, and multi-output roles are
  fitted together as a distinct-cell assignment. The evolved pads and fitted
  probes are frozen for held-out certification. Legacy nervous checkpoints
  without `input_layout` keep their target-declared pads.

- **Functional NV Net / FNV** (`substrates/fnv/`): a separate degree-3 honeycomb substrate whose cells are fixed physical functions rather than instances of the original NV pulse node. Every edge contains two independent antiparallel wires. A two-input component consumes two local directions and drives the third; a unary component has either one output direction or two identical output directions. The append-only catalogue contains 118 permanent type IDs: `EMPTY`; AND/OR/XOR and asymmetric VETO routes; fixed 1/2-tick delays; `NORMALIZER1/2`; `HOLD1/2`; Muller C-elements; rising-edge toggles; and enable-gated oscillators with fixed high/low times 1 or 2. Wider functions such as parity-3 and majority-3 must be assembled from those primitives. There is no free oscillator and no evolvable per-node parameter. Runs select whole component families only. Fresh genomes construct circuits from stable labelled physical branches instead of applying Edwards associative rules to a heterogeneous medium. Simulation is continuous-time and event-driven; all-zero power-on is a catalogue-wide invariant. Checkpoints store a SHA-256 catalogue hash so a type-ID remap cannot silently reinterpret saved hardware.

  The FNV control row includes a **Node number dictionary** that decodes every
  number drawn inside a component into its permanent name, family, routed
  inputs/outputs, and fixed timing. Code can use the catalogue-derived
  `NODE_TYPE_DICTIONARY` mapping for the same ID-to-name lookup.

  Each placement gene names a permanent ID, fixed component type, stable input
  port references, and branch-block ID. Its referenced ports must all face the
  same empty honeycomb site. Genes resolve by dependency, not list position. If
  two placements claim one cell, the earlier ready stable ID wins; only the
  colliding placement and its dependent descendants become dormant. Other
  branches still assemble. There is no target-shaped grid, desired-output
  query, automatic route planner, or phenotype-to-genome translation. New FNV
  genomes also evolve the source pads' **relative honeycomb positions**: input 0 is
  fixed at `(0, 0)` only to remove meaningless whole-body translation, while
  every other pad moves by local one-edge mutations and the whole layout is
  inherited as one collision-free physical module. Named dependencies make a
  pad move translate only branches rooted in that pad. Initial layouts are compact
  so random populations do not begin as disconnected islands. These pads are
  source-only during operation: neighbouring logic cannot reactivate them; only
  the target's external input pulse can.

  Mutation extends or joins live physical tips, changes compatible fixed
  components, constructs explicit DELAY-plus-LOGIC bridges across the empty
  frontier, and reroutes, duplicates, or deletes complete connected branch
  blocks. Plateau rescue samples bridge lengths across the available range and
  may offer a bounded two-bridge cascade, so a neutral first route can be
  extended without reading target answers. Transplantation remaps stable IDs;
  matching-layout crossover inherits a complete dependency cone, with
  different components at overlapping cells displaced locally. Abstract fan-out
  does not exist: multiple branches require a source pad or a fixed component
  with two output ports.

  Outputs are fitted probes, not extra component parameters. Every mature
  non-input component is eligible, regardless of distance from the target's
  drawn terminal, and multi-output targets use a global distinct-cell
  assignment that maximizes total contract score instead of greedily choosing
  one role at a time. The fitted pads and probes are then frozen for held-out
  certification. Old associative-v2 FNV checkpoints retain their original
  interpreter and target-declared fixed pads when no input layout exists.
  Reproducible small-budget comparisons live in
  `tools/benchmark_fnv.py`, including an all-temporal survey and selectable
  tournament/lexicase runs.

  FNV deliberately has **no small-genome or short-telomere preference**.
  Behavioral correctness dominates selection; FNV currently leaves the
  optional robustness and juvenile-development tiers inactive. Exact fitness
  ties are broken by a target-agnostic physical topology potential. The
  mature directed wiring graph is traced outward from the source pads and
  ranks uncapped counts of distinct input convergence, real multi-input
  junctions, reachable feedback, reachable edges, and reachable nodes. This
  deliberately supplies no small-body preference: longer connected routes can
  survive as neutral stepping stones, while disconnected bulk and unreachable
  loops score nothing. It retains computational stepping stones without recognizing any target-specific
  Boolean or temporal behavior. Static reports separately show the circuit's
  nonconstant truth-signature repertoire as diagnostic telemetry; it never
  enters selection.

- **LUT array** (`substrates/lut/`): the paper's **Architecture 2**, a faithful port of the `sim6` reference — a **square** grid where each cell wires to **4 neighbours** (N/S/E/W) and holds **four** directional 16-bit lookup tables (one per output; the `automaton_arrays` design specifies four lookup tables per cell, each with 16 states). Growth looks each direction's LUT up with the neighbour context **rotated** so the output direction is "front" (context rotation); dynamics are **asynchronous level logic** (`substrates/lut/pulse.py`): each cell is a logic element with a fixed propagation delay that re-indexes its four LUTs with the 4 bits its neighbours aim back at it whenever an input wire changes (inertial delay — sub-delay blips are filtered), simulated event-driven in continuous time. With delay = 1 tick and lattice stimuli this reproduces the `sim6` synchronous latched update bit for bit (the retained `lut.LutSim` is the audited quantization reference). The historical arbitrary-table mode can invert and therefore permits spontaneous activity (faithful to the paper's Fig. 14); the selectable named gate banks below are quiescent at all-zero input. Runs the same temporal and combinational targets.

  A LUT run can select whole **function banks**: `ROUTING`, `AND`, `OR`,
  `XOR`, asymmetric `VETO`, `THRESHOLD`, `MUX`, and `UNRESTRICTED`
  (“Arbitrary LUT” in the GUI). These are permanent physical 16-bit truth
  tables, not evolvable parameters: the named catalogue contains all routed
  2/3/4-input AND/OR/XOR variants, all directed vetoes, majority/threshold
  variants, and all directional 2:1 mux variants. `OFF` is always available.
  The selection constrains only executable directional outputs and the
  developmental germline; the five 16-bit CAM recognition fields remain fully
  expressive because they match developmental context rather than execute as
  runtime gates. Initialization selects a family before a table, and mutation
  usually moves to the nearest table within the current family, so larger
  families do not dominate merely by containing more variants. Arbitrary LUT
  alone is the default and preserves the previous all-65,536-table search
  exactly; old checkpoints without the setting load in that mode.

  LUT runs offer two physical input architectures. **Internal source pads**
  evolve one distinct square-lattice coordinate per logical input; the pads are
  developmental germlines and source-only runtime terminals, matching NV/FNV.
  **Exterior perimeter buses** instead grow from one neutral germline and use
  every exposed outer face. Faces in stable cyclic order are assigned round
  robin to logical inputs: one input owns the whole perimeter, two inputs form
  A/B/A/B around it, and larger input sets repeat A/B/C/… in the same way. A
  logical signal is fanned simultaneously to all taps of its bus. Every tap
  remains outside the organism and feeds only the facing N/S/E/W LUT input, so
  feedback can neither consume nor reactivate it; faces around enclosed holes
  are excluded. This bus geometry is fixed rather than genetic, removing port
  placement from the search while keeping the developed body evolvable.
  Both modes use the same globally fitted, distinct, read-only output probes.
  Checkpoints record the selected mode; obsolete point-layout fields from older
  exterior checkpoints are preserved on round-trip but ignored. Training and
  Interactive playback support both modes; the generic held-out
  certification adapter has not yet been specialized for exterior
  outside-to-facing-edge links, so exterior certification verdicts are not yet
  an audited claim.

  (The **hex** nervous array is the degree-3 case from the same `automaton_arrays` design — 3 neighbours and 3 LUTs of 8 states — drawn as a brick-wall; deliberately *not* the degree-6 triangular reading.)

- **Nervous-net substrate profile** (the **NV profile** display in the pulse-physics row; `runtime/config.py: NV_NEW_RUN_PROFILES`): a fresh nervous-net run uses **Analog tri-circuit** (`tri3` + `paper_analog`) — the paper's Fig. 2 tile with three independent 5-bit L/R/D channels (15 bits total) and the Fig. 1 analog node (`substrates/nervous/analog.py`). Capacitive down-steps, Vbp leak, and comparator hysteresis make coincidence, output width, and refractory **emergent**, so the Coinc control is disabled and Vth / step / tau / hysteresis have their own GUI row.
  The former width-preserving single-tile engine (`single` + `pulse_delay`) and digital tri-circuit ablation (`tri3` + `uniform`) remain solely for old checkpoints, tests, and controlled comparisons; `validate_new_nv_profile` rejects them for fresh runs. The digital tri engine now understands the same 15-bit, OR-capable channel representation, while checkpoint migration widens its older 12-bit AND-only states exactly once. A further variant made emitted width a per-node-type genome vector; **width evolution has been removed from the substrate** — the genome no longer carries a width vector, and a checkpoint saved under it loads on the paper's fixed-width node instead. Timing models are audited in `tests/test_pulse_models.py`, the tri tile in `tests/test_tritile.py`, and the analog node against an independent Δt integrator in `tests/test_analog_reference.py`. Width-sensitive temporal targets now exercise useful relations rather than isolated node contracts: **Pulse width sum (A+B)** emits a Q interval whose duration is `width(A)+width(B)`, while **Odd pulse selector** passes the 1st/3rd/5th input pulses and preserves their individual widths. Both score complete output waveforms (`PulseSim.pulse_intervals`, real rises *and* falls), rather than treating pulses with equal leading edges as equivalent. A related mixed-width stateful family counts every physical pulse once regardless of its 0.5-to-2.25-second duration: **A-count parity queried by B** retains cumulative odd/even parity, **A-count multiple-of-3 queried by B** recognizes positive multiples of three, and **Odd A batch closed by B** reports parity since the preceding B and then clears it.

The SNN and nervous backends share no code; FNV is independent of the nervous
runtime and shares only target contracts plus honeycomb geometry, not node
physics or genome types. Growth in the nervous and FNV models happens on an
**unbounded field** and runs until it converges to a stable attractor. Size is
bounded genetically rather than by a wall. Seed cells start with the genome's
telomere length *L*; a daughter inherits one less, so the organism halts its own
growth at radius *L*. Maintenance rules continue to act on existing cells:
telomeres limit replication, not function.

The GA never stops early at fitness 1.0. SNN applies a solved-only
senescence/parsimony tie-break. Nervous and FNV deliberately do **not** prefer
small genomes: their final tie-break favors input-reachable connectivity and
feedback, while LUT has no size tie-break. Before a solution exists, most
slots remain generational, while a bounded rotating reserve retains distinct
best-on-case behaviors so a missing truth-table row or trial cannot be forgotten.
After a solution appears, evaluated parent-plus-offspring survivor selection
retains it while exploratory offspring continue to be tested.

## How it works

### Compatibility spatial-I/O compiler for hard LUT targets

This subsection describes the retained programmatic
`spatial_chromosome` experiment, not the current app's native internal-pad or
exterior-edge modes.

For event-driven truth tables, silence alone cannot identify an all-zero row
that must emit a 1 (decoder D0, comparator EQ, NAND/NOR/XNOR). The periodic
wrapper adds an explicit case-valid strobe only for those tables; zero-row-low
targets retain their historical port count and schedule.

When such a spatial-I/O LUT run plateaus, `substrates/lut/synthesis.py` can compile up to
four data inputs and four outputs into a directional crossbar; the five-port
comparator uses a two-stage strobe/zero-detector cascade. The phenotype is
inverse-developed into a heritable genome, re-grown exactly, and scored by the
unchanged behavior contract. Verified witnesses reach 1.000 on Full adder,
2-bit adder, MUX, Majority-3, Parity-3, decoder, comparator, and the 2x2
multiplier within the default eight-step growth bound.

- **Indirect encoding**: the genome is an associative memory of rules mapping neighbourhood context → output state, looked up by minimum Hamming distance. A nervous-net rule contains five state-valued fields (`L context`, `R context`, `D context`, `self in`, `self out`). They are 5-bit values (0–31) for the retired single-circuit architecture and 15-bit values (three independently packed 5-bit channels) for `tri3`.
- **Context rotation** (nervous net): one scalar gene is evaluated independently for each of a tile's L/R/D core circuits. Its physical neighbour context is rotated so the circuit being updated is "front", letting the same rule grow symmetric structure ("right/left/down rotated to match topology", EH'02 Fig. 4).
- **Hierarchical crossover**: multi-gene chromosomes exchange reciprocal suffixes at a real interior boundary; a one-gene chromosome exchanges a proper subset of that rule's active fields, so minimal genomes can still produce recombinant children. Parent choice prefers different rule content, and multi-edit mutation transactions finish with a protected non-parent allele so inverse edits cannot quietly cancel back to a clone
- **Circuit ontogeny**: growth iterations expand the selected developmental germlines into a full circuit
- **LIF simulation** (SNN): neurons communicate via synaptic currents; static output is read as spike/no-spike, while temporal runs retain continuous event times and expose unit-width spike intervals to the shared contract scorer
- **Per-output encoding** (SNN): experimental targets may use complemented inputs and/or an inverted spike reading, but all bundled arithmetic carries are direct: input/output high means logic 1
- **Pluggable targets**: a `Target` (`substrates/snn/targets.py`) is the single source of truth for logical input count, output roles, truth table, and display/legacy coordinates. Current Nervous, FNV, and LUT runs resolve physical inputs from the genome and fit physical output probes from behavior.
- **Combinational target library** (`substrates/snn/targets.py`, run on **all four** models): logic gates, half/full/2-bit adders, plus the harder **2:1 MUX**, **Majority-3**, **Parity-3 (XOR3)**, **2-to-4 decoder**, **2-bit comparator** (GT/EQ/LT) and **2×2 multiplier** (4-bit product) — data routing, voting, wide parity, relational logic and arithmetic. Nervous and LUT encode every row as its own widely-spaced pulse window via `periodic_combinational_target(...)`. FNV instead uses its native level-gate semantics: each exhaustive row holds its input levels, resets the circuit, waits through a genotype-derived settling horizon, and reads the final level at globally fitted probes. This prevents harmless settling glitches from being scored as permanent logical highs while keeping every row/output as a separate exact lexicase case.
- **Temporal target library** (`substrates/nervous/targets.py`, run on **SNN + Nervous + FNV + LUT** except where target metadata records a physical exclusion): eight hand-built banks cover oscillator, `Pattern (1000)`, coincidence, Temporal XOR, Sequence A→B, Veto, Burst ×3, and Divide-by-3. The registered oracle-backed set adds pulse-width sum, odd-pulse selection, three A-count query functions, SR latch, C-element, refractory filtering, A-first rendezvous, collision serialization, watchdog timeout, toggle, echo, a 12-second non-retriggerable one-shot, period doubling/tripling/halving, temporal interval sum, two pair detectors, period stepper, gated oscillator, and resettable toggle. Each target declares its own backend/model exclusions; unsupported combinations are hidden rather than scored against impossible physics. FNV feeds its physical events and intervals into these same contracts.
- **Interval transformation targets**: **Period tripler (3×)** emits every third input edge, **Period halver (½×)** first measures a complete even input period and then inserts midpoint events, and **Temporal sum (ΔA + ΔB)** measures two intervals on each of two inputs and encodes their sum as the interval between two Q events. Their seeded banks mix periods/intervals and include silent, one-lane, and incomplete guards, preventing direct wires, fixed bursts, and free-running oscillators from solving them.
- **Physical pair target** (Nervous + LUT — the asynchronous backends): **Pair detection gap (2x pulse width)** supplies explicit fractional-phase input pulses and emits Q only when consecutive leading edges are separated by exactly twice the supplied input-pulse width; wrong relative gaps, chains, isolated pulses, and silence are included.
- **Temporal time units**: target names, descriptions, reports, and plots use seconds. Nervous events may occur at fractional seconds; one LUT gate delay is defined as one second.
- **Temporal targets** (`substrates/nervous/targets.py`): scored on raw events, cadence invariants, or persistence traces as the behavior requires, across several shifted trials — so only genuine timing/memory passes, not a lucky delay chain
- **Oracle targets** (`substrates/nervous/oracle.py`): rather than hand-picking input timings (which can be adversarial to a circuit's internal phase), a goal is specified as a *reference state machine* `oracle(inputs, state) -> (outputs, state)` plus a *stimulus generator*. Seeded schedules are labelled by the oracle; the circuit is scored on reproducing the input→output *relation*, and `holdout_score` re-samples fresh schedules to certify that it generalises rather than memorising timings. Exact-delay and interval targets disable free alignment where absolute timing is the behavior. Stateful banks include boundary, silence, re-arm, repeated-command, and long-horizon cases; for example, the period stepper must sustain cadences 2 → 4 → 6, and the watchdog mixes safe, exact-deadline, late, re-arm, and never-armed trials.
- **Fitness for temporal targets** (`substrates/nervous/contracts.py`, `substrates/nervous/scoring.py`; `substrates/nervous/temporal.py` only runs trials and collects observations): the question is *"did the network implement the declared idea?"*, not *"did it copy an arbitrary trace?"*. A target owns a serializable `BehaviorContract` containing one or more reusable restrictions, and every backend passes normalized observations to `score_contract(...)` —
  - *One-to-one timed events* (`event_correspondence`) pairs expected and produced continuous-time leading edges. Missing and spurious edges both cost. A target may fit one latency shared across every trial/output or require absolute timing, as Echo delay 3 does.
  - *Sustained and commanded cadence* (`sustained_cadence`, `commanded_cadence`) measures regular rhythm, event count, dwell coverage, silence outside active epochs, and command-induced rate changes without requiring an arbitrary absolute phase.
  - *Phase-invariant logical state* (`logical_state`) compiles the abstract expected state into active and quiet epochs. A nervous stored 1 may be a circulating pulse; the scorer measures its regular lap across commanded-active epochs (never below the smallest legal ring), uses that same lap for retention and transition timing, and grants one-lap stopping grace for the pulse already in flight when a reset arrives. Shifting an identical ring in phase or building a larger valid loop therefore cannot reduce fitness.
  - *Complete pulse intervals* (`pulse_intervals`) checks both rise and fall, so the right edge times with the wrong pulse widths cannot pass.
  - *Bounded persistent state* (`bounded_state`) expresses long-horizon hold, quiet, clear, reset influence, and reload cases through the same entry point used by ordinary targets.
  - A deterministic per-run event cap rejects pathological oscillators before one genome monopolises evaluation. Event/cadence evaluation schedules input edges directly, skips `T × cells` display snapshots, reuses one input cone across every trial, and uses allocation-free sparse matching; sampled states are built only for playback and persistence targets.
  - *Trace-matched output placement*: Nervous, FNV, and LUT fit each output role over the whole mature organism. Multi-output roles are assigned globally to distinct cells, so an early role cannot greedily consume the only strong probe for a later one. SNN retains its own backend-specific readout placement.
  Multiple restrictions aggregate as half weighted mean plus half worst restriction: useful partial progress remains visible, but an easy restriction cannot hide a failed one, and fitness 1.0 requires every restriction to pass.
- **Semantic period-stepper contract** (`commanded_cadence`): some behaviours have no single "correct" trace to match. A cadence controller must hold a regular output pulse rate and make it slower after each command. Every command-delimited dwell is checked for sustained regular cadence, and later dwells must have a strictly longer period. A fixed-rate oscillator scores **0** on the change term regardless of raw trace overlap.
- **Describe a target as spike events** (`substrates.nervous.spike_target`): define a new temporal function directly from test cases — `spike_target(name, cases, T, n_inputs=…)` where each case is `(input_spikes, output_spikes)` (input ticks per input, expected output ticks). Pair it with `substrates.nervous.ga.diversify` to turn one solution into a whole generation of genotypically-unique valid solvers.
- **Loop-aware GA** (`substrates/nervous/ga.py`, `substrates/topology.py`): a target-blind final ranking tier rewards directed hardware the source pads can actually reach — nodes, wires, multi-input convergence, cyclic nodes, independent loop rank, and distinct feedback regions; disconnected bulk and unreachable rings receive none. Nervous aggregates those counts with diminishing-return `log1p` credit. FNV keeps its separate physical extractor and uses the uncapped lexicographic construction potential described above. Behavioral fitness—and optional robustness/juvenile objectives where the backend implements them—always ranks above topology. Nervous also retains its older temporal-only loop shaping: below perfection, a writable/readable feedback loop can add at most 5% of the remaining score; periodic combinational wrappers do not receive that bonus.
- **Search dynamics** (`substrates/nervous/ga.py`, `substrates/lut/ga.py`, `substrates/fnv/ga.py`): **stress-induced hypermutation** raises the mutation rate after 12 genuinely flat generations and relaxes on either scalar improvement or leximin progress in one organism's declared case vector. This avoids calling improved weakest-case coverage a stall merely because the plotted aggregate is unchanged. **Simulated-annealing mutation decay** multiplies the base rate by editable *Anneal α*. **Plateau β** controls the reheat slope: 0 disables it, 1 preserves the tuned behavior, and larger values raise mutation faster. SNN alone uses a solved-only senescence/parsimony tie-break; Nervous/FNV use topology and LUT intentionally has no small-genome preference.
- **Evaluation performance**: GUI/controller runs use an explicit **Workers** limit (default `max(1, min(cores - 2, 8))`, allowed 1-16), reuse one persistent worker pool across generations, deduplicate identical genomes before submission, and keep a bounded fitness cache. Stop cancels queued work and drains running workers before another run can begin. Hot paths compile developmental lookups once: Nervous/SNN pack a complete context into one XOR + `bit_count`, constructive FNV resolves named dependencies once (legacy associative-v2 uses its categorical-distance matrix), and LUT retains its vectorized table engine. FNV also compiles phenotype wiring once and reuses it across output fitting, trials, cases, and topology. Acyclic stateless FNV truth tables use an exact bit-parallel settled evaluator; cycles and stateful components fall back to continuous-time events, and certification always replays physical dynamics. Ordinary Nervous evaluation grows once for both behavior and topology; LUT resets one compiled simulator between timing replicates and vectorizes steady-duty extraction; fitness-only SNN runs skip voltage-history recording. Nervous target-only expected windows are cached during global probe fitting, and an event overflow stops the remaining trials immediately because overflow already guarantees zero fitness. Structural cloning avoids recursive `deepcopy` while preserving each backend's mutation semantics. Representative local microbenchmarks for the earlier optimization pass improved FNV 2-bit evaluation by about 47%, FNV reproduction by 40%, Nervous selection evaluation by 52%, SNN evaluation by 18%, and LUT evaluation by 12%; these are implementation checks, not universal throughput guarantees.
- **Nervous sampled-state fast path**: persistence targets still receive exactly the same half-tick logical samples, but fitness no longer constructs `T × cells` full-grid dictionaries. It reconstructs only candidate/output traces from the physical pulse intervals already emitted by the engine; equivalence tests cover uniform, paper-analog, and tri-circuit physics.
- **Substrate topology** (both original lattices): the hex (degree-3) and square (degree-4) grids are **bipartite** — 2-colourable by `(x+y)` parity, hex girth 6 — so a single circulating pulse can only traverse an **even-length** loop, and its output period is therefore even. Odd output periods are *not* impossible, but they cost far more: a period-*p* output needs a length-*2p* loop carrying two evenly-spaced pulses (`output_period = loop_length / n_pulses`), a conjunction the GA path dips through lower fitness to reach and empirically never crosses. This is a real design constraint, not a bug: a period-2 target (toggle) solves trivially, and the **pattern generator** is deliberately set to an even period — `Pattern (1000)`, period 4, one pulse in a length-4 loop — which the cheap single-pulse route reaches directly. The earlier odd-period `Pattern (100)` was retired precisely because parity made it topologically out of reach (neither a bigger grid nor hundreds of generations moved it off ~0.76). Autonomous targets that must run on **both** original lattices are chosen with this in mind. FNV keeps the spatially bipartite honeycomb, but its separate one-/two-tick delays, holds, toggles, and gated oscillators add temporal state at vertices, so an even spatial loop is no longer restricted to one tick per edge; this is how FNV escapes the original timing bipartiteness without inventing nonphysical connections.
- **LUT logic view** (`substrates/lut/boolfn.py`): every 16-bit lookup table is really a boolean function of the four neighbour input bits, so it is decoded to a minimised sum-of-products expression `out = f(N,S,E,W)` (verified exact over all 65 536 tables) — the genome reads as logic instead of hex, and the Growth tab shows the mature organism's distinct tables as actual 4×4 truth grids
- **sim6 ontogeny** (`substrates/lut/ontogeny.py`): a faithful port of the reference `table_create` morphogenesis that *grows* a dense genome on the fly (inventing a gene per unseen context), reproducing sim6's varied biomorphs. Runs as a standalone shape-browser (`py -m substrates.lut.ontogeny`) and is **the** LUT seed path: every LUT population/immigrant genome (evolver and Designer alike) comes from `substrates.lut.ga.make_seed_genome` — a dense biomorph seed drawn from a cached pool so immigrants stay cheap — and the LUT GA drops the smaller-genome parsimony tie-break, which used to prune the rich shapes back to sparse uniform diamonds. (Diagnostic finding behind this: the "uninteresting" diamond growth was genome *sparsity*, not the growth engine.)

### Plateau rescue for difficult targets

After 12 flat generations, nervous and LUT runs keep producing mutated
descendants of the separately archived all-time champion. Independently, the
baseline contract-elite reserve keeps a bounded set of complementary
best-on-case behaviors available to breed.
Spatial-I/O LUT runs also propose coordinated compact port motifs and nearby
one-bit rules expressed at output cells. Every proposal is an ordinary
heritable genome, grown normally and scored by the unchanged Behavior Contract;
the rescue never inserts a hand-built phenotype or grants privileged fitness.
The hard-target compiler rescue is available only when `UNRESTRICTED` is in the
selected LUT inventory, because its synthesized truth tables are not guaranteed
to belong to a named physical bank.
This makes a plateau diagnosable: a rescued genome demonstrates search
difficulty, while failure of a separately hand-designed phenotype still points
to scoring or substrate representation.

### Current and compatibility I/O modes

The main app exposes only each substrate's current physical I/O:

- **Nervous and FNV** — evolved relative internal source pads plus globally
  fitted, distinct, read-only output probes. There is no placement dropdown
  choice for fresh runs.
- **LUT** — either evolved internal source pads or fixed alternating exterior
  perimeter buses, with the same global output fitter in both modes.
- **SNN** — the original fixed binding plus the retained `tag_rank`,
  `wiring_chromosome`, and `spatial_chromosome` experimental strategies.

The shared `io_placement` machinery remains for old checkpoints and controlled
programmatic comparisons. `spatial_chromosome` is still causal for SNN and
programmatic LUT runs: chromosome 3 encodes input germlines and target-blind
output anchors. Programmatic LUT runs may also use `terminal_nodes`, whose
heritable `io_kind` alleles create source-only inputs and sink-only outputs.
Fresh Nervous and FNV runs reject all of these older placement strategies
because their native pad/probe mechanism replaces them; SNN rejects directional
`terminal_nodes`. Fixed-input legacy Nervous documents still load unchanged,
but Nervous checkpoints that actually encoded one of the retired placement
strategies are rejected because converting their tags or anchors into native
pad coordinates would invent a different organism.

## Files

**Entry points**

| File | Description |
|------|-------------|
| `ui/app.py` | Single-window GUI: pick a model + target, inspect its executable Behavior Contract, evolve, inspect scored observations / growth / activity / genome, and drive circuits interactively |
| `ui/designer.py` | Manual circuit **designer** — build a circuit by hand at either encoding level: edit the genome (chromosomes/genes) and press Grow, or edit the grown grid directly (place cells, set routing states). Simulate and score it against any target; available as a GUI tab and standalone |
| `ui/diversity_ui.py` | "Diversity" tab — analyse a complete evaluated population, solved or failed: the four-level collapse funnel, fitness summary, and optional mutational-robustness histogram. Runs on a worker thread with progress and Stop; nervous/LUT only |
| `ui/interactive.py` | "Interactive / Test" tab — after evolving/loading a circuit, drive its inputs and watch the response play out. Temporal SNN, nervous-net, FNV, and LUT runs load the exact scored trial into an editable continuous-time pulse timeline; FNV displays its real fixed-function components and directed wires, and combinational FNV cases use the same genotype-derived settling window as fitness. SNN shows recurrent LIF voltage/spike playback in contract seconds, while static SNN truth tables retain input toggles |
| `substrates/fnv/playback.py` | Scorer-faithful FNV continuous-time player and preparation adapter used by Interactive; wraps the same `FunctionalSim`, evolved source pads, fitted probes, and temporal/logic observation horizons as FNV fitness |
| `substrates/nervous/playback.py` | Shared continuous-time nervous-net player and pulse-lane editor used by Interactive and Designer; wraps the same paper-faithful `PulseSim` used by Nervous evolution |
| `ui/concept_gui.py` | GUI playground for the proof-of-concept GAs in `experiments/concept/`, with live matplotlib fitness / best-organism visuals |

**Backends**

| File | Description |
|------|-------------|
| `substrates/snn/` | SNN backend (genome, growth, LIF sim, fitness, targets, GA) |
| `substrates/snn/targets.py` | Combinational target registry + builders (gates, adders, MUX, majority, parity, decoder, comparator, multiplier, custom truth tables) |
| `substrates/nervous/` | Nervous-net backend (hex genome, honeycomb growth, pulse dynamics, targets, GA) |
| `substrates/fnv/` | Functional NV Net backend (fixed component catalogue, directed-wire dynamics, indirect growth, GA) |
| `substrates/nervous/targets.py` | Temporal target registry — eight hand-built timing banks plus the oracle-backed memory, interval, cadence, handshake, filtering, serialization, watchdog, and pulse-width families from `substrates/nervous/oracle.py` |
| `substrates/nervous/oracle.py` | Reference-state-machine targets + stimulus generator + held-out generalisation scoring |
| `substrates/nervous/contracts.py` | Serializable behavior-contract and constraint data plus concise target-definition helpers |
| `substrates/nervous/scoring.py` | The single `score_contract` evaluator: truth tables, event correspondence, phase-invariant state, bounded retention, complete intervals, cadence, alignment, and reports |
| `substrates/nervous/temporal.py` | Temporal observation harness: trial running, contract-scored output placement, and `prepare_net` |
| `substrates/nervous/tritile.py` | The paper's three-circuit tile (Fig. 2): 15-bit tile states (three 5-bit AND/OR-capable channels) expanded onto the unchanged pulse engine via pre-resolved sources; legacy 12-bit states migrate once |
| `substrates/nervous/analog.py` | Analog Fig. 1 node (charge / leak / comparator / hysteresis) with emergent coincidence, width and refractory; event-driven with analytical crossings |
| `substrates/nervous/certification.py` | The held-out verdict rule (CERTIFIED / OVERFIT / BELOW THRESHOLD / SOLVED / PLATEAU / UNCERTIFIED) shared by the GUI and `reproduce.py` |
| `substrates/nervous/diversity.py` | Population diversity: the four-level collapse funnel (exact genotype → functional → phenotype → off-spec behaviour) plus mutational robustness. It works on unsuccessful populations too; once everyone solves, structural variety replaces the now-zero fitness spread |
| `substrates/nervous/hexgrid.py` | Honeycomb geometry and facing-channel wiring |
| `substrates/nervous/ga.py` | Loop/memory-tuned GA (topology tie-break, duplication, immigrants, caching, SOS hypermutation, annealing) |
| `substrates/topology.py` | Target-blind Nervous connectivity/feedback measurement; FNV's physical extractor is separate and its aggregation is contract-tested against this one |
| `runtime/` | Backend-agnostic run machinery: run/GA config (incl. NV profiles), controller, checkpointing, mutation schedule, worker pool |
| `tools/benchmark.py` | Local terminal solvability sweep driven through the GUI's own worker (`runtime/controller.py`). Select comma-separated architectures or named sets plus target names/categories, preview the Cartesian matrix with `--dry-run`, and resume local JSON/Markdown results after interruption. It performs no version-control, network, or remote-publication operations. |
| `tools/benchmark_contracts.py` | Resumable historical/ablation matrix runner across SNN, the two retired nervous profiles, the current analog profile, and LUT; writes JSON and Markdown after every row |
| `tools/probe_gradient_jitter.py` | Diagnostic separating developmental ruggedness from output-probe refitting movement across one-step Nervous mutants; it changes no evolution behavior |
| `tools/diversity_report.py` | Prints the diversity funnel (and optionally the robustness panel) for a saved population |
| `substrates/lut/` | LUT-array backend (16-bit lookup cells, paper Architecture 2) |
| `substrates/lut/boolfn.py` | Decode 16-bit LUTs to boolean logic (minimised SOP over N/S/E/W) |
| `substrates/lut/ontogeny.py` | sim6 `table_create` morphogenesis — shape browser + the LUT GA's seed factory |
| `experiments/concept/` | Standalone proof-of-concept GA sims (hierarchical / coderack / multi-layer experiments) that predate the main backends |
| `ui/ui_compat.py` | Cross-platform Tk helpers (monospace-font probing, ttk theming) so the GUI behaves the same on Windows / Linux / macOS |

## Usage

```bash
pip install -r requirements.txt

python -m ui.app     # packaged launch
python app.py        # compatibility launch (also safe for existing shortcuts)
```

In the GUI: pick a **Model** (SNN / Nervous / FNV / LUT) and a **Target** from the dropdown
(use the category filter or type part of a name to search; click **Custom…** to
enter your own truth table), set Population / Generations / Restarts / Workers plus the
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
mean, and effective mutation rate (secondary axis). GUI defaults are Population
50, Generations 500, Restarts 1, Workers up to 8, Chroms 2, and Elites 5, with a hot-start
anneal (Mutations 4.0, α 0.997) so slow, steady progress is visible. Growth is
self-limiting via the telomere, so there are no
grid-size or iteration controls; **Max telomere** caps how large organisms may
grow (a chromosome's telomere is its growth *radius*, and eval cost ≈ radius²) —
  LUT runs default to 8 while the other growing backends use 20; raise it when
  a task genuinely needs a larger organism. Use **Load Saved** to reload `results/best_genome.json`
and **Save PNGs** to export the growth and voltage figures.

For LUT runs, the **LUT function banks** row chooses the run-level hardware
inventory. Enable any combination of the named gate families, or leave the
default **Arbitrary LUT** selected for the original unrestricted substrate.
The choice is stored in checkpoints and restored into both evolution and
Interactive playback.

For nervous and LUT runs, every normal completion or Stop also atomically writes
the last fully evaluated generation (including failed genomes and their scores) to
`results/latest_population.json`; the Diversity tab selects it automatically.
`results/solver_generation.json` remains a separate solver-only snapshot and is
therefore empty when no genome reached the `0.999` validity threshold.

Tick **Graded** for harder targets (adders, large custom tables): instead of a binary
pass/fail per output it gives smooth partial credit, which keeps a usable fitness
gradient where binary scoring would otherwise flatline. A perfect circuit still scores
1.0, so it's safe to leave on.

> Note: large targets (full/multi-bit adders, big custom tables) have many truth-table
> rows and evolve slowly — the framework supports them, but small targets are best for
> interactive runs.

For a run trapped at a true local peak, the Escape panel includes **Lineage
walk**. It reserves a configurable share of the existing population for
one-edit, mutation-only lineages. Those lineages are allowed to remain worse
for several generations, which lets later edits build on otherwise-discarded
stepping stones; improvements flow back into ordinary breeding. It adds no
target-specific scoring and no extra evaluations. Rebirth now archives the
generation-zero champion and defaults to a 15-generation trigger, while island
migration ranks evaluated offspring rather than borrowing fitness from the old
genome that happened to occupy the same slot. Every escape mechanism is off in
new GUI sessions and serialized/headless defaults; **Reset escape** restores
that reproducible baseline. The 10% lineage share remains merely the starting
value if the mechanism is explicitly enabled: an early raw-score screen did
not measure solves and is not evidence that it improves solvability.

### Local terminal benchmark

`tools/benchmark.py` runs selectable architecture × target matrices through
the same `runtime/controller.py` path as the application:

```bash
py tools/benchmark.py --list-architectures
py tools/benchmark.py --list-targets --architectures cellular
py tools/benchmark.py --architectures paper --targets "Half adder,Full adder" --dry-run
py tools/benchmark.py --architectures nervous,fnv,lut --targets temporal --seeds 3 --gens 40 --pop 60
```

Architectures are `nervous`, `fnv`, `lut`, and `snn`. Named sets are `paper`
(Nervous + LUT), `nv` (Nervous + FNV), `cellular` (Nervous + FNV + LUT), and
`all`. Target selectors accept exact names plus `temporal`, `combinational`, or
`all`. `--dry-run` expands and prints the matrix without evolving or writing;
normal runs print progress and save resumable local JSON plus a Markdown table
under `results/`. `--help` exposes the optional GUI-equivalent tuning controls.
Benchmarks default to the same conservative worker limit and skip the optional
post-solve solver-bank search so elapsed time measures the requested evolution
and certification. Use `--workers N` or `--diversify-solvers` to opt in to a
different load or the extra solver-bank phase.
The script does not inspect repository state or communicate with external
services.

## Contract v1 benchmark

`tools/benchmark_contracts.py` runs every target/profile matrix row with a fixed
seed and atomically checkpoints JSON plus Markdown after every row, so an
interrupted survey resumes without repeating completed evolution. It retains
the two retired Nervous configurations specifically for historical ablations;
the application itself offers only analog tri-circuit for a fresh NV run.
Generated benchmark artifacts are intentionally run outputs under `results/`
and are not part of the current source tree. Small-population matrices are
diagnostics, not ceiling claims.

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

The current suite contains **533 tests across 28 test files**.

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
  emergent behaviours against an independent numerical integrator;
  `test_input_layout.py`, `test_source_pads.py`,
  `test_lut_input_layout.py`, and `test_lut_exterior_io.py` pin the native
  source architectures and source-only physics;
  `test_lut_function_banks.py` pins the permanent truth-table catalogues,
  unrestricted backward compatibility, configuration round-tripping, and the
  restricted initialization/mutation/growth invariant; `test_topology.py` pins the
  target-blind Nervous/FNV structural tie-break; and
  `test_performance_equivalence.py` compares each optimized hot path to its
  scalar or freshly-built reference, including batched LUT lattice trials
  against independent simulators. Finally,
  `test_scoring_equivalence.py` pins declarative Contract-v1 semantics and
  anti-cheat invariants rather than freezing old mode-specific scores.
- **`reproduce.py`** — each *claim* is a seeded evolution plus a certification
  gate: for input-driven targets it re-samples fresh held-out schedules and
  reports **CERTIFIED** (generalises), **OVERFIT** (memorised timing),
  **BELOW THRESHOLD**, **SOLVED**, **PLATEAU** (a documented substrate limit),
  or **UNCERTIFIED**, as appropriate.
- **Automatic certification** — every supported GUI/controller run of an
  oracle-backed temporal target certifies its winner when it finishes: the
  verdict is shown in the status line and saved into the checkpoint
  (`substrates/nervous/certification.py`, one shared rule with `reproduce.py`).
  Asynchronous truth tables are replayed exhaustively under freshly shuffled
  row schedules and require held-out fitness ≥0.999, catching circuits that
  memorized schedule position instead of computing the presented row.
  LUT exterior-edge runs are the current exception because their frozen
  outside-to-facing-edge validation adapter is still pending; they report
  **UNCERTIFIED** rather than presenting an unaudited held-out score.

## Requirements

- Python 3.10+
- numpy
- matplotlib
- pytest is **optional** — the suite also runs with bare Python via
  `py tests/run_tests.py`

## Concepts & provenance

The architectures and mechanisms trace to Andrew Edwards' *Evolvable Hardware* work on **grown**, indirectly-encoded circuits; the code cites specific figures inline where they apply:

- **Nervous net** — the honeycomb "nervous net" of Edwards **EH'02**: three independently configured directional circuits per tile (Fig. 2), the 16 coincidence/veto states per circuit (Fig. 3), and context rotation (Fig. 4). The temporal/oracle target suite is layered on that substrate.
- **LUT array** — the paper's **Architecture 2**, ported faithfully from the `sim6` reference implementation and the degree-4 case of `automaton_arrays` (4 neighbours × four 16-state lookup tables per cell; `table_create` morphogenesis in `substrates/lut/ontogeny.py`).
- **Substrate framing** — identical tiled cells, routing SRAM / lookup tables as the reconfigurable element, and the genome as an associative memory (`context → new state`) all map onto the papers' hardware picture; the code comments name the corresponding hardware element at each point.

Two design threads are this project's own, layered on that base: **temporal / memory circuits** scored on spike-event timing (SR latches, oscillators, cadence steppers, gated memory) rather than only combinational truth tables, and **biologically-inspired search** (telomere/Hayflick-bounded growth, stress-induced hypermutation, annealed mutation, SNN solved-only parsimony, and Nervous/FNV connectivity-and-feedback selection). See the section comments in `substrates/nervous/ga.py` and `substrates/nervous/temporal.py` for the details behind each.
