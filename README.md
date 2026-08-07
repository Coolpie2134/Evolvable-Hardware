# Evolvable Hardware: Logic Circuits Grown Instead of Designed

This project evolves asynchronous logic circuits. The unusual part is that the
thing being evolved is **not a netlist**. Each candidate carries a small set of
growth rules; a circuit is grown from those rules on a regular lattice, and it
is the grown circuit that gets simulated and scored. Following Andrew Edwards'
*Evolvable Hardware* papers, this is called an **indirect encoding**.

Four cell architectures are implemented: a spiking neuron array (SNN), a
hexagonal "nervous net", the Functional NV Net (FNV), and a square lookup-table
array (the papers' Architecture 2, ported from the `sim6` reference).

## Vocabulary

The search machinery borrows its terms from genetics, which can obscure what
are ordinary engineering ideas. Translations used throughout this document:

| Term used here | What it actually means |
|---|---|
| Genome | The heritable data for one candidate: a rule set, not a netlist |
| Growth / development | Running the rules to produce an actual circuit on the lattice |
| Phenotype | The grown circuit that results |
| Fitness | A score in [0, 1] from simulating the grown circuit against a spec |
| Mutation | A small random edit to the rule set |
| Crossover / recombination | Building a child's rule set from two parents' |
| Population / generation | The batch of candidates held at once, and one breed-and-score cycle |
| Elites | The top-scoring candidates, reserved as parents |
| Immigrants | Freshly randomized candidates injected each generation to restore variety |
| Telomere | A division counter carried per lineage; it caps growth radius, so circuit size is set by the rules rather than by an external size limit |
| Germline / seed cell | The lattice cell(s) growth starts from |
| Behavior contract | A machine-readable spec of what the circuit must do, stored as data |
| Certification | Re-testing a winning circuit on freshly generated stimuli it never saw while evolving, to separate real function from memorized timing |
| Target | One task: a truth table, or a timing/memory behavior |

One term is worth stating plainly because it drives much of the design. Parent
selection on truth-table tasks uses **lexicase selection**: instead of ranking
candidates by one averaged percentage, it takes the test cases in random order
and repeatedly keeps only the candidates that are best on the current case.
A circuit that nails one hard row survives even if its average is poor, which
is what keeps partial solutions available to recombine.

## What's new

- **Three-circuit tile** (`substrates/nervous/tritile.py`): the nervous net now
  implements Edwards EH'02 Fig. 2 as drawn, with **three independently configured
  circuits per tile**, one driving each of the L/R/D output directions (tile
  state is 15 bits: three 5-bit channels). The older single-circuit tile
  broadcast one result to all three outputs, so it physically could not route
  two different signals to two different outputs. This one can.
- **Analog node model** (`substrates/nervous/analog.py`): an event-driven model
  of the Fig. 1 cell: capacitive input steps, Vbp leak toward Vdd, and a
  comparator with hysteresis. The coincidence window, output pulse width,
  stretching under dense input, and refractory recovery are no longer separate
  tunable knobs; they all fall out of the same handful of device constants.
  Checked against an independent numerical integrator.
- **One current nervous-net configuration**: new runs use **Analog tri-circuit**
  (the three-output tile plus the analog cell above). Vth, step, tau, and
  hysteresis get their own GUI row. The older single-tile and all-digital
  tri-circuit engines still load, for reading old saved runs and for controlled
  comparisons, but `validate_new_nv_profile` refuses them for a fresh run.
- **Specs stored as data** (`substrates/nervous/contracts.py`,
  `substrates/nervous/scoring.py`): each task states what it requires in a
  machine-readable form: match a truth table, match timed edges one-for-one,
  hold a logical state regardless of phase, reproduce complete pulse intervals,
  or maintain a commanded rate. Every backend converts its physical output into
  observations, and a single `score_contract(...)` evaluator does all scoring.
  This replaced seven separate per-task scoring modes. State is scored by the
  longest silent gap across continuous intervals, so sliding the same
  circulating pulse in phase cannot change its score.
- **Truth tables scored row by row**: asynchronous combinational runs keep every
  row/output check separate and select parents with lexicase rather than one
  averaged percentage. Timing comparisons keep a tolerance derived from median
  absolute deviation; exact Boolean checks use zero tolerance. Where a task has
  several outputs, the reported score mixes the mean with the *weakest* output,
  and the output-probe fitter optimizes that same quantity, so a solved easy
  output cannot mask one performing at chance. A single number is still
  reported for plotting and for stopping at exactly 1.0.
- **Partial solutions survive to be recombined**: until something solves the
  task, a rotating 40% of the population is reserved for candidates that are
  each the current best on *some individual test case*, prioritized by the cases
  the population is currently worst at. The rest of the slots turn over
  normally. When lexicase is active, the second parent is chosen for covering
  the first parent's weak cases, rather than drawing two parents independently
  and often getting two specialists in the same rows.
- **FNV starting components**: when logic gates are in the selected component
  bank, exhaustive truth-table runs build their starting population from gates
  plus the delay/fan-out routing parts. Mutation still draws on the complete
  selected bank, so delays, holds, normalizers, and stateful parts stay
  reachable. Timing tasks and deliberately gate-free banks are unaffected.
- **FNV multi-output selection**: individual row/output cells remain separately
  selectable, and selection additionally sees per-output balanced accuracy,
  per-row all-outputs-correct, and weakest-output accuracy. Reported fitness and
  certification are unchanged. Recombination now moves a complete upstream cone
  (every component a chosen node depends on) rather than mixing fields of
  unrelated components.
- **FNV builds circuits by explicit reference**: new FNV runs no longer classify
  a cell by nearest-matching neighbourhood context. Each gene names a permanent
  component type, the specific output ports feeding it, and the connected run of
  components it belongs to. Placement order follows those dependencies rather
  than position in a list, so if two genes claim the same cell only the losing
  gene and whatever depends on it go dormant; unrelated structure still builds.
  Mutation can extend or join live wire ends, swap in a compatible component,
  bridge across empty lattice, move/duplicate/delete a connected run, or move an
  input pad. Fan-out stays physical: extra wires require an input pad or a real
  two-output component, never an abstract split. Saved runs in the older format
  still load under their original interpreter.

## Background: grown, not designed

The obvious way to evolve a circuit is to encode it **directly**: one gene per
gate or wire, so the genome is a netlist. That approach has three well-known
problems. The genome grows in proportion to the circuit. Repeated
sub-structures have to be discovered separately every time they appear. And the
search landscape is rough, because flipping one bit rewires exactly one gate.

This project uses the **indirect** approach from Edwards' work: the genome holds
**growth rules** instead of a circuit. Starting from a few seed cells, the rules
are applied repeatedly until a circuit exists. Every cell holds the same rule
set and consults it the same way. *Given the pattern of states around me, what
do I become?* The lookup is by nearest match (minimum Hamming distance), so a
cell's fate is decided by its local context, and one rule fires wherever that
context occurs.

Three consequences matter:

- **Genome size is decoupled from circuit size.** A fixed rule set can grow an
  arbitrarily large structure, and a rule that builds one motif builds it
  everywhere that motif's context appears.
- **Edits are structural, not pointwise.** Changing one rule reshapes every copy
  of the motif it builds at once, which makes the landscape smoother than
  one-bit-one-gate encoding.
- **Size is set from the inside.** Growth stops on its own at a radius the
  genome specifies (the division counter described in the vocabulary table
  above), rather than running into an external size cap.

The substrate is a regular lattice of reconfigurable cells. Four cell
architectures are implemented: a spiking neuron, the two cellular-automaton
architectures from the Edwards papers, and FNV's bank of fixed-function
components. Both kinds of task are evolved: **combinational** logic, scored
against a truth table, and **timing-dependent** circuits, where the answer lives
in *when* edges occur: latches, oscillators, counters, delay lines, and pattern
generators.

## The four substrates at a glance

| | **SNN** (`substrates/snn/`) | **Nervous net** (`substrates/nervous/`) | **FNV** (`substrates/fnv/`) | **LUT array** (`substrates/lut/`) |
|---|---|---|---|---|
| Origin | LIF variant | Edwards EH'02 honeycomb | standalone fixed-function extension | paper Architecture 2 / `sim6` |
| Lattice | square grid | honeycomb, **degree 3** | honeycomb, **degree 3**, antiparallel wires | square grid, **degree 4** |
| Cell function | leaky integrate-and-fire neuron | `(E1 op E2) AND NOT I1` routing node | one of 118 permanent routed component types | four directional 16-bit tables, optionally restricted to fixed gate banks |
| Dynamics | **event-driven** continuous LIF | **asynchronous** edge-triggered pulses | **event-driven** fixed-function logic/state | **asynchronous** level logic |
| Spontaneous activity? | **no** | **no** | **no**, every type is quiescent | arbitrary LUTs: **yes**; named gate banks: **no** |
| Computes | combinational **+ temporal** | combinational **+ temporal** | combinational **+ temporal** | combinational **+ temporal** |

## What it does

A genetic algorithm evolves a physical circuit. The nervous net, SNN, and LUT
backends each grow one from their own rule format. FNV is the control arm: it
skips context-matched growth entirely and assembles fixed-function components
outward from its input pads by explicit reference. Four backends interpret and
score the resulting lattice:

- **SNN** (`substrates/snn/`): a square grid of leaky integrate-and-fire
  neurons.

  Truth-table runs keep the original feed-forward reading. Timing runs instead
  enable reciprocal adjacencies (`Arch.recurrent=True`) and feed exact
  continuous-time input edges through the same specs as every other backend.
  Spec times are in seconds and are converted explicitly into the LIF model's
  milliseconds (4.8 ms per spec second), so a requested period is compared
  against feedback loops the model can physically produce rather than against a
  mismatched clock. That makes evolved oscillators, latches, and counters
  possible without disturbing existing truth-table saves.

  A deterministic cap on total events rejects runaway loops. A hand-built
  two-neuron feedback circuit is included as a witness: it keeps firing after
  its input stops and scores 1.000 on the period-2 oscillator spec, proving the
  spec is satisfiable rather than merely unmet.

  Circuit size is limited the same way as the nervous net. Each chromosome
  carries a division counter, so a lineage stops dividing when it runs out and
  the body settles at radius L from the seeds. Once a circuit solves the task,
  a tie-break prefers smaller ones, trimming genes and shortening the counter
  toward the smallest circuit that still works. (`GRID_SIZE` remains only as an
  outer wall keeping coordinates valid for the growth view and fixed I/O
  terminals; the division counter is what actually limits size.) The interactive
  view animates membrane voltage and spikes live; see `ui/interactive.py`.
- **Nervous net** (`substrates/nervous/`): a honeycomb array whose nodes fire on
  `(E1 op E2) AND NOT I1`, which is coincidence detection with a veto input.

  Each node's routing is one of 32 states. States 0-15 are the paper's (Edwards
  EH'02 Fig. 3): each is a buffer or an AND, optionally vetoed. The paper has no
  OR. States 16-31 are an addition here: identical wiring with `op = OR`, so
  the node fires on *either* excitatory input. The extra state bit is 0 in every
  older genome, so states 0-15 grow and behave bit-identically to before. Only
  the OR of two *different* inputs is genuinely new; where the state is a buffer
  or off, AND and OR coincide and the twin is a harmless alias.

  Directions are read in each node's own orientation frame, so a rule builds
  mirror-symmetric structure (the papers' context rotation; down-parity tiles
  are the circuit rotated 180 deg).

  Inputs are **wired-OR injections** onto the input cell's net, the equivalent
  of pulsing a perimeter wire. They are not level clamps holding a value. Two things
  follow. With no input, the array is completely inert. And a pulse injected
  into a loop of buffers routed through an input cell keeps circulating, giving
  delay-line memory that persists "until stopped by application of an inhibitory
  input" (section 3).

  Alongside truth tables it evolves timing circuits: SR latches, toggle
  flip-flops, oscillators, and delay lines, plus ones where edge timing *is* the
  computation: a two-input coincidence detector, a monostable that terminates
  its own burst through delayed self-inhibition, and a double-pulse detector
  built from a delay line and a coincidence node.

  Dynamics are asynchronous and edge-triggered (`substrates/nervous/pulse.py`).
  A triggered node emits a fixed-width pulse after a fixed delay, coincidence
  requires both excitatory edges inside a window, and the veto input suppresses
  at trigger time. Scoring uses the raw continuous edge timestamps; the
  once-per-tick samples exist only for playback, so pulses shorter than a tick
  are no longer thrown away.

  Every element maps onto hardware in the papers: identical tiled nodes, 5 bits
  of routing SRAM per node (4 in the paper, the extra bit selecting AND vs OR),
  the genome as a content-addressed lookup, and fixed delay, pulse-width, and
  coincidence-window constants.

  Input and output placement is itself evolved. Each logical input gets one
  honeycomb coordinate, and no two may collide. Input 0 is pinned at `(0, 0)`
  purely to stop the whole circuit from drifting around the lattice; every other
  pad moves one edge at a time under mutation. Pads seed growth and are
  source-only once running, so feedback cannot re-drive them.

  Outputs are not placed in advance. Any grown non-input tile may serve as an
  output probe, and where a task has several outputs they are assigned together
  so that each lands on a distinct cell. Pads and probes are then frozen before
  certification, so the held-out test cannot quietly re-fit a better readout.
  Older saves without an `input_layout` keep the pads their task declared.

- **Functional NV Net / FNV** (`substrates/fnv/`): a separate degree-3 honeycomb
  substrate whose cells are fixed physical functions rather than configurable
  pulse nodes. Every edge carries two independent antiparallel wires.

  A two-input component takes two of its three local directions and drives the
  third; a one-input component drives either one direction or two identical
  ones. The catalogue holds 118 permanent component types and is append-only:
  `EMPTY`, AND/OR/XOR and asymmetric VETO routings, fixed 1- and 2-tick delays,
  `NORMALIZER1/2`, `HOLD1/2`, Muller C-elements, rising-edge toggles, and
  enable-gated oscillators with fixed high/low times of 1 or 2.

  Anything wider, such as parity-3 or majority-3, has to be built from those parts.
  There is no free-running oscillator and no per-component tunable parameter; a
  run selects whole component families, nothing finer. Simulation is
  continuous-time and event-driven, and every catalogue entry is quiescent at
  power-on with all-zero inputs.

  Saves record a SHA-256 hash of the catalogue, so renumbering a component type
  can never silently reinterpret stored hardware as something else.

  The FNV control row includes a **Node number dictionary** that decodes every
  number drawn inside a component into its permanent name, family, routed
  inputs/outputs, and fixed timing. Code can use the catalogue-derived
  `NODE_TYPE_DICTIONARY` mapping for the same ID-to-name lookup.

  Each gene names a permanent gene ID, a fixed component type, the specific
  output ports feeding its inputs, and which connected run of components it
  belongs to. All the ports it references must face the same empty lattice site.
  Genes are resolved in dependency order, not list order. If two genes claim one
  cell, the one whose dependencies were satisfied first wins; the loser and
  anything depending on it go dormant, while unrelated structure still builds.

  Nothing in this path consults the task: there is no target-shaped grid, no
  query for a desired output, no automatic router, and no way to convert a
  finished circuit back into a genome.

  Input pad positions are evolved too, on the same terms as the nervous net:
  input 0 pinned at `(0, 0)` to remove pointless whole-circuit translation,
  others moving one edge at a time, the layout inherited as a single
  collision-free unit. Because dependencies are named explicitly, moving a pad
  translates only the structure rooted at that pad. Starting layouts are compact
  so random populations do not begin as disconnected islands. Pads are
  source-only while running: only the task's external input pulse drives them.

  Mutation extends or joins live wire ends, swaps a compatible component,
  bridges across empty lattice with an explicit delay-plus-gate chain, and
  reroutes, duplicates, or deletes a whole connected run. When progress stalls,
  the extra candidates offered include bridges sampled across the available
  length range and a bounded two-bridge chain, so a first route that changes
  nothing can still be extended later. None of this reads the task's answers.
  Recombination between parents with matching pad layouts inherits a complete
  upstream cone; components clashing at overlapping cells are displaced locally.

  Fan-out is strictly physical. Extra wires come from an input pad or a real
  two-output component; there is no abstract split.

  Outputs are fitted probes rather than component parameters. Any grown
  non-input component is eligible no matter how far it sits from where the task
  drew its terminal, and multi-output tasks solve one global assignment to
  distinct cells rather than greedily filling one role at a time. Pads and
  probes are frozen before certification. `tools/benchmark_fnv.py` holds
  reproducible small-budget comparisons.

  **FNV deliberately does not prefer smaller circuits.** Correct behavior
  dominates selection. Exact ties are broken by a structural measure that never
  inspects what the circuit computes: the wiring graph is traced outward from
  the input pads and ranked by how many distinct inputs converge anywhere, how
  many genuine multi-input junctions exist, and how much reachable feedback,
  wiring, and cells there are. Because there is no size penalty, a long
  connected route that currently does nothing can survive as a stepping stone,
  while disconnected bulk and unreachable loops earn nothing. Reports separately
  list the distinct non-constant truth signatures a circuit exhibits; that is
  diagnostic only and never reaches selection.

- **LUT array** (`substrates/lut/`): the papers' **Architecture 2**, ported from
  the `sim6` reference. A square grid where each cell connects to its four N/S/E/W
  neighbours and holds four directional 16-bit lookup tables, one per output
  direction.

  During growth, each direction's table is indexed with the neighbour pattern
  rotated so that the output direction is "front". This is the same context rotation
  the nervous net uses, which is what lets one rule build symmetric structure.

  At run time the array is **asynchronous level logic** (`substrates/lut/pulse.py`).
  Each cell behaves as a logic element with a fixed propagation delay: whenever
  an input changes, it re-indexes its four tables using the four bits its
  neighbours drive back at it. The delay is inertial, so glitches shorter than
  the delay are filtered rather than propagated. Simulation is event-driven in
  continuous time. Set the delay to one tick and drive it on the tick lattice
  and it reproduces `sim6`'s synchronous latched update bit for bit. The
  retained `lut.LutSim` is kept as the audited reference for exactly this check.

  Arbitrary tables can invert, so that mode permits spontaneous activity, which
  matches the papers' Fig. 14. The named gate banks below are all quiescent at
  all-zero input. Runs the same truth-table and timing tasks as the others.

  A LUT run can select whole **function banks**: `ROUTING`, `AND`, `OR`,
  `XOR`, asymmetric `VETO`, `THRESHOLD`, `MUX`, and `UNRESTRICTED`
  ("Arbitrary LUT" in the GUI). These are permanent physical 16-bit truth
  tables, not evolvable parameters: the named catalogue contains all routed
  2/3/4-input AND/OR/XOR variants, all directed vetoes, majority/threshold
  variants, and all directional 2:1 mux variants. `OFF` is always available.
  Restricting the bank limits only what a cell can *execute* and what growth may
  seed. The five 16-bit pattern-matching fields stay fully expressive, because
  they match growth context rather than acting as gates at run time. Starting
  genomes pick a family first and then a table within it, and mutation usually
  moves to the nearest table in the current family, so a large family does not
  win simply by containing more entries. Arbitrary LUT alone is the default and
  reproduces the earlier all-65,536-table search exactly; saves predating the
  setting load that way.

  There are two input arrangements. **Internal source pads** evolve one distinct
  coordinate per logical input, seeding growth and acting as source-only
  terminals, matching the nervous net and FNV.

  **Exterior perimeter buses** instead grow from one neutral seed and tap every
  exposed outer face. Faces are taken in stable cyclic order and assigned round
  robin: one input owns the entire perimeter, two inputs alternate A/B/A/B,
  and more inputs repeat A/B/C/... the same way. A logical signal drives all taps
  of its bus simultaneously. Every tap stays outside the circuit and feeds only
  the LUT input facing it, so feedback can neither consume nor re-drive it;
  faces around enclosed holes are excluded. This geometry is fixed rather than
  evolved, which takes port placement out of the search while leaving the grown
  circuit free to evolve.

  Both modes share the same globally fitted, distinct, read-only output probes,
  and the selected mode is recorded in saves. Certification is the exception:
  the frozen-replay adapter has not yet been specialized for the exterior
  outside-to-facing-edge links, so exterior runs report UNCERTIFIED rather than
  present a held-out number that has not been audited.

  (The hex nervous array is the degree-3 case of the same `automaton_arrays`
  design, with 3 neighbours and 3 tables of 8 states, drawn as a brick wall.
  Deliberately not the degree-6 triangular reading.)

- **Nervous-net substrate profile** (the **NV profile** display in the pulse-physics row; `runtime/config.py: NV_NEW_RUN_PROFILES`): a fresh nervous-net run uses **Analog tri-circuit** (`tri3` + `paper_analog`), the paper's Fig. 2 tile with three independent 5-bit L/R/D channels (15 bits total) and the Fig. 1 analog node (`substrates/nervous/analog.py`). Capacitive down-steps, Vbp leak, and comparator hysteresis make coincidence, output width, and refractory **emergent**, so the Coinc control is disabled and Vth / step / tau / hysteresis have their own GUI row.
  Two older engines remain, purely for loading old saves, running tests, and
  making controlled comparisons: the width-preserving single-tile engine
  (`single` + `pulse_delay`) and the all-digital tri-circuit engine
  (`tri3` + `uniform`). `validate_new_nv_profile` refuses both for a fresh run.
  The digital tri engine understands the same 15-bit OR-capable channel format,
  and older 12-bit AND-only saves are widened exactly once on load.

  One earlier variant let each node type evolve its own emitted pulse width.
  **That has been removed.** The genome no longer carries a width vector, and a
  save made under it now loads onto the fixed-width node instead. Timing models
  are checked in `tests/test_pulse_models.py`, the three-circuit tile in
  `tests/test_tritile.py`, and the analog cell against an independent deltat
  integrator in `tests/test_analog_reference.py`.

  Tasks that depend on pulse width now test useful relationships rather than
  isolated node behavior. **Pulse width sum (A+B)** must emit an output interval
  lasting `width(A) + width(B)`. **Odd pulse selector** must pass the 1st, 3rd,
  and 5th input pulses while preserving each one's individual width. Both score
  the complete output waveform, real rising *and* falling edges, so pulses
  that merely share a leading edge are not treated as equivalent.

  A related group requires counting pulses whose durations vary from 0.5 to 2.25
  seconds, each counted exactly once regardless of length: **A-count parity
  queried by B** holds running odd/even parity, **A-count multiple-of-3 queried
  by B** recognizes positive multiples of three, and **Odd A batch closed by B**
  reports the parity accumulated since the previous B and then clears it.

The SNN and nervous backends share no code. FNV is independent of the nervous
runtime and shares only the task specs and the honeycomb geometry, not the
node physics, and not the genome format.

Nervous and FNV grow on an **unbounded field**, iterating until the structure
stops changing. Nothing external caps the size; the genome does. Seed cells
start with a division counter *L*, each daughter inherits one less, and growth
therefore halts by itself at radius *L*. The counter limits only division;
rules that maintain existing cells keep acting normally.

The GA never stops early on reaching a perfect score. SNN prefers smaller
circuits, but only once the task is solved. Nervous and FNV deliberately do not:
their final tie-break rewards connectivity and feedback reachable from the
inputs. LUT has no size preference at all.

Before anything solves the task, most population slots turn over normally while
a bounded rotating reserve holds candidates that are each best on some specific
test case, so a single unsolved truth-table row cannot be quietly forgotten.
Once something does solve it, survivor selection over evaluated parents and
offspring keeps it while exploratory offspring continue to be tried.

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
comparator uses a two-stage strobe/zero-detector cascade. The resulting circuit is
converted back into a heritable genome, re-grown exactly, and scored by the
unchanged behavior contract. Verified witnesses reach 1.000 on Full adder,
2-bit adder, MUX, Majority-3, Parity-3, decoder, comparator, and the 2x2
multiplier within the default eight-step growth bound.

- **How a rule is looked up**: the genome is a set of rules mapping a
  neighbourhood pattern to an output state, matched by *nearest* pattern
  (minimum Hamming distance) rather than requiring an exact hit, so a finite
  rule set covers every context a cell might encounter. A nervous-net rule holds
  five state-valued fields: `L context`, `R context`, `D context`, `self in`,
  `self out`. Each is 5 bits (0-31) on the retired single-circuit tile and 15
  bits on `tri3`, where it packs three independent 5-bit channels.
- **Context rotation** (nervous net): one rule is applied separately to each of
  a tile's three L/R/D circuits, with the neighbour pattern rotated each time so
  the circuit being updated always sits "front". This is what lets a single rule
  grow mirror-symmetric structure (EH'02 Fig. 4).
- **Recombination**: chromosomes with several genes swap matching tail segments
  at a real interior boundary. A single-gene chromosome instead swaps a subset
  of that one rule's active fields, so even minimal genomes can produce genuinely
  mixed children rather than copies. Parents are picked to differ in rule
  content, and a multi-edit mutation always ends on a value neither parent had,
  so a pair of opposite edits cannot silently cancel back into a clone.
- **Growth**: repeated rule application expands the seed cells into a full circuit.
- **LIF simulation** (SNN): neurons drive each other through synaptic currents.
  Truth-table runs read the output as spike / no spike; timing runs keep the
  continuous event times and hand unit-width spike intervals to the shared scorer.
- **Per-output conventions** (SNN): some experimental tasks use complemented
  inputs or an inverted spike reading, but everything in the bundled arithmetic
  set is direct: high means logic 1 on both input and output.
- **Task definitions**: a `Target` (`substrates/snn/targets.py`) is the one place
  that defines input count, output roles, and truth table. Nervous, FNV, and LUT
  runs take their physical input positions from the genome and fit output probes
  from measured behavior, so a task never dictates circuit geometry.
- **Combinational target library** (`substrates/snn/targets.py`, run on **all four** models): logic gates, half/full/2-bit adders, plus the harder **2:1 MUX**, **Majority-3**, **Parity-3 (XOR3)**, **2-to-4 decoder**, **2-bit comparator** (GT/EQ/LT) and **2x2 multiplier** (4-bit product), covering data routing, voting, wide parity, relational logic and arithmetic. Nervous and LUT encode every row as its own widely-spaced window via `periodic_combinational_target(...)`: the row's inputs are **held high** for a grid diameter plus a read window, then released with a settle gap before the next row, and the output must **sit at its required level** through that read window (the `combinational_level` contract). Held level in, held level out - the same thing combinational means on every other backend (FNV holds each row's input levels, waits out a circuit-derived settling time and reads the final level at globally fitted probes; SNN holds its input current for the whole run). A momentary pulse is a partial answer, scored by its duty. This is a real physical demand and the two asynchronous substrates answer it differently - the LUT array holds levels natively, while the nervous net's analog node emits a fixed-width pulse per edge and has to evolve sustained activity to hold at all; see the "Held combinational levels" section of the design doc for the measurements. This prevents harmless settling glitches from being scored as permanent logical highs while keeping every row/output as a separate exact lexicase case.
- **Temporal target library** (`substrates/nervous/targets.py`, run on **SNN + Nervous + FNV + LUT** except where target metadata records a physical exclusion): eight hand-built banks cover oscillator, `Pattern (1000)`, coincidence, Temporal XOR, Sequence A->B, Veto, Burst x3, and Divide-by-3. The registered oracle-backed set adds pulse-width sum, odd-pulse selection, three A-count query functions, SR latch, C-element, refractory filtering, A-first rendezvous, collision serialization, watchdog timeout, toggle, echo, a 12-second non-retriggerable one-shot, period doubling/tripling/halving, temporal interval sum, two pair detectors, period stepper, gated oscillator, and resettable toggle. Each target declares its own backend/model exclusions; unsupported combinations are hidden rather than scored against impossible physics. FNV feeds its physical events and intervals into these same contracts.
- **Coincident-edge temporal twins** (`coincident_temporal_target`): every
  combinational truth table also appears as a `<name> (temporal)` entry, so each
  function can be evolved either as settled logic or as edge timing. The
  difference is what the trial looks like. The periodic combinational wrapper
  gives each row its own widely-spaced window, holds the row's inputs as a level
  throughout it, and requires the output to hold its answer while the row is
  applied. The temporal twin presents each row as one point event (no hold), packs
  every row into one trial a
  few seconds apart, and scores exact edge correspondence, so a circuit only passes if it
  also recovers between rows: lingering state, a stuck output, or a slowly
  ringing path all cost score even when the logic is right.

  Two encoding rules follow from the substrate being quiescent. A row whose
  inputs are all 0 delivers no edge at all, so it is dropped unless it carries
  real evidence, in which case a case-valid strobe lane is added under the same
  rule the combinational wrapper uses. And because one-to-one event scoring
  rewards a blanket responder in proportion to how often the table asserts,
  quiet rows are repeated on lopsided tables such as OR and NAND until firing on
  everything scores below 0.7. `Coincidence (2-in)`, `Temporal XOR (2-in)` and
  `Veto gate` remain the separate hand-built versions of temporal AND, XOR and
  veto, and keep their names.
- **Interval transformation targets**: **Period tripler (3x)** emits every third input edge, **Period halver (1/2x)** first measures a complete even input period and then inserts midpoint events, and **Temporal sum (deltaA + deltaB)** measures two intervals on each of two inputs and encodes their sum as the interval between two Q events. Their seeded banks mix periods/intervals and include silent, one-lane, and incomplete guards, preventing direct wires, fixed bursts, and free-running oscillators from solving them.
- **Physical pair target** (Nervous + LUT, the asynchronous backends): **Pair detection gap (2x pulse width)** supplies explicit fractional-phase input pulses and emits Q only when consecutive leading edges are separated by exactly twice the supplied input-pulse width; wrong relative gaps, chains, isolated pulses, and silence are included.
- **Temporal time units**: target names, descriptions, reports, and plots use seconds. Nervous events may occur at fractional seconds; one LUT gate delay is defined as one second.
- **Temporal targets** (`substrates/nervous/targets.py`): scored on raw events, cadence invariants, or persistence traces as the behavior requires, across several shifted trials, so only genuine timing or memory behavior passes rather than a lucky delay chain
- **Oracle targets** (`substrates/nervous/oracle.py`): rather than hand-picking input timings (which can be adversarial to a circuit's internal phase), a goal is specified as a *reference state machine* `oracle(inputs, state) -> (outputs, state)` plus a *stimulus generator*. Seeded schedules are labelled by the oracle; the circuit is scored on reproducing the input->output *relation*, and `holdout_score` re-samples fresh schedules to certify that it generalises rather than memorising timings. Exact-delay and interval targets disable free alignment where absolute timing is the behavior. Stateful banks include boundary, silence, re-arm, repeated-command, and long-horizon cases; for example, the period stepper must sustain cadences 2 -> 4 -> 6, and the watchdog mixes safe, exact-deadline, late, re-arm, and never-armed trials.
- **Scoring timing behavior** (`substrates/nervous/contracts.py`,
  `substrates/nervous/scoring.py`; `substrates/nervous/temporal.py` only runs the
  trials and collects observations): the question asked is *did this circuit
  implement the required behavior?*, not *did it reproduce one particular
  waveform?* Each task carries a spec listing one or more reusable requirements,
  and every backend hands normalized observations to `score_contract(...)`:
  - *One-to-one timed events* (`event_correspondence`) pairs expected and produced continuous-time leading edges. Missing and spurious edges both cost. A target may fit one latency shared across every trial/output or require absolute timing, as Echo delay 3 does.
  - *Sustained and commanded cadence* (`sustained_cadence`, `commanded_cadence`) measures regular rhythm, event count, dwell coverage, silence outside active epochs, and command-induced rate changes without requiring an arbitrary absolute phase.
  - *Phase-invariant logical state* (`logical_state`) compiles the abstract expected state into active and quiet epochs. A nervous stored 1 may be a circulating pulse; the scorer measures its regular lap across commanded-active epochs (never below the smallest legal ring), uses that same lap for retention and transition timing, and grants one-lap stopping grace for the pulse already in flight when a reset arrives. Shifting an identical ring in phase or building a larger valid loop therefore cannot reduce fitness.
  - *Complete pulse intervals* (`pulse_intervals`) checks both rise and fall, so the right edge times with the wrong pulse widths cannot pass.
  - *Bounded persistent state* (`bounded_state`) expresses long-horizon hold, quiet, clear, reset influence, and reload cases through the same entry point used by ordinary targets.
  - A deterministic per-run event cap rejects pathological oscillators before one genome monopolises evaluation. Event/cadence evaluation schedules input edges directly, skips `T x cells` display snapshots, reuses one input cone across every trial, and uses allocation-free sparse matching; sampled states are built only for playback and persistence targets.
  - *Trace-matched output placement*: Nervous, FNV, and LUT fit each output role over the whole mature organism. Multi-output roles are assigned globally to distinct cells, so an early role cannot greedily consume the only strong probe for a later one. SNN retains its own backend-specific readout placement.
  Multiple restrictions aggregate as half weighted mean plus half worst restriction: useful partial progress remains visible, but an easy restriction cannot hide a failed one, and fitness 1.0 requires every restriction to pass.
- **Semantic period-stepper contract** (`commanded_cadence`): some behaviours have no single "correct" trace to match. A cadence controller must hold a regular output pulse rate and make it slower after each command. Every command-delimited dwell is checked for sustained regular cadence, and later dwells must have a strictly longer period. A fixed-rate oscillator scores **0** on the change term regardless of raw trace overlap.
- **Describe a target as spike events** (`substrates.nervous.spike_target`): define a new temporal function directly from test cases. `spike_target(name, cases, T, n_inputs=...)` takes cases where each case is `(input_spikes, output_spikes)` (input ticks per input, expected output ticks). Pair it with `substrates.nervous.ga.diversify` to turn one solution into a whole generation of genotypically-unique valid solvers.
- **Structural tie-break** (`substrates/nervous/ga.py`, `substrates/topology.py`):
  when two circuits score identically, the tie is broken by structure that the
  input pads can actually reach: cells, wires, points where several inputs
  converge, cells on cycles, the number of independent loops, and how many
  separate feedback regions exist. Disconnected bulk and loops nothing can reach
  count for zero. This measure never inspects what the circuit computes, so it
  cannot smuggle in task-specific knowledge. Nervous applies diminishing returns
  (`log1p`) to the counts; FNV uses its own extractor and ranks the terms in
  strict order without caps. Correct behavior always outranks structure.
  Nervous also keeps an older timing-only nudge: short of a perfect score, a
  feedback loop that can be both written and read adds at most 5% of whatever
  score remains.
- **Search dynamics** (`substrates/nervous/ga.py`, `substrates/lut/ga.py`,
  `substrates/fnv/ga.py`): if 12 generations pass with no real progress, the
  mutation rate is raised until progress resumes. "Progress" counts either a
  better overall score *or* improvement on the worst-performing test cases,
  otherwise a run that is genuinely fixing its hardest case would be mistaken
  for a stalled one just because the averaged number has not moved.
  Independently, the base mutation rate decays each generation by **Anneal alpha**
  (a simulated-annealing schedule: start hot, cool down). **Plateau beta** sets how
  sharply the stall response raises the rate. 0 disables it, 1 is the tuned
  default, larger is more aggressive.
- **Evaluation performance**: GUI/controller runs use an explicit **Workers** limit (default `max(1, min(cores - 2, 8))`, allowed 1-16), reuse one persistent worker pool across generations, deduplicate identical genomes before submission, and keep a bounded fitness cache. Stop cancels queued work and drains running workers before another run can begin. Hot paths compile developmental lookups once: Nervous/SNN pack a complete context into one XOR + `bit_count`, constructive FNV resolves named dependencies once (legacy associative-v2 uses its categorical-distance matrix), and LUT retains its vectorized table engine. FNV also compiles the grown circuit's wiring once and reuses it across output fitting, trials, cases, and topology. Acyclic stateless FNV truth tables use an exact bit-parallel settled evaluator; cycles and stateful components fall back to continuous-time events, and certification always replays physical dynamics. Ordinary Nervous evaluation grows once for both behavior and topology; LUT resets one compiled simulator between timing replicates and vectorizes steady-duty extraction; fitness-only SNN runs skip voltage-history recording. Nervous target-only expected windows are cached during global probe fitting, and an event overflow stops the remaining trials immediately because overflow already guarantees zero fitness. Structural cloning avoids recursive `deepcopy` while preserving each backend's mutation semantics. Representative local microbenchmarks for the earlier optimization pass improved FNV 2-bit evaluation by about 47%, FNV reproduction by 40%, Nervous selection evaluation by 52%, SNN evaluation by 18%, and LUT evaluation by 12%; these are implementation checks, not universal throughput guarantees.
- **Nervous sampled-state fast path**: persistence targets still receive exactly the same half-tick logical samples, but fitness no longer constructs `T x cells` full-grid dictionaries. It reconstructs only candidate/output traces from the physical pulse intervals already emitted by the engine; equivalence tests cover uniform, paper-analog, and tri-circuit physics.
- **Substrate topology** (both original lattices): the hex (degree-3) and square (degree-4) grids are **bipartite** (2-colourable by `(x+y)` parity, hex girth 6), so a single circulating pulse can only traverse an **even-length** loop, and its output period is therefore even. Odd output periods are *not* impossible, but they cost far more: a period-*p* output needs a length-*2p* loop carrying two evenly-spaced pulses (`output_period = loop_length / n_pulses`), a conjunction the GA path dips through lower fitness to reach and empirically never crosses. This is a real design constraint, not a bug: a period-2 target (toggle) solves trivially, and the **pattern generator** is deliberately set to an even period: `Pattern (1000)`, period 4, one pulse in a length-4 loop, which the cheap single-pulse route reaches directly. The earlier odd-period `Pattern (100)` was retired precisely because parity made it topologically out of reach (neither a bigger grid nor hundreds of generations moved it off ~0.76). Autonomous targets that must run on **both** original lattices are chosen with this in mind. FNV keeps the spatially bipartite honeycomb, but its separate one-/two-tick delays, holds, toggles, and gated oscillators add temporal state at vertices, so an even spatial loop is no longer restricted to one tick per edge; this is how FNV escapes the original timing bipartiteness without inventing nonphysical connections.
- **LUT logic view** (`substrates/lut/boolfn.py`): every 16-bit lookup table is really a boolean function of the four neighbour input bits, so it is decoded to a minimised sum-of-products expression `out = f(N,S,E,W)` (verified exact over all 65 536 tables), so the genome reads as logic instead of hex, and the Growth tab shows the mature organism's distinct tables as actual 4x4 truth grids
- **LUT seed generation** (`substrates/lut/ontogeny.py`): a port of the reference
  `table_create` growth routine, which builds a dense genome as it goes,
  inventing a new rule whenever it meets a context it has not seen. This
  reproduces the varied shapes the `sim6` reference produces. It runs standalone
  as a shape browser (`py -m substrates.lut.ontogeny`) and is the *only* way LUT
  genomes are created. Every starting genome and every injected newcomer, in
  both the evolver and the Designer, comes from
  `substrates.lut.ga.make_seed_genome`, drawn from a cached pool so newcomers
  stay cheap. The LUT GA also has no preference for smaller genomes, which
  previously pruned these rich shapes back into sparse uniform diamonds.
  (The diagnosis behind that change: the dull diamond growth was caused by
  genomes being too *sparse*, not by the growth engine.)

### When progress stalls

After 12 generations without progress, nervous and LUT runs start generating
extra mutated descendants of the separately archived all-time best circuit.
Independently, the reserve described earlier keeps a bounded set of
complementary partial solutions available to breed from.

LUT runs using the experimental spatial I/O also propose compact port
arrangements and nearby one-bit rule changes at output cells.

The important constraint: every one of these proposals is an ordinary heritable
genome, grown by the normal rules and scored by the unchanged spec. Nothing here
injects a hand-built circuit or hands out privileged fitness.
The hard-target compiler rescue is available only when `UNRESTRICTED` is in the
selected LUT inventory, because its synthesized truth tables are not guaranteed
to belong to a named physical bank.
This makes a plateau diagnosable: a rescued genome demonstrates search
difficulty, while failure of a separately hand-designed circuit still points
to scoring or substrate representation.

### Current and compatibility I/O modes

The main app exposes only each substrate's current physical I/O:

- **Nervous and FNV**: evolved relative internal source pads plus globally
  fitted, distinct, read-only output probes. There is no placement dropdown
  choice for fresh runs.
- **LUT**: either evolved internal source pads or fixed alternating exterior
  perimeter buses, with the same global output fitter in both modes.
- **SNN**: the original fixed binding plus the retained `tag_rank`,
  `wiring_chromosome`, and `spatial_chromosome` experimental strategies.

The shared `io_placement` machinery remains for old checkpoints and controlled
programmatic comparisons. `spatial_chromosome` is still causal for SNN and
programmatic LUT runs: chromosome 3 encodes input seed cells and task-blind
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
| `ui/designer.py` | Manual circuit **designer**. Build a circuit by hand at either level: edit the genome (chromosomes/genes) and press Grow, or edit the grown grid directly (place cells, set routing states). Simulate and score it against any target; available as a GUI tab and standalone |
| `ui/diversity_ui.py` | "Diversity" tab. Analyses a complete evaluated population, solved or failed, showing the four-level collapse funnel, fitness summary, and optional mutational-robustness histogram. Runs on a worker thread with progress and Stop; nervous/LUT only |
| `ui/interactive.py` | "Interactive / Test" tab. After evolving or loading a circuit, drive its inputs and watch the response play out. Temporal SNN, nervous-net, FNV, and LUT runs load the exact scored case into an editable continuous-time pulse timeline - the Case dropdown offers whatever fitness actually grades, which for combinational targets on the asynchronous backends is the individual truth-table row in its own window, not just the multi-row schedule it is packed into; FNV displays its real fixed-function components and directed wires, and combinational FNV cases use the same circuit-derived settling window as fitness. SNN shows recurrent LIF voltage/spike playback in contract seconds, while static SNN truth tables retain input toggles |
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
| `substrates/nervous/targets.py` | Temporal target registry: eight hand-built timing banks plus the oracle-backed memory, interval, cadence, handshake, filtering, serialization, watchdog, and pulse-width families from `substrates/nervous/oracle.py` |
| `substrates/nervous/oracle.py` | Reference-state-machine targets + stimulus generator + held-out generalisation scoring |
| `substrates/nervous/contracts.py` | Serializable behavior-contract and constraint data plus concise target-definition helpers |
| `substrates/nervous/scoring.py` | The single `score_contract` evaluator: truth tables, event correspondence, phase-invariant state, bounded retention, complete intervals, cadence, alignment, and reports |
| `substrates/nervous/temporal.py` | Temporal observation harness: trial running, contract-scored output placement, and `prepare_net` |
| `substrates/nervous/tritile.py` | The paper's three-circuit tile (Fig. 2): 15-bit tile states (three 5-bit AND/OR-capable channels) expanded onto the unchanged pulse engine via pre-resolved sources; legacy 12-bit states migrate once |
| `substrates/nervous/analog.py` | Analog Fig. 1 node (charge / leak / comparator / hysteresis) with emergent coincidence, width and refractory; event-driven with analytical crossings |
| `substrates/nervous/certification.py` | The held-out verdict rule (CERTIFIED / OVERFIT / BELOW THRESHOLD / SOLVED / PLATEAU / UNCERTIFIED) shared by the GUI and `reproduce.py` |
| `substrates/nervous/diversity.py` | Population diversity: the four-level collapse funnel (identical genome -> identical function -> identical grown circuit -> off-spec behaviour) plus mutational robustness. It works on unsuccessful populations too; once everyone solves, structural variety replaces the now-zero fitness spread |
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
| `substrates/lut/ontogeny.py` | Port of the `sim6` `table_create` growth routine: shape browser plus the LUT GA's seed factory |
| `experiments/concept/` | Standalone proof-of-concept GA sims (hierarchical / coderack / multi-layer experiments) that predate the main backends |
| `ui/ui_compat.py` | Cross-platform Tk helpers (monospace-font probing, ttk theming) so the GUI behaves the same on Windows / Linux / macOS |

## Usage

```bash
pip install -r requirements.txt

python -m ui.app     # packaged launch
python app.py        # compatibility launch (also safe for existing shortcuts)
```

In the GUI: pick a **Model** (SNN / Nervous / FNV / LUT) and a **Target** from the dropdown
(use the category filter or type part of a name to search; click **Custom...** to
enter your own truth table), set Population / Generations / Restarts / Workers plus the
GA tuning row: **Mutations/child**, **Anneal alpha** (alpha < 1 cools the mutation rate
each generation for a hot-start simulated-annealing schedule; alpha = 1 is off),
**Plateau beta** (stagnation reheat strength; beta = 0 disables it, beta = 1 is default),
**Mutation cap** (the maximum effective mutations per child after the starting
rate, annealing, and plateau reheating are combined; default 8),
immigrants, tournament size and **Elites** (size of the elite *breeding pool*: the
top N genomes are the recombination parents for the next generation but are **not**
copied by the reproduction operator; after a terminal solve, evaluated parents may
survive environmental selection; 0 = breed from the whole population), then click
**Run**. **Chroms** is the exact chromosome count for the run (1-32), is preserved by
mutation, and is stored with the checkpoint. The fitness chart plots all-time
best, the best newly generated offspring before survivor selection, population
mean, and effective mutation rate (secondary axis). GUI defaults are Population
50, Generations 500, Restarts 1, Workers up to 8, Chroms 2, and Elites 5, with a hot-start
anneal (Mutations 4.0, alpha 0.997) so slow, steady progress is visible.

Circuits stop growing on their own, so there is no grid-size or iteration
control. **Max telomere** is the upper bound on that self-limit: each
chromosome carries a division counter that is effectively the circuit's growth
*radius*, and evaluation cost scales as roughly radius^2. LUT runs default to 8;
the other growing backends default to 20. Raise it only when a task genuinely
needs a larger circuit.

Use **Load Saved** to reload `results/best_genome.json` and **Save PNGs** to
export the growth and voltage figures.

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
> rows and evolve slowly. The framework supports them, but small targets are best for
> interactive runs.

For a run stuck at a genuine local optimum, the Escape panel offers **Lineage
walk**. It sets aside a configurable share of the population for lineages that
only mutate, one edit at a time, and are permitted to stay *worse* for several
generations. The point is that an edit which looks useless on its own is often
the prerequisite for the next one, and ordinary selection discards it before
that can happen. Anything that does improve rejoins normal breeding. It adds no
task-specific scoring and costs no extra evaluations.

Rebirth archives the generation-zero best and triggers after 15 flat
generations by default. Island migration ranks offspring that have actually been
evaluated, rather than inheriting a score from whichever genome previously held
that slot.

**Every escape mechanism is off by default**, in new GUI sessions and in headless
runs alike; **Reset escape** restores that baseline. Treat the 10% lineage share
as a starting value only. An early screen of these mechanisms compared raw
scores at a single seed and short budget, without measuring whether anything
actually solved, so it is not evidence that they help. The benchmark section
below explains why a score gathered that way says very little.

### Local terminal benchmark

`tools/benchmark.py` runs selectable architecture x target matrices through
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

**Read a solve rate next to its budget.** The report prints a
`Solve gen (min/med/max)` column and a **Budget caveats** section that flags a
sweep still producing first-time solves at the cutoff (rate is a lower bound)
or one that never solved at all (bounds nothing, since "unreachable" and "needs a
longer run" are indistinguishable). `--gens 40` is a fast smoke default, not an
operating point: the application itself runs 500 generations, and FNV Full
adder first solves anywhere from generation 18 to 243. Size `--gens` past a
target's known solve-generation spread before reading a plateau as a limit of
the substrate.
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
substrate can **evolve** useful async circuits, so the substrate's clock-freedom
and the headline results are both runnable, not asserted.

```bash
py tests/run_tests.py          # whole suite, no pytest needed (py -m pytest also works)
py reproduce.py                # list reproducible claims
py reproduce.py c_element      # evolve one claim from a fixed seed, certify on held-out
py reproduce.py async_substrate  # the metamorphic "no hidden clock" audit
```

The current suite contains **573 tests across 28 test files**.

- **`tests/`**: a metamorphic **synchrony audit** (`test_synchrony.py`) that
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
- **`reproduce.py`**: each *claim* is a seeded evolution plus a certification
  gate: for input-driven targets it re-samples fresh held-out schedules and
  reports **CERTIFIED** (generalises), **OVERFIT** (memorised timing),
  **BELOW THRESHOLD**, **SOLVED**, **PLATEAU** (a documented substrate limit),
  or **UNCERTIFIED**, as appropriate.
- **Automatic certification**: every supported GUI/controller run of an
  oracle-backed temporal target certifies its winner when it finishes: the
  verdict is shown in the status line and saved into the checkpoint
  (`substrates/nervous/certification.py`, one shared rule with `reproduce.py`).
  Asynchronous truth tables are replayed exhaustively under freshly shuffled
  row schedules and require held-out fitness >=0.999, catching circuits that
  memorized schedule position instead of computing the presented row.
  LUT exterior-edge runs are the current exception because their frozen
  outside-to-facing-edge validation adapter is still pending; they report
  **UNCERTIFIED** rather than presenting an unaudited held-out score.

## Requirements

- Python 3.10+
- numpy
- matplotlib
- pytest is **optional**; the suite also runs with bare Python via
  `py tests/run_tests.py`

## Concepts & provenance

The architectures and mechanisms trace to Andrew Edwards' *Evolvable Hardware* work on **grown**, indirectly-encoded circuits; the code cites specific figures inline where they apply:

- **Nervous net**: the honeycomb "nervous net" of Edwards **EH'02**: three independently configured directional circuits per tile (Fig. 2), the 16 coincidence/veto states per circuit (Fig. 3), and context rotation (Fig. 4). The temporal/oracle target suite is layered on that substrate.
- **LUT array**: the paper's **Architecture 2**, ported faithfully from the `sim6` reference implementation and the degree-4 case of `automaton_arrays` (4 neighbours x four 16-state lookup tables per cell; the `table_create` growth routine in `substrates/lut/ontogeny.py`).
- **Substrate framing**: identical tiled cells, routing SRAM or lookup tables
  as the reconfigurable element, and a genome that maps `context -> new state`
  all correspond to the papers' hardware picture. Code comments name the
  matching hardware element at each point.

Two threads are this project's own, built on that base.

The first is **timing and memory circuits** (SR latches, oscillators, rate
controllers, gated memory) scored on when edges occur, rather than only
combinational truth tables.

The second is a set of **search mechanisms borrowed from biology**: growth
bounded by a per-lineage division counter, a mutation rate that rises when
progress stalls and decays otherwise, SNN's preference for smaller circuits once
a task is solved, and the connectivity-and-feedback tie-break used by Nervous
and FNV. The section comments in `substrates/nervous/ga.py` and
`substrates/nervous/temporal.py` explain each in detail.
