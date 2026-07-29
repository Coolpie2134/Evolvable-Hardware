# Evolvable Hardware — Technical Design Documentation

Circuit morphologies and ontogenies, after Edwards, *Evolvable Hardware* 2002
(DOI [10.1109/EH.2002.1029891](https://doi.org/10.1109/EH.2002.1029891)).

## 1. Overview

The project tests whether a developmental (indirect) encoding can evolve useful
asynchronous circuits. A genome is not a netlist; it is a set of
context-to-state rules that grow a circuit from one or more seed cells. A cell
reads its neighbours, matches that context against its genes by minimum Hamming
distance, and adopts the winning gene's output state. The circuit is the fixed
point of that process. The genetic algorithm never edits a circuit, only the
rules that build one.

Four substrates are implemented. All share the genome shape (chromosomes of
context-matched rules, telomere-bounded growth) and the same GA. They differ in
what a grown cell is and how signals move through it.

| Architecture | Module | What a cell is |
| :---- | :---- | :---- |
| Nervous net (Arch 1) | `substrates/nervous/` | Hex lattice, 3 neighbours (L/R/D). A cell is a pulse node: coincidence or buffer, plus inhibition. Continuous-time, edge-triggered pulses. A tile carries either one circuit (legacy profile) or the paper's three independent L/R/D circuits (tri profiles), under digital or analog node physics. See 2.2. |
| Functional NV Net (FNV) | `substrates/fnv/` | Hex lattice, 3 neighbours and two antiparallel wires per physical edge. A cell is one fixed routed component: logic, delay, normalizer, hold, C-element, toggle, or enable-gated oscillator. Continuous-time event-driven logic. |
| LUT array (Arch 2) | `substrates/lut/` | Square lattice, 4 neighbours (N/S/E/W). A cell holds four 16-bit lookup tables, one per output direction. Continuous-time asynchronous level logic. |
| SNN (comparison) | `substrates/snn/` | Square lattice. A cell is a leaky integrate-and-fire neuron with an evolvable threshold and time constant, wired by excitatory and inhibitory synapses. |

The nervous net and the LUT array are the paper's two architectures and the
focus of this work. FNV is a deliberate standalone extension that retains the
developmental honeycomb while replacing the uniform pulse node with a bank of
physically realizable fixed components. The SNN is a comparison backend: same
indirect encoding, but a neuron model rather than the paper's hardware. The
Designer and Diversity tabs are hidden for SNN and FNV runs.

## 2. The four architectures under the hood

### 2.1 Shared developmental core

Every backend follows the same developmental outline
(`substrates/nervous/nervous.py: grow_nervous`,
`substrates/fnv/growth.py: grow_functional`,
`substrates/lut/lut.py: grow_lut`, and the `substrates/snn` equivalents):

1. **Seed.** The target's input terminals are placed as live cells in the seed
   state. Nothing else exists. The field is unbounded; there is no grid-size
   clip.

2. **Lookup.** Each live cell and each empty frontier cell builds a context (its
   neighbours' states plus its own) and finds the closest gene. The original
   substrates use Hamming distance. FNV uses a categorical physical distance,
   because its permanent type numbers are labels rather than bit fields. The
   winning gene's `self_out` becomes the cell's next state.
   `self_out == 0` means death, or a dead direction.

3. **Division is Hayflick-limited.** Every chromosome carries an evolvable
   telomere. Seed cells start at the germline length `L = germline_telomere(genome)`
   (the maximum across chromosomes); a daughter inherits parent − 1. At 0 a cell
   is senescent: alive and functional, but unable to birth a neighbour. Growth
   halts at radius `L` from the seeds, so the genome bounds its own body size.

4. **Settle.** Lookups continue until the state map reaches a fixed point or a
   2-cycle ("the growth stops due to the fact that all queries return the same
   value out as was passed in"). The iteration budget derives entirely from `L`
   (`_grow_budget(L) = 3L + 6`), not from a user setting. `grid_size` and `iters`
   are accepted but ignored for the nervous net and FNV; they survive for old
   callers and pickles.

The mature grid is then interpreted into hardware, which is where the
architectures diverge.

**Lookup cost.** The context match dominates growth: at 32 chromosomes it is 95%
of nervous growth time, and 99% under the tri tile. The four context fields
occupy disjoint bit ranges, so their summed Hamming distance equals the popcount
of a single packed XOR. `_compile_lookup` packs each gene's context once per
growth run and the inner loop becomes one XOR plus `int.bit_count()`, preserving
chromosome-major first-wins tie-breaking. Measured end to end: 2.8× faster
single-tile and 13.6× faster tri-tile at 32 chromosomes, with grown grids
bit-identical over a 320-genome corpus. The packed value is also the growth
cache key, so the field widths are derived from the state alphabets
(`MAX_STATE`, `TRI_STATE_MAX`) rather than hard-coded: a field narrower than its
alphabet would alias two distinct contexts onto one cache entry.

### 2.2 Architecture 1 — the nervous net (`substrates/nervous`)

A hexagonal lattice. Each tile touches exactly three neighbours, labelled L, R
and D, matching the paper's Figure 3 cell, which contains three independent
circuits. A cell's 5-bit state (0–31) is not a number the physics reads; it is an
index into a fixed routing table (`substrates/nervous/hexgrid.py: ROUTING_HEX`) naming which
neighbours feed the node's two excitatory inputs and its one inhibitory input.
`interpret_nervous` performs that decode. Under the tri-circuit profiles a tile
carries three such indices packed into 12 bits, one per output direction; see
"The three-circuit tile" below.

One geometric consequence matters: the honeycomb is bipartite, since every edge
joins an (x+y)-even cell to an odd one. Every cycle therefore has even length,
and a single circulating pulse can only produce an even period. Odd-period
patterns need two pulses half a loop apart, which the GA empirically never
reaches. This is why the default pattern target is 1000 (period 4) rather
than 100.

#### The four node types

Every routing state decodes to `(E1, E2, I1, op)`: two excitatory source
directions, one inhibitory source, and the combining operation. The node type
falls out of that tuple (`hexgrid.routing_kind`):

| Node type | Routing shape | Behaviour |
| :---- | :---- | :---- |
| Off (dead) | `E1 = None` (state 0) | No wire, no output. The cell exists on the lattice but drives nothing and can never fire. Growth uses this as death. |
| Buffer | `E1 == E2`, no `I1` (states 1–3) | A single connection; one edge on that input triggers it. This is the delay element. Chains of buffers are delay lines, and loops of buffers are the substrate's only memory: a pulse circulating until inhibited. |
| Coincidence | `E1 != E2`, no `I1` (states 4–6) | Fires only when edges arrive on both excitatory wires within the coincidence window `COINC`; "neither input alone can trigger a response". This is what makes timing computable, since a delay line meeting a direct line at a coincidence node detects a specific gap. |
| Inhibited | `I1` present (states 7–15) | A buffer or coincidence node with a veto. If the inhibitory wire is high when the node would trigger, the response is suppressed. This is how a loop is stopped, how a signal is gated, and how "pass A unless B" is built. |

**The OR extension.** States 0–15 are the paper's AND (coincidence) semantics
verbatim. States 16–31 are OR twins: identical routing, but the node fires on
either excitatory input. For a buffer (`E1 == E2`) or the off state, OR and AND
coincide, so those twins are benign aliases; the new capability is the OR of a
coincidence pair, where either of two different lines activates the node. The
extra bit is 0 for every pre-existing genome, so old genomes grow and behave
bit-identically.

#### Pulse physics

`substrates/nervous/pulse.py: PulseSim` is an event-driven, continuous-time simulation
rather than a clocked loop:

* A wire (each cell's output net) is idle or carries a pulse over the half-open
  interval `[start, end)`.
* All action is precipitated by a leading edge. A held-high input is one long
  pulse with one edge, not a train of edges.
* A triggered node emits a pulse of width `WIDTH` after a fixed delay `DELAY`,
  and is refractory until its own pulse ends (comparator hysteresis, so no
  chattering).
* External inputs are injected onto the input cell's net (wired-OR, as if
  driving the physical perimeter wire).
* There is no spontaneous activity. Every input comes from a neighbour, so with
  no external input the array stays all-zero. This is an audited invariant
  (`tests/test_synchrony.py`).

The physical constants live in an immutable
`PulseConfig(delay, width, coincidence, model, event_cap)` carried with the run
rather than in module globals, so a worker process cannot silently score under
different physics. With `DELAY = WIDTH = TICK` and `COINC < TICK`, behaviour on
the integer lattice matches the old synchronous engine exactly; that engine was
the quantization of this one.

#### The node engines

The paper's node regenerates one fixed pulse width after one fixed delay, so an
input pulse's length is discarded the moment the first node fires (every
internal wire has a single driver, so no merge restores it).
`PulseConfig.model` selects the node engine: two digital abstractions of the
paper's node, and one model of its analog circuit.

| Model | Emitted pulse | What evolves |
| :---- | :---- | :---- |
| `uniform` (default) | width `WIDTH` after delay `DELAY`, every node | Routing only. This is the paper, and it is byte-identical to the engine before the variants existed. |
| `pulse_delay` | the incoming waveform, transported: `[t, t+w)` becomes `[t+d, t+d+w)` | Routing plus `Genome.state_delays`, per-routing-state delay multipliers in [0.25, 4.0]. Width is preserved rather than regenerated, so pulse duration carries information. |
| `paper_analog` | emergent: the comparator holds its output while the node sits below threshold | Routing only. Coincidence, width and refractory follow from the analog constants rather than from knobs. A separate engine (`substrates/nervous/analog.py`); see "The analog node" below. |

The two digital models coincide when every pulse is one `WIDTH` wide and the
delay vector is neutral, so each is a strict superset of the paper. Node delay is
a physical property of a node type, orthogonal to routing, so growth is
untouched: `nervous.node_delays(genome, grid, config)` builds the per-cell map at
scoring time and returns `None` off its model. That gate lives inside the helper,
so every scoring path is consistent by construction.

A third model once made the emitted width itself a per-node-type genome vector.
Width evolution has been retired: the genome no longer carries a width vector,
`evolved_width` is not a constructible model, and a checkpoint saved under it
loads on the fixed-width `uniform` node (see 4.7).

**Refractory divergence** matters when comparing models. A node is refractory
for its own output duration. Under width preservation that duration follows the
input, so a `pulse_delay` node transporting a wide pulse is busy for `delay + w`,
while a `uniform` node is busy only for `DELAY + WIDTH`. On stimulus wider than
`WIDTH` (for example the mixed 0.5–2.25 s oracle banks) a neutral
width-preserving genome can drop a following edge that `uniform` passes. This is
deliberate physics, and evolution recovers the edge by shortening the node's
delay, but neutral `pulse_delay` starts measurably harder than `uniform` on
wide-pulse banks.

#### The three-circuit tile (`tri3`)

The paper's Figure 2 tile is not one circuit but three: "each node contains
three nervous network circuits. Each circuit receives inputs from three
directions (left, right, and down) and sends outputs to the same three
directions." The legacy interpretation collapses that to one circuit driving one
output net that every listening neighbour reads, so a tile can only broadcast a
single state. `substrates/nervous/tritile.py` implements the tile as written.

A tri tile's state is 12 bits: three independent 4-bit routing configurations,
`chanL` in bits 0–3, `chanR` in 4–7, `chanD` in 8–11 (`channel_configs` /
`pack_channels`), seeded at `TRI_SEED_STATE = 0x111`, all three channels
buffer-D, giving a live signal-passing tile. Each channel is a Figure 3 circuit
with the paper's pure coincidence semantics; the OR twins are a single-tile
extension with no place in the faithful tile.

Rather than re-derive the event engine, `interpret_tri` expands a grown tri grid
into ordinary single-circuit sub-nodes, one per (tile, output direction), and
resolves the cross-tile wiring: the signal arriving on tile P's input direction
`e` is the output of the neighbour `Q = hex_dirs(P)[e]` whose own output points
back at P (`back_dir`). Those pre-resolved sources are handed to the unchanged
`PulseSim` through its `sources=` hook, so the tri tile inherits the audited
pulse physics rather than re-implementing it.

**Readout.** The target schema names an output tile, not one of its three pins,
so scoring observes a normalised wired-OR union of that tile's three output
wires. `TriSim` presents tile-keyed `rise_times`, `pulse_intervals` and
`activity_at`, so the temporal scorer needs no tri-specific branch. The union is
an external readout convention, not a fourth circuit inside the tile.

Elsewhere: growth matches 12-bit states by Hamming distance; mutation flips one
of 12 bits, and because the three channels are disjoint fields a flip always
lands inside a single channel, giving per-channel mutability for free;
`Genome.arch` is part of the genome signature, so the same integers never share
a fitness-cache slot across architectures; and the GA's loop bonus is measured on
the expanded sub-node graph (`_loop_bonus_tri`), because a tile's three circuits
form loops that tile-level routing cannot express. The delay vector is a
single-tile node-type feature and is not consulted under `tri3`. A capability
test proves one tile routes two signals to two outputs independently, which a
single broadcast state cannot do (`tests/test_tritile.py`).

#### The analog node (`paper_analog`)

The digital engines regenerate a fixed-width pulse after a fixed delay and treat
the coincidence window as a hard rectangle. The real circuit (Figure 1) is
analog: excitatory edges couple capacitively onto a high-impedance comparator
node, a Vbp-controlled transistor leaks that node back toward Vdd, and the
comparator, with a little hysteresis, trips when the node crosses threshold.
`substrates/nervous/analog.py: AnalogPulseSim` models that mechanism directly.

Voltages are normalised (Vdd = rest = 1.0, ground = 0). The node idles at rest;
each excitatory terminal edge steps it down by `step`; between edges it recovers
as `v(t) = rest + (v0 - rest) * exp(-(t - t0) / tau_leak)`. Because the steps are
instantaneous and the recovery is monotone-up, a downward threshold crossing can
only happen at an edge. Firing is therefore decided at edge times with no
inter-event root-finding, and the output fall is solved analytically and
rescheduled, inertial-delay style, whenever a later edge extends the pulse.

Three behaviours the digital node hard-codes as independent knobs are
consequences of the same physical constants here:

| Behaviour | Digital node | Analog node |
| :---- | :---- | :---- |
| Coincidence | the `COINC` window, a hard rectangle | one edge steps the node by `step < (1 − Vth)`, so it cannot trip alone; two within a leak time constant sum past threshold. The window width follows from `tau_leak` and `step`, and `COINC` is not read at all. |
| Buffer vs coincidence | the routing op | a buffer ties both terminals to one source, so a single source edge delivers `2 × step` and fires alone. The paper's buffer-vs-coincidence distinction falls out of the wiring. |
| Output width | fixed `WIDTH`, every pulse identical | however long the node sits below threshold. A further edge during the pulse pushes it deeper and stretches the output, so the node is paralyzable and width becomes input-dependent. |
| Recovery / refractory | a fixed lock-out after the emitted pulse | the hysteretic comparator trips at threshold and re-arms only at threshold + hysteresis, so the pulse and its recovery are one physical state transition rather than a pulse plus a hidden timer. |

The constants live on `PulseConfig` (`analog_threshold`, `analog_step`,
`analog_tau_leak`, `analog_hysteresis`) and are validated as a coupled set,
because they only describe a working node together: `0 < Vth < 1`,
`(1 − Vth)/2 < step < (1 − Vth)` so that two terminal edges fire and one does
not, `tau_leak > 0`, `hysteresis >= 0`, and `Vth + hysteresis < 1`.
`simulation.create_simulator` maps the run's propagation delay onto the node's
delay and dispatches to the analog engine. `COINC` and output width are
deliberately not mapped in, since under this model they are outputs of the
physics rather than inputs to it. `PulseConfig.width` still sets the default
external stimulus length.

Figure 1's I1 input controls the leak/bias branch. Without transistor parameters
the portable model cannot claim a calibrated analogue conductance, so active I1
is represented as a fast reset-to-Vdd clamp at an excitatory edge: a qualitative
circuit approximation, stated as such, rather than an invented graded voltage.
`AnalogPulseSim` mirrors `PulseSim`'s external surface, so it is a drop-in for
scoring and playback and composes with the tri tile through the same `sources`
hook. `tests/test_analog_reference.py` validates it against an independent Δt
numerical integrator on E1/E2 separation, dense pulses, output width and
recovery.

#### NV run profiles: what a new run may select

**There is one profile.** `runtime/config.py: NV_NEW_RUN_PROFILES` declares it,
`validate_new_nv_profile` rejects anything else, and `nv_run_config()` builds it:

| Profile (GUI label) | `tile_arch` + `node_model` |
| :---- | :---- |
| Analog tri-circuit (3-output, paper Fig. 1 node) | `tri3` + `paper_analog` |

This is both the most physically faithful configuration — the paper's tile on
the paper's node, with coincidence window, output width and refractory all
emergent rather than knobs — and the best measured. The single-tile
(`single` + `pulse_delay`) and digital tri-circuit (`tri3` + `uniform`) profiles
are RETIRED from new runs.

**The ablation that retired them.** The digital tri profile existed to separate
the two ways the analog profile differed from the single-tile baseline — tile
topology and node physics — so that a win could be attributed to one of them.
Measured over 8 targets × 2 seeds, 40 generations, population 30:

| Profile | tile | node | mean | solved | seconds |
| :---- | :---- | :---- | ----: | ----: | ----: |
| legacy | single | digital | 0.9097 | 4/8 | 88 |
| digital_tri | paper | digital | 0.8823 | 3/8 | 180 |
| analog_tri | paper | analog | **0.9554** | **5/8** | 155 |

The paper's tile ALONE does not help — the digital tri profile scored *below*
the single-tile engine. The analog node physics is what does the work.

**Held-out certification decided it**, and it matters more than the training
means above. Five seeds each:

```
SR latch   legacy      mean 0.794   1/5 CERTIFIED
           analog_tri  mean 0.936   3/5 CERTIFIED
Toggle     legacy      5/5 train 1.000 but 3/5 OVERFIT (2/5 certified)
           analog_tri  5/5 train 1.000 and 5/5 CERTIFIED
```

Legacy "solves" Toggle on every seed, and three of those five are memorised
timing that does not generalise. The digital engine's fixed pulse width and
rectangular coincidence window are exploitable timing invariants; the analog
node has no fixed rectangle to memorise, so a circuit has to work by real
dynamics. For a project whose credibility gate is generalisation (5.5), that is
the decisive difference.

**What it costs.** Two of 31 targets become unavailable — Odd pulse selector and
Pulse width sum, both declaring `supported_models=('pulse_delay',)` because they
are waveform-DURATION contracts needing width-preserving transport. The target
picker filters them out automatically. Runs are roughly 1.8× slower than the
retired single-tile engine.

The retired engines remain in the codebase as reference implementations, audited
by `tests/test_pulse_models.py` and `tests/test_node_contracts.py`; they are
simply no longer offered for new runs. `GAConfig`'s own field defaults still
describe the single-tile engine because that class doubles as the checkpoint
deserialisation target and is constructed partially throughout the tests —
shifting those defaults made every partially-built config an invalid
tri3/pulse_delay pairing, so the live profile comes from `nv_run_config()`
instead.

Under the analog profile the app relabels the physics fields (Delay becomes
"Propagation delay", Width becomes "Input width"), disables Coinc because
coincidence is emergent, and shows the analog constants in their own row. A
checkpoint saved under a retired pairing still loads for playback, with a
status-line warning that new runs use one of the current profiles.

#### 2.2.1 The canonical alphabet

The 5-bit configuration register has 32 settings but only **22 distinct
circuits**. A routing whose two excitatory inputs are the same cell just relays
that one line, and `AND(x, x) == OR(x, x)`, so the OR twin of every buffer — and
of "off" — is the identical circuit to its AND original:

```
dead <- 0, 16    buffer D <- 1, 17   buffer R <- 2, 18   buffer L <- 3, 19
buffer R veto L <-  7, 23   buffer D veto L <-  8, 24   buffer L veto R <-  9, 25
buffer D veto R <- 10, 26   buffer L veto D <- 11, 27   buffer R veto D <- 12, 28
```

Only the six genuine coincidence routings (4, 5, 6, 13, 14, 15) differ from
their twins, giving 12 distinct coincidence circuits.

Left unmanaged that is a **2:1 prior against coincidence** — the substrate's only
computational primitive. Drawing a configuration uniformly over the 32 encodings
makes every buffer, and death, twice as likely as any coincidence detector;
measured over 60k random genes, buffers landed at ~6.2% each against ~3.1% for
each coincidence. It also let genomes accumulate alias encodings, so two genes
building the identical circuit displayed as different node types on the Genome
tab, and a gene emitting state 16 displayed as a node type that is really a dead
cell — a "node" that can never appear in the grown net.

Genomes therefore carry canonical states only (`hexgrid.CANONICAL_STATES`,
`canonical_state`). Gene generation draws over *circuits*, and the register is
still physically 5 bits with single-bit-flip mutation — the flip is simply
normalised onto the canonical representative afterwards. Two rules matter:

* `terminal_nodes` binding keeps 16 and 17 drawable. There they are not aliases
  but the dedicated input / output node types, and normalising them away would
  erase every terminal an organism has grown.
* the flip must land on a **different circuit**. Canonicalising a raw flip is not
  enough: flipping a buffer's AND/OR select bit produces that buffer's own alias,
  which normalises straight back, silently turning about a fifth of state
  mutations into no-ops — and a no-op mutation lets a multi-event transaction
  cancel back to an exact copy of its parent, which reproduction relies on never
  happening.

**This is a representation fix, not a search improvement.** Measured A/B over 24
runs (6 targets × 4 seeds, 40 generations, population 40): legacy mean 0.8913,
canonical 0.8794 — better on 5, worse on 8, unchanged on 11. The likely reading
is that the alias states were supplying free neutral drift, genotype changes with
no phenotype change, which is exactly how this substrate crosses plateaus. Do not
cite the canonical alphabet as a solving improvement; it removes a
misrepresentation, and it costs a neutral network to do so.

### 2.3 Architecture 2 — the LUT array (`substrates/lut`)

A square, 4-neighbour lattice. Each cell holds four 16-bit lookup tables, one per
output direction (N/S/E/W). A table is indexed by the four neighbour input bits
(index bits 1/2/4/8 for N/S/E/W), so a cell's output in one direction is an
arbitrary boolean function of its four inputs. That is the whole hardware: the
field is uniform, every cell is the same silicon, and only the table contents
differ. `substrates/lut/boolfn.py` decodes a 16-bit table back to a readable
sum-of-products so the genome and net can be shown as logic.

The gene is Figure 10 verbatim: a context of five 16-bit LUT states (the four
neighbours' facing tables plus the cell's own table for that direction) mapped to
a new 16-bit table, chosen by minimum Hamming distance over the full 80-bit
context. During growth each direction is looked up with the context rotated to
that direction. `self_out == 0` (the all-zero table) means that direction is
dead; a cell with all four tables zero is removed.

**Dynamics.** `substrates/lut/pulse.py: AsyncLutSim` is continuous-time asynchronous
level logic with inertial delay. A cell re-evaluates its tables when a neighbour
changes and the new value lands one gate delay later, but a pending change
cancelled before it lands never appears, because a real gate's output node
cannot follow a blip shorter than its delay. Because it is level logic, pulse
width matters everywhere natively, with no model variants needed. On the integer
lattice with `delay == TICK` a vectorised fast path reproduces the old
synchronous engine (`LutSim`) bit for bit; `LutSim` is retained as the
quantization reference.

Two consequences are worth knowing. The array is spontaneously active: a table
with its index-0 bit set outputs high for the all-zero neighbourhood, so it fires
at power-on with no input, at exactly `t = 0`. That is honest LUT physics and the
deliberate contrast with the nervous net's quiescence. Second, it is a chaotic
recurrent system that generally does not settle, so combinational scoring must
require settling and in practice caps at constants. LUT is a temporal-target
substrate.

### 2.4 Functional NV Net (`substrates/fnv`)

FNV is its own substrate rather than another nervous-net node model. It retains
the degree-3 honeycomb, local L/R/D orientation, indirect context-to-state
development, unbounded field, and telomere bound. Its mature hardware and state
alphabet are independent.

Each physical adjacency contains two separately driven antiparallel wires. A
component only reads the directions declared as its inputs and only drives its
declared outputs. Two-input components use two sides and drive the remaining
side. Unary components have all three input orientations; for each orientation
there are two one-output routes and one two-output route. Asymmetric VETO has
both A/not-B assignments for each possible output direction.

The initial append-only catalogue has 118 permanent IDs:

* `EMPTY`;
* AND, OR, XOR, and VETO routing forms;
* exact one- and two-tick transport delays;
* `NORMALIZER1/2`, which emits one canonical-width pulse per inactive-to-active
  input episode;
* `HOLD1/2`, which extends an input after its falling edge;
* Muller C-elements and rising-edge toggles; and
* enable-gated oscillators with independently fixed one- or two-tick high and
  low periods.

There is deliberately no NOT, NAND, NOR, free oscillator, per-node numeric
parameter, filter, tap delay, or reset toggle. Functions already expressible by
the logic catalogue are not duplicated. Every timing choice and route is a
separate type ID so a future physical realization can map a genotype directly
to a component inventory. Checkpoints carry a SHA-256 catalogue hash; changing
the type-to-ID mapping makes them fail to load rather than silently becoming a
different circuit. They also carry an FNV development version, so a change such
as the polarized germline policy cannot silently reinterpret an older genome.

`FunctionalSim` is a continuous-time event engine. Logic transitions are
inertial; delays transport both edges; stateful elements retain only their
defined internal state. Every type is quiescent at all-zero power-on, including
the gated oscillators. Inputs are the only external sources and output probes do
not drive the circuit. An input pad is explicitly source-only: it ignores
internal component evaluations and changes state only when its corresponding
external pulse is injected. It may therefore drive adjacent receivers but
neighbouring logic can never feed back into or reactivate it.

New FNV genomes carry one discrete relative honeycomb coordinate per logical
input. Input 0 is anchored at `(0, 0)` as a coordinate gauge because absolute
translation has no behavioral meaning on the unbounded field. Other pads begin
in a compact collision-free neighbourhood and mutate by one physical edge at a
time inside a bounded placement domain. Crossover inherits the entire layout as
one co-adapted module rather than recombining individual pads and manufacturing
collisions. The pad positions are germline seeds, so they affect development as
well as runtime injection. Legacy checkpoints that lack this field continue to
use the target-declared fixed inputs.

The first four logical inputs retain fixed, distinct dual-output delay-component
identities during growth. This is physical port polarization, not an evolvable
per-component parameter: external injection drives every pad identically, while
the distinct catalogue states let the developmental rules distinguish A, B,
select, and a fourth input instead of imposing an accidental reflection
symmetry on asymmetric circuits.

FNV output terminals are fitted read-only probes. The fitter scores every mature
non-input component rather than a radius-limited neighbourhood around the
target's display coordinate. For multiple roles, a dynamic-programming
assignment maximizes their joint score subject to using distinct components;
this avoids a greedy early role consuming the only strong cell for a later
role. Training fixes both the evolved input layout and the selected output
cells, and held-out certification reuses those exact bindings without refitting.

An FNV run enables whole component families, never individual types or
parameters. Initialization chooses a family before a member so the 36 oscillator
routes do not outweigh a three-entry family. Mutation normally stays in the
current family and favors the nearest routing/timing form; rarer cross-family
changes preserve exploration. The native one-component baseline shown in the
report is informational only: it never rejects a target or prevents evolution
from discovering a harder emergent circuit.

FNV initialization is strictly forward-developmental. Every initial candidate
and random immigrant is an unrestricted random `FunctionalGene` genome drawn
from the enabled component families plus a compact random relative input
layout. The ordinary growth engine alone determines its mature body. The
evolutionary path contains no component-grid scaffold, target-shaped route
planner, phenotype-to-genome translation, or special timing-motif coverage
bank.

FNV reproduction uses expression-aware mutation. Rules that win a lookup in the
mature body or its frontier receive most mutation events, with a minority left
fully unrestricted so dormant rules and new morphologies remain reachable.
Input-layout mutations are interleaved with those rule edits, so placement and
morphology can co-adapt. Combinational runs further favor expressed
LOGIC/C-element outputs. Incomplete
phenotypes still emit a zero for every declared contract case, keeping FNV case
vectors rectangular and making ε-lexicase safe. `tools/benchmark_fnv.py` runs
deterministic, process-isolated target and seed comparisons through the real
controller; it can survey every FNV-supported temporal target, select tournament
or lexicase, and show case vectors.

FNV selection does not rank gene count, chromosome length, body size, or
telomere cost. After fitness, robustness, and juvenile-development score, exact
ties are separated by `FunctionalTopology`, computed from the same effective
directed wires as `FunctionalSim`. A graph traversal starts at every source-only
input pad. It counts reachable non-input components and wires, components
reachable from at least two logical inputs, cyclic nodes, independent cycle
rank, and separate strongly connected feedback regions. Each term enters as
`log(1 + count)`, so more connected structure and feedback are always preferred
but returns diminish. Unreachable components, disconnected islands, and loops
that no input can activate contribute zero. No truth table, expected event,
component family, fitted output, or target name enters this topology objective.
Tournament selection and survivor ranking use it as their final key; ε-lexicase
uses it only after behavioral cases have filtered the acceptable parent set.

### 2.5 The SNN comparison backend (`substrates/snn`)

The same indirect encoding grows a square grid of leaky integrate-and-fire
neurons. A cell's state selects a threshold level and a membrane time constant
from the `Arch` substrate table (`vth_levels`, `tau_levels`, `syn_weight`);
neighbouring cells are wired by excitatory or inhibitory synapses.
`substrates/snn/lif_sim.py` is itself event-driven: synapses emit rectangular current
pulses and, between event times, each non-refractory membrane has a closed-form
exponential solution, so there is no integration step or global update clock. The
sampled `simulate_trace` output exists only for the GUI's voltage plots.

Static SNN truth-table runs use the fixed 20 ms horizon (`SIM_TIME`) and retain
the original feed-forward graph. Temporal runs use the target's horizon, inject
each external leading edge at its exact floating-point time, convert contract
seconds to physical LIF milliseconds at 4.8 ms/s, and set
`Arch.recurrent=True`, which adds the reverse of every physical adjacency.
Output spike times and unit-width spike intervals enter the same Contract v1
scorer used by nervous and LUT circuits. A per-trial event cap makes pathological
feedback fail deterministically. The GA constants remain SNN-specific.

## 3. Genome translation for all models

Genotype to phenotype is the same three-step pipeline everywhere: genes, grown
state map, interpreted hardware. Only the alphabet changes.

|  | Nervous net | FNV | LUT array |
| :---- | :---- | :---- | :---- |
| Gene | `HexGene(ctx_l, ctx_r, ctx_d, self_in, self_out)`: four 5-bit context fields plus output state | `FunctionalGene(ctx_l, ctx_r, ctx_d, self_in, self_out)`: five permanent component IDs | `LutGene(ctx_n, ctx_e, ctx_s, ctx_w, self_in, self_out)`: five 16-bit LUT-valued fields |
| State alphabet | 0–31 (5-bit): 0 dead, 1–15 paper routing, 16–31 OR twins. Under `tri3`, 12-bit: three packed 4-bit paper configurations (0 = dead channel), no OR twins. | 0–117: 0 empty, 117 fixed component-and-routing types | 0–65535 (16-bit table): 0 = dead direction |
| Context match | min Hamming over (L, R, D, self) | categorical physical distance over family, function, routing, and fixed timing | min Hamming over the full 80-bit rotated context |
| Growth rule | `self_in == 0` matches empty cells, the only kind that can birth a cell | same empty-cell convention | same convention; roughly 1 in 65536 by chance, so random genomes are seeded with them explicitly |
| Interpretation | `interpret_nervous`: state → `ROUTING_HEX[state & 0x1F]` → `(E1, E2, I1, op)` | catalogue ID directly names the physical component and ports | the four tables are the hardware; no decode step |
| Extra vectors | `state_delays` (32 floats, model-gated, not read during growth) | — | — |
| Tile architecture | `Genome.arch`: `'single'` (one Fig. 3 circuit per tile) or `'tri3'` (three independent L/R/D circuits). Part of the genome signature, so the two never share a cache slot. | one component per honeycomb vertex | — |

**Chromosomes.** A genome is a list of chromosomes; each has genes, a split point
(the crossover boundary), a tag (used to pair homologs), and a telomere. Gene
count is capped at `MAX_GENES = 24` for Nervous and LUT and 64 for FNV; the
higher FNV ceiling lets a reverse-developed multi-component motif preserve its
local context rules. Random FNV chromosomes still start with 3–12 genes.
Chromosome count is capped at 32. The app's "Chroms" field is a structural
constraint that reproduction enforces exactly, not an initial-population hint.

**SNN.** `Gene(state_n, state_s, state_e, state_w, self_in, self_out, limit)`
over a 16-state alphabet, with `self_out >= 1` (no death state) and a vestigial
per-gene `limit` retained only for old pickles. Interpretation reads the state as
a threshold/tau index plus synapse polarity rather than a routing table.

Genomes are shared structurally between parents and children (genes and the delay
vector are copy-on-write), so mutation operators must never edit a list in place.
`_mutate_state_delay` builds a fresh vector every time.

## 4. How the genetic algorithm works

### 4.1 The generation loop

The controller (`runtime/controller.py: run_evolution`) is backend-neutral;
it wires up one backend's routines and runs the same loop. Per generation:

1. **Evaluate.** `eval_batch_cases(genomes, target, cache, pool)` grows and
   scores every genome. Evaluation is pure (grow plus score, no RNG), so it is
   cached in an `LRUCache` keyed by `genome_signature(genome)`, which includes
   the delay vector and the architecture so timing variants are never aliased.
   Work is spread over a process pool by `runtime/parallel.py: map_ordered`,
   one saturated pass with no chunk barrier, polling the stop signal as each
   genome finishes.

2. **Select and reproduce.** `next_population(...)` builds the whole next
   generation: immigrants, then children from selected parents via
   `crossover_nv` (unless recombination is off) and `mutate_nv`.

3. **Survive.** Before a solution exists, offspring replace the parents, so the
   population's best can regress; that is intended exploration. Once any genome
   reaches 1.0, `consolidate_population(...)` switches to terminal (μ + λ)
   survivor selection over parents and offspring, so solved circuits accumulate
   and the charted mean converges.

4. **Anneal.** The base rate decays each generation (`mut_rate *= MUT_DECAY`),
   then `adaptive_mutation_rate(...)` applies plateau reheating.

The generation budget runs to completion even after a solve; reaching 1.0 only
skips the remaining restarts. There is therefore a genuine post-solve phase in
which the population drifts under survivor selection and its backend-specific
final tie-break (topology for FNV, parsimony where enabled).

Key constants (`substrates/nervous/ga.py`): `POPSIZE = 120`, `MEAN_MUTATIONS = 4.0` (a hot
start for simulated annealing), `MUT_DECAY = 0.997`, a deliberately slow cooldown
because hard recurrent tasks need late variation: 0.997 cools 4.0 to about 0.89
by generation 500, where the older 0.99 crashed it to about 0.03. Workers scale
with the machine (`N_WORKERS = min(cores - 2, 16)`) and the fitness cache is
bounded at 200,000 entries for long runs.

### 4.2 Selection

**Tournament (default).** `rank_key(genome, fitness)` is the ranking key. Fitness
dominates absolutely. SNN and nervous ties can break toward a cheaper organism,
fewer genes first and then a shorter germline telomere for body size; LUT has no
small-genome tie-break, and FNV uses its input-reachable connectivity/feedback
potential instead. No tie-break distorts the fitness value, so a solved run
still reads exactly 1.0. Elites form
the breeding pool (truncation selection) rather than a copied-forward elite:
parents are drawn by tournament within that pool.

**ε-lexicase (optional).** `_lexicase_parent(population, case_vecs)` streams the
test cases in random order and, at each case, keeps only candidates within ε
(median absolute deviation) of that case's best. A mean hides a single failing
trial, roughly 1/12 of the score; lexicase makes every case a hard filter some of
the time, so specialists on the currently-failing cases get selected and
recombined. It must stream over the whole population, so it bypasses the elite
pool.

### 4.3 Mutation operators

`mutate_nv(genome, mean_mutations, max_telomere, chromosome_count, evolve_delay)`
draws a Poisson number of events (minimum 1, so mutation is always constructive)
and applies `_mutate_once_nv` for each. One mandatory routing tweak guarantees
the child is not a clone. The operator menu (`_MUT_OPS` / `_MUT_WEIGHTS`),
filtered to what is currently feasible:

| Operator | Weight | Effect |
| :---- | :---- | :---- |
| `tweak` | 0.32 | Change one field of one gene. Biased to keep growth rules (`self_in = 0`) reachable. |
| `duplicate` | 0.14 | Copy a gene within its chromosome. |
| `add_gene` / `del_gene` | 0.14 / 0.11 | Grow or shrink a chromosome's rule set (cap `MAX_GENES`). |
| `add_chrom` / `del_chrom` | 0.05 / 0.05 | Only when chromosome count is unconstrained (`chromosome_count=None`). `del_chrom` removes the smallest chromosome, so a deletion prunes the least-carrying module rather than wiping a large functional one. |
| `split` | 0.11 | Move a chromosome's crossover boundary. |
| `telomere` | 0.08 | Nudge a chromosome's division limit by ±1–3, evolving body size. |
| `delay` | 0.30 | `pulse_delay` only: geometric walk of one node type's delay multiplier. |

**Delay-mutation toggle.** Which timing operator is live is normally implied by
the node model, but the two are decoupled so ablations are possible.
`GAConfig.evolve_delay` accepts `None` (follow the model pairing, the default and
legacy behaviour) or an explicit `False` to disable the paired mutation, giving
width-preserving transport at a fixed base delay. `True` is rejected unless
`pulse_delay` is selected, because the engine ignores the vector otherwise and
the run would evolve dead genes. `GAConfig.timing_mutations()` resolves it, and
`ga.timing_mutation_flags(model, ...)` is the shared resolver used by
`next_population`, `evolve_temporal` and `diversify`.

Mutation always changes the genotype, but not necessarily the circuit. Roughly
22% of single-event mutations grow an identical phenotype, because a mutation can
land on a chromosome tag, a split point, an unexpressed rule, or a non-maximal
telomere. Section 6.9 explains why the diversity analyser separates those from
real perturbations.

### 4.4 Crossover

`crossover_nv(pa, pb)` is tag-matched hierarchical crossover: chromosomes are
paired with their nearest homolog by tag distance, and multi-gene homologs cross
at a genuine interior gene boundary (the split). When the common homolog has only
one gene, the rule's fields are recombined instead, so a minimal chromosome still
gets useful sexual recombination. Recombination can be disabled at runtime
without disabling mutation or immigration.

### 4.5 Plateau response (SOS) and annealing

`runtime/mutation.py: adaptive_mutation_rate(annealed_rate, stagnation,
solved, beta, mutation_limit)`. Below `STRESS_PATIENCE` flat generations the
effective rate is just the annealed rate, floored at 1 and capped by
`mutation_limit`. Past that it ramps as `1 + beta × plateau_age` up to the same
hard cap; `beta = 0` disables reheating entirely.

**Solved runs do not reheat.** `solved=True` pins the rate to the capped annealed
value. This fixed a real defect: SOS hypermutation was reheating a solved run to
maximum, overriding the anneal and holding the population mean down even though
the best was 1.0.

Past `STRESS_PATIENCE`, reproduction also keeps producing mutated descendants
of the separately archived all-time champion. This does not copy an evaluated
parent into the live population: every archive descendant contains a real
mutation, so pre-solve generational replacement remains intact. Evolvable I/O
descendants may receive a coordinated multi-port bundle rather than one
independent port edit.

### 4.5.0 Escaping a local minimum (`runtime/escape.py`)

The SOS reheat above is one population-wide response to a stall. `runtime/escape.py`
adds six more, all **off by default** — an unconfigured run, and any checkpoint
written before the module existed, behaves exactly as it did before it.

Before reaching for any of them it is worth knowing which failure you have,
because the cures do not overlap. Sample ~200 single mutations of the champion
and look at the deltas: neutral-heavy means a flat **plateau** (no gradient
exists — more mutation will not help, you need finer resolution or different
selection pressure); all-negative means a true local **peak** (you need larger
or structured jumps); and a population converged on a trivial strategy that
already scores well is a **degenerate attractor**, which is not a search
problem at all — see 5.6 and `tools/probe_trivial_baselines.py`.

| Mechanism (`EscapeConfig` field) | Default | What it does |
| :---- | :---- | :---- |
| Lifespan (`lifespan_scoring`, `lifespan_checkpoints`) | off, 3 | Scores the organism at several points along its **development**, not only as a grown adult. |
| Crowding (`crowding`, `crowding_window`, `crowding_fraction`) | off, 16, 0.5 | Restricted tournament replacement over a **reserve** of the population: an offspring competes against the most genetically **similar** member of a random window. |
| Neutral drift (`neutral_drift`) | off | Accepts equal-ranked challengers rather than demanding strict improvement. |
| Self-adaptive mutation (`self_adaptive_mutation`, `adaptive_tau`) | off, 0.25 | Each individual carries its own heritable mutation rate. |
| Rebirth (`rebirth`, `rebirth_patience`, `rebirth_fraction`, `rebirth_ancestors`, `rebirth_mutation_multiplier`, `archive_interval`, `archive_size`) | off, 40, 0.5, 4, 3.0, 10, 24 | On a stall, rebuilds part of the population from **diverse** archived ancestors at an elevated rate. |
| Robustness (`robustness`, `robustness_jitter`, `robustness_samples`) | off, 0.15, 2 | A second objective scored under jittered physics, ranked strictly below correctness. |
| Lexicase sample (`lexicase_downsample`) | 1.0 | Fraction of cases ε-lexicase streams per generation. |

**Lifespan scoring** answers the flat-plateau case. A genome whose stage-6 body
half-works but whose stage-12 body is broken currently scores zero, dies, and
leaves no gradient behind. Growth already produces a full trajectory
(`grow_nervous_snapshots` / `grow_lut_snapshots`), so the interior stages are
interpreted through the same `prepare_net_grid` / `prepare_lut_grid` path the
adult uses and scored by the ordinary contract. Two rules keep it honest: the
**reported fitness is always the adult score** — a run that reads 1.0 still
means the fully grown circuit works — and juvenile scores enter selection only,
as extra ε-lexicase cases and as a `rank_key` tier below behavioral fitness.
The case vector is always exactly `contract_case_count + checkpoints` long,
padded with the adult score for an organism that matures in fewer stages than
there are checkpoints, because ε-lexicase requires every population member to
present the same number of cases. Cost is roughly one extra evaluation per
checkpoint.

**Robustness** re-scores the adult under a deterministic ± ladder of perturbed
physics (delay up while width goes down, and vice versa — the two ends of an
asynchronous circuit's real timing margin). Determinism is not cosmetic: the
fitness cache is keyed on the genome alone, so random jitter would freeze
whichever draw happened first into that genome's score. Aggregation is by
**worst case** across jitter variants, and the case-vector collapse anneals
from mean toward min as the run's best fitness climbs — the min is the right
demand once a run is near solving, but it is uniformly zero (and therefore
gradient-free) early. Because it is aggregated in the driver rather than the
worker, `EscapeState.apply_robustness_blend` runs before anything is ranked.

Both extra objectives sit **lexicographically below fitness** in every
backend's `rank_key`:

```
viability > fitness > robustness > juvenile > backend final tie-break
```

For FNV the final tier is topology; for parsimony-enabled backends it is cost.
That ordering is the entire safety argument. Robustness can only ever separate
two circuits that are already equally correct, so a robust-but-wrong circuit
can never outrank a correct one; juvenile credit likewise only breaks ties,
most usefully across the flat zero-fitness region where nothing else does. With
both mechanisms off, both tiers are `0.0` for every genome and the ordering
collapses to exactly the one that preceded them.

**Crowding is monotone by construction, and that has to be a deliberate
choice.** An incumbent is only ever replaced by a challenger that ranks at
least as well, so a crowded population's fitness multiset can only rise.
Measured over the whole population: **zero** mean-fitness decreases in 9 of 9
runs (three targets × three seeds, 60 generations), against 24–31 for the
ordinary loop. The mean rises as a smooth curve with no dips at all. That is
what niche preservation buys, and on its own it is the wrong shape for leaving
a basin — a population that can never move downhill cannot cross a valley. It
also silently overrides this project's deliberate pre-solve rule that elites
breed but never survive.

So `crowding_fraction` (default 0.5) crowds only part of the next population;
the remaining slots are filled from the offspring under ordinary strict
generational replacement, which is where the exploratory churn lives. At the
default the same runs show 7–33 mean decreases per 60 generations — churn
restored, niches still protected. Setting it to 1.0 restores textbook
whole-population RTR, monotone mean and all.

**Rebirth** keeps a ring buffer of champion snapshots and, on a stall, rebuilds
`rebirth_fraction` of the population from `rebirth_ancestors` *maximally
different* archive entries (greedy farthest-point selection over
`genome_descriptor`). Re-seeding from the single best ancestor would simply walk
the same path again; spreading the seeds is the point of backtracking to a
branch point. The current elites are retained — this is a backtrack, not an
extinction — the reborn cohort mutates at `rebirth_mutation_multiplier ×` the
run rate so it leaves in a different direction, and a cooldown equal to the
patience stops it re-firing every generation while one stall persists.

**Downsampled ε-lexicase** streams a fresh random subset of cases each
generation. At equal evaluation budget that buys several times more generations
for the same selection quality, and because the subset is redrawn every
generation it is also what "rotate the stimulus set" reduces to once the cases
already exist. The ε itself (median absolute deviation, `_lexicase_parent`) is
what makes lexicase usable on continuous scores at all: plain lexicase filters
on exact ties, which floats essentially never produce, so the first case drawn
would decide every selection on its own — single-case selection wearing a
diversity costume.

**One drive path.** These mechanisms exist in one backend-neutral module because
the project has two GA drivers per backend — the headless `evolve_*` and
`runtime/controller.py` — and they have drifted before, with benchmarks
measuring one while the application ran the other. `build_escape_state` is the
single construction point for the clone/mutate/rank closures, and
`EscapeState.merge_generation`, `.accepts`, `.record_champion`, `.maybe_rebirth`,
`.apply_robustness_blend` and `.tick` are called by every driver. Re-evaluation
of the reborn cohort is sequenced inside `maybe_rebirth` rather than in each
loop for the same reason. `tests/test_escape.py` asserts both that the defaults
are inert and that every driver calls every hook.

### 4.5.1 Developmental spatial I/O

The `spatial_chromosome` strategy reserves chromosome 3 for normalized `(x,y)`
port alleles in a stable target coordinate field. Input alleles are decoded
before growth and used as distinct germline seeds. Quantization collisions are
resolved deterministically to the nearest free lattice site; evaluation never
rewrites the genotype. Consequently, an input-placement mutation changes both
the driven cell and the organism grown around it. Output alleles remain
target-blind attachments to the nearest exclusive mature cell.

Fresh spatial genomes always prime input alleles from the declared viable
geometry. Seventy-five percent also prime output geometry; the remaining
quarter randomize outputs while retaining the same input prior. This separates
initial readout exploration from morphology failure without granting the
trace-fitting privilege used by `fixed` output placement.

All runtime consumers pass the genome into
`substrates.nervous.io_placement.growth_seeds(...)`, including nervous/LUT/SNN training,
temporal adapters, frozen held-out evaluation, robustness, diversity, replay,
and the GUI. A truth-table-compiler provenance marker preserves the older
single-centre inverse-grown LUT witness as an explicit compatibility exception.

### 4.5.2 Dedicated directional terminal nodes

`terminal_nodes` is a heritable developmental placement strategy for the
nervous and LUT substrates. Each body gene carries an `io_kind` allele:
ordinary body, source-only input, or sink-only output. The allele belonging to
the winning developmental gene is recorded on the mature cell. LUT cells have
four directional winners; a single non-body kind is expressed, while an
input/output conflict resolves conservatively to an ordinary body cell.

Terminal-mode development starts from one neutral centre rather than declared
target pads. Binding considers only live cells with a matching expressed kind
and orders them with a stable genotype-keyed site rank. It activates exactly
one cell per logical input and exactly one per declared output role; excess
marked cells remain ordinary runtime body cells. Consequently a single-output
target has one active sink while Half Adder has two (`sum` and `carry`).
Binding never reads target coordinates, expected traces, or truth-table values.

`io_kind` is independent of the routing/LUT state alphabet, so existing circuit
states keep their meaning. Mutation changes the allele among the three kinds;
crossover and checkpoint persistence carry it with the rest of the gene. Old
checkpoints omit it and therefore load as ordinary body cells. A source
terminal ignores its local routing/LUT and emits only externally injected
signals. A sink terminal computes and records its local response but is masked
from all downstream excitatory, inhibitory, and LUT-neighbour reads. The same
masks are passed through scoring, frozen validation, continuous playback,
Interactive, and Designer construction paths. Tri-circuit outputs mask all
three subnodes belonging to the selected output tile.

For the LUT backend with `spatial_chromosome` binding,
`plateau_rescue_candidates(...)` adds a small deterministic memetic
neighbourhood to that offspring generation. Two-input/two-output bodies try
compact 2x2 assignments (all four port identities remain heritable), and rules
expressed at bound output cells try every one-bit `self_out` neighbour.
The motif and one-bit proposals never inspect expected answers. In addition,
periodic truth tables now receive an explicit compiler proposal from
`substrates/lut/synthesis.py`. A four-input hub uses its four directional LUTs as four
independent output functions; the five-port comparator adds a strobe-driven
zero detector and merge stage. Unaddressed LUT bits label cells for inverse
development, and the inverse records exterior `self_in=0 -> 0` suppressors as
well as positive births. A heritable polarised seed breaks the forced fourfold
symmetry of the legacy isotropic centre seed (`None` retains legacy behaviour).
The compiler result is admitted only after it re-grows exactly as an ordinary
genome. Every proposal is still accepted or rejected by the unchanged growth
engine and Behavior Contract; no hand-injected phenotype is scored.

### 4.6 Diversification

`diversify(seeds, target, pop_size, valid=0.999, ...)` fills a population with
evaluated, genetically distinct valid solutions once one exists. Distinctness is
measured on the rule alleles crossover can actually exchange
(`_recombination_signature`), not on tags, split metadata, or a neutral telomere
edit. Where the target has a broad neutral network it fills the population; where
solutions are isolated spikes it returns however few exist.

Because uniqueness is enforced by construction, the genotype counts of a
`diversify` pool describe the tool rather than the substrate. Measure the GA's
own post-solve population separately (see 6.9).

### 4.7 The tuning options

| Control (`GAConfig` field) | Default | Meaning |
| :---- | :---- | :---- |
| Mutations (`mean_mutations`) | 4.0 | Poisson mean events per child, before annealing. |
| Limit (`mutation_limit`) | 8.0 | Hard cap on the effective rate; bounds both the initial rate and plateau reheating. |
| Immigrants (`immigrant_fraction`) | 0.08 | Fraction of each generation drawn fresh at random. |
| Tournament (`tournament_size`) | 4 | Candidates per tournament draw; 1 is uniform among elites. |
| Elites (`elite_count`) | 1 | Size of the breeding pool, not copies carried forward. |
| Anneal (`mutation_decay`) | 0.997 | Per-generation multiplier on the base rate. |
| Beta (`stagnation_beta`) | 1.0 | Plateau response strength; 0 disables SOS reheating, larger ramps faster after `STRESS_PATIENCE` (12) flat generations. |
| ε-lexicase (`selection`) | tournament | Switches parent selection to lexicase. |
| Recombine (`recombination_enabled`) | true | Off means clone-and-mutate, no crossover. |
| Max telomere (`max_telomere`) | 20 (LUT 8) | Ceiling on the evolvable growth radius. |
| Chroms (`chromosome_count`) | None | Fixed chromosome count, enforced by every operator. |
| NV profile (`tile_arch` + `node_model`) | analog tri-circuit | The ONE profile a new nervous run may use (2.2). The dropdown has a single entry; the single-tile and digital tri-circuit engines are retired. |
| Delay / Width / Coinc (`PulseConfig`) | 1.0 / 1.0 / 0.5 | Nervous-net physics: propagation delay, emitted pulse width, coincidence window. |
| Analog node (`analog_threshold`, `analog_step`, `analog_tau_leak`, `analog_hysteresis`) | 0.5 / 0.34 / 1.10 / 0.08 | Analog profile only: the Figure 1 constants. Validated as a coupled set (2.2), so an impossible combination is reported as invalid tuning rather than crashing a run. Untouched defaults reproduce the audited physics exactly. |
| FNV families (`FNVConfig.families`) | all seven | Enables or disables LOGIC, DELAY, NORMALIZER, HOLD, C_ELEMENT, TOGGLE, and GATED_OSCILLATOR as whole families. At least one is required. |
| Escape (`escape`) | all off | The local-minimum escape mechanisms of 4.5.0, on their own two-row control block. `EscapeConfig` validates its own ranges, so a nonsensical entry is reported as invalid tuning on Run. Lifespan, Robustness and Lexicase sample read the temporal contract's per-case vectors and are therefore disabled for the SNN backend; FNV also disables options its adapter cannot honor. The population-level mechanisms apply to all four. |

The whole configuration is an immutable, process-safe
`RunConfig(ga, pulse, fnv)`.
`RunConfig.__post_init__` enforces `ga.node_model == pulse.model`, so a run can
never score under one model while mutating for another, and
`validate_new_nv_profile` gates fresh nervous-net runs to the profiles in 2.2.
`RunConfig.from_dict` migrates old checkpoints: it drops the retired
`delay_gain`, promotes the model into `ga`, drops the retired `evolve_width`
toggle, and maps `node_model='evolved_width'` onto `uniform` while keeping the
run's width and delay physics. A genome's stored `state_widths` vector is ignored
on load. Such a run is flagged in the status line as retired physics, because
single-tile `uniform` is not one of the current profiles.

## 5. Targets: scoring, fitness, and testing

### 5.1 What a target is

A `TemporalTarget` (`substrates/nervous/targets.py`) declares the inputs (seed cells), the
output terminals (roles), a horizon `T`, and a bank of `Trial`s. A trial carries
a stimulus, either per-tick `streams` or explicit physical `input_events` as
`(start, width)` pairs at real, possibly fractional times, plus what the output
should do: an expected per-tick trace, point `expected_events`, or complete
`expected_intervals`.

The target also carries a `BehaviorContract`, built with helpers such as
`event_contract(...)`, `state_contract(...)`, `interval_contract(...)`, and
`cadence_contract(...)`. Adding a target normally means selecting and
parameterizing these reusable restrictions; no scorer branch or backend edit is
required.

**Why banks are diverse.** Every preset carries several trials with different
pulse timings. A net that merely matches one fixed schedule (a lucky delay chain)
fails the shifted trials; only genuine state passes all of them. This is not
theoretical: a "solved" toggle was once found locked to the exact pulse gaps it
had been trained on.

### 5.2 Executable behavior contracts

`substrates/nervous/contracts.py` defines serializable `BehaviorContract` and `Constraint`
data. A target declares the restrictions that express its idea; it does not
select a scoring pipeline. `substrates/nervous/scoring.py: score_contract` is the only
target-facing fitness entry point. SNN, nervous, FNV, and LUT code only translates
substrate output into normalized observations and passes those observations to
that evaluator. Output placement uses the same entry point, so evolution cannot
optimize a different approximation from the score later reported to the user.

A contract can combine restrictions. The current reusable relations are:

| Relation | Reads | Required behavior |
| :---- | :---- | :---- |
| `truth_table` | normalized logic value or confidence | Match every expected output bit for every input row. Missing or unstable readouts score zero. |
| `event_correspondence` | continuous rise times | Match required edges one-to-one. Missing and spurious edges both cost. A single optional latency offset is shared across the complete trial bank. |
| `transition_correspondence` | logical changes derived from complete intervals | Time distinguishable state activations with one shared latency. For a nervous pulse ring, ordinary circulation and the pulse already in flight after reset are not counted as new logical transitions. |
| `logical_state` | complete pulse intervals, with sampled levels as a backend fallback | Satisfy active and quiet epochs. A nervous active state may be a circulating pulse; a LUT active state is a held level. Exact tick phase is not part of the target. |
| `pulse_intervals` | complete rise/fall intervals | Match event identity and duration. Correct rise times with wrong widths cannot pass. |
| `sustained_cadence` | continuous rise times | Maintain the required period after a trigger, with free absolute phase, sufficient event count and horizon coverage, and no pre-trigger output. |
| `commanded_cadence` | command-delimited output activity | Maintain a cadence in every dwell and change it in the required direction after commands. |
| `bounded_state` | long-horizon rise trains | Satisfy hold, clear, quiet, reset-influence and reload cases. Retention and SR curricula are ordinary contract cases rather than separate evaluator dispatches. |

Multiple constraints use `mean_worst`: the weighted mean preserves a climbable
gradient and the worst component prevents easy restrictions from drowning out a
failed one. A total of 1.0 is possible only when every constraint is perfect.
`needs_samples(target)` derives collection requirements from the contract's
observables. Checkpoints serialize the contract; the loader performs a one-way
migration from historical `score_mode` fields, but runtime target objects no
longer contain or dispatch on that field.

#### Phase-invariant logical state

Nervous-net memory holds a bit as a pulse circulating in a loop, not as a DC
level. Contract v1 therefore converts the expected abstract state into active
and quiet intervals. During an active interval it measures the maximum silent
gap in continuous time. `2 × (DELAY + WIDTH)` is the smallest legal lap, not a
maximum circuit size: when at least two pulses inside commanded-active epochs
demonstrate a regular longer lap, that circuit's own lap becomes its retention
budget. The estimate is made once per trial, excludes cross-epoch gaps, and is
capped by the longest active epoch so isolated blips cannot claim an arbitrarily
slow ring.

Reset cannot recall a pulse already travelling around the loop. The leading
one-lap portion of the following quiet epoch is therefore stopping grace; any
judgeable remainder must still be quiet. `transition_correspondence` reuses the
same demonstrated lap when it merges physical pulses into logical activity, so
ordinary circulation and that final in-flight pulse are not charged again as
spurious activations. Restarts whose boundary is hidden by ring phase are
optional in the timing clause, while `logical_state` continues to verify the
active and cleared epochs.

A one-off burst still fails when its trailing gap grows, silence fails
immediately, and persistent output is caught by the quiet epochs. No fixed bin
grid is laid over the intervals, so translating the same ring in phase cannot
alter its score. Intervals too short to distinguish one legal ring phase are not
used to claim retention. The LUT adapter declares strict level semantics and
must cover the whole active interval. `tools/probe_ring_penalty.py` sweeps honest
rings across every registered state target and fails if any is penalized.

### 5.3 Latency fitting

Most behaviours care about relative timing, so their contract enables one shared
latency offset across the entire bundle. In `substrates/nervous/scoring.py`,
`_best_event_shift` and
`_best_waveform_shift` generate candidate shifts from observed-versus-expected
edge pairs, clamp to `event_max_shift`, score each, and keep the best; the result
is cached on the observations object. Precision-delay targets such as Echo set
the contract's `fit_latency` parameter false because expected timestamps already specify the
required physical delay, which is what stops a direct input-to-output wire from
passing. A fitted latency can never conceal an incorrect pulse duration.

### 5.4 Cadence in detail

Used by the kicked autonomous rhythms (Oscillator, and Pattern when it has
exactly one rise per cycle). Per trial, given a fitted startup latency `L`, the
kick is the trial's first input edge and the scored window runs from
`kick + L + cadence_settle` to `T + L`. The trial score is the product of four
factors:

* **quiet** = `1 / (1 + edges before the kick)`. Output before any input is
  penalised hard.
* **regular** = fraction of consecutive gaps within `cadence_tolerance` of
  `cadence_period`. Phase-free: only gap lengths matter.
* **count** = `min(1, steady events / cadence_min_events)`. A two-blip transient
  is not a rhythm.
* **coverage** = span of the steady train ÷ required span. A burst that rings
  correctly then dies decays exactly this factor.

Because one `L` is shared across trials that kick at different ticks, a genuine
kicked loop fits all of them, while an input-ignoring free-runner is caught by
**quiet** in the late-kick trials.

### 5.5 Oracle targets and held-out certification

Hand-written banks memorise. The stronger construction (`substrates/nervous/oracle.py`)
defines a target as a reference state machine plus a random stimulus generator:
`oracle_target(name, oracle, inputs, ...)` samples schedules with a seed and
labels them by running the state machine. Fresh, never-trained-on schedules can
therefore be sampled at any time, which is what certification does.

`certification.certify(genome, target, train, backend, seeds, threshold=0.90)`
re-samples the spec at held-out seeds (default 4242, 777, 31415), scores with the
training readout and alignment frozen, and returns a verdict:

* **CERTIFIED** — held-out at or above threshold.
* **OVERFIT** (memorised timing) — trained well but held-out collapsed by more
  than the allowed gap.
* **BELOW THRESHOLD** — with an explicit note on whether it still generalises.
* **UNCERTIFIED** — the target has no oracle reference. Autonomous rhythms have
  no input-to-output relation to sample, so they stay hand-built by necessity.

`certification.carry_physics(src, dst)` copies the run's `pulse_config` or
`lut_config` onto every freshly-built spec target. Without it, a champion evolved
under a non-default node model would be validated under default `uniform`
physics and every such run would read as OVERFIT.

Measurements motivate the oracle construction for most input-driven targets:
held-out Echo scores 0.55 hand-trained against 1.00 oracle-trained, and the latch
0.74 against 0.93. Coincidence measured better hand-built, 0.94 against 0.68, so
it stays hand-built. The registry keeps whichever construction measures better.

### 5.6 Anti-cheat: targets are probed before they are trusted

A target is only as good as the cheapest circuit that scores 1.0 on it. Two
degeneracies were found and fixed by probing with idealised cheats:

* **One-shot(3)** was degenerate: a single-pulse echo scored 1.0, because the ±1
  ring tolerance covered the whole 3-tick window. Widened to 5 ticks, so only
  real self-terminating bursts pass.
* **Odd pulse selector** did not require counting at all: a parity-free
  refractory filter with one fixed dead time (D ≈ 4.1) reproduced every schedule
  and scored a perfect 1.0, because every single gap in the bank was shorter than
  every double gap. Fixed by adding three adversarial gap banks that force
  contradictory dead times: long gaps where a pulse must be suppressed, quick
  gaps where one must be passed. The best fixed-dead-time filter now measures
  about 0.89, below the 0.90 certification bar, and a regression test sweeps both
  filter families on the training seed and two held-out seeds.

The `_mix_event_widths` helper carries a related guard: widened pulses are
clamped to stay clear of the next event on the same lane, because an overlap
would merge two labelled stimulus events into one physical edge and make the
oracle's expected output unreachable in every model.

### 5.7 Target catalogue

Periodic combinational targets retain their source truth table. If the
all-zero row has any high output, the wrapper adds one case-valid strobe input:
without an onset, an asynchronous circuit cannot distinguish that row from an
idle settle interval. Tables whose zero row is all low are unchanged.

| Folder (GUI) | Examples | Notes |
| :---- | :---- | :---- |
| Combinational logic | AND/OR/XOR/NAND/NOR/XNOR, half and full adder, 2-bit adder, 2×2 multiplier, 2:1 MUX, 2-to-4 decoder, comparator, majority-3, parity-3 | Native truth tables on SNN. On the asynchronous backends they are wrapped by `periodic_combinational_target(...)`: every input combination is tested in its own widely-spaced window (a 1 emits an event, a 0 is silent), with the onset-to-onset gap set to several times the grid's settling transient so one case cannot contaminate the next. Each table repeats under alternate row orders and two phases so a fixed oscillator cannot replace input-dependent logic. |
| Timed events | Coincidence, Temporal XOR, Sequence A→B, Veto gate, Burst ×3, Divide-by-3, Echo, Pair detector, C-element, A-first rendezvous, Collision serializer, Refractory filter, Watchdog, the A-count query family, Period doubler/tripler/halver, Temporal sum | Point-event (F1) scoring. The mixed-width A-count query family is the fairest cross-model comparison set. |
| Memory and state | SR latch, Toggle flip-flop, One-shot, Gated oscillator, Resettable toggle | Persistence-window scoring; the real memory tests. |
| Cadence and patterns | Oscillator (period 2), Pattern (1000), Period stepper | Autonomous rhythms. No oracle is possible, so these are hand-built by necessity. |
| Pulse width and duration | Pulse width sum (A+B), Odd pulse selector, Pair detection gap (2× pulse width), Pulse doubler | Duration semantics. The two waveform-contract targets declare `supported_models=('pulse_delay',)`. |

### 5.8 Which targets suit which model

This matters whenever results are compared across architectures or node models.

* **SNN** — native combinational truth tables plus recurrent temporal runs.
  Recurrence is an explicit architecture field, saved in checkpoints, and is
  enabled automatically only for temporal targets so old static circuits retain
  their original topology. Point-spike output cannot preserve arbitrary pulse
  duration, so waveform targets remain excluded by metadata.
* **LUT array** — temporal targets. It is a chaotic recurrent system that cannot
  settle to combinational logic, since that scoring caps at constants, so the
  picker hides raw combinational entries and offers the periodic wrappers
  instead. Because its wires carry levels, it holds state genuinely (strict hold
  scoring) rather than ringing.
* **Nervous net** — everything, with model restrictions. Waveform-contract
  targets (Pulse width sum, Odd pulse selector) are physically unreachable under
  `uniform`, because a node regenerates a fixed width and every internal wire has
  a single driver, so no mechanism can synthesise an input-dependent duration.
  They declare `supported_models`, and the picker hides them under the wrong
  model rather than letting a run silently cap below 1.0.
* **FNV** — combinational and temporal targets through periodic input schedules.
  It has fixed I/O placement and no per-node tuning. Enabled families define the
  physical parts bank for the run, but a native one-component match is never a
  target restriction. Held-out certification regrows the genome and reuses the
  exact training output cells and alignment without refitting.

**Fair cross-model comparison.** Use the event-scored, mixed-width oracle targets
(the A-count query family, toggle, C-element, sequence, coincidence, serializer):
same stimulus, rise-only read-out, no width semantics to advantage a variant.
Avoid comparing on targets one model can trivialise with a single parameter.
Echo (delay 3) is one `pulse_delay` node at delay ×3, where `uniform` needs a
3-node chain. Such a result demonstrates what the extra degree of freedom buys;
it does not measure circuit evolution.

**Ablation discipline.** Whenever two configurations differ in more than one way,
a straight comparison confounds them. The legacy and analog tri-circuit profiles
differ in both tile topology and node physics, which is why the digital
tri-circuit profile exists: it changes the tile alone and holds the node at the
digital abstraction, so a result can be attributed to one cause. The same
argument applies within the width-preserving model, where
`GAConfig.evolve_delay=False` isolates width preservation from delay
evolvability. That pairing is not offered for new runs and is reachable only
programmatically or from an old checkpoint.

### 5.9 Contract-v1 evolution matrix

`tools/benchmark_contracts.py` enumerates the Cartesian matrix of the 15 logic
and 31 temporal targets against SNN, nervous legacy, nervous digital tri,
nervous analog tri, and LUT. Unsupported combinations are recorded with a
reason. Every supported row receives its own deterministic target seed, and the
runner atomically checkpoints JSON plus a readable Markdown table after each
row; resuming never repeats a completed evolution.

The July 21, 2026 diagnostic used 100 generations, population 12, two
chromosomes, and base seed 20260721. It completed all 230 rows with zero runtime
errors. Because the population is deliberately small, these are search
diagnostics rather than ceiling claims. The later recurrent-SNN execution smoke
used two generations and population four: all 31 temporal rows terminated
cleanly, 28 were applicable, 26 exposed nonzero search signal, and Coincidence
plus One-shot reached training fitness 1.0. A separately designed recurrent LIF
loop reaches 1.0 on Oscillator, proving the zero evolutionary smoke score is a
search miss. The smaller search budget is deliberately not compared as if it
were a matched benchmark. Exact maxima, seeds, durations, exclusions, and the
interpretation of open results are recorded under `results/`.

## 6. The application: tabs and features

### 6.1 The control header

Above the tabs, and shared by all of them:

* **Model** — Nervous / FNV / LUT / SNN. The master switch: it reconfigures the window,
  filters the target list, retitles the activity tab, and shows or hides the
  Designer and Diversity tabs (6.7, 6.9).
* **Target** — a two-part picker: a category folder plus a searchable, editable
  name box that filters incrementally. Folders are described in 5.7.
* **Gens / Pop / Chroms / Tries / Seed** — run size and reproducibility.
  Evaluation is deterministic, so a seed pins the entire evolutionary
  trajectory.
* **Substrate (Vth / Syn / Input) and Graded** — SNN only; disabled and
  irrelevant for the paper architectures.
* **GA tuning row** — mutations, limit, immigrants, tournament, elites, anneal,
  beta, ε-lexicase, recombination, max telomere (4.7). Disabled for SNN, which
  uses its own fixed constants.
* **Pulse physics row** (nervous only) — Delay, Width, Coincidence, and the NV
  profile dropdown, which now has a single entry — analog tri-circuit (2.2).
  Changing the profile re-filters the target list, relabels the physics fields,
  and shows or hides the analog constants row (Vth / Step / Tau leak /
  Hysteresis), which appears only under the analog profile and locks while a run
  is in flight.
* **FNV family row** — whole-family switches for logic, delay, normalizer, hold,
  C-element, toggle, and gated oscillator. At least one family remains enabled;
  timing and routing variants are never exposed as numeric parameters.
* **Escape block** (two rows) — the local-minimum mechanisms of 4.5.0: Lifespan
  (+ stages), Crowding (+ window), Neutral drift, Self-adaptive mutation,
  Rebirth (+ stall, frac), Robustness (+ jitter), Lexicase sample, and a Reset
  escape button. Every control carries a hover tooltip explaining what it does
  and what it costs. Lifespan, Robustness and Lexicase sample are disabled for
  SNN, which has no per-case temporal contract; FNV also disables the mechanisms
  its adapter cannot honor. The population-level mechanisms stay live on all
  four backends. To the right, a live telemetry line reports
  the active mechanisms, rebirth count and the generations its ancestors came
  from, archive size, cumulative crowding replacements, how far the robustness
  aggregator has annealed toward worst-case, and the population's mean
  self-adaptive mutation rate. It reads `off` when nothing is enabled.
* **Run / Stop / Pause / Save / Load Saved / Reset tuning** — checkpoints carry
  the genome, target, run config, seed and certification verdict, and restore the
  UI state. Stop cancels queued population evaluations, lets the few already
  executing drain on the background controller thread, and only then re-enables
  Run. This prevents an abandoned Windows process pool from overlapping a later
  run. A final checkpoint/render error is reported after controls are released,
  so it cannot strand the window in its locked running state.

### 6.2 Evolution tab

The live run. The chart plots four series against generation: best-of-all-time,
best new offspring, population mean, and population fitness σ, with the effective
mutation rate on a second axis, so plateau reheating and annealing are visible as
they happen. Note that σ is a diversity read-out only while fitness still varies;
once the population is solved it is identically zero, which is what the Diversity
tab (6.9) exists to replace. The right panel is titled **Behavior Contract**
before a run and **Contract Score** after evaluation. Its first block is generated
directly from the target's executable `BehaviorContract`: contract version, the
single `score_contract` evaluator, aggregation rule, every restriction, its
physical observable, weight, and parameters. It then shows the target's tests;
after evaluation it adds actual observations, fitted shared latency, per-case
scores, and PASS/FAIL. This replaces the old "Truth Table" / "Trace Report"
presentation and makes it visible that every backend enters the same method.

### 6.3 Circuit Growth tab

The developmental trajectory: snapshots of the grid from the seeds to the mature
organism. Growth ends at the telomere-bounded attractor, so the last frames
repeat once the organism is mature.

### 6.4 Activity / Voltage Traces tab

Retitled by backend. For SNN it shows membrane voltages against thresholds plus a
spike raster. For the asynchronous substrates it shows wire activity over the
run.

### 6.5 Genome tab

The evolved genome as data and as logic: chromosomes, genes, split points, tags,
telomeres, and for the LUT array each 16-bit table decoded to a boolean
sum-of-products over N/S/E/W (`substrates/lut/boolfn.py`), so the result reads as a
circuit rather than as hex.

### 6.6 Interactive tab

Drive the evolved circuit by hand. Click **Load current solution** after a run or
a Load Saved. A persistent blue **Behavior Contract v1** badge names the active
restriction(s) and aggregation rule above the visualization, so raw playback is
never presented without saying what evolution actually judged.

* **Case dropdown** (temporal SNN / nervous / LUT) — every stored test case of the target,
  listed with its pulse times (for example `Case 3/10: A[3, 7, 14, 21]`; guard
  banks read silent). Selecting one loads that trial's physical schedule, widths
  included, so width-preserving playback sees what fitness scored, and resets
  playback to `t = 0`. Hand-editing the timeline afterwards flips the box to
  "(custom schedule)", so it never claims to show a case it no longer matches.
* **Pulse timeline** — click a lane for a default-width pulse, drag for a custom
  width, click an existing pulse to remove it. This is the stimulus.
* **Network view** — the grown circuit with its routing arrows (green
  excitatory, red inhibitory), the input cells ringed, and the trace-matched
  output cell labelled. Temporal SNN playback uses the same editable schedule
  but shows recurrent LIF membrane voltage and a spike raster in contract
  seconds. Nervous and LUT views show their native pulse/level activity; nervous
  nodes charge and discharge like capacitors during playback (6.8).
* **Output pulse edges** — nervous/LUT leading edges in real time. The SNN
  equivalent is the output-highlighted spike raster.
* **Output pulse widths (level view)** — directly below the edge strip: each
  output pulse as a bar spanning its real rise-to-fall interval, labelled with
  its physical width. A pulse still high at the cursor is drawn lighter and
  clipped to "now" with a `w…` label. This makes the width-semantics targets
  readable as duration rather than inferred from edge spacing.
* **Step / Run / Reset** — playback in real, sub-tick continuous time.

### 6.7 Designer tab

A manual circuit designer: build or refine hardware by hand and score it against
a real target. It is available for the nervous net and the LUT array only,
because the Designer edits grown hardware (routing states, lookup tables), which
is what the two paper architectures have and the SNN does not, so the tab is
hidden for SNN runs. Inside the app its architecture is not an independent
control: it follows the main window's Model selector
(`DesignerTab.follow_backend`), and its own Architecture box is a read-only
indicator. Run standalone (`py -m ui.designer`), that box becomes a live control.

* **Genome and grid editing** — edit the genome text and regrow, or edit the
  phenotype directly: place or delete cells, set a cell's Fig. 3 routing state
  (nervous) or its four lookup tables (LUT), assign inputs (growth seeds) and
  output roles. On the LUT array, place and delete only change contents, since
  the hardware field is uniform: "place" loads the relay seed table and "delete"
  zeroes all four tables.
* **Adopt target I/O** — copy a target's input terminals onto the grid as seeds
  and free its output roles for assignment or auto-placement.
* **Simulate and score** — the same engines, physics config and scorers as
  evolution, with a full per-trial report. A hand-built circuit is scored exactly
  as an evolved one.
* **Import and export** — pull in the app's current best, load
  `results/best_genome.json`, or save a design. A saved design carries the genome
  when one exists, and always the working grid, I/O and target, so a hand-built
  phenotype round-trips even without DNA.

### 6.8 Capacitor-style node playback

In both the Interactive and Designer views, nervous-net nodes do not snap between
grey and green. `substrates/nervous/playback.py: charge_levels(sim, t)` models each node as an
RC follower of its own binary waveform: charge rises toward 1 while the wire is
high (time constant `CHARGE_TAU`) and decays exponentially after it falls
(`DISCHARGE_TAU`), and `viz._activity_color` interpolates grey to green over that
0–1 level. A pulse visibly charges its node and the node fades afterwards, as the
paper's capacitively-coupled hardware behaves.

This is a display model only. It reads the engine's `pulse_intervals` waveform
log and never touches the simulation; the scored physics remains the binary
edge-triggered pulse. The helper returns `None` for an engine with no waveform
log, and playback falls back to binary activity.

### 6.9 Diversity tab

Population fitness σ is identically zero once every genome scores 1.0, so it
cannot report diversity in a solved population. This tab reads variety off
structure instead (`substrates/nervous/diversity.py`) and also works on unsuccessful
populations, where it prints the selected genomes' min/mean/max fitness and
valid count before the structural report. Analysis is opt-in, since it grows
every genome, and runs on a worker thread with progress reporting and a working
Stop. Both the digital and analog nervous profiles use their saved run physics.

Every normally completed or stopped nervous/LUT run writes two distinct atomic
population artifacts:

* `results/latest_population.json` contains every genome and fitness from the
  last fully evaluated generation. The Diversity tab selects this after a failed
  or stopped run, so a zero-solver run remains analyzable.
* `results/solver_generation.json` contains only genomes at or above the `0.999`
  validity threshold (or the post-solve diversified solver set). It is honestly
  empty when no solver exists rather than leaving stale results from an older
  run.

Stopping inside an evaluation never writes a partially scored generation: both
artifacts use the most recent complete generation boundary. If Stop arrives
before the initial generation completes there is no safe population to save;
the tab reports that zero-genome condition explicitly rather than drawing an
apparently broken empty chart.

Genomes are grouped four ways:

| Level | A group contains genomes identical in |
| :---- | :---- |
| Genotype (exact) | every inherited field: rule alleles, gene order, chromosome tags, split points, all telomeres, the delay vector, architecture |
| Functional genotype | rule alleles, germline telomere, the delay vector the run reads, architecture. Tags and split points are excluded as crossover bookkeeping |
| Phenotype (grown circuit) | architecture, the grown state grid, and the timing values of states present in it |
| Behaviour (off-spec probes) | quantised output edges on a frozen probe bank of stimuli the target does not specify, read at the fitted output cell |

Each level reports the group count, the largest group and its share, and
effective diversity, `exp(Shannon entropy)` of the group sizes, which is the
number of equal-sized groups with the same entropy. The count alone is
insufficient: 120 groups where one holds 90% of the population is not 120 kinds
of thing. The plot shows one composition pie per level, every group its own
wedge, and the text panel prints the same numbers plus the group-size
distribution and definitions of every term.

The behaviour level measures functional diversity under a declared probe suite,
not mechanism: two structurally different circuits can be identical on every
probe. The probe bank is versioned and seeded, and both are printed with the
result, because a different bank is a different measurement.

**Mutational robustness** (optional) samples the neighbourhood of each genome
under a documented kernel: one weighted GA mutation event, Poisson mean 1,
minimum 1, guaranteed non-clone. Validity uses the raw behavioural score rather
than loop-bonus-shaped fitness. Four numbers are reported:

* **Silent rate** — the share of mutants that grow the parent's circuit
  unchanged. The kernel never returns a clone, but a mutation can land on a tag,
  a split point, an unexpressed rule or a non-maximal telomere; measured at about
  22% for one-event mutations. Such a mutant scores identically by construction.
* **Local robustness** — the share of all mutants still scoring valid, silent
  ones included.
* **Effective local robustness** — the same over phenotype-changing mutants only.
  This measures the circuit rather than the mutation operator.
* **Novel-valid rate** — the share still valid and landing on a phenotype group
  not already present in the population. A genome whose neighbourhood is entirely
  copies of itself is robust but not evolvable.

## 7. Verification

`py tests/run_tests.py` runs the whole suite with bare Python; 397 checks across
19 files. Tests are organised by the claim they defend rather than by module.

| File | Claim it defends |
| :---- | :---- |
| `test_synchrony.py` | The nervous net is genuinely asynchronous: determinism, event-order independence, translation invariance at sub-tick offsets, scale covariance, no spontaneous activity. |
| `test_lut_synchrony.py` | The same audit for the LUT array, plus the lattice quantization contract against the synchronous reference engine, inertial blip filtering and honest power-on spontaneity. |
| `test_node_contracts.py` | Each node model obeys its own stated physics before any comparison means anything: coincidence window sweeps, refractory arithmetic, the four inhibitor timing cases, loop regimes, width drift, OR union tables. |
| `test_pulse_models.py` | The timing models: `uniform` is byte-identical to the pre-variant engine, width preservation follows its genetic delay map, checkpoints round-trip, and a run saved under the retired width model migrates rather than failing to load. |
| `test_nervous_lookup.py` | The packed growth lookup is exactly the fieldwise original: identical winners including first-wins tie-breaking, and bit-identical grown grids and snapshot sequences. |
| `test_tritile.py` | The three-circuit tile: one tile routes two signals to two outputs independently, 12-bit growth and per-channel mutation, mutual back-directions, the tile-keyed wired-OR readout, determinism, composition with the analog node. |
| `test_analog.py` | The analog node's stated physics: a single edge does not fire while two coincident edges do, a buffer's doubled coupling fires alone, dense input stretches the output, the hysteretic re-arm, the coupled parameter validation. |
| `test_analog_reference.py` | The analog engine against an independent small-Δt numerical integrator, so the fast engine is not merely self-consistent. |
| `test_scoring_equivalence.py` | Contract-v1 semantics: every target has a declarative contract; all observation requirements derive from it; logic, event, state and interval cheats cannot reach 1.0; translating the phase of the same memory ring cannot change its score. |
| `test_oracle_logic.py` | Each oracle state machine encodes its stated behaviour, and its banks contain the positive, negative, boundary, re-arm and silence cases they claim. |
| `test_ga_dynamics.py` | Reproduction invariants: chromosome-count constraints survive every operator, children are non-clones from distinct parents, recombination can be disabled without disabling reproduction, configs round-trip; Stop drains its pool and saves both the full evaluated generation and solver subset. |
| `test_diversity.py` / `test_diversity_ui.py` | The four diversity levels distinguish what they claim and nothing more, including an unsuccessful analog tri-tile population; silent mutants are separated from survivors, and the tab's worker, empty-population, cancel and error paths release their controls. |
| `test_escape.py` | The local-minimum escape mechanisms (4.5.0): every one is inert by default and leaves rank ordering, case-vector length and the bred generation untouched; neither escape objective can outrank correctness; lifespan scoring never inflates the reported fitness; jitter probes are deterministic and do not recurse; crowding displaces the nearest incumbent rather than the worst; rebirth fires only on a stall, draws diverse ancestors, keeps elites and cools down; and every GA driver calls every shared hook. |
| `test_io_placement.py` / `test_fnv.py` | Evolvable I/O binding for SNN, nervous, and LUT retains its declared strategies and compatibility behavior. FNV separately evolves relative source-pad geometry, globally fits frozen output probes, enforces source-only inputs, and measures only input-reachable physical connectivity and feedback for its final selection tie-break. |
| `test_null_models.py` | The gauntlet of idealised cheats — a run is only a result if a trivial strategy could not have produced it. |
| `test_certification.py` | The held-out verdict rule. |
| `test_designer_runtime.py` | GUI-free controller checks: playback loops, physics precedence, the Interactive case dropdown, the width strip, the capacitor charge model. |
| `test_engine_semantics.py` | Engine-level scoring and dynamics semantics. |

Two disciplines are worth stating, because both were learned from defects.

Probe every new target with idealised cheats (pass-all, first-only,
fixed-refractory) before trusting a 1.0; twice a "solved" target proved solvable
without the behaviour it was named after (5.6).

The old golden file pinned the numerical output of seven independent scoring
modes. Contract v1 deliberately changes semantics, especially logical state, so
that equivalence gate was retired rather than regenerated under a false claim of
equivalence. The permanent gate now pins behavioral invariants and anti-cheats.
`tools/benchmark_contracts.py` supplies the complementary empirical gate: a
resumable fixed-seed evolution matrix across all registered targets and all five
substrate profiles.

## Requirements

Python 3.10 or newer (the growth lookup and the mutation tests use
`int.bit_count()`; the controller's pool shutdown uses `cancel_futures`),
developed and tested on 3.12. NumPy and Matplotlib. pytest is optional, since the
suite also runs with bare Python via `py tests/run_tests.py`.
