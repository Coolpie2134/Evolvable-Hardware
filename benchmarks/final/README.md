# Final benchmark

This is the completed `multi_target_600gen_20260915` benchmark: developmental
evolution across Nervous net, FNV, and LUT substrates. The results were copied
from the completed campaign, without rerunning evolution or changing scores.

## Scope and settings

| Setting | Value |
| --- | --- |
| Targets | 63 |
| Supported target/substrate combinations | 170 |
| Planned seeds per combination | 10 |
| Population | 60 |
| Maximum generations | 600 |
| Per-run time budget | 600 seconds; separate 900-second process limit |
| Parallelism | 4 evaluation workers per run; 2 concurrent runs |
| Recorded runs | 1,692 |
| Skipped runs | 8 |
| Measured generation records | 581,468 |

The eight skipped runs are LUT seeds 2-9 for **2x2 multiplier (temporal)**.
The first two attempts both timed out before generation 50, triggering the
predefined slow-run cutoff. All 1,692 recorded final attempts have no recorded
error; 495 stopped at the time budget. No final attempt hit the hard process limit.

## Reading the results

The graphs show **population mean fitness**, averaged across the recorded runs
of each substrate. Shading is a pointwise 95% bootstrap interval from 2,000
whole-run resamples. The graphs use the same fixed cohort throughout each curve.
They stop at its shortest available history rather than dropping slow runs.

There were **626 population-perfect stops**. A stopped run is carried forward
at fitness 1.0 for plotting only; dashed segments include these assumed tails.
They are not additional measurements in the raw records. A target axis is
shortened to the last solve only when all ten runs of all three substrates
became population-perfect. Missing substrate/target combinations are unsupported
combinations, not failed runs.

The archived scorer separately marked **629 runs as trained** and **413 as
certified** on its held-out tests. These flags are not interchangeable with a
population-perfect stop. Their recorded threshold is 0.999; a flag alone does
not establish an exact score of 1.0. Simulation results also do not demonstrate
physical hardware performance.

## Files

| File | Contents |
| --- | --- |
| [runs.csv](runs.csv) | One summary row for every planned run, including the eight skips; opens in spreadsheet software. |
| [runs.zip](runs.zip) | All 1,692 final run records with full generation histories, events, training/held-out scores, and per-run configuration metadata. |
| [figures/](figures/) | All 63 graphs in PNG, SVG, and PDF, plus the plotted means as JSON. |
| [plan.json](plan.json) / [status.json](status.json) | Original campaign settings and final status snapshot. |
| [summary.json](summary.json) | Machine-readable coverage and outcome counts. |
| [source.zip](source.zip) | Frozen result-producing Python source, the logging amendment, and before/after source manifests. |
| [provenance.json](provenance.json) | Original record hashes, source paths relative to the campaign, and export transformations. |
| [checksums.sha256](checksums.sha256) | SHA-256 hashes of every other publication file. |

Each `jobs/<target>_<substrate>/<seed-index>/result.json` in `runs.zip` contains
the measured run. Its adjacent `benchmark_metadata.json` includes the exact
saved `config`, `config_fingerprint`, generation timestamp, and cell summary.
Use these settings rather than current application defaults. The configuration
can differ by substrate or target; `effective_selection` records the selection
method actually used. Missing CSV values are blank, never assumed to be zero.

The result records preserve their original numeric values. Only JSON formatting
and machine-specific path strings were changed. `champion_index` identifies an
original archive location; champion files and large population snapshots are
not included in this publication copy. Interrupted/retried earlier attempts,
logs, and the derived Excel workbook remain in the original archive. The ZIP
contains the final selected attempt for each recorded seed, not every retry.

## Verify or regenerate graphs

From the repository root:

```sh
python tools/benchmark_archive.py verify
python tools/benchmark_archive.py extract
python tools/multi_target_campaign.py --plot
```

Extraction creates `results/multi_target_600gen_20260915/` and refuses to
overwrite an existing directory. Plotting uses the saved observations and does
not run evolution. It writes rebuilt figures into that extracted directory;
the published figures here stay unchanged. Install the root requirements first.

For independent analysis, `runs.zip` is a standard ZIP of JSON files and
`runs.csv` requires no project imports. For example:

```python
import json
from zipfile import ZipFile

with ZipFile("benchmarks/final/runs.zip") as archive:
    run = json.loads(archive.read("jobs/half-adder-temporal_fnv/00/result.json"))
    print(run["history"][0])
```

The frozen source differs from the current application. A documented
September 16 amendment made optional live-history writes tolerate a Windows
permission error; its before/after hashes and previous driver are included.
The exact historical Python/dependency versions were not recorded, so this
package supports inspecting and replotting the saved results, without promising
bit-for-bit reproduction of a new evolutionary run. The original checkpoint
files would be needed to replay individual champions.

## All target graphs

Counts below are population-perfect stops / recorded runs. A dash means the
target was not supported on that substrate. See `runs.csv` for held-out results.

| Target | Nervous net | FNV | LUT | Figure formats |
| --- | ---: | ---: | ---: | --- |
| AND (temporal) | 10/10 | 10/10 | 10/10 | [PNG](figures/and-temporal.png) / [SVG](figures/and-temporal.svg) / [PDF](figures/and-temporal.pdf) |
| XOR (temporal) | 10/10 | 10/10 | 10/10 | [PNG](figures/xor-temporal.png) / [SVG](figures/xor-temporal.svg) / [PDF](figures/xor-temporal.pdf) |
| Half adder (temporal) | 0/10 | 4/10 | 10/10 | [PNG](figures/half-adder-temporal.png) / [SVG](figures/half-adder-temporal.svg) / [PDF](figures/half-adder-temporal.pdf) |
| Toggle flip-flop | 9/10 | 10/10 | 0/10 | [PNG](figures/toggle-flip-flop.png) / [SVG](figures/toggle-flip-flop.svg) / [PDF](figures/toggle-flip-flop.pdf) |
| Echo (delay 3) | 10/10 | 10/10 | 10/10 | [PNG](figures/echo-delay-3.png) / [SVG](figures/echo-delay-3.svg) / [PDF](figures/echo-delay-3.pdf) |
| Oscillator | 10/10 | 7/10 | 10/10 | [PNG](figures/oscillator.png) / [SVG](figures/oscillator.svg) / [PDF](figures/oscillator.pdf) |
| 2-bit adder | - | 0/10 | 0/10 | [PNG](figures/2-bit-adder.png) / [SVG](figures/2-bit-adder.svg) / [PDF](figures/2-bit-adder.pdf) |
| 2-bit adder (temporal) | 0/10 | 0/10 | 0/10 | [PNG](figures/2-bit-adder-temporal.png) / [SVG](figures/2-bit-adder-temporal.svg) / [PDF](figures/2-bit-adder-temporal.pdf) |
| 2-bit comparator | - | 0/10 | 0/10 | [PNG](figures/2-bit-comparator.png) / [SVG](figures/2-bit-comparator.svg) / [PDF](figures/2-bit-comparator.pdf) |
| 2-bit comparator (temporal) | 0/10 | 0/10 | 0/10 | [PNG](figures/2-bit-comparator-temporal.png) / [SVG](figures/2-bit-comparator-temporal.svg) / [PDF](figures/2-bit-comparator-temporal.pdf) |
| 2-to-4 decoder | - | 0/10 | 0/10 | [PNG](figures/2-to-4-decoder.png) / [SVG](figures/2-to-4-decoder.svg) / [PDF](figures/2-to-4-decoder.pdf) |
| 2-to-4 decoder (temporal) | 0/10 | 0/10 | 0/10 | [PNG](figures/2-to-4-decoder-temporal.png) / [SVG](figures/2-to-4-decoder-temporal.svg) / [PDF](figures/2-to-4-decoder-temporal.pdf) |
| 2:1 MUX | - | 8/10 | 10/10 | [PNG](figures/2-1-mux.png) / [SVG](figures/2-1-mux.svg) / [PDF](figures/2-1-mux.pdf) |
| 2:1 MUX (temporal) | 5/10 | 0/10 | 5/10 | [PNG](figures/2-1-mux-temporal.png) / [SVG](figures/2-1-mux-temporal.svg) / [PDF](figures/2-1-mux-temporal.pdf) |
| 2x2 multiplier | - | 0/10 | 0/10 | [PNG](figures/2x2-multiplier.png) / [SVG](figures/2x2-multiplier.svg) / [PDF](figures/2x2-multiplier.pdf) |
| 2x2 multiplier (temporal) | 0/10 | 0/10 | 0/2 | [PNG](figures/2x2-multiplier-temporal.png) / [SVG](figures/2x2-multiplier-temporal.svg) / [PDF](figures/2x2-multiplier-temporal.pdf) |
| A-count multiple-of-3 queried by B | 0/10 | 0/10 | 0/10 | [PNG](figures/a-count-multiple-of-3-queried-by-b.png) / [SVG](figures/a-count-multiple-of-3-queried-by-b.svg) / [PDF](figures/a-count-multiple-of-3-queried-by-b.pdf) |
| A-count parity queried by B | 0/10 | 5/10 | 0/10 | [PNG](figures/a-count-parity-queried-by-b.png) / [SVG](figures/a-count-parity-queried-by-b.svg) / [PDF](figures/a-count-parity-queried-by-b.pdf) |
| A-first rendezvous | 0/10 | 1/10 | 0/10 | [PNG](figures/a-first-rendezvous.png) / [SVG](figures/a-first-rendezvous.svg) / [PDF](figures/a-first-rendezvous.pdf) |
| AND | - | 10/10 | 10/10 | [PNG](figures/and.png) / [SVG](figures/and.svg) / [PDF](figures/and.pdf) |
| Burst x3 | 7/10 | 9/10 | 9/10 | [PNG](figures/burst-x3.png) / [SVG](figures/burst-x3.svg) / [PDF](figures/burst-x3.pdf) |
| C-element (2-in join) | 0/10 | 0/10 | 0/10 | [PNG](figures/c-element-2-in-join.png) / [SVG](figures/c-element-2-in-join.svg) / [PDF](figures/c-element-2-in-join.pdf) |
| Coincidence (2-in) | 10/10 | 10/10 | 10/10 | [PNG](figures/coincidence-2-in.png) / [SVG](figures/coincidence-2-in.svg) / [PDF](figures/coincidence-2-in.pdf) |
| Collision serializer (2-to-1) | 0/10 | 0/10 | 0/10 | [PNG](figures/collision-serializer-2-to-1.png) / [SVG](figures/collision-serializer-2-to-1.svg) / [PDF](figures/collision-serializer-2-to-1.pdf) |
| Count to 4 | - | 0/10 | 0/10 | [PNG](figures/count-to-4.png) / [SVG](figures/count-to-4.svg) / [PDF](figures/count-to-4.pdf) |
| Full adder | - | 2/10 | 0/10 | [PNG](figures/full-adder.png) / [SVG](figures/full-adder.svg) / [PDF](figures/full-adder.pdf) |
| Full adder (temporal) | 0/10 | 0/10 | 0/10 | [PNG](figures/full-adder-temporal.png) / [SVG](figures/full-adder-temporal.svg) / [PDF](figures/full-adder-temporal.pdf) |
| Gap band-pass (A->B gap 2-4) | 0/10 | 2/10 | 0/10 | [PNG](figures/gap-band-pass-a-b-gap-2-4.png) / [SVG](figures/gap-band-pass-a-b-gap-2-4.svg) / [PDF](figures/gap-band-pass-a-b-gap-2-4.pdf) |
| Gated D latch | - | 0/10 | 2/10 | [PNG](figures/gated-d-latch.png) / [SVG](figures/gated-d-latch.svg) / [PDF](figures/gated-d-latch.pdf) |
| Gated oscillator | 0/10 | 0/10 | 0/10 | [PNG](figures/gated-oscillator.png) / [SVG](figures/gated-oscillator.svg) / [PDF](figures/gated-oscillator.pdf) |
| Half adder | - | 10/10 | 10/10 | [PNG](figures/half-adder.png) / [SVG](figures/half-adder.svg) / [PDF](figures/half-adder.pdf) |
| Majority-3 | - | 3/10 | 10/10 | [PNG](figures/majority-3.png) / [SVG](figures/majority-3.svg) / [PDF](figures/majority-3.pdf) |
| Majority-3 (temporal) | 7/10 | 0/10 | 10/10 | [PNG](figures/majority-3-temporal.png) / [SVG](figures/majority-3-temporal.svg) / [PDF](figures/majority-3-temporal.pdf) |
| NAND | - | 0/10 | 9/10 | [PNG](figures/nand.png) / [SVG](figures/nand.svg) / [PDF](figures/nand.pdf) |
| NAND (temporal) | 3/10 | 5/10 | 3/10 | [PNG](figures/nand-temporal.png) / [SVG](figures/nand-temporal.svg) / [PDF](figures/nand-temporal.pdf) |
| NOR | - | 0/10 | 9/10 | [PNG](figures/nor.png) / [SVG](figures/nor.svg) / [PDF](figures/nor.pdf) |
| NOR (temporal) | 2/10 | 7/10 | 0/10 | [PNG](figures/nor-temporal.png) / [SVG](figures/nor-temporal.svg) / [PDF](figures/nor-temporal.pdf) |
| OR | - | 10/10 | 10/10 | [PNG](figures/or.png) / [SVG](figures/or.svg) / [PDF](figures/or.pdf) |
| OR (temporal) | 10/10 | 10/10 | 10/10 | [PNG](figures/or-temporal.png) / [SVG](figures/or-temporal.svg) / [PDF](figures/or-temporal.pdf) |
| Odd A batch closed by B | 1/10 | 1/10 | 0/10 | [PNG](figures/odd-a-batch-closed-by-b.png) / [SVG](figures/odd-a-batch-closed-by-b.svg) / [PDF](figures/odd-a-batch-closed-by-b.pdf) |
| Odd pulse selector | - | 0/10 | - | [PNG](figures/odd-pulse-selector.png) / [SVG](figures/odd-pulse-selector.svg) / [PDF](figures/odd-pulse-selector.pdf) |
| One-shot (12 seconds) | 4/10 | 0/10 | 0/10 | [PNG](figures/one-shot-12-seconds.png) / [SVG](figures/one-shot-12-seconds.svg) / [PDF](figures/one-shot-12-seconds.pdf) |
| Parity-3 (XOR3) | - | 10/10 | 9/10 | [PNG](figures/parity-3-xor3.png) / [SVG](figures/parity-3-xor3.svg) / [PDF](figures/parity-3-xor3.pdf) |
| Parity-3 (XOR3) (temporal) | 0/10 | 5/10 | 8/10 | [PNG](figures/parity-3-xor3-temporal.png) / [SVG](figures/parity-3-xor3-temporal.svg) / [PDF](figures/parity-3-xor3-temporal.pdf) |
| Pattern (1000) | 10/10 | 8/10 | 8/10 | [PNG](figures/pattern-1000.png) / [SVG](figures/pattern-1000.svg) / [PDF](figures/pattern-1000.pdf) |
| Period doubler (2x) | 2/10 | 10/10 | 0/10 | [PNG](figures/period-doubler-2x.png) / [SVG](figures/period-doubler-2x.svg) / [PDF](figures/period-doubler-2x.pdf) |
| Period halver (1/2x) | 0/10 | 0/10 | 0/10 | [PNG](figures/period-halver-1-2x.png) / [SVG](figures/period-halver-1-2x.svg) / [PDF](figures/period-halver-1-2x.pdf) |
| Period stepper | 0/10 | 0/10 | 0/10 | [PNG](figures/period-stepper.png) / [SVG](figures/period-stepper.svg) / [PDF](figures/period-stepper.pdf) |
| Period tripler (3x) | 4/10 | 0/10 | 0/10 | [PNG](figures/period-tripler-3x.png) / [SVG](figures/period-tripler-3x.svg) / [PDF](figures/period-tripler-3x.pdf) |
| Refractory filter (3 seconds) | 2/10 | 10/10 | 0/10 | [PNG](figures/refractory-filter-3-seconds.png) / [SVG](figures/refractory-filter-3-seconds.svg) / [PDF](figures/refractory-filter-3-seconds.pdf) |
| Resettable divide-by-4 | 0/10 | 0/10 | 0/10 | [PNG](figures/resettable-divide-by-4.png) / [SVG](figures/resettable-divide-by-4.svg) / [PDF](figures/resettable-divide-by-4.pdf) |
| Resettable toggle | 8/10 | 0/10 | 0/10 | [PNG](figures/resettable-toggle.png) / [SVG](figures/resettable-toggle.svg) / [PDF](figures/resettable-toggle.pdf) |
| Rhythm cascade | 0/10 | 0/10 | 0/10 | [PNG](figures/rhythm-cascade.png) / [SVG](figures/rhythm-cascade.svg) / [PDF](figures/rhythm-cascade.pdf) |
| Ring pattern | 0/10 | 0/10 | 0/10 | [PNG](figures/ring-pattern.png) / [SVG](figures/ring-pattern.svg) / [PDF](figures/ring-pattern.pdf) |
| SR latch | 3/10 | 0/10 | 1/10 | [PNG](figures/sr-latch.png) / [SVG](figures/sr-latch.svg) / [PDF](figures/sr-latch.pdf) |
| Sequence A->B | 2/10 | 9/10 | 3/10 | [PNG](figures/sequence-a-b.png) / [SVG](figures/sequence-a-b.svg) / [PDF](figures/sequence-a-b.pdf) |
| Temporal XOR (2-in) | 9/10 | 10/10 | 10/10 | [PNG](figures/temporal-xor-2-in.png) / [SVG](figures/temporal-xor-2-in.svg) / [PDF](figures/temporal-xor-2-in.pdf) |
| Temporal sum (deltaA + deltaB) | 0/10 | 0/10 | 0/10 | [PNG](figures/temporal-sum-deltaa-deltab.png) / [SVG](figures/temporal-sum-deltaa-deltab.svg) / [PDF](figures/temporal-sum-deltaa-deltab.pdf) |
| Veto gate | 10/10 | 10/10 | 10/10 | [PNG](figures/veto-gate.png) / [SVG](figures/veto-gate.svg) / [PDF](figures/veto-gate.pdf) |
| Watchdog timeout (5 seconds) | 2/10 | 0/10 | 0/10 | [PNG](figures/watchdog-timeout-5-seconds.png) / [SVG](figures/watchdog-timeout-5-seconds.svg) / [PDF](figures/watchdog-timeout-5-seconds.pdf) |
| XNOR | - | 0/10 | 10/10 | [PNG](figures/xnor.png) / [SVG](figures/xnor.svg) / [PDF](figures/xnor.pdf) |
| XNOR (temporal) | 0/10 | 4/10 | 10/10 | [PNG](figures/xnor-temporal.png) / [SVG](figures/xnor-temporal.svg) / [PDF](figures/xnor-temporal.pdf) |
| XOR | - | 10/10 | 10/10 | [PNG](figures/xor.png) / [SVG](figures/xor.svg) / [PDF](figures/xor.pdf) |
