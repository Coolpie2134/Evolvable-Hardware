# Maintained tools

Run from the repository root with Python. Generated outputs belong in `results/`.

| Tool | Purpose |
| --- | --- |
| `benchmark.py` | Main configurable multi-substrate benchmark; histories, checkpoints, certification and resume support. |
| `benchmark_fnv.py` | FNV-only search and matched-budget random baseline. Run as `python -m tools.benchmark_fnv`. |
| `benchmark_contracts.py` | Resumable contract matrix for SNN, current analog NV and LUT. |
| `audit_champion_history.py` | Re-evaluate archived champions on fresh schedules with training-fitted readouts frozen. |
| `diversity_report.py` | Inspect saved population diversity and optional mutation robustness. |
| `fnv_solvable.py` | Check zero-input feasibility of FNV truth-table targets. |
| `prune_developmental_checkpoint.py` | Remove dormant rules from a checkpoint with behavioral verification. |
| `make_scoring_golden.py` | Generate the scoring regression fixture. |
| `benchmark_archive.py` | Verify the published final benchmark or extract its raw data for replotting. `verify` checks hashes, coverage, and all plotted mean curves. |
| `long_half_adder_campaign.py` | Current 600-generation half-adder confirmation campaign. |
| `plot_long_half_adder.py` | Regenerate the current campaign's convergence figure and report. |
| `multi_target_campaign.py` | Resume the capped 600-generation target sweep; `--smoke` checks the backends and `--plot` rebuilds figures. Create a `STOP` file in its result directory to pause after active jobs. |
| `supervise_campaign.py --launch` | Launch the campaign independently of the console, log process exits, and retry up to three coordinator crashes. Respects the campaign's `STOP` file. |

The last two scripts intentionally retain their established paths so their
campaign source manifests remain interpretable. Re-running evolution against a
new source revision constitutes a new experiment; existing figures describe the
saved results, not automatically the current code.

Superseded experiment launchers, one-off diagnostics and older plotting variants
were deleted during the September 2026 cleanup. Their generated result files
remain intact in the Desktop analysis archive. Some historical reports name
scripts that are no longer shipped.

The completed final campaign is included in
[benchmarks/final](../benchmarks/final/README.md). To verify it, extract its records,
and regenerate all 63 target figures without starting evolution:

```sh
python tools/benchmark_archive.py verify
python tools/benchmark_archive.py extract
python tools/multi_target_campaign.py --plot
```

Extraction refuses to overwrite an existing campaign directory. Rebuilt figures
go into `results/multi_target_600gen_20260915/figures/`; the publication copy stays
unchanged. The per-run JSON metadata records historical configurations, and
`source.zip` preserves the result-producing source and its logging amendment.
Champion and population snapshots are not part of this compact publication copy.

Older campaigns and project reports remain in the separate analysis archive.
The half-adder-only campaign scripts still require their original local
`results/` files. The desktop app and main `benchmark.py` work without them.

Nervous execution supports only `tri3` with `paper_analog`. Retired digital
engines and checkpoints requiring them are not supported. Shared stimulus
configuration fields may still appear in non-NV checkpoints.
