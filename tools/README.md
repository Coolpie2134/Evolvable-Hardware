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
| `long_half_adder_campaign.py` | Current 600-generation half-adder confirmation campaign. |
| `plot_long_half_adder.py` | Regenerate the current campaign's convergence figure and report. |

The last two scripts intentionally retain their established paths so their
campaign source manifests remain interpretable. Re-running evolution against a
new source revision constitutes a new experiment; existing figures describe the
saved results, not automatically the current code.

Superseded experiment launchers, one-off diagnostics and older plotting variants
were deleted during the September 2026 cleanup. Their generated result files
remain intact. Some historical reports name scripts that are no longer shipped.

Nervous execution supports only `tri3` with `paper_analog`. Retired digital
engines and checkpoints requiring them are not supported. Shared stimulus
configuration fields may still appear in non-NV checkpoints.
