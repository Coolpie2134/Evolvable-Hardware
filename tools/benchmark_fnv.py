"""Deterministic search benchmark for the Functional NV Net.

This is intentionally a thin wrapper around ``runtime.controller`` so benchmark
runs exercise the same initialization, GA, process pool, scoring, and plateau
logic as the desktop application.

Example:
    py -m tools.benchmark_fnv --targets AND XOR "Half adder" --seeds 11 12 13
"""
from __future__ import annotations

import argparse
import queue
import statistics
import sys
import tempfile
import threading
import time

from runtime.config import FNVConfig, GAConfig, RunConfig
from runtime.controller import run_evolution
from substrates.nervous.targets import (
    TEMPORAL_TARGETS, periodic_combinational_target)
from substrates.snn.targets import TARGETS


def _print_line(value=""):
    """Print target names safely on legacy Windows console encodings."""
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    print(
        text.encode(encoding, "backslashreplace").decode(encoding),
        flush=True)


def _target(name):
    name = name.replace("_", " ")
    if name in TEMPORAL_TARGETS:
        return TEMPORAL_TARGETS[name]
    try:
        return periodic_combinational_target(TARGETS[name])
    except KeyError as exc:
        raise ValueError("unknown combinational target: %s" % name) from exc


def run_one(name, seed, generations, population, chromosomes, families,
            show_cases=False, selection="tournament"):
    messages = queue.Queue()
    config = RunConfig(
        ga=GAConfig(
            chromosome_count=chromosomes,
            max_telomere=8,
            mean_mutations=4.0,
            mutation_decay=0.997,
            stagnation_beta=1.0,
            selection=selection,
        ),
        fnv=FNVConfig(tuple(families)),
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fnv-benchmark-") as results:
        run_evolution(
            generations, population, chromosomes, 1, _target(name), None,
            messages, threading.Event(), base_seed=seed, backend="fnv",
            run_config=config, results_dir=results,
        )
    elapsed = time.perf_counter() - started
    best, last_mean, error, champion = 0.0, 0.0, None, None
    while not messages.empty():
        message = messages.get()
        if message[0] == "gen":
            best = max(best, float(message[3]))
            last_mean = float(message[4])
        elif message[0] == "error":
            error = message[1]
        elif message[0] == "done":
            champion = message[1]
    if error is not None:
        raise RuntimeError(error)
    case_scores = ()
    if show_cases and champion is not None:
        from substrates.fnv.evaluation import evaluate_functional_full
        _, case_scores = evaluate_functional_full(champion, _target(name))
    return best, last_mean, elapsed, case_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", nargs="+",
        default=["AND", "XOR", "Half adder", "2:1 MUX"])
    parser.add_argument(
        "--all-temporal", action="store_true",
        help="benchmark every temporal target that declares FNV support")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--chromosomes", type=int, default=2)
    parser.add_argument(
        "--selection", choices=("tournament", "lexicase"),
        default="tournament")
    parser.add_argument(
        "--show-cases", action="store_true",
        help="print the winning contract case vector after each row")
    parser.add_argument(
        "--families", nargs="+",
        default=list(FNVConfig().families))
    args = parser.parse_args(argv)
    targets = args.targets
    if args.all_temporal:
        targets = [
            name for name, target in TEMPORAL_TARGETS.items()
            if not getattr(target, "supported_backends", ())
            or "fnv" in target.supported_backends
        ]

    rows = []
    header = "target,seed,best,last_mean,seconds"
    _print_line(header)
    for name in targets:
        display_name = name.replace("_", " ")
        for seed in args.seeds:
            best, mean, elapsed, case_scores = run_one(
                display_name, seed, args.generations, args.population,
                args.chromosomes, args.families,
                args.show_cases, args.selection)
            rows.append((
                display_name, seed, best, mean, elapsed))
            line = "%s,%d,%.6f,%.6f,%.3f" % (
                display_name, seed, best, mean, elapsed)
            _print_line(line)
            if args.show_cases:
                _print_line("  cases=" + repr(tuple(case_scores)))
    _print_line()
    for name in targets:
        display_name = name.replace("_", " ")
        values = [row[2] for row in rows if row[0] == display_name]
        _print_line("%s: mean-best %.6f  solved %d/%d" % (
            display_name, statistics.mean(values),
            sum(value >= 0.999999 for value in values), len(values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
