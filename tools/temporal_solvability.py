"""Render a temporal solvability audit from reproducible benchmark artifacts.

The report deliberately separates:

* DEMONSTRATED: a run reached fitness 1.0 on the training contract.
* SEARCH-OPEN: the backend is supported and produced behavioral gradient.
* NO-SIGNAL: the bounded run did not demonstrate target behavior.
* UNSUPPORTED: target metadata excludes the backend's present physics.

A bounded search miss is never reported as a proof of impossibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "spatial_temporal_all_smoke.json"
DEFAULT_SUPPLEMENT = ROOT / "results" / "spatial_temporal_smoke.json"
DEFAULT_HARD_SUPPLEMENT = (
    ROOT / "results" / "spatial_temporal_hard_24gen.json")
DEFAULT_FIXED = ROOT / "results" / "fixed_temporal_smoke_matched.json"
DEFAULT_OUTPUT = ROOT / "results" / "temporal_solvability_audit.md"

SUBSTRATES = (
    "snn",
    "nervous_legacy",
    "nervous_digital_tri",
    "nervous_analog_tri",
    "lut",
)

LABELS = {
    "snn": "SNN",
    "nervous_legacy": "NV legacy",
    "nervous_digital_tri": "NV digital tri",
    "nervous_analog_tri": "NV analog tri",
    "lut": "LUT",
}


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rows(*documents):
    """Merge result sets in order, allowing focused supplements to win."""
    merged = {}
    for document in documents:
        merged.update({
            (row["substrate"], row["target"]): row
            for row in document["results"]
            if row["kind"] == "temporal"
        })
    return merged


def _target_names(document):
    names = [
        row["target"] for row in document["results"]
        if row["kind"] == "temporal" and row["substrate"] == "snn"]
    if names:
        return names
    return list(dict.fromkeys(
        row["target"] for row in document["results"]
        if row["kind"] == "temporal"))


def _budget(document):
    cfg = document["config"]
    return (
        "%d generations, population %d, %d chromosomes, seed %d, I/O=%s"
        % (
            cfg["generations"], cfg["population"], cfg["chromosomes"],
            cfg["base_seed"], cfg.get("io_placement", "fixed"),
        ))


def _classification(row):
    if row is None:
        return "MISSING"
    if row["status"] == "unsupported":
        return "UNSUPPORTED"
    if row["status"] != "ok" or row.get("max_fitness") is None:
        return "ERROR"
    fitness = float(row["max_fitness"])
    if fitness >= 0.999:
        return "DEMONSTRATED"
    # Nervous selection may award a 0.05 structural-loop tier even when the
    # behavioral contract itself is zero. Do not call that behavioral gradient.
    if fitness <= 0.050001:
        return "NO-SIGNAL"
    return "SEARCH-OPEN"


def _cell(row):
    classification = _classification(row)
    if classification == "MISSING":
        return "-"
    if classification == "UNSUPPORTED":
        return "UNSUPPORTED"
    if classification == "ERROR":
        return "ERROR"
    value = float(row["max_fitness"])
    if classification == "DEMONSTRATED":
        return "YES %.3f" % value
    if classification == "NO-SIGNAL":
        return "NO-SIGNAL %.3f" % value
    return "OPEN %.3f" % value


def _summary(rows):
    lines = [
        "| Backend | Supported | Demonstrated | Search-open | No-signal | "
        "Unsupported | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for substrate in SUBSTRATES:
        backend_rows = [
            row for (backend, _target), row in rows.items()
            if backend == substrate]
        counts = {
            label: sum(
                _classification(row) == label for row in backend_rows)
            for label in (
                "DEMONSTRATED", "SEARCH-OPEN", "NO-SIGNAL",
                "UNSUPPORTED", "ERROR")}
        supported = (
            counts["DEMONSTRATED"]
            + counts["SEARCH-OPEN"]
            + counts["NO-SIGNAL"])
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d |"
            % (
                LABELS[substrate], supported, counts["DEMONSTRATED"],
                counts["SEARCH-OPEN"], counts["NO-SIGNAL"],
                counts["UNSUPPORTED"], counts["ERROR"]))
    return lines


def _comparison_table(spatial_rows, fixed_document):
    if fixed_document is None:
        return []
    fixed_rows = _rows(fixed_document)
    common_targets = list(dict.fromkeys(
        row["target"] for row in fixed_document["results"]
        if row["kind"] == "temporal"))
    lines = [
        "## Matched fixed-I/O control",
        "",
        "Each cell is `spatial - fixed` training fitness under the same target, "
        "backend, seed, generations, population, and chromosome count. Positive "
        "means spatial won that bounded run; it is not a general superiority "
        "claim.",
        "",
        "| Target | SNN | NV legacy | NV digital tri | NV analog tri | LUT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in common_targets:
        cells = []
        for substrate in SUBSTRATES:
            spatial = spatial_rows.get((substrate, target))
            fixed = fixed_rows.get((substrate, target))
            if (
                    spatial is None or fixed is None
                    or spatial.get("max_fitness") is None
                    or fixed.get("max_fitness") is None):
                cells.append("-")
            else:
                cells.append("%+.3f" % (
                    float(spatial["max_fitness"])
                    - float(fixed["max_fitness"])))
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (target.replace("|", "\\|"), *cells))
    lines.append("")
    return lines


def _matching_rows(documents, reference):
    """Rows from the document with the same experimental budget as reference."""
    if reference is None:
        return {}
    keys = (
        "generations", "population", "chromosomes", "base_seed",
        "io_placement")
    wanted = reference.get("config", {})
    # I/O intentionally differs between the spatial treatment and fixed
    # control; every other experimental coordinate must match.
    keys = tuple(key for key in keys if key != "io_placement")
    for document in documents:
        cfg = document.get("config", {})
        if all(cfg.get(key) == wanted.get(key) for key in keys):
            return _rows(document)
    return {}


def render(document, supplements=(), fixed_document=None):
    documents = (document,) + tuple(supplements)
    rows = _rows(*documents)
    names = _target_names(document)
    supported_rows = [
        row for row in rows.values()
        if _classification(row) not in ("UNSUPPORTED", "MISSING", "ERROR")]
    unsupported_rows = [
        row for row in rows.values()
        if _classification(row) == "UNSUPPORTED"]
    error_rows = [
        row for row in rows.values()
        if _classification(row) == "ERROR"]
    demonstrated_targets = {
        target for (backend, target), row in rows.items()
        if _classification(row) == "DEMONSTRATED"}
    targets_with_support = {
        target for (backend, target), row in rows.items()
        if _classification(row) not in ("UNSUPPORTED", "MISSING", "ERROR")}

    lines = [
        "# Temporal target solvability audit",
        "",
        "## Answer",
        "",
        "**No: not every temporal target is solvable on every backend under the "
        "current physics.** The benchmark contains %d supported rows and %d "
        "explicitly unsupported rows. Every one of the %d temporal targets is "
        "supported by at least one backend, but only %d targets currently have "
        "a perfect spatial-I/O training witness in these bounded runs."
        % (
            len(supported_rows), len(unsupported_rows),
            len(targets_with_support), len(demonstrated_targets)),
        "",
        "An `OPEN` or `NO-SIGNAL` cell is not an impossibility result. It means "
        "the architecture is admitted by the target contract but this search "
        "budget did not produce a perfect circuit. `YES` is training "
        "reachability only; oracle-backed winners still need frozen-readout "
        "held-out certification.",
        "",
        "Base audit: %s." % _budget(document),
    ]
    if supplements:
        lines.append(
            "Focused harder-target supplement(s): %s."
            % "; ".join(_budget(item) for item in supplements))
    lines += [
        "",
        "Legend: `YES` = fitness 1.0 demonstrated; `OPEN` = nonzero behavioral "
        "gradient; `NO-SIGNAL` = no behavior demonstrated at this budget "
        "(including the nervous 0.05 structural tier); `UNSUPPORTED` = present "
        "backend physics is explicitly excluded.",
        "",
        "| Target | SNN | NV legacy | NV digital tri | NV analog tri | LUT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        cells = [_cell(rows.get((backend, name))) for backend in SUBSTRATES]
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (name.replace("|", "\\|"), *cells))

    lines += [
        "",
        "## Backend totals",
        "",
        *_summary(rows),
        "",
        "## What is physically excluded",
        "",
        "The unsupported rows are width-sensitive contracts on substrates that "
        "do not preserve the required continuous pulse intervals. SNN excludes "
        "`Pulse width sum`, `Odd pulse selector`, and `Pair detection gap "
        "(2x pulse width)`. Digital-tri NV, analog-tri NV, and LUT exclude the "
        "first two. Legacy width-preserving NV admits all 31 temporal targets.",
        "",
        "This is a real substrate limitation, not a GA-budget problem. Making "
        "those rows pass would require adding interval/width-preserving state to "
        "the backend; weakening the target would answer a different question.",
        "",
        "## What remains a search problem",
        "",
        "All supported backends now execute recurrent temporal evaluation. The "
        "hard open families are counters/dividers, latches and rendezvous "
        "memory, serializers, watchdogs, and period transforms. Their graded "
        "scores show that the contracts are connected to behavior, but do not "
        "prove a complete mechanism is reachable by the current local mutation "
        "operators.",
        "",
        "When this historical matrix selects `spatial_chromosome`, every "
        "heritable input anchor is a developmental germline and outputs remain "
        "target-blind spatial attachments. Current Nervous application runs "
        "instead use evolved coordinate pads plus globally fitted probes.",
        "",
        "Current forward search keeps target-shaped inverse development out of "
        "Nervous and FNV. Connectivity/feedback topology supplies a target-blind "
        "stepping-stone signal; separately hand-designed witnesses remain useful "
        "only to distinguish substrate impossibility from a search miss.",
        "",
        *_comparison_table(
            _matching_rows(documents, fixed_document), fixed_document),
    ]
    if error_rows:
        lines += [
            "## Errors",
            "",
            "%d rows errored and must be rerun before drawing conclusions."
            % len(error_rows),
            "",
        ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--supplement", action="append", type=Path,
        help="focused result set that overrides matching base rows")
    parser.add_argument("--fixed-input", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    supplement_paths = (
        args.supplement if args.supplement is not None
        else [
            path for path in (DEFAULT_SUPPLEMENT, DEFAULT_HARD_SUPPLEMENT)
            if path.is_file()])
    supplements = [_load(path) for path in supplement_paths]
    fixed = (
        _load(args.fixed_input)
        if args.fixed_input is not None and args.fixed_input.is_file()
        else None)
    text = render(_load(args.input), supplements, fixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
