#!/usr/bin/env python3
"""
tools/benchmark.py - headless solvability benchmark, driven through the GUI's
own evolution worker.

WHY THIS EXISTS, AND WHY IT USES runtime/controller.py
------------------------------------------------------
The project has two GA drive paths: each backend's standalone
``substrates/*/ga.py::evolve_*`` and ``runtime/controller.py``, which is what
the desktop application actually runs. They have drifted before - a mechanism
lands in one, and the benchmarks measure the other (tools/benchmark_contracts.py
drives the standalone path). This tool deliberately drives the CONTROLLER, so a
benchmark number is a claim about the evolution the application performs, and
every control on the GUI has a flag here with the same meaning.

WHAT IT MEASURES
----------------
CERTIFIED solve rate: the fraction of seeds whose champion both reached a
training fitness of 1.0 and passed ``substrates.nervous.certification.certify``
on FRESH held-out stimulus. A training-only "solve" is recorded but is never
counted as solvable - held-out certification exists precisely because a high
training fitness can be memorised timing.

Cells are classified from the controller's own certification verdict:

    CERTIFIED     trained to 1.0 and generalises            (counted solvable)
    OVERFIT       trained to 1.0, held-out collapsed        (not solvable)
    UNCERTIFIED   trained to 1.0, target has no oracle      (not counted; the
                  claim is unverifiable, not false - reported separately)
    BELOW         did not reach the bar at this budget      (not solvable)

OUTPUT
------
The terminal shows the plan, generation progress, per-seed result, and final
targets x architectures table. A local JSON record is rewritten atomically
after every seed so an interrupted sweep can resume; a local Markdown copy of
the final table is written beside it. The script performs no version-control,
network, or remote-publication operations.

USAGE
-----
    py tools/benchmark.py --list-targets
    py tools/benchmark.py --architectures nervous --targets temporal --seeds 3
    py tools/benchmark.py --architectures paper --targets "Half adder,Full adder"
    py tools/benchmark.py --report-only --out results/benchmark.json
    py tools/benchmark.py --architectures nervous,lut --gens 40 --pop 60 
        --crowding --neutral-drift
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from runtime.config import (                                       # noqa: E402
    DEFAULT_EVALUATION_WORKERS, FNV_FAMILIES, FNVConfig, GAConfig,
    MAX_EVALUATION_WORKERS, NV_NEW_RUN_PROFILES, RunConfig,
    default_max_telomere)
from runtime.controller import SOLVER_VALID, run_evolution         # noqa: E402
from runtime.escape import EscapeConfig                            # noqa: E402
from runtime.limits import MAX_CHROMOSOME_COUNT                    # noqa: E402
from runtime.mutation import DEFAULT_MUTATION_LIMIT                # noqa: E402
from substrates.nervous.ga import (                                # noqa: E402
    IMMIGRANT_FRAC, MEAN_MUTATIONS, MUT_DECAY, TOURNAMENT_K)
from substrates.nervous.pulse import COINC, DELAY, WIDTH, PulseConfig  # noqa: E402
from substrates.nervous.targets import (                           # noqa: E402
    TEMPORAL_TARGETS, periodic_combinational_target)
from substrates.snn.snn import DEFAULT_ARCH, Arch                  # noqa: E402
from substrates.snn.targets import TARGETS                         # noqa: E402

# LUT function-family banks are a newer GUI control. Imported tolerantly so this
# tool still runs against a checkout that predates them; when the constant is
# absent the flag disappears and GAConfig keeps its own default.
try:
    from runtime.config import LUT_FUNCTION_FAMILIES                # noqa: E402
except ImportError:                                                 # pragma: no cover
    LUT_FUNCTION_FAMILIES = ()


SCHEMA = 'solvability-benchmark/v2'
ARCHITECTURES = ('nervous', 'fnv', 'lut', 'snn')
ARCHITECTURE_SETS = {
    'paper': ('nervous', 'lut'),
    'nv': ('nervous', 'fnv'),
    'cellular': ('nervous', 'fnv', 'lut'),
    'all': ARCHITECTURES,
}
ARCHITECTURE_DESCRIPTIONS = {
    'nervous': 'paper-inspired analog tri-circuit nervous net',
    'fnv': 'functional NV net with selectable component families',
    'lut': 'square directional-LUT array with selectable I/O and gate banks',
    'snn': 'leaky integrate-and-fire comparison substrate',
}
DEFAULT_OUT = os.path.join(ROOT, 'results', 'benchmark.json')

#: Backends whose controller path runs held-out certification. SNN has no
#: oracle-replay path in the controller, so an SNN cell can never be CERTIFIED;
#: the report says so rather than charting it as a failure to generalise.
CERTIFYING_BACKENDS = ('nervous', 'lut', 'fnv')

VERDICT_SYMBOL = {
    'certified': 'OK',
    'overfit': 'XX',
    'uncertified': '??',
    'below': '..',
    'error': '!!',
    'none': '--',
}


# ----------------------------- target selection -----------------------------

def all_targets():
    """Every registered target, combinational and temporal (GUI _all_targets)."""
    catalogue = dict(TARGETS)
    catalogue.update(TEMPORAL_TARGETS)
    return catalogue


def targets_for_backend(backend, node_model=None):
    """The GUI's ``_targets_for_backend``, reproduced exactly.

    Nervous/LUT see combinational targets through the periodic pulse wrapper;
    FNV evaluates its native level gates on held exhaustive rows, and SNN sees
    the raw registry. Targets that
    declare ``supported_backends`` / ``supported_models`` are filtered out where
    their physics is unattainable, rather than being allowed to silently cap
    below 1.0 and read as an evolution failure.
    """
    catalogue = all_targets()
    if backend in ('nervous', 'lut'):
        catalogue = {
            name: (target if getattr(target, 'temporal', False)
                   else periodic_combinational_target(target))
            for name, target in catalogue.items()
        }
    model = node_model if backend == 'nervous' else None
    return {
        name: target for name, target in catalogue.items()
        if (not getattr(target, 'supported_backends', ())
            or backend in target.supported_backends
            or (backend == 'fnv'
                and any(candidate in target.supported_backends
                        for candidate in ('nervous', 'lut'))))
        if (model is None
            or not getattr(target, 'supported_models', ())
            or model in target.supported_models)
    }


def resolve_target_names(requested, backend, node_model):
    """Expand ``--targets`` selectors against one backend's catalogue.

    Accepts the keywords ``all`` / ``temporal`` / ``combinational`` and exact
    target names (case-insensitive). A name that exists in the registry but not
    for THIS backend is skipped with a note, so one multi-backend sweep can name
    a target that only some backends' physics supports; a name that is not a
    target at all is an error rather than a silently empty sweep.

    Returns ``(names, unsupported)``.
    """
    available = targets_for_backend(backend, node_model)
    lowered = {name.lower(): name for name in available}
    registry = {name.lower(): name for name in all_targets()}
    selected, unsupported, unknown = [], [], []
    for token in requested:
        token = token.strip()
        if not token:
            continue
        key = token.lower()
        if key == 'all':
            selected.extend(available)
        elif key == 'temporal':
            selected.extend(name for name in available
                            if name in TEMPORAL_TARGETS)
        elif key == 'combinational':
            selected.extend(name for name in available if name in TARGETS)
        elif key == 'logic-temporal':
            # The coincident-edge twins of the truth tables. They are genuine
            # temporal targets, so `temporal` already includes them; this picks
            # out just the derived set, or lets a sweep exclude it by asking for
            # the other categories instead.
            selected.extend(name for name in available
                            if name.endswith('(temporal)')
                            and name in TEMPORAL_TARGETS)
        elif key in lowered:
            selected.append(lowered[key])
        elif key in registry:
            unsupported.append(registry[key])
        else:
            unknown.append(token)
    if unknown:
        raise SystemExit(
            'unknown target(s): %s\n'
            '(run --list-targets to see what each backend supports)'
            % ', '.join(unknown))
    # dict.fromkeys keeps the registry order and drops duplicates.
    return list(dict.fromkeys(selected)), list(dict.fromkeys(unsupported))


def effective_target(target, high, graded):
    """The GUI's ``_effective_target``: a fresh copy, with the truth-table knobs
    applied only where they exist (temporal targets have neither)."""
    if getattr(target, 'temporal', False):
        return dataclasses.replace(target)
    return dataclasses.replace(
        target, high=(target.high if high is None else high), graded=graded)


# ----------------------------- configuration --------------------------------

def build_escape_config(args):
    return EscapeConfig(
        lifespan_scoring=args.lifespan,
        lifespan_checkpoints=args.lifespan_stages,
        crowding=args.crowding,
        crowding_window=args.crowding_window,
        crowding_fraction=args.crowding_reserve,
        neutral_drift=args.neutral_drift,
        self_adaptive_mutation=args.self_adaptive_mutation,
        rebirth=args.rebirth,
        rebirth_patience=args.rebirth_stall,
        rebirth_fraction=args.rebirth_fraction,
        lineage_walk=args.lineage_walk,
        lineage_walk_fraction=args.lineage_walk_fraction,
        robustness=args.robustness,
        robustness_jitter=args.robustness_jitter,
        islands=args.islands,
        island_count=args.island_demes,
        island_migration_interval=args.island_migrate,
        lexicase_downsample=args.lexicase_sample)


def build_run_config(args, backend, chromosome_count):
    """One immutable RunConfig, assembled from the same fields the GUI reads."""
    tile_arch, node_model, evolve_delay = NV_NEW_RUN_PROFILES[args.nv_profile]
    analog = {}
    if node_model == 'paper_analog':
        analog = dict(analog_threshold=args.analog_vth,
                      analog_step=args.analog_step,
                      analog_tau_leak=args.analog_tau,
                      analog_hysteresis=args.analog_hysteresis)
    pulse = PulseConfig(delay=args.delay, width=args.width,
                        coincidence=args.coincidence, model=node_model,
                        **analog)
    max_telomere = (default_max_telomere(backend) if args.max_telomere is None
                    else args.max_telomere)
    extra = ({'lut_function_families': tuple(args.lut_function_families)}
             if LUT_FUNCTION_FAMILIES else {})
    return RunConfig(
        ga=GAConfig(
            mean_mutations=args.mutations,
            immigrant_fraction=args.immigrants,
            mutation_limit=args.mutation_cap,
            tournament_size=args.tournament,
            elite_count=args.elites,
            mutation_decay=args.anneal,
            stagnation_beta=args.plateau_beta,
            selection='lexicase' if args.lexicase else 'tournament',
            recombination_enabled=not args.no_recombination,
            max_telomere=max_telomere,
            node_model=node_model,
            evolve_delay=evolve_delay,
            tile_arch=tile_arch,
            io_placement=args.io_placement,
            lut_io_mode=args.lut_io_mode,
            chromosome_count=chromosome_count,
            evaluation_workers=args.workers,
            diversify_solvers=args.diversify_solvers,
            pure_evolution=args.pure_evolution,
            plateau_rescue_limit=(None if args.rescue_limit < 0
                                  else args.rescue_limit),
            escape=build_escape_config(args),
            **extra),
        pulse=pulse,
        fnv=FNVConfig(tuple(args.fnv_families), args.fnv_readout))


def build_arch(args, target, backend):
    """The GUI's ``_read_arch``. SNN-only; the other backends ignore it."""
    vmin, vmax = args.vth_min, args.vth_max
    levels = tuple(round(vmin + (vmax - vmin) * i / 3.0, 4) for i in range(4))
    recurrent = backend == 'snn' and getattr(target, 'temporal', False)
    high = target.high if args.input_high is None else args.input_high
    return Arch(syn_weight=args.syn_weight, vth_levels=levels,
                tau_levels=DEFAULT_ARCH.tau_levels, recurrent=recurrent), high


def architecture_summary(architecture, args):
    """Compact resolved configuration shown by ``--dry-run``."""
    if architecture == 'nervous':
        tile, model, _delay = NV_NEW_RUN_PROFILES[args.nv_profile]
        return '%s/%s' % (tile, model)
    if architecture == 'fnv':
        return 'families=%s; readout=%s' % (
            '+'.join(args.fnv_families), args.fnv_readout)
    if architecture == 'lut':
        return '%s; functions=%s' % (
            args.lut_io_mode, '+'.join(args.lut_function_families))
    return 'recurrent for temporal targets'


def config_record(args, architectures):
    """The full resolved option surface, recorded in the JSON and hashed for
    resume safety. Every GUI control appears here."""
    tile_arch, node_model, evolve_delay = NV_NEW_RUN_PROFILES[args.nv_profile]
    return {
        'driver': 'runtime.controller.run_evolution',
        'architectures': list(architectures),
        'targets': list(args.targets),
        'exclude': list(args.exclude),
        'seeds': args.seeds,
        'seed_base': args.seed_base,
        'run': {
            'population': args.pop, 'generations': args.gens,
            'restarts': args.tries, 'chromosomes': args.chroms,
            'workers': args.workers,
            'graded': args.graded, 'input_high': args.input_high,
            # Both change what a cell can reach, so they belong in the
            # fingerprint: resuming a capped sweep into an uncapped one would
            # mix incomparable rows.
            'time_cap': args.time_cap, 'stop_on_solve': args.stop_on_solve,
            'rescue_limit': args.rescue_limit,
        },
        'ga': {
            'mutations': args.mutations, 'immigrants': args.immigrants,
            'tournament': args.tournament, 'elites': args.elites,
            'anneal': args.anneal, 'plateau_beta': args.plateau_beta,
            'mutation_cap': args.mutation_cap, 'lexicase': args.lexicase,
            'recombination': not args.no_recombination,
            'diversify_solvers': args.diversify_solvers,
            'max_telomere': args.max_telomere,
            'resolved_max_telomere': {
                backend: (default_max_telomere(backend)
                          if args.max_telomere is None else args.max_telomere)
                for backend in architectures},
        },
        'substrate': {
            'syn_weight': args.syn_weight, 'vth_min': args.vth_min,
            'vth_max': args.vth_max,
        },
        'pulse': {
            'delay': args.delay, 'width': args.width,
            'coincidence': args.coincidence,
            'nv_profile': args.nv_profile, 'tile_arch': tile_arch,
            'node_model': node_model, 'evolve_delay': evolve_delay,
            'analog_vth': args.analog_vth, 'analog_step': args.analog_step,
            'analog_tau': args.analog_tau,
            'analog_hysteresis': args.analog_hysteresis,
        },
        'io': {
            'io_placement': args.io_placement,
            'lut_io_mode': args.lut_io_mode,
            'lut_function_families': list(args.lut_function_families),
        },
        'fnv_families': list(args.fnv_families),
        'fnv_readout': args.fnv_readout,
        'escape': dataclasses.asdict(build_escape_config(args)),
        'certification': {
            'solver_valid': SOLVER_VALID,
            'certifying_backends': list(CERTIFYING_BACKENDS),
        },
    }


def config_fingerprint(config):
    """Stable hash of everything that changes a result, so resuming a sweep
    under different settings is refused instead of mixing incomparable rows."""
    payload = {key: value for key, value in config.items()
               if key not in ('targets', 'exclude')}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def cell_seed(base, backend, target_name, index):
    """Per-(backend, target, index) seed, so no two cells walk the same RNG
    path and a single-cell rerun reproduces the sweep's number exactly."""
    digest = hashlib.sha256(
        ('%s|%s' % (backend, target_name)).encode('utf-8')).digest()
    return int(base) + index * 1_000_003 + int.from_bytes(digest[:4], 'big') % 1_000_000


# ------------------------------ one evolution -------------------------------

def classify_seed(best_fit, certification, backend, error):
    """Map one run onto the report's categories (see the module docstring)."""
    if error:
        return 'error'
    solved = best_fit >= SOLVER_VALID
    if certification is None:
        # No certification attempted. SNN has no controller-side oracle replay,
        # so it can never be certified; on a certifying backend a missing
        # verdict on a solved run means certify() itself failed, which is an
        # unverified claim - not a pass.
        if not solved or backend not in CERTIFYING_BACKENDS:
            return 'none'
        return 'uncertified'
    verdict = str(certification.get('verdict') or '')
    if verdict.startswith('CERTIFIED'):
        # Certification alone is not a solve: the training contract still has
        # to have been met, otherwise a partially-correct circuit that happens
        # to generalise would be charted as solvable.
        return 'certified' if solved else 'below'
    if verdict.startswith('OVERFIT'):
        return 'overfit'
    if verdict.startswith('UNCERTIFIED'):
        return 'uncertified' if solved else 'below'
    return 'below'


def run_one(backend, target_name, target, args, seed, snapshot_dir, quiet):
    """Evolve one (backend, target, seed) cell through the controller."""
    run_config = build_run_config(args, backend, args.chroms)
    arch, high = build_arch(args, target, backend)
    live_target = effective_target(target, high, args.graded)
    setattr(live_target, 'pulse_config', run_config.pulse)
    setattr(live_target, 'io_placement', run_config.ga.io_placement)
    if backend == 'fnv':
        setattr(live_target, '_fnv_families', run_config.fnv.families)
    if backend == 'lut':
        setattr(live_target, 'lut_io_mode', run_config.ga.lut_io_mode)

    messages = queue.Queue()
    stop_event = threading.Event()
    state = {'certification': None, 'error': None, 'best': 0.0,
             'first_solved_gen': None, 'last_gen': 0, 'done': False,
             'stopped_early': None}

    started = time.time()
    time_cap = float(getattr(args, 'time_cap', 0) or 0)
    stop_on_solve = bool(getattr(args, 'stop_on_solve', False))

    def budget(best_fit):
        """Why this run should end early, or None to keep going.

        Checked once per generation by the controller. Ending here is an
        ORDINARY end-of-run - the champion is kept and still certified - unlike
        the stop_event, which means the user aborted.
        """
        if stop_on_solve and best_fit >= 1.0:
            return 'solved'
        if time_cap and (time.time() - started) >= time_cap:
            return 'time cap %gs' % time_cap
        return None

    def worker():
        try:
            run_evolution(
                args.gens, args.pop, args.chroms, args.tries, live_target,
                arch, messages, stop_event, base_seed=seed, backend=backend,
                run_config=run_config, results_dir=snapshot_dir,
                budget=(budget if (time_cap or stop_on_solve) else None))
        except BaseException:                       # noqa: BLE001 - recorded
            import traceback
            messages.put(('error', traceback.format_exc(limit=5)))
            messages.put(('done', None, 0.0))

    # `started` is set above, before the budget closure that reads it.
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Drain live so a long terminal run prints progress instead of going silent,
    # so a Ctrl-C reaches the controller's own cooperative stop.
    absolute_gen = 0
    try:
        while not state['done']:
            try:
                message = messages.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            kind = message[0]
            if kind == 'gen':
                _try_i, generation, best_fit = message[1], message[2], message[3]
                absolute_gen += 1
                state['best'] = max(state['best'], best_fit)
                state['last_gen'] = generation
                if (state['first_solved_gen'] is None
                        and best_fit >= SOLVER_VALID):
                    state['first_solved_gen'] = absolute_gen
                if not quiet and args.progress_every and generation and (
                        generation % args.progress_every == 0):
                    print('      gen %4d  best %.3f  mean %.3f'
                          % (generation, best_fit, message[4]), flush=True)
            elif kind == 'budget':
                state['stopped_early'] = message[1]
            elif kind == 'certified':
                state['certification'] = message[1]
            elif kind == 'error':
                state['error'] = message[1]
            elif kind == 'done':
                state['best'] = max(state['best'], message[2] or 0.0)
                state['done'] = True
    except KeyboardInterrupt:
        stop_event.set()
        thread.join()
        raise
    thread.join()

    certification = state['certification']
    category = classify_seed(state['best'], certification, backend,
                             state['error'])
    return {
        'seed': seed,
        'best': round(float(state['best']), 6),
        'train': (None if certification is None
                  else certification.get('train')),
        'holdout': (None if certification is None
                    else certification.get('holdout')),
        'holdouts': (None if certification is None
                     else certification.get('holdouts')),
        'verdict': (None if certification is None
                    else certification.get('verdict')),
        'category': category,
        'certified': category == 'certified',
        'trained': state['best'] >= SOLVER_VALID,
        'first_solved_gen': state['first_solved_gen'],
        'generations': state['last_gen'],
        # Why the run ended before its generation budget, if it did. A
        # capped run is NOT evidence the target is unreachable, so the
        # reason has to survive into the report.
        'stopped_early': state['stopped_early'],
        'elapsed_s': round(time.time() - started, 2),
        'error': state['error'],
    }


# ------------------------------ result store --------------------------------

def replace_atomic(source, destination):
    """Replace despite the brief OneDrive/antivirus sharing locks this repo
    lives under (same idiom as tools/benchmark_contracts.py)."""
    destination = os.path.abspath(destination)
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def write_json(path, document):
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        'w', delete=False, dir=directory, suffix='.tmp', encoding='utf-8')
    try:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    replace_atomic(handle.name, path)


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def summarise_cell(cell):
    """Roll a cell's seeds up into the numbers the grid shows."""
    seeds = cell['seeds']
    n = len(seeds)
    counts = {}
    for seed in seeds:
        counts[seed['category']] = counts.get(seed['category'], 0) + 1
    certified = counts.get('certified', 0)
    scores = [seed['best'] for seed in seeds if seed['error'] is None]
    holdouts = [seed['holdout'] for seed in seeds
                if seed.get('holdout') is not None]
    cell['n'] = n
    cell['counts'] = counts
    cell['certified'] = certified
    cell['trained'] = sum(1 for seed in seeds if seed['trained'])
    cell['certified_rate'] = (certified / n) if n else 0.0
    cell['best_max'] = round(max(scores), 6) if scores else 0.0
    cell['best_mean'] = round(sum(scores) / len(scores), 6) if scores else 0.0
    cell['holdout_mean'] = (round(sum(holdouts) / len(holdouts), 6)
                            if holdouts else None)
    cell['elapsed_s'] = round(sum(seed['elapsed_s'] for seed in seeds), 2)
    cell['solve_gens'] = sorted(
        seed['first_solved_gen'] for seed in seeds
        if seed.get('first_solved_gen') is not None)
    return cell


#: A first solve landing this far into the budget counts as "late". Used only
#: to caveat a rate, never to change one.
LATE_SOLVE_FRACTION = 0.6


def budget_caveat(cell, budget):
    """Say whether a cell's certified rate is limited by ``--gens``.

    Returns 'truncated', 'no-solves', or None. This reads recorded solve
    generations only; it never re-runs or re-scores anything.
    """
    gens = cell.get('solve_gens') or []
    unsolved = cell['n'] - len(gens)
    if not gens:
        # Nothing solved, so there is no solve-time distribution to argue
        # from. Such a cell bounds nothing: it cannot separate "unreachable"
        # from "needs more than this budget", and that is the exact reading
        # that turns a short probe into a false structural conclusion.
        return 'no-solves' if cell['n'] else None
    if unsolved and gens[-1] >= LATE_SOLVE_FRACTION * float(budget):
        # Seeds were still solving for the first time near the cutoff while
        # others had not solved at all: the rate is a lower bound.
        return 'truncated'
    return None


# ------------------------------- reporting ----------------------------------

def _cell_label(cell):
    if cell is None:
        return '-'
    if cell.get('counts', {}).get('error'):
        return '!! err'
    dominant = max(cell['counts'], key=lambda key: cell['counts'][key]) \
        if cell['counts'] else 'none'
    symbol = VERDICT_SYMBOL.get(
        'certified' if cell['certified'] else dominant, '--')
    return '%s %d/%d' % (symbol, cell['certified'], cell['n'])


def render_markdown(document):
    config = document['config']
    cells = {(cell['backend'], cell['target']): summarise_cell(cell)
             for cell in document['cells']}
    architectures = config.get('architectures', config.get('backends', ()))
    backends = [backend for backend in architectures
                if any(key[0] == backend for key in cells)]
    targets = list(dict.fromkeys(cell['target'] for cell in document['cells']))

    lines = []
    lines.append('# Solvability benchmark')
    lines.append('')
    lines.append('Generated %s  -  local driver `%s`'
                 % (document['generated_utc'], config['driver']))
    run = config['run']
    lines.append('')
    lines.append('Budget: **%d generations x population %d**, %d restart(s), '
                 '%d chromosomes, %d seed(s) from base %s.'
                 % (run['generations'], run['population'], run['restarts'],
                    run['chromosomes'], config['seeds'], config['seed_base']))
    escape = [name for name, value in config['escape'].items()
              if value is True]
    downsample = config['escape'].get('lexicase_downsample', 1.0)
    if downsample < 1.0:
        escape.append('lexicase_downsample=%.2f' % downsample)
    lines.append('')
    lines.append('Physics: NV profile `%s` (%s / %s), delay %s, width %s, '
                 'coincidence %s.  I/O: `%s`, LUT `%s`.'
                 % (config['pulse']['nv_profile'], config['pulse']['tile_arch'],
                    config['pulse']['node_model'], config['pulse']['delay'],
                    config['pulse']['width'], config['pulse']['coincidence'],
                    config['io']['io_placement'], config['io']['lut_io_mode']))
    lines.append('')
    lines.append('Escape mechanisms: %s'
                 % (', '.join('`%s`' % name for name in escape) if escape
                    else 'none (all off)'))
    lines.append('')
    lines.append('## Certified solve rate')
    lines.append('')
    lines.append('A cell counts a seed as solvable only when its champion '
                 'reached training fitness >= %.3f **and** passed held-out '
                 'certification. `%s` = certified, `%s` = trained but overfit '
                 '(memorised timing), `%s` = trained but the target has no '
                 'oracle to certify against, `%s` = not solved at this budget.'
                 % (SOLVER_VALID, VERDICT_SYMBOL['certified'],
                    VERDICT_SYMBOL['overfit'], VERDICT_SYMBOL['uncertified'],
                    VERDICT_SYMBOL['below']))
    lines.append('')
    header = '| Target | ' + ' | '.join(backends) + ' |'
    lines.append(header)
    lines.append('| --- | ' + ' | '.join('---' for _ in backends) + ' |')
    for target in targets:
        row = ['| ' + target]
        for backend in backends:
            row.append(_cell_label(cells.get((backend, target))))
        lines.append(' | '.join(row) + ' |')

    # Per-backend totals.
    lines.append('')
    lines.append('| Backend | Certified | Trained | Targets | Wall clock |')
    lines.append('| --- | --- | --- | --- | --- |')
    for backend in backends:
        rows = [cell for key, cell in cells.items() if key[0] == backend]
        seeds_total = sum(cell['n'] for cell in rows)
        certified = sum(cell['certified'] for cell in rows)
        trained = sum(cell['trained'] for cell in rows)
        elapsed = sum(cell['elapsed_s'] for cell in rows)
        note = ('' if backend in CERTIFYING_BACKENDS
                else ' *(no certification path)*')
        lines.append('| %s%s | %d/%d (%.0f%%) | %d/%d | %d | %s |'
                     % (backend, note, certified, seeds_total,
                        100.0 * certified / seeds_total if seeds_total else 0.0,
                        trained, seeds_total, len(rows),
                        _duration(elapsed)))

    # Budget caveats. A certified rate is only meaningful next to the budget it
    # was measured at: a target still solving for the first time near the
    # cutoff has a rate that is a lower bound, and a target that never solved
    # has no rate at all. Without this, a short probe reads as a structural
    # verdict.
    budget = run['generations']
    truncated = [(backend, target)
                 for target in targets for backend in backends
                 if cells.get((backend, target)) is not None
                 and budget_caveat(cells[(backend, target)],
                                   budget) == 'truncated']
    unsolved = [(backend, target)
                for target in targets for backend in backends
                if cells.get((backend, target)) is not None
                and budget_caveat(cells[(backend, target)],
                                  budget) == 'no-solves']
    if truncated or unsolved:
        lines.append('')
        lines.append('### Budget caveats')
        lines.append('')
    if truncated:
        lines.append(
            'These cells were **still producing first-time solves near the '
            '%d-generation cutoff** while other seeds had not solved at all. '
            'Their certified rate is a lower bound - raising `--gens` is '
            'expected to raise it:' % budget)
        lines.append('')
        for backend, target in truncated:
            gens = cells[(backend, target)]['solve_gens']
            lines.append('- `%s` / %s - solves at generations %s'
                         % (backend, target,
                            ', '.join(str(value) for value in gens)))
        lines.append('')
    if unsolved:
        lines.append(
            'These cells recorded **no solve at all**, so this run bounds '
            'nothing: at a %d-generation budget it cannot distinguish '
            '"unreachable on this substrate" from "needs a longer run". Do '
            'not read them as a structural limit without re-running past '
            'the target\'s known solve-generation spread:' % budget)
        lines.append('')
        for backend, target in unsolved:
            lines.append('- `%s` / %s - 0/%d solved'
                         % (backend, target, cells[(backend, target)]['n']))
        lines.append('')

    lines.append('')
    lines.append('## Cell detail')
    lines.append('')
    lines.append('| Backend | Target | Certified | Best (max) | Best (mean) | '
                 'Held-out (mean) | Solve gen (min/med/max) | Verdicts |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for target in targets:
        for backend in backends:
            cell = cells.get((backend, target))
            if cell is None:
                continue
            verdicts = ', '.join(
                '%s x%d' % (name, count)
                for name, count in sorted(cell['counts'].items()))
            solve_gens = cell.get('solve_gens') or []
            lines.append(
                '| %s | %s | %d/%d | %.3f | %.3f | %s | %s | %s |'
                % (backend, target, cell['certified'], cell['n'],
                   cell['best_max'], cell['best_mean'],
                   ('%.3f' % cell['holdout_mean']
                    if cell['holdout_mean'] is not None else '-'),
                   ('%d/%d/%d' % (solve_gens[0],
                                  solve_gens[len(solve_gens) // 2],
                                  solve_gens[-1])
                    if solve_gens else '-'),
                   verdicts))

    skipped = document.get('skipped') or []
    if skipped:
        lines.append('')
        lines.append('## Skipped (unsupported)')
        lines.append('')
        lines.append('Excluded by the target\'s own `supported_backends` / '
                     '`supported_models` declaration - physically unattainable '
                     'here, not a search failure.')
        lines.append('')
        for entry in skipped:
            lines.append('* `%s` / %s' % (entry['backend'], entry['target']))

    errors = [(cell['backend'], cell['target'], seed['error'])
              for cell in document['cells'] for seed in cell['seeds']
              if seed['error']]
    if errors:
        lines.append('')
        lines.append('## Errors')
        lines.append('')
        for backend, target, error in errors:
            first = error.strip().splitlines()[-1] if error.strip() else '?'
            lines.append('* `%s` / %s - %s' % (backend, target, first))
    lines.append('')
    return '\n'.join(lines)


def _duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return '%ds' % seconds
    if seconds < 3600:
        return '%dm %02ds' % (seconds // 60, seconds % 60)
    return '%dh %02dm' % (seconds // 3600, (seconds % 3600) // 60)


# -------------------------------- the sweep ---------------------------------

def run_sweep(args, architectures):
    config = config_record(args, architectures)
    fingerprint = config_fingerprint(config)
    document = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'config': config,
        'config_fingerprint': fingerprint,
        'cells': [],
    }

    done = {}
    if args.resume and os.path.exists(args.out):
        previous = load_json(args.out)
        if previous.get('config_fingerprint') != fingerprint:
            raise SystemExit(
                'refusing to resume %s: it was produced under a different '
                'configuration (%s != %s). Use a different --out, or drop '
                '--resume to overwrite.'
                % (args.out, previous.get('config_fingerprint'), fingerprint))
        for cell in previous.get('cells', []):
            done[(cell['backend'], cell['target'])] = cell
        document['cells'] = list(previous['cells'])
        print('resuming %s - %d cell(s) already recorded'
              % (args.out, len(done)), flush=True)

    plan = []
    skipped = []
    for backend in architectures:
        _tile, node_model, _delay = NV_NEW_RUN_PROFILES[args.nv_profile]
        catalogue = targets_for_backend(backend, node_model)
        names, unsupported = resolve_target_names(
            args.targets, backend, node_model)
        excluded = {name.lower() for name in args.exclude}
        names = [name for name in names if name.lower() not in excluded]
        for name in names:
            plan.append((backend, name, catalogue[name]))
        skipped.extend((backend, name) for name in unsupported
                       if name.lower() not in excluded)
    for backend, name in skipped:
        # A declared supported_backends/supported_models restriction, not a
        # search failure - say so instead of charting a physically unattainable
        # cell as an evolution that could not solve it.
        print('skipping %-8s %s - unsupported by this backend/node model'
              % (backend, name), flush=True)
    document['skipped'] = [{'backend': backend, 'target': name}
                           for backend, name in skipped]

    if not plan:
        raise SystemExit('nothing to run - the target selection is empty')

    total_cells = len(plan)
    print('%d cell(s) x %d seed(s) = %d run(s) of %d generations'
          % (total_cells, args.seeds, total_cells * args.seeds, args.gens),
          flush=True)
    if args.dry_run:
        for backend, name, _target in plan:
            print('  %-8s [%-38s] %s'
                  % (backend, architecture_summary(backend, args), name))
        return document

    snapshot_dir = args.snapshot_dir or tempfile.mkdtemp(prefix='ehbench-')
    os.makedirs(snapshot_dir, exist_ok=True)
    started = time.time()
    try:
        for index, (backend, name, target) in enumerate(plan, 1):
            existing = done.get((backend, name))
            if existing is not None and len(existing['seeds']) >= args.seeds:
                print('[%d/%d] %-8s %-34s  (cached)'
                      % (index, total_cells, backend, name), flush=True)
                continue
            cell = existing or {'backend': backend, 'target': name, 'seeds': []}
            if existing is None:
                document['cells'].append(cell)
                done[(backend, name)] = cell
            for seed_index in range(len(cell['seeds']), args.seeds):
                seed = cell_seed(args.seed_base, backend, name, seed_index)
                print('[%d/%d] %-8s %-34s seed %d/%d (%d) ...'
                      % (index, total_cells, backend, name, seed_index + 1,
                         args.seeds, seed), flush=True)
                result = run_one(backend, name, target, args, seed,
                                 snapshot_dir, args.quiet)
                cell['seeds'].append(result)
                summarise_cell(cell)
                document['generated_utc'] = datetime.now(
                    timezone.utc).isoformat(timespec='seconds')
                write_json(args.out, document)
                print('        -> %-11s best %.3f  held-out %s  %s'
                      % (result['category'], result['best'],
                         ('%.3f' % result['holdout']
                          if result['holdout'] is not None else '-'),
                         _duration(result['elapsed_s'])), flush=True)
    finally:
        if args.snapshot_dir is None:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
    print('sweep finished in %s' % _duration(time.time() - started), flush=True)
    return document


# -------------------------------- the CLI -----------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog='benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Local terminal solvability sweep driven through the '
                    "application's own evolution worker. Select architecture "
                    'and target sets, then optionally tune the same controls '
                    'as the GUI.',
        epilog='examples:\n'
               '  py tools/benchmark.py --list-targets\n'
               '  py tools/benchmark.py --architectures nervous '
               '--targets temporal '
               '--seeds 3 --gens 40 --pop 60\n'
               '  py tools/benchmark.py --architectures paper '
               '--targets "Half adder,Full adder" --dry-run\n'
               '  py tools/benchmark.py --report-only --out '
               'results/benchmark.json\n')

    what = parser.add_argument_group('what to sweep')
    what.add_argument('--architectures', '--backends', default='nervous',
                      dest='architectures',
                      help='comma-separated architectures (%s) or sets (%s); '
                           'default: nervous'
                           % (', '.join(ARCHITECTURES),
                              ', '.join(ARCHITECTURE_SETS)))
    what.add_argument('--targets', default='temporal',
                      help='comma-separated target names and/or the keywords '
                           'all, temporal, combinational, logic-temporal '
                           '(the coincident-edge twins of the truth tables, '
                           'which `temporal` also includes) '
                           '(default: temporal)')
    what.add_argument('--exclude', default='',
                      help='comma-separated target names to skip')
    what.add_argument('--seeds', type=int, default=3,
                      help='independent runs per cell (default: 3)')
    what.add_argument('--seed-base', type=int, default=20260730,
                      help='base for the per-cell seed derivation')

    run = parser.add_argument_group('run controls (GUI: top row)')
    run.add_argument('--pop', type=int, default=60, help='Population')
    run.add_argument('--gens', type=int, default=40, help='Generations')
    run.add_argument('--tries', type=int, default=1, help='Restarts')
    run.add_argument(
        '--workers', type=int, default=DEFAULT_EVALUATION_WORKERS,
        help='Parallel evaluation processes, 1-%d (default: %d)'
             % (MAX_EVALUATION_WORKERS, DEFAULT_EVALUATION_WORKERS))
    run.add_argument('--chroms', type=int, default=2, help='Genome: Chroms')
    run.add_argument('--max-telomere', type=int, default=None,
                     help='Genome: Max telomere (default: per-backend, '
                     'nervous 24, snn 20, fnv 32, lut 18)')
    run.add_argument('--pure-evolution', action='store_true',
                     help='Disable target-specific developmental seeds and '
                          'witness rescue for an unbiased evolution run')
    run.add_argument('--graded', action='store_true',
                     help='Graded logic fitness (SNN combinational only)')
    run.add_argument('--input-high', type=float, default=None,
                     help="Substrate: Input I (default: the target's own high)")

    ga = parser.add_argument_group('GA tuning (GUI: GA row)')
    ga.add_argument('--mutations', type=float, default=MEAN_MUTATIONS,
                    help='Mutations/child')
    ga.add_argument('--immigrants', type=float, default=IMMIGRANT_FRAC,
                    help='Immigrants')
    ga.add_argument('--tournament', type=int, default=TOURNAMENT_K,
                    help='Tournament')
    ga.add_argument('--elites', type=int, default=5, help='Elites')
    ga.add_argument('--rescue-limit', type=int, default=-1, metavar='N',
                    help='plateau-rescue candidates per stalled generation '
                         '(0 disables rescue, -1 = the backend default: 8 '
                         'for FNV from the measured ablation, pop//2 '
                         'elsewhere)')
    ga.add_argument('--anneal', type=float, default=MUT_DECAY,
                    help='Anneal alpha')
    ga.add_argument('--plateau-beta', type=float, default=1.0,
                    help='Plateau beta')
    ga.add_argument('--mutation-cap', type=float, default=DEFAULT_MUTATION_LIMIT,
                    help='Mutation cap')
    ga.add_argument('--lexicase', action='store_true',
                    help='epsilon-lexicase selection instead of tournament')
    ga.add_argument('--no-recombination', action='store_true',
                    help='clone and mutate parents without crossover')
    ga.add_argument(
        '--diversify-solvers', action='store_true',
        help='run the optional post-solve solver-bank search (off by default '
             'for benchmark timing)')

    sub = parser.add_argument_group('SNN substrate (GUI: Substrate row)')
    sub.add_argument('--syn-weight', type=float, default=DEFAULT_ARCH.syn_weight)
    sub.add_argument('--vth-min', type=float, default=DEFAULT_ARCH.vth_levels[0])
    sub.add_argument('--vth-max', type=float, default=DEFAULT_ARCH.vth_levels[-1])

    pulse = parser.add_argument_group('pulse physics (GUI: Pulse / Analog rows)')
    pulse.add_argument('--delay', type=float, default=DELAY)
    pulse.add_argument('--width', type=float, default=WIDTH)
    pulse.add_argument('--coincidence', type=float, default=COINC)
    pulse.add_argument('--nv-profile', default='analog_tri',
                       choices=sorted(NV_NEW_RUN_PROFILES),
                       help='NV profile (tile architecture + node physics)')
    _analog = PulseConfig()
    pulse.add_argument('--analog-vth', type=float,
                       default=_analog.analog_threshold)
    pulse.add_argument('--analog-step', type=float, default=_analog.analog_step)
    pulse.add_argument('--analog-tau', type=float,
                       default=_analog.analog_tau_leak)
    pulse.add_argument('--analog-hysteresis', type=float,
                       default=_analog.analog_hysteresis)

    io = parser.add_argument_group('I/O binding (GUI: I/O binding)')
    io.add_argument('--io-placement', default='fixed',
                    choices=('fixed', 'terminal_nodes', 'tag_rank',
                             'wiring_chromosome', 'spatial_chromosome'))
    io.add_argument('--lut-io-mode', default='source_pads',
                    choices=('source_pads', 'exterior_edges'))
    io.add_argument('--fnv-families', default=','.join(FNV_FAMILIES),
                    help='comma-separated FNV component families: %s'
                         % ', '.join(FNV_FAMILIES))
    io.add_argument('--fnv-readout', choices=('fitted', 'genetic'),
                    default=FNVConfig().readout_mode,
                    help='FNV output scoring: fitted probes (control) or the '
                         'evolved output-role sites')
    if LUT_FUNCTION_FAMILIES:
        io.add_argument('--lut-function-families', default='UNRESTRICTED',
                        help='comma-separated LUT truth-table banks: %s '
                             '(default: UNRESTRICTED, the all-table substrate)'
                             % ', '.join(LUT_FUNCTION_FAMILIES))

    esc = parser.add_argument_group('escape mechanisms (GUI: Escape rows)')
    esc.add_argument('--lifespan', action='store_true')
    esc.add_argument('--lifespan-stages', type=int,
                     default=EscapeConfig().lifespan_checkpoints)
    esc.add_argument('--crowding', action='store_true')
    esc.add_argument('--crowding-window', type=int,
                     default=EscapeConfig().crowding_window)
    esc.add_argument('--crowding-reserve', type=float,
                     default=EscapeConfig().crowding_fraction)
    esc.add_argument('--neutral-drift', action='store_true')
    esc.add_argument('--self-adaptive-mutation', action='store_true')
    esc.add_argument('--rebirth', action='store_true')
    esc.add_argument('--rebirth-stall', type=int,
                     default=EscapeConfig().rebirth_patience)
    esc.add_argument('--rebirth-fraction', type=float,
                     default=EscapeConfig().rebirth_fraction)
    esc.add_argument('--lineage-walk', action='store_true',
                     help='reserve fitness-blind mutation-only lineages that '
                          'can traverse multi-generation fitness valleys')
    esc.add_argument('--lineage-walk-fraction', type=float,
                     default=EscapeConfig().lineage_walk_fraction)
    esc.add_argument('--robustness', action='store_true')
    esc.add_argument('--robustness-jitter', type=float,
                     default=EscapeConfig().robustness_jitter)
    esc.add_argument('--islands', action='store_true')
    esc.add_argument('--island-demes', type=int,
                     default=EscapeConfig().island_count)
    esc.add_argument('--island-migrate', type=int,
                     default=EscapeConfig().island_migration_interval)
    esc.add_argument('--lexicase-sample', type=float,
                     default=EscapeConfig().lexicase_downsample,
                     help='fraction of cases epsilon-lexicase streams per '
                          'generation (1 = all)')

    out = parser.add_argument_group('output')
    out.add_argument('--out', default=DEFAULT_OUT,
                     help='result JSON (rewritten after every seed)')
    out.add_argument('--markdown', default=None,
                     help='markdown report path (default: --out with .md)')
    out.add_argument('--snapshot-dir', default=None,
                     help="directory for the controller's population "
                          'snapshots (default: a temporary dir, discarded)')
    out.add_argument('--resume', action='store_true',
                     help='continue an interrupted sweep from --out')
    out.add_argument('--report-only', action='store_true',
                     help='re-render the markdown from --out; run nothing')
    out.add_argument('--dry-run', action='store_true',
                     help='print the planned cells and exit')
    out.add_argument('--quiet', action='store_true',
                     help='suppress per-generation progress')
    out.add_argument('--time-cap', type=float, default=0.0, metavar='SECONDS',
                     help='abandon the remaining generations once one run has '
                          'spent this long (0 = no cap). The champion is kept '
                          'and still certified; the reason is recorded so a '
                          'capped run is not mistaken for an unreachable target')
    out.add_argument('--stop-on-solve', action='store_true',
                     help='end a run as soon as training fitness reaches 1.0 '
                          'instead of exhausting the generation budget')
    out.add_argument('--progress-every', type=int, default=10,
                     help='print a progress line every N generations (0 = off)')
    out.add_argument('--list-targets', action='store_true',
                      help='list each backend\'s supported targets and exit')
    out.add_argument('--list-architectures', action='store_true',
                     help='list individual architectures and named sets, then '
                          'exit')
    return parser


def parse_list(value):
    return [item.strip() for item in str(value).split(',') if item.strip()]


def resolve_architectures(value):
    """Expand individual architecture names and named architecture sets."""
    selected = []
    unknown = []
    for token in parse_list(value):
        key = token.lower()
        if key in ARCHITECTURE_SETS:
            selected.extend(ARCHITECTURE_SETS[key])
        elif key in ARCHITECTURES:
            selected.append(key)
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(
            'unknown architecture or set: %s' % ', '.join(unknown))
    if not selected:
        raise ValueError('select at least one architecture')
    return list(dict.fromkeys(selected))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        architectures = resolve_architectures(args.architectures)
    except ValueError as error:
        parser.error(str(error))
    args.targets = parse_list(args.targets)
    args.exclude = parse_list(args.exclude)
    args.fnv_families = parse_list(args.fnv_families)
    args.lut_function_families = parse_list(
        getattr(args, 'lut_function_families', 'UNRESTRICTED'))

    if args.list_architectures:
        print('Architectures')
        for architecture in ARCHITECTURES:
            print('  %-9s %s' % (
                architecture, ARCHITECTURE_DESCRIPTIONS[architecture]))
        print('\nNamed sets')
        for name, members in ARCHITECTURE_SETS.items():
            print('  %-9s %s' % (name, ', '.join(members)))
        return 0

    if args.list_targets:
        _tile, node_model, _delay = NV_NEW_RUN_PROFILES[args.nv_profile]
        for backend in architectures:
            names = targets_for_backend(backend, node_model)
            print('%s (%d targets)' % (backend, len(names)))
            for name in names:
                kind = 'temporal' if name in TEMPORAL_TARGETS else 'combinational'
                print('    %-38s %s' % (name, kind))
            print()
        return 0

    markdown_path = args.markdown or os.path.splitext(args.out)[0] + '.md'

    if args.report_only:
        if not os.path.exists(args.out):
            parser.error('no result file at %s' % args.out)
        document = load_json(args.out)
    else:
        if not 1 <= args.chroms <= MAX_CHROMOSOME_COUNT:
            parser.error('--chroms must be between 1 and %d'
                         % MAX_CHROMOSOME_COUNT)
        if args.seeds < 1:
            parser.error('--seeds must be at least 1')
        if (args.io_placement in ('wiring_chromosome', 'spatial_chromosome')
                and args.chroms < 3):
            parser.error('chromosome-based I/O needs --chroms 3 or more; '
                         'chromosome 3 is the evolvable port map')
        # Fail on an invalid configuration here, before any evolution burns
        # hours only to raise inside a worker.
        for backend in architectures:
            build_run_config(args, backend, args.chroms)
        document = run_sweep(args, architectures)
        if args.dry_run:
            return 0

    for cell in document['cells']:
        summarise_cell(cell)
    write_json(args.out, document)
    report = render_markdown(document)
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write(report)
    print()
    print(report)
    print('wrote %s and %s' % (args.out, markdown_path))

    return 0


if __name__ == '__main__':
    # The controller opens a ProcessPoolExecutor; on Windows spawn this module
    # is re-imported in each worker, so the entry point must stay guarded.
    raise SystemExit(main())
