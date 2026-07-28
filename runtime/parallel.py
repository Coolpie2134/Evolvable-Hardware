"""
evo_runtime/parallel.py — one saturated, cancellation-aware population map.

`ProcessPoolExecutor.map` and hand-rolled chunk loops both stall on a straggler:
the caller waits for a whole batch before the next one is submitted, so a single
slow genome idles every other worker until the batch's barrier clears. Nervous
evaluation has a ~6x per-genome tail (a big grown net next to many small ones),
so that barrier wasted a large fraction of wall-clock. `map_ordered` submits the
whole population at once and harvests results as they finish — the pool stays
full — while still checking the stop signal between completions, so a run stays
responsive to cancellation without paying the barrier.
"""
from __future__ import annotations

from concurrent.futures import as_completed


class EvolutionCancelled(Exception):
    """Raised to unwind a run when the stop signal fires mid-evaluation."""


def map_ordered(executor, fn, items, should_stop=None, on_progress=None):
    """Evaluate `fn` over `items` on `executor`, returning results in INPUT order.

    Unlike a chunked `executor.map`, every item is submitted immediately, so the
    pool is never idled waiting on a batch barrier. `should_stop` (a zero-arg
    predicate) is polled as each result arrives; when it fires, not-yet-started
    work is cancelled and `EvolutionCancelled` is raised. `on_progress(done,
    total)` is called after each completion for progress reporting.
    """
    items = list(items)
    futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
    results = [None] * len(items)
    done = 0
    try:
        for fut in as_completed(futures):
            if should_stop is not None and should_stop():
                raise EvolutionCancelled
            results[futures[fut]] = fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(items))
    except EvolutionCancelled:
        for f in futures:
            f.cancel()
        raise
    return results
