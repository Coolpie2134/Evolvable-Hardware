from __future__ import annotations

from collections import OrderedDict


class LRUCache(OrderedDict):
    """Bounded mapping that evicts only the least-recently-used evaluation."""

    def __init__(self, capacity=200_000):
        super().__init__()
        self.capacity = max(1, int(capacity))
        # These are run telemetry, not part of the mapping semantics.  A
        # ``store`` is one phenotype evaluation admitted to the cache; unlike
        # population x generation it excludes duplicate genomes and cache
        # hits, while still counting a phenotype again if eviction forced a
        # real re-evaluation.
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        try:
            value = self[key]
        except KeyError:
            self.misses += 1
            return default
        self.hits += 1
        return value

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.stores += 1
        while len(self) > self.capacity:
            self.popitem(last=False)
            self.evictions += 1

    def telemetry(self):
        """Return cumulative counters safe to serialize in run history."""
        return {
            'phenotype_evaluations': int(self.stores),
            'cache_hits': int(self.hits),
            'cache_misses': int(self.misses),
            'cache_evictions': int(self.evictions),
        }
