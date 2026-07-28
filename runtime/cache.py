from __future__ import annotations

from collections import OrderedDict


class LRUCache(OrderedDict):
    """Bounded mapping that evicts only the least-recently-used evaluation."""

    def __init__(self, capacity=200_000):
        super().__init__()
        self.capacity = max(1, int(capacity))

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self.capacity:
            self.popitem(last=False)

