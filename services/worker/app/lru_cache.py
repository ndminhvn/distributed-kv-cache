from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        # mark as recently used
        self.cache.move_to_end(key, last=True)
        return self.cache[key]

    def put(self, key: str, value: Any):
        # overwrite existing
        if key in self.cache:
            self.cache.move_to_end(key, last=True)
        self.cache[key] = value

        # evict if needed
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # LRU is at the front
