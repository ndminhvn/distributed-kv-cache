from collections import OrderedDict
from typing import Any, Dict, Tuple


class KVCache:
    """
    Simple in-memory KV cache for LLM inference.
    Stores:
        key = (seq_id, layer, step)
        value = {"k": [...], "v": [...]}
    """

    def __init__(self, max_entries: int = 50000):
        self.cache: OrderedDict[Tuple[str, int, int], Dict[str, Any]] = OrderedDict()
        self.max_entries = max_entries

    def _make_key(self, seq_id: str, layer: int, step: int):
        return (seq_id, layer, step)

    def put(self, seq_id: str, layer: int, step: int, k_tensor, v_tensor):
        key = self._make_key(seq_id, layer, step)

        # Update or insert the entry
        self.cache[key] = {"k": k_tensor, "v": v_tensor}

        # Move to end (most recently used)
        self.cache.move_to_end(key)

        # Perform LRU eviction
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)  # Evict least recently used

    def get(self, seq_id: str, layer: int, step: int):
        key = self._make_key(seq_id, layer, step)
        if key not in self.cache:
            return None

        # Mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def evict_sequence(self, seq_id: str):
        """Evict an entire sequence (all layers, all steps)."""
        keys_to_delete = [k for k in self.cache.keys() if k[0] == seq_id]
        for k in keys_to_delete:
            del self.cache[k]

    def stats(self):
        return {"entries": len(self.cache), "max_entries": self.max_entries}
