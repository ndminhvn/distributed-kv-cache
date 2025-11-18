import hashlib
from bisect import bisect_right
from typing import Dict, List, Tuple


def _hash(value: str) -> int:
    """Return a consistent 32-bit hash."""
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


class ConsistentHashRing:
    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        self.ring: List[Tuple[int, str]] = []  # (hash, worker_id)
        self.sorted_keys: List[int] = []
        self.nodes: Dict[str, List[int]] = {}  # worker -> its vnode hashes

    def add_node(self, worker_id: str):
        """Add worker and its virtual nodes"""
        vnode_hashes = []

        for i in range(self.virtual_nodes):
            vnode_key = f"{worker_id}-vn-{i}"
            h = _hash(vnode_key)
            self.ring.append((h, worker_id))
            vnode_hashes.append(h)

        self.nodes[worker_id] = vnode_hashes
        self._rebuild()

    def remove_node(self, worker_id: str):
        """Remove worker and its virtual nodes"""
        if worker_id not in self.nodes:
            return

        vnode_hashes = set(self.nodes[worker_id])
        self.ring = [(h, w) for (h, w) in self.ring if h not in vnode_hashes]

        del self.nodes[worker_id]
        self._rebuild()

    def _rebuild(self):
        """Sort ring after changes"""
        self.ring.sort(key=lambda x: x[0])
        self.sorted_keys = [h for (h, _) in self.ring]

    def get_node(self, key: str) -> str:
        """Return worker_id responsible for this key"""
        if not self.ring:
            return None

        h = _hash(key)
        idx = bisect_right(self.sorted_keys, h)

        if idx == len(self.sorted_keys):
            idx = 0  # wrap around

        return self.ring[idx][1]
