import torch
import logging
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Auto-detect the best available device for tensor operations.

    Returns:
        - "cuda" if NVIDIA GPU is available
        - "cpu" otherwise (including macOS)
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class KVCache:
    """
    PyTorch-based in-memory KV cache for LLM inference.

    Stores transformer KV cache with support for:
    - Multi-token sequences (autoregressive generation)
    - Efficient appending of new tokens
    - LRU eviction
    - Memory tracking

    Storage format:
        key = (seq_id, layer)
        value = {
            "k": torch.Tensor [seq_len, num_heads, head_dim],
            "v": torch.Tensor [seq_len, num_heads, head_dim]
        }
    where seq_len grows with each generated token.
    """

    def __init__(self, max_entries: int = 50000, device: Optional[str] = None):
        self.cache: OrderedDict[Tuple[str, int], Dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.max_entries = max_entries
        self.device = device if device is not None else get_device()
        logger.info(f"KVCache initialized with device: {self.device}")

    def _make_key(self, seq_id: str, layer: int) -> Tuple[str, int]:
        """Create cache key from sequence metadata."""
        return (seq_id, layer)

    def put(
        self,
        seq_id: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """
        Store or append KV tensors for a specific sequence and layer.

        Args:
            seq_id: Unique sequence identifier
            layer: Transformer layer index
            k_tensor: Key tensor [seq_len, num_heads, head_dim] - full sequence or new token
            v_tensor: Value tensor [seq_len, num_heads, head_dim] - full sequence or new token
        """
        key = self._make_key(seq_id, layer)

        # Move tensors to cache device and ensure they're stored
        k_stored = k_tensor.to(self.device).clone()
        v_stored = v_tensor.to(self.device).clone()

        # Update or insert the entry
        self.cache[key] = {"k": k_stored, "v": v_stored}

        # Move to end (most recently used)
        self.cache.move_to_end(key)

        # Perform LRU eviction if needed
        if len(self.cache) > self.max_entries:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            # Free GPU memory if applicable
            del evicted_value["k"]
            del evicted_value["v"]

    def append(
        self,
        seq_id: str,
        layer: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """
        Append new token's KV to existing cache for a sequence and layer.

        Args:
            seq_id: Unique sequence identifier
            layer: Transformer layer index
            k_tensor: Key tensor for new token [1, num_heads, head_dim]
            v_tensor: Value tensor for new token [1, num_heads, head_dim]
        """
        key = self._make_key(seq_id, layer)

        # Move tensors to device
        k_new = k_tensor.to(self.device)
        v_new = v_tensor.to(self.device)

        if key in self.cache:
            # Append to existing cache
            cached = self.cache[key]
            k_concat = torch.cat([cached["k"], k_new], dim=0)
            v_concat = torch.cat([cached["v"], v_new], dim=0)
            self.cache[key] = {"k": k_concat, "v": v_concat}
        else:
            # First token for this sequence/layer
            self.cache[key] = {"k": k_new.clone(), "v": v_new.clone()}

        # Move to end (most recently used)
        self.cache.move_to_end(key)

        # Perform LRU eviction if needed
        if len(self.cache) > self.max_entries:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            del evicted_value["k"]
            del evicted_value["v"]

    def get(self, seq_id: str, layer: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve KV tensors for a specific sequence and layer.

        Returns:
            Dict with "k" and "v" tensors [seq_len, num_heads, head_dim], or None if not found
        """
        key = self._make_key(seq_id, layer)
        if key not in self.cache:
            return None

        # Mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def get_seq_len(self, seq_id: str, layer: int) -> int:
        """
        Get the current sequence length for a given sequence and layer.

        Args:
            seq_id: Sequence identifier
            layer: Layer index

        Returns:
            Current sequence length, or 0 if not found
        """
        entry = self.get(seq_id, layer)
        if entry is None:
            return 0
        return entry["k"].shape[0]

    def evict_sequence(self, seq_id: str):
        """
        Evict all entries for a given sequence (all layers, all steps).

        Args:
            seq_id: Sequence identifier to evict
        """
        keys_to_delete = [k for k in self.cache.keys() if k[0] == seq_id]
        for k in keys_to_delete:
            entry = self.cache.pop(k)
            # Free memory
            del entry["k"]
            del entry["v"]

    def evict_sequence_layer(self, seq_id: str, layer: int):
        """
        Evict cache for a specific sequence and layer.

        Args:
            seq_id: Sequence identifier
            layer: Layer index
        """
        key = self._make_key(seq_id, layer)
        if key in self.cache:
            entry = self.cache.pop(key)
            del entry["k"]
            del entry["v"]

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Calculate approximate memory usage of cached tensors.

        Returns:
            Dict with memory statistics in bytes and MB
        """
        total_bytes = 0
        for entry in self.cache.values():
            total_bytes += entry["k"].element_size() * entry["k"].numel()
            total_bytes += entry["v"].element_size() * entry["v"].numel()

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "num_entries": len(self.cache),
        }

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics including memory usage."""
        memory_stats = self.get_memory_usage()

        # Count sequences
        unique_sequences = len(set(k[0] for k in self.cache.keys()))

        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "unique_sequences": unique_sequences,
            "device": self.device,
            **memory_stats,
        }
