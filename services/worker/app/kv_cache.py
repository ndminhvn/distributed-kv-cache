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
    - Multi-step sequences (autoregressive generation)
    - Concatenation of cached steps
    - LRU eviction
    - Memory tracking

    Storage format:
        key = (seq_id, layer, step)
        value = {
            "k": torch.Tensor [batch=1, num_heads, seq_len=1, head_dim],
            "v": torch.Tensor [batch=1, num_heads, seq_len=1, head_dim]
        }
    """

    def __init__(self, max_entries: int = 50000, device: Optional[str] = None):
        self.cache: OrderedDict[Tuple[str, int, int], Dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.max_entries = max_entries
        self.device = device if device is not None else get_device()
        logger.info(f"KVCache initialized with device: {self.device}")

    def _make_key(self, seq_id: str, layer: int, step: int) -> Tuple[str, int, int]:
        """Create cache key from sequence metadata."""
        return (seq_id, layer, step)

    def put(
        self,
        seq_id: str,
        layer: int,
        step: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """
        Store KV tensors for a specific sequence, layer, and step.

        Args:
            seq_id: Unique sequence identifier
            layer: Transformer layer index
            step: Generation step index
            k_tensor: Key tensor [batch, num_heads, seq_len, head_dim]
            v_tensor: Value tensor [batch, num_heads, seq_len, head_dim]
        """
        key = self._make_key(seq_id, layer, step)

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

    def get(
        self, seq_id: str, layer: int, step: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve KV tensors for a specific sequence, layer, and step.

        Returns:
            Dict with "k" and "v" tensors, or None if not found
        """
        key = self._make_key(seq_id, layer, step)
        if key not in self.cache:
            return None

        # Mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def get_sequence_steps(self, seq_id: str, layer: int) -> List[int]:
        """
        Get all cached step indices for a given sequence and layer.

        Args:
            seq_id: Sequence identifier
            layer: Layer index

        Returns:
            Sorted list of step indices
        """
        steps = [k[2] for k in self.cache.keys() if k[0] == seq_id and k[1] == layer]
        return sorted(steps)

    def concat_sequence_steps(
        self, seq_id: str, layer: int, max_step: Optional[int] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Concatenate all cached KV tensors for a sequence at a given layer.

        This reconstructs the full KV cache for attention computation.

        Args:
            seq_id: Sequence identifier
            layer: Layer index
            max_step: If provided, only concat up to this step (inclusive)

        Returns:
            Dict with concatenated "k" and "v" tensors, or None if no steps found
            K, V shape: [batch, num_heads, total_seq_len, head_dim]
        """
        steps = self.get_sequence_steps(seq_id, layer)
        if not steps:
            return None

        if max_step is not None:
            steps = [s for s in steps if s <= max_step]

        if not steps:
            return None

        # Gather all K and V tensors
        k_list = []
        v_list = []
        for step in steps:
            entry = self.get(seq_id, layer, step)
            if entry is not None:
                k_list.append(entry["k"])
                v_list.append(entry["v"])

        if not k_list:
            return None

        # Concatenate along sequence dimension (dim=2)
        k_concat = torch.cat(k_list, dim=2)
        v_concat = torch.cat(v_list, dim=2)

        return {"k": k_concat, "v": v_concat}

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
        Evict all steps for a specific sequence and layer.

        Args:
            seq_id: Sequence identifier
            layer: Layer index
        """
        keys_to_delete = [
            k for k in self.cache.keys() if k[0] == seq_id and k[1] == layer
        ]
        for k in keys_to_delete:
            entry = self.cache.pop(k)
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
