"""
Utilities for serializing/deserializing PyTorch tensors for HTTP transport.
"""

import base64
import io
import torch
from typing import Dict, Any


def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Serialize a PyTorch tensor to a JSON-compatible dict.

    Returns:
        {
            "data": base64-encoded bytes,
            "shape": list of dimensions,
            "dtype": string representation of dtype
        }
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)

    return {
        "data": base64.b64encode(buffer.read()).decode("utf-8"),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def deserialize_tensor(tensor_dict: Dict[str, Any]) -> torch.Tensor:
    """
    Deserialize a tensor from the format created by serialize_tensor.

    Args:
        tensor_dict: Dict with "data", "shape", and "dtype" keys

    Returns:
        PyTorch tensor
    """
    data_bytes = base64.b64decode(tensor_dict["data"])
    buffer = io.BytesIO(data_bytes)
    buffer.seek(0)

    tensor = torch.load(buffer, weights_only=True)
    return tensor


def is_tensor_dict(obj: Any) -> bool:
    """Check if an object is a serialized tensor dict."""
    return isinstance(obj, dict) and "data" in obj and "shape" in obj and "dtype" in obj


def generate_fake_kv_tensor(
    seq_len: int = 1, num_heads: int = 8, head_dim: int = 64
) -> torch.Tensor:
    """
    Generate a fake KV tensor for distributed KV cache.

    NEW SIMPLIFIED FORMAT: (seq_len, num_heads, head_dim)

    This format stores all tokens together per layer:
    - seq_len grows with each generated token
    - Direct access - no concatenation needed for attention
    - Standard PyTorch convention: [seq_len, heads, dim]

    Usage:
    - For first token: generate_fake_kv_tensor(1, 8, 64) → [1, 8, 64]
    - For full sequence: generate_fake_kv_tensor(10, 8, 64) → [10, 8, 64]
    - Append new token: torch.cat([cached_kv, new_token_kv], dim=0)

    Benefits:
    1. Simpler structure - no step tracking
    2. Faster retrieval - already in attention format
    3. Less memory overhead - single tensor per layer
    4. Matches PyTorch attention conventions
    """
    return torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
