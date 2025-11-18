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
    batch_size: int = 1, num_heads: int = 8, seq_len: int = 1, head_dim: int = 64
) -> torch.Tensor:
    """
    Generate a fake KV tensor similar to what would be used in transformer attention.

    Shape: (batch_size, num_heads, seq_len, head_dim)

    This is the STANDARD format used in production systems:
    - HuggingFace Transformers (GPT-2, LLaMA, etc.)
    - vLLM (optimized inference engine)
    - TensorRT-LLM
    - Text Generation Inference (TGI)

    Benefits of this format:
    1. Easy token appending: concat along seq_len dimension
    2. Compatible with standard attention: scores = Q @ K.transpose(-2, -1)
    3. Matches most pretrained model checkpoints
    4. Optimized GPU kernel support

    For distributed KV cache, each cache entry stores one "step" worth of KV:
    - At step t, we cache K[:, :, t:t+1, :] and V[:, :, t:t+1, :]
    - During inference, concat all cached steps: K_full = concat([K_0, K_1, ..., K_t])
    """
    return torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
