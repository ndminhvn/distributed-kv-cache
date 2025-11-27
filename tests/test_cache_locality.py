"""
Test suite for KV cache locality and behavior.

Tests verify:
- KV cache entries for a sequence are on the correct worker
- Cache reuse works across multiple generation steps
- Cache grows correctly as sequence length increases
"""

import httpx
import pytest
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.tensor_utils import serialize_tensor, deserialize_tensor

# Test configuration
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8081")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")


@pytest.mark.asyncio
async def test_cache_locality():
    """Test that KV cache entries are stored on the correct worker."""
    async with httpx.AsyncClient() as client:
        seq_id = "locality-test-seq"
        layer = 0

        # Get the worker that should handle this sequence
        route_resp = await client.get(
            f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0
        )
        assert route_resp.status_code == 200
        expected_worker_id = route_resp.json()["worker_id"]
        expected_worker_addr = route_resp.json()["address"]

        # Create test KV tensors
        k_tensor = torch.randn(5, 12, 64)  # [seq_len=5, heads=12, dim=64]
        v_tensor = torch.randn(5, 12, 64)

        # Store KV cache via gateway
        put_resp = await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "k": serialize_tensor(k_tensor),
                "v": serialize_tensor(v_tensor),
            },
            timeout=5.0,
        )
        assert put_resp.status_code == 200
        put_data = put_resp.json()

        # Verify it was stored on the expected worker
        assert (
            put_data["worker_id"] == expected_worker_id
        ), f"KV stored on wrong worker: expected {expected_worker_id}, got {put_data['worker_id']}"

        # Retrieve KV cache and verify it's from the same worker
        get_resp = await client.post(
            f"{GATEWAY_URL}/kv/get",
            json={"seq_id": seq_id, "layer": layer},
            timeout=5.0,
        )
        assert get_resp.status_code == 200
        get_data = get_resp.json()

        assert (
            get_data["worker_id"] == expected_worker_id
        ), f"KV retrieved from wrong worker: expected {expected_worker_id}, got {get_data['worker_id']}"

        # Verify tensor shapes match
        k_retrieved = deserialize_tensor(get_data["k"])
        v_retrieved = deserialize_tensor(get_data["v"])

        assert (
            k_retrieved.shape == k_tensor.shape
        ), f"K tensor shape mismatch: {k_retrieved.shape} vs {k_tensor.shape}"
        assert (
            v_retrieved.shape == v_tensor.shape
        ), f"V tensor shape mismatch: {v_retrieved.shape} vs {v_tensor.shape}"

        print(f"✓ Cache locality verified:")
        print(f"  Sequence: {seq_id}")
        print(f"  Worker: {expected_worker_id}")
        print(f"  KV shape: {k_tensor.shape}")


@pytest.mark.asyncio
async def test_cache_append_behavior():
    """Test that cache append correctly grows the sequence length."""
    async with httpx.AsyncClient() as client:
        seq_id = "append-test-seq"
        layer = 0

        # Initial KV cache (5 tokens)
        k_initial = torch.randn(5, 12, 64)
        v_initial = torch.randn(5, 12, 64)

        put_resp = await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "k": serialize_tensor(k_initial),
                "v": serialize_tensor(v_initial),
            },
            timeout=5.0,
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["seq_len"] == 5

        # Append 3 new tokens
        for i in range(3):
            k_new = torch.randn(1, 12, 64)  # Single token
            v_new = torch.randn(1, 12, 64)

            append_resp = await client.post(
                f"{GATEWAY_URL}/kv/append",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_new),
                    "v": serialize_tensor(v_new),
                },
                timeout=5.0,
            )
            assert append_resp.status_code == 200
            expected_len = 5 + i + 1
            assert (
                append_resp.json()["seq_len"] == expected_len
            ), f"Sequence length after append {i+1}: expected {expected_len}, got {append_resp.json()['seq_len']}"

        # Retrieve final cache and verify total length
        get_resp = await client.post(
            f"{GATEWAY_URL}/kv/get",
            json={"seq_id": seq_id, "layer": layer},
            timeout=5.0,
        )
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert (
            get_data["seq_len"] == 8
        ), f"Final seq_len should be 8, got {get_data['seq_len']}"

        # Verify tensor dimensions
        k_final = deserialize_tensor(get_data["k"])
        assert (
            k_final.shape[0] == 8
        ), f"K tensor first dim should be 8, got {k_final.shape[0]}"

        print(f"✓ Cache append behavior verified:")
        print(f"  Initial length: 5 tokens")
        print(f"  After 3 appends: 8 tokens")
        print(f"  Final KV shape: {k_final.shape}")


@pytest.mark.asyncio
async def test_multi_layer_cache():
    """Test that multiple layers can be cached for the same sequence."""
    async with httpx.AsyncClient() as client:
        seq_id = "multi-layer-seq"
        num_layers = 4

        # Store KV cache for multiple layers
        for layer in range(num_layers):
            k_tensor = torch.randn(10, 12, 64)
            v_tensor = torch.randn(10, 12, 64)

            put_resp = await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_tensor),
                    "v": serialize_tensor(v_tensor),
                },
                timeout=5.0,
            )
            assert put_resp.status_code == 200

        # Retrieve and verify each layer
        for layer in range(num_layers):
            get_resp = await client.post(
                f"{GATEWAY_URL}/kv/get",
                json={"seq_id": seq_id, "layer": layer},
                timeout=5.0,
            )
            assert get_resp.status_code == 200
            get_data = get_resp.json()
            assert get_data["layer"] == layer
            assert get_data["seq_len"] == 10

        print(f"✓ Multi-layer cache verified:")
        print(f"  Sequence: {seq_id}")
        print(f"  Layers: {num_layers}")
        print(f"  All layers retrieved successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
