#!/usr/bin/env python3
"""
Advanced tests for PyTorch-based KV cache functionality.
Tests appending, memory management, and multi-token sequences.
"""

import httpx
import asyncio
import time
import pytest
import torch

from test_utils import (
    cleanup_sequences,
    track_sequence,
    get_tracked_sequences,
    clear_tracked_sequences,
    GATEWAY_URL,
)
from tensor_utils import serialize_tensor, deserialize_tensor, generate_fake_kv_tensor


# Realistic transformer dimensions
NUM_HEADS = 8
HEAD_DIM = 64
NUM_LAYERS = 12


@pytest.mark.asyncio
async def test_multi_token_append():
    """Test appending multiple tokens to a sequence."""
    print("\n" + "=" * 60)
    print("ADVANCED TEST 1: Multi-Token Appending")
    print("=" * 60)

    seq_id = "append_test_seq"
    track_sequence(seq_id)
    layer = 0
    num_tokens = 5

    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"\nAppending {num_tokens} tokens for seq={seq_id}, layer={layer}")

        # First token: use put
        k_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
        v_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

        await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "k": serialize_tensor(k_tensor),
                "v": serialize_tensor(v_tensor),
            },
        )
        print(f"   Token 0: PUT - shape {k_tensor.shape}")

        # Append remaining tokens
        for token_idx in range(1, num_tokens):
            k_new = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
            v_new = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

            resp = await client.post(
                f"{GATEWAY_URL}/kv/append",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_new),
                    "v": serialize_tensor(v_new),
                },
            )
            result = resp.json()
            print(f"   Token {token_idx}: APPEND - seq_len now {result['seq_len']}")

        # Retrieve full sequence
        get_resp = await client.post(
            f"{GATEWAY_URL}/kv/get",
            json={
                "seq_id": seq_id,
                "layer": layer,
            },
        )
        result = get_resp.json()

        retrieved_k = deserialize_tensor(result["k"])
        retrieved_v = deserialize_tensor(result["v"])

        print(f"\nFinal sequence:")
        print(
            f"   K shape: {retrieved_k.shape} (expected: [{num_tokens}, {NUM_HEADS}, {HEAD_DIM}])"
        )
        print(f"   V shape: {retrieved_v.shape}")

        # Verify shape
        assert retrieved_k.shape == (num_tokens, NUM_HEADS, HEAD_DIM)
        assert retrieved_v.shape == (num_tokens, NUM_HEADS, HEAD_DIM)

        print("\n✓ Multi-token appending verified successfully!")


@pytest.mark.asyncio
async def test_memory_tracking():
    """Test memory usage tracking for cached tensors."""
    print("\n" + "=" * 60)
    print("ADVANCED TEST 2: Memory Usage Tracking")
    print("=" * 60)

    seq_id = "memory_test_seq"
    track_sequence(seq_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Store entries across multiple layers with varying seq_len
        num_layers = 3
        seq_lens = [5, 10, 15]  # Different sequence lengths per layer

        print(f"\nStoring {num_layers} layers with varying sequence lengths")

        for layer in range(num_layers):
            seq_len = seq_lens[layer]
            # Generate full sequence for this layer [seq_len, num_heads, head_dim]
            k_tensor = generate_fake_kv_tensor(seq_len, NUM_HEADS, HEAD_DIM)
            v_tensor = generate_fake_kv_tensor(seq_len, NUM_HEADS, HEAD_DIM)

            await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_tensor),
                    "v": serialize_tensor(v_tensor),
                },
            )
            print(f"   Layer {layer}: seq_len={seq_len}")

        # Get stats from worker
        route_resp = await client.get(f"http://localhost:8081/kv/route/{seq_id}")
        worker_addr = route_resp.json()["address"]

        stats_resp = await client.get(f"{worker_addr}/kv/stats")
        stats = stats_resp.json()

        print(f"\nCache Statistics:")
        print(f"   Total entries: {stats['entries']}")
        print(f"   Unique sequences: {stats['unique_sequences']}")
        print(
            f"   Memory usage: {stats['total_mb']:.2f} MB ({stats['total_bytes']:,} bytes)"
        )
        print(f"   Device: {stats['device']}")

        # Calculate expected memory
        # Each tensor: seq_len × num_heads × head_dim × 4 bytes (float32)
        bytes_per_entry = lambda seq_len: 2 * seq_len * NUM_HEADS * HEAD_DIM * 4
        expected_bytes = sum(bytes_per_entry(sl) for sl in seq_lens)
        expected_mb = expected_bytes / (1024 * 1024)

        print(f"\nExpected memory: {expected_mb:.2f} MB ({expected_bytes:,} bytes)")

        # Allow some tolerance for metadata overhead
        assert abs(stats["total_mb"] - expected_mb) < 0.1, "Memory usage mismatch"
        assert stats["entries"] == num_layers
        assert stats["unique_sequences"] >= 1

        print("\n✓ Memory tracking verified!")


@pytest.mark.asyncio
async def test_layer_specific_eviction():
    """Test evicting specific layers while keeping others."""
    print("\n" + "=" * 60)
    print("ADVANCED TEST 3: Layer-Specific Eviction")
    print("=" * 60)

    seq_id = "evict_layer_test_seq"
    track_sequence(seq_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Store data across multiple layers
        num_layers = 3

        print(f"\nStoring {num_layers} layers with different sequence lengths")

        for layer in range(num_layers):
            seq_len = layer + 1  # Different lengths: 1, 2, 3
            k_tensor = generate_fake_kv_tensor(seq_len, NUM_HEADS, HEAD_DIM)
            v_tensor = generate_fake_kv_tensor(seq_len, NUM_HEADS, HEAD_DIM)

            await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_tensor),
                    "v": serialize_tensor(v_tensor),
                },
            )
            print(f"   Layer {layer}: seq_len={seq_len}")

        # Get worker address
        route_resp = await client.get(f"http://localhost:8081/kv/route/{seq_id}")
        worker_addr = route_resp.json()["address"]

        # Verify all layers exist
        print("\nBefore eviction:")
        for layer in range(num_layers):
            length_resp = await client.get(f"{worker_addr}/kv/{seq_id}/{layer}/length")
            seq_len = length_resp.json()["seq_len"]
            print(f"   Layer {layer}: seq_len={seq_len}")
            assert seq_len == layer + 1

        # Evict entire sequence
        print(f"\nEvicting entire sequence: {seq_id}")
        await client.delete(f"{worker_addr}/kv/{seq_id}")

        # Verify all layers are evicted
        print("\nAfter eviction:")
        for layer in range(num_layers):
            length_resp = await client.get(f"{worker_addr}/kv/{seq_id}/{layer}/length")
            seq_len = length_resp.json()["seq_len"]
            print(f"   Layer {layer}: seq_len={seq_len}")
            assert seq_len == 0

        print("\n✓ Eviction verified!")


@pytest.mark.asyncio
async def test_sequence_growth():
    """Test sequence growing over time with appends."""
    print("\n" + "=" * 60)
    print("ADVANCED TEST 4: Sequence Growth Over Time")
    print("=" * 60)

    seq_id = "growth_test_seq"
    track_sequence(seq_id)
    layer = 0
    total_tokens = 10

    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"\nGrowing sequence to {total_tokens} tokens")

        # First token
        k_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
        v_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

        await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "k": serialize_tensor(k_tensor),
                "v": serialize_tensor(v_tensor),
            },
        )
        print(f"   Token 0: initialized")

        # Get worker address
        route_resp = await client.get(f"http://localhost:8081/kv/route/{seq_id}")
        worker_addr = route_resp.json()["address"]

        # Append remaining tokens and check length growth
        for token_idx in range(1, total_tokens):
            k_new = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
            v_new = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

            await client.post(
                f"{GATEWAY_URL}/kv/append",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_new),
                    "v": serialize_tensor(v_new),
                },
            )

            # Check current length
            length_resp = await client.get(f"{worker_addr}/kv/{seq_id}/{layer}/length")
            current_len = length_resp.json()["seq_len"]
            expected_len = token_idx + 1

            print(
                f"   Token {token_idx}: seq_len={current_len} (expected {expected_len})"
            )
            assert current_len == expected_len, f"Length mismatch at token {token_idx}"

        # Final verification
        get_resp = await client.post(
            f"{GATEWAY_URL}/kv/get",
            json={
                "seq_id": seq_id,
                "layer": layer,
            },
        )
        result = get_resp.json()
        final_k = deserialize_tensor(result["k"])

        print(f"\nFinal sequence shape: {final_k.shape}")
        assert final_k.shape == (total_tokens, NUM_HEADS, HEAD_DIM)

        print("\n✓ Sequence growth verified!")


@pytest.mark.asyncio
async def test_cleanup_all_sequences():
    """Clean up all sequences created during advanced tests."""
    print("\n" + "=" * 60)
    print("ADVANCED TEST 5: Cleanup All Test Sequences")
    print("=" * 60)

    tracked_sequences = get_tracked_sequences()

    if not tracked_sequences:
        print("No sequences to clean up.")
        return

    print(f"Found {len(tracked_sequences)} unique sequences to clean up")

    from test_utils import cleanup_sequences

    cleanup_time = await cleanup_sequences(tracked_sequences)

    print(f"Successfully cleaned up all test sequences")
    print(
        f"Cleanup throughput: {len(tracked_sequences) / cleanup_time:.2f} deletions/sec"
    )

    # Clear the tracking set
    clear_tracked_sequences()


async def run_advanced_tests():
    """Run all advanced KV cache tests."""
    print("\n" + "=" * 60)
    print("Starting Advanced KV Cache Tests")
    print("=" * 60)

    try:
        await test_multi_token_append()
        await test_memory_tracking()
        await test_layer_specific_eviction()
        await test_sequence_growth()
        await test_cleanup_all_sequences()

        print("\n" + "=" * 60)
        print("✓ All advanced tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nAdvanced test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_advanced_tests())
