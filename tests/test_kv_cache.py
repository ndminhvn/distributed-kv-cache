#!/usr/bin/env python3
"""
Synthetic test for distributed KV cache system.
Simulates LLM inference workload with multiple sequences, layers, and steps.
"""

import httpx
import asyncio
import time
import pytest
import torch
from typing import List, Dict

from test_utils import (
    cleanup_sequences,
    track_sequence,
    get_tracked_sequences,
    clear_tracked_sequences,
    GATEWAY_URL,
)
from tensor_utils import serialize_tensor, deserialize_tensor, generate_fake_kv_tensor


NUM_SEQUENCES = 5
NUM_LAYERS = 12  # Typical transformer model (e.g., GPT-2 small has 12 layers)
NUM_STEPS = 10  # Number of tokens to generate per sequence

# Realistic transformer KV cache dimensions
NUM_HEADS = 8
HEAD_DIM = 64


@pytest.mark.asyncio
async def test_kv_put_get():
    """Test basic KV put and get operations."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic KV Put/Get")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        seq_id = "test_seq_001"
        track_sequence(seq_id)
        layer = 0

        # Generate PyTorch tensors in new format: [seq_len, num_heads, head_dim]
        # Start with 1 token
        k_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
        v_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

        # Put KV
        print(f"\nPutting KV for seq={seq_id}, layer={layer}")
        print(f"   K tensor shape: {k_tensor.shape}, dtype: {k_tensor.dtype}")
        print(f"   V tensor shape: {v_tensor.shape}, dtype: {v_tensor.dtype}")

        put_resp = await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "k": serialize_tensor(k_tensor),
                "v": serialize_tensor(v_tensor),
            },
        )
        put_result = put_resp.json()
        print(f"PUT Response: {put_result}")

        # Get KV
        print(f"\nGetting KV for seq={seq_id}, layer={layer}")
        get_resp = await client.post(
            f"{GATEWAY_URL}/kv/get",
            json={
                "seq_id": seq_id,
                "layer": layer,
            },
        )
        result = get_resp.json()

        # Deserialize retrieved tensors
        retrieved_k = deserialize_tensor(result["k"])
        retrieved_v = deserialize_tensor(result["v"])

        print(f"GET Response (worker={result.get('worker_id')})")
        print(f"   Retrieved K shape: {retrieved_k.shape}, dtype: {retrieved_k.dtype}")
        print(f"   Retrieved V shape: {retrieved_v.shape}, dtype: {retrieved_v.dtype}")
        print(f"   Sequence length: {result.get('seq_len')}")

        # Verify data integrity
        assert torch.allclose(retrieved_k, k_tensor), "K tensor mismatch!"
        assert torch.allclose(retrieved_v, v_tensor), "V tensor mismatch!"
        print("✓ Data integrity verified!")


@pytest.mark.asyncio
async def test_sequence_distribution():
    """Test that sequences are distributed across workers."""
    print("\n" + "=" * 60)
    print("TEST 2: Sequence Distribution Across Workers")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        sequence_to_worker = {}

        # Generate multiple sequences
        for seq_num in range(NUM_SEQUENCES):
            seq_id = f"seq_{seq_num:03d}"
            track_sequence(seq_id)

            # Put KV for first layer with PyTorch tensors (1 token)
            k_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
            v_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

            put_resp = await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": 0,
                    "k": serialize_tensor(k_tensor),
                    "v": serialize_tensor(v_tensor),
                },
            )

            worker_id = put_resp.json()["worker_id"]
            sequence_to_worker[seq_id] = worker_id
            print(f"{seq_id} -> worker: {worker_id}")

        # Check distribution
        print(f"\nDistribution Summary:")
        worker_counts = {}
        for seq_id, worker_id in sequence_to_worker.items():
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

        for worker_id, count in sorted(worker_counts.items()):
            print(f"   {worker_id}: {count} sequences")

        print(f"\nTotal sequences distributed: {len(sequence_to_worker)}")
        print(f"Number of workers used: {len(worker_counts)}")


@pytest.mark.asyncio
async def test_full_inference_simulation():
    """Simulate full LLM inference: multiple sequences × layers × tokens."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Inference Simulation")
    print("=" * 60)
    print(
        f"Simulating {NUM_SEQUENCES} sequences × {NUM_LAYERS} layers × {NUM_STEPS} tokens"
    )
    print(f"Total operations: {NUM_SEQUENCES * NUM_LAYERS * NUM_STEPS}")

    start_time = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        total_appends = 0
        total_gets = 0

        for seq_num in range(NUM_SEQUENCES):
            seq_id = f"inference_seq_{seq_num:03d}"
            track_sequence(seq_id)
            print(f"\nProcessing {seq_id}...")

            # Simulate generation: for each token, append KV for all layers
            for token_idx in range(NUM_STEPS):
                for layer in range(NUM_LAYERS):
                    # Generate new token's KV [1, num_heads, head_dim]
                    k_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)
                    v_tensor = generate_fake_kv_tensor(1, NUM_HEADS, HEAD_DIM)

                    if token_idx == 0:
                        # First token: use put
                        await client.post(
                            f"{GATEWAY_URL}/kv/put",
                            json={
                                "seq_id": seq_id,
                                "layer": layer,
                                "k": serialize_tensor(k_tensor),
                                "v": serialize_tensor(v_tensor),
                            },
                        )
                    else:
                        # Subsequent tokens: use append
                        await client.post(
                            f"{GATEWAY_URL}/kv/append",
                            json={
                                "seq_id": seq_id,
                                "layer": layer,
                                "k": serialize_tensor(k_tensor),
                                "v": serialize_tensor(v_tensor),
                            },
                        )
                    total_appends += 1

                # Simulate retrieval (e.g., for attention computation)
                # Get KV from a random layer
                import random

                random_layer = random.randint(0, NUM_LAYERS - 1)
                await client.post(
                    f"{GATEWAY_URL}/kv/get",
                    json={
                        "seq_id": seq_id,
                        "layer": random_layer,
                    },
                )
                total_gets += 1

            print(
                f"   Completed {seq_id}: {NUM_LAYERS * NUM_STEPS} operations, {NUM_STEPS} gets"
            )

        elapsed = time.time() - start_time
        print(f"\n⚡ Performance Summary:")
        print(f"   Total operations: {total_appends}")
        print(f"   Total GETs: {total_gets}")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Throughput: {(total_appends + total_gets) / elapsed:.2f} ops/sec")


@pytest.mark.asyncio
async def test_sequence_locality():
    """Verify that all entries for a sequence go to the same worker."""
    print("\n" + "=" * 60)
    print("TEST 4: Sequence Locality (Same Worker)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        seq_id = "locality_test_seq"
        track_sequence(seq_id)
        workers_seen = set()

        print(f"\nPutting KV entries for {seq_id} across multiple layers...")

        # Put entries for multiple layers
        for layer in range(3):
            # Put initial KV with PyTorch tensors (3 tokens)
            k_tensor = generate_fake_kv_tensor(3, NUM_HEADS, HEAD_DIM)
            v_tensor = generate_fake_kv_tensor(3, NUM_HEADS, HEAD_DIM)

            put_resp = await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "k": serialize_tensor(k_tensor),
                    "v": serialize_tensor(v_tensor),
                },
            )
            worker_id = put_resp.json()["worker_id"]
            workers_seen.add(worker_id)
            print(f"   layer={layer} -> {worker_id}")

        print(f"\nLocality Check:")
        print(f"   Total layers: 3")
        print(f"   Workers used: {len(workers_seen)}")
        print(f"   Worker(s): {workers_seen}")

        if len(workers_seen) == 1:
            print("✓ Perfect locality! All entries on same worker.")
        else:
            print("⚠ Warning: Entries spread across multiple workers (unexpected!)")


@pytest.mark.asyncio
async def test_cleanup_all_sequences():
    """Clean up all sequences created during KV cache tests."""
    print("\n" + "=" * 60)
    print("TEST 5: Cleanup All Test Sequences")
    print("=" * 60)

    tracked_sequences = get_tracked_sequences()

    if not tracked_sequences:
        print("No sequences to clean up.")
        return

    print(f"Found {len(tracked_sequences)} unique sequences to clean up")

    cleanup_time = await cleanup_sequences(tracked_sequences)

    print(f"Successfully cleaned up all test sequences")
    print(
        f"Cleanup throughput: {len(tracked_sequences) / cleanup_time:.2f} deletions/sec"
    )

    # Clear the tracking set
    clear_tracked_sequences()


async def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "=" * 60)
    print("Starting Distributed KV Cache Tests")
    print("=" * 60)

    try:
        await test_kv_put_get()
        await test_sequence_distribution()
        await test_full_inference_simulation()
        await test_sequence_locality()
        await test_cleanup_all_sequences()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
