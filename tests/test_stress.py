#!/usr/bin/env python3
"""
Stress test for distributed KV cache system.
Tests concurrent load, throughput, and distribution across workers.
"""

import httpx
import asyncio
import random
import time
import pytest
from typing import List, Dict
from collections import defaultdict

from test_utils import (
    cleanup_sequences,
    track_sequence,
    get_tracked_sequences,
    clear_tracked_sequences,
    GATEWAY_URL,
)


K_DIM = 64
V_DIM = 64


def generate_fake_tensor(dim: int) -> List[float]:
    """Generate a fake tensor as a list of floats."""
    return [random.random() for _ in range(dim)]


@pytest.mark.asyncio
async def test_concurrent_writes():
    """Test concurrent write operations to simulate real-world load."""
    print("\n" + "=" * 60)
    print("STRESS TEST 1: Concurrent Writes")
    print("=" * 60)

    NUM_CONCURRENT = 500
    NUM_SEQUENCES = 50
    NUM_LAYERS = 12

    print(f"Simulating {NUM_CONCURRENT} concurrent requests...")
    print(f"Total operations: {NUM_CONCURRENT}")

    start_time = time.time()

    async def write_kv_entry(client, idx):
        """Write a single KV entry."""
        seq_id = f"concurrent_seq_{idx % NUM_SEQUENCES:03d}"
        track_sequence(seq_id)
        layer = idx % NUM_LAYERS
        step = idx // NUM_LAYERS

        await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "step": step,
                "k": generate_fake_tensor(K_DIM),
                "v": generate_fake_tensor(V_DIM),
            },
            timeout=30.0,
        )
        return seq_id

    async with httpx.AsyncClient() as client:
        # Execute all writes concurrently
        tasks = [write_kv_entry(client, i) for i in range(NUM_CONCURRENT)]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print(f"\nCompleted {NUM_CONCURRENT} concurrent writes")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {NUM_CONCURRENT / elapsed:.2f} writes/sec")
    print(f"   Avg latency: {(elapsed / NUM_CONCURRENT) * 1000:.2f}ms per write")


@pytest.mark.asyncio
async def test_mixed_read_write_load():
    """Test mixed read/write workload with concurrent operations."""
    print("\n" + "=" * 60)
    print("STRESS TEST 2: Mixed Read/Write Load")
    print("=" * 60)

    NUM_SEQUENCES = 50
    NUM_LAYERS = 24
    NUM_STEPS = 10
    READ_WRITE_RATIO = 0.7  # 70% reads, 30% writes

    print(f"Simulating mixed workload with {int(READ_WRITE_RATIO*100)}% reads...")

    # First, populate cache with some data
    print("\nPopulating initial cache...")
    populate_start = time.time()

    async with httpx.AsyncClient() as client:
        populate_tasks = []
        for seq_num in range(NUM_SEQUENCES):
            seq_id = f"mixed_seq_{seq_num:03d}"
            track_sequence(seq_id)
            for layer in range(NUM_LAYERS):
                for step in range(NUM_STEPS):
                    populate_tasks.append(
                        client.post(
                            f"{GATEWAY_URL}/kv/put",
                            json={
                                "seq_id": seq_id,
                                "layer": layer,
                                "step": step,
                                "k": generate_fake_tensor(K_DIM),
                                "v": generate_fake_tensor(V_DIM),
                            },
                            timeout=30.0,
                        )
                    )

        # Execute in batches to avoid overwhelming the system
        batch_size = 100
        for i in range(0, len(populate_tasks), batch_size):
            batch = populate_tasks[i : i + batch_size]
            await asyncio.gather(*batch)

    populate_elapsed = time.time() - populate_start
    total_entries = NUM_SEQUENCES * NUM_LAYERS * NUM_STEPS
    print(f"Populated cache with {total_entries} entries in {populate_elapsed:.2f}s")
    print(
        f"   Population throughput: {total_entries / populate_elapsed:.2f} writes/sec"
    )

    # Now run mixed workload
    print("\nRunning mixed read/write workload...")

    NUM_OPERATIONS = 1000
    read_count = 0
    write_count = 0

    async def random_operation(client, idx):
        """Perform a random read or write operation."""
        nonlocal read_count, write_count

        seq_id = f"mixed_seq_{random.randint(0, NUM_SEQUENCES-1):03d}"
        layer = random.randint(0, NUM_LAYERS - 1)
        step = random.randint(0, NUM_STEPS - 1)

        if random.random() < READ_WRITE_RATIO:
            # Read operation
            try:
                await client.post(
                    f"{GATEWAY_URL}/kv/get",
                    json={"seq_id": seq_id, "layer": layer, "step": step},
                    timeout=30.0,
                )
                read_count += 1
                return "read"
            except:
                pass
        else:
            # Write operation
            await client.post(
                f"{GATEWAY_URL}/kv/put",
                json={
                    "seq_id": seq_id,
                    "layer": layer,
                    "step": step,
                    "k": generate_fake_tensor(K_DIM),
                    "v": generate_fake_tensor(V_DIM),
                },
                timeout=30.0,
            )
            write_count += 1
            return "write"

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [random_operation(client, i) for i in range(NUM_OPERATIONS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    print(f"\nMixed Workload Results:")
    print(f"   Total operations: {NUM_OPERATIONS}")
    print(f"   Reads: {read_count} ({read_count/NUM_OPERATIONS*100:.1f}%)")
    print(f"   Writes: {write_count} ({write_count/NUM_OPERATIONS*100:.1f}%)")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {NUM_OPERATIONS / elapsed:.2f} ops/sec")


@pytest.mark.asyncio
async def test_worker_distribution_under_load():
    """Test how sequences are distributed across workers under heavy load."""
    print("\n" + "=" * 60)
    print("STRESS TEST 3: Worker Distribution Analysis")
    print("=" * 60)

    NUM_SEQUENCES = 500

    print(f"Creating {NUM_SEQUENCES} sequences to analyze distribution...")

    sequence_to_worker = {}

    async def get_worker_for_sequence(client, seq_num):
        """Get which worker handles a given sequence."""
        seq_id = f"dist_seq_{seq_num:04d}"
        track_sequence(seq_id)

        response = await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": 0,
                "step": 0,
                "k": generate_fake_tensor(K_DIM),
                "v": generate_fake_tensor(V_DIM),
            },
            timeout=30.0,
        )

        worker_id = response.json()["worker_id"]
        return seq_id, worker_id

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [get_worker_for_sequence(client, i) for i in range(NUM_SEQUENCES)]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Analyze distribution
    for seq_id, worker_id in results:
        sequence_to_worker[seq_id] = worker_id

    worker_counts = defaultdict(int)
    for worker_id in sequence_to_worker.values():
        worker_counts[worker_id] += 1

    print(f"\nDistribution Analysis ({NUM_SEQUENCES} sequences):")
    print(f"   Total workers: {len(worker_counts)}")

    for worker_id, count in sorted(
        worker_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / NUM_SEQUENCES) * 100
        bar = "#" * int(percentage / 2)
        print(f"   {worker_id}: {count:3d} sequences ({percentage:5.1f}%) {bar}")

    # Calculate distribution metrics
    avg_per_worker = NUM_SEQUENCES / len(worker_counts)
    max_deviation = max(abs(count - avg_per_worker) for count in worker_counts.values())

    print(f"\nDistribution Metrics:")
    print(f"   Average per worker: {avg_per_worker:.1f}")
    print(f"   Max deviation: {max_deviation:.1f}")
    print(f"   Time to route: {elapsed:.2f}s")
    print(f"   Routing throughput: {NUM_SEQUENCES / elapsed:.2f} routes/sec")


@pytest.mark.asyncio
async def test_high_throughput_burst():
    """Test system behavior under high-throughput burst traffic."""
    print("\n" + "=" * 60)
    print("STRESS TEST 4: High-Throughput Burst")
    print("=" * 60)

    BURST_SIZE = 2000
    NUM_UNIQUE_SEQUENCES = 100

    print(
        f"Sending burst of {BURST_SIZE} requests across {NUM_UNIQUE_SEQUENCES} sequences..."
    )

    async def burst_request(client, idx):
        """Send a single burst request."""
        seq_id = f"burst_seq_{idx % NUM_UNIQUE_SEQUENCES:03d}"
        track_sequence(seq_id)
        layer = random.randint(0, 23)  # 24 layers
        step = random.randint(0, 19)  # 20 steps

        await client.post(
            f"{GATEWAY_URL}/kv/put",
            json={
                "seq_id": seq_id,
                "layer": layer,
                "step": step,
                "k": generate_fake_tensor(K_DIM),
                "v": generate_fake_tensor(V_DIM),
            },
            timeout=30.0,
        )

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [burst_request(client, i) for i in range(BURST_SIZE)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Check for failures
    failures = sum(1 for r in results if isinstance(r, Exception))
    successes = BURST_SIZE - failures

    print(f"\nBurst Test Results:")
    print(f"   Total requests: {BURST_SIZE}")
    print(f"   Successful: {successes} ({successes/BURST_SIZE*100:.1f}%)")
    print(f"   Failed: {failures} ({failures/BURST_SIZE*100:.1f}%)")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {successes / elapsed:.2f} ops/sec")
    print(f"   Avg latency: {(elapsed / BURST_SIZE) * 1000:.2f}ms")

    if failures > 0:
        print(f"\nWarning: {failures} requests failed")


@pytest.mark.asyncio
async def test_cleanup_all_sequences():
    """Clean up all sequences created during stress tests."""
    print("\n" + "=" * 60)
    print("STRESS TEST 5: Cleanup All Test Sequences")
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


async def run_stress_tests():
    """Run all stress tests."""
    print("\n" + "=" * 60)
    print("Starting Stress Tests")
    print("=" * 60)

    try:
        await test_concurrent_writes()
        await test_mixed_read_write_load()
        await test_worker_distribution_under_load()
        await test_high_throughput_burst()
        await test_cleanup_all_sequences()

        print("\n" + "=" * 60)
        print("All stress tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nStress test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_stress_tests())
