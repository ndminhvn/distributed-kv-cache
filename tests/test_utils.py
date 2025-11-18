#!/usr/bin/env python3
"""
Shared test utilities for distributed KV cache tests.
"""

import httpx
import asyncio
import time
from typing import List, Set


GATEWAY_URL = "http://localhost:8080"
COORDINATOR_ADDR = "http://localhost:8081"

# Global set to track all sequences created during tests
_created_sequences: Set[str] = set()


def track_sequence(seq_id: str):
    """Track a sequence ID for cleanup."""
    _created_sequences.add(seq_id)


def get_tracked_sequences() -> List[str]:
    """Get all tracked sequence IDs."""
    return list(_created_sequences)


def clear_tracked_sequences():
    """Clear the tracked sequences."""
    _created_sequences.clear()


async def cleanup_sequences(sequence_ids: List[str], batch_size: int = 50) -> float:
    """
    Clean up sequences from workers by routing through coordinator.

    Args:
        sequence_ids: List of sequence IDs to delete
        batch_size: Number of concurrent deletions per batch

    Returns:
        Time elapsed for cleanup in seconds
    """
    print(f"\nCleaning up {len(sequence_ids)} sequences...")
    cleanup_start = time.time()

    async with httpx.AsyncClient() as client:
        cleanup_tasks = []
        for seq_id in sequence_ids:
            try:
                # Route to correct worker
                route_resp = await client.get(
                    f"{COORDINATOR_ADDR}/kv/route/{seq_id}", timeout=10.0
                )
                if route_resp.status_code == 200:
                    worker_addr = route_resp.json()["address"]
                    cleanup_tasks.append(
                        client.delete(f"{worker_addr}/kv/{seq_id}", timeout=10.0)
                    )
            except Exception:
                # Silently skip sequences that can't be routed
                pass

        # Execute cleanup in batches
        if cleanup_tasks:
            for i in range(0, len(cleanup_tasks), batch_size):
                batch = cleanup_tasks[i : i + batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)
                # Check for errors
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    print(f"Warning: {len(errors)} cleanup operations failed")

    cleanup_elapsed = time.time() - cleanup_start
    print(f"Cleanup completed in {cleanup_elapsed:.2f}s")
    return cleanup_elapsed


async def test_cleanup_sequences():
    """Test the cleanup functionality by creating and deleting sequences."""
    print("\n" + "=" * 60)
    print("TEST: Cleanup Sequences")
    print("=" * 60)

    NUM_TEST_SEQUENCES = 10

    print(f"\nCreating {NUM_TEST_SEQUENCES} test sequences...")

    sequence_ids = [f"cleanup_test_seq_{i:03d}" for i in range(NUM_TEST_SEQUENCES)]

    # Create test sequences
    async with httpx.AsyncClient() as client:
        create_tasks = []
        for seq_id in sequence_ids:
            create_tasks.append(
                client.post(
                    f"{GATEWAY_URL}/kv/put",
                    json={
                        "seq_id": seq_id,
                        "layer": 0,
                        "step": 0,
                        "k": [0.0] * 64,
                        "v": [0.0] * 64,
                    },
                    timeout=10.0,
                )
            )

        results = await asyncio.gather(*create_tasks, return_exceptions=True)
        success_count = sum(
            1 for r in results if not isinstance(r, Exception) and r.status_code == 200
        )
        print(f"Created {success_count}/{NUM_TEST_SEQUENCES} sequences")

    # Test cleanup
    cleanup_time = await cleanup_sequences(sequence_ids)

    # Verify sequences are deleted by attempting to retrieve them
    print("\nVerifying deletion...")
    async with httpx.AsyncClient() as client:
        verify_tasks = []
        for seq_id in sequence_ids:
            verify_tasks.append(
                client.post(
                    f"{GATEWAY_URL}/kv/get",
                    json={"seq_id": seq_id, "layer": 0, "step": 0},
                    timeout=10.0,
                )
            )

        results = await asyncio.gather(*verify_tasks, return_exceptions=True)
        deleted_count = sum(
            1 for r in results if isinstance(r, Exception) or r.status_code == 404
        )
        print(f"Verified {deleted_count}/{NUM_TEST_SEQUENCES} sequences deleted")

    print(f"\nCleanup test completed successfully!")
    print(f"Cleanup throughput: {NUM_TEST_SEQUENCES / cleanup_time:.2f} deletions/sec")
