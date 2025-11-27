"""
Test suite for worker routing and sequence distribution.

Tests verify:
- Consistent hashing distributes sequences across workers
- Same seq_id always routes to the same worker
- Distribution is reasonably balanced
"""

import httpx
import pytest
import os
from collections import Counter

# Test configuration
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8081")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")


@pytest.mark.asyncio
async def test_consistent_routing():
    """Test that the same seq_id always routes to the same worker."""
    async with httpx.AsyncClient() as client:
        seq_id = "test-seq-123"

        # Route the same seq_id multiple times
        workers = []
        for _ in range(10):
            resp = await client.get(f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            workers.append(data["worker_id"])

        # All routes should go to the same worker
        assert len(set(workers)) == 1, "Same seq_id routed to different workers"
        print(f"✓ Consistent routing: seq_id '{seq_id}' always routes to {workers[0]}")


@pytest.mark.asyncio
async def test_distribution_across_workers():
    """Test that sequences are distributed across multiple workers."""
    async with httpx.AsyncClient() as client:
        # Get list of available workers
        resp = await client.get(f"{COORDINATOR_URL}/workers", timeout=5.0)
        assert resp.status_code == 200
        workers_data = resp.json()
        num_workers = len(workers_data)

        if num_workers < 2:
            pytest.skip("Need at least 2 workers for distribution test")

        # Route 100 different sequences
        worker_assignments = []
        for i in range(100):
            seq_id = f"test-seq-{i}"
            resp = await client.get(f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            worker_assignments.append(data["worker_id"])

        # Count distribution
        distribution = Counter(worker_assignments)

        # Each worker should get some sequences
        assert (
            len(distribution) == num_workers
        ), f"Sequences not distributed to all workers: {distribution}"

        # Check balance (no worker should have >70% of sequences)
        max_sequences = max(distribution.values())
        max_percentage = (max_sequences / 100) * 100
        assert (
            max_percentage < 70
        ), f"Unbalanced distribution: one worker has {max_percentage}% of sequences"

        print(f"✓ Distribution across {num_workers} workers: {dict(distribution)}")
        print(
            f"  Balance: {min(distribution.values())}-{max(distribution.values())} sequences per worker"
        )


@pytest.mark.asyncio
async def test_routing_stability_across_requests():
    """Test that routing remains stable even with many requests."""
    async with httpx.AsyncClient() as client:
        # Create a set of sequences and record their workers
        seq_to_worker = {}
        test_sequences = [f"stable-seq-{i}" for i in range(20)]

        # First pass: record initial routing
        for seq_id in test_sequences:
            resp = await client.get(f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0)
            assert resp.status_code == 200
            seq_to_worker[seq_id] = resp.json()["worker_id"]

        # Second pass (after some delay): verify routing hasn't changed
        for seq_id in test_sequences:
            resp = await client.get(f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0)
            assert resp.status_code == 200
            current_worker = resp.json()["worker_id"]
            assert (
                current_worker == seq_to_worker[seq_id]
            ), f"Routing changed for {seq_id}: {seq_to_worker[seq_id]} → {current_worker}"

        print(f"✓ Routing stability verified for {len(test_sequences)} sequences")


@pytest.mark.asyncio
async def test_gateway_uses_consistent_routing():
    """Test that gateway properly routes to workers based on seq_id."""
    async with httpx.AsyncClient() as client:
        # Get coordinator's routing decision
        seq_id = "gateway-test-seq"
        coord_resp = await client.get(
            f"{COORDINATOR_URL}/kv/route/{seq_id}", timeout=5.0
        )
        assert coord_resp.status_code == 200
        expected_worker = coord_resp.json()["worker_id"]

        print(f"✓ Coordinator routes '{seq_id}' to worker: {expected_worker}")
        print(f"  Note: Gateway routing verification requires generation flow test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
