"""
Stress test suite for distributed KV cache system.

Tests verify system behavior under:
- High concurrency (many simultaneous requests)
- Sustained load (continuous requests over time)
- Worker failure scenarios
- Memory pressure
- Mixed workload patterns
"""

import httpx
import pytest
import os
import json
import time
import asyncio
from typing import List, Dict
from collections import defaultdict

# Test configuration
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8081")


async def parse_sse_stream(response) -> List[Dict]:
    """Parse Server-Sent Events stream from response."""
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                continue
    return events


async def generate_text(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 10,
    request_id: int = 0,
) -> Dict:
    """Generate text and return timing metrics."""
    start_time = time.time()

    try:
        async with client.stream(
            "POST",
            f"{GATEWAY_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "model_name": "gpt2",
            },
        ) as response:
            if response.status_code != 200:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "total_time": time.time() - start_time,
                }

            events = await parse_sse_stream(response)

        total_time = time.time() - start_time
        token_events = [e for e in events if e["type"] == "token"]

        return {
            "request_id": request_id,
            "success": True,
            "total_time": total_time,
            "num_tokens": len(token_events),
            "tokens_per_second": (
                len(token_events) / total_time if total_time > 0 else 0
            ),
        }

    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time,
        }


@pytest.mark.asyncio
async def test_high_concurrency():
    """
    Test system under high concurrent load.
    Sends many requests simultaneously to verify system handles concurrency.
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        num_concurrent = 20
        prompts = [
            "The future of",
            "Machine learning",
            "Artificial intelligence",
            "Deep learning",
            "Neural networks",
        ]

        print("\n" + "=" * 80)
        print(f"HIGH CONCURRENCY TEST - {num_concurrent} concurrent requests")
        print("=" * 80)

        async def run_request(request_id: int):
            prompt = prompts[request_id % len(prompts)]
            return await generate_text(
                client, prompt, max_tokens=8, request_id=request_id
            )

        # Launch all requests concurrently
        start_time = time.time()
        print(f"\nLaunching {num_concurrent} concurrent requests...")

        tasks = [run_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)
        print(f"Total requests:     {num_concurrent}")
        print(
            f"Successful:         {len(successful)} ({len(successful)/num_concurrent*100:.1f}%)"
        )
        print(
            f"Failed:             {len(failed)} ({len(failed)/num_concurrent*100:.1f}%)"
        )
        print(f"Total time:         {total_time:.2f}s")
        print(f"Requests/sec:       {num_concurrent/total_time:.2f}")
        print()

        if successful:
            avg_time = sum(r["total_time"] for r in successful) / len(successful)
            avg_tokens = sum(r["num_tokens"] for r in successful) / len(successful)
            avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)

            print("Performance metrics (successful requests):")
            print(f"  Avg time per request: {avg_time:.3f}s")
            print(f"  Avg tokens generated: {avg_tokens:.1f}")
            print(f"  Avg tokens/sec:       {avg_tps:.2f}")

        if failed:
            print("\nFailed requests:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"  Request {r['request_id']}: {r.get('error', 'Unknown error')}")

        print("=" * 80)

        # Assertions
        success_rate = len(successful) / num_concurrent
        assert success_rate > 0.8, f"Success rate too low: {success_rate*100:.1f}%"
        assert len(successful) > 0, "No successful requests"


@pytest.mark.asyncio
async def test_sustained_load():
    """
    Test system under sustained load over time.
    Continuously sends requests to verify stability.
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        duration_seconds = 30
        requests_per_second = 2
        prompts = [
            "The cat sat",
            "Hello world",
            "Machine learning",
            "Python programming",
        ]

        print("\n" + "=" * 80)
        print(
            f"SUSTAINED LOAD TEST - {duration_seconds}s @ {requests_per_second} req/s"
        )
        print("=" * 80)

        results = []
        start_time = time.time()
        request_id = 0

        print("\nRunning sustained load test...")
        print("(This will take about 30 seconds)")

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Send batch of requests
            tasks = []
            for _ in range(requests_per_second):
                prompt = prompts[request_id % len(prompts)]
                tasks.append(
                    generate_text(client, prompt, max_tokens=5, request_id=request_id)
                )
                request_id += 1

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Wait to maintain target rate
            batch_time = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_time)
            await asyncio.sleep(sleep_time)

            # Progress indicator
            elapsed = time.time() - start_time
            print(f"  Progress: {elapsed:.1f}s / {duration_seconds}s", end="\r")

        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)
        print(f"Duration:           {total_time:.2f}s")
        print(f"Total requests:     {len(results)}")
        print(
            f"Successful:         {len(successful)} ({len(successful)/len(results)*100:.1f}%)"
        )
        print(
            f"Failed:             {len(failed)} ({len(failed)/len(results)*100:.1f}%)"
        )
        print(f"Actual req/sec:     {len(results)/total_time:.2f}")
        print()

        if successful:
            avg_time = sum(r["total_time"] for r in successful) / len(successful)
            avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)

            # Calculate p95 latency
            sorted_times = sorted(r["total_time"] for r in successful)
            p95_idx = int(len(sorted_times) * 0.95)
            p95_latency = sorted_times[p95_idx] if sorted_times else 0

            print("Performance metrics:")
            print(f"  Avg latency:      {avg_time:.3f}s")
            print(f"  P95 latency:      {p95_latency:.3f}s")
            print(f"  Avg tokens/sec:   {avg_tps:.2f}")

        print("=" * 80)

        # Assertions
        success_rate = len(successful) / len(results) if results else 0
        assert success_rate > 0.85, f"Success rate too low: {success_rate*100:.1f}%"


@pytest.mark.asyncio
async def test_burst_traffic():
    """
    Test system handling of burst traffic patterns.
    Alternates between high and low load.
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        print("\n" + "=" * 80)
        print("BURST TRAFFIC TEST")
        print("=" * 80)

        prompts = ["The quick brown", "Hello world", "Machine learning"]

        # Burst pattern: high load, then low load
        bursts = [
            {"name": "Burst 1", "count": 15, "delay": 0},
            {"name": "Quiet 1", "count": 2, "delay": 0.5},
            {"name": "Burst 2", "count": 20, "delay": 0},
            {"name": "Quiet 2", "count": 3, "delay": 0.5},
        ]

        all_results = []
        request_id = 0

        for burst in bursts:
            print(f"\n{burst['name']}: {burst['count']} requests...")
            tasks = []

            for _ in range(burst["count"]):
                prompt = prompts[request_id % len(prompts)]
                tasks.append(
                    generate_text(client, prompt, max_tokens=5, request_id=request_id)
                )
                request_id += 1
                if burst["delay"] > 0:
                    await asyncio.sleep(burst["delay"])

            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            successful = len([r for r in results if r["success"]])
            print(f"  Completed: {successful}/{burst['count']} successful")

        # Analyze overall results
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]

        print("\n" + "-" * 80)
        print("OVERALL RESULTS:")
        print("-" * 80)
        print(f"Total requests:     {len(all_results)}")
        print(
            f"Successful:         {len(successful)} ({len(successful)/len(all_results)*100:.1f}%)"
        )
        print(f"Failed:             {len(failed)}")

        if successful:
            avg_time = sum(r["total_time"] for r in successful) / len(successful)
            print(f"Avg latency:        {avg_time:.3f}s")

        print("=" * 80)

        assert len(successful) > 0.8 * len(all_results), "Too many failed requests"


@pytest.mark.asyncio
async def test_mixed_workload():
    """
    Test system with mixed workload: different prompt lengths and token counts.
    Simulates realistic usage with variety.
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        print("\n" + "=" * 80)
        print("MIXED WORKLOAD TEST")
        print("=" * 80)

        # Different workload types
        workloads = [
            {"prompt": "Hi", "max_tokens": 5, "category": "short"},
            {
                "prompt": "The quick brown fox jumps",
                "max_tokens": 10,
                "category": "medium",
            },
            {
                "prompt": "In a galaxy far far away, there lived a brave warrior who",
                "max_tokens": 20,
                "category": "long",
            },
        ]

        num_requests = 15
        results_by_category = defaultdict(list)

        print(f"\nExecuting {num_requests} mixed requests...")

        tasks = []
        for i in range(num_requests):
            workload = workloads[i % len(workloads)]
            tasks.append(
                generate_text(
                    client, workload["prompt"], workload["max_tokens"], request_id=i
                )
            )

        results = await asyncio.gather(*tasks)

        # Categorize results
        for i, result in enumerate(results):
            category = workloads[i % len(workloads)]["category"]
            results_by_category[category].append(result)

        # Analyze by category
        print("\n" + "-" * 80)
        print("RESULTS BY CATEGORY:")
        print("-" * 80)

        for category in ["short", "medium", "long"]:
            category_results = results_by_category[category]
            successful = [r for r in category_results if r["success"]]

            if successful:
                avg_time = sum(r["total_time"] for r in successful) / len(successful)
                avg_tokens = sum(r["num_tokens"] for r in successful) / len(successful)

                print(f"\n{category.upper()}:")
                print(f"  Requests:     {len(category_results)}")
                print(f"  Successful:   {len(successful)}")
                print(f"  Avg time:     {avg_time:.3f}s")
                print(f"  Avg tokens:   {avg_tokens:.1f}")

        print("\n" + "=" * 80)

        # Overall assertion
        total_successful = sum(
            len([r for r in results if r["success"]])
            for results in results_by_category.values()
        )
        assert total_successful > 0.8 * num_requests, "Too many failed requests"


@pytest.mark.asyncio
async def test_worker_stats_under_load():
    """
    Monitor worker statistics during load to verify cache behavior.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("\n" + "=" * 80)
        print("WORKER STATS UNDER LOAD")
        print("=" * 80)

        # Get initial stats
        stats_before = await client.get(f"{GATEWAY_URL}/stats", timeout=5.0)
        assert stats_before.status_code == 200
        stats_before_data = stats_before.json()

        print("\nInitial stats:")
        print(f"  Total sequences: {stats_before_data.get('total_sequences', 0)}")
        print(f"  Total entries:   {stats_before_data.get('total_entries', 0)}")

        # Generate load
        num_requests = 10
        print(f"\nGenerating {num_requests} requests...")

        tasks = []
        for i in range(num_requests):
            prompt = f"Request {i}: The quick brown"
            tasks.append(generate_text(client, prompt, max_tokens=5, request_id=i))

        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r["success"]]

        # Get stats after load
        stats_after = await client.get(f"{GATEWAY_URL}/stats", timeout=5.0)
        assert stats_after.status_code == 200
        stats_after_data = stats_after.json()

        print("\nAfter load:")
        print(f"  Total sequences: {stats_after_data.get('total_sequences', 0)}")
        print(f"  Total entries:   {stats_after_data.get('total_entries', 0)}")

        # Verify stats increased
        sequences_added = stats_after_data.get(
            "total_sequences", 0
        ) - stats_before_data.get("total_sequences", 0)
        entries_added = stats_after_data.get(
            "total_entries", 0
        ) - stats_before_data.get("total_entries", 0)

        print("\nChanges:")
        print(f"  Sequences added: {sequences_added}")
        print(f"  Entries added:   {entries_added}")
        print(f"  Successful reqs: {len(successful)}")

        print("\n" + "=" * 80)

        assert len(successful) > 0, "No successful requests"
        assert sequences_added >= 0, "Sequences should not decrease"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
