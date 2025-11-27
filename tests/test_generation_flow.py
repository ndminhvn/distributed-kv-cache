"""
Test suite for end-to-end generation flow.

Tests verify:
- /generate endpoint works correctly
- Model auto-initialization
- Streaming response format
- Token generation quality
- Performance metrics
"""

import httpx
import pytest
import os
import json
import time
from typing import List, Dict

# Test configuration
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8081")


async def parse_sse_stream(response) -> List[Dict]:
    """Parse Server-Sent Events stream from response."""
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])  # Skip "data: " prefix
                events.append(data)
            except json.JSONDecodeError:
                continue
    return events


@pytest.mark.asyncio
async def test_basic_generation():
    """Test basic text generation with streaming."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = "Once upon a time"

        start_time = time.time()

        async with client.stream(
            "POST",
            f"{GATEWAY_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.7,
                "model_name": "gpt2",
            },
        ) as response:
            assert response.status_code == 200
            events = await parse_sse_stream(response)

        elapsed = time.time() - start_time

        # Verify event structure
        assert len(events) > 0, "No events received"

        # First event should be 'start'
        assert (
            events[0]["type"] == "start"
        ), f"First event should be 'start', got {events[0]['type']}"
        assert events[0]["prompt"] == prompt
        assert "seq_id" in events[0]

        # Middle events should be 'token'
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0, "No token events received"

        for token_event in token_events:
            assert "token" in token_event
            assert "token_id" in token_event

        # Last event should be 'done'
        assert (
            events[-1]["type"] == "done"
        ), f"Last event should be 'done', got {events[-1]['type']}"
        assert "generated_text" in events[-1]
        assert "tokens_generated" in events[-1]

        # Extract generated text
        tokens = [e["token"] for e in token_events]
        generated_text = "".join(tokens)

        print(f"✓ Basic generation test passed:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: '{generated_text}'")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens/sec: {len(tokens)/elapsed:.2f}")


@pytest.mark.asyncio
async def test_model_auto_initialization():
    """Test that model initializes automatically on first generation."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Get workers and check model status
        workers_resp = await client.get(f"{COORDINATOR_URL}/workers", timeout=5.0)
        assert workers_resp.status_code == 200
        workers = workers_resp.json()

        if not workers:
            pytest.skip("No workers available")

        # Try generation - model should auto-initialize
        async with client.stream(
            "POST",
            f"{GATEWAY_URL}/generate",
            json={"prompt": "Hello", "max_tokens": 5, "model_name": "gpt2"},
        ) as response:
            assert (
                response.status_code == 200
            ), f"Generation failed (auto-init should work): {response.status_code}"
            events = await parse_sse_stream(response)

        # Verify we got tokens (meaning model loaded successfully)
        token_events = [e for e in events if e["type"] == "token"]
        assert (
            len(token_events) > 0
        ), "Model did not generate tokens (auto-init may have failed)"

        print(f"✓ Model auto-initialization verified:")
        print(f"  Model: gpt2")
        print(f"  Generated {len(token_events)} tokens")


@pytest.mark.asyncio
async def test_generation_with_different_temperatures():
    """Test generation with different temperature settings."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = "The quick brown"
        temperatures = [0.1, 0.7, 1.0]

        results = {}

        for temp in temperatures:
            async with client.stream(
                "POST",
                f"{GATEWAY_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 8,
                    "temperature": temp,
                    "model_name": "gpt2",
                },
            ) as response:
                assert response.status_code == 200
                events = await parse_sse_stream(response)

            tokens = [e["token"] for e in events if e["type"] == "token"]
            generated = "".join(tokens)
            results[temp] = generated

        print(f"✓ Temperature variation test:")
        for temp, text in results.items():
            print(f"  T={temp}: '{text}'")


@pytest.mark.asyncio
async def test_concurrent_generations():
    """Test multiple concurrent generation requests."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompts = ["The cat sat on", "In a galaxy far", "Machine learning is"]

        import asyncio

        async def generate_one(prompt: str, idx: int):
            start = time.time()
            async with client.stream(
                "POST",
                f"{GATEWAY_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 5,
                    "temperature": 0.7,
                    "model_name": "gpt2",
                },
            ) as response:
                assert response.status_code == 200
                events = await parse_sse_stream(response)

            elapsed = time.time() - start
            tokens = [e["token"] for e in events if e["type"] == "token"]

            # Get seq_id from start event
            start_event = next(e for e in events if e["type"] == "start")
            seq_id = start_event["seq_id"]

            return {
                "idx": idx,
                "prompt": prompt,
                "seq_id": seq_id,
                "tokens": len(tokens),
                "time": elapsed,
                "generated": "".join(tokens),
            }

        # Run concurrently
        tasks = [generate_one(p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

        print(f"✓ Concurrent generations test:")
        print(f"  Requests: {len(prompts)}")
        for r in results:
            print(
                f"  [{r['idx']}] '{r['prompt']}' → {r['tokens']} tokens in {r['time']:.2f}s"
            )

        # Verify all completed successfully
        assert all(
            r["tokens"] > 0 for r in results
        ), "Some generations produced no tokens"


@pytest.mark.asyncio
async def test_generation_performance_metrics():
    """Test and collect performance metrics for generation."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        prompt = "The future of artificial intelligence"
        max_tokens = 20

        # Warm-up request
        async with client.stream(
            "POST",
            f"{GATEWAY_URL}/generate",
            json={"prompt": "warm up", "max_tokens": 3, "model_name": "gpt2"},
        ) as response:
            await parse_sse_stream(response)

        # Actual test
        start_time = time.time()
        first_token_time = None

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
            assert response.status_code == 200

            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append(data)

                    # Record first token time
                    if data["type"] == "token" and first_token_time is None:
                        first_token_time = time.time()

        total_time = time.time() - start_time

        token_events = [e for e in events if e["type"] == "token"]
        num_tokens = len(token_events)

        # Calculate metrics
        time_to_first_token = first_token_time - start_time if first_token_time else 0
        tokens_per_second = num_tokens / total_time if total_time > 0 else 0

        print(f"✓ Performance metrics:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Total tokens: {num_tokens}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Time to first token: {time_to_first_token:.3f}s")
        print(f"  Tokens/sec: {tokens_per_second:.2f}")
        print(f"  Avg time/token: {(total_time/num_tokens*1000):.1f}ms")

        # Performance assertions
        assert (
            time_to_first_token < 10.0
        ), f"First token took too long: {time_to_first_token:.2f}s"
        assert (
            tokens_per_second > 0.5
        ), f"Generation too slow: {tokens_per_second:.2f} tokens/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
