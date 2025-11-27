# Distributed KV Cache - Test Suite

Comprehensive test suite for the distributed KV cache system with a focus on routing, caching behavior, and end-to-end generation flow.

## Test Organization

### Active Test Suites

1. **`test_routing_distribution.py`** - Worker routing and sequence distribution

   - Consistent hashing verification
   - Load distribution across workers
   - Routing stability

2. **`test_cache_locality.py`** - KV cache locality and behavior

   - Cache locality verification
   - Cache append behavior
   - Multi-layer caching

3. **`test_generation_flow.py`** - End-to-end generation testing

   - Streaming response verification
   - Model auto-initialization
   - Performance metrics
   - Concurrent generation requests

4. **`test_cache_performance.py`** - Cache vs no-cache performance comparison

   - Single generation speedup analysis
   - Multiple round cache benefits
   - Memory efficiency validation
   - Long sequence cache benefits
   - Time-to-first-token improvements

5. **`test_stress.py`** - Stress testing and system limits
   - High concurrency testing (20+ concurrent requests)
   - Sustained load testing (30s continuous load)
   - Burst traffic patterns
   - Mixed workload scenarios
   - Worker statistics under load

### Supporting Files

- **`tensor_utils.py`** - Tensor serialization/deserialization
- **`test_utils.py`** - Common test utilities
- **`archived/`** - Legacy tests (kept for reference)

## Setup

### 1. Install Dependencies

```bash
cd tests
uv sync
```

### 2. Start Services

Start the services using `scripts/local_dev.sh`.

## Running Tests

### Individual Test Suites

To run a specific test suite, use one of these commands:

```bash
# Run with Python directly
uv run python test_routing_distribution.py
uv run python test_cache_locality.py
uv run python test_generation_flow.py
uv run python test_cache_performance.py
uv run python test_stress.py

# Or run with pytest (recommended for detailed output)
uv run pytest test_routing_distribution.py -v -s
uv run pytest test_cache_performance.py -v -s
uv run pytest test_stress.py -v -s
```

**Note:** Use the `-s` flag to see print statements and detailed metrics in the output.

### Run All Tests

```bash
uv run pytest -v -s
```

### Run Specific Test Categories

```bash
# Performance and cache tests
uv run pytest test_cache_performance.py test_cache_locality.py -v -s

# Stress and load tests
uv run pytest test_stress.py -v -s

# Quick validation (routing + generation)
uv run pytest test_routing_distribution.py test_generation_flow.py -v -s
```

## Test Descriptions

### Performance Tests (`test_cache_performance.py`)

These tests demonstrate the **core value proposition** of the distributed KV cache system:

- **`test_cache_vs_no_cache_single_generation`**: Compares first generation (no cache) vs second generation (with cache) of the same prompt. Shows time-to-first-token and total time improvements.

- **`test_cache_benefit_multiple_rounds`**: Runs multiple generation rounds to show cumulative cache benefits over a conversation-like scenario.

- **`test_cache_memory_efficiency`**: Monitors memory usage and cache statistics to verify the cache doesn't grow unbounded.

- **`test_long_sequence_cache_benefit`**: Tests longer sequences to demonstrate that cache benefits increase with sequence length.

**Expected Results:** Cache should provide 10-50% speedup for typical workloads, with larger improvements for longer sequences.

### Stress Tests (`test_stress.py`)

These tests verify system stability and performance under heavy load:

- **`test_high_concurrency`**: Sends 20+ concurrent requests to test parallel processing capabilities.

- **`test_sustained_load`**: Runs continuous load at 2 req/sec for 30 seconds to verify system stability over time.

- **`test_burst_traffic`**: Alternates between high and low load to test handling of traffic spikes.

- **`test_mixed_workload`**: Uses varied prompt lengths and token counts to simulate realistic usage patterns.

- **`test_worker_stats_under_load`**: Monitors worker statistics during load to verify cache is functioning correctly.

**Expected Results:** System should maintain >80% success rate under all load patterns, with graceful degradation under extreme load.
