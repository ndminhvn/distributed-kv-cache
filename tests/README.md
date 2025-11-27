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

To run the tests, use the following command, replacing `<file_name>` with the desired test file (e.g., `test_routing_distribution.py`):

```bash
uv run python <file_name>.py
```

Alternatively, you can run the tests using `pytest` for directly using `pytest`:
**Note: Use tag `-s` to see print statements in the output. If you don't include `-s`, print statements will be suppressed and only test results will be shown.**

```bash
uv run pytest <file_name>.py -v -s
```
