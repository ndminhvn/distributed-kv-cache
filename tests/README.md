<!-- README for distributed KV cache tests -->

# Distributed KV Cache Tests

To run the tests for the distributed key-value cache system, follow the instructions below.

## Setup

1. Navigate to the `tests` directory:
   ```bash
   cd tests
   ```
2. Install the required dependencies using `uv`:
   ```bash
   uv sync
   ```

## Running Tests

To execute the test suite, run the following command (replace `<file_name>` with the actual test file name, e.g., `test_kv_cache`):

```bash
uv run python <file_name>.py
```

Alternatively, you can run the tests using `pytest` for directly using `pytest`:
**Note: Use tag `-s` to see print statements in the output. If you don't include `-s`, print statements will be suppressed and only test results will be shown.**

```bash
uv run pytest <file_name>.py -v -s
```
