import os
import httpx
import logging
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Distributed KV Cache - Gateway")
COORDINATOR_ADDR = os.environ.get("COORDINATOR_ADDR", "http://coordinator:8081")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    model_name: str = "gpt2"


class KVPutRequest(BaseModel):
    seq_id: str
    layer: int
    k: Dict[str, Any]  # Serialized tensor dict
    v: Dict[str, Any]  # Serialized tensor dict


class KVAppendRequest(BaseModel):
    seq_id: str
    layer: int
    k: Dict[str, Any]  # Serialized tensor dict for new token
    v: Dict[str, Any]  # Serialized tensor dict for new token


class KVGetRequest(BaseModel):
    seq_id: str
    layer: int


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def get_worker_for_seq(seq_id: str) -> str:
    """
    Route a sequence ID to a worker address using consistent hashing.

    This ensures all KV cache entries for a sequence go to the same worker,
    enabling efficient cache locality.

    Args:
        seq_id: Unique sequence identifier

    Returns:
        Worker address (e.g., "http://worker-1:8082")

    Raises:
        HTTPException: If routing fails
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{COORDINATOR_ADDR}/kv/route/{seq_id}", timeout=5.0)
        route = r.json()

        if r.status_code != 200 or "address" not in route:
            raise HTTPException(
                status_code=500, detail=f"Failed to route seq_id {seq_id}"
            )

        return route["address"]


# ============================================================================
# HEALTH & MONITORING
# ============================================================================
# Note: /kv/* testing endpoints are defined at the bottom of this file


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "gateway"}


@app.get("/stats")
async def get_stats():
    """
    Aggregate statistics from all workers for monitoring and reporting.

    Returns:
        - Worker health status
        - KV cache statistics from each worker
        - Model information (if loaded)
        - Total cluster capacity
    """
    async with httpx.AsyncClient() as client:
        try:
            # Get list of workers
            workers_resp = await client.get(f"{COORDINATOR_ADDR}/workers", timeout=5.0)
            workers = workers_resp.json()

            if not workers:
                return {"workers": [], "total_workers": 0, "cluster_healthy": False}

            # Gather stats from all workers
            worker_stats = []
            for worker_id, worker_info in workers.items():
                worker_addr = worker_info["address"]

                try:
                    # Get worker stats (includes health, KV cache, and model info)
                    stats_resp = await client.get(f"{worker_addr}/stats", timeout=2.0)

                    if stats_resp.status_code == 200:
                        worker_data = stats_resp.json()
                        worker_stats.append(
                            {
                                "worker_id": worker_id,
                                "address": worker_addr,
                                **worker_data,  # Include all stats from worker
                            }
                        )
                    else:
                        worker_stats.append(
                            {
                                "worker_id": worker_id,
                                "address": worker_addr,
                                "status": "error",
                                "error": f"HTTP {stats_resp.status_code}",
                            }
                        )

                except Exception as e:
                    logger.error(f"Failed to get stats from {worker_id}: {e}")
                    worker_stats.append(
                        {
                            "worker_id": worker_id,
                            "address": worker_addr,
                            "health": {"status": "unreachable"},
                            "error": str(e),
                        }
                    )

            # Calculate aggregate metrics
            total_sequences = sum(
                w.get("kv_cache", {}).get("num_sequences", 0) for w in worker_stats
            )
            total_entries = sum(
                w.get("kv_cache", {}).get("total_entries", 0) for w in worker_stats
            )
            healthy_workers = sum(
                1 for w in worker_stats if w.get("worker_id") and "error" not in w
            )

            return {
                "workers": worker_stats,
                "total_workers": len(workers),
                "healthy_workers": healthy_workers,
                "cluster_healthy": healthy_workers == len(workers),
                "aggregate": {
                    "total_sequences": total_sequences,
                    "total_kv_entries": total_entries,
                },
            }

        except Exception as e:
            logger.error(f"Failed to gather cluster stats: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to gather stats: {str(e)}"
            )


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate text using the distributed inference KV cache system.

    This endpoint:
    1. Generates a unique seq_id for this generation request
    2. Routes to a worker based on the seq_id (for KV cache locality)
    3. Worker handles model initialization if needed, manages KV cache, and generates tokens
    4. Streams tokens back to the client as they're generated

    The seq_id ensures that all tokens for this sequence are cached
    on the same worker, enabling efficient KV cache reuse.
    """
    seq_id = str(uuid.uuid4())

    try:
        worker_addr = await get_worker_for_seq(seq_id)
        logger.info(f"Routing generation request seq={seq_id} to {worker_addr}")

        # Stream response from worker
        async def stream_from_worker():
            async with httpx.AsyncClient(timeout=300.0) as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{worker_addr}/generate",
                    json={
                        "seq_id": seq_id,
                        "prompt": req.prompt,
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                        "top_p": req.top_p,
                        "model_name": req.model_name,
                    },
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_from_worker(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================================
# TESTING ENDPOINTS (For test suites only)
# ============================================================================


@app.post("/kv/get")
async def get_kv(req: KVGetRequest):
    """[Testing] Retrieve KV cache entry. Routes by seq_id to the worker storing this sequence."""
    worker_addr = await get_worker_for_seq(req.seq_id)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{worker_addr}/kv/get",
            json={"seq_id": req.seq_id, "layer": req.layer},
            timeout=5.0,
        )

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="KV entry not found")

        return resp.json()


@app.post("/kv/put")
async def put_kv(req: KVPutRequest):
    """[Testing] Store KV cache entry. Routes by seq_id to ensure all entries for a sequence go to same worker."""
    worker_addr = await get_worker_for_seq(req.seq_id)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{worker_addr}/kv/put",
            json={
                "seq_id": req.seq_id,
                "layer": req.layer,
                "k": req.k,
                "v": req.v,
            },
            timeout=5.0,
        )
        return resp.json()


@app.post("/kv/append")
async def append_kv(req: KVAppendRequest):
    """[Testing] Append new token's KV to existing cache. Routes by seq_id."""
    worker_addr = await get_worker_for_seq(req.seq_id)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{worker_addr}/kv/append",
            json={
                "seq_id": req.seq_id,
                "layer": req.layer,
                "k": req.k,
                "v": req.v,
            },
            timeout=5.0,
        )
        return resp.json()
