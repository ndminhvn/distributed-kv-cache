import os
import socket
import asyncio
import httpx
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from .lru_cache import LRUCache
from .kv_cache import KVCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configuration via env vars
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8082"))
COORDINATOR_ADDR = os.environ.get("COORDINATOR_ADDR", "http://coordinator:8081")

cache = LRUCache(capacity=int(os.environ.get("LRU_CAPACITY", "1000")))
kv_cache = KVCache(max_entries=int(os.environ.get("KV_MAX_ENTRIES", "50000")))


class PutRequest(BaseModel):
    key: str
    value: str


class KVPutRequest(BaseModel):
    seq_id: str
    layer: int
    step: int
    k: list
    v: list


class KVGetRequest(BaseModel):
    seq_id: str
    layer: int
    step: int


# @app.on_event("startup")
async def startup_event():
    # register with coordinator
    address = f"http://{WORKER_ID}:{WORKER_PORT}"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{COORDINATOR_ADDR}/register",
                json={"worker_id": WORKER_ID, "address": address},
            )
            logger.info(f"[startup] Registered worker {WORKER_ID} -> {address}")
        except Exception as e:
            logger.error(f"[startup] Failed to register worker: {e}")


# @app.on_event("shutdown")
async def shutdown_event():
    address = f"http://{WORKER_ID}:{WORKER_PORT}"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{COORDINATOR_ADDR}/deregister",
                json={"worker_id": WORKER_ID, "address": address},
            )
            logger.info(f"[shutdown] Deregistered worker {WORKER_ID}")
        except Exception as e:
            logger.error(f"[shutdown] Failed to deregister worker: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(title="Distributed KV Cache - Worker", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "worker_id": WORKER_ID}


@app.get("/get/{key}")
def get_value(key: str):
    value = cache.get(key)
    return {"key": key, "value": value, "hit": value is not None}


@app.post("/put")
def put_value(req: PutRequest):
    cache.put(req.key, req.value)
    return {"key": req.key, "worker_id": WORKER_ID}


@app.post("/kv/get")
async def get_kv(req: KVGetRequest):
    out = kv_cache.get(req.seq_id, req.layer, req.step)
    if out is None:
        raise HTTPException(status_code=404, detail="KV entry not found")
    return {
        "seq_id": req.seq_id,
        "layer": req.layer,
        "step": req.step,
        "k": out["k"],
        "v": out["v"],
        "worker_id": WORKER_ID,
    }


@app.post("/kv/put")
async def put_kv(req: KVPutRequest):
    kv_cache.put(req.seq_id, req.layer, req.step, req.k, req.v)
    return {
        "status": "ok",
        "seq_id": req.seq_id,
        "layer": req.layer,
        "step": req.step,
        "tensor_shapes": {"k": len(req.k), "v": len(req.v)},
        "worker_id": WORKER_ID,
    }


@app.delete("/kv/{seq_id}")
async def evict_seq(seq_id: str):
    kv_cache.evict_sequence(seq_id)
    return {"status": "evicted", "seq_id": seq_id, "worker_id": WORKER_ID}


@app.get("/kv/stats")
async def stats():
    return kv_cache.stats()
