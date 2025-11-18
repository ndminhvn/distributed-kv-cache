import os
import httpx
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Distributed KV Cache - Gateway")
COORDINATOR_URL = os.environ.get("COORDINATOR_ADDR", "http://coordinator:8081")


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/get/{key}")
async def get_value(key: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{COORDINATOR_URL}/route/{key}", timeout=5.0)
        route = r.json()

        if r.status_code != 200 or "address" not in route:
            raise HTTPException(status_code=500, detail="No route for key")

        worker_addr = route["address"]

        resp = await client.get(f"{worker_addr}/get/{key}", timeout=5.0)
        return resp.json()


@app.post("/put")
async def put_value(req: PutRequest):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{COORDINATOR_URL}/route/{req.key}", timeout=5.0)
        route = r.json()

        if r.status_code != 200 or "address" not in route:
            raise HTTPException(status_code=500, detail="No route for key")

        worker_addr = route["address"]

        # To test distributed routing
        logger.debug(f"Routing PUT {req.key} to {worker_addr}")

        resp = await client.post(
            f"{worker_addr}/put", json={"key": req.key, "value": req.value}, timeout=5.0
        )
        return resp.json()


@app.post("/kv/get")
async def get_kv(req: KVGetRequest):
    """Retrieve KV cache entry. Routes by seq_id to the worker storing this sequence."""
    async with httpx.AsyncClient() as client:
        # Route by seq_id
        r = await client.get(f"{COORDINATOR_URL}/kv/route/{req.seq_id}", timeout=5.0)
        route = r.json()

        if r.status_code != 200 or "address" not in route:
            raise HTTPException(status_code=500, detail="No route for seq_id")

        worker_addr = route["address"]
        logger.info(
            f"Routing KV GET seq={req.seq_id} layer={req.layer} step={req.step} to {worker_addr}"
        )

        resp = await client.post(
            f"{worker_addr}/kv/get",
            json={"seq_id": req.seq_id, "layer": req.layer, "step": req.step},
            timeout=5.0,
        )

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="KV entry not found")

        return resp.json()


@app.post("/kv/put")
async def put_kv(req: KVPutRequest):
    """Store KV cache entry. Routes by seq_id to ensure all entries for a sequence go to same worker."""
    async with httpx.AsyncClient() as client:
        # Route by seq_id
        r = await client.get(f"{COORDINATOR_URL}/kv/route/{req.seq_id}", timeout=5.0)
        route = r.json()

        if r.status_code != 200 or "address" not in route:
            raise HTTPException(status_code=500, detail="No route for seq_id")

        worker_addr = route["address"]
        logger.info(
            f"Routing KV PUT seq={req.seq_id} layer={req.layer} step={req.step} to {worker_addr}"
        )

        resp = await client.post(
            f"{worker_addr}/kv/put",
            json={
                "seq_id": req.seq_id,
                "layer": req.layer,
                "step": req.step,
                "k": req.k,
                "v": req.v,
            },
            timeout=5.0,
        )
        return resp.json()
