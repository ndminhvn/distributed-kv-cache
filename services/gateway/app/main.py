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
