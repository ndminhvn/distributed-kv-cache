from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="Distributed KV Cache - Coordinator")

# In-memory cluster state (later we replace with etcd/Redis/GKE metadata)
workers: Dict[str, dict] = {}


class RegisterWorker(BaseModel):
    worker_id: str
    address: str  # internal DNS or IP


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register")
def register_worker(req: RegisterWorker):
    workers[req.worker_id] = {"address": req.address}
    return {"message": f"Worker {req.worker_id} registered."}


@app.get("/workers")
def list_workers():
    return workers


@app.get("/route/{key}")
def route_key(key: str):
    # TODO: consistent hashing to pick worker
    if not workers:
        return {"error": "no workers registered"}

    # Temporary: pick first worker
    worker_id = next(iter(workers.keys()))
    return {"worker_id": worker_id, "address": workers[worker_id]["address"]}
