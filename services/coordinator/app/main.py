from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from .hash_ring import ConsistentHashRing

app = FastAPI(title="Distributed KV Cache - Coordinator")

# In-memory cluster state (later we replace with etcd/Redis/GKE metadata)
workers: Dict[str, dict] = {}

# Consistent hashing ring for worker assignment
ring = ConsistentHashRing(virtual_nodes=100)


class RegisterWorker(BaseModel):
    worker_id: str
    address: str  # internal DNS or IP


# ============================================================================
# HEALTH & MONITORING
# ============================================================================


@app.get("/health")
def health():
    return {"status": "ok", "service": "coordinator"}


@app.get("/stats")
def get_coordinator_stats():
    """
    Get coordinator statistics for monitoring.

    Returns:
        - Total workers registered
        - Hash ring configuration
        - Worker list
    """
    return {
        "total_workers": len(workers),
        "workers": workers,
        "hash_ring": {
            "virtual_nodes": ring.virtual_nodes,
            "total_nodes": len(ring.nodes),
        },
    }


@app.post("/register")
def register_worker(req: RegisterWorker):
    workers[req.worker_id] = {"address": req.address}

    # Add worker to consistent hashing ring
    ring.add_node(req.worker_id)

    return {"message": f"Worker {req.worker_id} registered."}


@app.post("/deregister")
def deregister_worker(req: RegisterWorker):
    # same body shape for simplicity
    worker_id = req.worker_id
    if worker_id in workers:
        del workers[worker_id]
        ring.remove_node(worker_id)
        return {"message": f"Worker {worker_id} deregistered."}
    return {"error": "worker not found"}


@app.get("/workers")
def list_workers():
    """Get list of all registered workers."""
    return workers


@app.get("/kv/route/{seq_id}")
def route_kv(seq_id: str):
    """Route KV cache requests by seq_id to ensure all entries for a sequence go to the same worker."""
    worker_id = ring.get_node(seq_id)
    if worker_id is None:
        return {"error": "No workers available"}

    return {
        "worker_id": worker_id,
        "address": workers[worker_id]["address"],
        "seq_id": seq_id,
    }
