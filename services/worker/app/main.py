from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Distributed KV Cache - Worker")

store = {}  # simple in-memory KV, later replace with LMDB or RocksDB


class PutRequest(BaseModel):
    key: str
    value: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/put")
def put_value(req: PutRequest):
    store[req.key] = req.value
    return {"message": "ok"}


@app.get("/get/{key}")
def get_value(key: str):
    return {"key": key, "value": store.get(key)}
