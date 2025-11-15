from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Distributed KV Cache - Gateway")


class PutRequest(BaseModel):
    key: str
    value: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/put")
async def put_value(req: PutRequest):
    # TODO: call coordinator for key assignment
    return {"message": f"Received key={req.key}, value={req.value}"}


@app.get("/get/{key}")
async def get_value(key: str):
    # TODO: call coordinator to find worker
    return {"key": key, "value": None}
