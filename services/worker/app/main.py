import os
import socket
import asyncio
import httpx
import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Any, Dict
from .kv_cache import KVCache
from .tensor_utils import serialize_tensor, deserialize_tensor
from .model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# configuration via env vars
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
WORKER_PORT = int(os.environ.get("WORKER_PORT", "8082"))
COORDINATOR_ADDR = os.environ.get("COORDINATOR_ADDR", "http://coordinator:8081")

kv_cache = KVCache(max_entries=int(os.environ.get("KV_MAX_ENTRIES", "50000")))
model_loader = (
    ModelLoader()
)  # Lazy loading - model loaded on first /inference/init call


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


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


class InferenceInitRequest(BaseModel):
    model_name: str = "gpt2"  # Default to GPT-2 for testing


class GenerateRequest(BaseModel):
    seq_id: str
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    model_name: str = "gpt2"  # Model to use for generation


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


# ============================================================================
# HEALTH & MONITORING
# ============================================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "worker_id": WORKER_ID}


@app.get("/stats")
async def get_worker_stats():
    """
    Get comprehensive worker statistics for monitoring.

    Returns:
        - KV cache statistics
        - Model information (if loaded)
        - Worker metadata
    """
    stats = {
        "worker_id": WORKER_ID,
        "kv_cache": kv_cache.stats(),
    }

    # Add model info if loaded
    if model_loader.is_loaded():
        stats["model"] = {
            "loaded": True,
            "info": model_loader._get_model_info(),
            "memory": model_loader.get_memory_usage(),
        }
    else:
        stats["model"] = {"loaded": False}

    return stats


# ============================================================================
# KV CACHE MANAGEMENT (Testing & Debugging Endpoints)
# ============================================================================
# Note: These endpoints are for testing/debugging only.
# In production, /generate handles KV cache internally.


@app.post("/kv/get")
async def get_kv(req: KVGetRequest):
    """[Testing] Retrieve KV cache entry for a specific sequence and layer."""
    out = kv_cache.get(req.seq_id, req.layer)
    if out is None:
        raise HTTPException(status_code=404, detail="KV entry not found")

    # Serialize tensors for transport
    return {
        "seq_id": req.seq_id,
        "layer": req.layer,
        "seq_len": out["k"].shape[0],
        "k": serialize_tensor(out["k"]),
        "v": serialize_tensor(out["v"]),
        "worker_id": WORKER_ID,
    }


@app.post("/kv/put")
async def put_kv(req: KVPutRequest):
    """[Testing] Store KV cache entry for a specific sequence and layer."""
    # Deserialize tensors from request
    k_tensor = deserialize_tensor(req.k)
    v_tensor = deserialize_tensor(req.v)

    kv_cache.put(req.seq_id, req.layer, k_tensor, v_tensor)

    return {
        "status": "ok",
        "seq_id": req.seq_id,
        "layer": req.layer,
        "seq_len": k_tensor.shape[0],
        "tensor_shapes": {"k": list(k_tensor.shape), "v": list(v_tensor.shape)},
        "tensor_dtypes": {"k": str(k_tensor.dtype), "v": str(v_tensor.dtype)},
        "worker_id": WORKER_ID,
    }


@app.post("/kv/append")
async def append_kv(req: KVAppendRequest):
    """[Testing] Append a new token's KV to existing cache."""
    # Deserialize tensors from request
    k_tensor = deserialize_tensor(req.k)
    v_tensor = deserialize_tensor(req.v)

    kv_cache.append(req.seq_id, req.layer, k_tensor, v_tensor)

    # Get updated sequence length
    new_seq_len = kv_cache.get_seq_len(req.seq_id, req.layer)

    return {
        "status": "ok",
        "seq_id": req.seq_id,
        "layer": req.layer,
        "seq_len": new_seq_len,
        "tensor_shapes": {"k": list(k_tensor.shape), "v": list(v_tensor.shape)},
        "tensor_dtypes": {"k": str(k_tensor.dtype), "v": str(v_tensor.dtype)},
        "worker_id": WORKER_ID,
    }


@app.delete("/kv/{seq_id}")
async def evict_seq(seq_id: str):
    """[Operational] Evict all KV cache entries for a sequence (cleanup/memory management)."""
    kv_cache.evict_sequence(seq_id)
    return {"status": "evicted", "seq_id": seq_id, "worker_id": WORKER_ID}


@app.get("/kv/{seq_id}/{layer}/length")
async def get_sequence_length(seq_id: str, layer: int):
    """[Operational] Get the current sequence length for debugging/monitoring."""
    seq_len = kv_cache.get_seq_len(seq_id, layer)
    return {
        "seq_id": seq_id,
        "layer": layer,
        "seq_len": seq_len,
        "worker_id": WORKER_ID,
    }


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================


@app.post("/inference/init")
async def inference_init(req: InferenceInitRequest):
    """
    Initialize model for inference.

    This loads the LLM model into GPU memory. Should be called once before
    generating tokens. Supports any HuggingFace AutoModelForCausalLM.

    Examples:
    - "gpt2" (124M parameters, good for testing)
    - "gpt2-medium" (355M)
    - "gpt2-large" (774M)
    - "meta-llama/Llama-2-7b-hf" (requires auth token)
    """
    try:
        model_info = model_loader.load_model(req.model_name)
        memory_info = model_loader.get_memory_usage()

        return {
            "status": "initialized",
            "worker_id": WORKER_ID,
            "model_info": model_info,
            "memory_usage": memory_info,
        }

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise HTTPException(
            status_code=500, detail=f"Model initialization failed: {str(e)}"
        )


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate text auto-regressively using the loaded model.

    This endpoint handles the full generation loop internally:
    1. Auto-initializes model if not already loaded
    2. Tokenizes the initial prompt
    3. Runs forward passes through the model
    4. Samples next tokens (greedy or with temperature/top_p)
    5. Appends KV cache automatically for each layer
    6. Streams tokens back to client as they're generated

    Returns a streaming response with Server-Sent Events format.
    Each event contains a JSON object with the generated token.
    """
    # Auto-initialize model if not loaded
    if not model_loader.is_loaded():
        try:
            logger.info(f"Auto-initializing model {req.model_name} for generation")
            model_loader.load_model(req.model_name)
        except Exception as e:
            logger.error(f"Failed to auto-initialize model: {e}")
            raise HTTPException(
                status_code=500, detail=f"Model initialization failed: {str(e)}"
            )

    async def generate_stream():
        try:
            import torch
            import torch.nn.functional as F

            # Tokenize initial prompt
            input_ids = model_loader.tokenize(req.prompt)
            generated_tokens = []
            eos_token_id = model_loader.tokenizer.eos_token_id

            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'seq_id': req.seq_id, 'prompt': req.prompt})}\n\n"

            # Generation loop
            for step in range(req.max_tokens):
                # Build past_key_values from KV cache if we have prior tokens
                past_key_values = None
                if step > 0:
                    # Retrieve cached KV from previous steps
                    num_layers = model_loader.model.config.num_hidden_layers
                    past_key_values = []

                    for layer_idx in range(num_layers):
                        cached = kv_cache.get(req.seq_id, layer_idx)
                        if cached is None:
                            # No cache yet, break and use full sequence
                            past_key_values = None
                            break

                        # cached is a dict with "k" and "v" keys
                        k_cached = cached["k"]
                        v_cached = cached["v"]
                        # Convert from our format [seq_len, heads, dim] to HuggingFace format
                        # [batch, heads, seq_len, dim]
                        k_hf = k_cached.transpose(0, 1).unsqueeze(
                            0
                        )  # [1, heads, seq_len, dim]
                        v_hf = v_cached.transpose(0, 1).unsqueeze(0)
                        past_key_values.append((k_hf, v_hf))

                    if past_key_values:
                        past_key_values = tuple(past_key_values)

                # Forward pass
                if past_key_values is not None and step > 0:
                    # Use only the last token as input when we have cached KV
                    current_input = input_ids[:, -1:]
                else:
                    # First step or no cache: use full sequence
                    current_input = input_ids

                logits, new_past_key_values = model_loader.generate_step(
                    current_input, past_key_values=past_key_values
                )

                # Sample next token
                next_token_logits = logits[0, -1, :] / req.temperature

                # Apply top-p (nucleus) sampling if needed
                if req.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > req.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample from the distribution
                if req.temperature > 0:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token_id = torch.argmax(next_token_logits).unsqueeze(0)

                # Decode token
                next_token_text = model_loader.decode(next_token_id.unsqueeze(0))
                generated_tokens.append(next_token_text)

                # Store/append KV cache for each layer
                if new_past_key_values is not None:
                    num_layers = len(new_past_key_values)
                    for layer_idx in range(num_layers):
                        layer_kv = new_past_key_values[layer_idx]
                        k_tensor = layer_kv[0]  # [batch, num_heads, seq_len, head_dim]
                        v_tensor = layer_kv[1]

                        # Convert from HuggingFace format to our format
                        k_converted = (
                            k_tensor.squeeze(0).transpose(0, 1).contiguous()
                        )  # [seq_len, heads, dim]
                        v_converted = v_tensor.squeeze(0).transpose(0, 1).contiguous()

                        if step == 0:
                            # First token: store full KV
                            kv_cache.put(
                                req.seq_id, layer_idx, k_converted, v_converted
                            )
                        else:
                            # Subsequent tokens: append to existing KV
                            # Extract only the new token's KV (last position in seq_len dimension)
                            k_new = k_converted[-1:, :, :]  # [1, heads, dim]
                            v_new = v_converted[-1:, :, :]
                            kv_cache.append(req.seq_id, layer_idx, k_new, v_new)

                # Stream the token back to client
                yield f"data: {json.dumps({'type': 'token', 'token': next_token_text, 'token_id': int(next_token_id)})}\n\n"

                # Append to input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

                # Check for EOS
                if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break

            # Send completion metadata
            generated_text = req.prompt + "".join(generated_tokens)
            yield f"data: {json.dumps({'type': 'done', 'generated_text': generated_text, 'tokens_generated': len(generated_tokens)})}\n\n"

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/inference/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    if not model_loader.is_loaded():
        return {
            "loaded": False,
            "worker_id": WORKER_ID,
        }

    model_info = model_loader._get_model_info()
    memory_info = model_loader.get_memory_usage()

    return {
        "loaded": True,
        "worker_id": WORKER_ID,
        "model_info": model_info,
        "memory_usage": memory_info,
    }


@app.delete("/inference/model")
async def unload_model():
    """Unload the currently loaded model to free GPU memory."""
    if not model_loader.is_loaded():
        return {
            "status": "no_model_loaded",
            "worker_id": WORKER_ID,
        }

    model_name = model_loader.model_name
    model_loader.unload_model()

    return {
        "status": "unloaded",
        "model_name": model_name,
        "worker_id": WORKER_ID,
    }
