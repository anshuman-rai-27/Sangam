"""
Central Sangam server — rooms, WebSocket routing, SSE token streaming, static frontend.

Supports two worker modes:
  Browser workers  — connect via WS, run ONNX slices locally, no Python needed
  Python workers   — connect via HTTP /register (existing worker/main.py)

Usage:
    python -m server.main
"""
import asyncio
import base64 as _b64
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tokenizers import Tokenizer

from server.registry import DeviceRegistry
from server.room import RoomManager
from shared.serialization import payload_to_tensor

WEB_DIR = Path(__file__).parent.parent / "web"

# ── Model registry ────────────────────────────────────────────────────────────
_qwen_hf  = os.environ.get("HF_REPO",       "anshumanrai/sangam-qwen-slices")
_qwen_cdn = os.environ.get("ONNX_CDN_BASE",  f"https://huggingface.co/{_qwen_hf}/resolve/main")
_gpt2_hf  = os.environ.get("GPT2_HF_REPO",  "anshumanrai/sangam-gpt2-slices")
_gpt2_cdn = os.environ.get("GPT2_CDN_BASE",  f"https://huggingface.co/{_gpt2_hf}/resolve/main")

MODELS: Dict[str, dict] = {
    "qwen2.5-0.5b": {
        "name":             "Qwen2.5-0.5B-Instruct",
        "description":      "0.5B parameter instruction-following model",
        "hf_repo":          _qwen_hf,
        "onnx_cdn_base":    _qwen_cdn,
        "local_dir":        os.environ.get("MODEL_SLICES_DIR", "model_slices"),
        "tokenizer_id":     "Qwen/Qwen2.5-0.5B-Instruct",
        "eos_tokens":       (151645, 151643),
        "architecture":     "qwen2",
        "slice_assignments": [(0, 8), (8, 16), (16, 24)],
    },
    "gpt2": {
        "name":             "GPT-2 (117M)",
        "description":      "OpenAI GPT-2 base language model",
        "hf_repo":          _gpt2_hf,
        "onnx_cdn_base":    _gpt2_cdn,
        "local_dir":        "model_slices/gpt2",
        "tokenizer_id":     "gpt2",
        "eos_tokens":       (50256,),
        "architecture":     "gpt2",
        "slice_assignments": [(0, 4), (4, 8), (8, 12)],
    },
}
DEFAULT_MODEL = "qwen2.5-0.5b"

registry     = DeviceRegistry()
room_manager = RoomManager()

# Per-model resources loaded at startup
_tokenizers:   Dict[str, Any]  = {}
_server_heads: Dict[str, Any]  = {}

_room_ws:          Dict[str, Set[WebSocket]] = {}
_room_locks:       Dict[str, asyncio.Lock]   = {}
_pending_forwards: Dict[str, asyncio.Future] = {}


# ─── Float32 helpers (browser ↔ server) ──────────────────────────────────────

def _t2b(arr: np.ndarray) -> Tuple[str, list]:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return _b64.b64encode(arr.tobytes()).decode(), list(arr.shape)

def _b2t(data: str, shape: list) -> np.ndarray:
    raw = _b64.b64decode(data)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


# ─── Server heads (embedding + LM-head projection) ───────────────────────────

class _QwenHead:
    """Qwen2.5: RoPE (no wpe), weight-tied lm_head = wte.T, hidden=896."""
    def __init__(self, path: str):
        data = np.load(path)
        self._wte = data["wte"]  # float16, (151936, 896)
        print(f"[server] qwen head: wte={self._wte.shape} {self._wte.dtype}")

    def embed(self, input_ids: list) -> Tuple[str, list]:
        ids    = np.array(input_ids, dtype=np.int32)
        hidden = self._wte[ids].astype(np.float32)  # (n, 896)
        return _t2b(hidden[np.newaxis])              # (1, n, 896)

    def project(self, data: str, shape: list) -> np.ndarray:
        hidden = _b2t(data, shape)
        last   = hidden[0, -1, :].astype(np.float16)
        logits = (last @ self._wte.T).astype(np.float32)
        return logits.reshape(1, 1, -1)


class _GPT2Head:
    """GPT-2: learned absolute positional embeddings (wpe), hidden=768."""
    def __init__(self, path: str):
        data = np.load(path)
        self._wte = data["wte"]  # float16, (50257, 768)
        self._wpe = data["wpe"]  # float16, (1024, 768)
        print(f"[server] gpt2 head: wte={self._wte.shape} wpe={self._wpe.shape}")

    def embed(self, input_ids: list) -> Tuple[str, list]:
        ids    = np.array(input_ids, dtype=np.int32)
        pos    = np.arange(len(ids), dtype=np.int32)
        hidden = (self._wte[ids] + self._wpe[pos]).astype(np.float32)  # (n, 768)
        return _t2b(hidden[np.newaxis])                                  # (1, n, 768)

    def project(self, data: str, shape: list) -> np.ndarray:
        hidden = _b2t(data, shape)
        last   = hidden[0, -1, :].astype(np.float16)
        logits = (last @ self._wte.T).astype(np.float32)
        return logits.reshape(1, 1, -1)


_HEAD_CLASSES = {"qwen2": _QwenHead, "gpt2": _GPT2Head}


# ─── Sampling ─────────────────────────────────────────────────────────────────

def _sample_next(logits_row: np.ndarray, temperature: float, top_p: float) -> int:
    logits_row = logits_row.astype(np.float64) / max(temperature, 1e-6)
    logits_row -= logits_row.max()
    probs = np.exp(logits_row)
    probs /= probs.sum()

    sorted_idx   = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative   = np.cumsum(sorted_probs)

    cutoff = int(np.searchsorted(cumulative - sorted_probs, top_p))
    sorted_probs[cutoff + 1:] = 0.0
    sorted_probs /= sorted_probs.sum()

    chosen = int(np.random.choice(len(sorted_probs), p=sorted_probs))
    return int(sorted_idx[chosen])


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import urllib.request

    for model_key, cfg in MODELS.items():
        # Tokenizer
        try:
            tok = Tokenizer.from_pretrained(cfg["tokenizer_id"])
            _tokenizers[model_key] = tok
            print(f"[server] tokenizer loaded: {cfg['tokenizer_id']}")
        except Exception as e:
            print(f"[server] WARNING: tokenizer failed for {model_key}: {e}")

        # Server head
        local_dir = cfg["local_dir"]
        head_path = os.path.join(local_dir, "server_head.npz")
        if not os.path.isfile(head_path):
            hf_url = f"https://huggingface.co/{cfg['hf_repo']}/resolve/main/server_head.npz"
            print(f"[server] downloading server_head.npz for {model_key} from {hf_url} …")
            os.makedirs(local_dir, exist_ok=True)
            try:
                urllib.request.urlretrieve(hf_url, head_path)
                print(f"[server] downloaded server_head.npz for {model_key}")
            except Exception as e:
                print(f"[server] WARNING: could not download server_head for {model_key}: {e}")
                continue

        arch = cfg["architecture"]
        HeadClass = _HEAD_CLASSES.get(arch)
        if HeadClass:
            _server_heads[model_key] = HeadClass(head_path)

    print("[server] ready at http://0.0.0.0:" + os.environ.get("PORT", "8000"))
    yield


app = FastAPI(lifespan=lifespan)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ─── Broadcast helper ─────────────────────────────────────────────────────────

async def _broadcast(room_id: str, msg: dict) -> None:
    dead: Set[WebSocket] = set()
    for ws in list(_room_ws.get(room_id, [])):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.add(ws)
    if dead:
        _room_ws.get(room_id, set()).difference_update(dead)


# ─── Inference ────────────────────────────────────────────────────────────────

async def _run_inference(room_id: str, text: str, max_new_tokens: int,
                         temperature: float = 0.9, top_p: float = 0.9):
    room = room_manager.get(room_id)
    if room is None:
        yield {"error": "room not found"}
        return

    pipeline = room.get_pipeline()
    if pipeline is None:
        yield {"error": "pipeline not ready — all 3 devices must be connected"}
        return

    model_key = room.model
    cfg       = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    tok       = _tokenizers.get(model_key)
    head      = _server_heads.get(model_key)
    eos       = cfg["eos_tokens"]

    if tok is None:
        yield {"error": f"tokenizer not loaded for model '{model_key}'"}
        return

    # Apply prompt template
    if cfg["architecture"] == "qwen2":
        wrapped = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        wrapped = text  # GPT-2 base: no chat template

    input_ids = tok.encode(wrapped).ids
    generated = list(input_ids)

    all_browser = all(d.ws is not None for d in pipeline)

    if all_browser and head is None:
        yield {"error": f"server_head not loaded for model '{model_key}' — restart server"}
        return

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for _ in range(max_new_tokens):

            if all_browser:
                try:
                    b64, shape = head.embed(generated)
                except Exception as exc:
                    yield {"error": f"embed failed: {exc}"}
                    return

                for device in pipeline:
                    rid  = uuid.uuid4().hex
                    loop = asyncio.get_event_loop()
                    fut  = loop.create_future()
                    _pending_forwards[rid] = fut
                    try:
                        await device.ws.send_text(json.dumps({
                            "type":       "forward_request",
                            "request_id": rid,
                            "slice_id":   device.slice_id,
                            "data":       b64,
                            "shape":      shape,
                        }))
                        result = await asyncio.wait_for(asyncio.shield(fut), timeout=30.0)
                        b64, shape = result["data"], result["shape"]
                    except asyncio.TimeoutError:
                        room.mark_dropped(device.device_id)
                        await _broadcast(room_id, {"type": "device_dropped",
                                                    "device_id": device.device_id,
                                                    **room.to_dict()})
                        yield {"error": f"device '{device.device_id}' timed out"}
                        return
                    except Exception as exc:
                        yield {"error": str(exc)}
                        return
                    finally:
                        _pending_forwards.pop(rid, None)

                logits = head.project(b64, shape)

            else:
                hidden_payload = None
                logits_payload = None

                for device in pipeline:
                    body = ({"input_ids": generated} if device.is_first
                            else {"hidden": hidden_payload})
                    try:
                        r = await http_client.post(f"{device.url}/forward", json=body)
                        r.raise_for_status()
                    except Exception:
                        room.mark_dropped(device.device_id)
                        await _broadcast(room_id, {"type": "device_dropped",
                                                    "device_id": device.device_id,
                                                    **room.to_dict()})
                        yield {"error": f"device '{device.device_id}' unreachable"}
                        return
                    data = r.json()
                    if device.is_last:
                        logits_payload = data["logits"]
                    else:
                        hidden_payload = data["hidden"]

                logits = payload_to_tensor(logits_payload).numpy()

            next_token = _sample_next(logits[0, -1, :], temperature, top_p)
            generated.append(next_token)
            decoded = tok.decode([next_token])
            yield {"token": decoded}

            if next_token in eos:
                break

    yield {"done": True, "full": tok.decode(generated)}


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    device_id:   str
    url:         str
    layer_start: int
    layer_end:   int
    is_first:    bool
    is_last:     bool
    room_id:     Optional[str] = None

class InferRequest(BaseModel):
    text:           str
    max_new_tokens: int = 20

class JoinRequest(BaseModel):
    device_id: str
    ram_mb:    int = 2048

class CreateRoomRequest(BaseModel):
    model: Optional[str] = None


# ─── Static pages ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    p = WEB_DIR / "index.html"
    if not p.is_file():
        return {"status": "Sangam server running", "docs": "/docs"}
    return FileResponse(str(p))

@app.get("/room/{room_id}")
def room_page(room_id: str):
    p = WEB_DIR / "room.html"
    if not p.is_file():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(str(p))


# ─── Room API ─────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models():
    return [
        {"id": k, "name": v["name"], "description": v["description"]}
        for k, v in MODELS.items()
    ]

@app.post("/room/create")
def room_create(req: CreateRoomRequest = CreateRoomRequest()):
    model_key = req.model or DEFAULT_MODEL
    if model_key not in MODELS:
        raise HTTPException(400, f"Unknown model '{model_key}'. Available: {list(MODELS)}")
    cfg  = MODELS[model_key]
    room = room_manager.create(
        model=model_key,
        model_name=cfg["name"],
        slice_assignments=cfg["slice_assignments"],
    )
    return {"room_id": room.room_id, "model": model_key, "model_name": cfg["name"]}

@app.post("/room/{room_id}/join")
def room_join(room_id: str, req: JoinRequest):
    room = room_manager.get_or_create(room_id)
    dev  = room.join(req.device_id, req.ram_mb)
    if dev is None:
        raise HTTPException(409, "Room full — all 3 slice slots are taken")

    cfg        = MODELS.get(room.model, MODELS[DEFAULT_MODEL])
    local_dir  = cfg["local_dir"]
    local_path = Path(local_dir) / f"slice_{dev.slice_id}.onnx"
    if local_path.is_file():
        onnx_url = f"/onnx/{room.model}/{dev.slice_id}"
    else:
        onnx_url = f"{cfg['onnx_cdn_base']}/slice_{dev.slice_id}.onnx"

    return {
        "device_id": dev.device_id,
        "slice_id":  dev.slice_id,
        "layers":    [dev.layer_start, dev.layer_end],
        "onnx_url":  onnx_url,
    }

@app.get("/room/{room_id}/status")
def room_status(room_id: str):
    room = room_manager.get(room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    return room.to_dict()


# ─── Slice serving ────────────────────────────────────────────────────────────

@app.get("/onnx/{model_key}/{slice_id}")
def get_slice_onnx(model_key: str, slice_id: int):
    if model_key not in MODELS:
        raise HTTPException(404, f"Unknown model '{model_key}'")
    local_dir = MODELS[model_key]["local_dir"]
    path = Path(local_dir) / f"slice_{slice_id}.onnx"
    if not path.is_file():
        raise HTTPException(404, f"ONNX slice {slice_id} not found for {model_key}")
    return FileResponse(
        str(path), media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=slice_{slice_id}.onnx"},
    )

# Legacy single-model endpoint (backward compat)
@app.get("/onnx/{slice_id}")
def get_slice_onnx_legacy(slice_id: int):
    local_dir = MODELS[DEFAULT_MODEL]["local_dir"]
    path = Path(local_dir) / f"slice_{slice_id}.onnx"
    if not path.is_file():
        raise HTTPException(404, f"ONNX slice {slice_id} not found")
    return FileResponse(
        str(path), media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=slice_{slice_id}.onnx"},
    )


# ─── WebSocket: live room state + browser worker routing ─────────────────────

@app.websocket("/ws/room/{room_id}")
async def ws_room(websocket: WebSocket, room_id: str):
    await websocket.accept()
    room = room_manager.get_or_create(room_id)
    _room_ws.setdefault(room_id, set()).add(websocket)
    await websocket.send_text(json.dumps({"type": "room_state", **room.to_dict()}))

    my_device_id: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            t   = msg.get("type")

            if t == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            elif t == "worker_ready":
                device_id    = msg.get("device_id", "")
                ram_mb       = msg.get("ram_mb", 2048)
                my_device_id = device_id

                dev = room.get_device(device_id)
                if dev is None:
                    dev = room.join(device_id, ram_mb)

                if dev is not None:
                    dev.status = "ready"
                    dev.ws     = websocket
                    state = room.to_dict()
                    await _broadcast(room_id, {"type": "device_ready", **state})
                    if room.pipeline_ready:
                        await _broadcast(room_id, {"type": "pipeline_ready", **state})

            elif t == "forward_result":
                rid = msg.get("request_id", "")
                if rid in _pending_forwards:
                    fut = _pending_forwards[rid]
                    if not fut.done():
                        fut.set_result(msg)

            elif t == "forward_error":
                rid = msg.get("request_id", "")
                if rid in _pending_forwards:
                    fut = _pending_forwards[rid]
                    if not fut.done():
                        fut.set_exception(RuntimeError(msg.get("error", "forward error")))

    except WebSocketDisconnect:
        _room_ws.get(room_id, set()).discard(websocket)
        if my_device_id:
            room.mark_dropped(my_device_id)
            await _broadcast(room_id, {"type": "device_dropped",
                                        "device_id": my_device_id,
                                        **room.to_dict()})


# ─── SSE: token streaming ─────────────────────────────────────────────────────

@app.get("/room/{room_id}/stream")
async def stream(room_id: str, prompt: str, max_new_tokens: int = 80,
                 temperature: float = 0.9, top_p: float = 0.9):
    _room_locks.setdefault(room_id, asyncio.Lock())
    lock = _room_locks[room_id]

    async def event_gen():
        async with lock:
            async for payload in _run_inference(room_id, prompt, max_new_tokens,
                                                temperature, top_p):
                yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Legacy endpoints (Python worker compatibility) ───────────────────────────

@app.post("/register")
async def register(req: RegisterRequest):
    result = registry.register(req.model_dump(exclude={"room_id"}))

    if req.room_id:
        room = room_manager.get(req.room_id)
        if room:
            room.mark_ready(req.device_id, req.url)
            state = room.to_dict()
            await _broadcast(req.room_id, {"type": "device_ready", **state})
            if room.pipeline_ready:
                await _broadcast(req.room_id, {"type": "pipeline_ready", **state})

    print(f"[server] registered: {result}")
    return result


@app.get("/status")
def status():
    pipeline = registry.get_pipeline()
    return {
        "devices":        registry.all_devices(),
        "pipeline_ready": pipeline is not None,
        "pipeline_order": (
            [{"device_id": d.device_id, "layers": [d.layer_start, d.layer_end]}
             for d in pipeline] if pipeline else []
        ),
    }


@app.post("/infer")
async def infer(req: InferRequest):
    pipeline = registry.get_pipeline()
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready — check /status")

    tok = _tokenizers.get(DEFAULT_MODEL)
    input_ids = tok.encode(req.text).ids
    generated = list(input_ids)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for _ in range(req.max_new_tokens):
            hidden_payload = None
            logits_payload = None
            for device in pipeline:
                body = {"input_ids": generated} if device.is_first else {"hidden": hidden_payload}
                try:
                    r = await client.post(f"{device.url}/forward", json=body)
                    r.raise_for_status()
                except Exception as exc:
                    registry.mark_unavailable(device.device_id)
                    raise HTTPException(503, f"Device '{device.device_id}' unreachable: {exc}")
                data = r.json()
                if device.is_last:
                    logits_payload = data["logits"]
                else:
                    hidden_payload = data["hidden"]

            logits     = payload_to_tensor(logits_payload).numpy()
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            if next_token in MODELS[DEFAULT_MODEL]["eos_tokens"]:
                break

    return {"input": req.text, "output": tok.decode(generated)}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0").strip()
    port = int(os.environ.get("PORT", "8000").strip())
    uvicorn.run(app, host=host, port=port, reload=False)
