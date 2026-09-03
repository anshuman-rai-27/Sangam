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
from typing import Dict, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import httpx
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import GPT2Tokenizer

from server.registry import DeviceRegistry
from server.room import RoomManager
from shared.serialization import payload_to_tensor

MODEL_SLICES_DIR = os.environ.get("MODEL_SLICES_DIR", "model_slices").strip()
TOKENIZER_PATH   = os.environ.get("TOKENIZER_PATH",   "model_slices/tokenizer").strip()
WEB_DIR          = Path(__file__).parent.parent / "web"

registry     = DeviceRegistry()
room_manager = RoomManager()
tokenizer    = None
server_head  = None   # loaded from server_head.pt if available

_room_ws:        Dict[str, Set[WebSocket]]  = {}
_room_locks:     Dict[str, asyncio.Lock]    = {}
_pending_forwards: Dict[str, asyncio.Future] = {}


# ─── Float32 helpers (browser ↔ server) ─────────────────────────────────────

def _t2b(t: torch.Tensor) -> Tuple[str, list]:
    """Tensor → (base64-float32, shape-list)."""
    arr = t.detach().cpu().float().numpy()
    return _b64.b64encode(arr.tobytes()).decode(), list(arr.shape)


def _b2t(data: str, shape: list) -> torch.Tensor:
    """base64-float32 + shape → Tensor."""
    raw = _b64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


# ─── Server head (embedding + LM-head projection) ────────────────────────────

class _ServerHead:
    """Holds wte, wpe, lm_head — server does embedding and projection."""

    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._wte = nn.Embedding(50257, 768)
        self._wte.load_state_dict(data["wte"])
        self._wpe = nn.Embedding(1024, 768)
        self._wpe.load_state_dict(data["wpe"])
        # LM head shares weights with wte (GPT-2 weight tying)
        self._lm_head = nn.Linear(768, 50257, bias=False)
        self._lm_head.weight = self._wte.weight
        for m in [self._wte, self._wpe, self._lm_head]:
            m.eval()

    @torch.no_grad()
    def embed(self, input_ids: list) -> Tuple[str, list]:
        ids = torch.tensor([input_ids], dtype=torch.long)
        pos = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
        return _t2b(self._wte(ids) + self._wpe(pos))

    @torch.no_grad()
    def project(self, data: str, shape: list) -> torch.Tensor:
        return self._lm_head(_b2t(data, shape))


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, server_head

    # Auto-generate model slices on first start (e.g. Render persistent disk is empty)
    head_path = os.path.join(MODEL_SLICES_DIR, "server_head.pt")
    if not os.path.isfile(head_path):
        print("[server] model slices not found — running splitter (first run, ~5 min)…")
        from splitter.split_model import split
        split(MODEL_SLICES_DIR)

    tok_path = TOKENIZER_PATH if os.path.isdir(TOKENIZER_PATH) else "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(tok_path)
    print(f"[server] tokenizer loaded from '{tok_path}'")

    head_path = os.path.join(MODEL_SLICES_DIR, "server_head.pt")
    if os.path.isfile(head_path):
        server_head = _ServerHead(head_path)
        print(f"[server] server_head loaded (browser workers enabled)")
    else:
        print("[server] server_head.pt not found — browser workers disabled until you re-run splitter")

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


# ─── Sampling ─────────────────────────────────────────────────────────────────

import torch.nn.functional as F

def _sample_next(logits_row: torch.Tensor, temperature: float, top_p: float) -> int:
    logits_row = logits_row / max(temperature, 1e-6)
    probs = F.softmax(logits_row, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep the minimal set whose cumulative prob >= top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()
    chosen = sorted_idx[torch.multinomial(sorted_probs, num_samples=1)]
    return int(chosen)


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

    # Wrap in Q&A format — nudges GPT-2 toward a response rather than random continuation
    wrapped = f"Q: {text}\nA:"
    input_ids = tokenizer.encode(wrapped)
    generated = list(input_ids)

    all_browser = all(d.ws is not None for d in pipeline)

    if all_browser and server_head is None:
        yield {"error": "server_head.pt missing — re-run: python -m splitter.split_model then restart server"}
        return

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for _ in range(max_new_tokens):

            if all_browser:
                # ── Browser workers: server embeds → WS routing → server projects ──
                try:
                    b64, shape = server_head.embed(generated)
                except Exception as exc:
                    yield {"error": f"embed failed: {exc}"}
                    return

                for device in pipeline:
                    rid = uuid.uuid4().hex
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
                        room.mark_dropped(device.device_id)
                        await _broadcast(room_id, {"type": "device_dropped",
                                                    "device_id": device.device_id,
                                                    **room.to_dict()})
                        yield {"error": str(exc)}
                        return
                    finally:
                        _pending_forwards.pop(rid, None)

                logits = server_head.project(b64, shape)

            else:
                # ── Python workers: existing HTTP /forward routing ───────────────
                hidden_payload = None
                logits_payload = None

                for device in pipeline:
                    body = ({"input_ids": generated} if device.is_first
                            else {"hidden": hidden_payload})
                    try:
                        r = await http_client.post(f"{device.url}/forward", json=body)
                        r.raise_for_status()
                    except Exception as exc:
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

                logits = payload_to_tensor(logits_payload)

            next_token = _sample_next(logits[0, -1, :], temperature, top_p)
            generated.append(next_token)
            decoded = tokenizer.decode([next_token])
            yield {"token": decoded}

            if next_token == tokenizer.eos_token_id or decoded == "\n":
                break

    yield {"done": True, "full": tokenizer.decode(generated)}


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

@app.post("/room/create")
def room_create():
    return {"room_id": room_manager.create().room_id}

@app.post("/room/{room_id}/join")
def room_join(room_id: str, req: JoinRequest):
    room = room_manager.get_or_create(room_id)
    dev  = room.join(req.device_id, req.ram_mb)
    if dev is None:
        raise HTTPException(409, "Room full — all 3 slice slots are taken")
    return {
        "device_id":    dev.device_id,
        "slice_id":     dev.slice_id,
        "layers":       [dev.layer_start, dev.layer_end],
        "download_url": f"/slice/{dev.slice_id}",
        "onnx_url":     f"/onnx/{dev.slice_id}",
    }

@app.get("/room/{room_id}/status")
def room_status(room_id: str):
    room = room_manager.get(room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    return room.to_dict()


# ─── Slice serving ────────────────────────────────────────────────────────────

@app.get("/slice/{slice_id}")
def get_slice_pt(slice_id: int):
    """Serves .pt slice files for Python workers."""
    path = Path(MODEL_SLICES_DIR) / f"slice_{slice_id}.pt"
    if not path.is_file():
        raise HTTPException(404, f"Slice {slice_id} not found — run: python -m splitter.split_model")
    return FileResponse(
        str(path), media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=slice_{slice_id}.pt"},
    )

@app.get("/onnx/{slice_id}")
def get_slice_onnx(slice_id: int):
    """Serves .onnx slice files for browser workers."""
    path = Path(MODEL_SLICES_DIR) / f"slice_{slice_id}.onnx"
    if not path.is_file():
        raise HTTPException(
            404,
            f"ONNX slice {slice_id} not found — re-run: python -m splitter.split_model"
        )
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
                # Browser worker announces ONNX loaded and ready
                device_id = msg.get("device_id", "")
                ram_mb    = msg.get("ram_mb", 2048)
                my_device_id = device_id

                dev = room.get_device(device_id)
                if dev is None:
                    # Auto-assign if device hasn't called /join yet
                    dev = room.join(device_id, ram_mb)

                if dev is not None:
                    dev.status = "ready"
                    dev.ws     = websocket
                    state = room.to_dict()
                    await _broadcast(room_id, {"type": "device_ready", **state})
                    if room.pipeline_ready:
                        await _broadcast(room_id, {"type": "pipeline_ready", **state})

            elif t == "forward_result":
                # Browser returns result of a forward pass
                rid = msg.get("request_id", "")
                if rid in _pending_forwards:
                    fut = _pending_forwards[rid]
                    if not fut.done():
                        fut.set_result(msg)

            elif t == "forward_error":
                # Browser reports an error during forward pass
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


# ─── Legacy endpoints (Python worker / Phase 1 compatibility) ─────────────────

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

    input_ids = tokenizer.encode(req.text)
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

            logits     = payload_to_tensor(logits_payload)
            next_token = int(logits[0, -1, :].argmax())
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    return {"input": req.text, "output": tokenizer.decode(generated)}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0").strip()
    port = int(os.environ.get("PORT", "8000").strip())
    uvicorn.run(app, host=host, port=port, reload=False)
