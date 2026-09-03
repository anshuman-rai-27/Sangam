"""
Device worker — run one per device.

Environment variables:
    DEVICE_ID    unique name for this device          (default: device_0)
    DEVICE_PORT  port this worker listens on          (default: 8001)
    DEVICE_HOST  host this worker advertises          (default: localhost)
    SERVER_URL   URL of the central server            (default: http://localhost:8000)
    SLICE_PATH   path to the .pt slice file           (default: model_slices/slice_0.pt)

Usage:
    DEVICE_ID=device_0 DEVICE_PORT=8001 SLICE_PATH=model_slices/slice_0.pt \
        python -m worker.main
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.serialization import tensor_to_payload, payload_to_tensor
from worker.model_slice import GPT2Slice

DEVICE_ID   = os.environ.get("DEVICE_ID",   "device_0").strip()
DEVICE_PORT = int(os.environ.get("DEVICE_PORT", "8001").strip())
DEVICE_HOST = os.environ.get("DEVICE_HOST", "localhost").strip()
SERVER_URL  = os.environ.get("SERVER_URL",  "http://localhost:8000").strip()
SLICE_PATH  = os.environ.get("SLICE_PATH",  "model_slices/slice_0.pt").strip()
ROOM_ID     = os.environ.get("ROOM_ID",     "").strip()

model = GPT2Slice()


# ------------------------------------------------------------------
# Lifespan: load model slice + register with server
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load(SLICE_PATH)
    advertised_url = f"http://{DEVICE_HOST}:{DEVICE_PORT}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{SERVER_URL}/register",
                json={
                    "device_id":   DEVICE_ID,
                    "url":         advertised_url,
                    "layer_start": model.layer_start,
                    "layer_end":   model.layer_end,
                    "is_first":    model.is_first,
                    "is_last":     model.is_last,
                    "room_id":     ROOM_ID or None,
                },
                timeout=10.0,
            )
            print(f"[worker] registered: {resp.json()}")
        except Exception as exc:
            print(f"[worker] WARNING: could not register with server: {exc}")
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class ForwardRequest(BaseModel):
    input_ids: Optional[list] = None   # only first device
    hidden:    Optional[dict] = None   # middle / last devices

class ForwardResponse(BaseModel):
    hidden: Optional[dict] = None      # first / middle devices
    logits: Optional[dict] = None      # last device only


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/ping")
def ping():
    return {"status": "ok", "device_id": DEVICE_ID,
            "layers": [model.layer_start, model.layer_end]}


@app.post("/forward", response_model=ForwardResponse)
def forward(req: ForwardRequest):
    if model.is_first:
        if req.input_ids is None:
            raise HTTPException(400, "input_ids required for first device")
        hidden = model.forward_first(req.input_ids)
        return ForwardResponse(hidden=tensor_to_payload(hidden))

    if req.hidden is None:
        raise HTTPException(400, "hidden tensor required for non-first device")

    hidden = payload_to_tensor(req.hidden)

    if model.is_last:
        logits = model.forward_last(hidden)
        return ForwardResponse(logits=tensor_to_payload(logits))

    hidden = model.forward_middle(hidden)
    return ForwardResponse(hidden=tensor_to_payload(hidden))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEVICE_PORT, reload=False)
