# Sangam

Distributed ML inference — GPT-2 Small (117M) split across 3 devices, orchestrated by a central server.
Devices join via a website, each browser downloads its model slice (~113 MB), and compute together.
**No Python or install required on participant devices — just a browser.**

---

## How it works

```
Browser / device                    Server
─────────────────                   ──────────────────────────────────
Open room URL
Click "Join as worker"
  → POST /room/{id}/join            assign slice slot (0, 1, or 2)
  ← {slice_id: 0}
  → GET /onnx/0                     stream slice_0.onnx (~113 MB)
  ← file (progress bar shown)
  ort.InferenceSession.create()
  → WS worker_ready                 mark device ready; broadcast state

User types prompt → Send
  → GET /room/{id}/stream           start SSE token stream
                                      embed(input_ids) via server_head
                                      → WS forward_request → device 0
                                      ← WS forward_result
                                      → WS forward_request → device 1
                                      ← WS forward_result
                                      → WS forward_request → device 2
                                      ← WS forward_result
                                      lm_head → greedy decode
  ← data: {"token": "Hello"}        token appears in browser
  ← data: {"token": " world"}
  ...
```

**Slice layout** (GPT-2 Small, 12 transformer blocks):

| Slot | Layers | ONNX size | Handled by |
|------|--------|-----------|------------|
| 0 | blocks 0–3 | ~113 MB | browser |
| 1 | blocks 4–7 | ~113 MB | browser |
| 2 | blocks 8–11 + ln_f | ~113 MB | browser |
| — | wte + wpe + lm_head | ~157 MB | server RAM only |

---

## Server setup (run once)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Split the model (downloads ~500 MB, run once)
```bash
# Linux / Mac / Git Bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m splitter.split_model

# Windows CMD
set PYTHONIOENCODING=utf-8&& set PYTHONPATH=.&& python -m splitter.split_model
```

Produces in `model_slices/`:
- `slice_0.pt`, `slice_1.pt`, `slice_2.pt` — for Python workers
- `slice_0.onnx`, `slice_1.onnx`, `slice_2.onnx` — for browser workers
- `server_head.pt` — embedding + LM head (server only)
- `tokenizer/` — GPT-2 tokenizer

### 3. Start server
```bash
# Linux / Mac / Git Bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m server.main

# Windows CMD
set PYTHONIOENCODING=utf-8&& set PYTHONPATH=.&& python -m server.main
```

Open **http://localhost:8000**

---

## Connecting devices (browser workers — no install needed)

1. Open the server URL on any device (phone, laptop, tablet)
2. Create or join a room
3. Click **"Join as worker"** — the browser downloads its assigned slice and runs ONNX inference via WebAssembly
4. Once all 3 slots show **ready**, type a prompt and press **Send**

Tokens stream back in real time across all connected tabs.

---

## Python workers (optional, for devices with Python)

If a device has Python installed, it can contribute via terminal instead:

```bash
# Linux / Mac
PYTHONPATH=. python -m worker.join --server http://<SERVER_IP>:8000 --room <ROOM_ID>

# Windows CMD
set PYTHONPATH=.&& python -m worker.join --server http://<SERVER_IP>:8000 --room <ROOM_ID>
```

Python workers and browser workers can coexist in the same room.

---

## Local test (3 browser tabs, one machine)

1. Start server: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python -m server.main`
2. Open **http://localhost:8000** → Create room
3. Open the room URL in **3 browser tabs**
4. Click **"Join as worker"** in each tab — they each download a different slice
5. Once all 3 show ready, type a prompt

To test with Python workers instead, open 3 extra terminals:
```bash
# (replace ROOM_ID with the code from step 2)
PYTHONPATH=. python -m worker.join --server http://localhost:8000 --room <ROOM_ID> --port 8001
PYTHONPATH=. python -m worker.join --server http://localhost:8000 --room <ROOM_ID> --port 8002
PYTHONPATH=. python -m worker.join --server http://localhost:8000 --room <ROOM_ID> --port 8003
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Web UI home |
| `GET`  | `/room/{id}` | Room page |
| `POST` | `/room/create` | Create a new room |
| `POST` | `/room/{id}/join` | Join room → get slice assignment |
| `GET`  | `/room/{id}/status` | Room state (JSON) |
| `GET`  | `/room/{id}/stream?prompt=...&max_new_tokens=80` | SSE token stream |
| `GET`  | `/onnx/{n}` | Download ONNX slice n (for browser workers) |
| `GET`  | `/slice/{n}` | Download .pt slice n (for Python workers) |
| `WS`   | `/ws/room/{id}` | Live room state + browser worker routing |
| `POST` | `/register` | Register a Python worker (legacy) |
| `GET`  | `/status` | Legacy pipeline status |
| `POST` | `/infer` | Legacy inference (full output, no streaming) |

---

## Project structure

```
server/
  main.py           HTTP + WebSocket + SSE + browser inference routing
  registry.py       legacy device registry
  room.py           RoomManager — rooms, slice slot assignment

worker/
  model_slice.py    GPT2Slice — forward pass for a layer range
  main.py           FastAPI worker: /forward /ping
  download.py       downloads a .pt slice with progress bar
  join.py           join room → download → start Python worker

shared/
  serialization.py  tensor <-> base64 for Python worker transport

splitter/
  split_model.py    downloads GPT-2, exports .pt + .onnx slices + tokenizer

web/
  index.html        create / join room
  room.html         room UI (devices + inference panel)
  room.js           WebSocket + ONNX worker + SSE streaming
  style.css         dark theme

model_slices/       (generated by splitter, not committed)
  slice_{0,1,2}.pt
  slice_{0,1,2}.onnx
  server_head.pt
  tokenizer/
```

---

## Docker

```bash
docker compose up --build   # first run — downloads and splits model
docker compose up -d        # subsequent starts
```

Server at http://localhost:8000
