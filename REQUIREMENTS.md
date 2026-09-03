# Sangam — Product Requirements

---

## Core Vision

Multiple devices open a website, form a "room", and collectively run a language model.
The model is split across them automatically. Anyone in the room can send a prompt and
see the response — generated live by the distributed pipeline.

---

## User Flow

```
Device A opens  sangam.local/room/abc  ──┐
Device B opens  sangam.local/room/abc  ──┤──► Server assigns layers to each device
Device C opens  sangam.local/room/abc  ──┘    Each device downloads its slice

User types prompt on the website
         │
         ▼
Server routes activations: Device A → Device B → Device C
         │
         ▼
Response streams back to the website in real time
```

---

## Requirements

### R1 — Room System
- Any device can create a room; gets a shareable URL (e.g. `/room/abc123`)
- Other devices join by opening the same URL
- Website shows live list of connected devices and their status
- Room stays active as long as at least one device is connected

### R2 — Automatic Device Assignment
- When a device joins, server assigns it a layer range based on:
  - How many devices are in the room
  - Device's reported available RAM (sent on join)
- Assignment is shown to the user on the website: "You are running layers 4–7"

### R3 — Slice Download on the Device
- After assignment, the device automatically downloads its slice from the server
- Download progress is shown on the website: "Downloading model slice… 43%"
- Slice is cached locally so rejoining the same room skips re-download
- Once downloaded, device status changes to "ready"

### R4 — Pipeline Ready Gate
- Inference is only available when all required layer ranges are covered by ready devices
- Website shows pipeline status: "2/3 devices ready" or "Pipeline ready"
- If a device drops mid-session, the website shows an error and waits for reconnect

### R5 — Inference from the Website
- Input box on the website accepts a text prompt
- Submitted only when pipeline is ready
- Response streams token-by-token to the website (not a single bulk response)
- Multiple users in the same room see the same response stream

### R6 — Device Worker (what runs on the device)
- Two options (decision needed — see open questions):
  - **Option A: Python script** — user downloads and runs a script; more capable, works on mobile via Termux
  - **Option B: Browser tab** — device just keeps the tab open; uses WebAssembly (ONNX.js / transformers.js); zero install, works on any device including iPhone
- Device must stay connected (tab open / script running) during inference

---

## What Needs to Be Built

| Component | Status | Notes |
|---|---|---|
| Room creation + join | Missing | Need room ID system, URL routing |
| Device registration UI | Missing | Show assigned layers, download progress |
| Slice assignment logic | Partial | registry.py assigns layers but not dynamically based on device count |
| Slice download endpoint | Missing | `GET /slice/{slice_id}` on server |
| Slice download on device | Missing | Client-side download + cache |
| Pipeline status UI | Missing | Live "N/3 ready" indicator |
| Inference input box | Missing | Web frontend |
| Streaming response | Missing | Server-Sent Events or WebSocket from server to browser |
| Device worker in browser | Missing | If Option B: ONNX.js inference in a Web Worker |
| Device worker as script | Exists | worker/main.py — but manual, no auto-download |

---

## Open Questions (decisions needed before building)

1. **Device worker: browser tab or Python script?**
   Browser tab = zero friction, works on iPhone, but limited to ONNX.js/WebAssembly performance.
   Python script = faster inference, works on Termux (Android), requires install.
   *Recommendation: browser tab (Option B) — removes all friction for mobile.*

2. **Streaming: SSE or WebSocket?**
   SSE is simpler (one-way server→browser). WebSocket is bidirectional (needed if devices also communicate via WS).
   *Recommendation: WebSocket for devices (already need bidirectional), SSE for browser inference stream.*

3. **Where does the server run?**
   Currently localhost only. For real rooms across devices: needs a public IP or LAN.
   *Recommendation: LAN for now (all devices on same WiFi), public deployment later.*

4. **Model caching on device**
   Where to cache the slice? Browser: IndexedDB (up to ~1GB on mobile). Python: local filesystem.

---

## Out of Scope (for now)

- Authentication / private rooms
- Multiple simultaneous inference requests
- More than one model (GPT-2 Small only)
- GPU acceleration
- Public internet deployment
