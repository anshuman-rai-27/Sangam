# Sangam Inference Flow — "Hello" Example

When you type **"hello"** and press Send, here is exactly what happens.

---

## High-Level Flow

```
Browser UI
    │
    │  GET /room/{id}/stream?prompt=hello
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        SERVER                               │
│                                                             │
│  1. Tokenize                                                │
│     "hello"  ──►  [token_ids]  e.g. [14990]                │
│                                                             │
│  2. Apply chat template                                     │
│     "<|im_start|>user\nhello<|im_end|>\n                    │
│      <|im_start|>assistant\n"                               │
│     ──► input_ids = [151644, 872, 198, 14990, 151645, ...]  │
│                                                             │
│  3. Embed  (wte lookup, float16 → float32)                  │
│     input_ids ──► hidden  shape: (1, seq_len, 896)          │
└──────────────────┬──────────────────────────────────────────┘
                   │  WS: forward_request {data, shape}
                   ▼
        ┌──────────────────┐
        │   DEVICE 0       │  slice_0.onnx
        │   layers 0–7     │  hidden (1, seq, 896) → hidden (1, seq, 896)
        └────────┬─────────┘
                 │  WS: forward_result {data, shape}
                 ▼
┌────────────────────────────────────────────────────────────┐
│                        SERVER                              │
│  receives hidden from device 0, sends to device 1          │
└──────────────────┬─────────────────────────────────────────┘
                   │  WS: forward_request {data, shape}
                   ▼
        ┌──────────────────┐
        │   DEVICE 1       │  slice_1.onnx
        │   layers 8–15    │  hidden (1, seq, 896) → hidden (1, seq, 896)
        └────────┬─────────┘
                 │  WS: forward_result {data, shape}
                 ▼
┌────────────────────────────────────────────────────────────┐
│                        SERVER                              │
│  receives hidden from device 1, sends to device 2          │
└──────────────────┬─────────────────────────────────────────┘
                   │  WS: forward_request {data, shape}
                   ▼
        ┌──────────────────┐
        │   DEVICE 2       │  slice_2.onnx
        │   layers 16–23   │  hidden (1, seq, 896) → hidden (1, seq, 896)
        │   + RMSNorm      │
        └────────┬─────────┘
                 │  WS: forward_result {data, shape}
                 ▼
┌────────────────────────────────────────────────────────────┐
│                        SERVER                              │
│                                                             │
│  4. Project last token                                      │
│     hidden[0, -1, :]  shape: (896,)  fp16                  │
│     logits = hidden @ wte.T  shape: (151936,)               │
│                                                             │
│  5. Sample next token  (temperature=0.9, top-p=0.9)        │
│     logits ──► next_token_id  e.g. 13  → " Hi"             │
│                                                             │
│  6. Stream token via SSE                                    │
│     data: {"token": " Hi"}   ──────────────────────────►   │
│                                                             │
│  7. Append next_token to generated[], repeat from step 3   │
│     (now seq_len grows by 1 each iteration)                 │
│                                                             │
│  8. Stop when EOS token (151645) is sampled                 │
│     or max_new_tokens (80) reached                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Per-Token Timing

Each generated token costs exactly **one full pipeline round-trip**:

```
Token N:
  Server embed  →  Device0 (8 layers)  →  Device1 (8 layers)  →  Device2 (8 layers + norm)  →  Server project + sample
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────►
                                                                                               stream token to UI
Token N+1:
  Server embed  →  Device0  →  Device1  →  Device2  →  Server
  ───────────────────────────────────────────────────────────►
                                           stream token to UI
...
```

This is why generation is **word-by-word** — each token depends on all previous ones so parallelism is impossible across tokens.

---

## Data Shapes at Each Stage

| Stage | Tensor shape | dtype |
|-------|-------------|-------|
| input_ids | `(seq_len,)` | int32 |
| After embed (wte) | `(1, seq_len, 896)` | float32 |
| After device 0 | `(1, seq_len, 896)` | float32 |
| After device 1 | `(1, seq_len, 896)` | float32 |
| After device 2 + norm | `(1, seq_len, 896)` | float32 |
| Last token slice | `(896,)` | float16 |
| Logits (lm_head) | `(151936,)` | float32 |
| next_token | scalar | int32 |

---

## Transport Between Server and Devices

Hidden states are serialised as **base64-encoded raw float32 bytes** over WebSocket JSON messages:

```
Server → Device:
{
  "type":       "forward_request",
  "request_id": "a3f9...",
  "slice_id":   0,
  "data":       "<base64 float32 bytes>",
  "shape":      [1, 12, 896]
}

Device → Server:
{
  "type":       "forward_result",
  "request_id": "a3f9...",
  "data":       "<base64 float32 bytes>",
  "shape":      [1, 12, 896]
}
```

A sequence of length 12 costs `1 × 12 × 896 × 4 bytes ≈ 43 KB` per device hop, or ~86 KB total per token (two round-trips: server→device and device→server, ×3 devices).
