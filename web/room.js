// room.js — WebSocket room state + browser worker (ONNX) + streaming inference

const roomId      = window.location.pathname.split('/').pop();
const serverOrigin = window.location.origin;

// ── Worker state ──────────────────────────────────────────────────────────────
let ortSession   = null;  // onnxruntime-web InferenceSession
let mySliceId    = null;
let myDeviceId   = sessionStorage.getItem('sangam_device') || null;
let isWorker     = false;
let pipelineOk   = false;
let ws           = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.getElementById('room-id-chip').textContent = roomId;
document.title = `Sangam · ${roomId}`;
document.getElementById('join-cmd').innerHTML =
  `python -m worker.join --server <em>${serverOrigin}</em> --room <em>${roomId}</em>`;

connectWS();

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/room/${roomId}`);

  ws.onopen = () => {
    setWsStatus('connected', false);
    // Keepalive — Render free tier drops idle WS after ~55s
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:'ping'}));
      else clearInterval(ping);
    }, 25000);
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    if (['room_state','device_ready','pipeline_ready','device_dropped'].includes(msg.type)) {
      updateRoom(msg);
    }

    // Server is asking THIS browser to run a forward pass
    if (msg.type === 'forward_request' && isWorker && msg.slice_id === mySliceId) {
      handleForwardRequest(msg);
    }
  };

  ws.onclose = () => {
    setWsStatus('disconnected — reconnecting…', true);
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => ws.close();
}

function setWsStatus(text, isErr) {
  const el = document.getElementById('ws-status');
  el.textContent = text;
  el.className   = 'status-line' + (isErr ? ' error' : '');
}

// ── Room state rendering ──────────────────────────────────────────────────────

const SLICE_LABELS = [
  'blocks 0–3 (first)',
  'blocks 4–7 (middle)',
  'blocks 8–11 + ln (last)',
];

function updateRoom(data) {
  const ready = data.ready_count  ?? 0;
  const total = data.total_slots  ?? 3;

  document.getElementById('pipeline-fill').style.width = `${(ready / total) * 100}%`;

  const lbl = document.getElementById('pipeline-label');
  lbl.textContent = `${ready} / ${total} ready`;
  lbl.className   = 'status-line' + (data.pipeline_ready ? ' ready' : '');

  // Device slots
  const list    = document.getElementById('device-list');
  list.innerHTML = '';
  const slotMap  = {};
  (data.devices || []).forEach(d => { slotMap[d.slice_id] = d; });

  for (let i = 0; i < total; i++) {
    const dev = slotMap[i];
    if (!dev) {
      const el = document.createElement('div');
      el.className   = 'slot-empty';
      el.textContent = `Slot ${i+1} — ${SLICE_LABELS[i]} — waiting…`;
      list.appendChild(el);
    } else {
      const isMe = (dev.device_id === myDeviceId);
      const el   = document.createElement('div');
      el.className = 'device-item';
      el.innerHTML = `
        <div class="device-info">
          <div class="name">${esc(dev.device_id)}${isMe ? ' <span class="you">[YOU]</span>' : ''}</div>
          <div class="layers">${SLICE_LABELS[i]}${dev.is_browser ? ' · browser' : ' · python'}</div>
        </div>
        <span class="badge badge-${dev.status}">${dev.status}</span>
      `;
      list.appendChild(el);
    }
  }

  pipelineOk = !!data.pipeline_ready;
  document.getElementById('send-btn').disabled = !pipelineOk;

  const inferStatus = document.getElementById('infer-status');
  if (pipelineOk) {
    inferStatus.textContent = 'Pipeline ready — type a prompt and press Send.';
    inferStatus.className   = 'status-line ready';
  } else {
    inferStatus.textContent = `Waiting for all ${total} device slots to connect…`;
    inferStatus.className   = 'status-line';
  }
}

// ── Browser worker: join + download ONNX + run forward passes ─────────────────

async function joinAsWorker() {
  const btn = document.getElementById('join-worker-btn');
  btn.disabled = true;

  // Persist device ID across refreshes
  if (!myDeviceId) {
    myDeviceId = 'device_' + Math.random().toString(36).substr(2, 6);
    sessionStorage.setItem('sangam_device', myDeviceId);
  }

  setWorkerStatus('joining room…');

  // 1. Join room (HTTP) — get slice assignment
  let assignment;
  try {
    const resp = await fetch(`/room/${roomId}/join`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device_id: myDeviceId, ram_mb: estimateRAM()}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    assignment = await resp.json();
  } catch (e) {
    setWorkerStatus('join failed: ' + e.message, true);
    btn.disabled = false;
    return;
  }

  mySliceId = assignment.slice_id;
  setWorkerStatus(`assigned slice ${mySliceId} — downloading ONNX (~112 MB)…`);
  showDlBar(true);

  // 2. Download ONNX with progress
  let onnxBuffer;
  try {
    const onnxUrl = `https://huggingface.co/anshumanrai/sangam-gpt2-slices/resolve/main/slice_${mySliceId}.onnx`;
  onnxBuffer = await downloadWithProgress(onnxUrl, (pct) => {
      document.getElementById('dl-bar').style.width = pct + '%';
      document.getElementById('dl-label').textContent = `downloading slice ${mySliceId}: ${pct}%`;
    });
  } catch (e) {
    setWorkerStatus('download failed: ' + e.message, true);
    btn.disabled = false;
    showDlBar(false);
    return;
  }

  setWorkerStatus('loading model (WASM)…');
  document.getElementById('dl-bar').style.width = '100%';

  // 3. Load ONNX into onnxruntime-web
  try {
    // Point wasm loader at the CDN (avoids CSP issues with inline WASM)
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
    ortSession = await ort.InferenceSession.create(onnxBuffer, {
      executionProviders: ['webgpu', 'wasm'],
    });
  } catch (e) {
    setWorkerStatus('ONNX load failed: ' + e.message, true);
    btn.disabled = false;
    showDlBar(false);
    return;
  }

  // 4. Announce ready via WebSocket
  ws.send(JSON.stringify({
    type:      'worker_ready',
    device_id: myDeviceId,
    slice_id:  mySliceId,
    ram_mb:    estimateRAM(),
  }));

  isWorker = true;
  showDlBar(false);
  setWorkerStatus(`active — slice ${mySliceId} ready`);
  btn.textContent = `Worker active (slice ${mySliceId})`;
  document.getElementById('worker-panel').style.borderColor = 'var(--green)';
}

// ── Forward pass (called when server sends forward_request) ───────────────────

async function handleForwardRequest(msg) {
  if (!ortSession) return;

  try {
    const f32   = b64ToFloat32(msg.data);
    const shape = msg.shape;                     // [1, seq_len, 768]
    const input = new ort.Tensor('float32', f32, shape);

    const outputs   = await ortSession.run({hidden_states: input});
    const outTensor = outputs[Object.keys(outputs)[0]];  // 'output'

    ws.send(JSON.stringify({
      type:       'forward_result',
      request_id: msg.request_id,
      data:       float32ToB64(outTensor.data),
      shape:      Array.from(outTensor.dims),
    }));
  } catch (e) {
    ws.send(JSON.stringify({
      type:       'forward_error',
      request_id: msg.request_id,
      error:      e.message,
    }));
  }
}

// ── Inference (user presses Send) ─────────────────────────────────────────────

function handleKey(ev) {
  if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); sendPrompt(); }
}

async function sendPrompt() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt || !pipelineOk) return;

  const sendBtn    = document.getElementById('send-btn');
  const output     = document.getElementById('output-box');
  const inferStatus= document.getElementById('infer-status');

  sendBtn.disabled = true;
  inferStatus.textContent = 'Generating…';
  inferStatus.className   = 'status-line';

  output.innerHTML = `<span class="prompt-echo">&gt; ${esc(prompt)}</span>\n`;
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  output.appendChild(cursor);

  const url = `/room/${roomId}/stream?prompt=${encodeURIComponent(prompt)}&max_new_tokens=80&temperature=0.9&top_p=0.9`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let payload;
        try { payload = JSON.parse(line.slice(6)); } catch { continue; }

        if (payload.token !== undefined) {
          cursor.remove();
          const span = document.createElement('span');
          span.className   = 'token';
          span.textContent = payload.token;
          output.appendChild(span);
          output.appendChild(cursor);
          output.scrollTop = output.scrollHeight;
        }

        if (payload.done || payload.error) {
          cursor.remove();
          if (payload.error) {
            const err = document.createElement('span');
            err.className   = 'error-msg';
            err.textContent = `\n[error: ${payload.error}]`;
            output.appendChild(err);
          }
          break;
        }
      }
    }
  } catch (err) {
    cursor.remove();
    output.insertAdjacentHTML('beforeend',
      `<span class="error-msg">\n[fetch error: ${esc(err.message)}]</span>`);
  }

  inferStatus.textContent = 'Done.';
  inferStatus.className   = 'status-line ready';
  sendBtn.disabled = !pipelineOk;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

async function downloadWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const total  = parseInt(resp.headers.get('content-length') || '0');
  const reader = resp.body.getReader();
  const chunks = [];
  let downloaded = 0;

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    downloaded += value.length;
    if (total && onProgress) onProgress(Math.round(100 * downloaded / total));
  }

  const buf = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) { buf.set(chunk, offset); offset += chunk.length; }
  return buf.buffer;
}

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function float32ToB64(f32arr) {
  const bytes = new Uint8Array(f32arr.buffer);
  let bin = '';
  // Process in chunks to avoid stack overflow on large tensors
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function estimateRAM() {
  return Math.round((navigator.deviceMemory || 2) * 1024);
}

function setWorkerStatus(text, isErr) {
  const el = document.getElementById('worker-status');
  el.textContent = text;
  el.className   = 'status-line' + (isErr ? ' error' : '');
}

function showDlBar(show) {
  document.getElementById('dl-progress-wrap').style.display = show ? 'block' : 'none';
}

function copyLink() {
  navigator.clipboard.writeText(window.location.href)
    .then(() => showToast('Link copied!'))
    .catch(() => showToast(window.location.href));
}

function copyJoinCmd() {
  const cmd = `python -m worker.join --server ${serverOrigin} --room ${roomId}`;
  navigator.clipboard.writeText(cmd).then(() => showToast('Command copied!')).catch(() => {});
}

function showToast(msg) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    Object.assign(t.style, {
      position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)',
      background: 'var(--surface2)', border: '1px solid var(--border)',
      color: 'var(--text)', padding: '8px 18px', borderRadius: '6px',
      fontSize: '13px', zIndex: '999', transition: 'opacity .3s',
    });
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = '1';
  setTimeout(() => { t.style.opacity = '0'; }, 2000);
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
