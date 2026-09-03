"""
Local integration test — boots server + 3 workers in subprocesses, runs one inference.

Prerequisites:
    1. python -m splitter.split_model      (download + split GPT-2 once)
    2. pip install -r requirements.txt

Usage:
    python -m tests.run_local
"""
import os
import subprocess
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(__file__))

WORKERS = [
    {"DEVICE_ID": "device_0", "DEVICE_PORT": "8001", "SLICE_PATH": "model_slices/slice_0.pt"},
    {"DEVICE_ID": "device_1", "DEVICE_PORT": "8002", "SLICE_PATH": "model_slices/slice_1.pt"},
    {"DEVICE_ID": "device_2", "DEVICE_PORT": "8003", "SLICE_PATH": "model_slices/slice_2.pt"},
]

SERVER_URL = "http://localhost:8000"


def start(module: str, extra_env: dict = None) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": ROOT, **(extra_env or {})}
    cmd = [sys.executable, "-m", module]
    # DEVNULL prevents pipe-buffer deadlock; output goes nowhere (use terminal directly if debugging)
    return subprocess.Popen(cmd, cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_http(url: str, retries: int = 40, delay: float = 1.0) -> bool:
    for _ in range(retries):
        try:
            httpx.get(url, timeout=2.0)
            return True
        except Exception:
            time.sleep(delay)
    return False


def run():
    procs = []

    # --- Server ---
    print("Starting central server...")
    procs.append(start("server.main"))
    if not wait_for_http(f"{SERVER_URL}/status"):
        print("ERROR: server did not start")
        sys.exit(1)
    print("  server up")

    # --- Workers ---
    for w in WORKERS:
        print(f"Starting {w['DEVICE_ID']} (port {w['DEVICE_PORT']})...")
        env = {**w, "SERVER_URL": SERVER_URL, "DEVICE_HOST": "localhost"}
        procs.append(start("worker.main", extra_env=env))
        if not wait_for_http(f"http://localhost:{w['DEVICE_PORT']}/ping"):
            print(f"ERROR: {w['DEVICE_ID']} did not start")
            for p in procs:
                p.terminate()
            sys.exit(1)
        print(f"  {w['DEVICE_ID']} up")

    # Give workers time to register
    time.sleep(2)

    # --- Check pipeline status ---
    with httpx.Client(timeout=10.0) as client:
        st = client.get(f"{SERVER_URL}/status").json()
    print(f"\nPipeline ready: {st['pipeline_ready']}")
    for d in st["devices"]:
        print(f"  {d['device_id']}  layers {d['layers']}  status={d['status']}")

    if not st["pipeline_ready"]:
        print("ERROR: pipeline not complete after all workers registered")
        for p in procs:
            p.terminate()
        sys.exit(1)

    # --- Run inference ---
    prompt = "The future of artificial intelligence is"
    print(f"\nRunning inference: '{prompt}'")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{SERVER_URL}/infer",
                           json={"text": prompt, "max_new_tokens": 15})
        resp.raise_for_status()
        result = resp.json()

    print(f"\nPrompt : {result['input']}")
    print(f"Output : {result['output']}")
    print("\nAll tests passed.")

    for p in procs:
        p.terminate()


if __name__ == "__main__":
    run()
