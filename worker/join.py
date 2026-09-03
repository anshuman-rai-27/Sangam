"""
Single-command device join: join room -> download slice -> start worker.

Usage:
    python -m worker.join --server http://192.168.1.x:8000 --room abc123
    python -m worker.join --server http://192.168.1.x:8000 --room abc123 --port 8002 --host 192.168.1.y
"""
import argparse
import os
import subprocess
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="Join a Sangam room and start the worker")
    parser.add_argument("--server",    required=True, help="Server URL, e.g. http://192.168.1.x:8000")
    parser.add_argument("--room",      required=True, help="Room ID")
    parser.add_argument("--device-id", default=None,  help="Device ID (auto-generated if omitted)")
    parser.add_argument("--ram",       type=int, default=2048, help="Available RAM in MB")
    parser.add_argument("--port",      type=int, default=8001, help="Port for this worker to listen on")
    parser.add_argument("--host",      default="localhost",
                        help="Hostname/IP this worker advertises to the server (use LAN IP for multi-device)")
    args = parser.parse_args()

    device_id  = args.device_id or f"device_{uuid.uuid4().hex[:6]}"
    server_url = args.server.rstrip("/")

    from worker.download import join_room, download_slice

    # 1. Join room — get slice assignment
    print(f"[join] joining room '{args.room}' as '{device_id}'")
    try:
        assignment = join_room(server_url, args.room, device_id, args.ram)
    except Exception as exc:
        print(f"[join] ERROR: could not join room: {exc}")
        sys.exit(1)

    slice_id = assignment["slice_id"]
    layers   = assignment["layers"]
    print(f"[join] assigned slice {slice_id} (layers {layers[0]}-{layers[1]})")

    # 2. Download slice (cached if already present)
    cache_path = os.path.join("model_slices", f"room_{args.room}_slice{slice_id}.pt")
    if os.path.isfile(cache_path):
        print(f"[join] using cached slice: {cache_path}")
    else:
        try:
            download_slice(server_url, slice_id, cache_path)
        except Exception as exc:
            print(f"[join] ERROR: download failed: {exc}")
            sys.exit(1)

    # 3. Start worker (replaces this process)
    print(f"[join] starting worker on {args.host}:{args.port}")
    env = {
        **os.environ,
        "PYTHONPATH":  ROOT,
        "DEVICE_ID":   device_id,
        "DEVICE_PORT": str(args.port),
        "DEVICE_HOST": args.host,
        "SERVER_URL":  server_url,
        "SLICE_PATH":  cache_path,
        "ROOM_ID":     args.room,
    }
    # Use subprocess.run so Ctrl-C propagates naturally
    subprocess.run(
        [sys.executable, "-m", "worker.main"],
        env=env,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
