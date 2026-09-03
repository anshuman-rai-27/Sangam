"""
Downloads a model slice from the server with a progress bar.

Can also be used standalone:
    python -m worker.download --server http://192.168.1.x:8000 --room abc123
"""
import argparse
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import httpx


def join_room(server_url: str, room_id: str, device_id: str, ram_mb: int) -> dict:
    with httpx.Client() as client:
        resp = client.post(
            f"{server_url}/room/{room_id}/join",
            json={"device_id": device_id, "ram_mb": ram_mb},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


def download_slice(server_url: str, slice_id: int, dest_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    url = f"{server_url}/slice/{slice_id}"
    print(f"[download] fetching slice {slice_id} from {url}")

    with httpx.Client() as client:
        with client.stream("GET", url, timeout=300.0) as resp:
            resp.raise_for_status()
            total      = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded // total
                        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                        print(f"\r  [{bar}] {pct:3d}%  {downloaded//1024}KB/{total//1024}KB",
                              end="", flush=True)
            print()

    print(f"[download] saved -> {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Join a room and download its assigned slice")
    parser.add_argument("--server",    required=True, help="Server URL, e.g. http://192.168.1.x:8000")
    parser.add_argument("--room",      required=True, help="Room ID")
    parser.add_argument("--device-id", default=None,  help="Device ID (auto-generated if omitted)")
    parser.add_argument("--ram",       type=int, default=2048, help="Available RAM in MB")
    parser.add_argument("--out-dir",   default="model_slices", help="Directory to save slice")
    args = parser.parse_args()

    device_id  = args.device_id or f"device_{uuid.uuid4().hex[:6]}"
    server_url = args.server.rstrip("/")

    print(f"[download] joining room '{args.room}' as '{device_id}'")
    assignment = join_room(server_url, args.room, device_id, args.ram)
    slice_id   = assignment["slice_id"]
    layers     = assignment["layers"]
    print(f"[download] assigned slice {slice_id} (layers {layers[0]}-{layers[1]})")

    dest = os.path.join(args.out_dir, f"room_{args.room}_slice{slice_id}.pt")
    if os.path.isfile(dest):
        print(f"[download] already cached at {dest}")
    else:
        download_slice(server_url, slice_id, dest)

    print(f"[download] done. slice path: {dest}")


if __name__ == "__main__":
    main()
