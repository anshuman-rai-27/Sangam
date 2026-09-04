"""
Upload model slices to HuggingFace Hub.

Usage:
    # Qwen2.5
    python scripts/upload_to_hf.py --repo your-username/sangam-qwen-slices

    # GPT-2
    python scripts/upload_to_hf.py --repo your-username/sangam-gpt2-slices --dir model_slices/gpt2
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Load .env from repo root if present
_env = Path(__file__).parent.parent / ".env"
if _env.is_file():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

FILES = [
    "slice_0.onnx",
    "slice_1.onnx",
    "slice_2.onnx",
    "server_head.npz",
]

def upload(repo_id: str, slices_dir: str):
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    print(f"Repo: https://huggingface.co/{repo_id}")
    print(f"Dir:  {slices_dir}")

    for fname in FILES:
        path = os.path.join(slices_dir, fname)
        if not os.path.isfile(path):
            print(f"  SKIP {fname} (not found)")
            continue
        size_mb = os.path.getsize(path) / 1e6
        print(f"  uploading {fname} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"  done: {fname}")

    print(f"\nAll done. CDN base URL:")
    print(f"  https://huggingface.co/{repo_id}/resolve/main/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. your-username/sangam-gpt2-slices")
    parser.add_argument("--dir", default="model_slices", help="local directory containing slice files")
    args = parser.parse_args()
    upload(args.repo, args.dir)
