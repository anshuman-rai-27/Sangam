"""
Convert server_head.pt (PyTorch) → server_head.npz (numpy).
Run once locally: python scripts/convert_head_to_npz.py
"""
import os
import numpy as np
import torch

SRC = os.path.join("model_slices", "server_head.pt")
DST = os.path.join("model_slices", "server_head.npz")

print(f"Loading {SRC} …")
data = torch.load(SRC, map_location="cpu", weights_only=False)

wte = data["wte"]["weight"].float().numpy()   # (50257, 768)
wpe = data["wpe"]["weight"].float().numpy()   # (1024,  768)

print(f"wte shape: {wte.shape}, wpe shape: {wpe.shape}")
np.savez_compressed(DST, wte=wte, wpe=wpe)

size_mb = os.path.getsize(DST) / 1e6
print(f"Saved {DST}  ({size_mb:.1f} MB)")
