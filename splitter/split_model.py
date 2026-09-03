"""
Run once to download GPT-2 Small and split into slice files.

Produces:
  slice_{0,1,2}.pt      — for Python workers (existing format)
  slice_{0,1,2}.onnx    — for browser workers (~112 MB each, blocks only)
  server_head.pt        — wte + wpe + lm_head for server-side embedding/projection
  tokenizer/            — GPT-2 tokenizer

Usage:
    python -m splitter.split_model
    python -m splitter.split_model --output model_slices
"""
import argparse
import os

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

TOTAL_LAYERS = 12

SLICES = [
    {"name": "slice_0", "layer_range": [0, 4],  "is_first": True,  "is_last": False},
    {"name": "slice_1", "layer_range": [4, 8],  "is_first": False, "is_last": False},
    {"name": "slice_2", "layer_range": [8, 12], "is_first": False, "is_last": True},
]


# ─── Browser ONNX wrapper ─────────────────────────────────────────────────────

class _BrowserSlice(nn.Module):
    """
    Uniform interface for all 3 browser slices: float32 hidden_states in/out.
    The server handles embedding (wte+wpe) and LM-head projection.
    Slice 2 additionally applies the final layer-norm (ln_f) before returning.
    """
    def __init__(self, blocks: list, ln_f: nn.Module = None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln_f   = ln_f

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            out = blk(hidden_states, use_cache=False)
            # transformers 5.x returns tensor directly; 4.x returns tuple
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
        return hidden_states


# ─── Main ─────────────────────────────────────────────────────────────────────

def split(output_dir: str = "model_slices"):
    os.makedirs(output_dir, exist_ok=True)

    # Skip if all outputs already present (Render persistent disk reuse)
    needed = [f"slice_{i}.pt" for i in range(3)] + \
             [f"slice_{i}.onnx" for i in range(3)] + \
             ["server_head.pt"]
    if all(os.path.isfile(os.path.join(output_dir, f)) for f in needed):
        print(f"[splitter] all model files found in '{output_dir}', skipping.")
        return

    print("Downloading / loading GPT-2 Small (117M)...")
    model  = GPT2LMHeadModel.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    model.eval()
    gpt2 = model.transformer

    # ── .pt slices for Python workers (existing format) ──────────────────────
    for s in SLICES:
        start, end = s["layer_range"]
        payload = {
            "config":      config,
            "layer_range": s["layer_range"],
            "is_first":    s["is_first"],
            "is_last":     s["is_last"],
            "blocks":      {i: gpt2.h[i].state_dict() for i in range(start, end)},
        }
        if s["is_first"]:
            payload["wte"] = gpt2.wte.state_dict()
            payload["wpe"] = gpt2.wpe.state_dict()
        if s["is_last"]:
            payload["ln_f"]    = gpt2.ln_f.state_dict()
            payload["lm_head"] = model.lm_head.state_dict()

        path = os.path.join(output_dir, f"{s['name']}.pt")
        torch.save(payload, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  saved {path}  ({size_mb:.1f} MB)  layers {start}-{end-1}")

    # ── server_head.pt (wte + wpe; lm_head shares wte weights) ──────────────
    head_path = os.path.join(output_dir, "server_head.pt")
    torch.save(
        {"wte": gpt2.wte.state_dict(), "wpe": gpt2.wpe.state_dict()},
        head_path,
    )
    size_mb = os.path.getsize(head_path) / 1e6
    print(f"  saved {head_path}  ({size_mb:.1f} MB)  [wte + wpe for server]")

    # ── ONNX slices for browser workers ──────────────────────────────────────
    print("Exporting ONNX slices for browser workers...")
    try:
        _export_onnx(model, gpt2, output_dir)
    except Exception as exc:
        print(f"  [WARNING] ONNX export failed: {exc}")
        print("  Server will still work with Python workers.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_dir = os.path.join(output_dir, "tokenizer")
    GPT2Tokenizer.from_pretrained("gpt2").save_pretrained(tok_dir)
    print(f"  saved tokenizer -> {tok_dir}")
    print("Done.")


@torch.no_grad()
def _export_onnx(model: GPT2LMHeadModel, gpt2, output_dir: str):
    dummy_h = torch.randn(1, 16, 768)

    for s in SLICES:
        start, end  = s["layer_range"]
        blocks      = [gpt2.h[i] for i in range(start, end)]
        ln_f        = gpt2.ln_f if s["is_last"] else None
        wrapper     = _BrowserSlice(blocks, ln_f).eval()
        onnx_path   = os.path.join(output_dir, f"{s['name']}.onnx")

        torch.onnx.export(
            wrapper,
            (dummy_h,),
            onnx_path,
            input_names=["hidden_states"],
            output_names=["output"],
            dynamic_axes={
                "hidden_states": {1: "seq_len"},
                "output":        {1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  exported {onnx_path}  ({size_mb:.1f} MB)  layers {start}-{end-1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="model_slices")
    args = parser.parse_args()
    split(args.output)
