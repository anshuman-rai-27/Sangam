"""
Split GPT-2 (117M) into 3 browser ONNX slices + server_head.npz.

Requires (locally only, not on server):
    pip install torch transformers

Produces in model_slices/gpt2/:
  slice_{0,1,2}.onnx   — browser ONNX workers (~112 MB each, float32)
  server_head.npz      — wte + wpe weights in float16 for server (~54 MB)

Usage:
    python -m splitter.split_gpt2
    python -m splitter.split_gpt2 --output model_slices/gpt2
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_ID   = "gpt2"
NUM_LAYERS = 12          # GPT-2 base has 12 transformer blocks
NUM_SLICES = 3
LAYERS_PER = NUM_LAYERS // NUM_SLICES   # 4 layers per slice
MAX_LEN    = 1024        # GPT-2 context window


# ─── Browser ONNX wrapper ─────────────────────────────────────────────────────

class _GPT2BrowserSlice(nn.Module):
    """
    Wraps GPT-2 decoder blocks for ONNX export.
    Single input: hidden_states (1, seq_len, 768).
    GPT-2 uses standard attention (no RoPE); positional embeddings
    are added server-side before the first slice.
    """

    def __init__(self, blocks: list, ln_f: nn.Module = None):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.ln_f   = ln_f

        causal = torch.triu(torch.full((MAX_LEN, MAX_LEN), float("-inf")), diagonal=1)
        self.register_buffer("_causal_bias", causal, persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len     = hidden_states.shape[1]
        causal_mask = self._causal_bias[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            # transformers ≥5.x returns a tensor; ≤4.x returns a tuple
            out = block(hidden_states, attention_mask=causal_mask, use_cache=False)
            hidden_states = out[0] if isinstance(out, tuple) else out

        if self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)

        return hidden_states


# ─── Main ─────────────────────────────────────────────────────────────────────

def split(output_dir: str = "model_slices/gpt2"):
    os.makedirs(output_dir, exist_ok=True)

    needed = [f"slice_{i}.onnx" for i in range(NUM_SLICES)] + ["server_head.npz"]
    if all(os.path.isfile(os.path.join(output_dir, f)) for f in needed):
        print(f"[splitter] all files present in '{output_dir}', skipping.")
        return

    print(f"Downloading {MODEL_ID}…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    transformer = model.transformer   # GPT2Model
    blocks      = transformer.h       # list of 12 GPT2Block

    print(f"  {NUM_LAYERS} layers  hidden={model.config.n_embd}  "
          f"vocab={model.config.vocab_size}")

    # ── ONNX slices ──────────────────────────────────────────────────────────
    print("Exporting ONNX slices…")
    dummy = torch.randn(1, 16, model.config.n_embd)

    for i in range(NUM_SLICES):
        start   = i * LAYERS_PER
        end     = (i + 1) * LAYERS_PER
        is_last = (i == NUM_SLICES - 1)
        ln_f    = transformer.ln_f if is_last else None

        wrapper   = _GPT2BrowserSlice([blocks[j] for j in range(start, end)],
                                      ln_f=ln_f).eval()
        onnx_path = os.path.join(output_dir, f"slice_{i}.onnx")

        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    (dummy,),
                    onnx_path,
                    input_names  = ["hidden_states"],
                    output_names = ["output"],
                    dynamic_axes = {"hidden_states": {1: "seq_len"}, "output": {1: "seq_len"}},
                    opset_version = 14,
                    dynamo        = False,
                )
            ext_path = onnx_path + ".data"
            if os.path.isfile(ext_path):
                print(f"  consolidating external data for slice_{i}…")
                model_proto = onnx.load(onnx_path)
                onnx.save(model_proto, onnx_path, save_as_external_data=False)
                os.remove(ext_path)

            mb = os.path.getsize(onnx_path) / 1e6
            print(f"  slice_{i}.onnx  {mb:.0f} MB  (layers {start}–{end-1}"
                  + (" + ln_f" if is_last else "") + ")")
        except Exception as exc:
            print(f"  [ERROR] ONNX export failed for slice {i}: {exc}")
            raise

    # ── server_head.npz ───────────────────────────────────────────────────────
    print("Saving server_head.npz (float16)…")
    wte_fp16  = transformer.wte.weight.detach().float().numpy().astype(np.float16)
    wpe_fp16  = transformer.wpe.weight.detach().float().numpy().astype(np.float16)
    head_path = os.path.join(output_dir, "server_head.npz")
    np.savez_compressed(head_path, wte=wte_fp16, wpe=wpe_fp16)
    mb = os.path.getsize(head_path) / 1e6
    print(f"  server_head.npz  {mb:.0f} MB  wte={wte_fp16.shape} wpe={wpe_fp16.shape} float16")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="model_slices/gpt2")
    args = parser.parse_args()
    split(args.output)
