"""
Split Qwen2.5-0.5B-Instruct into 3 browser ONNX slices + server_head.npz.

Requires (locally only, not on server):
    pip install torch transformers

Produces in model_slices/:
  slice_{0,1,2}.onnx   — browser ONNX workers (~450-480 MB each, float32)
  server_head.npz      — embed_tokens weights in float16 for server (~272 MB)

Usage:
    python -m splitter.split_model
    python -m splitter.split_model --output model_slices
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS  = 24          # Qwen2.5-0.5B has 24 transformer layers
NUM_SLICES  = 3
LAYERS_PER  = NUM_LAYERS // NUM_SLICES   # 8 layers per slice
MAX_LEN     = 2048        # max sequence length for causal mask buffer


# ─── Browser ONNX wrapper ─────────────────────────────────────────────────────

class _BrowserSlice(nn.Module):
    """
    Wraps Qwen2.5 decoder layers for ONNX export.
    Single input: hidden_states (1, seq_len, 896).

    Transformers 5.x requires pre-computed RoPE embeddings (position_embeddings)
    passed to each layer — the attention no longer computes them from position_ids.
    Layer forward now returns a plain Tensor (not a tuple).
    """

    def __init__(self, layers: list, rotary_emb: nn.Module, final_norm: nn.Module = None):
        super().__init__()
        self.layers     = nn.ModuleList(layers)
        self.rotary_emb = rotary_emb   # Qwen2RotaryEmbedding shared across all slices
        self.final_norm = final_norm

        causal = torch.triu(torch.full((MAX_LEN, MAX_LEN), float("-inf")), diagonal=1)
        self.register_buffer("_causal_bias", causal, persistent=True)
        self.register_buffer("_positions", torch.arange(MAX_LEN).unsqueeze(0), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len      = hidden_states.shape[1]
        position_ids = self._positions[:, :seq_len]
        causal_mask  = self._causal_bias[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        cos, sin     = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=(cos, sin),
                use_cache=False,
            )

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


# ─── Main ─────────────────────────────────────────────────────────────────────

def split(output_dir: str = "model_slices"):
    os.makedirs(output_dir, exist_ok=True)

    needed = [f"slice_{i}.onnx" for i in range(NUM_SLICES)] + ["server_head.npz"]
    if all(os.path.isfile(os.path.join(output_dir, f)) for f in needed):
        print(f"[splitter] all files present in '{output_dir}', skipping.")
        return

    print(f"Downloading {MODEL_ID} (eager attention for ONNX compatibility)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",   # avoid F.scaled_dot_product_attention (ONNX unfriendly)
    )
    model.eval()
    transformer = model.model    # Qwen2Model
    layers      = transformer.layers

    print(f"  {NUM_LAYERS} layers  hidden={model.config.hidden_size}  "
          f"vocab={model.config.vocab_size}")

    # ── ONNX slices ──────────────────────────────────────────────────────────
    print("Exporting ONNX slices…")
    dummy = torch.randn(1, 16, model.config.hidden_size)

    for i in range(NUM_SLICES):
        start      = i * LAYERS_PER
        end        = (i + 1) * LAYERS_PER
        is_last    = (i == NUM_SLICES - 1)
        final_norm = transformer.norm if is_last else None

        wrapper   = _BrowserSlice([layers[j] for j in range(start, end)],
                                  rotary_emb=transformer.rotary_emb,
                                  final_norm=final_norm).eval()
        onnx_path = os.path.join(output_dir, f"slice_{i}.onnx")

        try:
            # PyTorch 2.9+ new torch.export-based ONNX exporter (no TorchScript tracing)
            seq_dim = torch.export.Dim("seq_len", min=1, max=MAX_LEN)
            torch.onnx.export(
                wrapper,
                (dummy,),
                onnx_path,
                input_names    = ["hidden_states"],
                output_names   = ["output"],
                dynamic_shapes = {"hidden_states": {1: seq_dim}},
                opset_version  = 17,
            )
            # Merge external data file (slice_N.onnx.data) into the .onnx
            ext_path = onnx_path + ".data"
            if os.path.isfile(ext_path):
                print(f"  consolidating external data for slice_{i}…")
                model_proto = onnx.load(onnx_path)   # loads graph + external data into RAM
                onnx.save(model_proto, onnx_path, save_as_external_data=False)
                os.remove(ext_path)

            mb = os.path.getsize(onnx_path) / 1e6
            print(f"  slice_{i}.onnx  {mb:.0f} MB  (layers {start}–{end-1}"
                  + (" + norm" if is_last else "") + ")")
        except Exception as exc:
            print(f"  [ERROR] ONNX export failed for slice {i}: {exc}")
            raise

    # ── server_head.npz ───────────────────────────────────────────────────────
    # Store embed_tokens as float16 to fit Render 512 MB RAM limit.
    # lm_head shares the same weights (tie_word_embeddings=True).
    print("Saving server_head.npz (float16)…")
    wte_fp16  = transformer.embed_tokens.weight.detach().float().numpy().astype(np.float16)
    head_path = os.path.join(output_dir, "server_head.npz")
    np.savez_compressed(head_path, wte=wte_fp16)
    mb = os.path.getsize(head_path) / 1e6
    print(f"  server_head.npz  {mb:.0f} MB  wte={wte_fp16.shape} float16")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="model_slices")
    args = parser.parse_args()
    split(args.output)
