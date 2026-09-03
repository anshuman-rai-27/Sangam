"""
Loads one .pt slice produced by splitter/split_model.py and exposes
forward_first / forward_middle / forward_last depending on position in pipeline.
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2Slice(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_first: bool = False
        self.is_last: bool = False
        self.layer_start: int = 0
        self.layer_end: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, slice_path: str) -> None:
        data = torch.load(slice_path, map_location="cpu", weights_only=False)
        config = data["config"]
        self.layer_start, self.layer_end = data["layer_range"]
        self.is_first = data["is_first"]
        self.is_last = data["is_last"]

        # Instantiate a full model so transformers 5.x wires up all internal
        # state (including _attn_implementation) before we extract our layers.
        full = GPT2LMHeadModel(config)
        gpt2 = full.transformer

        if self.is_first:
            self.wte = gpt2.wte
            self.wpe = gpt2.wpe
            self.drop = gpt2.drop
            self.wte.load_state_dict(data["wte"])
            self.wpe.load_state_dict(data["wpe"])

        # Pull out only the blocks we own, loading saved weights into them.
        self.blocks = nn.ModuleList()
        for orig_idx in range(self.layer_start, self.layer_end):
            block = gpt2.h[orig_idx]
            block.load_state_dict(data["blocks"][orig_idx])
            self.blocks.append(block)

        if self.is_last:
            self.ln_f = gpt2.ln_f
            self.lm_head = full.lm_head
            self.ln_f.load_state_dict(data["ln_f"])
            self.lm_head.load_state_dict(data["lm_head"])

        self.eval()
        print(f"[slice] loaded layers {self.layer_start}-{self.layer_end - 1}  "
              f"first={self.is_first}  last={self.is_last}")

    # ------------------------------------------------------------------
    # Forward helpers (all @no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_first(self, input_ids: list) -> torch.Tensor:
        ids = torch.tensor([input_ids], dtype=torch.long)
        seq_len = ids.shape[-1]
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        hidden = self.wte(ids) + self.wpe(pos)
        hidden = self.drop(hidden)
        for block in self.blocks:
            hidden = block(hidden)   # transformers 5.x returns tensor directly
        return hidden  # (1, seq_len, 768)

    @torch.no_grad()
    def forward_middle(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden  # (1, seq_len, 768)

    @torch.no_grad()
    def forward_last(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)  # (1, seq_len, vocab_size)
