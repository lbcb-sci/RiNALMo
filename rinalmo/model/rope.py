import torch
import torch.nn as nn

# Code heavily inspired by https://blog.eleuther.ai/rotary-embeddings/, GPT-NeoX (Pytorch) implementation
# and ESM2 implementation https://github.com/facebookresearch/esm/blob/main/esm/rotary_embedding.py

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: int = 10000
    ):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cached(self, x, seq_dim):
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k):
        self._update_cached(k, seq_dim=-2)
        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)
