import torch
from torch import nn

import math

from rinalmo.model.rope import RotaryPositionEmbedding

import torch.nn.functional as F

from einops import rearrange

def dot_product_attention(q, k, v, attn_mask=None, key_pad_mask=None, dropout=None):
    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float("-inf"))

    if key_pad_mask is not None:
        attn = attn.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    attn = attn.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()
        assert c_in % num_heads == 0, "Embedding dimensionality must be divisible with number of attention heads!"

        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.to_q = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_k = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_v = nn.Linear(self.c_in, self.c_qkv, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, q, k, v, attn_mask=None, key_pad_mask=None):
        bs = q.shape[0]

        q = self.to_q(q).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        k = self.to_k(k).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(v).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        output, attn = dot_product_attention(q, k, v, attn_mask, key_pad_mask, self.attention_dropout)

        output = output.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.c_head)
        output = self.out_proj(output)

        return output, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()

        self.mh_attn = MultiHeadAttention(c_in, num_heads, attention_dropout, use_rot_emb, bias)

    def forward(self, x, attn_mask=None, key_pad_mask=None):
        return self.mh_attn(x, x, x, attn_mask, key_pad_mask)


class FlashMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, causal=False, use_rot_emb=True, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_dim = self.head_dim * num_heads * 3
        self.causal = causal
        self.use_rot_emb = use_rot_emb

        self.Wqkv = nn.Linear(embed_dim, self.qkv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attention_dropout = nn.Dropout(attention_dropout)

        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None, return_attn_probs=False):
        """
        x: (batch, seqlen, embed_dim)
        key_padding_mask: (batch, seqlen) - False means masked
        """
        B, S, _ = x.shape

        qkv = self.Wqkv(x)  # (B, S, 3 * H * D)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        qkv = qkv.permute(0, 3, 2, 1, 4)

        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)
        attn_scores /= self.head_dim ** 0.5

        if key_padding_mask is not None:
            # key_padding_mask: (B, S) → (B, 1, 1, S)
            mask = ~key_padding_mask[:, None, None, :].to(torch.bool)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        if self.causal:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attention_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, H, S, D)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        out = self.out_proj(attn_output)

        if return_attn_probs:
            return out, attn_probs
        return out, None
