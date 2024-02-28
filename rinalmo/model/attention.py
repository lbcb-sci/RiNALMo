import torch
from torch import nn

import math

from rinalmo.model.rope import RotaryPositionEmbedding

from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding

from flash_attn.bert_padding import unpad_input, pad_input

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

class FlashAttention(nn.Module):
    """
    Implement the scaled dot product attention with softmax.
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        """
        Args:
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            softmax_scale: float. The scaling of QK^T before applying softmax.
                           Default to 1 / sqrt(headdim).
            attention_dropout: float. The dropout rate to apply to the attention
                               (default: 0.0)
        """
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.causal = causal

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None, return_attn_probs=False):
        """
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                 If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                 If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                 (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                        of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
            return_attn_probs: bool. Whether to return the attention probabilities. This option is for
                               testing only. The returned probabilities are not guaranteed to be correct
                               (they might not have the right scaling).
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                 else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        unpadded = cu_seqlens is not None

        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_attn_probs=return_attn_probs
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                return_attn_probs=return_attn_probs
            )

class FlashMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented using FlashAttention.
    """
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0, causal=False, use_rot_emb=True, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimensionality must be divisible with number of attention heads!"

        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_dim = self.head_dim * num_heads * 3

        self.rotary_emb_dim = self.head_dim
        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryEmbedding(
                dim=self.rotary_emb_dim,
                base=10000.0,
                interleaved=False,
                scale_base=None,
                pos_idx_in_fp32=True,  # fp32 RoPE precision
                device=None
            )
        self.flash_self_attn = FlashAttention(causal=self.causal, softmax_scale=None, attention_dropout=attention_dropout)

        self.Wqkv = nn.Linear(self.embed_dim, self.qkv_dim, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def forward(self, x, key_padding_mask=None, return_attn_probs=False):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num_heads * head_dim)
            key_pad_mask: boolean mask, True means to keep, False means to mask out.
                          (batch, seqlen)
            return_attn_probs: whether to return attention masks (False by default)
        """

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.use_rot_emb:
            qkv = self.rotary_emb(qkv, seqlen_offset=0)

        if return_attn_probs:
            bs = qkv.shape[0]
            qkv = torch.permute(qkv, (0, 3, 2, 1, 4))
            q = qkv[:, :, 0, :, :]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]
            out, attn = dot_product_attention(q, k, v, key_pad_mask=torch.logical_not(key_padding_mask) if key_padding_mask is not None else None, dropout=self.attention_dropout)
            output = out.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.head_dim)
            output = self.out_proj(output)
            return output, attn

        if key_padding_mask is not None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            x_unpad, indices, cu_seqlens, max_s = unpad_input(qkv, key_padding_mask)
            output_unpad = self.flash_self_attn(
                    x_unpad, 
                    cu_seqlens=cu_seqlens, 
                    max_seqlen=max_s, 
                    return_attn_probs=return_attn_probs
                    )
            out = pad_input(rearrange(output_unpad, '... h d -> ... (h d)'), indices, batch_size, seqlen)
        else:
            output = self.flash_self_attn(
                    qkv, 
                    cu_seqlens=None, 
                    max_seqlen=None, 
                    return_attn_probs=return_attn_probs
                    )
            out = rearrange(output, '... h d -> ... (h d)')

        out = self.out_proj(out)
        return out, None
