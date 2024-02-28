import torch
from torch import nn
from torch.nn import functional as F

from rinalmo.model.attention import MultiHeadSelfAttention, FlashMultiHeadSelfAttention

import torch.utils.checkpoint as checkpoint

class TokenDropout(nn.Module):
    def __init__(
        self,
        active: bool,
        mask_ratio: float,
        mask_tkn_prob: float,
        mask_tkn_idx: int,
        pad_tkn_idx: int,
    ):
        super().__init__()

        self.active = active

        self.mask_ratio_train = mask_ratio * mask_tkn_prob

        self.mask_tkn_idx = mask_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx

    def forward(self, x, tokens):
        if self.active:
            pad_mask = tokens.eq(self.pad_tkn_idx)
            src_lens = (~pad_mask).sum(dim=-1)

            x = torch.where((tokens == self.mask_tkn_idx).unsqueeze(dim=-1), 0.0, x)
            mask_ratio_observed = (tokens == self.mask_tkn_idx).sum(dim=-1) / src_lens
            x = x * (1 - self.mask_ratio_train) / (1 - mask_ratio_observed[..., None, None])

        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, use_rot_emb, attn_qkv_bias, transition_dropout, attention_dropout, residual_dropout, transition_factor, use_flash_attn) for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, need_attn_weights=False):
        attn_weights = None
        if need_attn_weights:
            attn_weights = []

        for block in self.blocks:
            x, attn = checkpoint.checkpoint(
                block, 
                x,
                key_padding_mask=key_padding_mask,
                need_attn_weights=need_attn_weights,
                use_reentrant=False
                )

            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)

        return x, attn_weights

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    In the cited paper beta is set to 1 and is not learnable;
    but by the Swish definition it is learnable parameter otherwise
    it is SiLU activation function (https://paperswithcode.com/method/swish)
    """
    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        """
        Args:
            size_in: input embedding dimension
            size_out: output embedding dimension
            beta_is_learnable: whether beta is learnable or set to 1, learnable by default
            bias: whether use bias term, enabled by default
        """
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)  

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.mh_attn = FlashMultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, causal=False, use_rot_emb=use_rot_emb, bias=attn_qkv_bias)
        else:
            self.mh_attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, use_rot_emb, attn_qkv_bias)
        
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.transition = nn.Sequential(
                SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
                nn.Dropout(p=transition_dropout),
                nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)

        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x, key_padding_mask=None, need_attn_weights=None):
        x = self.attn_layer_norm(x)
        if self.use_flash_attn:
            mh_out, attn = self.mh_attn(x, key_padding_mask=key_padding_mask, return_attn_probs=need_attn_weights)
        else:
            mh_out, attn = self.mh_attn(x, attn_mask=None, key_pad_mask=key_padding_mask)
        x = x + self.residual_dropout_1(mh_out)

        residual = x
        x = self.out_layer_norm(x)
        x = residual + self.residual_dropout_2(self.transition(x))

        return x, attn

class MaskedLanguageModelHead(nn.Module):
    def __init__(self, embed_dim, alphabet_size):
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, alphabet_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.linear2(x)

        return x
