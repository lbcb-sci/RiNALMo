import torch
from torch import nn

from rinalmo.model.modules import Transformer, MaskedLanguageModelHead, TokenDropout

class RiNALMo(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_tkn_idx = self.config.model.embedding['padding_idx']

        self.embedding = nn.Embedding(**self.config.model['embedding'])
        self.transformer = Transformer(**self.config.model['transformer'])

        self.lm_mask_head = MaskedLanguageModelHead(**self.config.model['lm_mask_head'])

        self.token_dropout = TokenDropout(**self.config.model['token_dropout'])

    def forward(self, tokens, need_attn_weights=False):
        pad_mask = tokens.eq(self.pad_tkn_idx)
        x = self.embedding(tokens)
        x = self.token_dropout(x, tokens)

        if self.config.model.transformer.use_flash_attn:
            representation, attn_weights = self.transformer(
                x,
                key_padding_mask=torch.logical_not(pad_mask) if pad_mask is not None else None,
                need_attn_weights=need_attn_weights
                )
        else:
            representation, attn_weights = self.transformer(x, key_padding_mask=pad_mask, need_attn_weights=need_attn_weights)
        x = self.lm_mask_head(representation)

        result = {"logits": x, "representation": representation}
        if need_attn_weights:
            result["attentions"] = torch.stack(attn_weights, dim=1)
        
        return result
