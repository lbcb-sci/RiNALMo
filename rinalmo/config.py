import ml_collections as mlc
import copy

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import *

def model_config(name):
    c = copy.deepcopy(default_config)

    if name == "nano":
        c.globals.embed_dim = 320

        c.model.transformer.num_blocks = 6
        c.model.transformer.num_heads = 20
    elif name == "micro":
        c.globals.embed_dim = 480

        c.model.transformer.num_blocks = 12
        c.model.transformer.num_heads = 20
    elif name == "mega":
        c.globals.embed_dim = 640

        c.model.transformer.num_blocks = 30
        c.model.transformer.num_heads = 20
    elif name == "giga":
        c.globals.embed_dim = 1280

        c.model.transformer.num_blocks = 33
        c.model.transformer.num_heads = 20

        c.training.optimizer.lr = 5e-5
        c.training.lr_scheduler.cosine_decay.eta_min = 1e-5
    else:
        raise ValueError("Invalid configuration name!")
    
    assert not any_tokenizer_discrepancies(c), "Found discrepancies in tokenizer configuration!"

    return c

def any_tokenizer_discrepancies(config):
    alphabet = Alphabet(**config['alphabet'])

    if alphabet.get_idx(MASK_TKN) != config['globals'].mask_tkn_idx:
        return True
    
    if alphabet.get_idx(PAD_TKN) != config['globals'].pad_tkn_idx:
        return True
    
    if len(alphabet) != config['globals'].alphabet_size:
        return True
    
    return False

embed_dim = mlc.FieldReference(480, field_type=int)

default_alphabet = Alphabet()
alphabet_size = mlc.FieldReference(len(default_alphabet), field_type=int)
mask_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(MASK_TKN), field_type=int)
pad_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(PAD_TKN), field_type=int)

mask_ratio = mlc.FieldReference(0.15, field_type=float)
mask_tkn_prob = mlc.FieldReference(0.8, field_type=float)

default_config = mlc.ConfigDict(
    {
        "globals": {
            "embed_dim": embed_dim,
            "alphabet_size": alphabet_size,
            "mask_tkn_idx": mask_tkn_idx,
            "pad_tkn_idx": pad_tkn_idx,
            "mask_ratio": mask_ratio,
            "mask_tkn_prob": mask_tkn_prob,
        },
        "alphabet": {
            "standard_tkns": RNA_TOKENS,
            "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
        },
        "training": {
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "warm_up": {
                    "iters": 2000,
                },
                "cosine_decay": {
                    "T_max": 200_000,
                    "eta_min": 1e-5,
                },
            },
            "masking": {
                "bert_masking": {
                    "mask_ratio": mask_ratio,
                    "mask_tkn_prob": mask_tkn_prob,
                    "random_tkn_prob": 0.1,
                }
            },
        },
        "model": {
            "embedding": {
                "num_embeddings": alphabet_size,
                "embedding_dim": embed_dim,
                "padding_idx": pad_tkn_idx,
            },
            "token_dropout": {
                "active": True,
                "mask_ratio": mask_ratio,
                "mask_tkn_prob": mask_tkn_prob,
                "mask_tkn_idx": mask_tkn_idx,
                "pad_tkn_idx": pad_tkn_idx,
            },
            "transformer": {
                "embed_dim": embed_dim,
                "num_blocks": 12,
                "num_heads": 20,
                "use_rot_emb": True,
                "attn_qkv_bias": False,
                "attention_dropout": 0.1,
                "transition_dropout": 0.0,
                "residual_dropout": 0.1,
                "transition_factor": 4,
                "use_flash_attn": True,
            },
            "lm_mask_head": {
                "embed_dim": embed_dim,
                "alphabet_size": alphabet_size,
            }
        }
    }
)
