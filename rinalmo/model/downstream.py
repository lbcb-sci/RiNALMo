import torch
from torch import nn

from rinalmo.model.modules import Transformer

def _outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class SecStructPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_blocks, conv_dim=64, kernel_size=3):
        super().__init__()

        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        
    def forward(self, x):
        x = _outer_concat(x, x) # B x L x F => B x L x L x 2F

        x = self.linear_in(x)
        x = x.permute(0, 3, 1, 2) # B x L x L x E  => B x E x L x L

        x = self.resnet(x)
        x = self.conv_out(x)
        x = x.squeeze(-3) # B x 1 x L x L => B x L x L

        # Symmetrize the output
        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)

        return x

class RibonanzaPredictionHead(nn.Module):
    def __init__(self, c_in, embed_dim, num_blocks, num_attn_heads):
        super().__init__()

        self.linear_in = nn.Linear(c_in, embed_dim)
        self.transformer = Transformer(embed_dim=embed_dim, num_blocks=num_blocks, num_heads=num_attn_heads)
        self.linear_out = nn.Linear(embed_dim, 2)

    def forward(self, x, padding_mask=None):
        x = self.linear_in(x)
        x, _ = self.transformer(x, key_padding_mask=padding_mask)
        x = self.linear_out(x)

        return x

class ResNet1DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, stride=1, bias=False):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding="same"),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        residual = x
        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet1D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet1DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class RibosomeLoadingPredictionHead(nn.Module):
    def __init__(self, c_in, embed_dim, num_blocks, dropout=0.2):
        super().__init__()

        self.linear_in = nn.Linear(c_in, embed_dim)

        self.resnet = ResNet1D(embed_dim, num_blocks)
        self.dropout = nn.Dropout(p=dropout)

        self.linear_out = nn.Linear(embed_dim, 1)

    def forward(self, x, padding_mask=None):
        x = self.linear_in(x)

        x = x.permute(0, 2, 1) # B x L x E => B x E x L
        x = self.resnet(x)
        x = x.permute(0, 2, 1) # B x E x L => B x L x E

        # Global pooling (B x L x E => B x E)
        if padding_mask is not None:
            x[padding_mask, :] = 0.0
            x = x.sum(dim=-2) / (~padding_mask).sum(dim=-1)[:, None]
        else:
            x = x.mean(dim=-2)

        x = self.dropout(x)
        x = self.linear_out(x).squeeze(-1)

        return x

class SpliceSitePredictionHead(nn.Module):
    def __init__(self, c_in, embed_dim) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(c_in, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
