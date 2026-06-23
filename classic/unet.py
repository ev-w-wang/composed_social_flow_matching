import torch
import torch.nn as nn
import math
import numpy as np

from classic.helpers import SinusoidalEmbed, Conv1dBlock, Downsample1d, Upsample1d, LayerNorm, PreNorm, LinearAttention, Residual

class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t).unsqueeze(-1)
        out = self.blocks[1](out) + self.residual_conv(x)
        return out

class TempUNet(nn.Module):
    def __init__(
        self,
        horizon,
        out_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        dims = [out_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalEmbed(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim),
        )


        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, out_dim, 1),
        )

    def forward(self, x, cond, time):
        emb = self.time_mlp(time) + self.cond_emb(cond)
        h = []
        x = x.permute(0, 2, 1)
        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x

