import math
import torch
import torch.nn as nn
from x_transformers.x_transformers import (
    FixedPositionalEmbedding,
    AbsolutePositionalEmbedding,
)


class PatchyfiedSpecPosEmbs(nn.Module):
    def __init__(self, n_freq_patches, dim):
        super().__init__()
        self.n_freq_patches = n_freq_patches
        self.dim = dim
        self.pos_emb_freq = nn.Embedding(n_freq_patches, dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.d_sqrt = dim**-0.5

    def pos_emb_time(self, indices):
        div_term = torch.exp(
            torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(1, indices.shape[0], self.dim)
        pe[0, :, 0::2] = torch.sin(indices * div_term)
        pe[0, :, 1::2] = torch.cos(indices * div_term)
        return pe

    def forward(self, x, **kwargs):
        T, D = x.shape[-2:]
        n = self.n_freq_patches
        assert T % n == 0, "Number of frequency patches seems to be wrong"

        temp_pos = torch.tensor(sum([[i % n] * n for i in range(T // n)], [])).view(
            -1, 1
        )
        freq_pos = torch.tensor(
            sum([list(range(0, n)) for _ in range(T // n)], [])
        )  # .view(-1, 1)
        pe_temp = self.pos_emb_time(temp_pos).to(x.device).squeeze()
        pe_freq = self.pos_emb_freq(freq_pos.to(x.device))
        # pe_freq = self.pos_emb_time(freq_pos).to(x.device)
        # pos = (self.linear(pe_temp) + pe_freq) * self.d_sqrt
        pos = self.linear(torch.cat([pe_freq, pe_temp], -1)) * self.d_sqrt
        return pos
