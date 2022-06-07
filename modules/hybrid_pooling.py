import torch
import torch.nn as nn
from modules.patchmerger import PatchMerger


class HPMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(HPMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class HybridPooler(nn.Module):
    def __init__(self, hidden_size, dropout=0.2, f_patchmerger=-1):
        super(HybridPooler, self).__init__()
        self.hidden_size = hidden_size
        self.f_patchmerger = f_patchmerger
        self.pmp = PatchMerger(self.hidden_size, 2)
        self.mlp1 = HPMLP(3*self.hidden_size, self.hidden_size, dropout)
        self.mlp2 = HPMLP(3*self.hidden_size, self.hidden_size, dropout)

    def forward(self, tokens, lengths):
        tokens, clf_pooled = tokens[:, 1:], tokens[:, 0]
        # tokens = torch.fft.fft2(tokens, dim=(-1, -2)).real
        if self.f_patchmerger > 0:
            lengths = (lengths / self.f_patchmerger).long()

        mean_pooled = torch.cat([torch.mean(i[0:l], dim=0).view(1, -1)
                                 for i, l in zip(tokens, lengths)], dim=0)
        max_pooled = torch.cat([torch.max(i[0:l], dim=0)[0].view(1, -1)
                                for i, l in zip(tokens, lengths)], dim=0)
        min_pooled = torch.cat([torch.min(i[0:l], dim=0)[0].view(1, -1)
                                for i, l in zip(tokens, lengths)], dim=0)
        pmp_pooled = self.pmp(tokens, lengths)

        pooled_traditional = torch.cat([mean_pooled, max_pooled, min_pooled], -1)
        pooled_learned = torch.cat([pmp_pooled.flatten(1), clf_pooled], dim=-1)

        pooled = torch.cat([self.mlp1(pooled_traditional), self.mlp2(pooled_learned)], -1)
        return pooled
