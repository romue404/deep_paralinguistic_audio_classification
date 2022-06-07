import torch
import torch.nn as nn
from src.modules.patchmerger import PatchMerger


class HPMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(HPMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    @property
    def out_features(self):
        return self.net[-2].out_features

    def forward(self, x):
        return self.net(x)


class HybridPooler(nn.Module):
    def __init__(self, hidden_size, num_patchmerges, dropout, how):
        super(HybridPooler, self).__init__()
        assert how in ['both', 'fixed', 'learned']
        self.how = how
        self.hidden_size = hidden_size
        self.num_patchmerges = num_patchmerges
        self.pmp = PatchMerger(self.hidden_size, num_patchmerges)
        self.mlp1 = HPMLP(3 * self.hidden_size, self.hidden_size, dropout) \
            if how in ['both', 'fixed'] else nn.Identity()
        self.mlp2 = HPMLP((1 + num_patchmerges) * self.hidden_size, self.hidden_size, dropout) \
            if how in ['both', 'learned'] else nn.Identity()

    def fixed_path(self, tokens, lengths):
        tokens, clf_pooled = tokens[:, 1:], tokens[:, 0]
        # tokens = torch.fft.fft2(tokens, dim=(-1, -2)).real
        mean_pooled = torch.cat(
            [torch.mean(i[0:l], dim=0).view(1, -1) for i, l in zip(tokens, lengths)],
            dim=0,
        )
        max_pooled = torch.cat(
            [torch.max(i[0:l], dim=0)[0].view(1, -1) for i, l in zip(tokens, lengths)],
            dim=0,
        )
        min_pooled = torch.cat(
            [torch.min(i[0:l], dim=0)[0].view(1, -1) for i, l in zip(tokens, lengths)],
            dim=0,
        )
        pooled_traditional = torch.cat([mean_pooled, max_pooled, min_pooled], -1)
        return self.mlp1(pooled_traditional)

    def learned_path(self, tokens, lengths):
        tokens, clf_pooled = tokens[:, 1:], tokens[:, 0]
        pmp_pooled = self.pmp(tokens, lengths)
        pooled_learned = torch.cat([pmp_pooled.flatten(1), clf_pooled], dim=-1)
        return self.mlp2(pooled_learned)

    def forward(self, tokens, lengths):
        pooled_traditional = self.fixed_path(tokens, lengths)
        pooled_learned = self.learned_path(tokens, lengths)
        if self.how == 'learned':
            return pooled_learned
        elif self.how == 'fixed':
            return pooled_traditional
        return torch.cat([pooled_traditional, pooled_learned], -1)

    @property
    def out_features(self):
        d = 0
        for mlp in [self.mlp1, self.mlp2]:
            try:
                d += mlp.out_features
            except AttributeError:
                pass
        return d
