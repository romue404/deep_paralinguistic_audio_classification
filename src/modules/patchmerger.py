import torch
import torch.nn as nn


class PatchMerger(nn.Module):
    def __init__(self, dim, m=16):
        super(PatchMerger, self).__init__()
        self.mem_tokens = nn.Linear(dim, m, bias=False)
        self.scale = nn.Parameter(torch.ones(m))
        self.d_sqrt = dim ** -0.5

    def make_mask(self, lengths):
        max_len = lengths.max().item()
        return torch.arange(max_len).expand(len(lengths), max_len).to(
            lengths.device
        ) < (lengths).unsqueeze(1)

    def forward(self, x, mask_or_lengths=None):
        raw_scores = self.mem_tokens(x) * self.scale * self.d_sqrt
        if mask_or_lengths is not None:
            if len(mask_or_lengths.shape) == 1:
                mask_or_lengths = self.make_mask(mask_or_lengths)
            raw_scores[~mask_or_lengths] = -torch.inf
        attn_scores = torch.softmax(raw_scores, dim=1)
        x_ = x.transpose(-1, -2)  # b x d x n
        out = (x_ @ attn_scores).transpose(-1, -2)
        return out