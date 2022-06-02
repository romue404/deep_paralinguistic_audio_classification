import torch
import torch.nn as nn
import einops


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


# todo currently assumes clf token!
class FrequencyPatchMerger(nn.Module):
    def __init__(self, n_freq_patches: int, dim: int):
        super(FrequencyPatchMerger, self).__init__()
        self.n = n_freq_patches
        self.dim = dim
        self.projector = nn.Linear(dim, 1, bias=True)
        self.scale = nn.Parameter(torch.ones(1))
        self.d_sqrt = dim ** -0.5

    def forward(self, x, mask):
        x_ = einops.rearrange(x[:, 1:], "b (t n) d -> b t n d", n=self.n)
        scores = torch.softmax(self.projector(x_) * self.d_sqrt * self.scale, -2)

        mask_ = einops.rearrange(mask[:, 1:], "b (t n) -> b t n", n=self.n).prod(-1)

        x_ = torch.cat((x[:, 0, None], x_), 1)
        mask_ = torch.cat((mask[:, 0, None], mask_), 1).bool()

        return x_, mask_
