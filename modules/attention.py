import torch
import torch.nn as nn
from x_transformers.x_transformers import Attention, Rezero


class AttnFn(nn.Module):
    def __init__(self):
        super(AttnFn, self).__init__()
        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, dots, dim=-1):
        r, c = dots.shape[-2], dots.shape[-1]
        mask_value = -torch.finfo(dots.dtype).max
        num_mem_k = c - r

        mask = torch.zeros(r, c, device=dots.device)
        i, j = torch.arange(r - 1, -1, -1), torch.arange(c - 1, num_mem_k - 1, -1)
        mask[i, j] = 1

        dots = dots * self.scale

        dots = dots.masked_fill(mask.bool(), mask_value)

        return dots.softmax(dim=dim)


def attn_fn(dots, dim=-1):
    r, c = dots.shape[-2], dots.shape[-1]
    mask_value = -torch.finfo(dots.dtype).max
    num_mem_k = c - r

    mask = torch.zeros(r, c, device=dots.device)
    i, j = torch.arange(r-1, -1, -1), torch.arange(c-1, num_mem_k-1, -1)
    mask[i, j] = 1
    dots = dots.masked_fill(mask.bool(), mask_value)

    return dots.softmax(dim=dim)


class MyAttention(Attention):
    def __init__(self, *args, **kwargs):
        super(MyAttention, self).__init__(*args, **kwargs)
        self.attn_fn = attn_fn