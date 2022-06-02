import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class MelSpec(Dataset):
    def __init__(self, path, label, transform=None):
        self.path = path
        self.transform = transform
        self.label = label

    def __getitem__(self, index: int):
        x = torch.from_numpy(np.load(file=self.path)).float()
        x = x.squeeze().unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.label, Path(self.path).stem

    def __len__(self) -> int:
        return 1


class RepeatLastSpectrogramFrame(nn.Module):
    def __init__(self, height, width, v_stride, h_stride):
        super(RepeatLastSpectrogramFrame, self).__init__()
        self.height = height
        self.width = width
        self.v_stride = v_stride
        self.h_stride = h_stride

    def forward(self, x):
        F, T = x.shape[-2:]

        n_v = (F - self.height) / self.v_stride
        n_h = (T - self.width) / self.h_stride

        pad_v = math.ceil(n_v) * self.v_stride - int(n_v * self.v_stride)
        pad_h = math.ceil(n_h) * self.h_stride - int(n_h * self.h_stride)

        if pad_h > 0:
            s = [x.select(-1, -1).unsqueeze(-1) for _ in range(pad_h)]
            x = torch.cat([x] + s, -1)
        if pad_v > 0:
            s = [x.select(-2, 0).unsqueeze(-2) for _ in range(pad_v)]
            x = torch.cat([x] + s, -2)
        return x


class PatchifySpectrogram(nn.Module):
    def __init__(self, height, width, v_stride, h_stride):
        super(PatchifySpectrogram, self).__init__()
        assert v_stride <= height and h_stride <= width, "Dialation not supported"
        self.repeat_frame = RepeatLastSpectrogramFrame(
            height, width, v_stride, h_stride
        )
        # self.unfold = nn.Unfold(kernel_size=(height, width), stride=(v_stride, h_stride))
        self.unfoldT = nn.Unfold(
            kernel_size=(width, height), stride=(h_stride, v_stride), padding=0
        )

    def forward(self, x):
        x = x.view(1, 1, *x.shape[-2:])
        rpted = self.repeat_frame(x)
        unfolded = self.unfoldT(rpted.transpose(-1, -2))
        return unfolded.squeeze(0)


class NormalizeSpectrogram(nn.Module):
    def __init__(self, min_level_db=-80):
        super(NormalizeSpectrogram, self).__init__()
        self.min_level_db = min_level_db

    def forward(self, x):
        # x = np.clip((x - min_level_db) / -min_level_db, 0, 1)
        # x = (x - (-50.35543)) / 10.555636
        return (x - (self.min_level_db // 2)) / -(self.min_level_db // 2)


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.
    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(
        self,
        virtual_crop_scale=(1.0, 1.5),
        freq_scale=(0.6, 1.5),
        time_scale=(0.6, 1.5),
    ):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = "bicubic"
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [
            int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)
        ]
        virtual_crop_area = (
            torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
            .to(torch.float)
            .to(lms.device)
        )
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y : y + h, x : x + w] = lms
        # get random area
        i, j, h, w = self.get_params(
            virtual_crop_area.shape[-2:],
            lms.shape[-2:],
            self.time_scale,
            self.freq_scale,
        )
        crop = virtual_crop_area[:, i : i + h, j : j + w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(
            crop.unsqueeze(0),
            size=lms.shape[-2:],
            mode=self.interpolation,
            align_corners=True,
        ).squeeze(0)
        return lms.to(torch.float)

    def __repr__(self):
        format_string = (
            self.__class__.__name__ + f"(virtual_crop_size={self.virtual_crop_scale}"
        )
        format_string += ", time_scale={0}".format(
            tuple(round(s, 4) for s in self.time_scale)
        )
        format_string += ", freq_scale={0})".format(
            tuple(round(r, 4) for r in self.freq_scale)
        )
        return format_string


class SpecMixup(nn.Module):
    def __init__(self, num_classes, alpha=0.4):
        super(SpecMixup, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.dist = torch.distributions.Uniform(0, alpha)

    def forward(self, specs, labels):
        # idxs = lengths.squeeze().argsort(0)
        # s, lab, len = specs[idxs], labels[idxs], lengths[idxs]
        labels_oh = F.one_hot(labels, self.num_classes)
        lambda_ = self.dist.sample((specs.shape[0], 1, 1)).to(specs.device)
        mixed_specs = (1.0 - lambda_) * specs + lambda_ * specs.roll(1, dims=0)
        mixed_labels = (1.0 - lambda_.view(-1, 1)) * labels_oh + lambda_.view(
            -1, 1
        ) * labels_oh.roll(1, dims=0)
        return mixed_specs, mixed_labels


class SpecDrloc(nn.Module):
    def __init__(self, n_freq_patches, n_time_patches, dim, mlp_dim=256, m=16):
        super(SpecDrloc, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 2),
        )
        self.m, self.n_freq_patches, self.n_time_patches = (
            m,
            n_freq_patches,
            n_time_patches,
        )
        self.l1_loss = nn.L1Loss()

    def forward(self, tokens, lengths):
        device = tokens.device
        i = torch.tensor([random.choices(range(0, l), k=self.m) for l in lengths]).to(
            device
        )
        j = torch.tensor([random.choices(range(0, l), k=self.m) for l in lengths]).to(
            device
        )

        freq_delta = (
            ((i % self.n_freq_patches) - (j % self.n_freq_patches))
            / self.n_freq_patches
        ).flatten()
        time_delta = (
            ((i / self.n_freq_patches).floor() - (j / self.n_freq_patches).floor())
            / self.n_time_patches
        ).flatten()

        bs = (
            torch.tensor([[x] * self.m for x in range(tokens.shape[0])])
            .to(device)
            .flatten()
        )

        bi = tokens[bs, i.flatten()]
        bj = tokens[bs, j.flatten()]

        predictions = self.mlp(torch.cat((bi, bj), -1))
        targets = torch.stack((freq_delta, time_delta), -1)
        loss = self.l1_loss(predictions, targets)
        return loss


if __name__ == "__main__":
    drloc = SpecDrloc(5, 256, 64)
    S = torch.randn(32, 128, 64)
    lengths = [random.randint(50, 120) for _ in range(32)]

    loss = drloc(S, lengths)
    print(loss)
