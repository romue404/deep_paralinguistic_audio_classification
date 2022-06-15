from os import environ
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.seed import seed_everything
from copy import deepcopy


class NormalizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
        trainable_magnitude=True,
        **kwargs,
    ):
        super(NormalizedLinear, self).__init__(
            in_features, out_features, False, device, dtype
        )
        self.d_sqrt = in_features**0.5
        self.trainable_magnitude = trainable_magnitude
        self.scale = nn.Parameter(
            torch.tensor([0.5]), requires_grad=trainable_magnitude
        )

    def forward(self, input):
        normalized_input = F.normalize(input, dim=-1, p=2, eps=1e-5)
        normalized_weight = F.normalize(self.weight, dim=-1, p=2, eps=1e-5)
        return F.linear(normalized_input, normalized_weight) * self.d_sqrt * self.scale


class EMA(nn.Module):
    """Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def exclude_from_wt_decay(named_params, weight_decay, skip_list):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if ".g" in name:
            # print(f"skipped ReZero param {name}")
            excluded_params.append(param)
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
            # print(f"skipped param {name}")
        else:
            params.append(param)
    return [
        {"params": params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0},
    ]


def initialize_rng(seed_value: int):
    """seed everything"""
    gen = torch.Generator()
    gen.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True)
    seed_everything(seed_value, workers=True)
    return gen
