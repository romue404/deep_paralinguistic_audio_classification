from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    patch_h: int
    patch_w: int
    v_stride: int
    h_stride: int
    use_spec_pos_emb: bool
    lr: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    num_heads: int
    ff_mult: int
    use_attn_hack: bool
    mask_prob: float
    emb_dropout: float
    attn_dropout: float
    ff_dropout: float
    freq_mask_param: int
    label_smoothing: float
    num_mem_kv: int
    num_warmup_epochs: int
    prob_crop: float
    crop_scale: float
    mixup_alpha: float
    drloc_alpha: float
    drloc_m: int
    drloc_time_patches: int
    use_normalized_linear: bool
    num_patchmerges: int
    pooling_type: str
    norm: str
    ff: str



@dataclass
class RunConfig:
    seed: int
    max_epochs: int


@dataclass
class DatasetConfig:
    dir: str
    sr: int


@dataclass
class RootConfig:
    model: ModelConfig
    run: RunConfig
    dataset: DatasetConfig


cs = ConfigStore.instance()
cs.store(name="root_config", node=RootConfig)
