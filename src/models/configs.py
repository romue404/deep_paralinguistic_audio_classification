from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    lr: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    num_heads: int
    ff_mult: int
    mask_prob: float
    dropout: float
    label_smoothing: float
    num_mem_kv: int
    num_warmup_epochs: int
    f_patchmerger: bool


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
