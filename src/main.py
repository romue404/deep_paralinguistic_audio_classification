import wandb
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from src.models.configs import RootConfig
from src.models.dataset import RawDataset
from src.modules.data_modules import BalancedSimpleSpectrogramDataModule
from src.rnn_model import ClassificationTFMR
from src.modules.spec_utils import (
    RandomResizeCrop,
    PatchifySpectrogram,
    NormalizeSpectrogram,
)
import torch
from torchvision.transforms import Compose, RandomApply
from torchaudio.transforms import FrequencyMasking
import hydra
from src.utils import initialize_rng


def train(cfg: RootConfig, seed: int):
    initialize_rng(seed)

    ds = RawDataset(cfg.dataset.dir, cfg.dataset.sr)

    config_defaults = dict(**cfg.model)

    wandb_logger = WandbLogger(
        entity="mdsg",
        project="test-compare-22",
        config=config_defaults,
        log_model="all",
        reinit=True,
    )

    wandb.define_metric("uar", summary="max")
    # use config ant NOT cfg - cfg is only for defaults!
    config = wandb.config

    patchify = PatchifySpectrogram(
        config.patch_h, config.patch_w, config.v_stride, config.h_stride
    )

    train_tfms = Compose(
        [
            NormalizeSpectrogram(),
            RandomApply(
                [
                    RandomResizeCrop(
                        virtual_crop_scale=(1.0, 1.0 + config.crop_scale),
                        freq_scale=(1.0 - config.crop_scale, 1.0 + config.crop_scale),
                        time_scale=(1.0 - config.crop_scale, 1.0 + config.crop_scale),
                    )
                ],
                config.prob_crop,
            ),
            FrequencyMasking(freq_mask_param=config.freq_mask_param),
            patchify,
        ]
    )
    test_tfms = Compose([NormalizeSpectrogram(), patchify])

    data_module = BalancedSimpleSpectrogramDataModule(
        specs_path=ds.specs_path,
        name2class=ds.name2class,
        label_csv=ds.full_csv,
        train_transform=train_tfms,
        test_transform=test_tfms,
        num_workers=4,
        batch_size=32,
    )

    data_module.setup()

    model = ClassificationTFMR(
        classes=ds.classes,
        input_dim=config.patch_h * config.patch_w,
        num_freq_patches=int(((128 - config.patch_h) / config.v_stride) + 1),
        num_warmup_epochs=int(0.1 * cfg.run.max_epochs),
        **config
    )

    wandb_logger.watch(model, log="all")
    checkpoint_callback = ModelCheckpoint(monitor="uar", mode="max")

    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        max_epochs=cfg.run.max_epochs,
        logger=wandb_logger,
        check_val_every_n_epoch=2,
        log_every_n_steps=2,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)
    # trainer.test(model, data_module)
    wandb_logger.experiment.finish()

# TODO hyparams for the run -> specify config name
@hydra.main(version_base=None, config_path="../configs", config_name="config-vocc")
def init_training(cfg: RootConfig):
    for seed in [1337]:
        train(cfg, seed)

# TODO hyparams for the run -> specify config name
@hydra.main(version_base=None, config_path="../configs", config_name="config-vocc")
def test(cfg: RootConfig):
    initialize_rng(cfg.run.seed)

    ds = RawDataset(cfg.dataset.dir, cfg.dataset.sr)

    config = cfg.model

    patchify = PatchifySpectrogram(
        config.patch_h, config.patch_w, config.v_stride, config.h_stride
    )

    train_tfms = Compose(
        [
            NormalizeSpectrogram(),
            RandomApply(
                [
                    RandomResizeCrop(
                        virtual_crop_scale=(1.0, 1.0 + config.crop_scale),
                        freq_scale=(1.0 - config.crop_scale, 1.0 + config.crop_scale),
                        time_scale=(1.0 - config.crop_scale, 1.0 + config.crop_scale),
                    )
                ],
                config.prob_crop,
            ),
            FrequencyMasking(freq_mask_param=config.freq_mask_param),
            patchify,
        ]
    )
    test_tfms = Compose([NormalizeSpectrogram(), patchify])

    data_module = BalancedSimpleSpectrogramDataModule(
        specs_path=ds.specs_path,
        name2class=ds.name2class,
        label_csv=ds.full_csv,
        train_transform=train_tfms,
        test_transform=test_tfms,
        num_workers=4,
        batch_size=32,
    )

    data_module.setup()

    model = ClassificationTFMR(
        classes=ds.classes,
        input_dim=config.patch_h * config.patch_w,
        num_freq_patches=int(((128 - config.patch_h) / config.v_stride) + 1),
        num_warmup_epochs=int(0.1 * cfg.run.max_epochs),
        **config
    )

    logger = CSVLogger("logs", name="results", flush_logs_every_n_steps=1)

    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        max_epochs=cfg.run.max_epochs,
        logger=logger,
        check_val_every_n_epoch=2,
        log_every_n_steps=2,
    )
    # TODO edit path to weights
    trainer.test(model, data_module, ckpt_path="./checkpoints/ksf-best2", verbose=True)


if __name__ == "__main__":
    # TODO train or test
    # init_training()
    # test()
