import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from src.models.configs import RootConfig
from src.models.dataset import RawDataset
from src.modules.data_modules import SimpleSpectrogramDataModule
from src.rnn_model import ClassificationTFMR
from src.modules.spec_utils import (
    RandomResizeCrop,
    PatchifySpectrogram,
    NormalizeSpectrogram,
)
from torchvision.transforms import Compose, RandomApply
from torchaudio.transforms import FrequencyMasking
import hydra
from src.utils import initialize_rng



def train(cfg, seed):
    initialize_rng(seed)

    ds = RawDataset(cfg.dataset.dir, cfg.dataset.sr)

    num_classes = len(ds.classes) - 1  # remove ? label from testset TODO remove -1 when training on primates!!!
    config_defaults = dict(**cfg.model)

    wandb_logger = WandbLogger(
        entity="mdsg", project="test-compare-22", config=config_defaults, log_model=False, reinit=True
    )

    wandb.define_metric("uar", summary="max")
    # use config ant NOT cfg - cfg is only for defaults!
    config = wandb.config

    patchify = PatchifySpectrogram(config.patch_h,
                                   config.patch_w,
                                   config.v_stride,
                                   config.h_stride)

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
    test_tfms = Compose(
        [
            NormalizeSpectrogram(),
            patchify
        ]
    )

    data_module = SimpleSpectrogramDataModule(
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
        num_classes=num_classes,
        input_dim=config.patch_h * config.patch_w,
        num_freq_patches=int(((128 - config.patch_h) / config.v_stride) + 1),
        num_warmup_epochs=int(0.1*cfg.run.max_epochs),
        **config
    )

    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        max_epochs=cfg.run.max_epochs,
        logger=wandb_logger,
        check_val_every_n_epoch=2,
        log_every_n_steps=2
    )

    wandb_logger.watch(model, log="all")
    trainer.fit(model, data_module)
    #trainer.test(model, data_module)
    wandb_logger.experiment.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def init_training(cfg: RootConfig):
    for seed in [1337]:
        train(cfg, seed)

if __name__ == "__main__":
    init_training()
    #train(cfg, seed)