import pytorch_lightning as pl
from paralinguistic_data_modules import BalancedSimpleSpectrogramDataModule, SimpleSpectrogramDataModule
from rnn_model import MySpecTFMR, ClassificationTFMR
from modules.spec_utils import RandomResizeCrop, PatchifySpectrogram, NormalizeSpectrogram
import torch
import numpy as np
from torchvision.transforms import Compose, RandomApply
from torchaudio.transforms import FrequencyMasking
import constants as c


seed = 1337
torch.manual_seed(seed)
np.random.seed(seed)

patch_h, patch_w, v_stride, h_stride = 64, 6, 32, 4

train_tfms = Compose([NormalizeSpectrogram(),
                      RandomApply([RandomResizeCrop(
                          virtual_crop_scale=(1.0, 1.25),
                          freq_scale=(0.75, 1.25),
                          time_scale=(0.75, 1.25)
                      )], .75),
                      PatchifySpectrogram(patch_h, patch_w, v_stride, h_stride)
                      ])
test_tfms = Compose([
    NormalizeSpectrogram(),
    PatchifySpectrogram(patch_h, patch_w, v_stride, h_stride)
])


data_module = SimpleSpectrogramDataModule(
    specs_path=c.vocc_specs_path, name2class=c.vocc_name2class, label_csv=c.vocc_full_csv,
    train_transform=train_tfms, test_transform=test_tfms, num_workers=4, batch_size=32
)
num_classes = len(c.vocc_classes) - 1  #remove ? label drom testset
data_module.setup()

# vocc:150, primates:217, kfv:322
# 0.0578

lr = 5e-4
model = ClassificationTFMR(**dict(
    num_classes=num_classes,
    lr=lr,
    weight_decay=5e-2,
    input_dim=patch_h*patch_w,
    num_freq_patches=int(((128 - patch_h) / v_stride) + 1),
    hidden_size=128,
    num_layers=8,
    num_heads=4,
    ff_mult=4,
    mask_prob=0.05,
    dropout=0.0,
    label_smoothing=0.0,
    num_mem_kv=32,
    num_warmup_epochs=30,
    f_patchmerger=False
))

trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=10, max_epochs=300)
trainer.fit(model, data_module)
print(len(trainer.train_dataloader))
print(f'Best devel UAR: {model.best_uar}')

#trainer.test(model, data_module)