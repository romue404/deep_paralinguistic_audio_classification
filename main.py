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

patch_h, patch_w, v_stride, h_stride = 32, 4, 16, 2
#patch_h, patch_w, v_stride, h_stride = 128, 2, 1, 1
train_tfms = Compose([NormalizeSpectrogram(),
                      RandomApply([RandomResizeCrop(
                          virtual_crop_scale=(1.0, 1.25),
                          freq_scale=(0.75, 1.25),
                          time_scale=(0.75, 1.25)
                      )], .75),
                      FrequencyMasking(freq_mask_param=8),
                      PatchifySpectrogram(patch_h, patch_w, v_stride, h_stride)
                      ])
test_tfms = Compose([
    NormalizeSpectrogram(),
    PatchifySpectrogram(patch_h, patch_w, v_stride, h_stride)
])


#data_module = SimpleSpectrogramDataModule(train_transform=train_tfms, test_transform=test_tfms,
#                                          num_workers=4, batch_size=32, specs_path=c.ksf_specs_path,
#                                          name2class=c.ksf_name2class, label_csv=c.ksf_full_csv)

data_module = SimpleSpectrogramDataModule(
    specs_path=c.vocc_specs_path, name2class=c.vocc_name2class, label_csv=c.vocc_full_csv,
    train_transform=train_tfms, test_transform=test_tfms, num_workers=4, batch_size=32
)
num_classes = len(c.vocc_classes) - 1  #remove ? label drom testset

print(num_classes, c.ksf_name2class)
data_module.setup()

# 0.0578
lr = 3e-4
model = ClassificationTFMR(**dict(
    num_classes=num_classes,
    lr=3e-4,
    weight_decay=1e-4,
    input_dim=patch_h*patch_w,
    num_freq_patches=int(((128 - patch_h) / v_stride) + 1),
    hidden_size=128,
    num_layers=10,
    num_heads=4,
    ff_mult=3,
    mask_prob=0.2,
    dropout=0.2,
    label_smoothing=0.0,
    num_mem_kv=32,
    num_warmup_epochs=30,
    f_patchmerger=False
))

trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=5, max_epochs=750)

trainer.fit(model, data_module)
print(len(trainer.train_dataloader))
print(f'Best devel UAR: {model.best_uar}')

trainer.test(model, data_module)