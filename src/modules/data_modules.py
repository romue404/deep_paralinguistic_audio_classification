from typing import Union, List, Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from pathlib import Path
from modules.spec_utils import MelSpec
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleSpectrogramDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        train_transform,
        test_transform,
        name2class,
        specs_path,
        label_csv,
    ):
        super(SimpleSpectrogramDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.class_counts = None
        self.name2class = name2class
        self.specs_path = specs_path
        self.df = label_csv

    def setup(self, stage: Optional[str] = None):
        df = self.df.copy()
        df.label = df.label.map(self.name2class)
        df.filename = df.filename.apply(
            lambda x: f"{self.specs_path / Path(x).stem}.npy"
        )
        datasets = {"train": [], "test": [], "devel": []}
        for filename, label, partition in df.values:
            tfm = self.train_transform if partition == "train" else self.test_transform
            datasets[partition].append(MelSpec(filename, label=label, transform=tfm))
        self.datasets = {k: ConcatDataset(v) for k, v in datasets.items()}

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            *args,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_variable_len_sequence,
            batch_size=self.batch_size,
            **kwargs,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.datasets["devel"],
            *args,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_variable_len_sequence,
            batch_size=self.batch_size,
            **kwargs,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.datasets["test"],
            *args,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_variable_len_sequence,
            batch_size=self.batch_size,
            **kwargs,
        )


class BalancedSimpleSpectrogramDataModule(SimpleSpectrogramDataModule):
    def __init__(self, *args, **kwargs):
        super(BalancedSimpleSpectrogramDataModule, self).__init__(*args, **kwargs)
        counts = dict(self.df.query("partition == 'train'")["label"].value_counts())
        self.class_counts = {self.name2class[k]: v for k, v in counts.items()}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        weights = []
        for b in self.datasets["train"]:
            packed, label, names = b
            weights.append(1 / self.class_counts[int(label)])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=weights, num_samples=len(self.datasets["train"]), replacement=True
        )
        return DataLoader(
            self.datasets["train"],
            *args,
            sampler=sampler,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,  # shuffle false due to sampler
            collate_fn=collate_variable_len_sequence,
            batch_size=self.batch_size,
            **kwargs,
        )


def collate_variable_len_sequence(batch):
    labels = torch.tensor([l for _, l, _ in batch]).long()
    mels = [m for m, _, _ in batch]
    names = [name for _, _, name in batch]
    lengths = torch.tensor([m.shape[-1] for m in mels]).long()
    zero_column = torch.zeros_like(mels[0].select(-1, 0)).unsqueeze(-1)
    mels = (
        torch.stack(
            [
                torch.cat(([mel] + [zero_column] * (max(lengths) - mel.shape[-1])), -1)
                if mel.shape[-1] != max(lengths)
                else mel
                for mel in mels
            ],
            0,
        )
        .float()
        .squeeze()
    )
    # expected by pack padded sequence
    mels = mels.permute(
        0, 2, 1
    )  # 'batch token_size num_tokens -> batch num_tokens token_size'
    packed = pack_padded_sequence(mels, lengths, batch_first=True, enforce_sorted=False)
    return packed, labels, names
