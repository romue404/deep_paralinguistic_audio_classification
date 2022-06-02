from pathlib import Path
import pandas as pd


class RawDataset:
    def __init__(self, dir: str, sr: int) -> None:
        self.dir = Path(dir)
        self.sr = sr
        self.wavs_path = Path.joinpath(self.dir, "wav")
        self.specs_path = Path.joinpath(self.dir, "melspecs")
        self.labels_path = Path.joinpath(self.dir, "lab", "labels.csv")

        if not self.specs_path.exists():
            raise Exception(
                "Melspectogams not found! You can create them with the preprocessing script."
            )

        (
            self.full_csv,
            self.classes,
            self.name2class,
            self.class2name,
        ) = self.all_labels_csv(Path.joinpath(self.dir, "lab"), test_available=False)

    def all_labels_csv(
        self,
        path,
        names=["train", "devel", "test"],
        test_available=False,
        is_combined=False,
    ):
        if not is_combined:
            csvs = [pd.read_csv(path / f"{name}.csv") for name in names]
            for csv_, partition in zip(csvs, names):
                csv_["partition"] = partition
            full_csv = pd.concat(csvs)
        else:
            full_csv = pd.read_csv(path)
            full_csv["partition"] = "N/A"
            for partition in names:
                full_csv.loc[
                    full_csv.file_name.str.contains(partition), "partition"
                ] = partition
        classes = sorted(list(full_csv.label.unique()))
        name2class = (
            {n: c for c, n in enumerate(classes)}
            if test_available
            else {n: c - 1 for c, n in enumerate(classes)}
        )
        class2name = {c: n for n, c in name2class.items()}
        return full_csv, classes, name2class, class2name
