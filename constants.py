from pathlib import Path
import pandas as pd


def all_labels_csv(path, names=['train', 'devel', 'test'], test_available=False, is_combined=False):
    if not is_combined:
        csvs = [pd.read_csv(path / f'{name}.csv') for name in names]
        for csv_, partition in zip(csvs, names):
            csv_['partition'] = partition
        full_csv = pd.concat(csvs)
    else:
        full_csv = pd.read_csv(path)
        full_csv['partition'] = 'N/A'
        for partition in names:
            full_csv.loc[full_csv.file_name.str.contains(partition), 'partition'] = partition
    classes = sorted(list(full_csv.label.unique()))
    name2class = {n: c for c, n in enumerate(classes)} if test_available else {n: c-1 for c, n in enumerate(classes)}
    class2name = {c: n for n, c in name2class.items()}
    return full_csv, classes, name2class, class2name

# constants
sr = 16000
# PRIMATES
primates_data_path = Path('/mnt/robert/datasets/compare21-primates')
primates_wavs_path = primates_data_path / 'wav'
primates_specs_path = primates_data_path / 'melspecs'
primates_full_csv, primates_classes, primates_name2class, primates_class2name  = all_labels_csv(
    primates_data_path / 'lab', test_available=True
)

# VOC-C
vocc_data_path = Path('/mnt/robert/datasets/compare22-VOC-C')
vocc_wavs_path = vocc_data_path   / 'wav'
vocc_specs_path = vocc_data_path / 'melspecs'
vocc_full_csv, vocc_classes, vocc_name2class, vocc_class2name  = all_labels_csv(
    vocc_data_path / 'lab', test_available=False
)


# KSF
ksf_data_path = Path('/mnt/robert/datasets/compare22-KSF')
ksf_wavs_path = ksf_data_path   / 'wav'
ksf_specs_path = ksf_data_path / 'melspecs'
ksf_full_csv, ksf_classes, ksf_name2class, ksf_class2name  = all_labels_csv(
    ksf_data_path / 'lab', test_available=False
)


# MASKS
masks_data_path = primates_data_path = Path('/mnt/robert/datasets/compare20-masks')
masks_wavs_path = masks_data_path / 'wav'
masks_specs_path = masks_data_path / 'melspecs'
masks_labels_path = masks_data_path / 'lab' / 'labels.csv'

masks_full_csv, masks_classes, masks_name2class, masks_class2name  = all_labels_csv(
    masks_data_path / 'lab' / 'labels.csv',
    test_available=False, is_combined=True
)
masks_full_csv.rename(columns={'file_name': 'filename'}, inplace=True)