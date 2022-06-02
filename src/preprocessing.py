import numpy as np
from models.configs import RootConfig
from models.dataset import RawDataset
import librosa
import librosa.display
from tqdm import tqdm
from matplotlib import pyplot as plt
import hydra


def preprocess_dataset(ds: RawDataset):
    specs_path = ds.specs_path
    wav_path = ds.wavs_path
    # data_path = ds.dir
    wavs = sorted(list(wav_path.glob("*.wav")))

    specs_path.mkdir(exist_ok=True, parents=False)
    melspec_params = dict(n_fft=1024, hop_length=512, n_mels=128, power=2)

    print(
        f"... Converting wavs from {wav_path} into mel-spectrograms in {specs_path} ..."
    )
    for wav in tqdm(wavs):
        path = specs_path / f"{wav.stem}.npy"
        # path_img = specs_path / f"{wav.stem}.jpg"
        audio, sr = librosa.load(str(wav), sr=ds.sr)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, **melspec_params)
        mel_spec_db = librosa.amplitude_to_db(S=mel_spec, ref=np.max)
        # mel_spec_norm = normalize(mel_spec_db)
        mel_spec_norm = mel_spec_db
        np.save(file=path, arr=mel_spec_norm)
        # save images
        # _, n = mel_spec_db.shape
        # mel_spec_to_img(mel_spec_db, sr, melspec_params["hop_length"], path_img)
        plt.close()

    print(f"Preprocessing done\tmel-spectrograms can be found in {specs_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: RootConfig):
    # def mel_spec_to_img(spectrogram, sr, hop_len, out_path, size=227):
    #     # prepare plotting
    #     fig = plt.figure(frameon=False, tight_layout=False)
    #     fig.set_size_inches(1, 1)
    #     ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    #     ax.set_axis_off()
    #     fig.add_axes(ax)
    #     spectrogram_axes = librosa.display.specshow(
    #         spectrogram,
    #         hop_length=hop_len,
    #         sr=sr,
    #         cmap="viridis",
    #         y_axis="mel",
    #         x_axis="time",
    #     )
    #     fig.add_axes(spectrogram_axes, id="spectrogram")
    #     fig.savefig(out_path, format="jpg", dpi=size)
    #     plt.clf()

    ds = RawDataset(cfg.dataset.dir, cfg.dataset.sr)
    preprocess_dataset(ds)


if __name__ == "__main__":
    main()
